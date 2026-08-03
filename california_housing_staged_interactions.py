"""Test staged GAM -> depth-2 interaction screen -> constrained deep residual model.

All models target log1p(MedHouseVal).  Pair discovery uses out-of-fold GAM
residuals; the final GAM and deep residual model are refit on the full training
split and evaluated only on the untouched test split.
"""
import itertools
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor

import config as C


def xgb(depth, *, constraints=None):
    args = dict(n_estimators=C.SWEEP_ITERS, max_depth=depth,
                learning_rate=C.SWEEP_LR, objective="reg:squarederror",
                tree_method="hist", n_jobs=-1, random_state=C.RANDOM_STATE)
    if constraints is not None:
        args["interaction_constraints"] = constraints
    return XGBRegressor(**args)


def pair_scores(model, names):
    """Score pairs by gain of depth-2 child splits below each root feature."""
    frame = model.get_booster().trees_to_dataframe()
    score = {}
    for _, tree in frame.groupby("Tree"):
        root = tree.loc[tree.Node == 0]
        if root.empty or root.iloc[0].Feature == "Leaf":
            continue
        parent = root.iloc[0].Feature
        # In a max-depth-2 tree, nodes 1 and 2 are the two child splits.  Each
        # produces a root-child interaction; siblings never form an interaction.
        for _, child in tree.loc[tree.Node.isin([1, 2])].iterrows():
            if child.Feature == "Leaf" or child.Feature == parent:
                continue
            a, b = sorted((parent, child.Feature), key=lambda s: int(s[1:]))
            score[(a, b)] = score.get((a, b), 0.0) + float(child.Gain)
    return sorted(score.items(), key=lambda kv: kv[1], reverse=True)


def main():
    data = fetch_california_housing(as_frame=True)
    names = list(data.data.columns)
    X = data.data.copy()
    y = np.log1p(data.target.to_numpy(float))  # required response transform
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=C.TEST_SIZE,
                                           random_state=C.RANDOM_STATE)
    Xtr, Xte = Xtr.reset_index(drop=True), Xte.reset_index(drop=True)
    cols = [f"f{i}" for i in range(X.shape[1])]
    A, B = Xtr.copy(), Xte.copy(); A.columns = cols; B.columns = cols
    singletons = [[c] for c in cols]
    print(f"California housing; target=log1p(MedHouseVal); train={A.shape}, test={B.shape}")

    print("[1/4] cross-fitted GAM residuals")
    oof = np.empty(len(A))
    folds = KFold(n_splits=5, shuffle=True, random_state=C.RANDOM_STATE)
    for fold, (tr, va) in enumerate(folds.split(A), 1):
        gam = xgb(10, constraints=singletons).fit(A.iloc[tr], ytr[tr])
        oof[va] = gam.predict(A.iloc[va])
        print(f"  fold {fold}/5")

    print("[2/4] depth-2 screen on OOF GAM residuals")
    screen = xgb(2).fit(A, ytr - oof)
    ranked = pair_scores(screen, names)
    top = ranked[:8]
    print("  top candidate interactions (child split gain):")
    for (a, b), gain in top:
        print(f"    {names[int(a[1:])]} x {names[int(b[1:])]}: {gain:.3f}")
    if not top:
        raise RuntimeError("No two-feature paths found in the depth-2 screen.")

    print("[3/4] refit full GAM, then deep residual model constrained by screened pairs")
    gam = xgb(10, constraints=singletons).fit(A, ytr)
    residual = ytr - gam.predict(A)
    selected_groups = [[a, b] for (a, b), _ in top]
    deep_resid = xgb(6, constraints=selected_groups).fit(A, residual)
    staged_pred = gam.predict(B) + deep_resid.predict(B)
    gam_pred = gam.predict(B)
    unrestricted = xgb(6).fit(A, ytr).predict(B)

    print("[4/4] held-out evaluation (log1p MAE)")
    results = pd.DataFrame([
        {"model": "GAM main effects", "mae_log1p": np.abs(gam_pred - yte).mean()},
        {"model": "GAM + screened deep residual", "mae_log1p": np.abs(staged_pred - yte).mean()},
        {"model": "unrestricted depth-6 XGBoost", "mae_log1p": np.abs(unrestricted - yte).mean()},
    ])
    print(results.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    pairs = pd.DataFrame([{"feature_1": names[int(a[1:])], "feature_2": names[int(b[1:])],
                           "screen_gain": gain} for (a, b), gain in ranked])
    results.to_csv(C.OUT_DIR / "california_housing_staged_results.csv", index=False)
    pairs.to_csv(C.OUT_DIR / "california_housing_screened_pairs.csv", index=False)
    np.savez(C.OUT_DIR / "california_housing_staged_predictions.npz", y_test=yte,
             gam=gam_pred, staged=staged_pred, unrestricted=unrestricted,
             oof_gam=oof)
    print(f"\nsaved -> {C.OUT_DIR / 'california_housing_staged_results.csv'}")


if __name__ == "__main__":
    main()
