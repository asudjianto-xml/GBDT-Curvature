"""Deep fANOVA interaction-kernel curvature on bikeshare, with log1p(cnt)."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

import config as C
import data as D
from california_housing_deep_anova_kernel_curvature import (
    background_predictions, model_kernel_curvature, SumModel, main_effect_table,
)


SEMANTIC_CAT = ["yr", "holiday", "workingday", "weathersit", "mnth", "hr", "weekday"]
NUMERIC = ["temp", "hum", "windspeed"]


def xgb(depth, constraints=None):
    kw = dict(n_estimators=C.SWEEP_ITERS, max_depth=depth, learning_rate=C.SWEEP_LR,
              objective="reg:squarederror", tree_method="hist", n_jobs=-1,
              random_state=C.RANDOM_STATE)
    if constraints is not None:
        kw["interaction_constraints"] = constraints
    return XGBRegressor(**kw)


def design_matrices(Xtr, Xte):
    ct = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), SEMANTIC_CAT),
        ("num", "passthrough", NUMERIC),
    ]).fit(Xtr)
    A0, B0 = ct.transform(Xtr), ct.transform(Xte)
    cols = [f"f{i}" for i in range(A0.shape[1])]
    A, B = pd.DataFrame(A0, columns=cols), pd.DataFrame(B0, columns=cols)
    groups, label_of = [], {}
    start = 0
    for name, categories in zip(SEMANTIC_CAT, ct.named_transformers_["cat"].categories_):
        group = [f"f{i}" for i in range(start, start + len(categories))]
        groups.append(group)
        for f in group: label_of[f] = name
        start += len(categories)
    num_start = ct.output_indices_["num"].start
    for q, name in enumerate(NUMERIC):
        group = [f"f{num_start + q}"]
        groups.append(group); label_of[group[0]] = name
    return A, B, groups, label_of


def screen_pairs(model, label_of):
    frame = model.get_booster().trees_to_dataframe()
    score = {}
    for _, tree in frame.groupby("Tree"):
        root = tree.loc[tree.Node == 0]
        if root.empty or root.iloc[0].Feature == "Leaf":
            continue
        a = label_of[root.iloc[0].Feature]
        for _, child in tree.loc[tree.Node.isin([1, 2])].iterrows():
            if child.Feature == "Leaf":
                continue
            b = label_of[child.Feature]
            if a == b:
                continue
            key = tuple(sorted((a, b)))
            score[key] = score.get(key, 0.0) + float(child.Gain)
    return sorted(score.items(), key=lambda kv: kv[1], reverse=True)


def main():
    if not C.LOG1P_TARGET:
        raise RuntimeError("This analysis requires config.LOG1P_TARGET=True.")
    X, y, raw_y = D.load()                          # y = log1p(cnt)
    split = D.split(X, y, raw_y)
    Xtr, ytr, Xte, yte = split["X_train"], split["y_train"], split["X_test"], split["y_test"]
    A, B, groups, label_of = design_matrices(Xtr, Xte)
    group_of = {name: group for group, name in zip(groups, SEMANTIC_CAT + NUMERIC)}
    print(f"bikeshare: train={A.shape}, test={B.shape}; response=log1p(cnt)")

    print("[1/4] cross-fitted XGBoost GAM residuals")
    oof = np.empty(len(A))
    for fold, (tr, va) in enumerate(KFold(n_splits=5, shuffle=True,
                                           random_state=C.RANDOM_STATE).split(A), 1):
        gam = xgb(10, groups).fit(A.iloc[tr], ytr[tr])
        oof[va] = gam.predict(A.iloc[va])
        print(f"  fold {fold}/5")

    print("[2/4] depth-2 interaction screen")
    screen = xgb(2).fit(A, ytr - oof)
    ranked = screen_pairs(screen, label_of)
    selected = ranked[:8]
    print("  selected pairs:")
    for (a, b), gain in selected:
        print(f"    {a} x {b}: {gain:.3f}")
    if not selected:
        raise RuntimeError("No cross-feature depth-2 interactions were selected.")

    print("[3/4] final GAM + screened deep residual model")
    gam = xgb(10, groups).fit(A, ytr)
    deep = xgb(6, [group_of[a] + group_of[b] for (a, b), _ in selected]).fit(A, ytr - gam.predict(A))
    gam_pred, final_pred = gam.predict(B), gam.predict(B) + deep.predict(B)
    print(f"  test log1p MAE: GAM={np.abs(gam_pred-yte).mean():.4f}, "
          f"GAM+screened-deep={np.abs(final_pred-yte).mean():.4f}")

    print("[4/4] deep fANOVA kernels + curvature")
    rng = np.random.default_rng(C.RANDOM_STATE)
    background = A.iloc[rng.choice(len(A), size=96, replace=False)].reset_index(drop=True)
    base = deep.predict(background)[None, :]
    records = []
    for (a, b), gain in selected:
        ga, gb = group_of[a], group_of[b]
        hab = background_predictions(deep, B, background, ga + gb)
        ha = background_predictions(deep, B, background, ga)
        hb = background_predictions(deep, B, background, gb)
        psi = hab - ha - hb + base
        component = psi.mean(axis=1)
        component = component - component.mean()    # center: A_S is variance, not 2nd moment
        curvature, raw, states = model_kernel_curvature(psi / np.sqrt(psi.shape[1]), component)
        records.append({"feature_1": a, "feature_2": b, "screen_gain": gain,
                        "anova_energy": float(np.mean(component ** 2)),
                        "component_std": float(np.std(component)),
                        "model_kernel_curvature": curvature,
                        "laplacian_energy": raw, "kernel_states": states,
                        "kernel_rank": int(np.linalg.matrix_rank(psi))})
    out = pd.DataFrame(records).sort_values("anova_energy", ascending=False)
    print("\nDeep-model pair fANOVA importance + model-kernel curvature:")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    out.to_csv(C.OUT_DIR / "bikeshare_deep_anova_kernel_curvature.csv", index=False)
    np.savez(C.OUT_DIR / "bikeshare_deep_anova_kernel_curvature.npz",
             **{col: out[col].to_numpy() for col in out.columns})
    print(f"\nsaved -> {C.OUT_DIR / 'bikeshare_deep_anova_kernel_curvature.csv'}")

    full = SumModel(gam, deep)
    features = SEMANTIC_CAT + NUMERIC
    mains = main_effect_table(full, B, background, features, group_of)
    print("\nMain-effect fANOVA importance + model-kernel curvature:")
    print(mains.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    mains.to_csv(C.OUT_DIR / "bikeshare_main_effect_curvature.csv", index=False)


if __name__ == "__main__":
    main()
