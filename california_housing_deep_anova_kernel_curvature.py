"""Deep-model fANOVA interaction kernels and curvature for California housing.

Stage 1: XGBoost GAM.  Stage 2: the existing OOF depth-2 screen supplies
candidate pairs.  Stage 3: a depth-6 residual model constrained by those pairs.

For each selected pair S={j,k}, Psi_S(x) is a finite fANOVA contrast feature map
over empirical background draws.  K_S = Psi_S Psi_S' / M is therefore PSD and
is induced by the FINAL deep model, not by raw feature distance or the screen.
"""
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBRegressor

import config as C


def xgb(depth, constraints=None):
    kw = dict(n_estimators=C.SWEEP_ITERS, max_depth=depth, learning_rate=C.SWEEP_LR,
              objective="reg:squarederror", tree_method="hist", n_jobs=-1,
              random_state=C.RANDOM_STATE)
    if constraints is not None:
        kw["interaction_constraints"] = constraints
    return XGBRegressor(**kw)


def background_predictions(model, Xeval, background, changed, chunk=48):
    """Matrix [point, background]: model(x_changed, b_unchanged)."""
    n, m = len(Xeval), len(background)
    ans = np.empty((n, m))
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        frame = pd.concat([background] * (hi - lo), ignore_index=True)
        for col in changed:
            frame[col] = np.repeat(Xeval.iloc[lo:hi][col].to_numpy(), m)
        ans[lo:hi] = model.predict(frame).reshape(hi - lo, m)
    return ans


def model_kernel_curvature(psi, component, k=C.KNN, n_perm=200, seed=0):
    """Support-invariant component curvature on K=Psi Psi'/M's feature graph.

    The roughness R(g) is the count-weighted mean squared graph Laplacian of the
    component. We normalize it by its mean under random permutations of the
    component values over the graph nodes, R(g o pi). The permutation null absorbs
    the graph's size and structure, so Q_S is comparable across components with
    very different numbers of distinct states -- a binary feature on a two-node
    graph no longer inflates. It is also amplitude-invariant, since numerator and
    null scale together. Q_S below 1 is smoother than chance; a degenerate
    few-node graph sits near 1.
    """
    # Equivalent rows have exactly the same learned feature map; collapse them
    # before kNN so duplicate states cannot make a zero bandwidth.
    states, inv, counts = np.unique(psi, axis=0, return_inverse=True, return_counts=True)
    values = np.array([component[inv == q].mean() for q in range(len(states))])
    if len(states) < 2 or np.var(component) < 1e-12:
        return 0.0, 0.0, len(states)
    kk = min(k, len(states) - 1)
    d, ix = NearestNeighbors(n_neighbors=kk + 1).fit(states).kneighbors(states)
    d, ix = d[:, 1:], ix[:, 1:]
    fallback = np.median(d[d > 1e-12]) if np.any(d > 1e-12) else 1.0
    scale = np.where(d[:, -1] > 1e-12, d[:, -1], fallback)
    w = np.exp(-d ** 2 / (2 * scale[:, None] ** 2 + 1e-12)); w /= w.sum(axis=1, keepdims=True)

    def roughness(v):
        lap = v - (w * v[ix]).sum(axis=1)
        return float(np.average(lap ** 2, weights=counts))

    raw = roughness(values)
    rng = np.random.default_rng(seed)
    null = float(np.mean([roughness(rng.permutation(values)) for _ in range(n_perm)]))
    return (raw / null if null > 0 else 0.0), raw, len(states)


class SumModel:
    """Full model f = f_1 + g_deep, so first-order contrasts give the model's
    main effects and second-order contrasts give the interactions."""
    def __init__(self, a, b):
        self.a, self.b = a, b

    def predict(self, X):
        return self.a.predict(X) + self.b.predict(X)


def main_effect_table(full, Xeval, background, features, changed_of):
    """Importance A_j and curvature Q_j for each main effect (first-order
    contrast of the full model), computed like the pair components."""
    base = full.predict(background)[None, :]
    rows = []
    for name in features:
        h = background_predictions(full, Xeval, background, changed_of[name])
        psi = h - base                              # first-order fANOVA contrast
        comp = psi.mean(axis=1)
        comp = comp - comp.mean()                   # center: A_j is variance
        curv, raw, states = model_kernel_curvature(psi / np.sqrt(psi.shape[1]), comp)
        rows.append({"feature": name, "anova_energy": float(np.mean(comp ** 2)),
                     "component_std": float(np.std(comp)),
                     "model_kernel_curvature": curv, "laplacian_energy": raw,
                     "kernel_states": states})
    return pd.DataFrame(rows).sort_values("anova_energy", ascending=False)


def main():
    data = fetch_california_housing(as_frame=True)
    raw_names = list(data.data.columns)
    X, y = data.data.copy(), np.log1p(data.target.to_numpy(float))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=C.TEST_SIZE,
                                           random_state=C.RANDOM_STATE)
    Xtr, Xte = Xtr.reset_index(drop=True), Xte.reset_index(drop=True)
    cols = [f"f{i}" for i in range(X.shape[1])]
    raw_to_col = dict(zip(raw_names, cols))
    A, B = Xtr.copy(), Xte.copy(); A.columns = cols; B.columns = cols

    screened = pd.read_csv(C.OUT_DIR / "california_housing_screened_pairs.csv").head(8)
    pairs = [(raw_to_col[row.feature_1], raw_to_col[row.feature_2])
             for row in screened.itertuples(index=False)]
    print("fit final deep residual model; target=log1p(MedHouseVal)")
    gam = xgb(10, [[c] for c in cols]).fit(A, ytr)
    deep = xgb(6, [list(pair) for pair in pairs]).fit(A, ytr - gam.predict(A))
    pred = gam.predict(B) + deep.predict(B)
    print(f"  held-out log1p MAE: GAM+screened-deep={np.abs(pred-yte).mean():.4f}")

    rng = np.random.default_rng(C.RANDOM_STATE)
    background = A.iloc[rng.choice(len(A), size=96, replace=False)].reset_index(drop=True)
    base = deep.predict(background)[None, :]
    records = []
    print("construct deep fANOVA PSD kernels and component curvature")
    for (a, b), row in zip(pairs, screened.itertuples(index=False)):
        hab = background_predictions(deep, B, background, [a, b])
        ha = background_predictions(deep, B, background, [a])
        hb = background_predictions(deep, B, background, [b])
        psi = hab - ha - hb + base                 # finite fANOVA feature map
        component = psi.mean(axis=1)               # f_{jk} under this measure
        component = component - component.mean()    # center: A_S is variance, not 2nd moment
        curvature, raw, states = model_kernel_curvature(psi / np.sqrt(psi.shape[1]), component)
        records.append({"feature_1": row.feature_1, "feature_2": row.feature_2,
                        "screen_gain": row.screen_gain,
                        "anova_energy": float(np.mean(component ** 2)),
                        "component_std": float(np.std(component)),
                        "model_kernel_curvature": curvature,
                        "laplacian_energy": raw, "kernel_states": states,
                        "kernel_rank": int(np.linalg.matrix_rank(psi))})
    out = pd.DataFrame(records).sort_values("anova_energy", ascending=False)
    print("\nDeep-model pair fANOVA importance + model-kernel curvature:")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    out.to_csv(C.OUT_DIR / "california_housing_deep_anova_kernel_curvature.csv", index=False)
    np.savez(C.OUT_DIR / "california_housing_deep_anova_kernel_curvature.npz",
             **{col: out[col].to_numpy() for col in out.columns})
    print(f"\nsaved -> {C.OUT_DIR / 'california_housing_deep_anova_kernel_curvature.csv'}")

    full = SumModel(gam, deep)
    changed_of = {name: [raw_to_col[name]] for name in raw_names}
    mains = main_effect_table(full, B, background, raw_names, changed_of)
    print("\nMain-effect fANOVA importance + model-kernel curvature:")
    print(mains.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    mains.to_csv(C.OUT_DIR / "california_housing_main_effect_curvature.csv", index=False)


if __name__ == "__main__":
    main()
