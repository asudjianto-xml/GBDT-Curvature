"""Diagnostics for the IIGBM paper.

Part A -- Orthogonality.  Fitting the interaction (deep) model on GAM residuals
should leave the interaction components approximately uncorrelated with the main
effects, so the ANOVA energies A_S behave like an orthogonal decomposition. We
check this directly on California housing: the empirical correlation between each
main effect and each pair component, and between distinct pair components.

Part B -- Curvature validation.  Q_S is meant to rank a component by local
sharpness. We test it on synthetic data with two planted interactions of known
shape: one smooth and broad, one sharp and localized. A valid Q_S must rank the
localized one higher even when its ANOVA energy is smaller. A label-permutation
null gives Q_S a reference scale, and repeats over seeds / k / M give stability.
"""
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBRegressor

import config as C
from california_housing_deep_anova_kernel_curvature import (
    background_predictions, xgb,
)


def IC(groups):
    """XGBoost interaction_constraints as an integer-index string.

    Required because XGBoost 3.1.x + pandas 3.0 does not expose DataFrame column
    names to the booster, so name-based constraints raise 'not a subset'.
    """
    return "[" + ",".join("[" + ",".join(map(str, g)) + "]" for g in groups) + "]"


# ----- component extraction (finite-background fANOVA contrasts) -----

def main_effect(model, Xeval, background, col, base):
    h = background_predictions(model, Xeval, background, [col])
    m = h.mean(axis=1) - base.mean()
    return m - m.mean()                             # center


def pair_component(model, Xeval, background, a, b, base):
    hab = background_predictions(model, Xeval, background, [a, b])
    ha = background_predictions(model, Xeval, background, [a])
    hb = background_predictions(model, Xeval, background, [b])
    psi = hab - ha - hb + base
    comp = psi.mean(axis=1)
    return comp - comp.mean(), psi                  # center: A_S is variance


# ----- curvature with a permutation null (mirrors model_kernel_curvature) -----

def curvature_null(psi, component, k=C.KNN, n_perm=200, seed=0):
    states, inv, counts = np.unique(psi, axis=0, return_inverse=True,
                                    return_counts=True)
    values = np.array([component[inv == q].mean() for q in range(len(states))])
    if len(states) < 2 or np.var(component) < 1e-12:
        return dict(q=0.0, z=0.0, states=len(states))
    kk = min(k, len(states) - 1)
    d, ix = NearestNeighbors(n_neighbors=kk + 1).fit(states).kneighbors(states)
    d, ix = d[:, 1:], ix[:, 1:]
    fb = np.median(d[d > 1e-12]) if np.any(d > 1e-12) else 1.0
    scale = np.where(d[:, -1] > 1e-12, d[:, -1], fb)
    w = np.exp(-d ** 2 / (2 * scale[:, None] ** 2 + 1e-12))
    w /= w.sum(axis=1, keepdims=True)

    def energy(v):
        lap = v - (w * v[ix]).sum(axis=1)
        return float(np.average(lap ** 2, weights=counts))

    raw = energy(values)
    rng = np.random.default_rng(seed)
    null = np.array([energy(rng.permutation(values)) for _ in range(n_perm)])
    mean, sd = null.mean(), (null.std() or 1e-12)
    return dict(q=raw / mean if mean > 0 else 0.0,          # support-invariant Q_S
                z=(raw - mean) / sd, states=len(states))


# ----- Part A -----

def orthogonality():
    print("=== Part A: orthogonality of residual-fit interaction components ===")
    data = fetch_california_housing(as_frame=True)
    names = list(data.data.columns)
    X, y = data.data.copy(), np.log1p(data.target.to_numpy(float))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=C.TEST_SIZE,
                                          random_state=C.RANDOM_STATE)
    Xtr, Xte = Xtr.reset_index(drop=True), Xte.reset_index(drop=True)
    cols = [f"f{i}" for i in range(X.shape[1])]
    raw_to_col = dict(zip(names, cols))
    A, B = Xtr.copy(), Xte.copy(); A.columns = cols; B.columns = cols

    # constraints as integer indices (name-based constraints break under
    # this env's pandas/XGBoost feature-name mapping)
    gam = xgb(10, IC([[i] for i in range(len(cols))])).fit(A, ytr)
    screened = pd.read_csv(C.OUT_DIR / "california_housing_screened_pairs.csv").head(6)
    pairs = [(raw_to_col[r.feature_1], raw_to_col[r.feature_2])
             for r in screened.itertuples(index=False)]
    deep = xgb(6, IC([[int(a[1:]), int(b[1:])] for a, b in pairs])).fit(A, ytr - gam.predict(A))

    rng = np.random.default_rng(C.RANDOM_STATE)
    bg = A.iloc[rng.choice(len(A), size=96, replace=False)].reset_index(drop=True)
    base_g = gam.predict(bg)[None, :]
    base_d = deep.predict(bg)[None, :]

    mains = {names[i]: main_effect(gam, B, bg, c, base_g) for i, c in enumerate(cols)}
    inters = {f"{r.feature_1}x{r.feature_2}":
              pair_component(deep, B, bg, a, b, base_d)[0]
              for (a, b), r in zip(pairs, screened.itertuples(index=False))}

    def corr(u, v):
        return float(np.corrcoef(u, v)[0, 1])

    mi = [abs(corr(m, g)) for m in mains.values() for g in inters.values()]
    ii = [abs(corr(inters[a], inters[b])) for a in inters for b in inters if a < b]
    centering = max(abs(g.mean()) / (g.std() + 1e-12) for g in inters.values())
    print(f"  main-effect vs interaction |corr|:  max={max(mi):.3f}  mean={np.mean(mi):.3f}")
    print(f"  interaction vs interaction |corr|:  max={max(ii):.3f}  mean={np.mean(ii):.3f}")
    print(f"  interaction centering |mean|/std:   max={centering:.3f}")
    print("  (near-zero main-vs-interaction corr => residual fit yields "
          "approx. orthogonal A_S)")
    pd.DataFrame({"pair": list(inters),
                  "max_corr_with_any_main": [max(abs(corr(m, g)) for m in mains.values())
                                             for g in inters.values()]}
                 ).to_csv(C.OUT_DIR / "iigbm_orthogonality.csv", index=False)


# ----- Part B -----

def make_synthetic(n, seed):
    rng = np.random.default_rng(seed)
    U = rng.uniform(0, 1, size=(n, 6))
    u1, u2, u3, u4, u5, u6 = U.T
    main = (1.2 * np.sin(2 * np.pi * u5) + 0.8 * (u6 - 0.5)
            + 0.5 * (u1 - 0.5) + 0.5 * (u2 - 0.5) + 0.3 * (u3 - 0.5) + 0.3 * (u4 - 0.5))
    broad = 3.0 * (u1 - 0.5) * (u2 - 0.5)                      # smooth bilinear
    local = 1.2 * np.exp(-((u3 - 0.8) ** 2 + (u4 - 0.8) ** 2) / (2 * 0.06 ** 2))  # sharp bump
    y = main + broad + local + rng.normal(0, 0.05, n)
    X = pd.DataFrame(U, columns=[f"f{i}" for i in range(6)])
    return X, y


def synthetic_curvature(seed, k=C.KNN, M=96):
    X, y = make_synthetic(8000, seed)
    cols = list(X.columns)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed)
    Xtr, Xte = Xtr.reset_index(drop=True), Xte.reset_index(drop=True)
    gam = xgb(6, IC([[i] for i in range(len(cols))])).fit(Xtr, ytr)
    deep = xgb(4, IC([[0, 1], [2, 3]])).fit(Xtr, ytr - gam.predict(Xtr))
    rng = np.random.default_rng(seed)
    bg = Xtr.iloc[rng.choice(len(Xtr), size=M, replace=False)].reset_index(drop=True)
    base = deep.predict(bg)[None, :]
    out = {}
    for name, (a, b) in [("broad(f0,f1)", ("f0", "f1")), ("local(f2,f3)", ("f2", "f3"))]:
        comp, psi = pair_component(deep, Xte, bg, a, b, base)
        res = curvature_null(psi / np.sqrt(M), comp, k=k, seed=seed)
        out[name] = dict(A=float(np.mean(comp ** 2)), **res)
    return out


def curvature_validation():
    print("\n=== Part B: Q_S validation on planted smooth vs localized interactions ===")
    r0 = synthetic_curvature(seed=0)
    print("  single run (seed 0):")
    for name, r in r0.items():
        print(f"    {name:14s} A={r['A']:.4f}  Q={r['q']:.4f}  z={r['z']:+.1f}  "
              f"states={r['states']}")

    seeds = range(5)
    stab = {n: [] for n in r0}
    for s in seeds:
        for n, r in synthetic_curvature(seed=s).items():
            stab[n].append(r["q"])
    print("  stability over 5 seeds  (mean +/- sd):")
    for n, v in stab.items():
        print(f"    {n:14s} Q = {np.mean(v):.3f} +/- {np.std(v):.3f}")

    print("  sensitivity to k (seed 0):")
    for k in (10, 15, 25):
        r = synthetic_curvature(seed=0, k=k)
        print(f"    k={k:2d}  Q_broad={r['broad(f0,f1)']['q']:.3f}  "
              f"Q_local={r['local(f2,f3)']['q']:.3f}")

    rows = [dict(component=n, seed=s, q=q) for n, v in stab.items()
            for s, q in zip(seeds, v)]
    pd.DataFrame(rows).to_csv(C.OUT_DIR / "iigbm_qs_validation.csv", index=False)


if __name__ == "__main__":
    orthogonality()
    curvature_validation()
    print(f"\nsaved -> {C.OUT_DIR / 'iigbm_orthogonality.csv'}, "
          f"{C.OUT_DIR / 'iigbm_qs_validation.csv'}")
