"""Compare two curvatures of the same fitted model on planted-interaction data:

  finite-difference field  K_{01}(x,x)   -- coordinate-resolved (off-diagonal Hessian)
  GBDT-kernel Laplacian    |(Lf)(x)|      -- isotropic (trace of the Hessian)

Claim: the finite-difference field localizes the planted (u0,u1) interaction; the
kernel Laplacian tracks total bending and does not isolate the pair.
"""
import numpy as np
import pandas as pd
import xgboost as xgblib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr

from california_housing_deep_anova_kernel_curvature import xgb, background_predictions

cols = [f"u{i}" for i in range(5)]


def true_bump(a, b):
    return 1.5 * np.exp(-(((a - 0.75) ** 2 + (b - 0.75) ** 2) / (2 * 0.10 ** 2)))


def main():
    rng = np.random.default_rng(0)
    n = 8000
    U = rng.uniform(0, 1, (n, 5)); u = U.T
    X = pd.DataFrame(U, columns=cols)
    mains = sum((0.7 + 0.1 * j) * np.sin(2 * np.pi * u[j]) for j in range(5))  # curved mains
    y = mains + true_bump(u[0], u[1]) + rng.normal(0, 0.03, n)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    Xtr, Xte = Xtr.reset_index(drop=True), Xte.reset_index(drop=True)

    model = xgb(6).fit(Xtr, ytr)
    f = model.predict(Xte)

    # (1) finite-difference interaction field for (u0,u1)
    bg = Xtr.iloc[rng.choice(len(Xtr), 96, replace=False)].reset_index(drop=True)
    base = model.predict(bg)[None, :]
    h0 = background_predictions(model, Xte, bg, ["u0"])
    h1 = background_predictions(model, Xte, bg, ["u1"])
    psi01 = background_predictions(model, Xte, bg, ["u0", "u1"]) - h0 - h1 + base
    fd_field = (psi01 ** 2).mean(axis=1)

    # (2) GBDT leaf co-membership graph, Laplacian of the predictions
    leaves = model.get_booster().predict(xgblib.DMatrix(Xte), pred_leaf=True)
    d, idx = NearestNeighbors(n_neighbors=16, metric="hamming").fit(leaves).kneighbors(leaves)
    d, idx = d[:, 1:], idx[:, 1:]
    w = (1.0 - d); w /= w.sum(axis=1, keepdims=True)     # leaf-proximity weights
    Lf = np.abs(f - (w * f[idx]).sum(axis=1))            # |graph Laplacian of f|

    # (3) total local bending (Hessian trace proxy): sum of self second differences
    trace = np.zeros(len(Xte))
    for c in cols:
        step = 0.25 * Xte[c].std()
        Xp = Xte.copy(); Xp[c] = np.clip(Xte[c] + step, Xte[c].min(), Xte[c].max())
        Xm = Xte.copy(); Xm[c] = np.clip(Xte[c] - step, Xte[c].min(), Xte[c].max())
        trace += np.abs(model.predict(Xp) - 2 * f + model.predict(Xm))

    # purified true interaction energy (same background contrast on the true bump)
    x0 = Xte.u0.to_numpy()[:, None]; x1 = Xte.u1.to_numpy()[:, None]
    b0 = bg.u0.to_numpy()[None, :]; b1 = bg.u1.to_numpy()[None, :]
    sv = true_bump(Xte.u0.to_numpy(), Xte.u1.to_numpy())[:, None]
    psi_true = sv - true_bump(x0, b1) - true_bump(b0, x1) + true_bump(b0, b1)
    true_energy = (psi_true ** 2).mean(axis=1)

    print("localize the planted (u0,u1) interaction:")
    print(f"  finite-difference field  vs true interaction : {spearmanr(fd_field, true_energy).statistic:+.3f}")
    print(f"  GBDT-kernel Laplacian     vs true interaction : {spearmanr(Lf, true_energy).statistic:+.3f}")
    print("what the kernel Laplacian tracks instead:")
    print(f"  GBDT-kernel Laplacian     vs Hessian trace     : {spearmanr(Lf, trace).statistic:+.3f}")
    print(f"  GBDT-kernel Laplacian     vs finite-diff field : {spearmanr(Lf, fd_field).statistic:+.3f}")


if __name__ == "__main__":
    main()
