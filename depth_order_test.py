"""Test that depth correlates with interaction ORDER (the relationship that
survives, unlike per-point curvature).

Plant a pure 2-way interaction on (u0,u1) and a pure 3-way interaction on
(u2,u3,u4), both centered products so they have no lower-order fANOVA content.
Fit unconstrained XGBoost at depth 1..4 and measure:

  MSE by depth        -- the pair should be captured at depth 2, the triple at 3
  component recovery  -- corr(model |S|-way contrast, planted component) should
                         turn on exactly when depth reaches |S|
"""
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr

import config as C
from california_housing_deep_anova_kernel_curvature import xgb, background_predictions

cols = [f"u{i}" for i in range(6)]


def contrast(model, Xeval, bg, S):
    """|S|-way fANOVA contrast by inclusion-exclusion over subsets of S."""
    n, M = len(Xeval), len(bg)
    base = model.predict(bg)[None, :]
    total = np.zeros((n, M))
    for r in range(len(S) + 1):
        for T in combinations(S, r):
            sign = (-1) ** (len(S) - len(T))
            h = (background_predictions(model, Xeval, bg, [cols[i] for i in T])
                 if T else np.broadcast_to(base, (n, M)))
            total = total + sign * h
    comp = total.mean(axis=1)
    return comp - comp.mean()


def main():
    rng = np.random.default_rng(0)
    n = 8000
    U = rng.uniform(0, 1, (n, 6))
    u = U.T
    X = pd.DataFrame(U, columns=cols)
    mains = sum((0.6 + 0.1 * j) * np.sin(2 * np.pi * u[j]) for j in range(6))
    g2 = 4.0 * (u[0] - .5) * (u[1] - .5)                       # pure 2-way
    g3 = 8.0 * (u[2] - .5) * (u[3] - .5) * (u[4] - .5)         # pure 3-way
    y = mains + g2 + g3 + rng.normal(0, 0.05, n)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    Xtr, Xte = Xtr.reset_index(drop=True), Xte.reset_index(drop=True)
    bg = Xtr.iloc[rng.choice(len(Xtr), 96, replace=False)].reset_index(drop=True)

    g2_te = (4.0 * (Xte.u0 - .5) * (Xte.u1 - .5)).to_numpy()
    g3_te = (8.0 * (Xte.u2 - .5) * (Xte.u3 - .5) * (Xte.u4 - .5)).to_numpy()
    print(f"planted energies:  Var(g2)={np.var(g2_te):.4f}  Var(g3)={np.var(g3_te):.4f}  "
          f"noise Var=0.0025")

    depths = [1, 2, 3, 4]
    rows = []
    for d in depths:
        m = xgb(d).fit(Xtr, ytr)
        mse = float(np.mean((m.predict(Xte) - yte) ** 2))
        rec2 = spearmanr(contrast(m, Xte, bg, (0, 1)), g2_te).statistic
        rec3 = spearmanr(contrast(m, Xte, bg, (2, 3, 4)), g3_te).statistic
        rows.append({"depth": d, "test_MSE": mse, "recover_2way": rec2, "recover_3way": rec3})
        print(f"  depth {d}:  MSE={mse:.4f}  recover(2-way)={rec2:+.3f}  recover(3-way)={rec3:+.3f}")

    tab = pd.DataFrame(rows)
    mse = tab.test_MSE.to_numpy()
    print(f"\nMSE drop depth1->2 = {mse[0]-mse[1]:.4f}  (vs planted Var(g2)={np.var(g2_te):.4f})")
    print(f"MSE drop depth2->3 = {mse[1]-mse[2]:.4f}  (vs planted Var(g3)={np.var(g3_te):.4f})")
    print(f"MSE drop depth3->4 = {mse[2]-mse[3]:.4f}  (should be ~0: no order-4 structure)")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(tab.depth, tab.test_MSE, "o-"); ax[0].set(xlabel="tree depth",
        ylabel="held-out MSE", title="accuracy vs depth"); ax[0].set_xticks(depths)
    ax[1].plot(tab.depth, tab.recover_2way, "o-", label="2-way (u0,u1)")
    ax[1].plot(tab.depth, tab.recover_3way, "s-", label="3-way (u2,u3,u4)")
    ax[1].set(xlabel="tree depth", ylabel="component recovery (Spearman)",
              title="each order turns on at its depth"); ax[1].set_xticks(depths)
    ax[1].axhline(0, color="k", lw=0.5); ax[1].legend()
    fig.tight_layout(); fig.savefig(C.OUT_DIR / "depth_order_test.png", dpi=130)
    print(f"saved -> {C.OUT_DIR / 'depth_order_test.png'}")


if __name__ == "__main__":
    main()
