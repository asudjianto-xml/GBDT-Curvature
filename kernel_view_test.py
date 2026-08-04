"""Test the model-induced-kernel view of interactions on synthetic data.

Plant one localized interaction on (u0,u1) and leave (u2,u3) additive. Fit the
staged model with interaction constraints on both pairs, then read the component
kernels. Claims tested:

  detection    : A_01 (planted) >> A_23 (permitted but absent)
  localization : the kernel diagonal K_S(x,x)=||Psi_S(x)||^2/M is large exactly
                 where the true interaction is active
  recovery     : the estimated component recovers the planted mixed difference
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr

import config as C
from california_housing_deep_anova_kernel_curvature import (
    xgb, background_predictions, SumModel,
)


def IC(groups):
    return "[" + ",".join("[" + ",".join(map(str, g)) + "]" for g in groups) + "]"


def true_bump(a, b):
    return 1.5 * np.exp(-(((a - 0.75) ** 2 + (b - 0.75) ** 2) / (2 * 0.10 ** 2)))


def main():
    rng = np.random.default_rng(0)
    n = 8000
    U = rng.uniform(0, 1, (n, 5))
    u = U.T
    cols = [f"u{i}" for i in range(5)]
    X = pd.DataFrame(U, columns=cols)
    mains = (1.0 * np.sin(2 * np.pi * u[4]) + 0.8 * (u[0] - .5) + 0.8 * (u[1] - .5)
             + 0.5 * (u[2] - .5) + 0.5 * (u[3] - .5))
    y = mains + true_bump(u[0], u[1]) + rng.normal(0, 0.03, n)   # interaction only on (u0,u1)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    Xtr, Xte = Xtr.reset_index(drop=True), Xte.reset_index(drop=True)

    gam = xgb(8, IC([[i] for i in range(5)])).fit(Xtr, ytr)
    deep = xgb(4, IC([[0, 1], [2, 3]])).fit(Xtr, ytr - gam.predict(Xtr))  # real + null pair
    full = SumModel(gam, deep)
    print(f"MAE  gam={np.abs(gam.predict(Xte)-yte).mean():.4f}  "
          f"full={np.abs(full.predict(Xte)-yte).mean():.4f}")

    bg = Xtr.iloc[rng.choice(len(Xtr), 96, replace=False)].reset_index(drop=True)
    base = full.predict(bg)[None, :]

    def component(a, b):
        hab = background_predictions(full, Xte, bg, [cols[a], cols[b]])
        ha = background_predictions(full, Xte, bg, [cols[a]])
        hb = background_predictions(full, Xte, bg, [cols[b]])
        psi = hab - ha - hb + base
        comp = psi.mean(axis=1); comp = comp - comp.mean()
        diag = (psi ** 2).mean(axis=1)          # kernel diagonal K_S(x,x)
        return comp, diag

    comp01, diag01 = component(0, 1)             # planted interaction
    comp23, diag23 = component(2, 3)             # permitted but absent

    # ground truth: the PURIFIED interaction of the planted bump under the SAME
    # background contrast (removes the main-effect projections the GAM absorbed).
    x0 = Xte.u0.to_numpy()[:, None]; x1 = Xte.u1.to_numpy()[:, None]
    b0 = bg.u0.to_numpy()[None, :]; b1 = bg.u1.to_numpy()[None, :]
    self_val = true_bump(Xte.u0.to_numpy(), Xte.u1.to_numpy())[:, None]
    psi_true = self_val - true_bump(x0, b1) - true_bump(b0, x1) + true_bump(b0, b1)
    comp_true = psi_true.mean(axis=1); comp_true = comp_true - comp_true.mean()
    diag_true = (psi_true ** 2).mean(axis=1)

    print(f"detection    A_01(planted)={np.var(comp01):.5f}   "
          f"A_23(absent)={np.var(comp23):.5f}   ratio={np.var(comp01)/max(np.var(comp23),1e-9):.0f}x")
    print(f"recovery     spearman(est comp_01, TRUE purified interaction) = "
          f"{spearmanr(comp01, comp_true).statistic:+.3f}")
    print(f"localization spearman(kernel diag_01, TRUE interaction diagonal) = "
          f"{spearmanr(diag01, diag_true).statistic:+.3f}")
    print(f"null field   mean diag_23={diag23.mean():.2e}   vs mean diag_01={diag01.mean():.2e}")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for a, (xx, yy, c, t) in zip(ax, [
        (Xte.u0, Xte.u1, comp_true, "true purified interaction"),
        (Xte.u0, Xte.u1, comp01, "estimated component (kernel)"),
        (Xte.u0, Xte.u1, diag01, "kernel diagonal K_01(x,x)")]):
        s = a.scatter(xx, yy, c=c, s=7, cmap="viridis")
        a.set_title(t); a.set_xlabel("feature a"); a.set_ylabel("feature b")
        fig.colorbar(s, ax=a)
    fig.tight_layout(); fig.savefig(C.OUT_DIR / "kernel_view_test.png", dpi=130)
    print(f"saved -> {C.OUT_DIR / 'kernel_view_test.png'}")


if __name__ == "__main__":
    main()
