"""Metric dependence of the two curvatures under a monotone feature reparametrization.

Trees are invariant to monotone transforms of a split feature, so squaring u0 and
u2 leaves the fitted function unchanged. The interaction contrast substitutes
values and should be invariant; the self second difference uses a step in the
coordinate's own scale and should change.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from california_housing_deep_anova_kernel_curvature import xgb, background_predictions

cols = [f"u{i}" for i in range(4)]


def interaction_energy(model, Xe, bg, a, b):
    base = model.predict(bg)[None, :]
    ha = background_predictions(model, Xe, bg, [a])
    hb = background_predictions(model, Xe, bg, [b])
    psi = background_predictions(model, Xe, bg, [a, b]) - ha - hb + base
    comp = psi.mean(axis=1)
    return float(np.var(comp - comp.mean()))


def self_curvature(model, Xe, bg, a):
    h0 = background_predictions(model, Xe, bg, [a])
    step = 0.25 * Xe[a].std()
    Xp = Xe.copy(); Xp[a] = np.clip(Xe[a] + step, Xe[a].min(), Xe[a].max())
    Xm = Xe.copy(); Xm[a] = np.clip(Xe[a] - step, Xe[a].min(), Xe[a].max())
    psi = background_predictions(model, Xp, bg, [a]) - 2 * h0 + background_predictions(model, Xm, bg, [a])
    comp = psi.mean(axis=1)
    return float(np.var(comp - comp.mean()))


def evaluate(X, y, tag):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    Xtr, Xte = Xtr.reset_index(drop=True), Xte.reset_index(drop=True)
    model = xgb(6).fit(Xtr, ytr)
    bg = Xtr.iloc[np.random.default_rng(0).choice(len(Xtr), 96, replace=False)].reset_index(drop=True)
    Xe = Xte.iloc[:1500]
    a01 = interaction_energy(model, Xe, bg, "u0", "u1")
    s2 = self_curvature(model, Xe, bg, "u2")
    print(f"  {tag:24s} interaction A_01 = {a01:.5f}   self-curv(u2) = {s2:.5f}")
    return a01, s2


def main():
    rng = np.random.default_rng(0)
    n = 8000
    U = rng.uniform(0, 1, (n, 4)); u = U.T
    bump = 1.5 * np.exp(-(((u[0] - .75) ** 2 + (u[1] - .75) ** 2) / (2 * 0.10 ** 2)))
    y = 1.5 * np.sin(2 * np.pi * u[2]) + 0.6 * (u[3] - .5) + bump + rng.normal(0, 0.03, n)

    X0 = pd.DataFrame(U, columns=cols)                     # original coordinates
    Xt = X0.copy(); Xt["u0"] = X0.u0 ** 2; Xt["u2"] = X0.u2 ** 2   # monotone squash of u0, u2

    print("monotone reparametrization of u0 (interaction feature) and u2 (main effect):")
    a0, s0 = evaluate(X0, y, "original")
    a1, s1 = evaluate(Xt, y, "u0,u2 squared")
    print(f"\ninteraction A_01 change: {abs(a1 - a0) / a0 * 100:5.1f}%   (metric-free, expect ~0)")
    print(f"self-curv(u2)   change: {abs(s1 - s0) / s0 * 100:5.1f}%   (needs a step, expect large)")


if __name__ == "__main__":
    main()
