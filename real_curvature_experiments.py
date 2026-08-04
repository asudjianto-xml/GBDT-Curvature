"""Real-data experiments for the curvature paper: interaction-energy (off-diagonal
curvature) matrix, a local interaction-energy field, and the depth-order curve,
on California housing and UCI bikeshare.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

import config as C
import data as D
from california_housing_deep_anova_kernel_curvature import xgb, background_predictions


def _center(v):
    return v - v.mean()


def self_curvature(model, Xe, bg, h0, a, kind):
    """Energy of the discrete self second difference d2f/da^2 (main-effect bending).
    kind: ('cont',) | ('cyc', P, off) | ('ord', lo, hi) | ('bin',)."""
    if kind[0] == "bin":
        return 0.0                                  # two levels: no second difference
    Xp, Xm = Xe.copy(), Xe.copy()
    if kind[0] == "cont":
        step = 0.25 * Xe[a].std()
        Xp[a] = np.clip(Xe[a] + step, Xe[a].min(), Xe[a].max())
        Xm[a] = np.clip(Xe[a] - step, Xe[a].min(), Xe[a].max())
    elif kind[0] == "cyc":
        P, off = kind[1], kind[2]
        Xp[a] = (Xe[a] - off + 1) % P + off
        Xm[a] = (Xe[a] - off - 1) % P + off
    elif kind[0] == "ord":
        lo, hi = kind[1], kind[2]
        Xp[a] = np.minimum(Xe[a] + 1, hi); Xm[a] = np.maximum(Xe[a] - 1, lo)
    hp = background_predictions(model, Xp, bg, [a])
    hm = background_predictions(model, Xm, bg, [a])
    psi = hp - 2 * h0 + hm                           # discrete second derivative field
    return float(np.var(_center(psi.mean(axis=1))))


def run(name, Xtr, ytr, Xte, yte, feats, ftypes):
    print(f"\n===== {name} =====")
    # depth-order: unconstrained model at each depth
    mse = {}
    for d in range(1, 7):
        m = xgb(d).fit(Xtr, ytr)
        mse[d] = float(np.mean((m.predict(Xte) - yte) ** 2))
    print("depth-order held-out MSE:", {d: round(v, 4) for d, v in mse.items()})

    # curvature matrix on an unconstrained deep model
    model = xgb(6).fit(Xtr, ytr)
    Xe = Xte.sample(min(1500, len(Xte)), random_state=0).reset_index(drop=True)
    bg = Xtr.sample(96, random_state=0).reset_index(drop=True)
    base = model.predict(bg)[None, :]
    h = {a: background_predictions(model, Xe, bg, [a]) for a in feats}
    main = {a: _center((h[a] - base).mean(axis=1)) for a in feats}

    p = len(feats)
    Aimp = np.zeros((p, p))     # importance matrix: diagonal = main-effect energy A_j
    Ahes = np.zeros((p, p))     # Hessian matrix:   diagonal = self second difference
    for i, a in enumerate(feats):
        Aimp[i, i] = float(np.var(main[a]))
        Ahes[i, i] = self_curvature(model, Xe, bg, h[a], a, ftypes[a])
    pair_e = {}
    for i, a in enumerate(feats):
        for j in range(i + 1, p):
            b = feats[j]
            psi = background_predictions(model, Xe, bg, [a, b]) - h[a] - h[b] + base
            e = float(np.var(_center(psi.mean(axis=1))))     # interaction (shared off-diagonal)
            Aimp[i, j] = Aimp[j, i] = e
            Ahes[i, j] = Ahes[j, i] = e
            pair_e[(a, b)] = e
    top = sorted(pair_e.items(), key=lambda kv: -kv[1])[:8]
    print("top interaction pairs (off-diagonal, mixed-curvature energy):")
    for (a, b), e in top:
        print(f"    {a} x {b}: {e:.5f}")
    diag = sorted(((a, Aimp[i, i], Ahes[i, i]) for i, a in enumerate(feats)),
                  key=lambda t: -t[1])
    print("main effects (importance A_j, self-curvature):")
    for a, imp, hes in diag:
        print(f"    {a}: A_j={imp:.5f}  self-curv={hes:.5f}")

    # local interaction-energy field for the top pair
    (a, b), _ = top[0]
    psi_top = background_predictions(model, Xe, bg, [a, b]) - h[a] - h[b] + base
    field = (psi_top ** 2).mean(axis=1)

    slug = name.lower().replace(" ", "_")
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    for k, (Mtx, ttl) in enumerate([
            (Aimp, "importance matrix (diag = main-effect A_j)"),
            (Ahes, "Hessian matrix (diag = self-curvature)")]):
        im = ax[k].imshow(Mtx, cmap="viridis")
        ax[k].set_xticks(range(p)); ax[k].set_xticklabels(feats, rotation=90, fontsize=7)
        ax[k].set_yticks(range(p)); ax[k].set_yticklabels(feats, fontsize=7)
        ax[k].set_title(f"{name}: {ttl}")
        fig.colorbar(im, ax=ax[k], fraction=0.046)
    s = ax[2].scatter(Xe[a], Xe[b], c=field, s=8, cmap="magma")
    ax[2].set_xlabel(a); ax[2].set_ylabel(b)
    ax[2].set_title(f"local interaction field  K(x,x): {a} x {b}")
    fig.colorbar(s, ax=ax[2], fraction=0.046)
    fig.tight_layout(); fig.savefig(C.OUT_DIR / f"curvature_{slug}.png", dpi=130)

    pd.DataFrame([{"feature_1": a, "feature_2": b, "interaction_energy": e}
                 for (a, b), e in sorted(pair_e.items(), key=lambda kv: -kv[1])]
                 ).to_csv(C.OUT_DIR / f"curvature_pairs_{slug}.csv", index=False)
    pd.DataFrame([{"depth": d, "mse": v} for d, v in mse.items()]
                 ).to_csv(C.OUT_DIR / f"depth_order_{slug}.csv", index=False)
    print(f"saved -> curvature_{slug}.png, curvature_pairs_{slug}.csv, depth_order_{slug}.csv")


def main():
    # California housing (all continuous)
    cal = fetch_california_housing(as_frame=True)
    X, y = cal.data.copy(), np.log1p(cal.target.to_numpy(float))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=C.RANDOM_STATE)
    ft_cal = {c: ("cont",) for c in X.columns}
    run("California Housing", Xtr.reset_index(drop=True), ytr,
        Xte.reset_index(drop=True), yte, list(X.columns), ft_cal)

    # Bikeshare (raw features; log1p cnt)
    Xb, yb, yb_raw = D.load()
    s = D.split(Xb, yb, yb_raw)
    ft_bike = {"yr": ("bin",), "mnth": ("cyc", 12, 1), "hr": ("cyc", 24, 0),
               "holiday": ("bin",), "weekday": ("cyc", 7, 0), "workingday": ("bin",),
               "weathersit": ("ord", 1, 4), "temp": ("cont",), "hum": ("cont",),
               "windspeed": ("cont",)}
    run("Bikeshare", s["X_train"], s["y_train"], s["X_test"], s["y_test"],
        list(Xb.columns), ft_bike)


if __name__ == "__main__":
    main()
