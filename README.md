# GBDT-Curvature

A curvature framework for gradient-boosted decision trees. Curvature is the model's
second-order structure: how each feature's effect bends along its own axis (the
diagonal of a discrete Hessian) and how two features bend jointly (the
off-diagonal, which is the interaction). Tree depth sets the order of curvature the
model can carry, and every object is read from the fitted model or its induced
kernel, with no surrogate.

## Two papers

- **`gbdt_curvature.tex`** — *Curvature of Gradient-Boosted Trees: Main Effects as
  Bending, Interactions as Off-Diagonal Curvature, and Depth as Curvature Order.*
  Defines curvature by finite-difference operators against a background (main
  effect = first difference, interaction = mixed second difference, self-curvature
  = self second difference), builds the importance and Hessian matrices, gives the
  model-induced-kernel view (the kernel diagonal `K_S(x,x)` is a per-point
  interaction-energy field), and shows depth corresponds to interaction order.
  Validated on simulated data plus California housing and bikeshare.

- **`iigbm_model_induced_curvature.tex`** — *Inherently Interpretable GBM.* A staged
  additive-then-interaction model, PSD component kernels induced by the fitted deep
  model, and a support-invariant component curvature `Q_S` paired with importance
  `A_S`.

The two use the same object with different normalizations: the curvature paper's
self-curvature `A_jj` is a raw Hessian energy (zero for binary features); the IIGBM
paper's `Q_S` is a permutation-null-normalized roughness (near 1 for low-cardinality
components).

## Method

1. **Discrete curvature.** Against a background sample, `Ψⱼ` is the first difference
   (main effect), `Ψⱼₖ` the mixed second difference (interaction, off-diagonal
   Hessian) and `Ψⱼⱼ` the self second difference (main-effect bending). Order-`|S|`
   effects are `|S|`-fold differences; this is the functional ANOVA grading read as
   curvature.
2. **Model-induced kernel.** A fitted tree induces a PSD leaf co-membership kernel
   `K_T(x,x') = 1{leaf(x)=leaf(x')}`. The same recipe on the mixed-difference
   response gives a per-component kernel `K_S`; the model kernel decomposes as
   `K = Σ_S K_S`, an interaction is a non-separable block, and `K_S(x,x)` maps where
   an interaction fires.
3. **Depth and order.** Depth bounds the order of mixed difference a model can
   represent, so the depth at which held-out error flattens measures a dataset's
   effective interaction order (near 2 for California, near 3 for bikeshare).
4. **Staged IIGBM.** Additive GAM, depth-2 interaction screen, constrained deep
   residual model; component importance `A_S` and support-invariant curvature `Q_S`.

## Files

| File | Role |
|------|------|
| `gbdt_curvature.tex` | Paper: curvature framework (Hessian, kernel view, depth-order) |
| `iigbm_model_induced_curvature.tex` | Paper: IIGBM (staged model, component kernels, `Q_S`) |
| `IIGBM_tutorial.ipynb` | Tutorial: trees→kernel, staged model, components, curvature, plots |
| `kernel_view_test.py` | Simulated: interaction = off-diagonal curvature; kernel detection, recovery, localization |
| `depth_order_test.py` | Simulated: depth = interaction order (each order captured at its depth) |
| `gbdt_laplacian_test.py` | Simulated: finite-difference curvature vs the GBDT-kernel graph Laplacian (which does not localize interactions) |
| `invariance_test.py` | Simulated: the interaction is reparametrization-invariant (metric-free), the self-curvature is not |
| `real_curvature_experiments.py` | California + bikeshare: importance and Hessian matrices, interaction fields, depth-order curves |
| `california_housing_staged_interactions.py` | Stage I–III on California; writes screened pairs + MAE |
| `california_housing_deep_anova_kernel_curvature.py` | Core kernel + curvature functions; California component tables |
| `bikeshare_deep_anova_kernel_curvature.py` | Same analysis on UCI hourly bikeshare |
| `iigbm_diagnostics.py` | Orthogonality test and `Q_S` validation (synthetic + permutation null + stability) |
| `make_tutorial.py` | Regenerates the tutorial notebook |
| `config.py`, `data.py` | Shared configuration and bikeshare loading |

## Environment

Use Python 3.11 with **xgboost 3.0** and **pandas 2.2**. Under pandas 3.0, XGBoost
drops DataFrame feature names and name-based interaction constraints fail.

```
pip install "xgboost==3.0.*" "pandas==2.2.*" "scikit-learn>=1.4" numpy scipy matplotlib jupyter
```

## Running

```bash
# Curvature paper experiments
python kernel_view_test.py            # interaction = off-diagonal curvature (simulated)
python depth_order_test.py            # depth = interaction order (simulated)
python gbdt_laplacian_test.py         # finite differences vs GBDT-kernel Laplacian
python invariance_test.py             # metric-free interaction vs metric-dependent self-curvature
python real_curvature_experiments.py  # matrices + fields + depth-order (California, bikeshare)

# IIGBM pipeline (run staged first: it writes the screened-pairs file the next script reads)
python california_housing_staged_interactions.py
python california_housing_deep_anova_kernel_curvature.py
python bikeshare_deep_anova_kernel_curvature.py
python iigbm_diagnostics.py

# Tutorial
python make_tutorial.py
jupyter nbconvert --to notebook --execute --inplace IIGBM_tutorial.ipynb
```

Figures and tables are written to `artifacts/` (git-ignored). The papers reference
those figures, so run the scripts above before compiling the `.tex`.

## Data

California housing loads from scikit-learn, so its scripts and the tutorial need no
download. Bikeshare uses the UCI Bike Sharing hourly file (`hour.csv`); set its path
in `config.py` (`DATA_PATH`).

## Citation

Cite the papers together with the constructions they build on: Sudjianto & Zhang,
*Generalized Nadaraya–Watson Operators for Learned Kernel Geometry* (SSRN 6823639,
2026) for the model-induced kernel; Hu, Nair, Sudjianto, Zhang & Chen,
*Interpretable Machine Learning based on Functional ANOVA Framework*
(arXiv:2305.15670, 2023) for hierarchical orthogonality.
