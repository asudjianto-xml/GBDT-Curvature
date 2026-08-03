# GBDT-Curvature

Read a fitted gradient boosting model through three quantities derived from the model itself:
functional ANOVA **order** (which features act alone or in pairs), component **importance**
`A_S` (how much variation an effect explains), and component **curvature** `Q_S` (whether that
effect is broad and smooth or sharp and local). Importance says which learned effects matter;
curvature says how each behaves locally.

The method and proofs are in [`iigbm_model_induced_curvature.tex`](iigbm_model_induced_curvature.tex).
Start with the tutorial notebook.

## Method

1. **Staged model (IIGBM).** Fit an additive XGBoost GAM (one feature group per tree), screen
   pairwise interactions with a depth-2 model on the GAM residual, then fit a deep residual
   model constrained to the screened pairs. Interaction constraints keep the model readable.
2. **Model-induced kernel.** A fitted tree induces a PSD leaf co-membership kernel
   `K_T(x,x') = 1{leaf(x)=leaf(x')}`. Applying the same recipe to the deep model's fANOVA
   interaction responses gives a per-component kernel `K_S`.
3. **Component curvature.** On the k-NN graph of `K_S`, the graph Laplacian of the component
   values, `c_S = (I-W) g_S`, measures local roughness. `Q_S` normalizes that roughness by its
   mean under random permutations of the values over the graph, so it is comparable across
   components of any cardinality: `Q_S < 1` is smoother than chance, `Q_S ≈ 1` means no
   smoothness signal (rough, or too few states to resolve).

## Files

| File | Role |
|------|------|
| `IIGBM_tutorial.ipynb` | Runnable tutorial: trees→kernel, staged model, components, curvature, plots |
| `california_housing_staged_interactions.py` | Stage I–III on California housing; writes screened pairs + MAE |
| `california_housing_deep_anova_kernel_curvature.py` | Core kernel + curvature functions; California component tables |
| `bikeshare_deep_anova_kernel_curvature.py` | Same analysis on UCI hourly bikeshare |
| `iigbm_diagnostics.py` | Orthogonality test and `Q_S` validation (synthetic + permutation null + stability) |
| `config.py`, `data.py` | Shared configuration and bikeshare loading |
| `iigbm_model_induced_curvature.tex` | The paper |

## Environment

Use Python 3.11 with **xgboost 3.0** and **pandas 2.2**. Under pandas 3.0, XGBoost drops
DataFrame feature names and the name-based interaction constraints fail.

```
pip install "xgboost==3.0.*" "pandas==2.2.*" "scikit-learn>=1.4" numpy matplotlib jupyter
```

## Running

```bash
# California pipeline (writes artifacts/california_housing_screened_pairs.csv, then the tables)
python california_housing_staged_interactions.py
python california_housing_deep_anova_kernel_curvature.py

# Bikeshare (see data note below)
python bikeshare_deep_anova_kernel_curvature.py

# Orthogonality + Q_S validation
python iigbm_diagnostics.py

# Regenerate and run the tutorial
python make_tutorial.py
jupyter nbconvert --to notebook --execute --inplace IIGBM_tutorial.ipynb
```

Run `california_housing_staged_interactions.py` before the California curvature script: it
produces the screened-pairs file the curvature script reads.

## Data

California housing loads from scikit-learn, so the tutorial and California scripts need no
download. Bikeshare uses the UCI Bike Sharing hourly file (`hour.csv`); set its path in
`config.py` (`DATA_PATH`).

## Citation

If you use this, cite the paper (`iigbm_model_induced_curvature.tex`) and the model-induced
kernel construction it builds on: Sudjianto & Zhang, *Generalized Nadaraya–Watson Operators for
Learned Kernel Geometry* (SSRN 6823639, 2026); and the hierarchical-orthogonality result of Hu,
Nair, Sudjianto, Zhang & Chen, *Interpretable Machine Learning based on Functional ANOVA
Framework* (arXiv:2305.15670, 2023).
