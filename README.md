# F4F TreeLens — reproduction and improvement

An independent reproduction of Farmers for Forests' TreeLens DBH-to-carbon pipeline (August 2025), replacing the pooled GPR baseline with a hierarchical Bayesian model. Two pipeline-level findings (Jensen back-transform; 80% credit haircut) are independent of the model choice.

## Contents

- `hierarchical_dbh_v3.py` — current model. PyMC, hierarchical with partial pooling across species and districts, quadratic crown and age predictors.
- `hierarchical_dbh_v31.py` — same model, with the Jensen-correct back-transform from log(DBH) posterior to plot-level CO₂.
- `hierarchical_dbh_model.py`, `hierarchical_dbh_v2.py` — earlier iterations, kept for history.
- `f4f_ground_data_13Oct25.csv` — public ground-truth release, used unchanged.
- `out_v3/` — saved MCMC trace, residuals, per-species posterior parameters.
- `make_figures.py` — regenerates the six paper figures from `out_v3/` into `figures/`.
- `figures/` — PNGs included by the white paper.
- `white_paper/` — white paper (LaTeX source + PDF).

## Model iterations

The hierarchical model was developed in three steps; v3 is the current model. v31 is a diagnostic add-on that reuses v3's posterior.

### v1 → v2

| | v1 (`hierarchical_dbh_model.py`) | v2 (`hierarchical_dbh_v2.py`) |
|---|---|---|
| Predictors | crown, height, species, district | + log(age), + crown² (per-species curvature) |
| Species coefficients | 3 (α intercept, β crown, γ height) | 5 (adds ζ curvature, η age) |
| Data | full cleaned dataset (~7,100 trees) | requires non-missing age → ~6,364 trees, 19 species |
| Carbon calc | `carbon_from_dbh_samples()` — full Verra chain helper | none |
| Diagnostics | basic — RMSE on test set | full audit: train-test gap, residuals by size/species/district, calibration coverage, LODO, species-params printout |

### v3 and v31

| | v3 (`hierarchical_dbh_v3.py`) | v31 (`hierarchical_dbh_v31.py`) |
|---|---|---|
| What it does | Fits the hierarchical model | Reuses v3's posterior to fix the carbon back-transform |
| MCMC | yes (~5–8 min) | no (~30 sec) |
| Changes from previous | drops bamboo; age slope ≥ 0; age feature masked for low-CV species; sapling floor noted | model unchanged; back-transform uses per-sample biomass instead of median-then-transform |
| Output | `out_v3/` (trace + residuals + species params + train/test splits) | `out_v3/v3_1_corrected_carbon.csv` |
| Plot CO₂ bias | −10% (uncorrected diagnostic) | +4.5% (Jensen-corrected) |

## Setup and run

```sh
uv sync                                   # installs dependencies into .venv
.venv/bin/python hierarchical_dbh_v3.py   # fits the model, saves posterior to out_v3/
.venv/bin/python hierarchical_dbh_v31.py  # reuses out_v3/ for Jensen-corrected carbon
```

Run v3 first — v31 reuses the posterior saved in `out_v3/trace_v3.nc` and will fail if it doesn't exist. Model fit takes ~3–7 minutes on a laptop (4 chains × 1,000 draws); v31 runs in ~30 seconds.

## Writeup

See `white_paper/white_paper.pdf` for the full technical report.

## Data and prior work

Ground truth from the [F4F public datasets repository](https://github.com/Farmers-For-Forests/public-datasets) (13 October 2025 release), used without modification. Original methodology: [TreeLens v1.0 white paper](https://www.farmersforforests.org/FFF/images/our-tech/tree-lens-white-paper.pdf).
