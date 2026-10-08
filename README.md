# Uncertainty propagation in drone-based agroforestry carbon estimation

Code for a study of what happens *after* the tree-diameter (DBH) model in a drone-based
carbon MRV pipeline: back-transforming predictions through the biomass allometry,
aggregating to plot totals, and turning those totals into credit claims. Seven DBH models,
from a pooled Gaussian process to hierarchical Bayesian models, are compared on the same
open ground-truth data.

Paper: [`white_paper/main.pdf`](white_paper/main.pdf). Working notes: [`FINDINGS.md`](FINDINGS.md).

## Contents

| File | What it does | Output |
|---|---|---|
| `hierarchical_dbh_v3.py` | Fits the full hierarchical model (species, district, age); defines the cleaning and train/test split | `out_v3/` |
| `baseline_gpr.py` | Pooled GPR and OLS baselines on the same split | `out_baseline/` |
| `ablation_hier.py` | Hierarchical model with covariates removed (drone-only, species, species + district) | `out_ablation/` |
| `pseudo_plot_eval.py` | Plot-level evaluation of credit-issuance policies; optional low-rainfall allometry (`brown_eq322`) | `out_plots/`, `out_plots_eq322/` |
| `grouped_split_eval.py` | Same evaluation with whole plots held out (5-fold, models refit per fold) | `out_grouped/` |
| `make_paper_figures.py` | Figures for the paper | `figures/paper_*.pdf` |
| `hierarchical_dbh_v31.py` | Shared helpers (posterior prediction, CO₂ conversion) | — |
| `hierarchical_dbh_model.py`, `hierarchical_dbh_v2.py`, `make_figures.py` | Earlier iterations, kept for history | `out_v2/` |

## Run

```sh
uv sync
.venv/bin/python hierarchical_dbh_v3.py     # ~5 min; everything below reuses out_v3/
.venv/bin/python baseline_gpr.py            # ~5 min
.venv/bin/python ablation_hier.py           # ~5 min
.venv/bin/python pseudo_plot_eval.py        # ~2 min (needs out_ablation/)
.venv/bin/python grouped_split_eval.py      # ~45-60 min
.venv/bin/python make_paper_figures.py      # ~5 min
```

Posterior traces (`*.nc`) are not tracked in git; `hierarchical_dbh_v3.py` and
`ablation_hier.py` regenerate them.

## Data

Ground truth: the 13 October 2025 public release of the
[Farmers for Forests datasets repository](https://github.com/Farmers-For-Forests/public-datasets)
(`f4f_ground_data_13Oct25.csv`), used without modification. Biomass uses Eq. 3.2.1 of
Brown (1997), *FAO Forestry Paper 134*, with the CO₂ conversion of Verra VMD0001.

## License

MIT
