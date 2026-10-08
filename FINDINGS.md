# Findings log — pre-submission experiments (ACML 2026 SustainableML workshop)

Reference notes from the experiments run on 2026-10-07 to harden the white paper for the
SustainableML-Afrasia workshop (deadline 8 Oct 2026, 23:59 AoE). All numbers below are on
the **same v3 split** (`out_v3/train_v3.csv` 5,035 trees / `out_v3/test_v3.csv` 1,257 trees,
18 species) unless stated otherwise.

## TL;DR

1. **The −53% RMSE headline does not survive a same-split baseline.** TreeLens-spec GPR
   reproduced on our split scores 2.67–2.87 cm, not 4.78 cm. v3's real gain is **16–22%**.
2. **The gain needs species identification.** With drone + location only (A0), the
   hierarchical model ties GPR (2.83 cm). Species ID gives most of the gain (→ 2.34 cm);
   field age adds the rest (→ 2.24 cm).
3. **Pipeline finding 1 (Jensen back-transform) holds for every model** — the point-estimate
   back-transform under-reports CO₂ by 8–15% under all seven models.
4. **Pipeline finding 2 (80% haircut) is half right.** The 80% rule does leave 12–18% of
   credits on the table in aggregate. But the proposed fix — a posterior p25 — is **not
   audit-safe**: at plot level it over-claims on 46–58% of plots against a nominal 25%.
5. **New, stronger finding: tree-level calibration does not transfer to plots.** Every model
   is slightly *under*-confident per tree (90% intervals cover ~93%), but nominal 80% plot
   intervals cover only 46–67% of pseudo-plots. Plot errors (15–27% mean absolute) do not
   shrink with plot size — they are systematic plot effects no model in this study captures.

## 1. Audit of the existing white paper / repo

| Issue | Where | Status |
|---|---|---|
| Table 1 compares GPR on TreeLens's 3,500-tree data vs. hierarchical on the 7,326-tree data; caption says "same held-out test" | white paper §4.1 | Fixed by §2 below — use same-split numbers |
| No GPR reproduction existed in the repo, though paper Goal 1 claims one | repo | Added `baseline_gpr.py` |
| "Posterior p25/p10" in Fig. 6 are a **bootstrap over trees of point predictions**, not posterior quantiles | `make_figures.py:205-210` | Real v3 posterior: p25 **+1.9%**, p10 **−0.4%** (paper: +1.0%, −2.3%) |
| Text says p25 = +1.0% (over-claims); Fig. 6 caption says it "under-reports by 1.0%" | white paper §6.2 / Fig. 6 | Sign contradiction; moot after §4 |
| Calibration "opposite direction" (GPR 1.08 vs ours 0.88) | white paper Table 1 | Does not hold on same data: GPR slope 0.86 |
| Dataset counts differ: README ~7,100 / ~6,364 / 19 species; paper 7,326 / 6,283 / 18 | README vs paper | Saved split totals **6,292** (5,035 + 1,257), not the paper's 6,283; 18 species is correct. Reconcile paper + README |
| "$100k–500k per 1,000 ha" extrapolation | white paper §6.3 | Unsupported; drop |
| README: "LaTeX source + PDF" but `.tex` was not in repo; `*.tex` and `*.nc` are gitignored | README / `.gitignore` | Files now present locally; still gitignored |
| README run-time: "~5–8 min" vs "~3–7 minutes" | README | Cosmetic |
| **Jensen coefficient wrong:** text simplifies exp((2.32σ)²/2) to exp(1.345σ²); correct is exp(**2.691**σ²). Hierarchical range "5.5–13%" is really 5.7–27% (σ 0.144–0.298). `make_figures.py` uses the correct expression | white paper §5.1–5.2 | Fixed in `main.tex`; observed gaps match corrected formula within 2 pp (§5) |
| GitHub URL + author name in repo | — | Must anonymise for double-blind (anonymous.4open.science) |

## 2. Same-split GPR baseline — `baseline_gpr.py` → `out_baseline/`

TreeLens spec: `ConstantKernel × RBF + WhiteKernel` on standardised (TC, TH), target log(DBH),
`normalize_y=True`, 2 optimiser restarts. Per-tree predictive samples (independent noise).
The saved v3 posterior is re-scored with the same metric code and reproduces the paper's
v3 numbers exactly (2.24 / +4.49% / −10.0% / −16.4%), so the metric code is consistent.

| Same 1,257 test trees | GPR (TC, TH) | GPR (log TC, log TH) | Pooled OLS (log) | Hierarchical v3 |
|---|---|---|---|---|
| RMSE (cm) | 2.87 | 2.67 | 2.84 | **2.24** |
| MAE (cm) | 1.75 | 1.73 | 1.86 | **1.48** |
| MAPE on log(DBH) | 9.56% | 9.56% | 10.05% | 8.59% |
| R² (log) | 0.811 | 0.816 | 0.798 | 0.850 |
| 90% coverage | 93.3% | 93.4% | 93.2% | 93.4% |
| Calibration slope | 0.864 | 0.862 | 0.887 | 0.878 |
| CO₂ point back-transform | −14.9% | −13.5% | −14.8% | −10.0% |
| CO₂ Jensen-correct mean | +2.0% | +2.9% | +3.2% | +4.5% |
| CO₂ posterior p25 | −0.6% | +0.4% | +0.2% | +1.9% |
| CO₂ posterior p10 | −2.7% | −1.6% | −2.0% | −0.4% |
| CO₂ 80% rule | −18.4% | −17.6% | −17.5% | −16.4% |

- GPR MAPE on log(DBH) of 9.56% is close to TreeLens's published 8.59% → reproduction credible.
- Learned kernels: raw `1.19² · RBF(0.935) + White(0.203)`; log `0.942² · RBF(0.946) + White(0.204)`.
- Jensen gap under GPR (−13.5 to −14.9%) sits inside the paper's analytical 11–18% range.
- Aggregate test-set CO₂: pooled models land *closer* to truth (+2–3%) than v3 (+4.5%).

## 3. Covariate ablation — `ablation_hier.py` → `out_ablation/`

v3's priors and structure with terms switched off. NUTS, 4 chains × 1,000 draws, target
accept 0.95, seed 42. All arms: max R̂ 1.00 on hyperparameters, 0 divergences, 67–126 s each.

| Model | Deployable from drone? | RMSE | MAE | R² log | Cal. slope | CO₂ point | CO₂ mean | CO₂ p25 | CO₂ p10 | CO₂ 80% |
|---|---|---|---|---|---|---|---|---|---|---|
| GPR (TC, TH) | yes | 2.67–2.87 | 1.73–1.75 | 0.81 | 0.86 | −13.5 to −14.9% | +2.0 to +2.9% | −0.6 to +0.4% | −2.7 to −1.6% | −18% |
| **A0** crown, crown², height + district; species pooled | yes (+ GPS) | 2.83 | 1.81 | 0.804 | 0.870 | −12.1% | +5.2% | +2.2% | −0.2% | −15.8% |
| **A1** species-level crown, crown², height | needs species ID | 2.34 | 1.57 | 0.835 | 0.903 | −8.0% | +9.2% | +5.8% | +3.2% | −12.6% |
| **A2** species + district | needs species ID | 2.39 | 1.57 | 0.833 | 0.891 | −7.8% | +9.2% | +5.9% | +3.3% | −12.6% |
| **v3** species + district + age | needs field age | 2.24 | 1.48 | 0.850 | 0.878 | −10.0% | +4.5% | +1.9% | −0.4% | −16.4% |

- District adds nothing on top of crown/height (A0 ≈ GPR) or on top of species (A2 ≈ A1).
- Species ID: 2.83 → 2.34 cm (−17%). Age on top: 2.39 → 2.24 cm (−6%).
- **Per-tree accuracy and aggregate CO₂ bias move in opposite directions** for A1/A2: better
  RMSE, but +9.2% aggregate over-prediction (vs +2–3% for GPR). Plausibly the Mango/Lemon
  over-prediction from paper §4.3 — **not tested**.

## 4. Pseudo-plot policy evaluation — `pseudo_plot_eval.py` → `out_plots/`

The public data has no plot ID, so test trees are grouped:

- **block_year** (primary): District × Block × Plantation Year, ≥10 test trees → **26 plots,
  1,208 trees**, median 25 trees (10–187). 22 of 26 are in Ahmednagar.
- **latlon3** (robustness): lat/lon rounded to 3 dp (~110 m), ≥10 trees → **39 plots, 894 trees**,
  median 20 (10–66).

Posterior samples are joint within a draw (shared parameters), so plot totals carry
parameter-correlated error. GPR is re-fit and sampled from its **full joint predictive
covariance** (`return_cov=True`), not independent per-tree noise.

Over-claim = policy's claim > field-measured plot total. Wilson 95% CIs in brackets.

### 4.1 Over-claim rate across plots (block_year, n = 26)

| Model | point | mean | p25 (nominal 25%) | p10 (nominal 10%) | 80% rule |
|---|---|---|---|---|---|
| GPR (TC, TH) | 31% [17–50] | 65% [46–81] | 46% [29–65] | 31% [17–50] | 23% [11–42] |
| A0 | 38% [22–57] | 62% [43–78] | 54% [35–71] | 42% [26–61] | 31% [17–50] |
| A1 | 38% [22–57] | 77% [58–89] | 58% [39–74] | 31% [17–50] | 19% [9–38] |
| A2 | 42% [26–61] | 77% [58–89] | 54% [35–71] | 42% [26–61] | 35% [19–54] |
| v3 | 35% [19–54] | 65% [46–81] | 46% [29–65] | 31% [17–50] | 19% [9–38] |

latlon3 (n = 39) agrees: p25 over-claims 41–54%, p10 26–44%, 80% rule 20–41%.

### 4.2 Plot-level error and credits (block_year)

| Model | Mean abs. plot error (p25) | Aggregate err. p25 | 80% rule: left on table | 80% rule: over-claimed |
|---|---|---|---|---|
| GPR | 19.2% | −3.9% | 16.7% | 1.4% |
| A0 | 22.7% | −0.9% | 15.7% | 4.0% |
| A1 | 17.1% | −1.4% | 13.3% | 1.2% |
| A2 | 20.2% | 0.0% | 14.3% | 3.6% |
| v3 | **14.7%** | −1.0% | 14.5% | 1.5% |

"Left on table" / "over-claimed" = sum over plots of under/over-shoot, as % of total true CO₂.

### 4.3 Plot-level interval coverage

| Model | 50% interval (block_year / latlon3) | 80% interval (block_year / latlon3) |
|---|---|---|
| GPR | 35% / 41% | 54% / 59% |
| A0 | 19% / 26% | 46% / 46% |
| A1 | 23% / 44% | 62% / 67% |
| A2 | 27% / 38% | 50% / 67% |
| v3 | 35% / 38% | 58% / 62% |

### 4.4 Reading

- **Tree-level calibration ≠ plot-level calibration.** Per tree, all models are mildly
  under-confident (§2). Per plot, all are strongly over-confident. Summing per-tree errors
  that are independent given the parameters makes the plot posterior too narrow.
- **Plot error is systematic, not sampling noise.** v3's per-plot mean-policy error ranges
  −35% to +55%, and |error| is uncorrelated with plot size (Spearman −0.07). Bigger plots
  do not average it away → a missing plot/farm-level effect (management, site, measurement
  crew), which none of the models has.
- **Consequence for paper finding 2.** "Replace the 80% rule with posterior p25" is not
  supported: p25 over-claims on roughly half of plots. The 80% rule is itself only ~60–80%
  plot-safe (over-claims on 19–41% of plots), while leaving ~13–17% on the table in aggregate. The defensible recommendation
  is **plot-level calibration** — e.g. a plot random effect, or conformal / empirical
  quantile calibration on held-out plots — before any quantile is used for issuance.
- **Hierarchical helps per plot, modestly.** v3 has the lowest mean absolute plot error
  (14.7% vs GPR 19.2% on block_year; 16.3% vs 20.3% on latlon3).

### 4.5 Caveats

- Pseudo-plots, not surveyed plots; 26 / 39 groups → wide CIs (see brackets).
- The split is by tree, so test plots share farms with training trees. True out-of-plot
  error is likely **larger** than reported here, which strengthens §4.4.
- Geographic concentration: 22 / 26 block_year plots are in Ahmednagar.
- Ground truth = Verra chain on field DBH; allometric error is out of scope.

## 5. Back-transform gap: analytic vs observed — `make_paper_figures.py`

Mean ÷ point-estimate test-set CO₂ total, against exp(2.691·σ²) at each model's median
per-tree predictive σ:

| Model | Median σ | Observed gap | Analytic | Point vs truth | Mean vs truth |
|---|---|---|---|---|---|
| GPR | 0.251 | 19.9% | 18.5% | −14.9% | +2.0% |
| Pooled OLS | 0.265 | 21.0% | 20.9% | −14.8% | +3.2% |
| A0 | 0.258 | 19.7% | 19.7% | −12.1% | +5.2% |
| A1 | 0.241 | 18.7% | 16.9% | −8.0% | +9.2% |
| A2 | 0.246 | 18.5% | 17.7% | −7.8% | +9.2% |
| v3 | 0.228 | 16.1% | 15.0% | −10.0% | +4.5% |

Point total is 14–17% below the per-sample mean for every model.

## 6. Workshop paper — `white_paper/main.tex`

- Standalone framing: F4F cited only for the dataset, the GPR spec and the 80% rule.
  No reproduction/critique sections; no speculation about any production pipeline.
- ACML 2026 `jmlr.cls` [wcp] from the official zip. The zip lacks `jmlrutils.sty`;
  it is generated from CTAN `jmlr` v1.30 (same version as the class) and copied
  next to `main.tex`.
- Build: `pdflatex main && bibtex main && pdflatex main && pdflatex main` → 9 pages.
- Figures: `figures/paper_jensen.pdf`, `figures/paper_calibration.pdf`.
- Model names in paper: H-drone = A0, H-species = A1, H-species+district = A2, H-full = v3.

## 7. Implications for the workshop paper

- **Lead with pipeline-level findings across seven models:** (1) Jensen back-transform trap,
  model-independent; (2) tree-level calibration does not transfer to plots, so both the flat
  80% haircut and naive posterior quantiles are miscalibrated as issuance rules.
- **Secondary:** covariate ablation — partial pooling pays off only with species ID (~17%);
  per-tree accuracy does not imply less-biased plot totals.
- **Drop:** −53% headline, calibration "opposite direction", "same audit-safety" p25 claim,
  $/ha extrapolation.
- **Future work becomes concrete:** plot-level random effect + held-out-plot calibration;
  split by plot rather than by tree.

## 8. Reproduce

```sh
uv sync
.venv/bin/python baseline_gpr.py        # ~4.5 min  → out_baseline/baseline_comparison.csv
.venv/bin/python ablation_hier.py       # ~5 min    → out_ablation/{trace_*.nc, ablation_comparison.csv}
.venv/bin/python pseudo_plot_eval.py    # ~2 min    → out_plots/{per_plot.csv, policy_summary.csv}
```

All three read the saved split and posterior from `out_v3/` (requires `out_v3/trace_v3.nc`;
`*.nc` is gitignored). `pseudo_plot_eval.py` also needs `out_ablation/trace_A{0,1,2}.nc`.

## 9. Allometry provenance (verified 2026-10-08)

- AGB = exp(−1.996 + 2.32·ln D) is **Eq. 3.2.1 of Brown (1997), FAO Forestry Paper 134**
  ("dry" zone; revised from Brown et al. 1989 for dry forest in India; DBH 5–40 cm; 28 trees;
  adj. r² 0.89). TreeLens cites it only as "from Verra literature"; VMD0001 supplies only the
  biomass → C → CO₂ conversion.
- Brown restricts Eq. 3.2.1 to dry zones with **> 900 mm/yr** rainfall; below that, use
  Eq. 3.2.2 (AGB ∝ basal area, i.e. b = 2). Ahmednagar (92% of trees) has normal annual
  rainfall **561.6 mm** (ICAR-CRIDA district contingency plan, 2011); 15% of test trees
  are below the 5 cm DBH floor.
- Effect: absolute CO₂ levels are uncertain; the paper's conclusions are not, since they hold
  for any power law. With b = 2 the back-transform gap at σ = 0.24 is 12.2% (vs 16.8% at b = 2.32).
- Bib fixes: PyMC title comma; VMD0001 v1.2 is 2023 (approved 27 Nov 2023), not 2022.

### 9.1 Re-run with Brown Eq. 3.2.2 (b = 2) — `pseudo_plot_eval.py brown_eq322` → `out_plots_eq322/`

Eq. 3.2.2: AGB = 10^−0.535 · BA, BA = πD²/4 in cm² (191 trees, DBH 3–30 cm). Default run
(`brown_eq321`) re-checked byte-identical to the earlier `out_plots/`. Block-year plots:

| Model | Gap b=2.32 | Gap b=2 | p25 over-claim b=2 | p10 b=2 | 80% rule b=2 | 80% cov. 2.32 / 2 | 80% rule left on table b=2 |
|---|---|---|---|---|---|---|---|
| GPR | 19.3% | 13.9% | 46% | 31% | 19% | 54 / 58% | 17.6% |
| A0 | 19.8% | 14.3% | 54% | 42% | 27% | 46 / 46% | 17.0% |
| A1 | 18.6% | 13.3% | 46% | 27% | 15% | 62 / 65% | 15.4% |
| A2 | 18.5% | 13.2% | 54% | 38% | 27% | 50 / 54% | 15.5% |
| v3 | 16.0% | 11.5% | 46% | 23% | 15% | 58 / 65% | 15.8% |

- Gap shrinks as exp(2σ²) predicts (v3: 10.9% predicted vs 11.5% observed).
- Plot calibration failure unchanged: p25 over-claims 46–54% (block-year), 36–51% (geo-cell);
  80% intervals cover 46–65% / 41–72%. v3 plot errors still −26% to +44%, uncorrelated with
  plot size (Spearman −0.11).
- Plot MAE drops ~3 pp for every model (v3 14.7 → 12.3%), and the 80% rule over-claims a bit
  less (15–27% vs 19–35%) — less curvature, smaller tails. Ranking of models unchanged.
- Paper: Limitations rewritten with these numbers; new Appendix C (Table `tab:allometry`).
