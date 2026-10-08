"""Plot-level evaluation of credit-issuance policies. The public dataset has no plot
ID, so test trees are grouped into pseudo-plots and each policy is scored per plot:
how often does it over-claim, and how much does it leave on the table?

  block_year  District x Block x Plantation Year   (primary)
  latlon3     lat/lon rounded to 3 dp (~110 m cell) (robustness)

Groups with fewer than MIN_TREES test trees are dropped. Posterior samples are joint
across trees within a draw (shared parameters), so plot totals carry correlated error.
Reuses traces from out_v3/ and out_ablation/; refits only the GPR. ~3 minutes.

Usage: python pseudo_plot_eval.py [brown_eq321 | brown_eq322]
  brown_eq321  AGB = exp(-1.996 + 2.32 ln D)      Brown (1997) Eq. 3.2.1, dry >900 mm  (default)
  brown_eq322  AGB = 10^-0.535 * BA, BA = pi D^2/4  Brown (1997) Eq. 3.2.2, dry <900 mm"""

import sys
import warnings
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from ablation_hier import ARMS, predict as predict_ablation
from hierarchical_dbh_v31 import co2_kg as co2_kg_eq321, predict_log, rebuild_encoders

warnings.filterwarnings('ignore')

V3_DIR    = Path('out_v3')
ABL_DIR   = Path('out_ablation')
N_SAMPLES = 1000
MIN_TREES = 10
SEED      = 0

POLICIES = ['point', 'mean', 'p25', 'p10', 'rule80']


def co2_kg_eq322(dbh_cm):
    biomass = 10 ** -0.535 * np.pi * dbh_cm**2 / 4      # Brown (1997) Eq. 3.2.2, kg
    return biomass * 1.27 * 0.5 * (44 / 12)


ALLOMETRIES = {
    'brown_eq321': (co2_kg_eq321, Path('out_plots')),
    'brown_eq322': (co2_kg_eq322, Path('out_plots_eq322')),
}
ALLOMETRY = sys.argv[1] if len(sys.argv) > 1 else 'brown_eq321'
co2_kg, OUT_DIR = ALLOMETRIES[ALLOMETRY]
OUT_DIR.mkdir(exist_ok=True)


def gpr_joint_samples(train, test):
    """TreeLens-spec GPR; samples from the full predictive covariance (incl. noise)."""
    cols = ['crown_m', 'height_m']
    m, s = train[cols].values.mean(0), train[cols].values.std(0)
    X_tr, X_te = (train[cols].values - m) / s, (test[cols].values - m) / s
    gpr = GaussianProcessRegressor(
        kernel=ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(0.05),
        normalize_y=True, n_restarts_optimizer=2, random_state=SEED,
    ).fit(X_tr, np.log(train['DBH_cm'].values))
    mu, cov = gpr.predict(X_te, return_cov=True)
    L = np.linalg.cholesky(cov + 1e-8 * np.eye(len(mu)))
    rng = np.random.default_rng(SEED)
    return mu[None, :] + rng.standard_normal((N_SAMPLES, len(mu))) @ L.T


def plot_keys(test):
    return {
        'block_year': test['District'] + '|' + test['Block'] + '|' + test['Plantation Year'].astype(str),
        'latlon3':    test['Latitude'].round(3).astype(str) + '_' + test['Longitude'].round(3).astype(str),
    }


def per_plot(model, log_pred, test, keys):
    co2_true = co2_kg(test['DBH_cm'].values)
    co2_samp = co2_kg(np.exp(log_pred))                       # (samples, trees)
    co2_pt   = co2_kg(np.exp(np.median(log_pred, axis=0)))
    rows = []
    for grouping, key in keys.items():
        for plot, idx in key.groupby(key).groups.items():
            pos = test.index.get_indexer(idx)
            if len(pos) < MIN_TREES:
                continue
            totals = co2_samp[:, pos].sum(axis=1)
            true = co2_true[pos].sum()
            claims = {
                'point':  co2_pt[pos].sum(),
                'mean':   totals.mean(),
                'p25':    np.quantile(totals, 0.25),
                'p10':    np.quantile(totals, 0.10),
                'rule80': 0.8 * totals.mean(),
            }
            row = dict(model=model, grouping=grouping, plot=plot, n_trees=len(pos),
                       n_species=test.iloc[pos]['Tree species'].nunique(),
                       true_kg=true, pit=float(np.mean(totals < true)))
            row.update({f'{p}_kg': v for p, v in claims.items()})
            rows.append(row)
    return rows


def summarise(plots):
    out = []
    for (model, grouping), g in plots.groupby(['model', 'grouping'], sort=False):
        true = g['true_kg'].values
        for p in POLICIES:
            claim = g[f'{p}_kg'].values
            rel = 100 * (claim - true) / true
            k, n = int((claim > true).sum()), len(true)
            ci = binomtest(k, n).proportion_ci(method='wilson')
            out.append({
                'model': model, 'grouping': grouping, 'policy': p, 'n_plots': n,
                'overclaim_rate_pct':  100 * k / n,
                'overclaim_ci95':      f'{100*ci.low:.0f}-{100*ci.high:.0f}',
                'median_rel_err_pct':  np.median(rel),
                'mean_abs_rel_err_pct': np.mean(np.abs(rel)),
                'aggregate_err_pct':   100 * (claim.sum() - true.sum()) / true.sum(),
                'overclaimed_pct':     100 * np.clip(claim - true, 0, None).sum() / true.sum(),
                'left_on_table_pct':   100 * np.clip(true - claim, 0, None).sum() / true.sum(),
            })
        out.append({'model': model, 'grouping': grouping, 'policy': 'PIT',
                    'n_plots': len(g),
                    'pit_in_central50_pct': 100 * g['pit'].between(0.25, 0.75).mean(),
                    'pit_in_central80_pct': 100 * g['pit'].between(0.10, 0.90).mean()})
    return pd.DataFrame(out)


def main():
    train, encoders = rebuild_encoders(V3_DIR / 'train_v3.csv')
    test = pd.read_csv(V3_DIR / 'test_v3.csv')
    keys = plot_keys(test)
    for name, key in keys.items():
        sizes = key.value_counts()
        kept = sizes[sizes >= MIN_TREES]
        print(f"{name}: {len(kept)} plots >= {MIN_TREES} trees, {kept.sum():,} trees, "
              f"median size {int(kept.median())}, range {kept.min()}-{kept.max()}")

    preds = {}
    print("\nFitting GPR (TreeLens spec, joint covariance)...", flush=True)
    preds['GPR (TC, TH)'] = gpr_joint_samples(train, test)
    for name, arm in ARMS.items():
        preds[arm['label']] = predict_ablation(az.from_netcdf(ABL_DIR / f'trace_{name}.nc'),
                                               encoders, test, arm)
    preds['v3: + field age (full model)'] = predict_log(
        az.from_netcdf(V3_DIR / 'trace_v3.nc'), encoders, test, n_samples=N_SAMPLES, seed=SEED)

    plots = pd.DataFrame([r for m, lp in preds.items() for r in per_plot(m, lp, test, keys)])
    summary = summarise(plots)
    plots.to_csv(OUT_DIR / 'per_plot.csv', index=False)
    summary.to_csv(OUT_DIR / 'policy_summary.csv', index=False)

    pd.set_option('display.width', 220)
    pd.set_option('display.max_columns', 20)
    for grouping in keys:
        s = summary[(summary['grouping'] == grouping) & (summary['policy'] != 'PIT')]
        print(f"\n=== {grouping}: policy outcomes across plots ===")
        print(s.drop(columns=['grouping', 'pit_in_central50_pct', 'pit_in_central80_pct'],
                     errors='ignore').round(1).to_string(index=False))
        pit = summary[(summary['grouping'] == grouping) & (summary['policy'] == 'PIT')]
        print(f"\n{grouping}: plot-level interval coverage (nominal 50% / 80%)")
        print(pit[['model', 'n_plots', 'pit_in_central50_pct', 'pit_in_central80_pct']]
              .round(1).to_string(index=False))
    print(f"\nSaved {OUT_DIR}/per_plot.csv and {OUT_DIR}/policy_summary.csv")


if __name__ == '__main__':
    main()
