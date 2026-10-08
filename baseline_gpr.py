"""Reproduces the TreeLens pooled GPR baseline on the exact v3 train/test split, so
Table 1 compares like with like. Also scores a pooled OLS reference and re-scores
the saved v3 posterior with the same metric code. No MCMC; ~1-3 minutes."""

import warnings
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from hierarchical_dbh_v31 import co2_kg, predict_log, rebuild_encoders

warnings.filterwarnings('ignore')

V3_DIR    = Path('out_v3')
OUT_DIR   = Path('out_baseline')
N_SAMPLES = 1000
SEED      = 0

OUT_DIR.mkdir(exist_ok=True)


def fit_gpr(train, test, log_inputs):
    """TreeLens spec: RBF + white noise on (TC, TH), target log(DBH)."""
    cols = ['crown_m', 'height_m']
    X_tr, X_te = train[cols].values, test[cols].values
    if log_inputs:
        X_tr, X_te = np.log(X_tr), np.log(X_te)
    m, s = X_tr.mean(0), X_tr.std(0)
    X_tr, X_te = (X_tr - m) / s, (X_te - m) / s
    y_tr = np.log(train['DBH_cm'].values)

    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.05)
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                   n_restarts_optimizer=2, random_state=SEED)
    gpr.fit(X_tr, y_tr)
    print(f"  kernel: {gpr.kernel_}")
    mu, sd = gpr.predict(X_te, return_std=True)   # sd includes the white-noise term
    rng = np.random.default_rng(SEED)
    return mu[None, :] + sd[None, :] * rng.standard_normal((N_SAMPLES, len(mu)))


def fit_pooled_ols(train, test):
    """log-log OLS on (TC, TH) with Gaussian residuals — the simplest pooled reference."""
    feats = lambda d: np.column_stack([np.log(d['crown_m']), np.log(d['height_m'])])
    y_tr = np.log(train['DBH_cm'].values)
    ols = LinearRegression().fit(feats(train), y_tr)
    sd = np.std(y_tr - ols.predict(feats(train)), ddof=3)
    mu = ols.predict(feats(test))
    rng = np.random.default_rng(SEED)
    return mu[None, :] + sd * rng.standard_normal((N_SAMPLES, len(mu)))


def score(name, log_pred, test):
    y = np.log(test['DBH_cm'].values)
    med = np.median(log_pred, axis=0)
    dbh_true, dbh_pt = np.exp(y), np.exp(med)

    nominal = np.array([0.50, 0.60, 0.70, 0.80, 0.90, 0.95])
    empirical = np.array([
        np.mean((y >= np.quantile(log_pred, (1-l)/2, axis=0)) &
                (y <= np.quantile(log_pred, 1-(1-l)/2, axis=0)))
        for l in nominal])

    co2_true   = co2_kg(dbh_true).sum()
    totals     = co2_kg(np.exp(log_pred)).sum(axis=1)   # per-sample plot totals
    pct = lambda v: 100 * (v - co2_true) / co2_true

    return {
        'model':           name,
        'rmse_cm':         np.sqrt(mean_squared_error(dbh_true, dbh_pt)),
        'mae_cm':          mean_absolute_error(dbh_true, dbh_pt),
        'mape_logdbh_pct': 100 * np.mean(np.abs(med - y) / np.abs(y)),
        'r2_log':          1 - np.var(y - med) / np.var(y),
        'cov90_pct':       100 * empirical[4],
        'calib_slope':     np.polyfit(nominal, empirical, 1)[0],
        'co2_point_pct':   pct(co2_kg(dbh_pt).sum()),
        'co2_mean_pct':    pct(totals.mean()),
        'co2_p25_pct':     pct(np.quantile(totals, 0.25)),
        'co2_p10_pct':     pct(np.quantile(totals, 0.10)),
        'co2_80rule_pct':  pct(0.8 * totals.mean()),
    }


def main():
    train, encoders = rebuild_encoders(V3_DIR / 'train_v3.csv')
    test = pd.read_csv(V3_DIR / 'test_v3.csv')
    print(f"Split reused from {V3_DIR}/: train {len(train):,} | test {len(test):,} | "
          f"{train['Tree species'].nunique()} species\n")

    preds = {}
    print("Fitting GPR (TreeLens spec, raw TC/TH)...")
    preds['GPR (TC, TH)'] = fit_gpr(train, test, log_inputs=False)
    print("Fitting GPR (log TC/TH)...")
    preds['GPR (log TC, log TH)'] = fit_gpr(train, test, log_inputs=True)
    print("Fitting pooled OLS...")
    preds['Pooled OLS (log TC, log TH)'] = fit_pooled_ols(train, test)
    print("Re-scoring saved v3 posterior...")
    preds['Hierarchical v3 (all covariates)'] = predict_log(
        az.from_netcdf(V3_DIR / 'trace_v3.nc'), encoders, test,
        n_samples=N_SAMPLES, seed=SEED)

    results = pd.DataFrame([score(k, v, test) for k, v in preds.items()]).set_index('model')
    pd.set_option('display.width', 200)
    print("\nAll models on the same held-out test set (CO2 columns = % vs ground truth):")
    print(results.round(3).T.to_string())

    results.to_csv(OUT_DIR / 'baseline_comparison.csv')
    print(f"\nSaved {OUT_DIR / 'baseline_comparison.csv'}")


if __name__ == '__main__':
    main()
