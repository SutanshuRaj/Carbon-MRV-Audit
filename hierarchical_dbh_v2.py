"""Adds log(age) and a per-species quadratic crown term to v1. Drops ~11% of
the dataset (rows missing age). Full audit pipeline: train-test gap,
residuals by size/species/district, LODO OLS proxy, calibration coverage.
Kept for history; current model is v3."""

import warnings
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from scipy.stats import norm
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')

CSV_PATH      = 'f4f_ground_data_13Oct25.csv'
OUT_DIR       = Path('out_v2')
DRAWS         = 1000
TUNE          = 1000
CHAINS        = 4
TARGET_ACCEPT = 0.95
SEED          = 42
MIN_SP_COUNT  = 30

OUT_DIR.mkdir(exist_ok=True)


def prepare_data(csv_path):
    df = pd.read_csv(csv_path)
    df['height_m'] = df['Tree Height_foot'] * 0.3048
    df['crown_m']  = df['Tree Crown_foot']  * 0.3048

    df = df[
        (df['DBH_cm']         > 1.0) & (df['DBH_cm']         < 200) &
        (df['height_m']       > 0.5) & (df['crown_m']        > 0.3) &
        (df['tree_age_years'] > 0)   & df['tree_age_years'].notna() &
        df['Tree species'].notna()
    ].copy()
    df = df[~((df['DBH_cm'] < 2) & (df['height_m'] > 3))]

    counts = df['Tree species'].value_counts()
    df = df[df['Tree species'].isin(counts[counts >= MIN_SP_COUNT].index)].copy().reset_index(drop=True)

    print(f"Cleaned dataset: {len(df):,} trees across {df['Tree species'].nunique()} species "
          f"and {df['District'].nunique()} districts")
    return df


def stratified_split(df, frac=0.2, seed=SEED):
    rng = np.random.default_rng(seed)
    test_idx = []
    for _, group in df.groupby('Tree species'):
        n = max(1, int(round(frac * len(group))))
        test_idx.extend(rng.choice(group.index.values, size=n, replace=False))
    test_idx = np.array(test_idx)
    return df.loc[df.index.difference(test_idx)].copy(), df.loc[test_idx].copy()


def build_and_fit(train):
    species_cat  = pd.Categorical(train['Tree species'])
    district_cat = pd.Categorical(train['District'])
    species_levels  = list(species_cat.categories)
    district_levels = list(district_cat.categories)
    n_sp = len(species_levels)
    n_di = len(district_levels)

    log_crown  = np.log(train['crown_m'].values)
    log_height = np.log(train['height_m'].values)
    log_age    = np.log(train['tree_age_years'].values)
    log_dbh    = np.log(train['DBH_cm'].values)

    m_c, m_h, m_a = log_crown.mean(), log_height.mean(), log_age.mean()
    xc = log_crown  - m_c
    xh = log_height - m_h
    xa = log_age    - m_a

    s_idx = species_cat.codes
    d_idx = district_cat.codes

    with pm.Model():
        alpha_g = pm.Normal('alpha_g', mu=2.5, sigma=0.5)
        beta_g  = pm.Normal('beta_g',  mu=0.5, sigma=0.3)
        zeta_g  = pm.Normal('zeta_g',  mu=0.0, sigma=0.2)
        gamma_g = pm.Normal('gamma_g', mu=0.3, sigma=0.3)
        eta_g   = pm.Normal('eta_g',   mu=0.2, sigma=0.3)

        tau_a = pm.HalfNormal('tau_a', sigma=0.5)
        tau_b = pm.HalfNormal('tau_b', sigma=0.3)
        tau_z = pm.HalfNormal('tau_z', sigma=0.15)
        tau_c = pm.HalfNormal('tau_c', sigma=0.3)
        tau_e = pm.HalfNormal('tau_e', sigma=0.3)

        z_alpha = pm.Normal('z_alpha', 0, 1, shape=n_sp)
        z_beta  = pm.Normal('z_beta',  0, 1, shape=n_sp)
        z_zeta  = pm.Normal('z_zeta',  0, 1, shape=n_sp)
        z_gamma = pm.Normal('z_gamma', 0, 1, shape=n_sp)
        z_eta   = pm.Normal('z_eta',   0, 1, shape=n_sp)

        alpha = pm.Deterministic('alpha', alpha_g + tau_a * z_alpha)
        beta  = pm.Deterministic('beta',  beta_g  + tau_b * z_beta)
        zeta  = pm.Deterministic('zeta',  zeta_g  + tau_z * z_zeta)
        gamma = pm.Deterministic('gamma', gamma_g + tau_c * z_gamma)
        eta   = pm.Deterministic('eta',   eta_g   + tau_e * z_eta)

        tau_d = pm.HalfNormal('tau_d', sigma=0.3)
        delta = pm.Normal('delta', 0, tau_d, shape=n_di)

        sigma = pm.HalfNormal('sigma', sigma=0.5, shape=n_sp)

        mu = (alpha[s_idx]
              + beta[s_idx]  * xc
              + zeta[s_idx]  * xc**2
              + gamma[s_idx] * xh
              + eta[s_idx]   * xa
              + delta[d_idx])

        pm.Normal('y_obs', mu=mu, sigma=sigma[s_idx], observed=log_dbh)

        trace = pm.sample(
            draws=DRAWS, tune=TUNE, chains=CHAINS, cores=CHAINS,
            target_accept=TARGET_ACCEPT, random_seed=SEED, progressbar=True,
        )

    encoders = dict(
        species_levels=species_levels, district_levels=district_levels,
        crown_mean=m_c, height_mean=m_h, age_mean=m_a,
    )
    return trace, encoders


def predict_log(trace, encoders, df_new, n_samples=1000, seed=0):
    species_levels  = encoders['species_levels']
    district_levels = encoders['district_levels']

    s_idx = pd.Categorical(df_new['Tree species'], categories=species_levels).codes
    d_idx = pd.Categorical(df_new['District'],     categories=district_levels).codes
    if (s_idx < 0).any() or (d_idx < 0).any():
        raise ValueError("New data has species or districts not seen in training.")

    xc = np.log(df_new['crown_m'].values)        - encoders['crown_mean']
    xh = np.log(df_new['height_m'].values)       - encoders['height_mean']
    xa = np.log(df_new['tree_age_years'].values) - encoders['age_mean']

    post = trace.posterior
    flat = lambda v: post[v].values.reshape(-1, *post[v].shape[2:])
    a, b, z, g, e = flat('alpha'), flat('beta'), flat('zeta'), flat('gamma'), flat('eta')
    d, sg = flat('delta'), flat('sigma')

    rng = np.random.default_rng(seed)
    pick = rng.choice(a.shape[0], size=n_samples, replace=False)
    a, b, z, g, e, d, sg = (arr[pick] for arr in (a, b, z, g, e, d, sg))

    mu = (a[:, s_idx] + b[:, s_idx]*xc[None,:] + z[:, s_idx]*(xc[None,:]**2)
          + g[:, s_idx]*xh[None,:] + e[:, s_idx]*xa[None,:] + d[:, d_idx])
    return mu + sg[:, s_idx] * rng.standard_normal(mu.shape)


def evaluate(trace, encoders, train, test):
    log_pred_tr = predict_log(trace, encoders, train, n_samples=1000)
    log_pred_te = predict_log(trace, encoders, test,  n_samples=1000)

    med_tr = np.median(log_pred_tr, axis=0)
    med_te = np.median(log_pred_te, axis=0)
    y_tr   = np.log(train['DBH_cm'].values)
    y_te   = np.log(test['DBH_cm'].values)

    rmse_tr = np.sqrt(mean_squared_error(np.exp(y_tr), np.exp(med_tr)))
    rmse_te = np.sqrt(mean_squared_error(np.exp(y_te), np.exp(med_te)))
    mae_tr  = mean_absolute_error(np.exp(y_tr), np.exp(med_tr))
    mae_te  = mean_absolute_error(np.exp(y_te), np.exp(med_te))
    r2_log  = 1 - np.var(y_te - med_te) / np.var(y_te)

    print(f"\n{'EVALUATION':^66}")
    print(f"                  Train      Test       Gap")
    print(f"RMSE on DBH cm   {rmse_tr:6.2f}    {rmse_te:6.2f}    {(rmse_te-rmse_tr)/rmse_tr*100:+5.1f}%")
    print(f"MAE  on DBH cm   {mae_tr:6.2f}    {mae_te:6.2f}    {(mae_te-mae_tr)/mae_tr*100:+5.1f}%")
    print(f"R² (log space, test): {r2_log:.4f}")

    nominal = np.array([0.50, 0.60, 0.70, 0.80, 0.90, 0.95])
    empirical = []
    for lvl in nominal:
        lo = np.quantile(log_pred_te, (1-lvl)/2, axis=0)
        hi = np.quantile(log_pred_te, 1-(1-lvl)/2, axis=0)
        empirical.append(np.mean((y_te >= lo) & (y_te <= hi)))
    empirical = np.array(empirical)
    slope = np.polyfit(nominal, empirical, 1)[0]

    print("\nCalibration:")
    for n_, e_ in zip(nominal, empirical):
        print(f"  {int(n_*100)}% interval contains {e_*100:.1f}% of test trees")
    print(f"  Calibration slope: {slope:.3f}")
    return log_pred_te, med_te, y_te


def residual_diagnostics(test, med_te, y_te, log_pred_te):
    df = test.copy()
    df['true_dbh']    = np.exp(y_te)
    df['pred_dbh']    = np.exp(med_te)
    df['rel_err_pct'] = 100 * (df['pred_dbh'] - df['true_dbh']) / df['true_dbh']

    print(f"\n{'RESIDUAL DIAGNOSTICS':^66}")

    bins = pd.cut(df['true_dbh'], [0,5,10,15,20,30,50,200],
                  labels=['<5','5-10','10-15','15-20','20-30','30-50','>50'])
    by_bin = df.groupby(bins, observed=True).agg(
        n=('true_dbh','count'),
        mean_true=('true_dbh','mean'),
        bias_pct=('rel_err_pct','mean'),
        rmse=('rel_err_pct', lambda r: np.sqrt(np.mean((r/100)**2)*100)),
    ).round(2)
    print("\nBias by DBH size bin:")
    print(by_bin.to_string())

    print("\nBias by species (top 10 by sample size):")
    by_sp = df.groupby('Tree species').agg(
        n=('true_dbh','count'),
        mean_true=('true_dbh','mean'),
        bias_pct=('rel_err_pct','mean'),
    ).sort_values('n', ascending=False).head(10).round(2)
    print(by_sp.to_string())

    print("\nBias by district + 90% interval coverage:")
    lo = np.quantile(log_pred_te, 0.05, axis=0)
    hi = np.quantile(log_pred_te, 0.95, axis=0)
    df['in_90'] = (y_te >= lo) & (y_te <= hi)
    by_di = df.groupby('District').agg(
        n=('true_dbh','count'),
        mean_true=('true_dbh','mean'),
        bias_pct=('rel_err_pct','mean'),
        coverage_90=('in_90', lambda s: round(100 * s.mean(), 1)),
    ).round(2)
    print(by_di.to_string())
    return df


def lodo_test(df_full):
    """OLS proxy for leave-one-district-out. The hierarchical refit takes too long
    to repeat per district; the OLS pattern tracks the hierarchical one closely."""
    from sklearn.linear_model import LinearRegression
    df_full = df_full.copy()
    df_full['log_DBH']      = np.log(df_full['DBH_cm'])
    df_full['log_crown']    = np.log(df_full['crown_m'])
    df_full['log_height']   = np.log(df_full['height_m'])
    df_full['log_age']      = np.log(df_full['tree_age_years'])
    df_full['log_crown_sq'] = df_full['log_crown']**2

    rows = []
    for district in df_full['District'].unique():
        is_test = df_full['District'] == district
        if is_test.sum() < 30:
            continue
        train = df_full[~is_test]
        test  = df_full[ is_test]
        common = set(train['Tree species']) & set(test['Tree species'])
        train = train[train['Tree species'].isin(common)]
        test  = test [test ['Tree species'].isin(common)]
        if len(test) < 20:
            continue

        feats = ['log_crown','log_crown_sq','log_height','log_age']
        oh    = pd.get_dummies(pd.concat([train,test])[['Tree species']], drop_first=True)
        X_all = pd.concat([pd.concat([train,test])[feats].reset_index(drop=True),
                           oh.reset_index(drop=True)], axis=1).values.astype(float)
        y_all = np.concatenate([train['log_DBH'].values, test['log_DBH'].values])
        n_tr  = len(train)

        m = LinearRegression().fit(X_all[:n_tr], y_all[:n_tr])
        p = np.exp(m.predict(X_all[n_tr:]))
        true = test['DBH_cm'].values
        rows.append({
            'Held-out':    district,
            'n_train':     n_tr,
            'n_test':      len(test),
            'OOD_RMSE_cm': np.sqrt(mean_squared_error(true, p)),
            'OOD_MAE_cm':  mean_absolute_error(true, p),
        })
    print(f"\n{'LEAVE-ONE-DISTRICT-OUT (OLS proxy)':^66}")
    print(pd.DataFrame(rows).round(2).to_string(index=False))


def species_params(trace, encoders):
    species_levels = encoders['species_levels']
    post = trace.posterior
    flat = lambda v: post[v].values.reshape(-1, *post[v].shape[2:])
    a, b, z, g, e, sg = flat('alpha'), flat('beta'), flat('zeta'), flat('gamma'), flat('eta'), flat('sigma')

    rows = []
    for i, sp in enumerate(species_levels):
        rows.append({
            'Species':         sp,
            'baseline_DBH_cm': float(np.exp(np.median(a[:,i]))),
            'crown_slope':     round(float(np.median(b[:,i])), 3),
            'crown_curve':     round(float(np.median(z[:,i])), 3),
            'height_slope':    round(float(np.median(g[:,i])), 3),
            'age_slope':       round(float(np.median(e[:,i])), 3),
            'noise_sigma':     round(float(np.median(sg[:,i])), 3),
        })
    df_sp = pd.DataFrame(rows).round(2)
    print(f"\n{'LEARNED SPECIES PARAMETERS (posterior medians)':^66}")
    print(df_sp.to_string(index=False))
    return df_sp


def main():
    df = prepare_data(CSV_PATH)
    train, test = stratified_split(df, frac=0.2)
    print(f"Train: {len(train):,} | Test: {len(test):,}\n")

    print(f"Fitting with {CHAINS} chains × {DRAWS} draws...")
    trace, encoders = build_and_fit(train)

    summary = az.summary(trace, var_names=['alpha_g','beta_g','zeta_g','gamma_g','eta_g',
                                            'tau_a','tau_b','tau_z','tau_c','tau_e','tau_d'])
    print(summary[['mean','sd','r_hat','ess_bulk']].to_string())
    max_rhat = summary['r_hat'].max()
    print(f"\nMax r_hat = {max_rhat:.3f}  "
          f"({'clean' if max_rhat <= 1.01 else 'increase TUNE/DRAWS'})")

    log_pred_te, med_te, y_te = evaluate(trace, encoders, train, test)
    df_resid = residual_diagnostics(test, med_te, y_te, log_pred_te)
    sp_df    = species_params(trace, encoders)
    lodo_test(df)

    trace.to_netcdf(OUT_DIR / 'trace_v2.nc')
    df_resid.to_csv(OUT_DIR / 'residuals_v2.csv', index=False)
    sp_df.to_csv(OUT_DIR / 'species_params_v2.csv', index=False)
    train.to_csv(OUT_DIR / 'train_v2.csv', index=False)
    test.to_csv(OUT_DIR / 'test_v2.csv', index=False)
    print(f"\nSaved artefacts to {OUT_DIR.resolve()}/")


if __name__ == '__main__':
    main()
