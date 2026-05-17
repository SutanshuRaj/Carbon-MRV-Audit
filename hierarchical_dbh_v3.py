"""Hierarchical Bayesian DBH model with species partial pooling, district random
effects, and quadratic crown + log-age predictors. Fits via NUTS, ~5-8 minutes."""

import warnings
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')

CSV_PATH      = 'f4f_ground_data_13Oct25.csv'
OUT_DIR       = Path('out_v3')
DRAWS         = 1000
TUNE          = 1000
CHAINS        = 4
TARGET_ACCEPT = 0.95
SEED          = 42

MIN_SP_COUNT  = 30
EXCLUDE_SP    = ['Bamboo']   # monocot grass — Verra hardwood allometry doesn't apply
AGE_CV_FLOOR  = 0.10         # below this, age has no signal to learn from
SAPLING_DBH   = 5.0          # tape-measurement noise floor; flag in output only

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
    df = df[~df['Tree species'].isin(EXCLUDE_SP)]

    counts = df['Tree species'].value_counts()
    df = df[df['Tree species'].isin(counts[counts >= MIN_SP_COUNT].index)].copy().reset_index(drop=True)

    cv = df.groupby('Tree species')['tree_age_years'].agg(lambda s: s.std() / s.mean())
    age_masked = cv[cv < AGE_CV_FLOOR].index.tolist()
    df['_age_informative'] = (~df['Tree species'].isin(age_masked)).astype(float)

    print(f"Cleaned dataset: {len(df):,} trees, {df['Tree species'].nunique()} species, "
          f"{df['District'].nunique()} districts")
    print(f"Excluded species: {EXCLUDE_SP}")
    print(f"Age-masked species (CV < {AGE_CV_FLOOR}): {age_masked or '(none)'}")
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
    n_sp, n_di = len(species_levels), len(district_levels)

    log_crown  = np.log(train['crown_m'].values)
    log_height = np.log(train['height_m'].values)
    log_age    = np.log(train['tree_age_years'].values)
    log_dbh    = np.log(train['DBH_cm'].values)

    m_c, m_h, m_a = log_crown.mean(), log_height.mean(), log_age.mean()
    xc = log_crown  - m_c
    xh = log_height - m_h
    xa = (log_age   - m_a) * train['_age_informative'].values

    s_idx = species_cat.codes
    d_idx = district_cat.codes

    with pm.Model():
        alpha_g = pm.Normal('alpha_g', mu=2.5, sigma=0.5)
        beta_g  = pm.Normal('beta_g',  mu=0.5, sigma=0.3)
        zeta_g  = pm.Normal('zeta_g',  mu=0.0, sigma=0.2)
        gamma_g = pm.Normal('gamma_g', mu=0.3, sigma=0.3)
        eta_g   = pm.TruncatedNormal('eta_g', mu=0.2, sigma=0.3, lower=0.0)

        tau_a = pm.HalfNormal('tau_a', sigma=0.5)
        tau_b = pm.HalfNormal('tau_b', sigma=0.3)
        tau_z = pm.HalfNormal('tau_z', sigma=0.15)
        tau_c = pm.HalfNormal('tau_c', sigma=0.3)
        tau_e = pm.HalfNormal('tau_e', sigma=0.2)

        z_alpha = pm.Normal('z_alpha', 0, 1, shape=n_sp)
        z_beta  = pm.Normal('z_beta',  0, 1, shape=n_sp)
        z_zeta  = pm.Normal('z_zeta',  0, 1, shape=n_sp)
        z_gamma = pm.Normal('z_gamma', 0, 1, shape=n_sp)

        alpha = pm.Deterministic('alpha', alpha_g + tau_a * z_alpha)
        beta  = pm.Deterministic('beta',  beta_g  + tau_b * z_beta)
        zeta  = pm.Deterministic('zeta',  zeta_g  + tau_z * z_zeta)
        gamma = pm.Deterministic('gamma', gamma_g + tau_c * z_gamma)

        # eta_sp = eta_g + |offset| guarantees age slope >= eta_g >= 0
        # (older trees, all else equal, are at least as fat)
        eta_offset = pm.HalfNormal('eta_offset', sigma=tau_e, shape=n_sp)
        eta = pm.Deterministic('eta', eta_g + eta_offset)

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
        raise ValueError("New data has species or districts not in training.")

    xc = np.log(df_new['crown_m'].values)        - encoders['crown_mean']
    xh = np.log(df_new['height_m'].values)       - encoders['height_mean']
    xa = (np.log(df_new['tree_age_years'].values) - encoders['age_mean']) * df_new['_age_informative'].values

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
    df['true_dbh'] = np.exp(y_te)
    df['pred_dbh'] = np.exp(med_te)
    df['rel_err_pct'] = 100 * (df['pred_dbh'] - df['true_dbh']) / df['true_dbh']
    lo90 = np.quantile(log_pred_te, 0.05, axis=0)
    hi90 = np.quantile(log_pred_te, 0.95, axis=0)
    df['in_90'] = (y_te >= lo90) & (y_te <= hi90)
    df['below_sapling_floor'] = df['true_dbh'] < SAPLING_DBH

    print(f"\n{'RESIDUAL DIAGNOSTICS':^66}")

    bins = pd.cut(df['true_dbh'], [0, SAPLING_DBH, 10, 15, 20, 30, 50, 200],
                  labels=[f'<{SAPLING_DBH:g}','5-10','10-15','15-20','20-30','30-50','>50'])
    by_bin = df.groupby(bins, observed=True).agg(
        n=('true_dbh','count'),
        mean_true=('true_dbh','mean'),
        bias_pct=('rel_err_pct','mean'),
        coverage_90=('in_90', lambda s: round(100*s.mean(), 1)),
    ).round(2)
    print("\nBias by DBH size bin:")
    print(by_bin.to_string())

    print("\nBias by species (top 12 by sample size):")
    by_sp = df.groupby('Tree species').agg(
        n=('true_dbh','count'),
        mean_true=('true_dbh','mean'),
        bias_pct=('rel_err_pct','mean'),
        coverage_90=('in_90', lambda s: round(100*s.mean(), 1)),
    ).sort_values('n', ascending=False).head(12).round(2)
    print(by_sp.to_string())

    print("\nBias by district + 90% coverage:")
    by_di = df.groupby('District').agg(
        n=('true_dbh','count'),
        mean_true=('true_dbh','mean'),
        bias_pct=('rel_err_pct','mean'),
        coverage_90=('in_90', lambda s: round(100*s.mean(), 1)),
    ).round(2)
    print(by_di.to_string())

    # NOTE: this plot-level CO2 number uses the buggy point-estimate back-transform.
    # See hierarchical_dbh_v31.py for the Jensen-corrected version.
    def co2(dbh_cm):
        biomass = np.exp(-1.996 + 2.32 * np.log(dbh_cm))
        return biomass * 1.27 * 0.5 * (44/12)
    df['co2_true'] = co2(df['true_dbh'])
    df['co2_pred'] = co2(df['pred_dbh'])
    total_true = df['co2_true'].sum()
    total_pred = df['co2_pred'].sum()
    print(f"\nPlot-level carbon bias (uncorrected):  {100*(total_pred-total_true)/total_true:+5.2f}%")
    above = df[~df['below_sapling_floor']]
    print(f"Plot-level carbon bias excluding saplings:  "
          f"{100*(above['co2_pred'].sum()-above['co2_true'].sum())/above['co2_true'].sum():+5.2f}%")
    print(f"(Saplings: {100*df['below_sapling_floor'].mean():.1f}% of trees, "
          f"{100*df.loc[df['below_sapling_floor'],'co2_true'].sum()/total_true:.2f}% of carbon.)")
    return df


def lodo_test(df_full):
    """Leave-one-district-out via OLS proxy. Measures generalization within
    agroforestry across districts — not directly comparable to a hierarchical refit."""
    df = df_full.copy()
    df['log_DBH']      = np.log(df['DBH_cm'])
    df['log_crown']    = np.log(df['crown_m'])
    df['log_height']   = np.log(df['height_m'])
    df['log_age']      = np.log(df['tree_age_years']) * df['_age_informative']
    df['log_crown_sq'] = df['log_crown']**2

    rows = []
    for district in df['District'].unique():
        is_te = df['District'] == district
        if is_te.sum() < 30:
            continue
        train, test = df[~is_te], df[is_te]
        common = set(train['Tree species']) & set(test['Tree species'])
        train = train[train['Tree species'].isin(common)]
        test  = test [test ['Tree species'].isin(common)]
        if len(test) < 20:
            continue

        feats = ['log_crown', 'log_crown_sq', 'log_height', 'log_age']
        oh = pd.get_dummies(pd.concat([train,test])[['Tree species']], drop_first=True)
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
            'baseline_DBH_cm': round(float(np.exp(np.median(a[:,i]))), 2),
            'crown_slope':     round(float(np.median(b[:,i])), 3),
            'crown_curve':     round(float(np.median(z[:,i])), 3),
            'height_slope':    round(float(np.median(g[:,i])), 3),
            'age_slope':       round(float(np.median(e[:,i])), 3),
            'noise_sigma':     round(float(np.median(sg[:,i])), 3),
        })
    df_sp = pd.DataFrame(rows)
    print(f"\n{'LEARNED SPECIES PARAMETERS (posterior medians)':^66}")
    print(df_sp.to_string(index=False))
    if (df_sp['age_slope'] < 0).any():
        print("\nWARNING: negative age slope leaked through constraint.")
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

    trace.to_netcdf(OUT_DIR / 'trace_v3.nc')
    df_resid.to_csv(OUT_DIR / 'residuals_v3.csv', index=False)
    sp_df.to_csv(OUT_DIR / 'species_params_v3.csv', index=False)
    train.to_csv(OUT_DIR / 'train_v3.csv', index=False)
    test.to_csv(OUT_DIR / 'test_v3.csv', index=False)
    print(f"\nSaved artefacts to {OUT_DIR.resolve()}/")


if __name__ == '__main__':
    main()
