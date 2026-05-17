"""Reuses the v3 posterior to compute plot-level CO2 with the Jensen-correct
back-transform (per-sample biomass, then average) and compare it to the buggy
point-estimate version. No refitting; runs in ~10-30 seconds."""

import warnings
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

OUT_DIR        = Path('out_v3')
SAPLING_DBH    = 5.0
N_PRED_SAMPLES = 1000
SEED           = 0


def rebuild_encoders(train_path):
    train = pd.read_csv(train_path)
    species_cat  = pd.Categorical(train['Tree species'])
    district_cat = pd.Categorical(train['District'])
    log_crown  = np.log(train['crown_m'].values)
    log_height = np.log(train['height_m'].values)
    log_age    = np.log(train['tree_age_years'].values)
    return train, dict(
        species_levels  = list(species_cat.categories),
        district_levels = list(district_cat.categories),
        crown_mean      = log_crown.mean(),
        height_mean     = log_height.mean(),
        age_mean        = log_age.mean(),
    )


def predict_log(trace, encoders, df_new, n_samples=N_PRED_SAMPLES, seed=SEED):
    species_levels  = encoders['species_levels']
    district_levels = encoders['district_levels']

    s_idx = pd.Categorical(df_new['Tree species'], categories=species_levels).codes
    d_idx = pd.Categorical(df_new['District'],     categories=district_levels).codes
    if (s_idx < 0).any() or (d_idx < 0).any():
        raise ValueError("New data has species/districts unseen in training.")

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


def co2_kg(dbh_cm):
    biomass = np.exp(-1.996 + 2.32 * np.log(dbh_cm))   # Verra VMD0001 AGB, kg
    return biomass * 1.27 * 0.5 * (44 / 12)


def main():
    print("v3.1 — Jensen-corrected plot-level carbon diagnostic")
    print(f"Reusing posterior from {OUT_DIR}/trace_v3.nc — no MCMC.\n")

    trace = az.from_netcdf(OUT_DIR / 'trace_v3.nc')
    train, encoders = rebuild_encoders(OUT_DIR / 'train_v3.csv')
    test = pd.read_csv(OUT_DIR / 'test_v3.csv')

    if '_age_informative' not in test.columns:
        cv = train.groupby('Tree species')['tree_age_years'].agg(lambda s: s.std() / s.mean())
        age_masked = cv[cv < 0.10].index.tolist()
        test['_age_informative'] = (~test['Tree species'].isin(age_masked)).astype(float)
        if age_masked:
            print(f"Reapplied age mask for: {age_masked}")

    log_pred = predict_log(trace, encoders, test, n_samples=N_PRED_SAMPLES)

    # Point-estimate (buggy): exp(median(log_pred)) -> geometric mean of lognormal
    dbh_wrong   = np.exp(np.median(log_pred, axis=0))
    # Correct: posterior arithmetic mean of DBH
    dbh_samples = np.exp(log_pred)
    dbh_correct = np.mean(dbh_samples, axis=0)

    co2_wrong   = co2_kg(dbh_wrong)
    co2_correct = np.mean(co2_kg(dbh_samples), axis=0)

    true_dbh = test['DBH_cm'].values
    co2_true = co2_kg(true_dbh)

    print(f"Test set: {len(test):,} trees\n")

    print("Plot-level carbon bias — buggy vs corrected back-transform")
    sum_true, sum_wrong, sum_corr = co2_true.sum(), co2_wrong.sum(), co2_correct.sum()
    print(f"  Total true CO2:             {sum_true:11.1f} kg")
    print(f"  Point-estimate predicted:   {sum_wrong:11.1f} kg   bias {100*(sum_wrong-sum_true)/sum_true:+6.2f}%")
    print(f"  Jensen-corrected:           {sum_corr:11.1f} kg   bias {100*(sum_corr-sum_true)/sum_true:+6.2f}%")
    delta = 100 * (sum_corr - sum_wrong) / sum_true
    print(f"  Jensen correction lifted plot-level estimate by {delta:+.2f} pp.")

    mask = true_dbh >= SAPLING_DBH
    print(f"\nExcluding saplings (DBH < {SAPLING_DBH:g} cm; "
          f"{(~mask).sum()} trees, {100*(~mask).mean():.1f}% of test):")
    st, sw, sc = co2_true[mask].sum(), co2_wrong[mask].sum(), co2_correct[mask].sum()
    print(f"  True:            {st:11.1f} kg")
    print(f"  Point-estimate:  {sw:11.1f} kg   bias {100*(sw-st)/st:+6.2f}%")
    print(f"  Jensen-corrected:{sc:11.1f} kg   bias {100*(sc-st)/st:+6.2f}%")

    print("\nCO2 bias by DBH size bin")
    bins = pd.cut(true_dbh, [0, SAPLING_DBH, 10, 15, 20, 30, 50, 200],
                  labels=[f'<{SAPLING_DBH:g}','5-10','10-15','15-20','20-30','30-50','>50'])
    df = pd.DataFrame({
        'bin':         bins,
        'co2_true':    co2_true,
        'co2_wrong':   co2_wrong,
        'co2_correct': co2_correct,
        'dbh_true':    true_dbh,
        'dbh_wrong':   dbh_wrong,
        'dbh_correct': dbh_correct,
        'species':     test['Tree species'].values,
    })
    by_bin = df.groupby('bin', observed=True).apply(
        lambda d: pd.Series({
            'n':                len(d),
            'co2_share_pct':    100*d['co2_true'].sum()/co2_true.sum(),
            'bias_wrong_pct':   100*(d['co2_wrong'].sum()-d['co2_true'].sum())/d['co2_true'].sum(),
            'bias_correct_pct': 100*(d['co2_correct'].sum()-d['co2_true'].sum())/d['co2_true'].sum(),
        })
    ).round(2)
    print(by_bin.to_string())

    print("\nCO2 bias by species (top 12 by sample size)")
    by_sp = df.groupby('species').apply(
        lambda d: pd.Series({
            'n':                len(d),
            'co2_share_pct':    100*d['co2_true'].sum()/co2_true.sum(),
            'bias_wrong_pct':   100*(d['co2_wrong'].sum()-d['co2_true'].sum())/d['co2_true'].sum(),
            'bias_correct_pct': 100*(d['co2_correct'].sum()-d['co2_true'].sum())/d['co2_true'].sum(),
        })
    ).round(2).sort_values('n', ascending=False).head(12)
    print(by_sp.to_string())

    print("\nDBH-level bias (model-level, separate from biomass back-transform)")
    by_dbh = df.groupby('bin', observed=True).apply(
        lambda d: pd.Series({
            'n':                len(d),
            'mean_dbh_true':    d['dbh_true'].mean(),
            'dbh_bias_wrong':   100*(d['dbh_wrong'].mean()  - d['dbh_true'].mean()) / d['dbh_true'].mean(),
            'dbh_bias_correct': 100*(d['dbh_correct'].mean() - d['dbh_true'].mean()) / d['dbh_true'].mean(),
        })
    ).round(2)
    print(by_dbh.to_string())

    out = OUT_DIR / 'v3_1_corrected_carbon.csv'
    pd.DataFrame({
        'species':     test['Tree species'].values,
        'district':    test['District'].values,
        'dbh_true':    true_dbh,
        'dbh_wrong':   dbh_wrong,
        'dbh_correct': dbh_correct,
        'co2_true':    co2_true,
        'co2_wrong':   co2_wrong,
        'co2_correct': co2_correct,
    }).to_csv(out, index=False)
    print(f"\nSaved per-tree corrected predictions to {out.resolve()}")

    bias_corr = 100*(sum_corr-sum_true)/sum_true
    print(f"\nPlot-level CO2 bias after Jensen fix: {bias_corr:+.2f}%")


if __name__ == '__main__':
    main()
