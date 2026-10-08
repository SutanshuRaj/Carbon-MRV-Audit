"""Covariate ablation of the v3 hierarchical model on the exact v3 split. Separates
what is deployable from drone imagery alone from what needs field data.

  A0  drone + location:  crown, crown², height, district         (species pooled)
  A1  + species ID:      species-level crown, crown², height      (no district)
  A2  + both:            species-level + district                 (no age)
  v3  + field age:       saved posterior from out_v3/, not refit

Same priors and structure as hierarchical_dbh_v3.py with terms switched off.
Usage: python ablation_hier.py [A0 A1 A2]   (~5-8 min per arm)"""

import sys
import warnings
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from baseline_gpr import score
from hierarchical_dbh_v31 import predict_log, rebuild_encoders

warnings.filterwarnings('ignore')

V3_DIR        = Path('out_v3')
OUT_DIR       = Path('out_ablation')
DRAWS         = 1000
TUNE          = 1000
CHAINS        = 4
TARGET_ACCEPT = 0.95
SEED          = 42
N_SAMPLES     = 1000

ARMS = {
    'A0': dict(species=False, district=True,  label='A0: drone + district (species pooled)'),
    'A1': dict(species=True,  district=False, label='A1: + species ID, no district'),
    'A2': dict(species=True,  district=True,  label='A2: species + district, no age'),
}

OUT_DIR.mkdir(exist_ok=True)


def design(df, encoders, use_species):
    xc = np.log(df['crown_m'].values)  - encoders['crown_mean']
    xh = np.log(df['height_m'].values) - encoders['height_mean']
    s_idx = (pd.Categorical(df['Tree species'], categories=encoders['species_levels']).codes
             if use_species else np.zeros(len(df), dtype=int))
    d_idx = pd.Categorical(df['District'], categories=encoders['district_levels']).codes
    if (s_idx < 0).any() or (d_idx < 0).any():
        raise ValueError("Species or district unseen in training.")
    return xc, xh, s_idx, d_idx


def fit(train, encoders, arm):
    xc, xh, s_idx, d_idx = design(train, encoders, arm['species'])
    n_sp = len(encoders['species_levels']) if arm['species'] else 1
    n_di = len(encoders['district_levels'])

    with pm.Model():
        alpha_g = pm.Normal('alpha_g', mu=2.5, sigma=0.5)
        beta_g  = pm.Normal('beta_g',  mu=0.5, sigma=0.3)
        zeta_g  = pm.Normal('zeta_g',  mu=0.0, sigma=0.2)
        gamma_g = pm.Normal('gamma_g', mu=0.3, sigma=0.3)

        if arm['species']:
            tau_a = pm.HalfNormal('tau_a', sigma=0.5)
            tau_b = pm.HalfNormal('tau_b', sigma=0.3)
            tau_z = pm.HalfNormal('tau_z', sigma=0.15)
            tau_c = pm.HalfNormal('tau_c', sigma=0.3)
            alpha = pm.Deterministic('alpha', alpha_g + tau_a * pm.Normal('z_alpha', 0, 1, shape=n_sp))
            beta  = pm.Deterministic('beta',  beta_g  + tau_b * pm.Normal('z_beta',  0, 1, shape=n_sp))
            zeta  = pm.Deterministic('zeta',  zeta_g  + tau_z * pm.Normal('z_zeta',  0, 1, shape=n_sp))
            gamma = pm.Deterministic('gamma', gamma_g + tau_c * pm.Normal('z_gamma', 0, 1, shape=n_sp))
        else:
            alpha = pm.Deterministic('alpha', pm.math.stack([alpha_g]))
            beta  = pm.Deterministic('beta',  pm.math.stack([beta_g]))
            zeta  = pm.Deterministic('zeta',  pm.math.stack([zeta_g]))
            gamma = pm.Deterministic('gamma', pm.math.stack([gamma_g]))

        if arm['district']:
            tau_d = pm.HalfNormal('tau_d', sigma=0.3)
            delta = pm.Normal('delta', 0, tau_d, shape=n_di)
        else:
            delta = pm.Deterministic('delta', pm.math.zeros(n_di))

        sigma = pm.HalfNormal('sigma', sigma=0.5, shape=n_sp)

        mu = (alpha[s_idx] + beta[s_idx] * xc + zeta[s_idx] * xc**2
              + gamma[s_idx] * xh + delta[d_idx])
        pm.Normal('y_obs', mu=mu, sigma=sigma[s_idx], observed=np.log(train['DBH_cm'].values))

        return pm.sample(draws=DRAWS, tune=TUNE, chains=CHAINS, cores=CHAINS,
                         target_accept=TARGET_ACCEPT, random_seed=SEED, progressbar=False)


def predict(trace, encoders, df_new, arm):
    xc, xh, s_idx, d_idx = design(df_new, encoders, arm['species'])
    post = trace.posterior
    flat = lambda v: post[v].values.reshape(-1, *post[v].shape[2:])
    a, b, z, g, d, sg = (flat(v) for v in ('alpha', 'beta', 'zeta', 'gamma', 'delta', 'sigma'))

    rng = np.random.default_rng(0)
    pick = rng.choice(a.shape[0], size=N_SAMPLES, replace=False)
    a, b, z, g, d, sg = (arr[pick] for arr in (a, b, z, g, d, sg))

    mu = (a[:, s_idx] + b[:, s_idx] * xc[None, :] + z[:, s_idx] * xc[None, :]**2
          + g[:, s_idx] * xh[None, :] + d[:, d_idx])
    return mu + sg[:, s_idx] * rng.standard_normal(mu.shape)


def max_rhat(trace):
    hyper = [v for v in trace.posterior.data_vars
             if v.endswith('_g') or v.startswith('tau_') or v == 'sigma']
    return float(az.summary(trace, var_names=hyper)['r_hat'].max())


def main():
    names = sys.argv[1:] or list(ARMS)
    train, encoders = rebuild_encoders(V3_DIR / 'train_v3.csv')
    test = pd.read_csv(V3_DIR / 'test_v3.csv')
    print(f"Split reused from {V3_DIR}/: train {len(train):,} | test {len(test):,}")

    rows = []
    for name in names:
        arm = ARMS[name]
        print(f"\n=== {arm['label']} ===", flush=True)
        trace = fit(train, encoders, arm)
        trace.to_netcdf(OUT_DIR / f'trace_{name}.nc')
        row = score(arm['label'], predict(trace, encoders, test, arm), test)
        row['max_rhat'] = max_rhat(trace)
        row['divergences'] = int(trace.sample_stats['diverging'].sum())
        print(pd.Series(row).to_string(), flush=True)
        rows.append(row)
        pd.DataFrame([row]).to_csv(OUT_DIR / f'metrics_{name}.csv', index=False)

    v3 = score('v3: + field age (full model)',
               predict_log(az.from_netcdf(V3_DIR / 'trace_v3.nc'), encoders, test,
                           n_samples=N_SAMPLES, seed=0), test)
    rows.append(v3)
    table = pd.DataFrame(rows).set_index('model')
    pd.set_option('display.width', 220)
    print("\nAblation on the same held-out test set (CO2 columns = % vs ground truth):")
    print(table.round(3).T.to_string())
    table.to_csv(OUT_DIR / 'ablation_comparison.csv')


if __name__ == '__main__':
    main()
