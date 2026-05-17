"""First hierarchical model. Species + district partial pooling, no age,
no crown curvature. Includes the Verra biomass / CO2 helper. Kept for history;
current model is v3."""

import numpy as np
import pandas as pd
import pymc as pm


def prepare_data(csv_path, min_species_count=30):
    df = pd.read_csv(csv_path)
    df['height_m'] = df['Tree Height_foot'] * 0.3048
    df['crown_m']  = df['Tree Crown_foot']  * 0.3048

    df = df[
        (df['DBH_cm']   > 1.0)  & (df['DBH_cm']   < 200) &
        (df['height_m'] > 0.5)  & (df['crown_m']  > 0.3) &
        df['Tree species'].notna()
    ].copy()

    # thin-but-tall rows are usually data-entry errors (DBH and height swapped, etc)
    df = df[~((df['DBH_cm'] < 2) & (df['height_m'] > 3))]

    counts = df['Tree species'].value_counts()
    return df[df['Tree species'].isin(counts[counts >= min_species_count].index)].copy()


def stratified_split(df, frac=0.2, seed=42):
    rng = np.random.default_rng(seed)
    test_idx = []
    for _, group in df.groupby('Tree species'):
        n_test = max(1, int(round(frac * len(group))))
        test_idx.extend(rng.choice(group.index.values, size=n_test, replace=False))
    test_idx = np.array(test_idx)
    return df.loc[df.index.difference(test_idx)].copy(), df.loc[test_idx].copy()


def build_and_fit(train, draws=800, tune=800, chains=2, target_accept=0.92, seed=42):
    species_cat  = pd.Categorical(train['Tree species'])
    district_cat = pd.Categorical(train['District'])
    species_levels  = list(species_cat.categories)
    district_levels = list(district_cat.categories)

    s_idx = species_cat.codes
    d_idx = district_cat.codes
    n_sp  = len(species_levels)
    n_di  = len(district_levels)

    log_crown  = np.log(train['crown_m'].values)
    log_height = np.log(train['height_m'].values)
    log_dbh    = np.log(train['DBH_cm'].values)
    crown_mean, height_mean = log_crown.mean(), log_height.mean()
    log_crown_c  = log_crown  - crown_mean
    log_height_c = log_height - height_mean

    with pm.Model():
        alpha_global = pm.Normal('alpha_global', mu=2.5, sigma=0.5)
        beta_global  = pm.Normal('beta_global',  mu=1.0, sigma=0.3)
        gamma_global = pm.Normal('gamma_global', mu=0.3, sigma=0.3)

        tau_alpha = pm.HalfNormal('tau_alpha', sigma=0.5)
        tau_beta  = pm.HalfNormal('tau_beta',  sigma=0.3)
        tau_gamma = pm.HalfNormal('tau_gamma', sigma=0.3)

        # non-centered species effects keep the sampler stable under uneven n per species
        z_alpha = pm.Normal('z_alpha', 0, 1, shape=n_sp)
        z_beta  = pm.Normal('z_beta',  0, 1, shape=n_sp)
        z_gamma = pm.Normal('z_gamma', 0, 1, shape=n_sp)

        alpha_sp = pm.Deterministic('alpha_sp', alpha_global + tau_alpha * z_alpha)
        beta_sp  = pm.Deterministic('beta_sp',  beta_global  + tau_beta  * z_beta)
        gamma_sp = pm.Deterministic('gamma_sp', gamma_global + tau_gamma * z_gamma)

        tau_delta = pm.HalfNormal('tau_delta', sigma=0.3)
        delta_di  = pm.Normal('delta_dis', 0, tau_delta, shape=n_di)

        sigma_sp = pm.HalfNormal('sigma_sp', sigma=0.5, shape=n_sp)

        mu = (alpha_sp[s_idx]
              + beta_sp[s_idx]  * log_crown_c
              + gamma_sp[s_idx] * log_height_c
              + delta_di[d_idx])

        pm.Normal('y_obs', mu=mu, sigma=sigma_sp[s_idx], observed=log_dbh)

        trace = pm.sample(draws=draws, tune=tune, chains=chains, cores=chains,
                          target_accept=target_accept, random_seed=seed,
                          progressbar=False)

    encoders = dict(species_levels=species_levels, district_levels=district_levels,
                    crown_mean=crown_mean, height_mean=height_mean)
    return trace, encoders


def predict(trace, encoders, df_new, n_pred_samples=1000, seed=0):
    species_levels  = encoders['species_levels']
    district_levels = encoders['district_levels']

    s_idx = pd.Categorical(df_new['Tree species'], categories=species_levels).codes
    d_idx = pd.Categorical(df_new['District'],     categories=district_levels).codes
    if (s_idx < 0).any() or (d_idx < 0).any():
        raise ValueError("New data contains species or districts not in training set.")

    log_crown_c  = np.log(df_new['crown_m'].values)  - encoders['crown_mean']
    log_height_c = np.log(df_new['height_m'].values) - encoders['height_mean']

    post = trace.posterior
    alpha = post['alpha_sp'].values.reshape(-1, len(species_levels))
    beta  = post['beta_sp'].values.reshape(-1, len(species_levels))
    gamma = post['gamma_sp'].values.reshape(-1, len(species_levels))
    delta = post['delta_dis'].values.reshape(-1, len(district_levels))
    sigma = post['sigma_sp'].values.reshape(-1, len(species_levels))

    rng = np.random.default_rng(seed)
    pick = rng.choice(alpha.shape[0], size=n_pred_samples, replace=False)
    alpha, beta, gamma, delta, sigma = (a[pick] for a in [alpha, beta, gamma, delta, sigma])

    mu = (alpha[:, s_idx]
          + beta[:, s_idx]  * log_crown_c[None, :]
          + gamma[:, s_idx] * log_height_c[None, :]
          + delta[:, d_idx])
    sd = sigma[:, s_idx]
    return mu + sd * rng.standard_normal(mu.shape)


def summarize_predictions(log_dbh_samples, level=0.90):
    dbh = np.exp(log_dbh_samples)
    median = np.median(dbh, axis=0)
    lo = np.quantile(dbh, (1 - level) / 2, axis=0)
    hi = np.quantile(dbh, 1 - (1 - level) / 2, axis=0)
    return pd.DataFrame({
        'DBH_median_cm':                       median,
        f'DBH_p{int((1-level)/2*100)}_cm':     lo,
        f'DBH_p{int((1-(1-level)/2)*100)}_cm': hi,
    })


def carbon_from_dbh_samples(log_dbh_samples):
    """Verra VMD0001 applied per posterior sample → CO2 with credible intervals."""
    dbh           = np.exp(log_dbh_samples)
    agb           = np.exp(-1.996 + 2.32 * np.log(dbh))   # AGB in kg
    total_biomass = agb * 1.27                             # + roots
    carbon        = total_biomass * 0.5                    # 50% carbon
    co2           = carbon * (44 / 12)
    return pd.DataFrame({
        'CO2_median_kg': np.median(co2, axis=0),
        'CO2_p5_kg':     np.quantile(co2, 0.05, axis=0),
        'CO2_p95_kg':    np.quantile(co2, 0.95, axis=0),
    })


if __name__ == '__main__':
    df = prepare_data('f4f_ground_data_13Oct25.csv', min_species_count=30)
    train, test = stratified_split(df, frac=0.2)
    print(f"Train: {len(train):,} | Test: {len(test):,}")

    trace, encoders = build_and_fit(train)

    log_dbh_samples = predict(trace, encoders, test, n_pred_samples=1000)
    pred_df = summarize_predictions(log_dbh_samples, level=0.90)
    pred_df['DBH_true_cm'] = test['DBH_cm'].values
    pred_df['Species']     = test['Tree species'].values

    rmse = np.sqrt(np.mean((pred_df['DBH_median_cm'] - pred_df['DBH_true_cm'])**2))
    print(f"Test RMSE: {rmse:.2f} cm")

    co2_df = carbon_from_dbh_samples(log_dbh_samples)
    print("\nFirst 5 trees with 90% credible interval on CO2:")
    print(pd.concat([pred_df.head(), co2_df.head()], axis=1).to_string())
