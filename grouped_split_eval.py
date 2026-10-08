"""Plot-held-out evaluation. The main results use a tree-level split, so test plots
share farms with training trees. Here whole pseudo-plots (District x Block x Plantation
Year) are held out with 5-fold GroupKFold, every model is refit per fold, and plot-level
calibration is scored on farms the models never saw.

  - Test trees whose species is absent from a fold's training set are dropped (counted).
  - A district absent from training gets district effect 0 (population-level prediction).
  - Per-fold predictions are cached in OUT_DIR/fold{k}.npz, so a rerun resumes.

~45-60 min on a laptop (4 MCMC fits + 1 GPR per fold)."""

import sys
import warnings
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import ablation_hier
import hierarchical_dbh_v3 as v3
from baseline_gpr import score
from pseudo_plot_eval import gpr_joint_samples, per_plot, summarise

warnings.filterwarnings('ignore')

OUT_DIR   = Path('out_grouped')
N_FOLDS   = 5
N_SAMPLES = 1000
MIN_TREES = 10
SEED      = 0

MODELS = {
    'GPR (TC, TH)':                          None,
    'A0: drone + district (species pooled)': dict(species=False, district=True,  age=False),
    'A1: + species ID, no district':         dict(species=True,  district=False, age=False),
    'A2: species + district, no age':        dict(species=True,  district=True,  age=False),
    'v3: + field age (full model)':          dict(species=True,  district=True,  age=True),
}

OUT_DIR.mkdir(exist_ok=True)


def encoders_for(train):
    return dict(
        species_levels  = list(pd.Categorical(train['Tree species']).categories),
        district_levels = list(pd.Categorical(train['District']).categories),
        crown_mean  = np.log(train['crown_m']).mean(),
        height_mean = np.log(train['height_m']).mean(),
        age_mean    = np.log(train['tree_age_years']).mean(),
    )


def predict(trace, enc, df, spec):
    """Posterior predictive log(DBH); unseen district -> delta = 0."""
    post = trace.posterior
    flat = lambda v: post[v].values.reshape(-1, *post[v].shape[2:])
    rng = np.random.default_rng(SEED)
    pick = rng.choice(post.sizes['chain'] * post.sizes['draw'], size=N_SAMPLES, replace=False)
    get = lambda v: flat(v)[pick]

    s_idx = (pd.Categorical(df['Tree species'], categories=enc['species_levels']).codes
             if spec['species'] else np.zeros(len(df), dtype=int))
    assert (s_idx >= 0).all(), "unseen species must be dropped before predict"
    d_idx = pd.Categorical(df['District'], categories=enc['district_levels']).codes
    d_known = d_idx >= 0

    xc = np.log(df['crown_m'].values)  - enc['crown_mean']
    xh = np.log(df['height_m'].values) - enc['height_mean']
    a, b, z, g, sg = (get(v) for v in ('alpha', 'beta', 'zeta', 'gamma', 'sigma'))
    mu = (a[:, s_idx] + b[:, s_idx] * xc + z[:, s_idx] * xc**2 + g[:, s_idx] * xh)
    if spec['age']:
        xa = (np.log(df['tree_age_years'].values) - enc['age_mean']) * df['_age_informative'].values
        mu = mu + get('eta')[:, s_idx] * xa
    if spec['district']:
        d = get('delta')
        mu = mu + np.where(d_known, d[:, np.clip(d_idx, 0, None)], 0.0)
    return mu + sg[:, s_idx] * rng.standard_normal(mu.shape)


def fit(train, enc, spec):
    if spec['age']:
        return v3.build_and_fit(train)[0]
    arm = dict(species=spec['species'], district=spec['district'])
    return ablation_hier.fit(train, enc, arm)


def run_fold(k, df, tr_idx, te_idx):
    cache = OUT_DIR / f'fold{k}.npz'
    train, test = df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()
    seen = test['Tree species'].isin(train['Tree species'].unique())
    test = test[seen]
    if cache.exists():
        print(f"fold {k}: cached", flush=True)
        z = np.load(cache, allow_pickle=True)
        return test, {m: z[f'm{i}'] for i, m in enumerate(MODELS)}, int((~seen).sum())

    enc = encoders_for(train)
    print(f"\nfold {k}: train {len(train):,} | test {len(test):,} "
          f"(dropped {(~seen).sum()} unseen-species trees) | "
          f"{test['plot'].nunique()} held-out plots", flush=True)
    preds = {}
    for name, spec in MODELS.items():
        print(f"  fitting {name}", flush=True)
        if spec is None:
            preds[name] = gpr_joint_samples(train, test)
        else:
            trace = fit(train, enc, spec)
            rhat = float(az.summary(trace, var_names=[v for v in trace.posterior.data_vars
                                                       if v.startswith('tau_') or v.endswith('_g')])
                         ['r_hat'].max())
            div = int(trace.sample_stats['diverging'].sum())
            print(f"    max r_hat {rhat:.3f}, divergences {div}", flush=True)
            preds[name] = predict(trace, enc, test, spec)
    np.savez(cache, **{f'm{i}': preds[m] for i, m in enumerate(MODELS)})
    return test, preds, int((~seen).sum())


def main():
    df = v3.prepare_data(v3.CSV_PATH)
    df['plot'] = df['District'] + '|' + df['Block'] + '|' + df['Plantation Year'].astype(str)
    print(f"{df['plot'].nunique()} block-year groups in {len(df):,} trees; {N_FOLDS}-fold GroupKFold")

    tests, preds, dropped = [], {m: [] for m in MODELS}, 0
    for k, (tr, te) in enumerate(GroupKFold(n_splits=N_FOLDS).split(df, groups=df['plot'])):
        test, p, d = run_fold(k, df, tr, te)
        tests.append(test.assign(fold=k))
        dropped += d
        for m in MODELS:
            preds[m].append(p[m])

    test = pd.concat(tests)
    test.index = pd.RangeIndex(len(test))
    preds = {m: np.concatenate(v, axis=1) for m, v in preds.items()}

    tree = pd.DataFrame([score(m, lp, test) for m, lp in preds.items()]).set_index('model')
    keys = {'block_year_heldout': test['plot']}
    plots = pd.DataFrame([r for m, lp in preds.items() for r in per_plot(m, lp, test, keys)])
    summary = summarise(plots)

    tree.to_csv(OUT_DIR / 'tree_level.csv')
    plots.to_csv(OUT_DIR / 'per_plot.csv', index=False)
    summary.to_csv(OUT_DIR / 'policy_summary.csv', index=False)

    pd.set_option('display.width', 220)
    pd.set_option('display.max_columns', 20)
    print(f"\nPooled over folds: {len(test):,} held-out trees ({dropped} dropped as unseen species)")
    print(tree.round(3).T.to_string())
    s = summary[summary['policy'] != 'PIT']
    print("\nHeld-out plot policy outcomes:")
    print(s.drop(columns=['grouping', 'pit_in_central50_pct', 'pit_in_central80_pct'],
                 errors='ignore').round(1).to_string(index=False))
    print("\nHeld-out plot coverage (nominal 50% / 80%):")
    print(summary[summary['policy'] == 'PIT'][['model', 'n_plots', 'pit_in_central50_pct',
                                                 'pit_in_central80_pct']].round(1).to_string(index=False))


if __name__ == '__main__':
    if len(sys.argv) > 1:
        sys.exit("grouped_split_eval.py takes no arguments")
    main()
