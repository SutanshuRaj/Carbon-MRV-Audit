"""Figures for the workshop paper (white_paper/main.tex), written to figures/paper_*.pdf.

  paper_jensen.pdf       analytic back-transform bias vs. per-tree predictive sigma,
                         with the observed mean/point ratio for each model overlaid
  paper_calibration.pdf  tree-level vs. plot-level interval coverage (GPR, v3)

Refits the GPR (~2 min); everything else reuses saved traces and out_plots/per_plot.csv."""

import warnings
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ablation_hier import ARMS, predict as predict_ablation
from baseline_gpr import fit_gpr, fit_pooled_ols
from hierarchical_dbh_v31 import co2_kg, predict_log, rebuild_encoders

warnings.filterwarnings('ignore')

V3_DIR, ABL_DIR, PLOT_DIR = Path('out_v3'), Path('out_ablation'), Path('out_plots')
FIG_DIR = Path('figures')
SEED    = 0

INK, INK_2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
BLUE, ORANGE     = '#2a78d6', '#eb6834'
EXPONENT         = 2.32 ** 2 / 2          # bias factor = exp(EXPONENT * sigma^2)

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 9, 'axes.edgecolor': INK_2,
    'axes.labelcolor': INK, 'xtick.color': INK_2, 'ytick.color': INK_2,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'legend.frameon': False, 'savefig.bbox': 'tight', 'savefig.dpi': 300, 'pdf.fonttype': 42,
})


def all_predictions(train, encoders, test):
    preds = {'GPR': fit_gpr(train, test, log_inputs=False),
             'GPR (log inputs)': fit_gpr(train, test, log_inputs=True),
             'Pooled OLS': fit_pooled_ols(train, test)}
    for name, arm in ARMS.items():
        preds[name] = predict_ablation(az.from_netcdf(ABL_DIR / f'trace_{name}.nc'),
                                       encoders, test, arm)
    preds['v3'] = predict_log(az.from_netcdf(V3_DIR / 'trace_v3.nc'), encoders, test,
                              n_samples=1000, seed=SEED)
    return preds


def fig_jensen(preds):
    sigma = np.linspace(0.0, 0.40, 200)
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.plot(sigma, 100 * (np.exp(EXPONENT * sigma**2) - 1), color=INK, lw=2,
            label=r'Analytic: $\exp((2.32\sigma)^2/2) - 1$')

    rows = []
    for name, lp in preds.items():
        point = co2_kg(np.exp(np.median(lp, axis=0))).sum()
        mean  = co2_kg(np.exp(lp)).sum(axis=1).mean()
        rows.append((name, np.median(lp.std(axis=0)), 100 * (mean / point - 1)))
    obs = pd.DataFrame(rows, columns=['model', 'sigma', 'gap_pct'])
    ax.scatter(obs['sigma'], obs['gap_pct'], s=42, color=BLUE, edgecolor='white',
               linewidth=1.2, zorder=3, label='Observed, test set (7 models)')

    ax.set_xlabel(r'Median per-tree predictive $\sigma$ on $\log(\mathrm{DBH})$')
    ax.set_ylabel('Mean ÷ point-estimate\nCO$_2$ total − 1 (%)')
    ax.set_xlim(0, 0.40)
    ax.set_ylim(0, 55)
    ax.legend(loc='upper left')
    fig.savefig(FIG_DIR / 'paper_jensen.pdf')
    plt.close(fig)
    return obs


def tree_coverage(lp, y, levels):
    return np.array([np.mean((y >= np.quantile(lp, (1-l)/2, axis=0)) &
                             (y <= np.quantile(lp, 1-(1-l)/2, axis=0))) for l in levels])


def plot_coverage(pit, levels):
    return np.array([np.mean((pit >= (1-l)/2) & (pit <= 1-(1-l)/2)) for l in levels])


def fig_calibration(preds, test):
    levels = np.linspace(0.1, 0.9, 9)
    y = np.log(test['DBH_cm'].values)
    plots = pd.read_csv(PLOT_DIR / 'per_plot.csv')
    plots = plots[plots['grouping'] == 'block_year']
    panels = {'GPR (crown, height)': ('GPR', 'GPR (TC, TH)'),
              'Hierarchical (+ species, district, age)': ('v3', 'v3: + field age (full model)')}

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), sharey=True)
    for ax, (title, (key, plot_model)) in zip(axes, panels.items()):
        pit = plots.loc[plots['model'] == plot_model, 'pit'].values
        ax.plot([0, 1], [0, 1], color=INK_2, lw=1, ls=(0, (3, 3)), label='Ideal')
        ax.plot(levels, tree_coverage(preds[key], y, levels), color=BLUE, lw=2,
                marker='o', ms=5, mec='white', mew=1, label='Per tree (n = 1,257)')
        ax.plot(levels, plot_coverage(pit, levels), color=ORANGE, lw=2,
                marker='o', ms=5, mec='white', mew=1, label=f'Per plot (n = {len(pit)})')
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xlabel('Nominal central-interval level')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel('Empirical coverage')
    axes[0].legend(loc='upper left')
    fig.savefig(FIG_DIR / 'paper_calibration.pdf')
    plt.close(fig)


def main():
    train, encoders = rebuild_encoders(V3_DIR / 'train_v3.csv')
    test = pd.read_csv(V3_DIR / 'test_v3.csv')
    preds = all_predictions(train, encoders, test)
    obs = fig_jensen(preds)
    obs['analytic_pct'] = 100 * (np.exp(EXPONENT * obs['sigma']**2) - 1)
    print(obs.round(3).to_string(index=False))
    fig_calibration(preds, test)
    print(f"Saved {FIG_DIR}/paper_jensen.pdf and {FIG_DIR}/paper_calibration.pdf")


if __name__ == '__main__':
    main()
