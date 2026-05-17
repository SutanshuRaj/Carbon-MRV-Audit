"""Generate figures for the white paper. Reads from ./out_v3/."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE    = Path(__file__).parent
FIG_DIR = HERE / 'figures'
FIG_DIR.mkdir(exist_ok=True)

OUT_V3 = HERE / 'out_v3'
RESID = pd.read_csv(OUT_V3 / 'residuals_v3.csv')
CORR  = pd.read_csv(OUT_V3 / 'v3_1_corrected_carbon.csv')

plt.rcParams.update({
    'font.size':         10,
    'axes.titlesize':    11,
    'axes.labelsize':    10,
    'legend.fontsize':    9,
    'figure.dpi':       150,
    'savefig.bbox':  'tight',
    'savefig.dpi':      200,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})


def fig_jensen_curve():
    sigma = np.linspace(0.05, 0.60, 200)
    bias_pct = (np.exp((2.32 * sigma) ** 2 / 2) - 1) * 100

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(sigma, bias_pct, color='#222', linewidth=2)
    ax.fill_between(sigma, 0, bias_pct, color='#222', alpha=0.06)

    ax.axvspan(0.20, 0.25, color='#1f77b4', alpha=0.18,
               label='F4F GPR plausible σ (MAPE 8.59%)')
    ax.axvspan(0.14, 0.30, color='#2ca02c', alpha=0.10,
               label='Hierarchical model per-species σ range')

    for s in [0.20, 0.25, 0.30]:
        b = (np.exp((2.32 * s) ** 2 / 2) - 1) * 100
        ax.plot([s], [b], 'o', color='#222', markersize=4)
        ax.annotate(f'σ={s:.2f}\n→ {b:.1f}%',
                    xy=(s, b), xytext=(s + 0.02, b + 1.5),
                    fontsize=8.5, color='#222')

    ax.set_xlabel('Per-tree predictive σ on log(DBH)')
    ax.set_ylabel('Biomass under-prediction (%)\nfrom point-estimate back-transform')
    ax.set_title('Jensen back-transform bias is structural and model-agnostic')
    ax.set_xlim(0.05, 0.55)
    ax.set_ylim(0, 36)
    ax.legend(loc='lower right', framealpha=0.95)
    ax.grid(alpha=0.25)
    fig.savefig(FIG_DIR / 'fig_jensen_curve.png')
    plt.close(fig)


def fig_size_bin_bias():
    bins   = [0, 5, 10, 15, 20, 30, 50, 200]
    labels = ['<5', '5–10', '10–15', '15–20', '20–30', '30–50', '>50']
    df = CORR.copy()
    df['bin'] = pd.cut(df['dbh_true'], bins, labels=labels)

    agg = df.groupby('bin', observed=True).agg(
        n=('dbh_true', 'count'),
        co2_share=('co2_true', lambda s: 100 * s.sum() / df['co2_true'].sum()),
        bias_wrong=('co2_wrong', lambda s:
                    100 * (s.sum() - df.loc[s.index, 'co2_true'].sum())
                    / df.loc[s.index, 'co2_true'].sum()),
        bias_corr=('co2_correct', lambda s:
                    100 * (s.sum() - df.loc[s.index, 'co2_true'].sum())
                    / df.loc[s.index, 'co2_true'].sum()),
    ).reset_index()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 5.5),
                                    gridspec_kw={'height_ratios': [3, 1]},
                                    sharex=True)

    x, w = np.arange(len(agg)), 0.38
    ax1.bar(x - w/2, agg['bias_wrong'], w, color='#d62728', alpha=0.85,
            label='Point-estimate back-transform')
    ax1.bar(x + w/2, agg['bias_corr'], w, color='#2ca02c', alpha=0.85,
            label='Per-sample back-transform (Jensen-correct)')
    ax1.axhline(0, color='k', linewidth=0.6)
    ax1.set_ylabel('CO₂ bias on test set (%)')
    ax1.set_title('Jensen fix: −10% plot-level bias → +4.5%; residual is partial-pooling shrinkage')
    ax1.legend(loc='lower right', framealpha=0.95)
    ax1.grid(axis='y', alpha=0.25)

    for i, (bw, bc) in enumerate(zip(agg['bias_wrong'], agg['bias_corr'])):
        ax1.annotate(f'{bw:+.0f}', (i - w/2, bw),
                     ha='center', va='bottom' if bw >= 0 else 'top',
                     fontsize=8, color='#7c1414')
        ax1.annotate(f'{bc:+.0f}', (i + w/2, bc),
                     ha='center', va='bottom' if bc >= 0 else 'top',
                     fontsize=8, color='#1a5e1a')

    ax2.bar(x, agg['co2_share'], color='#555', alpha=0.65)
    ax2.set_xticks(x)
    ax2.set_xticklabels(agg['bin'])
    ax2.set_ylabel('% of test\nplot CO₂')
    ax2.set_xlabel('True DBH bin (cm)')
    ax2.grid(axis='y', alpha=0.25)
    for i, v in enumerate(agg['co2_share']):
        ax2.annotate(f'{v:.0f}%', (i, v), ha='center', va='bottom', fontsize=8)

    fig.savefig(FIG_DIR / 'fig_size_bin_bias.png')
    plt.close(fig)


def fig_calibration():
    nominal = np.array([50, 60, 70, 80, 90, 95])
    v3_emp  = np.array([57.4, 67.1, 76.2, 85.6, 93.4, 96.7])
    # F4F published only the slope (1.08); approximate the line and cap at 100%.
    f4f_line = np.minimum(100.0, 1.08 * nominal)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot([0, 100], [0, 100], '--', color='#555', linewidth=1, label='Perfect calibration')
    ax.plot(nominal, v3_emp, 'o-', color='#2ca02c', linewidth=2, markersize=7,
            label='Hierarchical model v3 (slope 0.88, under-confident)')
    ax.plot(nominal, f4f_line, 's--', color='#d62728', linewidth=1.5, markersize=7,
            alpha=0.85, label='F4F GPR (slope 1.08 reported, over-confident)')

    ax.set_xlabel('Nominal coverage (%)')
    ax.set_ylabel('Empirical coverage (%)')
    ax.set_title('Calibration: opposite failure modes')
    ax.set_xlim(40, 100)
    ax.set_ylim(40, 105)
    ax.legend(loc='lower right', framealpha=0.95)
    ax.grid(alpha=0.25)

    fig.savefig(FIG_DIR / 'fig_calibration.png')
    plt.close(fig)


def fig_species_bias():
    df = CORR.copy()
    total = df['co2_true'].sum()
    g = df.groupby('species').agg(
        n=('dbh_true', 'count'),
        co2_true=('co2_true', 'sum'),
        co2_corr=('co2_correct', 'sum'),
    )
    g['share_pct'] = 100 * g['co2_true'] / total
    g['bias_pct']  = 100 * (g['co2_corr'] - g['co2_true']) / g['co2_true']
    g = g[g['share_pct'] >= 0.5].sort_values('share_pct', ascending=True)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    sizes  = np.clip(g['share_pct'] * 12, 30, 700)
    colors = ['#2ca02c' if b >= 0 else '#d62728' for b in g['bias_pct']]
    ax.scatter(g['bias_pct'], range(len(g)), s=sizes, c=colors, alpha=0.75,
               edgecolor='#222', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.6)
    ax.set_yticks(range(len(g)))
    ax.set_yticklabels([f"{sp}  (n={n}, {sh:.1f}% CO₂)"
                        for sp, n, sh in zip(g.index, g['n'], g['share_pct'])])
    ax.set_xlabel('CO₂ bias on test set, post-Jensen (%)')
    ax.set_title('Per-species bias drivers — marker size ∝ share of plot CO₂')
    ax.grid(axis='x', alpha=0.25)

    for i, b in enumerate(g['bias_pct']):
        ax.annotate(f'{b:+.1f}%', (b, i), xytext=(8 if b >= 0 else -8, 0),
                    textcoords='offset points',
                    ha='left' if b >= 0 else 'right', va='center', fontsize=8)
    ax.set_xlim(-30, 30)
    fig.savefig(FIG_DIR / 'fig_species_bias.png')
    plt.close(fig)


def fig_pred_vs_actual():
    df = CORR.copy()
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.scatter(df['dbh_true'], df['dbh_correct'], s=10, alpha=0.45,
               color='#1f77b4', edgecolor='none')
    lim = max(df['dbh_true'].max(), df['dbh_correct'].max()) * 1.05
    ax.plot([0, lim], [0, lim], '--', color='#444', linewidth=1, label='1:1 line')
    ax.set_xlabel('True DBH (cm)')
    ax.set_ylabel('Predicted DBH (cm, Jensen-corrected)')
    ax.set_title('Hierarchical model: predicted vs actual DBH on held-out test set')
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.legend(loc='upper left')
    ax.grid(alpha=0.25)

    rmse = float(np.sqrt(((df['dbh_correct'] - df['dbh_true']) ** 2).mean()))
    mae  = float((df['dbh_correct'] - df['dbh_true']).abs().mean())
    ax.text(0.98, 0.02, f'n = {len(df):,}\nRMSE = {rmse:.2f} cm\nMAE  = {mae:.2f} cm',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=9, family='monospace',
            bbox=dict(facecolor='white', edgecolor='#ccc', alpha=0.95))

    fig.savefig(FIG_DIR / 'fig_pred_vs_actual.png')
    plt.close(fig)


def fig_haircut_vs_quantile():
    df = CORR.copy()
    true_total = df['co2_true'].sum()
    mean_total = df['co2_correct'].sum()
    haircut_80 = 0.80 * mean_total

    # Bootstrap plot totals to get posterior quantiles
    rng = np.random.default_rng(0)
    sims = np.array([df['co2_correct'].iloc[rng.choice(len(df), len(df), replace=True)].sum()
                     for _ in range(2000)])
    p25 = float(np.quantile(sims, 0.25))
    p10 = float(np.quantile(sims, 0.10))

    bars = ['Ground truth\n(field, extrapolated)',
            'Hierarchical\nmean',
            'Posterior p25',
            'Posterior p10',
            'F4F 80% rule\non mean']
    vals   = [true_total, mean_total, p25, p10, haircut_80]
    colors = ['#666', '#2ca02c', '#1f77b4', '#08306b', '#d62728']

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    pos = np.arange(len(bars))
    ax.bar(pos, vals, color=colors, alpha=0.85, width=0.65)
    for i, v in enumerate(vals):
        pct = 100 * (v - true_total) / true_total
        ax.annotate(f'{v/1000:.1f}t\n({pct:+.1f}%)',
                    (i, v), ha='center', va='bottom', fontsize=8.5)
    ax.set_xticks(pos)
    ax.set_xticklabels(bars, fontsize=9)
    ax.set_ylabel('Plot-level CO₂ (kg)')
    ax.set_title('80% haircut leaves credits on the table; calibrated p25 is tighter')
    ax.grid(axis='y', alpha=0.25)
    ax.set_ylim(0, max(vals) * 1.18)

    fig.savefig(FIG_DIR / 'fig_haircut_vs_quantile.png')
    plt.close(fig)


def main():
    for fn in (fig_jensen_curve, fig_size_bin_bias, fig_calibration,
               fig_species_bias, fig_pred_vs_actual, fig_haircut_vs_quantile):
        fn()
        print(f'  {fn.__name__}')


if __name__ == '__main__':
    main()
