"""Barplot resumen de eegnet_augmentation.csv: accuracy y F1 promedio (± std
entre sujetos) por config y ratio de aumento, con la baseline sin aumento
(config=none, ratio=0.0) como línea de referencia.

Uso:
    uv run python plot_eegnet_augmentation.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = 'eegnet_augmentation.csv'
CONFIGS = ['tts_gan_baseline', 'eeg_gan_vanilla_full', 'GAN_009_fm50_postnet1_stack0']
# paleta categórica fija (dataviz skill), misma asignación que summary_comparison.py
COLORS = {
    'tts_gan_baseline': '#2a78d6',
    'eeg_gan_vanilla_full': '#eb6834',
    'GAN_009_fm50_postnet1_stack0': '#1baf7a',
}
LABELS = {
    'tts_gan_baseline': 'TTS-GAN (baseline)',
    'eeg_gan_vanilla_full': 'EEG-GAN vanilla (full)',
    'GAN_009_fm50_postnet1_stack0': 'DDGAN (fm50/postnet/no-stack)',
}
METRICS = [('accuracy', 'Accuracy'), ('f1', 'F1')]

INK = '#0b0b0b'
INK_MUTED = '#898781'
GRID = '#e1e0d9'
SURFACE = '#fcfcfb'

df = pd.read_csv(CSV_PATH)
baseline = df[df.config == 'none']
ratios = sorted(df.loc[df.config != 'none', 'ratio'].unique())
n_configs = len(CONFIGS)
bar_w = 0.8 / n_configs
x = np.arange(len(ratios))

fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=SURFACE)
fig.suptitle('EEGNet: aumento de datos por config y ratio (media ± std entre sujetos)', color=INK, fontsize=12.5)

for ax, (col, title) in zip(axes, METRICS):
    ax.set_facecolor(SURFACE)
    base_mean, base_std = baseline[col].mean(), baseline[col].std()
    ax.axhline(base_mean, color=INK_MUTED, linestyle='--', linewidth=1.2, zorder=1,
               label=f'sin aumento (baseline, {base_mean:.3f})')
    ax.axhspan(base_mean - base_std, base_mean + base_std, color=INK_MUTED, alpha=0.08, zorder=0)

    for i, cfg in enumerate(CONFIGS):
        g = df[df.config == cfg].groupby('ratio')[col].agg(['mean', 'std']).reindex(ratios)
        offset = (i - (n_configs - 1) / 2) * bar_w
        ax.bar(x + offset, g['mean'], width=bar_w * 0.9, color=COLORS[cfg], label=LABELS[cfg],
               yerr=g['std'], capsize=2, error_kw=dict(linewidth=0.8, ecolor=INK_MUTED), zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels([f'{r:.1f}' for r in ratios], color=INK)
    ax.set_xlabel('ratio de aumento (fracción de trials Target)', color=INK, fontsize=9)
    ax.set_title(title, color=INK, fontsize=11)
    ax.tick_params(colors=INK_MUTED)
    ax.grid(axis='y', color=GRID, linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_color(GRID)

axes[0].legend(fontsize=7.5, loc='lower right', facecolor=SURFACE, edgecolor=GRID)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out_path = 'comparison_plots/eegnet_augmentation_barplot.png'
fig.savefig(out_path, dpi=150, facecolor=SURFACE)
print(f'-> {out_path}')
