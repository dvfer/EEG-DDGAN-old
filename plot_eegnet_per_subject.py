"""Barplot de F1 (clase Target) por sujeto -- un subplot por sujeto, sin
promediar entre sujetos (a diferencia de plot_eegnet_augmentation.py). Sirve
para ver directo los casos puntuales que el std entre sujetos esconde (ej.
la caída de eeg_gan_vanilla_full en sujeto 4/ratio 0.3).

Uso:
    uv run python plot_eegnet_per_subject.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = 'eegnet_augmentation.csv'
METRIC = 'f1'
CONFIGS = ['tts_gan_baseline', 'eeg_gan_vanilla_full', 'GAN_009_fm50_postnet1_stack0']
COLORS = {
    'tts_gan_baseline': '#2a78d6',
    'eeg_gan_vanilla_full': '#eb6834',
    'GAN_009_fm50_postnet1_stack0': '#1baf7a',
}
LABELS = {
    'tts_gan_baseline': 'TTS-GAN',
    'eeg_gan_vanilla_full': 'EEG-GAN vanilla',
    'GAN_009_fm50_postnet1_stack0': 'DDGAN',
}

INK = '#0b0b0b'
INK_MUTED = '#898781'
GRID = '#e1e0d9'
SURFACE = '#fcfcfb'

df = pd.read_csv(CSV_PATH)
df = df[df.subject >= 2]  # sujeto 1 no tiene tts_gan_baseline/DDGAN -- fuera de la comparación (ver metrics_*.csv)
subjects = sorted(df.subject.unique())
ratios = sorted(df.loc[df.config != 'none', 'ratio'].unique())
x = np.arange(len(ratios))
n_configs = len(CONFIGS)
bar_w = 0.8 / n_configs

ncols = 5
nrows = int(np.ceil(len(subjects) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.4 * nrows), facecolor=SURFACE, squeeze=False)
fig.suptitle(f'EEGNet por sujeto -- {METRIC} (clase Target)', color=INK, fontsize=13)

for i, subject in enumerate(subjects):
    ax = axes[i // ncols, i % ncols]
    ax.set_facecolor(SURFACE)
    sdf = df[df.subject == subject]

    base_row = sdf[sdf.config == 'none']
    if not base_row.empty:
        base_val = base_row[METRIC].iloc[0]
        ax.axhline(base_val, color=INK_MUTED, linestyle='--', linewidth=1.1, zorder=1)

    present_configs = [c for c in CONFIGS if not sdf[sdf.config == c].empty]
    for j, cfg in enumerate(present_configs):
        vals = sdf[sdf.config == cfg].set_index('ratio')[METRIC].reindex(ratios)
        offset = (j - (len(present_configs) - 1) / 2) * bar_w
        ax.bar(x + offset, vals, width=bar_w * 0.9, color=COLORS[cfg], label=LABELS[cfg], zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels([f'{r:.1f}' for r in ratios], color=INK, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title(f'Sujeto {subject}', color=INK, fontsize=10.5)
    ax.tick_params(colors=INK_MUTED)
    ax.grid(axis='y', color=GRID, linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_color(GRID)

for i in range(len(subjects), nrows * ncols):
    axes[i // ncols, i % ncols].axis('off')

handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[c]) for c in CONFIGS]
handles.append(plt.Line2D([0], [0], color=INK_MUTED, linestyle='--'))
fig.legend(handles, [LABELS[c] for c in CONFIGS] + ['sin aumento (baseline)'],
           loc='lower center', ncol=4, fontsize=9, facecolor=SURFACE, edgecolor=GRID, bbox_to_anchor=(0.5, -0.02))

fig.tight_layout(rect=[0, 0.03, 1, 0.95])
out_path = 'comparison_plots/eegnet_per_subject.png'
fig.savefig(out_path, dpi=150, facecolor=SURFACE, bbox_inches='tight')
print(f'-> {out_path}')
