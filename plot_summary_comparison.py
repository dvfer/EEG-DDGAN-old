"""Resume en un plot las métricas de las 3 tablas comparadas (sujetos 2-10):
tts_gan_baseline vs eeg_gan_vanilla_full vs GAN_009_fm50_postnet1_stack0 (tesis).

Boxplot (cuartiles) + puntos por sujeto (n=9 es poco para confiar solo en la
caja) por métrica, escala log en las 3 que abarcan >100x de rango.

Uso:
    uv run python plot_summary_comparison.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CONFIGS = {
    'tts_gan_baseline': 'metrics_tts_gan_baseline.csv',
    'eeg_gan_vanilla_full': 'metrics_eeg_gan_vanilla_full.csv',
    'GAN_009_fm50_postnet1_stack0': 'metrics_GAN_009_fm50_postnet1_stack0.csv',
}
# paleta categórica fija (dataviz skill): slot1 azul, slot2 naranja, slot3 aqua
COLORS = {
    'tts_gan_baseline': '#2a78d6',
    'eeg_gan_vanilla_full': '#eb6834',
    'GAN_009_fm50_postnet1_stack0': '#1baf7a',
}
LABELS = {
    'tts_gan_baseline': 'TTS-GAN\n(baseline)',
    'eeg_gan_vanilla_full': 'EEG-GAN\nvanilla (full)',
    'GAN_009_fm50_postnet1_stack0': 'DDGAN\n(fm50/postnet/no-stack)',
}
METRICS = [
    ('jsd_mean', 'JSD (espectral)', True),
    ('mmd2', 'MMD²', True),
    ('erp_amp_err', 'Error amplitud pico ERP', True),
    ('erp_lat_err_ms', 'Error latencia pico ERP (ms)', False),
]

INK = '#0b0b0b'
INK_MUTED = '#898781'
GRID = '#e1e0d9'
SURFACE = '#fcfcfb'

frames = []
for name, path in CONFIGS.items():
    df = pd.read_csv(path)
    df = df[df['subject'] >= 2].copy()
    df['config'] = name
    frames.append(df)
data = pd.concat(frames, ignore_index=True)

fig, axes = plt.subplots(2, 2, figsize=(11, 8), facecolor=SURFACE)
fig.suptitle('Comparación de métricas por config (sujetos 2-10, n=9 c/u)', color=INK, fontsize=13)

for ax, (col, title, use_log) in zip(axes.flat, METRICS):
    ax.set_facecolor(SURFACE)
    box_data = [data.loc[data['config'] == cfg, col].values for cfg in CONFIGS]
    bp = ax.boxplot(box_data, patch_artist=True, widths=0.5, showfliers=False,
                     medianprops=dict(color=INK, linewidth=1.5))
    for patch, cfg in zip(bp['boxes'], CONFIGS):
        patch.set_facecolor(COLORS[cfg])
        patch.set_alpha(0.35)
        patch.set_edgecolor(COLORS[cfg])
    for i, cfg in enumerate(CONFIGS, start=1):
        vals = data.loc[data['config'] == cfg, col].values
        jitter = np.random.default_rng(0).uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, color=COLORS[cfg], s=22,
                   zorder=3, edgecolor=SURFACE, linewidth=0.5)

    ax.set_xticks(range(1, len(CONFIGS) + 1))
    ax.set_xticklabels([LABELS[c] for c in CONFIGS], color=INK, fontsize=8.5)
    if use_log:
        ax.set_yscale('log')
    ax.set_title(title, color=INK, fontsize=10.5)
    ax.tick_params(colors=INK_MUTED)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color(GRID)

fig.tight_layout(rect=[0, 0, 1, 0.96])
out_path = 'comparison_plots/summary_comparison.png'
fig.savefig(out_path, dpi=150, facecolor=SURFACE)
print(f'-> {out_path}')
