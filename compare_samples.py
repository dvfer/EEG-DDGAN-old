"""Genera muestras sintéticas desde un checkpoint entrenado (moabb_pipeline.py)
y las compara con los datos reales del sujeto: ERP overlay, PSD (Welch) +
divergencia espectral (JSD), MMD² distribucional, y PCA/t-SNE.

Reutiliza eeggan.generate_samples_main (generación) y
eeggan.helpers.visualize_pca.visualization_dim_reduction (PCA/t-SNE) — no
reimplementa nada que el paquete ya resuelva.

Estilo de gráficos y métricas (JSD, MMD²) inspirados en
/home/dvfer/Documents/GANNTHESIS/gann/EEG-DDGAN/signal_analysis.ipynb.

Uso:
    python compare_samples.py --subjects 1 2 3
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')  # sin display en el servidor de entrenamiento
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from eeg_metrics import compute_psd, spectral_jsd, mmd_rbf, erp_peak_metrics
from eeggan.helpers.dataloader import Dataloader
from eeggan.helpers.visualize_pca import visualization_dim_reduction
from moabb_pipeline import MODEL_PREFIX  # single source of truth (fm/postnet/stack en el nombre)

DATA_DIR      = 'subject_data/train'    # usado para entrenar (y como fallback si no hay test)
TEST_DATA_DIR = 'subject_data/test'     # held-out — lo que se usa como "real" para las métricas
GAN_DIR   = 'trained_models'
GEN_DIR   = 'generated_samples'
PLOT_DIR  = 'comparison_plots'

CONDITIONS = {'NonTarget': 0, 'Target': 1}
MAX_SAMPLES_PER_COND = 200  # cap para no generar batches gigantes
FS = 256  # sampling rate de BNCI2014_009; ajustar si el pipeline resamplea
BP_HZ = 24     # banda real del paradigm P300 de MOABB es [1,24] Hz
PSD_FMAX = 25  # límite de vista del PSD (referencia: signal_analysis.ipynb)

# Estilo inspirado en signal_analysis.ipynb (figuras estilo paper)
COLORS = {'real': '#0072BD', 'gen': '#D65F5F'}
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.axisbelow': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def _generate_condition_csv(model_path, out_csv, condition, n_samples):
    from eeggan.generate_samples_main import main as generate_main
    generate_main([
        f'model={model_path}',
        f'save_name={out_csv}',
        f'conditions={condition}',
        f'num_samples_total={n_samples}',
        f'num_samples_parallel={n_samples}',
    ])


def generate_synthetic(model_path, real_csv, out_csv, n_per_cond=None):
    """Genera tantas muestras sintéticas como trials reales por condición
    (hasta MAX_SAMPLES_PER_COND) y las junta en un único CSV con el mismo
    formato largo que el real.

    n_per_cond: dict {nombre_condición: n} para fijar el tamaño a mano en vez
    de derivarlo de los trials reales (y saltear el cap de MAX_SAMPLES_PER_COND).
    n=0 saltea esa condición. Lo usa el pool de aumento, que solo necesita
    Target y en cantidad >> trials reales -- ver sample_pool() en
    train_eegnet_augmentation.py."""
    real_df = pd.read_csv(real_csv)
    n_trials_per_cond = real_df.groupby('Condition')['Trial'].nunique()

    os.makedirs(GEN_DIR, exist_ok=True)
    parts = []
    for name, cond in CONDITIONS.items():
        if n_per_cond is not None:
            n = int(n_per_cond.get(name, 0))
        else:
            n = min(int(n_trials_per_cond.get(cond, 0)), MAX_SAMPLES_PER_COND)
        if n == 0:
            continue
        tmp_csv = os.path.join(GEN_DIR, f'_tmp_{name}.csv')
        _generate_condition_csv(model_path, tmp_csv, cond, n)
        parts.append(pd.read_csv(tmp_csv))
        os.remove(tmp_csv)

    df = pd.concat(parts, ignore_index=True)
    df.to_csv(out_csv, index=False)
    return out_csv


def load(csv_path, norm_data, norm_min=None, norm_max=None):
    """norm_min/norm_max: si se dan, normaliza con ESTOS valores en vez de los
    propios del csv (para comparar el held-out de test en la misma escala en la
    que el generador vio los datos de train — ver train_norm_stats())."""
    if norm_min is not None:
        dl = Dataloader(path=csv_path, norm_data=False,
                         kw_time='Time', kw_conditions='Condition', kw_channel='Electrode')
        data = dl.get_data(shuffle=False)[:, 1:].numpy()
        data = (data - norm_min) / (norm_max - norm_min)
    else:
        dl = Dataloader(path=csv_path, norm_data=norm_data,
                         kw_time='Time', kw_conditions='Condition', kw_channel='Electrode')
        data = dl.get_data(shuffle=False)[:, 1:].numpy()       # (trial, seq, channel)
    labels = dl.get_labels()[:, 0, 0].numpy()              # (trial,)
    return data, labels


def train_norm_stats(train_csv):
    """Min/max del CSV de train — la misma escala que Dataloader(norm_data=True)
    le dio al generador durante el entrenamiento (ver dataloader.py: dataset_min/
    dataset_max se calculan siempre, se apliquen o no)."""
    dl = Dataloader(path=train_csv, norm_data=False,
                     kw_time='Time', kw_conditions='Condition', kw_channel='Electrode')
    return dl.dataset_min.item(), dl.dataset_max.item()


def _long_df(data_2d, tipo_senal):
    """(trials, tiempo) -> formato largo para sns.lineplot (promedio+banda automáticos)."""
    n_trials, seq_len = data_2d.shape
    return pd.DataFrame({
        'tiempo': np.tile(np.arange(seq_len), n_trials),
        'amplitud': data_2d.flatten(),
        'tipo_senal': tipo_senal,
    })


def plot_erp_overlay(real_data, real_labels, gen_data, gen_labels, out_path):
    n_channels = real_data.shape[-1]
    ncols = min(4, n_channels)
    nrows = int(np.ceil(n_channels / ncols))

    for name, cond in CONDITIONS.items():
        fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.5 * nrows), squeeze=False)
        fig.suptitle(f'ERP promedio ± desvío estándar por canal — {name} (real vs sintético)')
        for ch in range(n_channels):
            ax = axs[ch // ncols, ch % ncols]
            df = pd.concat([
                _long_df(real_data[real_labels == cond][:, :, ch], 'Real'),
                _long_df(gen_data[gen_labels == cond][:, :, ch], 'Sintético'),
            ], ignore_index=True)
            sns.lineplot(data=df, x='tiempo', y='amplitud', hue='tipo_senal',
                         palette={'Real': COLORS['real'], 'Sintético': COLORS['gen']},
                         errorbar='sd', ax=ax, legend=(ch == 0))
            ax.set_title(f'Canal {ch}', fontsize=8)
            ax.set_xlabel('')
            ax.set_ylabel('')
        if n_channels:
            axs[0, 0].legend(fontsize=7)
        for ch in range(n_channels, nrows * ncols):
            axs[ch // ncols, ch % ncols].axis('off')
        fig.tight_layout()
        path = out_path.format(cond=name)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f'  Guardado: {path}')


def plot_psd_overlay(real_data, real_labels, gen_data, gen_labels, out_path, fs=FS):
    """PSD (Welch) promedio por canal/condición, real vs sintético.
    Devuelve {(condición, canal): jsd} -- divergencia espectral en banda real."""
    n_channels = real_data.shape[-1]
    ncols = min(4, n_channels)
    nrows = int(np.ceil(n_channels / ncols))
    jsd_scores = {}

    for name, cond in CONDITIONS.items():
        fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.5 * nrows), squeeze=False)
        fig.suptitle(f'PSD (Welch) por canal — {name} (real vs sintético)')
        for ch in range(n_channels):
            ax = axs[ch // ncols, ch % ncols]
            means = {}
            for label, data, labels, color, style in [
                ('Real', real_data, real_labels, COLORS['real'], '-'),
                ('Sintético', gen_data, gen_labels, COLORS['gen'], '--'),
            ]:
                trials = data[labels == cond][:, :, ch]
                if trials.shape[0] == 0:
                    continue
                f, mean, std = compute_psd(trials, fs)
                means[label] = mean
                ax.plot(f, mean, color=color, linestyle=style, label=label)
                ax.fill_between(f, mean - std, mean + std, color=color, alpha=0.2)
            if 'Real' in means and 'Sintético' in means:
                jsd_scores[(name, ch)] = spectral_jsd(f, means['Real'], means['Sintético'], fmax=BP_HZ)
            ax.axvline(BP_HZ, color='gray', linewidth=1, linestyle=':', label=f'Corte real ({BP_HZ} Hz)')
            ax.set_xlim(0, PSD_FMAX)
            ax.set_title(f'Canal {ch}', fontsize=8)
            ax.set_xlabel('')
            ax.set_ylabel('')
        if n_channels:
            axs[0, 0].legend(fontsize=7)
        for ch in range(n_channels, nrows * ncols):
            axs[ch // ncols, ch % ncols].axis('off')
        fig.supxlabel('Frequency (Hz)', fontsize=9)
        fig.supylabel('Power Spectral Density (a.u.)', fontsize=9)
        fig.tight_layout()
        path = out_path.format(cond=name)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f'  Guardado: {path}')

    return jsd_scores


def compare_subject(subject, model_path=None, real_csv=None, train_csv=None, gen_csv=None,
                     plot_dir=None, skip_generation=False):
    """Compara un checkpoint con los datos HELD-OUT (test) de un sujeto: genera
    muestras, grafica ERP/PSD/PCA/t-SNE y devuelve un dict de métricas resumen.

    Los paths son opcionales — por defecto usan la convención de moabb_pipeline.py
    (MODEL_PREFIX/GAN_DIR/DATA_DIR/TEST_DATA_DIR/etc). Parametrizables para que
    ablation_pipeline.py pueda reusar esta misma función con checkpoints/carpetas
    de salida distintas.

    real_csv: CSV de evaluación (por defecto, el held-out de TEST_DATA_DIR — el
    generador nunca lo vio en entrenamiento).
    train_csv: CSV de entrenamiento (por defecto, DATA_DIR) — se usa SOLO para
    tomar el min/max con el que normalizar real_csv, así el "real" de test queda
    en la MISMA escala en la que el generador vio los datos de train (si cada CSV
    se normalizara con su propio min/max, la comparación quedaría ligeramente
    desalineada en escala). Si no existe, cae a normalizar con el min/max propio
    de real_csv (con aviso).

    skip_generation=True: no entrena/genera nada, usa el `gen_csv` ya existente tal
    cual (debe pasarse explícitamente). Para evaluar checkpoints de otra rama/repo
    (ej. EEG-GAN vanilla en `main`) cuyo formato esta rama no puede cargar — ver
    eval_external_config.py.
    """
    if real_csv is None:
        real_csv = os.path.join(TEST_DATA_DIR, f'subject_{subject:03d}.csv')
    if train_csv is None:
        train_csv = os.path.join(DATA_DIR, f'subject_{subject:03d}.csv')
    if plot_dir is None:
        plot_dir = PLOT_DIR

    if not os.path.exists(real_csv):
        print(f'  Aviso: no existe {real_csv}, se omite sujeto {subject}.')
        return None

    if skip_generation:
        if gen_csv is None or not os.path.exists(gen_csv):
            print(f'  Aviso: skip_generation=True pero no existe gen_csv={gen_csv}.')
            return None
        os.makedirs(plot_dir, exist_ok=True)
        print(f'  Usando muestras sintéticas ya generadas -> {gen_csv}')
    else:
        if model_path is None:
            model_path = os.path.join(GAN_DIR, f'{MODEL_PREFIX}_s{subject:03d}.pt')
        if gen_csv is None:
            gen_csv = os.path.join(GEN_DIR, f'{MODEL_PREFIX}_s{subject:03d}_synthetic.csv')
        if not os.path.exists(model_path):
            print(f'  Aviso: no existe {model_path}, se omite sujeto {subject}.')
            return None

        os.makedirs(plot_dir, exist_ok=True)

        print(f'  Generando muestras sintéticas -> {gen_csv}')
        generate_synthetic(model_path, real_csv, gen_csv)

    if os.path.exists(train_csv):
        norm_min, norm_max = train_norm_stats(train_csv)
    else:
        print(f'  Aviso: no existe train_csv={train_csv}; normalizando el held-out con su propio '
              f'min/max (puede no coincidir exactamente con la escala del generador).')
        norm_min, norm_max = None, None
    real_data, real_labels = load(real_csv, norm_data=True, norm_min=norm_min, norm_max=norm_max)
    gen_data, gen_labels   = load(gen_csv, norm_data=False)  # ya está en la escala normalizada del generador

    print('  Graficando ERP promedio (real vs sintético)...')
    plot_erp_overlay(real_data, real_labels, gen_data, gen_labels,
                      os.path.join(plot_dir, f's{subject:03d}_erp_{{cond}}.png'))

    print('  Graficando PSD (Welch)...')
    jsd_scores = plot_psd_overlay(real_data, real_labels, gen_data, gen_labels,
                                   os.path.join(plot_dir, f's{subject:03d}_psd_{{cond}}.png'))

    print(f'  Métricas de fidelidad espectral/distribucional (sujeto {subject}):')
    mmd_vals, amp_errs, lat_errs = [], [], []
    for name, cond in CONDITIONS.items():
        cond_jsd = [v for (n, _ch), v in jsd_scores.items() if n == name]
        if cond_jsd:
            print(f'    JSD espectral (0-{BP_HZ}Hz) {name}: media={np.mean(cond_jsd):.4f} '
                  f'(por canal: {[round(v, 3) for v in cond_jsd]})')

        real_c = real_data[real_labels == cond].reshape(-1, real_data.shape[1] * real_data.shape[2])
        gen_c  = gen_data[gen_labels == cond].reshape(-1, gen_data.shape[1] * gen_data.shape[2])
        if len(real_c) >= 4 and len(gen_c) >= 4:
            half = len(real_c) // 2
            baseline = mmd_rbf(real_c[:half], real_c[half:])
            mmd_val = mmd_rbf(real_c, gen_c)
            mmd_vals.append(mmd_val)
            print(f'    MMD² {name}: real-vs-sintético={mmd_val:.6f}  (baseline real/real split={baseline:.6f}, esperado ~0)')

        # Pico ERP (P300) por canal, promediado — ver eeg_metrics.erp_peak_metrics
        real_cond, gen_cond = real_data[real_labels == cond], gen_data[gen_labels == cond]
        if len(real_cond) and len(gen_cond):
            for ch in range(real_cond.shape[-1]):
                amp_err, lat_err = erp_peak_metrics(real_cond[:, :, ch], gen_cond[:, :, ch], fs=FS)
                amp_errs.append(amp_err)
                lat_errs.append(lat_err)

    if amp_errs:
        print(f'    Pico ERP (250-600ms): error amplitud media={np.mean(amp_errs):.4f}, '
              f'error latencia media={np.mean(lat_errs):.1f}ms')

    print('  Graficando PCA...')
    visualization_dim_reduction(real_data, gen_data, 'pca', save=True,
                                 save_name=os.path.join(plot_dir, f's{subject:03d}_pca.png'))

    print('  Graficando t-SNE...')
    visualization_dim_reduction(real_data, gen_data, 'tsne', save=True,
                                 save_name=os.path.join(plot_dir, f's{subject:03d}_tsne.png'))

    return {
        'jsd_mean': float(np.mean(list(jsd_scores.values()))) if jsd_scores else float('nan'),
        'mmd2': float(np.mean(mmd_vals)) if mmd_vals else float('nan'),
        'erp_amp_err': float(np.mean(amp_errs)) if amp_errs else float('nan'),
        'erp_lat_err_ms': float(np.mean(lat_errs)) if lat_errs else float('nan'),
    }


def main():
    parser = argparse.ArgumentParser(description='Compara muestras generadas por la GAN con los datos reales.')
    parser.add_argument('--subjects', type=int, nargs='+', default=[1, 2, 3])
    args = parser.parse_args()

    for subject in args.subjects:
        print(f'\n{"="*55}\n  Sujeto {subject}\n{"="*55}')
        compare_subject(subject)

    print('\nComparación completada. Ver PNGs en', PLOT_DIR)


if __name__ == '__main__':
    main()
