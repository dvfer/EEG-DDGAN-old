"""Pipeline: MOABB P300 (BNCI2014_009 / BNCI2014_008) → CSV multi-canal → GAN (ttsgan-direct)

Reemplaza los bucles bash de entrenamiento. Llama directamente a las
funciones Python de eeggan sin subprocesses.

Nota (rama ttsgan-direct): esta rama entrena la GAN directamente sobre los
datos multicanal en crudo, sin el paso de autoencoder que usaba `main`
(ver openspec/changes/ttsgan-native-multichannel). `patch_size` debe ser
divisor de la longitud de secuencia cruda del dataset (ya no de `time_out`
del AE) — `gan_training_main.py` lanza un `ValueError` claro si no lo es.

Uso:
    python moabb_pipeline.py                        # usa DATASET_NAME de abajo
    python moabb_pipeline.py --dataset BNCI2014_008
    python moabb_pipeline.py --dataset BNCI2014_008 --subjects 1 2

Ajusta las variables de configuración de abajo antes de ejecutar.
"""

import argparse
import os
import numpy as np
import pandas as pd

# ────────────────────────────────────────────────────────────
# CONFIGURACIÓN  (equivalente a las variables del script bash)
# ────────────────────────────────────────────────────────────

# Dataset MOABB a usar: 'BNCI2014_009' (10 sujetos) o 'BNCI2014_008' (8 sujetos)
DATASET_NAME = 'BNCI2014_009'

# Sujetos por defecto por dataset (override con --subjects)
DEFAULT_SUBJECTS = {
    'BNCI2014_009': [1, 2, 3],
    'BNCI2014_008': list(range(1, 9)),  # los 8 sujetos del dataset
}

# Condición a exportar: 'Target', 'NonTarget', o 'Both'
CONDITION = 'Both'

# Normalización z-score por trial antes de exportar (False = sin normalizar)
NORM = False

# Directorios de salida. BNCI2014_009 mantiene las rutas planas originales
# (compare_samples.py/ablation_pipeline.py/train_eeggan_vanilla.sh las tienen
# hardcodeadas así) — otros datasets van a un directorio hermano sufijado para
# no pisar esos CSVs/checkpoints.
_DIR_SUFFIX = {'BNCI2014_009': '', 'BNCI2014_008': '_008'}


def _data_dirs(dataset_name):
    suffix = _DIR_SUFFIX[dataset_name]
    return f'subject_data/train{suffix}', f'subject_data/test{suffix}'


DATA_DIR, TEST_DATA_DIR = _data_dirs(DATASET_NAME)
TEST_SIZE     = 0.2                    # fracción de trials para el held-out set
GAN_DIR  = 'trained_models'

# ── GAN ─────────────────────────────────────────────────────
GAN_PATCH_SIZE = 10
GAN_N_EPOCHS   = 2000
GAN_SEED       = 42
GAN_USE_DWT    = True   # usar MultiscaleDWTDiscriminator
GAN_HIGH_FREQ  = True   # incluir coeficientes de alta frecuencia en DWT
GAN_DWT_J      = 4      # niveles de descomposición DWT (techo ~log2(seq_len))
GAN_LAMBDA_FM  = 50     # peso de la feature-matching loss (50: mejores resultados en ablation)
GAN_USE_POSTNET = True  # smoothing residual de costuras de patch en el generador
GAN_USE_STACKING = False # combinar D1 (TTS) + D2 (DWT) vía StackingDiscriminator

# Preset alternativo: TTS-GAN puro (sin segundo discriminador DWT, sin PostNet,
# sin stacking) -- el 'tts_gan_baseline'/'baseline' del ablation, usado como
# comparación contra el modelo completo ('main', los GAN_USE_* de arriba).
# Seleccionable con --config tts_baseline; no afecta el preset 'main' por defecto.
GAN_CONFIGS = {
    'tts_baseline': dict(use_dwt=False, high_freq=False, dwt_j=GAN_DWT_J, lambda_fm=0,
                          use_postnet=False, use_stacking=False),
}

# Prefijo para los archivos de modelo -- codifica el dataset + los hiperparámetros
# que más solemos variar entre corridas, para no pisar/confundir checkpoints de
# config distinta (mismo motivo que el nombre por target en train_eeggan_vanilla.sh).
def _model_prefix(dataset_name, config_name='main'):
    code = dataset_name.split('_')[-1]  # 'BNCI2014_009' -> '009'
    if config_name != 'main':
        return f'GAN_{code}_{config_name}'
    return f'GAN_{code}_fm{GAN_LAMBDA_FM}_postnet{int(GAN_USE_POSTNET)}_stack{int(GAN_USE_STACKING)}'


MODEL_PREFIX = _model_prefix(DATASET_NAME)

# ────────────────────────────────────────────────────────────


# Mapeo de etiquetas string (P300) → entero
_P300_LABEL_MAP = {'Target': 1, 'NonTarget': 0}


def _to_int_labels(y):
    """Convierte etiquetas string ('Target'/'NonTarget') a int si es necesario."""
    y = np.asarray(y)
    if y.dtype.kind in ('U', 'S', 'O'):            # dtype string/object
        return np.vectorize(_P300_LABEL_MAP.get)(y).astype(int)
    return y.astype(int)


def get_channel_names(dataset, subject):
    """Extrae nombres de canales EEG desde los datos raw MNE del sujeto."""
    try:
        raws = dataset.get_data(subjects=[subject])
        # estructura: {subject: {session: {run: MNE_Raw}}}
        first_subject = next(iter(raws.values()))
        first_session = next(iter(first_subject.values()))
        first_run     = next(iter(first_session.values()))
        picks = first_run.pick('eeg', verbose=False)
        return list(picks.ch_names)
    except Exception as exc:
        print(f"  Aviso: no se pudieron extraer nombres de canales ({exc}). "
              f"Usando Ch0, Ch1, ...")
        return None   # se determina más tarde a partir de X.shape


def export_to_csv(X, y, ch_names, condition='Both', output_path='data.csv',
                  norm=False, patch_size=None):
    """Exporta EEG multi-canal a CSV en formato largo compatible con Dataloader.

    Args:
        X:          numpy array (n_trials, n_channels, n_timepoints)
        y:          array-like (n_trials,) — etiquetas int o str ('Target'/'NonTarget')
        ch_names:   list[str] de longitud n_channels
        condition:  'Target' (solo clase 1), 'NonTarget' (clase 0), o 'Both'
        output_path: ruta de salida del CSV
        norm:       si True, aplica z-score por trial antes de exportar
        patch_size: si se indica, recorta los timepoints finales para que
                    n_timepoints sea múltiplo de patch_size (requerido por la
                    GAN, ver gan_training_main.pad_warning)

    CSV resultante (formato largo):
        ParticipantID | Condition | Trial | Electrode | Time1 | Time2 | ...
        una fila por (trial, canal)

    Uso posterior con Dataloader:
        Dataloader(output_path, kw_channel='Electrode',
                   kw_time='Time', kw_conditions='Condition')
    """
    X = np.asarray(X, dtype=np.float32)
    y = _to_int_labels(y)

    if X.ndim != 3:
        raise ValueError(
            f"X debe ser 3-D (n_trials, n_channels, n_timepoints), "
            f"recibido {X.shape}"
        )

    if patch_size:
        n_keep = (X.shape[2] // patch_size) * patch_size
        if n_keep == 0:
            raise ValueError(
                f"n_timepoints ({X.shape[2]}) < patch_size ({patch_size})"
            )
        if n_keep != X.shape[2]:
            print(f"  Recortando timepoints {X.shape[2]} -> {n_keep} "
                  f"(múltiplo de patch_size={patch_size})")
        X = X[:, :, :n_keep]

    n_trials, n_channels, n_timepoints = X.shape

    if ch_names is None:
        ch_names = [f'Ch{i}' for i in range(n_channels)]
    if len(ch_names) != n_channels:
        raise ValueError(
            f"ch_names tiene {len(ch_names)} entradas pero X tiene {n_channels} canales"
        )

    # Filtrar por condición
    if condition == 'Target':
        mask = y == 1
    elif condition == 'NonTarget':
        mask = y == 0
    else:
        mask = np.ones(n_trials, dtype=bool)

    X_out = X[mask]
    y_out = y[mask]
    n_export = X_out.shape[0]

    if n_export == 0:
        raise ValueError(
            f"Sin datos para exportar con condition='{condition}'. "
            f"Etiquetas únicas presentes: {np.unique(y)}"
        )

    # Normalización z-score por trial (sobre todos los canales y tiempo)
    if norm:
        mean = X_out.mean(axis=(1, 2), keepdims=True)
        std  = X_out.std(axis=(1, 2), keepdims=True) + 1e-8
        X_out = (X_out - mean) / std

    # Shuffle si exportamos ambas condiciones
    if condition == 'Both':
        perm   = np.random.permutation(n_export)
        X_out  = X_out[perm]
        y_out  = y_out[perm]

    # Construir DataFrame en formato largo (una fila por canal por trial)
    # Nota: la columna Electrode guarda el ÍNDICE del canal (entero) en vez del
    # nombre, para que todo el CSV sea numérico y compatible con el Dataloader
    # (df.to_numpy() debe ser float, no object). El mapeo índice→nombre se
    # guarda en un archivo sidecar '<csv>.channels.txt' para remapear después.
    time_cols = [f'Time{t + 1}' for t in range(n_timepoints)]
    rows = []
    for trial_idx in range(n_export):
        for ch_idx in range(n_channels):
            row = {
                'ParticipantID': 1,
                'Condition':     int(y_out[trial_idx]),
                'Trial':         trial_idx,
                'Electrode':     ch_idx,
            }
            row.update(zip(time_cols, X_out[trial_idx, ch_idx].tolist()))
            rows.append(row)

    col_order = ['ParticipantID', 'Condition', 'Trial', 'Electrode'] + time_cols
    df = pd.DataFrame(rows, columns=col_order)
    df.to_csv(output_path, index=False)

    # Sidecar con el mapeo índice→nombre de canal (un nombre por línea)
    channels_path = f'{output_path}.channels.txt'
    with open(channels_path, 'w') as f:
        f.write('\n'.join(ch_names) + '\n')

    print(
        f"  CSV exportado: {output_path}\n"
        f"  Channels map:  {channels_path}\n"
        f"  Trials: {n_export} | Canales: {n_channels} | "
        f"Timepoints: {n_timepoints} | Filas totales: {len(df)}"
    )
    return df


def train_gan(csv_path, gan_save_path, patch_size=10,
              n_epochs=2000, seed=42, use_dwt=True, high_freq=True, dwt_j=4, lambda_fm=20,
              use_postnet=False, use_stacking=False):
    """Entrena la GAN llamando directamente a eeggan (sin autoencoder — ttsgan-direct)."""
    from eeggan.gan_training_main import main as gan_main

    print(f"  Entrenando GAN → {gan_save_path}")
    args = [
        f'data={csv_path}',
        f'save_name={os.path.basename(gan_save_path)}',
        'kw_channel=Electrode',
        'kw_conditions=Condition',
        'kw_time=Time',
        f'patch_size={patch_size}',
        f'n_epochs={n_epochs}',
        f'seed={seed}',
    ]
    if use_dwt:
        args.append('use_multiscale_dwt_discriminator')
    if high_freq:
        args.append('multiscale_dwt_high_freq=True')
    if use_dwt:
        args.append(f'dwt_j={dwt_j}')
        args.append(f'lambda_fm={lambda_fm}')  # solo tiene efecto si hay D2 (DWT)
    if use_postnet:
        args.append('use_postnet')
    if use_stacking:
        args.append('use_stacking')
    gan_main(args)


def main():
    from moabb.datasets import BNCI2014_008, BNCI2014_009
    from moabb.paradigms import P300

    dataset_classes = {'BNCI2014_009': BNCI2014_009, 'BNCI2014_008': BNCI2014_008}

    parser = argparse.ArgumentParser(
        description="Pipeline MOABB P300 -> CSV -> GAN (ttsgan-direct, sin autoencoder)"
    )
    parser.add_argument(
        '--dataset', choices=list(dataset_classes), default=DATASET_NAME,
        help=f'Dataset MOABB a usar (default: {DATASET_NAME})'
    )
    parser.add_argument(
        '--subjects', type=int, nargs='+', default=None,
        help='IDs de sujetos a procesar (default: DEFAULT_SUBJECTS[dataset] del archivo). Ej: --subjects 1'
    )
    parser.add_argument(
        '--config', choices=['main'] + list(GAN_CONFIGS), default='main',
        help="Preset de hiperparámetros de GAN: 'main' (DWT+FM+PostNet, default) "
             "o 'tts_baseline' (TTS-GAN puro, sin DWT/PostNet/stacking)"
    )
    args = parser.parse_args()
    if args.subjects is None:
        args.subjects = DEFAULT_SUBJECTS[args.dataset]

    dataset  = dataset_classes[args.dataset]()
    paradigm = P300()

    # Recalcular rutas/prefijo si --dataset/--config difieren de los defaults del archivo
    data_dir, test_data_dir = _data_dirs(args.dataset)
    model_prefix  = _model_prefix(args.dataset, args.config)
    gan_kwargs = (
        dict(use_dwt=GAN_USE_DWT, high_freq=GAN_HIGH_FREQ, dwt_j=GAN_DWT_J,
             lambda_fm=GAN_LAMBDA_FM, use_postnet=GAN_USE_POSTNET, use_stacking=GAN_USE_STACKING)
        if args.config == 'main' else GAN_CONFIGS[args.config]
    )

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(test_data_dir, exist_ok=True)
    os.makedirs(GAN_DIR,  exist_ok=True)

    for subject in args.subjects:
        print(f'\n{"="*55}')
        print(f'  Sujeto {subject}')
        print(f'{"="*55}')

        # Rutas de archivos
        csv_path      = os.path.join(data_dir, f'subject_{subject:03d}.csv')       # 80% train
        test_csv_path = os.path.join(test_data_dir, f'subject_{subject:03d}.csv')  # 20% held-out
        gan_save_path = os.path.join(GAN_DIR,  f'{model_prefix}_s{subject:03d}.pt')

        # 1. Exportar CSVs train/test (se omite si ambos ya existen)
        if os.path.exists(csv_path) and os.path.exists(test_csv_path):
            print(f'  CSVs ya existen (train+test), se omite la exportación: {csv_path}')
        else:
            from sklearn.model_selection import train_test_split

            print('  Cargando datos MOABB...')
            X, y, _ = paradigm.get_data(dataset=dataset, subjects=[subject])
            ch_names = get_channel_names(dataset, subject)

            n_ch = X.shape[1]
            if ch_names is None:
                ch_names = [f'Ch{i}' for i in range(n_ch)]

            print(f'  Shape: {X.shape} | Canales: {ch_names[:5]}{"..." if n_ch > 5 else ""}')

            # Split 80/20 estratificado por condición, ANTES de exportar — el
            # generador nunca debe entrenar con trials del held-out de test.
            idx = np.arange(X.shape[0])
            idx_train, idx_test = train_test_split(
                idx, test_size=TEST_SIZE, stratify=_to_int_labels(y), random_state=GAN_SEED
            )
            print(f'  Split train/test: {len(idx_train)}/{len(idx_test)} trials '
                  f'({1 - TEST_SIZE:.0%}/{TEST_SIZE:.0%}, estratificado, seed={GAN_SEED})')

            print('  Exportando CSV de train...')
            export_to_csv(
                X[idx_train], y[idx_train], ch_names,
                condition=CONDITION,
                output_path=csv_path,
                norm=NORM,
                patch_size=GAN_PATCH_SIZE,
            )
            print('  Exportando CSV de test (held-out)...')
            export_to_csv(
                X[idx_test], y[idx_test], ch_names,
                condition=CONDITION,
                output_path=test_csv_path,
                norm=NORM,
                patch_size=GAN_PATCH_SIZE,
            )

        # 2. Entrenar GAN directamente sobre los datos crudos multicanal (sin AE)
        train_gan(
            csv_path, gan_save_path,
            patch_size=GAN_PATCH_SIZE,
            n_epochs=GAN_N_EPOCHS,
            seed=GAN_SEED,
            **gan_kwargs,
        )

    print('\nPipeline completado.')


if __name__ == '__main__':
    main()
