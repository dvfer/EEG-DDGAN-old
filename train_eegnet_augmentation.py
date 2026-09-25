"""Entrena EEGNet (P300 target/non-target) con aumento de datos sintéticos de
los 3 modelos comparados en ablation_results/metrics_*.csv -- TTS-GAN baseline,
EEG-GAN vanilla (full, AE-coupled, rama `main`) y DDGAN (esta rama, el mejor) --
en ratios 0.1, 0.2, 0.3, 0.4, 0.5, más un baseline sin aumento (ratio 0.0,
compartido entre los 3 -- es el mismo dato real, entrenarlo 3 veces sería
recomputar lo mismo).

Escala: el generador entrena sobre datos normalizados [0,1] con el min/max del
CSV de train (dataloader.py). Para que cada pool sintético y los reales
(train/test) queden en la MISMA escala, reusa la normalización de
compare_samples.py: train_norm_stats()+load() (evita repetir ese bug ya
arreglado ahí -- ver S1478 en memoria del proyecto).

Cada (config, sujeto, ratio) se corre N_REPEATS veces (seed distinta cada vez
-- init del modelo + shuffling + qué trials sintéticos se samplean del pool) y
se reporta media ± std sobre esas repeticiones: mide qué tan CONSISTENTE es el
resultado de entrenar con esa config, no la variabilidad entre sujetos (esa ya
se ve directo comparando filas de sujetos distintos, sin necesidad de repetir
nada -- ver plot_eegnet_per_subject.py).

ratio r: n_aug = round(r * n_trials_TARGET_train) -- solo se aumenta la clase
Target (minoritaria en P300; NonTarget no se toca), tomados de un pool
sintético por sujeto (con reemplazo si el pool no alcanza):
  - tts_gan_baseline / GAN_009_fm50_postnet1_stack0 (DDGAN): esta misma rama
    (ttsgan-direct) -- el pool se genera acá mismo, in-process, si no existe.
  - eeg_gan_vanilla_full: checkpoint AE-coupled de la rama `main` -- esta rama
    NO puede cargarlo (generate_samples_main lo rechaza a propósito), así que
    el pool se genera invocando ./generate_eeggan_vanilla_augpool.sh como
    subproceso (worktree de `main`, otro venv) si todavía no existe.

Uso:
    uv run python train_eegnet_augmentation.py            # sujetos 1..10
    uv run python train_eegnet_augmentation.py 4 5 6       # sujetos específicos
"""
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from compare_samples import load, train_norm_stats, generate_synthetic, DATA_DIR, TEST_DATA_DIR, GAN_DIR, GEN_DIR
from moabb_pipeline import MODEL_PREFIX

RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5]
N_EPOCHS = 60
BATCH_SIZE = 32
SEED = 42
N_REPEATS = 100  # repeticiones por (config, sujeto, ratio) -- mide consistencia, no varianza entre sujetos
RESULTS_CSV = 'ablation_results/eegnet_augmentation.csv'

# name -> checkpoint (None si el pool se genera aparte, fuera de esta rama)
CONFIGS = {
    'tts_gan_baseline': os.path.join(GAN_DIR, 'GAN_009_tts_gan_baseline_s{:03d}.pt'),
    'GAN_009_fm50_postnet1_stack0': os.path.join(GAN_DIR, f'{MODEL_PREFIX}_s{{:03d}}.pt'),
    'eeg_gan_vanilla_full': None,  # pool generado por generate_eeggan_vanilla_augpool.sh
}

# Pool vanilla: nombre legacy (BNCI2014_009, sin código de dataset) + dataset a
# pasarle al script. Pisables a nivel módulo para correr otro dataset -- mismo
# mecanismo que CONFIGS (ver eeg_net_aug/train_eegnet_augmentation_v2.py).
VANILLA_POOL_TEMPLATE = 'EEG_GAN_vanilla_full_s{:03d}_augpool.csv'
VANILLA_DATASET = 'BNCI2014_009'
# Tamaño del pool sintético. None = el de siempre (trials reales, capeado a
# compare_samples.MAX_SAMPLES_PER_COND=200, ambas condiciones). Un int fija N
# muestras SOLO de Target -- que es lo único que sample_pool() consume -- y
# escribe a un archivo con sufijo propio para no pisar los pools viejos.
POOL_N_TARGET = None


class EEGNet(nn.Module):
    """Lawhern et al. 2018, versión compacta. Input: (batch, 1, n_channels, n_times)."""

    def __init__(self, n_channels, n_times, n_classes=2, F1=8, D=2, F2=16, dropout=0.5):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            feat_dim = self.block2(self.block1(torch.zeros(1, 1, n_channels, n_times))).flatten(1).shape[1]
        self.classify = nn.Linear(feat_dim, n_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return self.classify(x.flatten(1))


def _selfcheck():
    """Chequeo mínimo: el forward corre y da la forma de salida esperada."""
    m = EEGNet(n_channels=8, n_times=100)
    out = m(torch.zeros(4, 1, 8, 100))
    assert out.shape == (4, 2), out.shape


def to_eegnet_input(X):
    """(trial, seq, channel) -> (trial, 1, channel, seq)."""
    return torch.tensor(X, dtype=torch.float32).permute(0, 2, 1).unsqueeze(1)


def sample_pool(X_pool, y_pool, n, rng, target_class=1):
    """Samplea SOLO de la clase target -- el aumento busca compensar el
    desbalance P300 (Target << NonTarget), no inflar ambas clases por igual."""
    pool_idx = np.flatnonzero(y_pool == target_class)
    replace = n > len(pool_idx)
    idx = rng.choice(pool_idx, size=n, replace=replace)
    return X_pool[idx], y_pool[idx]


def train_and_eval(X_train, y_train, X_test, y_test, device, seed):
    n_channels, n_times = X_train.shape[-1], X_train.shape[1]
    torch.manual_seed(seed)
    model = EEGNet(n_channels, n_times).to(device)

    x_tr = to_eegnet_input(X_train).to(device)
    y_tr = torch.tensor(y_train, dtype=torch.long).to(device)
    x_te = to_eegnet_input(X_test).to(device)

    class_counts = torch.bincount(y_tr, minlength=2).float()
    weight = (class_counts.sum() / (2 * class_counts.clamp(min=1))).to(device)  # P300: clases desbalanceadas
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    model.train()
    perm_rng = np.random.default_rng(seed)
    for _ in range(N_EPOCHS):
        idx = perm_rng.permutation(len(y_tr))
        for start in range(0, len(idx), BATCH_SIZE):
            batch = idx[start:start + BATCH_SIZE]
            optimizer.zero_grad()
            loss = criterion(model(x_tr[batch]), y_tr[batch])
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(x_te), dim=1)[:, 1].cpu().numpy()
    preds = (probs >= 0.5).astype(int)
    return {
        'accuracy': accuracy_score(y_test, preds),
        'f1': f1_score(y_test, preds, zero_division=0),  # binario, clase Target -- la métrica de interés acá
        'f1_macro': f1_score(y_test, preds, average='macro', zero_division=0),  # (F1_Target+F1_NonTarget)/2, referencia
        'auc': roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else float('nan'),
    }


def repeat_train_eval(X_train, y_train, X_test, y_test, device, X_pool=None, y_pool=None, n_aug=0):
    """Corre train_and_eval() N_REPEATS veces (seed = SEED+r cada vez, tanto
    para el sampleo del pool como para el entrenamiento) y agrega media/std."""
    runs = []
    for r in range(N_REPEATS):
        seed = SEED + r
        if n_aug:
            rng = np.random.default_rng(seed)
            X_aug, y_aug = sample_pool(X_pool, y_pool, n_aug, rng)
            X_tr = np.concatenate([X_train, X_aug])
            y_tr = np.concatenate([y_train, y_aug])
        else:
            X_tr, y_tr = X_train, y_train
        runs.append(train_and_eval(X_tr, y_tr, X_test, y_test, device, seed=seed))

    runs_df = pd.DataFrame(runs)
    agg = {}
    for col in runs_df.columns:
        agg[f'{col}_mean'] = runs_df[col].mean()
        agg[f'{col}_std'] = runs_df[col].std()
    agg['n_repeats'] = N_REPEATS
    return agg


def get_pool(config_name, subject, train_csv):
    """Devuelve (X_pool, y_pool) del config dado, generando el pool si no existe:
    in-process si el checkpoint vive en esta rama (ttsgan-direct), o vía
    subprocess a generate_eeggan_vanilla_augpool.sh si no (eeg_gan_vanilla_full,
    que necesita el worktree/venv de `main`)."""
    ckpt_template = CONFIGS[config_name]
    if ckpt_template is not None:
        model_path = ckpt_template.format(subject)
        sfx = '' if POOL_N_TARGET is None else f'_p{POOL_N_TARGET}'
        pool_csv = os.path.join(GEN_DIR, f'{os.path.splitext(os.path.basename(model_path))[0]}_augpool{sfx}.csv')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Falta checkpoint {model_path}')
        if not os.path.exists(pool_csv):
            print(f'    Generando pool sintético ({config_name}) -> {pool_csv}')
            n_per_cond = None if POOL_N_TARGET is None else {'Target': POOL_N_TARGET, 'NonTarget': 0}
            generate_synthetic(model_path, train_csv, pool_csv, n_per_cond)
        else:
            print(f'    Pool sintético ya existe, se reusa: {pool_csv}')
    else:
        pool_csv = os.path.join(GEN_DIR, VANILLA_POOL_TEMPLATE.format(subject))
        if not os.path.exists(pool_csv):
            print(f'    Generando pool sintético ({config_name}) -> {pool_csv} (subproceso, otro venv)')
            script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'generate_eeggan_vanilla_augpool.sh')
            # falla clara si falta el worktree/checkpoint
            subprocess.run([script, str(subject), 'full', VANILLA_DATASET, pool_csv,
                            str(POOL_N_TARGET or 0)], check=True)
        else:
            print(f'    Pool sintético ya existe, se reusa: {pool_csv}')
    return load(pool_csv, norm_data=False)  # ya en la escala [0,1] del generador que lo produjo


def run_subject(subject, device):
    train_csv = os.path.join(DATA_DIR, f'subject_{subject:03d}.csv')
    test_csv = os.path.join(TEST_DATA_DIR, f'subject_{subject:03d}.csv')
    for path in (train_csv, test_csv):
        if not os.path.exists(path):
            raise FileNotFoundError(f'Falta {path}')

    norm_min, norm_max = train_norm_stats(train_csv)
    X_train, y_train = load(train_csv, norm_data=True, norm_min=norm_min, norm_max=norm_max)
    X_test, y_test = load(test_csv, norm_data=True, norm_min=norm_min, norm_max=norm_max)

    rows = []
    print(f'  ratio=0.0 (baseline, sin aumento -- compartido entre configs, {N_REPEATS} corridas)')
    metrics = repeat_train_eval(X_train, y_train, X_test, y_test, device)
    print(f'    -> {metrics}')
    rows.append({'config': 'none', 'subject': subject, 'ratio': 0.0, 'n_aug': 0, **metrics})

    n_train_target = int((y_train == 1).sum())  # el ratio de aumento es sobre la clase Target, no sobre el total

    for config_name in CONFIGS:
        print(f'  config={config_name}')
        try:
            X_pool, y_pool = get_pool(config_name, subject, train_csv)
        except Exception as exc:
            print(f'    Aviso: se omite ({exc}).')
            continue
        for ratio in RATIOS:
            n_aug = round(ratio * n_train_target)
            metrics = repeat_train_eval(X_train, y_train, X_test, y_test, device, X_pool, y_pool, n_aug)
            print(f'    ratio={ratio}: n_aug={n_aug} ({N_REPEATS} corridas) -> {metrics}')
            rows.append({'config': config_name, 'subject': subject, 'ratio': ratio, 'n_aug': n_aug, **metrics})
    return rows


if __name__ == '__main__':
    _selfcheck()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    subjects = [int(s) for s in sys.argv[1:]] or list(range(1, 11))
    os.makedirs('ablation_results', exist_ok=True)

    n_runs = len(subjects) * N_REPEATS * (1 + len(CONFIGS) * len(RATIOS))
    print(f'Total de entrenamientos planeados: ~{n_runs} '
          f'({len(subjects)} sujetos x {N_REPEATS} repeticiones x '
          f'(1 baseline + {len(CONFIGS)} configs x {len(RATIOS)} ratios)) -- '
          f'esto tarda, correlo detached (screen).')

    rows = []
    if os.path.exists(RESULTS_CSV):
        rows = pd.read_csv(RESULTS_CSV).to_dict('records')
        rows = [r for r in rows if int(r['subject']) not in subjects]  # se recalculan

    for subject in subjects:
        print(f'\n{"="*55}\n  Sujeto {subject}\n{"="*55}')
        try:
            rows.extend(run_subject(subject, device))
        except Exception as exc:
            print(f'  Aviso: sujeto {subject} omitido ({exc}).')
            continue
        pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)  # progreso incremental

    if not rows:
        print('\nNingún sujeto produjo métricas.')
        sys.exit(1)

    df = pd.DataFrame(rows).sort_values(['config', 'subject', 'ratio'])
    df.to_csv(RESULTS_CSV, index=False)
    print(f'\nTabla -> {RESULTS_CSV}')
    print(df.to_string(index=False))
