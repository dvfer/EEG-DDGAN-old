"""EEGNet + aumento sintético, v2: almacenamiento por-corrida en vez de un
único CSV mutable (ver openspec/changes/eegnet-augmentation-v2/).

Motivación: train_eegnet_augmentation.py (raíz del repo) guarda todo en
ablation_results/eegnet_augmentation.csv, reescrito incrementalmente. Cuando
el script cambió de esquema (de 1 corrida a N_REPEATS=100 agregado), las
filas de sujetos corridos con cada versión quedaron mezcladas en la misma
tabla sin chequeo de compatibilidad -> columnas NaN silenciosas
(comparison_plots/eegnet_augmentation.csv). Este script es nuevo, no toca
ni reemplaza al viejo, y evita esa clase de bug: cada corrida (config,
sujeto, ratio, seed) es su propio archivo inmutable; el resumen
(media/std) se recalcula siempre desde los archivos presentes en disco.

Reusa (no reimplementa) lo ya probado en train_eegnet_augmentation.py: el
modelo EEGNet, sample_pool(), get_pool() (generación de pools sintéticos,
incluido el mecanismo por subprocess para eeg_gan_vanilla_full), y de
compare_samples.py: load()/train_norm_stats() (misma normalización que ve
el generador).

Configs comparadas (3 con aumento + baseline sin aumento):
  - fm_lambda_50_postnet: checkpoint de ablation_pipeline.py
    (trained_models/ABLATION_<name>_s<subj>.pt).
  - tts_gan_baseline: trained_models/GAN_009_tts_gan_baseline_s<subj>.pt.
  - eeg_gan_vanilla_full: worktree `main` (EEG-DDGAN-main-vanilla/), pool
    generado vía generate_eeggan_vanilla_augpool.sh.
  - none: EEGNet sin ningún aumento (solo datos reales de train).

  (fm_lambda_0 se sacó del barrido: su checkpoint solo existe para sujeto 1
  -- ABLATION_SUBJECT hardcodeado en ablation_pipeline.py -- así que nunca
  pudo compararse contra el resto de la cohorte. Los datos ya calculados de
  sujeto 1 quedan en eeg_net_aug/results/fm_lambda_0/, no se borran.)

50 repeticiones por (config, sujeto, ratio) con seeds LITERALES 1..50 (no
SEED+r) -- menos que las 100 del script viejo, por tiempo de cómputo.

Uso (correr desde la RAÍZ del repo, como el resto de los scripts -- los
paths de datos/checkpoints son relativos a ahí):
    uv run python eeg_net_aug/train_eegnet_augmentation_v2.py        # sujetos 1..10
    uv run python eeg_net_aug/train_eegnet_augmentation_v2.py 4 5 6  # sujetos específicos
    uv run python eeg_net_aug/train_eegnet_augmentation_v2.py --dataset BNCI2014_008
"""
import argparse
import glob
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from compare_samples import DATA_DIR, GAN_DIR, TEST_DATA_DIR, load, train_norm_stats  # noqa: E402
import moabb_pipeline as _mp  # noqa: E402 -- convención de rutas/nombres por dataset
import train_eegnet_augmentation as _tea  # noqa: E402 -- EEGNet, sample_pool, get_pool, N_EPOCHS/BATCH_SIZE

DATASET = 'BNCI2014_009'  # pisado por --dataset (ver use_dataset())
DEFAULT_EPOCHS = _tea.N_EPOCHS  # 60 -- lo ya corrido; otro valor va a results..._ep<N>
DEFAULT_POOL = None  # None = pool de siempre (200/cond); un int va a results..._pool<N>
DEFAULT_MODEL = 'eegnet'  # lo ya corrido; otro clasificador va a results..._<modelo>

# Receta de Adam por modelo. eegnet: la que se corrió siempre. conformer: la
# del código oficial de Song et al. 2023 (lr 2e-4, betas (0.5, 0.999), sin
# weight decay) -- no la de EEGNet, que con el Conformer no está calibrada.
# --lr pisa el lr; un lr distinto al default DEL MODELO va a results..._lr<LR>.
OPTIM = {
    'eegnet':    {'lr': 1e-3, 'betas': (0.9, 0.999), 'weight_decay': 1e-4},
    'conformer': {'lr': 2e-4, 'betas': (0.5, 0.999), 'weight_decay': 0.0},
}
DEFAULT_LR = OPTIM[DEFAULT_MODEL]['lr']
LR = DEFAULT_LR    # pisado por --lr / por el default del modelo (ver use_dataset())
MODEL = DEFAULT_MODEL  # pisado por --model (ver use_dataset())
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
DEFAULT_SUBJECTS = {'BNCI2014_009': list(range(1, 11)), 'BNCI2014_008': list(range(1, 9))}
RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5]
SEEDS = list(range(1, 51))  # literales 1..50 -- a diferencia de SEED+r del script viejo

# Set de configs de ESTE experimento -- distinto al de train_eegnet_augmentation.py.
# Pisa el CONFIGS del módulo importado: get_pool() de ese módulo lee CONFIGS a
# nivel de módulo, así reusamos su lógica de generación de pools (incluido el
# subprocess de eeg_gan_vanilla_full) sin reimplementarla.
_tea.CONFIGS = {
    # fm_lambda_0 sacado del barrido: checkpoint solo existe para sujeto 1
    # (ABLATION_SUBJECT hardcodeado en ablation_pipeline.py), nunca comparable
    # contra el resto de la cohorte. Datos ya calculados quedan en
    # eeg_net_aug/results/fm_lambda_0/ (no se borran, solo no se recalculan).
    'fm_lambda_50_postnet': os.path.join(GAN_DIR, 'ABLATION_fm_lambda_50_postnet_s{:03d}.pt'),
    'tts_gan_baseline': os.path.join(GAN_DIR, 'GAN_009_tts_gan_baseline_s{:03d}.pt'),
    'eeg_gan_vanilla_full': None,  # pool generado aparte (worktree `main`), ver get_pool() de _tea
}
# Control SIN GAN: aumenta con trials Target REALES del propio train (re-muestreo).
# Aporta el mismo n_aug y el mismo corrimiento de balance de clases que los
# configs con GAN, pero cero información nueva -- separa "el GAN genera P300
# útil" de "la clase minoritaria necesitaba más peso", que la curva de ratios
# por sí sola no distingue.
REAL_OVERSAMPLE = 'real_oversample'
CONFIGS_WITH_AUG = list(_tea.CONFIGS) + [REAL_OVERSAMPLE]

# GAN_009_fm50_postnet1_stack0 (moabb_pipeline.py) es la MISMA arquitectura que
# fm_lambda_50_postnet (use_dwt=True, lambda_fm=50, use_postnet=True,
# use_stacking=False, mismo patch_size/seed) pero entrenada a presupuesto
# completo (GAN_N_EPOCHS=2000) para sujetos 2-10, mientras que el checkpoint
# ABLATION_* (solo sujeto 1) se entrenó a ABLATION_N_EPOCHS=200 ("bajo a
# propósito para iterar rápido"). Preferimos el checkpoint completo cuando
# existe -- OJO: esto deja la fila de sujeto 1 en desventaja de presupuesto de
# entrenamiento (200 vs. 2000 épocas) frente a sujetos 2-10 para este config;
# no hay forma de evitarlo sin re-entrenar GAN_009_fm50_postnet1_stack0_s001.
_CHECKPOINT_ALIASES = {
    'fm_lambda_50_postnet': [
        os.path.join(GAN_DIR, 'GAN_009_fm50_postnet1_stack0_s{:03d}.pt'),
        os.path.join(GAN_DIR, 'ABLATION_fm_lambda_50_postnet_s{:03d}.pt'),
    ],
}


def build_model(n_channels, n_times):
    """Clasificador según MODEL, y el tensor de entrada que espera.

    'eegnet' es la reimplementación de Lawhern et al. 2018 que vive en
    train_eegnet_augmentation.py y con la que se corrió todo hasta ahora --
    NO se cambia por braindecode.EEGNetv4: los 50 seeds x sujetos x ratios ya
    calculados dejarían de ser comparables. 'conformer' sí sale de
    braindecode (Song et al. 2023) y toma (batch, canal, tiempo), sin el
    unsqueeze(1) que mete to_eegnet_input()."""
    if MODEL == 'eegnet':
        return _tea.EEGNet(n_channels, n_times), _tea.to_eegnet_input
    if MODEL == 'conformer':
        from braindecode.models import EEGConformer  # import perezoso: eegnet no lo necesita
        to_input = lambda X: torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
        return EEGConformer(n_outputs=2, n_chans=n_channels, n_times=n_times), to_input
    raise ValueError(f'modelo desconocido: {MODEL}')


def use_dataset(dataset_name, n_epochs=None, lr=None, pool_n_target=None, model=None):
    """Repunta datos, checkpoints y resultados al dataset dado.

    n_epochs: épocas de EEGNet (default DEFAULT_EPOCHS=60). Si difiere del
    default, los resultados van a un árbol aparte (results..._ep<N>): corridas
    con presupuesto de entrenamiento distinto NO son comparables y no deben
    caer en el mismo summary -- es la misma clase de bug que motivó este
    script (ver docstring del módulo).

    BNCI2014_009 no se toca: mantiene rutas planas (subject_data/train),
    los nombres de checkpoint de siempre y eeg_net_aug/results/, así lo ya
    calculado (50 seeds x 10 sujetos) sigue siendo válido y no se recalcula.
    Otro dataset usa el sufijo de moabb_pipeline (_DIR_SUFFIX) para datos y
    resultados, y deriva los nombres de checkpoint de _model_prefix() -- que es
    como moabb_pipeline.py los guarda: GAN_008_fm50_postnet1_stack0_sXXX.pt y
    GAN_008_tts_baseline_sXXX.pt (el 'tts_gan_baseline' de 009 vino de otro
    script, por eso el nombre distinto; la CLAVE del config se mantiene para
    que los scripts de análisis/plots sigan funcionando sin cambios)."""
    global DATASET, DATA_DIR, TEST_DATA_DIR, RESULTS_DIR, CONFIGS_WITH_AUG, LR, MODEL
    DATASET = dataset_name
    if n_epochs is not None:
        _tea.N_EPOCHS = n_epochs
    if model is not None:
        MODEL = model
    # el modelo primero: sin --lr, el lr es el default DEL MODELO (OPTIM)
    LR = lr if lr is not None else OPTIM[MODEL]['lr']
    if pool_n_target is not None:
        # pool grande y SOLO Target: sample_pool() nunca toca NonTarget, y un
        # pool chico hace que todos los seeds sampleen casi las mismas muestras
        # (a ratio 0.3 con pool=200 dos seeds comparten el 84%), lo que
        # subestima la varianza entre seeds.
        _tea.POOL_N_TARGET = pool_n_target

    suffix = _mp._DIR_SUFFIX[dataset_name]
    ep_suffix = '' if _tea.N_EPOCHS == DEFAULT_EPOCHS else f'_ep{_tea.N_EPOCHS}'
    lr_suffix = '' if LR == OPTIM[MODEL]['lr'] else f'_lr{LR:g}'
    pool_suffix = '' if _tea.POOL_N_TARGET is None else f'_pool{_tea.POOL_N_TARGET}'
    model_suffix = '' if MODEL == DEFAULT_MODEL else f'_{MODEL}'
    RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               f'results{suffix}{ep_suffix}{lr_suffix}{pool_suffix}{model_suffix}')
    if dataset_name == 'BNCI2014_009':
        return

    code = dataset_name.split('_')[-1]
    DATA_DIR, TEST_DATA_DIR = _mp._data_dirs(dataset_name)

    _tea.CONFIGS = {
        'fm_lambda_50_postnet': os.path.join(GAN_DIR, f'{_mp._model_prefix(dataset_name)}_s{{:03d}}.pt'),
        'tts_gan_baseline': os.path.join(
            GAN_DIR, f'{_mp._model_prefix(dataset_name, "tts_baseline")}_s{{:03d}}.pt'),
        'eeg_gan_vanilla_full': None,
    }
    CONFIGS_WITH_AUG = list(_tea.CONFIGS) + [REAL_OVERSAMPLE]
    _CHECKPOINT_ALIASES.clear()  # los alias ABLATION_* son artefactos solo de 009
    _tea.VANILLA_POOL_TEMPLATE = f'EEG_GAN_vanilla_{code}_full_s{{:03d}}_augpool.csv'
    _tea.VANILLA_DATASET = dataset_name


def _resolve_checkpoint_template(config_name, subject):
    """Primer template de _CHECKPOINT_ALIASES cuyo .pt exista para `subject`;
    si ninguno existe, el primero (para que el mensaje de error de
    _tea.get_pool() señale el checkpoint "principal"). Configs sin alias
    (todas menos fm_lambda_50_postnet) devuelven su único template de
    siempre, sin cambios."""
    candidates = _CHECKPOINT_ALIASES.get(config_name, [_tea.CONFIGS[config_name]])
    for template in candidates:
        if os.path.exists(template.format(subject)):
            return template
    return candidates[0]


def get_pool(config_name, subject, train_csv):
    """Como _tea.get_pool(), pero devuelve None en vez de lanzar si falta el
    checkpoint/pool -- así el caller escribe el sentinel MISSING_CHECKPOINT
    en vez de abortar el sujeto entero."""
    if _tea.CONFIGS[config_name] is not None:
        _tea.CONFIGS[config_name] = _resolve_checkpoint_template(config_name, subject)
    try:
        return _tea.get_pool(config_name, subject, train_csv)
    except Exception as exc:
        print(f'    Aviso: no se pudo obtener el pool de {config_name} (sujeto {subject}): {exc}')
        return None


def run_single(X_train, y_train, X_test, y_test, device, seed):
    """Misma receta que _tea.train_and_eval(), pero además devuelve el
    classification_report -- esa función no expone las predicciones, y no se
    puede extender sin tocar train_eegnet_augmentation.py (fuera de alcance
    de este cambio), así que se duplica el loop de entrenamiento acá."""
    n_channels, n_times = X_train.shape[-1], X_train.shape[1]
    torch.manual_seed(seed)
    net, to_input = build_model(n_channels, n_times)
    model = net.to(device)

    x_tr = to_input(X_train).to(device)
    y_tr = torch.tensor(y_train, dtype=torch.long).to(device)
    x_te = to_input(X_test).to(device)

    class_counts = torch.bincount(y_tr, minlength=2).float()
    weight = (class_counts.sum() / (2 * class_counts.clamp(min=1))).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=OPTIM[MODEL]['betas'],
                                 weight_decay=OPTIM[MODEL]['weight_decay'])

    model.train()
    perm_rng = np.random.default_rng(seed)
    for _ in range(_tea.N_EPOCHS):
        idx = perm_rng.permutation(len(y_tr))
        for start in range(0, len(idx), _tea.BATCH_SIZE):
            batch = idx[start:start + _tea.BATCH_SIZE]
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
        'f1': f1_score(y_test, preds, zero_division=0),
        'f1_macro': f1_score(y_test, preds, average='macro', zero_division=0),
        'auc': roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else float('nan'),
        'classification_report': classification_report(y_test, preds, output_dict=True, zero_division=0),
    }


def _subject_dir(config, subject, results_dir=None):
    # RESULTS_DIR se resuelve acá (no como default de la firma) porque
    # use_dataset() puede repuntarlo después de importar el módulo.
    return os.path.join(results_dir or RESULTS_DIR, config, f'subject_{subject:03d}')


def run_path(config, subject, ratio, seed, results_dir=None):
    return os.path.join(_subject_dir(config, subject, results_dir), f'ratio_{float(ratio)}_seed_{seed}.json')


def missing_sentinel_path(config, subject, results_dir=None):
    return os.path.join(_subject_dir(config, subject, results_dir), 'MISSING_CHECKPOINT')


def mark_missing_checkpoint(config, subject, results_dir=None):
    path = missing_sentinel_path(config, subject, results_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, 'a').close()


def has_missing_checkpoint(config, subject, results_dir=None):
    return os.path.exists(missing_sentinel_path(config, subject, results_dir))


def clear_missing_checkpoint(config, subject, results_dir=None):
    """Borra el sentinel si quedó de una corrida anterior donde faltaba el
    checkpoint -- necesario para que un checkpoint que aparece después (o un
    resolver que ahora sabe encontrarlo, como el alias de
    fm_lambda_50_postnet) no deje corridas válidas enterradas bajo un resumen
    'missing_checkpoint' desactualizado."""
    path = missing_sentinel_path(config, subject, results_dir)
    if os.path.exists(path):
        os.remove(path)


def ensure_run(config, subject, ratio, seed, X_train, y_train, X_test, y_test, device,
                X_pool=None, y_pool=None, n_aug=0, results_dir=None):
    """Corre (config, sujeto, ratio, seed) si no existe ya su archivo --
    resume gratis: relanzar el script no reentrena lo que ya está en disco."""
    path = run_path(config, subject, ratio, seed, results_dir)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)

    if n_aug:
        rng = np.random.default_rng(seed)
        X_aug, y_aug = _tea.sample_pool(X_pool, y_pool, n_aug, rng)
        X_tr, y_tr = np.concatenate([X_train, X_aug]), np.concatenate([y_train, y_aug])
    else:
        X_tr, y_tr = X_train, y_train

    result = run_single(X_tr, y_tr, X_test, y_test, device, seed)
    result.update({'config': config, 'subject': subject, 'ratio': ratio, 'seed': seed, 'n_aug': n_aug,
                   'n_epochs': _tea.N_EPOCHS, 'lr': LR, 'pool_n_target': _tea.POOL_N_TARGET,
                   'model': MODEL})  # presupuesto/lr/modelo registrados: corridas distintas no se mezclan

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(result, f)
    print(f'    seed={seed} ratio={ratio}: acc={result["accuracy"]:.4f} f1={result["f1"]:.4f}')
    return result


def summarize_subject(config, subject, results_dir=None):
    """Recalcula el resumen de (config, sujeto) DESDE CERO a partir de los
    .json presentes en disco -- nunca lee ni mergea un summary previo (así
    no puede repetir el bug de esquemas mezclados del script viejo)."""
    summary_path = os.path.join(results_dir or RESULTS_DIR, config, f'subject_{subject:03d}_summary.csv')
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    if has_missing_checkpoint(config, subject, results_dir):
        pd.DataFrame([{'config': config, 'subject': subject, 'status': 'missing_checkpoint', 'ratio': np.nan,
                        'n_runs': 0, 'accuracy_mean': np.nan, 'accuracy_std': np.nan, 'f1_mean': np.nan,
                        'f1_std': np.nan, 'f1_macro_mean': np.nan, 'f1_macro_std': np.nan,
                        'auc_mean': np.nan, 'auc_std': np.nan}]).to_csv(summary_path, index=False)
        print(f'    {config} sujeto {subject}: checkpoint faltante, resumen marcado.')
        return

    subj_dir = _subject_dir(config, subject, results_dir)
    by_ratio = {}
    for f in sorted(glob.glob(os.path.join(subj_dir, 'ratio_*_seed_*.json'))):
        run = json.load(open(f))
        # el ratio sale del JSON, no del nombre de archivo: el nombre es una
        # etiqueta y podría redondear (antes usaba :.1f, que escribía 0.75
        # como "0.8"); el campo del JSON es el valor con el que se corrió.
        by_ratio.setdefault(float(run['ratio']), []).append(run)

    rows = []
    for ratio, runs in sorted(by_ratio.items()):
        df = pd.DataFrame(runs)
        row = {'config': config, 'subject': subject, 'status': 'ok', 'ratio': ratio, 'n_runs': len(runs)}
        for col in ('accuracy', 'f1', 'f1_macro', 'auc'):
            row[f'{col}_mean'] = df[col].mean()
            row[f'{col}_std'] = df[col].std()
        rows.append(row)

    if not rows:
        print(f'    Aviso: sin corridas para {config} sujeto {subject}, no se genera resumen.')
        return
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f'    Resumen -> {summary_path}')


def run_subject(subject, device):
    train_csv = os.path.join(DATA_DIR, f'subject_{subject:03d}.csv')
    test_csv = os.path.join(TEST_DATA_DIR, f'subject_{subject:03d}.csv')
    for path in (train_csv, test_csv):
        if not os.path.exists(path):
            raise FileNotFoundError(f'Falta {path}')

    norm_min, norm_max = train_norm_stats(train_csv)
    X_train, y_train = load(train_csv, norm_data=True, norm_min=norm_min, norm_max=norm_max)
    X_test, y_test = load(test_csv, norm_data=True, norm_min=norm_min, norm_max=norm_max)

    print(f'  ratio=0.0 (baseline "none", sin aumento -- compartido, {len(SEEDS)} corridas)')
    for seed in SEEDS:
        ensure_run('none', subject, 0.0, seed, X_train, y_train, X_test, y_test, device)
    summarize_subject('none', subject)

    n_train_target = int((y_train == 1).sum())  # el ratio de aumento es sobre la clase Target

    for config in CONFIGS_WITH_AUG:
        print(f'  config={config}')
        # el control usa el train ya cargado: misma normalización que X_train,
        # sin pasar por generate_synthetic ni por ningún checkpoint
        pool = (X_train, y_train) if config == REAL_OVERSAMPLE else get_pool(config, subject, train_csv)
        if pool is None:
            mark_missing_checkpoint(config, subject)
            summarize_subject(config, subject)
            continue
        clear_missing_checkpoint(config, subject)  # el checkpoint sí apareció esta vez
        X_pool, y_pool = pool
        for ratio in RATIOS:
            n_aug = round(ratio * n_train_target)
            for seed in SEEDS:
                ensure_run(config, subject, ratio, seed, X_train, y_train, X_test, y_test, device,
                           X_pool, y_pool, n_aug)
        summarize_subject(config, subject)


def _selfcheck():
    """Chequeo mínimo de summarize_subject(): sobre 3 corridas fake
    (accuracy 1.0/0.5/0.5) la media debe dar 2/3 y n_runs=3 -- corre contra
    un directorio temporal, nunca toca RESULTS_DIR real.

    Más el cableado de build_model(): el modelo seleccionado tiene que comer
    lo que le da su propio to_input() y devolver logits de 2 clases -- es el
    único punto donde eegnet (batch,1,canal,tiempo) y conformer
    (batch,canal,tiempo) difieren, y un mismatch de forma ahí revienta recién
    después de cargar los datos."""
    net, to_input = build_model(n_channels=8, n_times=206)
    assert net(to_input(np.zeros((4, 206, 8), dtype=np.float32))).shape == (4, 2)
    # un lr que NO es el default del modelo tiene que ir a un árbol aparte:
    # correr con otro lr encima de results/ mezclaría recetas en un mismo summary
    assert LR == OPTIM[MODEL]['lr'] or f'_lr{LR:g}' in RESULTS_DIR, (MODEL, LR, RESULTS_DIR)

    with tempfile.TemporaryDirectory() as tmp:
        for seed, acc in zip((1, 2, 3), (1.0, 0.5, 0.5)):
            path = run_path('fake_cfg', 1, 0.1, seed, results_dir=tmp)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                json.dump({'accuracy': acc, 'f1': acc, 'f1_macro': acc, 'auc': acc, 'ratio': 0.1}, f)
        summarize_subject('fake_cfg', 1, results_dir=tmp)
        row = pd.read_csv(os.path.join(tmp, 'fake_cfg', 'subject_001_summary.csv')).iloc[0]
        assert row['n_runs'] == 3, row['n_runs']
        assert abs(row['accuracy_mean'] - 2 / 3) < 1e-9, row['accuracy_mean']

        mark_missing_checkpoint('fake_cfg_missing', 2, results_dir=tmp)
        summarize_subject('fake_cfg_missing', 2, results_dir=tmp)
        row2 = pd.read_csv(os.path.join(tmp, 'fake_cfg_missing', 'subject_002_summary.csv')).iloc[0]
        assert row2['status'] == 'missing_checkpoint', row2['status']
        assert pd.isna(row2['accuracy_mean']), row2['accuracy_mean']

        # Regresión: un checkpoint que aparece DESPUÉS de haberse marcado
        # missing (p.ej. un alias de resolver nuevo que ahora sí lo encuentra)
        # no debe dejar corridas válidas enterradas bajo un resumen viejo.
        clear_missing_checkpoint('fake_cfg_missing', 2, results_dir=tmp)
        path = run_path('fake_cfg_missing', 2, 0.1, 1, results_dir=tmp)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump({'accuracy': 0.9, 'f1': 0.9, 'f1_macro': 0.9, 'auc': 0.9, 'ratio': 0.1}, f)
        summarize_subject('fake_cfg_missing', 2, results_dir=tmp)
        row3 = pd.read_csv(os.path.join(tmp, 'fake_cfg_missing', 'subject_002_summary.csv')).iloc[0]
        assert row3['status'] == 'ok', row3['status']
        assert row3['n_runs'] == 1, row3['n_runs']

        # Regresión: un ratio no representable con 1 decimal (0.75) no debe
        # colisionar con 0.8 en el nombre de archivo ni reportarse como 0.8 en
        # el resumen -- el formato viejo ({ratio:.1f}) hacía las dos cosas.
        assert run_path('c', 1, 0.75, 1, tmp) != run_path('c', 1, 0.8, 1, tmp)
        # ...y sin romper los nombres ya en disco, escritos con {ratio:.1f}
        for r in (0.0, 0.1, 0.5):
            assert os.path.basename(run_path('c', 1, r, 1, tmp)) == f'ratio_{r:.1f}_seed_1.json'
        for ratio in (0.75, 0.8):
            path = run_path('fake_cfg_ratios', 3, ratio, 1, results_dir=tmp)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                json.dump({'accuracy': 0.5, 'f1': 0.5, 'f1_macro': 0.5, 'auc': 0.5, 'ratio': ratio}, f)
        summarize_subject('fake_cfg_ratios', 3, results_dir=tmp)
        ratios = pd.read_csv(os.path.join(tmp, 'fake_cfg_ratios', 'subject_003_summary.csv'))['ratio'].tolist()
        assert ratios == [0.75, 0.8], ratios


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('subjects', nargs='*', type=int, help='IDs de sujeto (default: todos los del dataset)')
    parser.add_argument('--dataset', choices=list(DEFAULT_SUBJECTS), default=DATASET)
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                        help=f'épocas de EEGNet (default {DEFAULT_EPOCHS}); otro valor escribe en results..._ep<N>')
    parser.add_argument('--configs', nargs='+', default=None, metavar='NOMBRE',
                        help='configs con aumento a correr (default: todas). El baseline "none" '
                             'siempre se corre: no depende de ninguna GAN.')
    parser.add_argument('--pool-size', type=int, default=DEFAULT_POOL, dest='pool_size',
                        help='N muestras Target del pool sintético (default: el de siempre, '
                             '200/cond con NonTarget incluido); otro valor escribe en results..._pool<N>')
    parser.add_argument('--ratios', nargs='+', type=float, default=None,
                        help=f'ratios de aumento a correr (default {RATIOS}); los ya calculados se saltean')
    parser.add_argument('--lr', type=float, default=None,
                        help='learning rate de Adam (default: el del modelo, '
                             + ', '.join(f'{m}={o["lr"]:g}' for m, o in OPTIM.items())
                             + '); otro valor escribe en results..._lr<LR>')
    parser.add_argument('--model', choices=['eegnet', 'conformer'], default=DEFAULT_MODEL,
                        help=f'clasificador (default {DEFAULT_MODEL}); otro valor escribe en results..._<modelo>')
    args = parser.parse_args()

    use_dataset(args.dataset, args.epochs, args.lr, args.pool_size, args.model)
    _selfcheck()
    if args.configs:
        desconocidas = set(args.configs) - set(CONFIGS_WITH_AUG)
        if desconocidas:
            parser.error(f'config(s) desconocida(s): {sorted(desconocidas)}; hay {CONFIGS_WITH_AUG}')
        CONFIGS_WITH_AUG = args.configs
    if args.ratios:
        RATIOS = args.ratios
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    subjects = args.subjects or DEFAULT_SUBJECTS[args.dataset]
    print(f'Sujetos {subjects} | ratios {RATIOS} | configs {CONFIGS_WITH_AUG}')
    print(f'Dataset {args.dataset}, {MODEL} {_tea.N_EPOCHS} épocas, lr={LR:g}, pool={_tea.POOL_N_TARGET}: '
          f'datos en {DATA_DIR}/{TEST_DATA_DIR}, resultados en {RESULTS_DIR}')
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for subject in subjects:
        print(f'\n{"="*55}\n  Sujeto {subject}\n{"="*55}')
        try:
            run_subject(subject, device)
        except Exception as exc:
            print(f'  Aviso: sujeto {subject} omitido ({exc}).')
            continue

    print(f'\nListo. Resultados en {RESULTS_DIR}')
