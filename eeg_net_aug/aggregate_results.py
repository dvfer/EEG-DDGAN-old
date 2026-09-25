"""Agrega los JSON por corrida a un único CSV por celda, que es lo que consume
el paper (ver openspec/changes/journal-paper-multidataset/design.md D2).

Por qué existe: los crudos son ~35.000 archivos / 229 MB y no se versionan.
El CSV resultante son ~10^3 filas y sí puede acompañar al manuscrito, de modo
que las tablas se regeneren sin acceso a los árboles completos ni a demerzel.

Granularidad: una fila por (dataset, régimen, config, sujeto, ratio), con
media/desvío/n sobre los 50 seeds. Se agrega hasta SUJETO y no hasta cohorte
a propósito: los tests pareados y el boxplot necesitan el valor por sujeto.

Uso (desde la raíz del repo):
    uv run python eeg_net_aug/aggregate_results.py
    uv run python eeg_net_aug/aggregate_results.py --out ../escrito/journal/data/results.csv
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
METRICS = ('accuracy', 'f1', 'f1_macro', 'auc')

# Cada árbol es un RÉGIMEN experimental distinto. No se mezclan entre sí (D3):
# el sufijo del directorio codifica los hiperparámetros con los que se corrió.
TREES = {
    'results_ep100_pool2000':     dict(dataset='BNCI2014_009', regime='pool2000', n_epochs=100, lr=1e-3, pool=2000),
    'results_008_ep100_pool2000': dict(dataset='BNCI2014_008', regime='pool2000', n_epochs=100, lr=1e-3, pool=2000),
    'results_ep100':              dict(dataset='BNCI2014_009', regime='pool200',  n_epochs=100, lr=1e-3, pool=200),
    'results_008_ep100':          dict(dataset='BNCI2014_008', regime='pool200',  n_epochs=100, lr=1e-3, pool=200),
}


def collect(tree_dir, meta):
    """Una fila por (config, sujeto, ratio). El ratio sale del JSON, no del
    nombre de archivo -- el nombre es una etiqueta y en su día redondeó 0.75
    a 0.8 (ver el fix en train_eegnet_augmentation_v2.py)."""
    runs = {}
    for path in glob.glob(os.path.join(tree_dir, '*', 'subject_*', 'ratio_*.json')):
        with open(path) as f:
            r = json.load(f)
        key = (path.split(os.sep)[-3], int(r['subject']), float(r['ratio']))
        runs.setdefault(key, []).append(r)

    rows = []
    for (config, subject, ratio), rs in sorted(runs.items()):
        row = dict(meta, config=config, subject=subject, ratio=ratio, n_runs=len(rs))
        for m in METRICS:
            vals = np.array([x[m] for x in rs if m in x], dtype=float)
            vals = vals[~np.isnan(vals)]
            row[f'{m}_mean'] = vals.mean() if len(vals) else np.nan
            row[f'{m}_std'] = vals.std(ddof=1) if len(vals) > 1 else np.nan
        rows.append(row)
    return rows


def missing_cells(tree_dir):
    """Configs marcadas como checkpoint faltante -- se reportan explícitamente
    en vez de quedar como una ausencia silenciosa en la tabla (spec
    results-analysis: 'Celda faltante')."""
    out = []
    for path in glob.glob(os.path.join(tree_dir, '*', 'subject_*', 'MISSING_CHECKPOINT')):
        parts = path.split(os.sep)
        out.append((parts[-3], int(parts[-2].removeprefix('subject_'))))
    return sorted(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--root', default=HERE, help='directorio que contiene los árboles results*')
    ap.add_argument('--out', default=os.path.join(HERE, 'aggregated_results.csv'))
    args = ap.parse_args(argv)

    all_rows, faltantes = [], []
    for tree, meta in TREES.items():
        tree_dir = os.path.join(args.root, tree)
        if not os.path.isdir(tree_dir):
            print(f'  Aviso: no existe {tree_dir}, se omite ese régimen.')
            continue
        rows = collect(tree_dir, meta)
        all_rows += rows
        for config, subject in missing_cells(tree_dir):
            faltantes.append(dict(meta, config=config, subject=subject))
        print(f'  {tree}: {len(rows)} celdas, {sum(r["n_runs"] for r in rows)} corridas')

    if not all_rows:
        sys.exit('No se encontró ningún árbol de resultados.')

    df = pd.DataFrame(all_rows).sort_values(['dataset', 'regime', 'config', 'subject', 'ratio'])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False, float_format='%.6f')
    print(f'\n{len(df)} celdas -> {args.out}')

    if faltantes:
        fpath = os.path.splitext(args.out)[0] + '_missing.csv'
        pd.DataFrame(faltantes).to_csv(fpath, index=False)
        print(f'{len(faltantes)} celdas sin checkpoint -> {fpath}')
    return df


def _selfcheck():
    """Chequeo mínimo de collect(): 2 corridas fake de la misma celda deben
    colapsar a una fila con n_runs=2 y la media correcta, y el ratio debe
    salir del JSON aunque el nombre de archivo diga otra cosa."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        d = os.path.join(tmp, 'cfg', 'subject_003')
        os.makedirs(d)
        for seed, acc in ((1, 1.0), (2, 0.5)):
            # nombre de archivo con el ratio MAL redondeado a propósito
            with open(os.path.join(d, f'ratio_0.8_seed_{seed}.json'), 'w') as f:
                json.dump({'accuracy': acc, 'f1': acc, 'f1_macro': acc, 'auc': acc,
                           'subject': 3, 'ratio': 0.75}, f)
        rows = collect(tmp, {'dataset': 'X'})
        assert len(rows) == 1, rows
        assert rows[0]['n_runs'] == 2
        assert rows[0]['ratio'] == 0.75, rows[0]['ratio']   # del JSON, no del nombre
        assert abs(rows[0]['accuracy_mean'] - 0.75) < 1e-12
        assert abs(rows[0]['accuracy_std'] - np.std([1.0, 0.5], ddof=1)) < 1e-12


if __name__ == '__main__':
    _selfcheck()
    main()
