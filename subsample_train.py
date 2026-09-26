"""Submuestrea el train real a una fracción de sus trials, estratificado por
condición, para medir el aumento sintético en régimen de pocos datos.

Motivación: el baseline de EEGNet sin aumento subió de 0.573 (single-channel,
paper original) a 0.829 de Macro-F1 al pasar a multicanal, y con ese techo las
ganancias del aumento cayeron de +7.03 a +3.14 puntos. No queda margen para
mejorar. Recortando el train real se vuelve al régimen donde el aumento tiene
algo que aportar -- que además es el caso de uso que motiva el método: la
calibración por sujeto es cara.

Submuestrea por TRIAL, nunca por fila: el CSV está en formato largo y cada
trial ocupa n_channels filas consecutivas (16 en BNCI2014_009). Partir un
trial por la mitad dejaría trials con canales faltantes que el Dataloader
reshapea mal, sin error visible.

El test NO se toca: el held-out se mantiene completo e idéntico en todas las
fracciones, que es lo que hace comparables los números entre fracciones y
contra los resultados ya calculados.

Uso (desde la raíz del repo):
    uv run python subsample_train.py --frac 0.25              # sujetos 1..10 de 009
    uv run python subsample_train.py --frac 0.5 --dataset BNCI2014_008
    uv run python subsample_train.py --frac 0.25 --subjects 1
"""
import argparse
import os

import numpy as np
import pandas as pd

import moabb_pipeline as _mp

SEED = 42  # fijo: la misma fracción tiene que dar el mismo subconjunto siempre


def frac_tag(frac):
    """0.25 -> 'f25'. El sufijo va en el nombre del directorio de datos, del
    checkpoint y del árbol de resultados, para que corridas con distinta
    cantidad de datos reales nunca caigan en el mismo summary."""
    return f'f{round(frac * 100)}'


def subsample_csv(in_path, out_path, frac, seed=SEED):
    """Escribe en out_path una fracción de los TRIALS de in_path, estratificada
    por Condition. Devuelve (n_trials_in, n_trials_out)."""
    df = pd.read_csv(in_path)
    # Condition es constante dentro de un trial: basta la primera fila de cada uno
    por_trial = df.groupby('Trial')['Condition'].first()

    rng = np.random.default_rng(seed)
    elegidos = []
    for cond, trials in por_trial.groupby(por_trial):
        ids = trials.index.to_numpy()
        n = max(1, round(len(ids) * frac))  # nunca dejar una clase vacía
        elegidos.append(rng.choice(ids, size=n, replace=False))
    elegidos = np.concatenate(elegidos)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df[df['Trial'].isin(elegidos)].to_csv(out_path, index=False)
    return len(por_trial), len(elegidos)


def _selfcheck():
    """Un trial nunca se parte (todas sus filas entran o ninguna), la
    proporción de clases se mantiene y la fracción sale dentro de un trial de
    lo pedido."""
    import tempfile

    filas = []
    for trial in range(100):
        cond = 1 if trial < 20 else 0  # 1:5, como el desbalance del P300
        for ch in range(4):
            filas.append({'ParticipantID': 1, 'Condition': cond, 'Trial': trial,
                          'Electrode': f'Ch{ch}', 'Time1': float(trial), 'Time2': float(ch)})
    df = pd.DataFrame(filas)

    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, 'in.csv')
        dst = os.path.join(tmp, 'sub', 'out.csv')
        df.to_csv(src, index=False)
        n_in, n_out = subsample_csv(src, dst, 0.25)
        assert (n_in, n_out) == (100, 25), (n_in, n_out)

        out = pd.read_csv(dst)
        conteo = out.groupby('Trial').size()
        assert (conteo == 4).all(), conteo[conteo != 4]           # ningún trial partido
        por_trial = out.groupby('Trial')['Condition'].first()
        assert (por_trial == 1).sum() == 5, por_trial.sum()        # 20 * 0.25, estratificado
        assert (por_trial == 0).sum() == 20, len(por_trial)

        # mismo seed -> mismo subconjunto (reproducible entre maquinas/corridas)
        dst2 = os.path.join(tmp, 'sub2', 'out.csv')
        subsample_csv(src, dst2, 0.25)
        assert pd.read_csv(dst2).equals(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--frac', type=float, required=True, help='fracción de trials a conservar (0-1)')
    parser.add_argument('--dataset', choices=list(_mp._DIR_SUFFIX), default='BNCI2014_009')
    parser.add_argument('--subjects', nargs='+', type=int, default=None,
                        help='IDs de sujeto (default: todos los del dataset)')
    args = parser.parse_args()
    if not 0 < args.frac < 1:
        parser.error(f'--frac debe estar entre 0 y 1 (dado: {args.frac})')

    _selfcheck()
    data_dir, _ = _mp._data_dirs(args.dataset)
    out_dir = f'{data_dir}_{frac_tag(args.frac)}'
    subjects = args.subjects or _mp.DEFAULT_SUBJECTS[args.dataset]
    print(f'{data_dir} -> {out_dir} (frac={args.frac}, seed={SEED})')

    for subject in subjects:
        src = os.path.join(data_dir, f'subject_{subject:03d}.csv')
        if not os.path.exists(src):
            print(f'  sujeto {subject}: falta {src}, se omite')
            continue
        dst = os.path.join(out_dir, f'subject_{subject:03d}.csv')
        n_in, n_out = subsample_csv(src, dst, args.frac)
        print(f'  sujeto {subject}: {n_in} -> {n_out} trials')
