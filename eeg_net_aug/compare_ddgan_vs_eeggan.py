"""F1 de la clase positiva (campo `f1`): DDGAN vs EEG-GAN vanilla, pareado por seed.

El análisis que ya existe (results/f1_analysis.md) compara cada config contra el
baseline sin aumento. Esto responde la otra pregunta: ¿DDGAN le gana a EEG-GAN?
Pareado seed a seed (mismo split, misma inicialización) dentro de cada ratio.

Uso:
    uv run python eeg_net_aug/compare_ddgan_vs_eeggan.py
    uv run python eeg_net_aug/compare_ddgan_vs_eeggan.py --metric f1_macro
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
TREES = {
    'results_ep100_pool2000': 'BNCI2014_009 / pool2000',
    'results_ep100': 'BNCI2014_009 / pool200',
    'results_008_ep100_pool2000': 'BNCI2014_008 / pool2000',
    'results_008_ep100': 'BNCI2014_008 / pool200',
}
DDGAN, VANILLA, NONE = 'fm_lambda_50_postnet', 'eeg_gan_vanilla_full', 'none'


def load(tree_dir, metric):
    rows = []
    for path in glob.glob(os.path.join(tree_dir, '*', 'subject_*', 'ratio_*.json')):
        with open(path) as f:
            r = json.load(f)
        rows.append(dict(config=path.split(os.sep)[-3], subject=int(r['subject']),
                         ratio=float(r['ratio']), seed=int(r['seed']), val=float(r[metric])))
    return pd.DataFrame(rows)


def paired(df, a, b):
    """Une a y b por (subject, ratio, seed). Devuelve delta = a - b."""
    ka = df[df.config == a].set_index(['subject', 'ratio', 'seed']).val
    kb = df[df.config == b].set_index(['subject', 'ratio', 'seed']).val
    j = pd.concat([ka.rename('a'), kb.rename('b')], axis=1, join='inner').reset_index()
    j['delta'] = j.a - j.b
    return j


def vs_baseline(df, a):
    """a (cualquier ratio) contra `none` del mismo sujeto y seed."""
    ka = df[df.config == a].set_index(['subject', 'ratio', 'seed']).val
    kb = df[df.config == NONE].set_index(['subject', 'seed']).val
    j = ka.rename('a').reset_index().join(kb.rename('b'), on=['subject', 'seed'], how='inner')
    j['delta'] = j.a - j.b
    return j.dropna(subset=['delta'])


def _p(x):
    """t-test y Wilcoxon sobre las medias por sujeto (n = nº sujetos)."""
    per_subj = x.groupby('subject').delta.mean()
    if len(per_subj) < 3:
        return per_subj, np.nan, np.nan
    t = stats.ttest_1samp(per_subj, 0).pvalue
    w = stats.wilcoxon(per_subj).pvalue
    return per_subj, t, w


def report(tree, label, metric):
    df = load(tree, metric)
    if df.empty:
        return
    print(f'\n{"=" * 72}\n{label}  ({os.path.basename(tree)})  métrica: {metric}\n{"=" * 72}')

    d = paired(df, DDGAN, VANILLA)
    per_subj, tp, wp = _p(d)
    print(f'\nDDGAN - EEG-GAN  (n={len(d)} pares, {len(per_subj)} sujetos, '
          f'ratios {sorted(d.ratio.unique())})')
    print(f'  Δ medio por sujeto: {per_subj.mean():+.4f}   '
          f'gana en {int((per_subj > 0).sum())}/{len(per_subj)} sujetos   '
          f'p(t)={tp:.4f}  p(wilcoxon)={wp:.4f}')

    print('\n  por sujeto (paired t-test sobre ratios×seeds):')
    print(f'  {"suj":>4} {"Δ":>8} {"p":>9}  {"DDGAN":>7} {"EEGGAN":>7}')
    for s, g in d.groupby('subject'):
        p = stats.ttest_rel(g.a, g.b).pvalue
        mark = '*' if p < 0.05 else ' '
        print(f'  {s:>4} {g.delta.mean():+8.4f} {p:9.4f}{mark} {g.a.mean():7.4f} {g.b.mean():7.4f}')

    print('\n  por ratio:')
    for r, g in d.groupby('ratio'):
        ps, tp_, wp_ = _p(g)
        print(f'  ratio {r:<4} Δ={ps.mean():+.4f}  gana {int((ps > 0).sum())}/{len(ps)} suj  '
              f'p(t)={tp_:.4f}')

    print('\n  contra baseline sin aumento (`none`):')
    for cfg in sorted(df.config.unique()):
        if cfg == NONE:
            continue
        b = vs_baseline(df, cfg)
        ps, tp_, wp_ = _p(b)
        print(f'  {cfg:<24} Δ={ps.mean():+.4f}  gana {int((ps > 0).sum())}/{len(ps)} suj  '
              f'p(t)={tp_:.4f}  p(w)={wp_:.4f}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--root', default=HERE)
    ap.add_argument('--metric', default='f1', help='f1 = clase positiva (Target)')
    args = ap.parse_args(argv)
    for tree, label in TREES.items():
        d = os.path.join(args.root, tree)
        if os.path.isdir(d):
            report(d, label, args.metric)


def _selfcheck():
    df = pd.DataFrame([
        dict(config=DDGAN, subject=1, ratio=0.1, seed=1, val=0.6),
        dict(config=VANILLA, subject=1, ratio=0.1, seed=1, val=0.5),
        dict(config=NONE, subject=1, ratio=0.0, seed=1, val=0.4),
    ])
    assert paired(df, DDGAN, VANILLA).delta.iloc[0] == pytest_approx(0.1)
    assert vs_baseline(df, DDGAN).delta.iloc[0] == pytest_approx(0.2)


def pytest_approx(x, tol=1e-12):
    class _A:
        def __eq__(self, other):
            return abs(other - x) < tol
    return _A()


if __name__ == '__main__':
    _selfcheck()
    main()
