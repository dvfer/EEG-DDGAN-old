"""Fidelidad espectral medida POR TRIAL, para la tabla de ablación.

Por qué hace falta otra métrica: las heredadas del paper de conferencia no
pueden ver lo que el PostNet corrige.
  - La JSD se computa sobre 0.1-20 Hz, que excluye la banda donde el suavizado
    actúa.
  - Los errores de latencia/amplitud del ERP se miden sobre el gran promedio,
    que cancela el ruido de alta frecuencia de cada trial.
Una señal generada puede tener el ERP promedio perfecto y aun así ser ruido
por trial. Eso es justo lo que separa las configs con y sin PostNet.

Dos medidas, ambas por trial y por canal, comparadas contra los trials reales:
  - hf: fracción de potencia sobre 20 Hz (banda fuera del P300 fisiológico).
  - tv: variación total media, |diff| promedio -- rugosidad de la traza.

En ambas, más cerca del real es mejor; quedar POR DEBAJO significa sobre-suavizar.

Uso (desde la raíz del repo):
    uv run python eeg_net_aug/fidelity_per_trial.py --n 1000
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import signal as sg

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import compare_samples as cs  # noqa: E402
from compare_samples import DATA_DIR, GAN_DIR, GEN_DIR, load, train_norm_stats  # noqa: E402

FS = 256        # BNCI2014_009
HF_CUTOFF = 20  # límite superior de la banda P300 fisiológica
SUBJECT = 1     # el sujeto piloto: es donde vive la ablación

CONFIGS = ['baseline', 'postnet_only', 'dwt', 'dwt_stacking', 'fm_lambda_0',
           'fm_lambda_20_postnet', 'fm_lambda_50', 'fm_lambda_0_postnet',
           'fm_lambda_50_postnet', 'eeg_gan_vanilla_full']


def per_trial(X):
    """X: (trial, seq, canal) -> (hf medio, tv media) sobre trials y canales."""
    x = X.transpose(0, 2, 1).reshape(-1, X.shape[1])
    f, P = sg.welch(x, fs=FS, nperseg=min(128, x.shape[1]), axis=-1)
    hf = P[:, f > HF_CUTOFF].sum(axis=1) / P.sum(axis=1)
    tv = np.abs(np.diff(x, axis=-1)).mean(axis=-1)
    return float(hf.mean()), float(tv.mean())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--n', type=int, default=1000, help='trials Target sintéticos por config')
    ap.add_argument('--out', default='ablation_results/fidelity_per_trial.csv')
    args = ap.parse_args(argv)

    train_csv = os.path.join(DATA_DIR, f'subject_{SUBJECT:03d}.csv')
    norm_min, norm_max = train_norm_stats(train_csv)
    Xr, yr = load(train_csv, norm_data=True, norm_min=norm_min, norm_max=norm_max)
    hf_r, tv_r = per_trial(Xr[yr == 1])
    rows = [dict(config='real', n_trials=int((yr == 1).sum()), hf=hf_r, tv=tv_r,
                 hf_excess=0.0, tv_excess=0.0)]
    print(f'real: hf={hf_r:.4f} tv={tv_r:.5f} ({rows[0]["n_trials"]} trials)')

    for cfg in CONFIGS:
        ckpt = os.path.join(GAN_DIR, f'ABLATION_{cfg}_s{SUBJECT:03d}.pt')
        if not os.path.exists(ckpt):
            print(f'  falta {ckpt}, se omite')
            continue
        out_csv = os.path.join(GEN_DIR, f'_fidelity_{cfg}.csv')
        # solo Target y en cantidad >> los trials reales: la métrica es por
        # trial, así que cuantos más trials, menos ruido en la estimación
        cs.generate_synthetic(ckpt, train_csv, out_csv, {'Target': args.n, 'NonTarget': 0})
        Xg, yg = load(out_csv, norm_data=False)
        os.remove(out_csv)
        hf, tv = per_trial(Xg[yg == 1])
        rows.append(dict(config=cfg, n_trials=int((yg == 1).sum()), hf=hf, tv=tv,
                         hf_excess=hf - hf_r, tv_excess=tv - tv_r))
        print(f'{cfg:24s} hf={hf:.4f} ({hf / hf_r:5.1f}x real)  tv={tv:.5f}')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False, float_format='%.6f')
    print(f'\n-> {args.out}')


def _selfcheck():
    """Una señal suavizada debe medir menos HF y menos TV que una ruidosa."""
    rng = np.random.default_rng(0)
    t = np.linspace(0, 1, 200)
    limpia = np.sin(2 * np.pi * 5 * t)[None, :, None].repeat(20, 0)
    ruidosa = limpia + rng.normal(0, 0.3, limpia.shape)
    hf_l, tv_l = per_trial(limpia)
    hf_r, tv_r = per_trial(ruidosa)
    assert hf_l < hf_r, (hf_l, hf_r)
    assert tv_l < tv_r, (tv_l, tv_r)


if __name__ == '__main__':
    _selfcheck()
    main()
