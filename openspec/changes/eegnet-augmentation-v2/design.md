## Context

`train_eegnet_augmentation.py` guarda todo en un único CSV
(`ablation_results/eegnet_augmentation.csv`) reescrito incrementalmente
(`pd.read_csv(...).to_dict('records')` + filas nuevas + `to_csv`). Cuando el
esquema de columnas cambió entre corridas (de una sola corrida a
`N_REPEATS=100` agregado), el merge no lo detectó: pandas simplemente rellenó
con NaN las columnas que cada fila no tenía. Root cause: el formato de
almacenamiento (una tabla ancha mutable) no tiene forma de distinguir "no se
corrió" de "se corrió con otro esquema".

Este cambio corre un experimento nuevo (menos repeticiones, distinto set de
configs) y aprovecha para eliminar esa clase de bug: cada corrida individual
es su propio archivo inmutable; el resumen agregado se recalcula siempre
desde cero a partir de los archivos de corrida presentes, nunca se
mergea con un resumen previo.

## Goals / Non-Goals

**Goals:**
- Un archivo por corrida (config, sujeto, ratio, seed) con su
  `classification_report`, para que agregar/recortar corridas no pueda
  corromper un archivo de otra corrida.
- Resumen (mean/std) recalculado por config+sujeto a partir de los archivos
  de corrida existentes en disco — nunca por merge incremental de dicts con
  distinto esquema.
- Reusar el código ya probado de `train_eegnet_augmentation.py` (modelo
  EEGNet, `sample_pool`, `get_pool`, `generate_synthetic`, normalización de
  `compare_samples.py`) en vez de reimplementarlo.
- Resolución de checkpoint específica por config (3 fuentes distintas:
  `ablation_pipeline.py`, `train_tts_gan_baseline.py`, worktree `main`).

**Non-Goals:**
- No migra ni corrige `ablation_results/eegnet_augmentation.csv` ni
  `comparison_plots/eegnet_augmentation.csv` (quedan como están; ver
  conversación previa para homogeneizarlos si hace falta).
- No reentrena GANs faltantes — si un checkpoint no existe para un sujeto,
  esa combinación se saltea (mismo comportamiento que `get_pool()` ya tiene
  en el script viejo).
- No paraleliza entre sujetos/configs — corre secuencial como el script
  viejo (`train_eegnet_augmentation.sh` ya aísla por sujeto vía subprocess
  si se quiere paralelizar a nivel shell).

## Decisions

**1. Almacenamiento por-corrida en vez de CSV acumulativo, todo bajo una
carpeta propia `eeg_net_aug/`** (script + resultados autocontenidos, no
mezclado con la raíz del repo ni con `ablation_results/`/`comparison_plots/`):
```
eeg_net_aug/
  train_eegnet_augmentation_v2.py
  results/
    <config>/                          # fm_lambda_0, fm_lambda_50_postnet,
                                        # eeg_gan_vanilla_full, tts_gan_baseline, none
      subject_<SSS>/
        ratio_<R>_seed_<N>.json        # classification_report(output_dict=True)
                                        # + accuracy/f1/f1_macro/auc/ratio/seed/n_aug
        MISSING_CHECKPOINT              # sentinel vacío si falta el .pt de esa config/sujeto
      subject_<SSS>_summary.csv         # recalculado desde los .json de esa carpeta
```
Cada corrida escribe su propio `.json` (nombre determinístico por
`ratio_seed`) — si ya existe, se saltea (resume gratis, mismo patrón
"ya existe, se omite" de los scripts bash). El resumen se recalcula
completo con `glob` sobre los `.json` presentes, nunca leyendo un CSV
anterior — así no hay forma de que se cuelen filas de un esquema viejo.
Alternativa descartada: seguir con un CSV único pero versionado (columna
`schema_version`) — más frágil, sigue permitiendo mezclar por error si
alguien no chequea la versión antes de leer.

**2. `none` (baseline) es una config más en el mismo árbol, no un caso
especial.** Sus corridas viven en `none/subject_<SSS>/ratio_0.0_seed_<N>.json`
y se computan una sola vez (compartidas, igual que en el script viejo) antes
de iterar las configs con augmentation — evita entrenar el baseline 4 veces.

**3. Resolución de checkpoint por config (3 fuentes distintas):**
| config | checkpoint | requiere |
|---|---|---|
| `fm_lambda_0` | `trained_models/ABLATION_fm_lambda_0_s{:03d}.pt` | mismo patrón que `ablation_pipeline.checkpoint_path()` |
| `fm_lambda_50_postnet` | `trained_models/ABLATION_fm_lambda_50_postnet_s{:03d}.pt` | ídem |
| `tts_gan_baseline` | `trained_models/GAN_009_tts_gan_baseline_s{:03d}.pt` | ya generado in-process (misma rama) |
| `eeg_gan_vanilla_full` | `EEG-DDGAN-main-vanilla/trained_models/EEG_GAN_vanilla_full_s{:03d}.pt` | pool vía `generate_eeggan_vanilla_augpool.sh` (subprocess, otro venv) — reusa `get_pool()` tal cual |
| `none` | (sin checkpoint) | — |

Para las 2 primeras (`fm_lambda_0`, `fm_lambda_50_postnet`), el pool
sintético se genera **in-process** con `generate_synthetic()` (misma rama
`ttsgan-direct`, mismo mecanismo que ya usa `get_pool()` para
`tts_gan_baseline`/DDGAN) — no necesitan subprocess porque no son
AE-coupled.

**4. `N_REPEATS=10`, seeds literales `1..10`** (no `SEED+r=42+r` como el
script viejo) — así queda explícito en el nombre de archivo (`seed_3.json`)
cuál semilla es cada corrida, en vez de tener que reconstruir `42+r`.

**5. Checkpoint faltante → sentinel `MISSING_CHECKPOINT`, no fila ausente.**
El resumen por sujeto lee ese sentinel y escribe una fila explícita con
`status=missing_checkpoint` y métricas `NaN`, en vez de simplemente no
generar la fila (el bug que motivó este cambio era justamente "faltante
silencioso" — acá el resumen dice por qué falta).

## Risks / Trade-offs

- [10 corridas en vez de 100 → std menos estable] → aceptado explícitamente
  por el usuario por tiempo de cómputo; el resumen igual reporta `n_runs`
  real (puede ser <10 si alguna corrida individual tira excepción) para que
  quede trazable.
- [3 mecanismos de resolución de checkpoint distintos → más superficie de
  fallo] → cada uno reusa código ya probado (`checkpoint_path()` de
  `ablation_pipeline`, `get_pool()` de `train_eegnet_augmentation.py`) en vez
  de reimplementar; el sentinel `MISSING_CHECKPOINT` hace explícito cuál
  combinación no se pudo evaluar.
- [Muchos archivos chicos (config × sujeto × ratio × seed ≈ 4×10×5×10 = 2000
  `.json` + 500 de `none`)] → aceptable para este volumen; cada uno es
  trivial de inspeccionar/borrar individualmente para reintentar una corrida
  puntual, a diferencia de un CSV único.

## Migration Plan

No aplica (carpeta nueva, no toca lo existente). Rollback: borrar
`eeg_net_aug/` completa.

## Open Questions

- Ninguna bloqueante. Si en el futuro se quiere una tabla única "ancha" para
  graficar (como hace `plot_eegnet_augmentation.py` con el CSV viejo), se
  puede generar aparte concatenando los `subject_<SSS>_summary.csv` de todas
  las configs — no forma parte de este cambio.
