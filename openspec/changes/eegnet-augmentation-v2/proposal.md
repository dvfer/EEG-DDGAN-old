## Why

`train_eegnet_augmentation.py` acumula resultados en un único CSV mutable
(`ablation_results/eegnet_augmentation.csv`) que se reescribe incrementalmente
sujeto por sujeto. Cuando el script cambió de esquema (de una corrida simple
a `N_REPEATS=100` agregado con media/std), las filas de sujetos corridos con
versiones distintas del código quedaron mezcladas en la misma tabla sin
ningún chequeo de compatibilidad, produciendo columnas con NaN silencioso
(`comparison_plots/eegnet_augmentation.csv`, sujeto 1 vs. sujetos 2-10).
Además, el nuevo experimento compara un set de configs distinto al actual
(`fm_lambda_0` y `fm_lambda_50_postnet` del ablation study en vez del modelo
único de tesis) y necesita menos repeticiones por tiempo de cómputo (10 en
vez de 100). En vez de parchear el script viejo, se necesita un script nuevo
cuyo formato de almacenamiento no pueda mezclar esquemas incompatibles.

## What Changes

- Todo lo nuevo vive en una carpeta propia `eeg_net_aug/` (script +
  resultados), separada del resto del repo y de
  `train_eegnet_augmentation.py`.
- Nuevo script `eeg_net_aug/train_eegnet_augmentation_v2.py` (no reemplaza ni modifica
  `train_eegnet_augmentation.py`, que sigue existiendo para su propio
  experimento/CSV).
- Por cada sujeto (1-10) y cada config de generador, corre EEGNet **10 veces**
  (seeds literales 1-10, no `SEED+r`) por cada ratio de aumento
  `{0.1, 0.2, 0.3, 0.4, 0.5}`, más un baseline `none` (ratio 0.0, sin
  aumento, compartido entre configs).
- Configs de generador: `fm_lambda_0`, `fm_lambda_50_postnet` (checkpoints de
  `ablation_pipeline.py`), `eeg_gan_vanilla_full` (worktree `main`,
  AE-coupled), `tts_gan_baseline`, y `none` (EEGNet sin aumento).
- Almacenamiento **por-corrida**: cada corrida individual (config, sujeto,
  ratio, seed) escribe su propio archivo con el `classification_report` de
  sklearn — nunca se reescribe ni se mezcla con corridas de otro esquema.
  Estructura de carpetas por config → por sujeto → un archivo por corrida.
- Un resumen por config+sujeto (CSV, recalculado desde cero a partir de los
  archivos de corrida existentes) con media y std de accuracy/f1/f1_macro/auc
  por ratio sobre las 10 corridas — nunca por merge incremental con datos de
  otra corrida previa.
- Si falta el checkpoint de una config para un sujeto, se saltea esa
  combinación con aviso (igual que el comportamiento ya existente de
  `get_pool()`) y el resumen dice explícitamente que falta (no una fila
  ausente sin explicación).
- Reusa (no reimplementa) el modelo EEGNet, `sample_pool()`, `get_pool()`,
  `generate_synthetic()`, `load()`/`train_norm_stats()` ya probados en
  `train_eegnet_augmentation.py`/`compare_samples.py`.

## Capabilities

### New Capabilities
- `eegnet-augmentation-repeated-eval`: evaluación repetida (N corridas con
  seeds fijas) de EEGNet con datos aumentados por distintos generadores GAN,
  con almacenamiento por-corrida (sin CSV mutable acumulativo) y resumen
  agregado recalculado desde los archivos de corrida.

### Modified Capabilities
(ninguna — `train_eegnet_augmentation.py` no se toca)

## Impact

- Carpeta nueva `eeg_net_aug/` (raíz del repo, `ttsgan-direct`), autocontenida:
  - `eeg_net_aug/train_eegnet_augmentation_v2.py` — el script.
  - `eeg_net_aug/results/` — resultados por-corrida + resúmenes (ver design.md).
  No colisiona con `ablation_results/eegnet_augmentation.csv` ni con
  `comparison_plots/`.
- Lee (no modifica) checkpoints ya producidos por `ablation_pipeline.py`
  (`trained_models/ABLATION_fm_lambda_0_s*.pt`,
  `ABLATION_fm_lambda_50_postnet_s*.pt`), `train_tts_gan_baseline.py`
  (`trained_models/GAN_009_tts_gan_baseline_s*.pt`), y del worktree `main`
  (`EEG-DDGAN-main-vanilla/trained_models/EEG_GAN_vanilla_full_s*.pt`, vía
  `generate_eeggan_vanilla_augpool.sh`).
- Depende de que esos checkpoints existan por sujeto en la máquina donde se
  corra (GPU); si faltan para algún sujeto/config, esa combinación se
  saltea con aviso explícito en el resumen.
