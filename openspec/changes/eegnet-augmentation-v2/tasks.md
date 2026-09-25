## 1. Esqueleto y resolución de checkpoints

- [x] 1.1 Crear `eeg_net_aug/train_eegnet_augmentation_v2.py`. Como
  `compare_samples.py`/`train_eegnet_augmentation.py` viven en la raíz del
  repo (un nivel arriba de `eeg_net_aug/`), agregar esa raíz a `sys.path`
  (`sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))`)
  antes de importar `load`, `train_norm_stats`, `generate_synthetic`,
  `DATA_DIR`, `TEST_DATA_DIR`, `GAN_DIR`, `GEN_DIR` de `compare_samples.py`
  y `EEGNet`, `to_eegnet_input`, `sample_pool` de
  `train_eegnet_augmentation.py` — sin reimplementarlos.
- [x] 1.2 Definir `RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')`
  (relativo al propio script, no al cwd, para que `eeg_net_aug/` quede
  autocontenida sin importar desde dónde se invoque), `RATIOS = [0.1, 0.2,
  0.3, 0.4, 0.5]`, `SEEDS = list(range(1, 11))` (literal 1..10, no `SEED+r`).
- [x] 1.3 Definir `CHECKPOINT_RESOLVERS`: dict `config_name -> función(subject) -> path | None`,
  con paths relativos vía `GAN_DIR` (igual convención que el resto del repo
  — el script se invoca con cwd=raíz del repo, como todos los demás; solo
  `RESULTS_DIR` se ancla a `__file__` porque es exclusivo de `eeg_net_aug/`)
  cubriendo las 4 fuentes (`ABLATION_{name}_s{:03d}.pt` en `GAN_DIR` para
  `fm_lambda_0`/`fm_lambda_50_postnet`; `GAN_009_tts_gan_baseline_s{:03d}.pt`
  en `GAN_DIR` para `tts_gan_baseline`; checkpoint del worktree `main` para
  `eeg_gan_vanilla_full`).
- [x] 1.4 Reusar/adaptar `get_pool()` de `train_eegnet_augmentation.py` para
  las 4 configs con augmentation (in-process para las 3 de esta rama, vía
  `generate_eeggan_vanilla_augpool.sh` para `eeg_gan_vanilla_full`); si el
  checkpoint no existe, devolver `None` (no lanzar) para que el caller pueda
  escribir el sentinel `MISSING_CHECKPOINT`.

## 2. Corrida individual y almacenamiento por-archivo

- [x] 2.1 Escribir `run_single(config, subject, ratio, seed, X_train, y_train, X_test, y_test, device, X_pool=None, y_pool=None, n_aug=0)`
  que entrena EEGNet una vez con esa seed (reusando `train_and_eval`/lógica
  equivalente de `train_eegnet_augmentation.py`) y devuelve
  `classification_report(y_test, preds, output_dict=True)` + accuracy/f1/f1_macro/auc.
- [x] 2.2 Escribir `run_path(config, subject, ratio, seed)` →
  `RESULTS_DIR/<config>/subject_<SSS>/ratio_<r>_seed_<n>.json`.
- [x] 2.3 Escribir `ensure_run(...)`: si `run_path(...)` ya existe, lo carga y
  lo reusa (log "ya existe, se omite"); si no, corre `run_single` y escribe
  el JSON (incluye seed, ratio, n_aug, y las métricas/report).
- [x] 2.4 Escribir el sentinel `MISSING_CHECKPOINT` (archivo vacío) en
  `RESULTS_DIR/<config>/subject_<SSS>/` cuando el checkpoint no exista, y
  saltear esa (config, sujeto) sin correr ninguna seed/ratio.

## 3. Resumen agregado

- [x] 3.1 Escribir `summarize_subject(config, subject)`: recorre con `glob`
  los `.json` de `RESULTS_DIR/<config>/subject_<SSS>/`, agrupa por `ratio`,
  calcula media/std de accuracy/f1/f1_macro/auc y `n_runs` (cantidad de
  archivos encontrados, puede ser <10), y escribe
  `RESULTS_DIR/<config>/subject_<SSS>_summary.csv` desde cero (no lee ni
  mergea un summary previo).
- [x] 3.2 Si existe el sentinel `MISSING_CHECKPOINT` para esa (config,
  sujeto), `summarize_subject` escribe una única fila con
  `status=missing_checkpoint` y métricas `NaN` en vez de filas por ratio.
- [x] 3.3 Confirmar que `summarize_subject` nunca lee el CSV de
  `train_eegnet_augmentation.py` ni escribe en `ablation_results/` /
  `comparison_plots/` (namespaces separados).

## 4. Orquestación por sujeto y CLI

- [x] 4.1 Escribir `run_subject(subject, device)`: carga train/test del
  sujeto (igual que `train_eegnet_augmentation.py:204-213`), corre el
  baseline `none` (ratio 0.0) una sola vez con las 10 seeds vía `ensure_run`,
  y luego itera las 4 configs con augmentation × 5 ratios × 10 seeds,
  llamando `summarize_subject` al final de cada config para ese sujeto.
- [x] 4.2 CLI en `if __name__ == '__main__'`: `sys.argv[1:]` como lista de
  sujetos (default `1..10`), mismo patrón que `train_eegnet_augmentation.py`.
- [x] 4.3 Aislar excepciones por sujeto (try/except con aviso y `continue`),
  igual que el resto de los scripts del repo, para que un sujeto que falle
  no mate la corrida completa.

## 5. Verificación

- [x] 5.1 Self-check (`_selfcheck()` o `assert`s en `__main__` antes de
  correr): valida que `run_path()` genera nombres determinísticos y que
  `summarize_subject` sobre un directorio con 3 `.json` de prueba
  (accuracy=1.0, 0.5, 0.5, por ejemplo) calcula la media/std esperada y
  `n_runs=3`, sin leer nada de disco fuera del directorio de prueba.
- [ ] 5.2 Correr `python eeg_net_aug/train_eegnet_augmentation_v2.py <un sujeto con
  checkpoints disponibles>` end-to-end y confirmar que se generan los `.json`
  por corrida y el `_summary.csv` esperado.
- [ ] 5.3 Confirmar el caso de checkpoint faltante: correr para un sujeto sin
  alguno de los 4 checkpoints y verificar el sentinel + la fila
  `missing_checkpoint` en el summary de esa config.
