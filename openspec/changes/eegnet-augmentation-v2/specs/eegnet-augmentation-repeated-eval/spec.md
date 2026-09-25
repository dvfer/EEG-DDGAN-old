## ADDED Requirements

### Requirement: Corridas repetidas por-archivo
El sistema SHALL entrenar y evaluar EEGNet `N_REPEATS=10` veces (seeds
literales 1 a 10) para cada combinación (config, sujeto, ratio), y SHALL
escribir el `classification_report` de cada corrida individual en su propio
archivo, identificado de forma determinística por (config, sujeto, ratio,
seed).

#### Scenario: Corrida nueva
- **WHEN** no existe todavía el archivo de resultado para (config, sujeto,
  ratio, seed)
- **THEN** el sistema entrena EEGNet con esa semilla y ese ratio de aumento y
  escribe un archivo nuevo con el `classification_report` y las métricas
  escalares (accuracy, f1, f1_macro, auc) de esa corrida

#### Scenario: Corrida ya existente (resume)
- **WHEN** ya existe el archivo de resultado para (config, sujeto, ratio,
  seed)
- **THEN** el sistema NO reentrena esa combinación y la reusa tal cual para
  el resumen

### Requirement: Sin merge de esquemas incompatibles
El sistema NUNCA SHALL combinar en un mismo archivo de resumen corridas
generadas con un número distinto de repeticiones o un esquema de métricas
distinto al esperado. El resumen agregado SHALL recalcularse por completo a
partir de los archivos de corrida presentes en disco, nunca por edición
incremental de un resumen previo.

#### Scenario: Regeneración de resumen
- **WHEN** se solicita el resumen (media/std) de un (config, sujeto)
- **THEN** el sistema lee todos los archivos de corrida presentes para esa
  combinación, calcula media y std sobre ellos, y escribe el resumen desde
  cero (sobreescribiendo cualquier resumen previo de esa misma combinación)

#### Scenario: Menos corridas que las esperadas
- **WHEN** existen menos de 10 archivos de corrida para un (config, sujeto,
  ratio) al momento de generar el resumen
- **THEN** el resumen se calcula igual sobre las corridas disponibles y
  reporta explícitamente cuántas corridas (`n_runs`) se usaron, sin fallar

### Requirement: Configs de generador soportadas
El sistema SHALL soportar las configs `fm_lambda_0`, `fm_lambda_50_postnet`,
`eeg_gan_vanilla_full`, `tts_gan_baseline` (con aumento sintético) y `none`
(sin aumento, baseline), resolviendo el checkpoint de cada una según su
fuente correspondiente (ablation study, entrenamiento propio de esta rama, o
worktree `main`).

#### Scenario: Baseline sin aumento compartido
- **WHEN** se evalúa la config `none` para un sujeto y ratio 0.0
- **THEN** el sistema la computa una única vez por sujeto, y ese resultado no
  se recalcula por separado para cada una de las otras configs

#### Scenario: Ratios de aumento
- **WHEN** se evalúa una config con aumento para un sujeto
- **THEN** el sistema corre esa config para cada ratio en
  `{0.1, 0.2, 0.3, 0.4, 0.5}`, aumentando solo la clase Target según
  `n_aug = round(ratio * n_trials_target_train)`

### Requirement: Checkpoint faltante explícito
Si el checkpoint requerido por una config no existe para un sujeto dado, el
sistema SHALL saltear esa combinación sin interrumpir el resto de la
ejecución, y SHALL dejar constancia explícita de la ausencia en el resumen
de ese sujeto (no una fila o combinación faltante sin explicación).

#### Scenario: Checkpoint ausente
- **WHEN** el `.pt` requerido por una config no existe para un sujeto
- **THEN** el sistema continúa con las demás combinaciones (config, sujeto)
  y el resumen de ese sujeto incluye una entrada para esa config marcada
  como faltante (checkpoint ausente), con métricas nulas en vez de omitirla
  en silencio
