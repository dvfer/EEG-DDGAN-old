#!/usr/bin/env bash
# Genera un pool sintético de EEG-GAN vanilla (rama `main`, AE-coupled) sizeado
# contra TRAIN (no contra test, a diferencia de train_eeggan_vanilla.sh, que lo
# genera sizeado a test solo para la tabla de ablation) -- lo necesita
# train_eegnet_augmentation.py para poder armar ratios de aumento de hasta 0.5
# de train. Requiere que train_eeggan_vanilla.sh ya haya entrenado ese sujeto
# (no entrena nada acá, solo genera).
#
# Cap de 200 trials por condición (misma convención que
# compare_samples.MAX_SAMPLES_PER_COND) para no generar pools gigantes.
#
# Uso:
#   ./generate_eeggan_vanilla_augpool.sh [subject_id] [target] [dataset] [out_csv] [n_target]
#   # defaults: 1 full BNCI2014_009 <generated_samples/${GAN_NAME}_augpool.csv> 0
#   # n_target > 0: genera EXACTAMENTE ese número de trials Target y NINGÚN
#   # NonTarget (lo único que consume sample_pool()); 0 = comportamiento viejo
#   # (trials reales por condición, capeado a MAX_PER_COND).

set -euo pipefail

SUBJECT="${1:-1}"
SUBJECT_FMT=$(printf "%03d" "$SUBJECT")
TARGET="${2:-full}"
DATASET="${3:-BNCI2014_009}"
MAX_PER_COND=200

case "$DATASET" in
    BNCI2014_009) DIR_SUFFIX="" ;;
    BNCI2014_008) DIR_SUFFIX="_008" ;;
    *) echo "Dataset desconocido: $DATASET (usar BNCI2014_009 o BNCI2014_008)" >&2; exit 1 ;;
esac
DATASET_CODE="${DATASET##*_}"

WORKTREE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../EEG-DDGAN-main-vanilla"

# Nombre con código de dataset (el que produce train_eeggan_vanilla.sh hoy);
# si no está, se cae al nombre viejo sin código (checkpoints 009 previos).
GAN_NAME="EEG_GAN_vanilla_${DATASET_CODE}_${TARGET}_s${SUBJECT_FMT}"
if [[ ! -f "${WORKTREE_DIR}/trained_models/${GAN_NAME}.pt" ]]; then
    GAN_NAME="EEG_GAN_vanilla_${TARGET}_s${SUBJECT_FMT}"
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
TRAIN_CSV="${REPO_ROOT}/subject_data/train${DIR_SUFFIX}/subject_${SUBJECT_FMT}.csv"
OUT_CSV="${4:-${REPO_ROOT}/generated_samples/${GAN_NAME}_augpool.csv}"
[[ "$OUT_CSV" = /* ]] || OUT_CSV="${REPO_ROOT}/${OUT_CSV}"   # el script hace cd al worktree

if [[ ! -f "$TRAIN_CSV" ]]; then
    echo "No existe $TRAIN_CSV -- corré moabb_pipeline.py --dataset $DATASET --subjects $SUBJECT primero." >&2
    exit 1
fi
if [[ ! -d "$WORKTREE_DIR" ]]; then
    echo "No existe $WORKTREE_DIR -- corré ./train_eeggan_vanilla.sh $SUBJECT $TARGET $DATASET primero." >&2
    exit 1
fi

cd "$WORKTREE_DIR"
PY=".venv/bin/python"
GAN_CKPT="trained_models/${GAN_NAME}.pt"
if [[ ! -f "$GAN_CKPT" ]]; then
    echo "No existe $GAN_CKPT -- corré ./train_eeggan_vanilla.sh $SUBJECT $TARGET $DATASET primero." >&2
    exit 1
fi

N_TARGET_ARG="${5:-0}"
if [[ "$N_TARGET_ARG" -gt 0 ]]; then
    N_TARGET="$N_TARGET_ARG"
    N_NONTARGET=0
else
    N_TARGET=$("$PY" -c "import pandas as pd; df=pd.read_csv('$TRAIN_CSV'); print(min(df[df.Condition==1]['Trial'].nunique(), $MAX_PER_COND))")
    N_NONTARGET=$("$PY" -c "import pandas as pd; df=pd.read_csv('$TRAIN_CSV'); print(min(df[df.Condition==0]['Trial'].nunique(), $MAX_PER_COND))")
fi
echo "Generando pool: Target=${N_TARGET}, NonTarget=${N_NONTARGET}"

mkdir -p generated_samples
rm -f generated_samples/_tmp_target.csv generated_samples/_tmp_nontarget.csv  # restos de una corrida abortada
for cond_pair in "1 target $N_TARGET" "0 nontarget $N_NONTARGET"; do
    read -r cond name n <<< "$cond_pair"
    if [[ "$n" -eq 0 ]]; then continue; fi   # if, no '&&': con set -e un && falso aborta el script
    "$PY" - <<PYEOF
import functools, torch
torch.load = functools.partial(torch.load, weights_only=False)

import eeggan.helpers.initialize_gan as _ig
_orig_init_gan = _ig.init_gan
def _init_gan_2tuple(*a, **kw):
    result = _orig_init_gan(*a, **kw)
    return result[:2] if len(result) > 2 else result
_ig.init_gan = _init_gan_2tuple

from eeggan.generate_samples_main import main
main([
    "model=$GAN_CKPT",
    "save_name=_tmp_${name}.csv",
    "conditions=$cond",
    "num_samples_total=$n",
    "num_samples_parallel=$n",
    "sequence_length=-1",
])
PYEOF
done

mkdir -p "$(dirname "$OUT_CSV")"
"$PY" -c "
import pandas as pd
import os
parts = [pd.read_csv(f) for f in ['generated_samples/_tmp_target.csv', 'generated_samples/_tmp_nontarget.csv'] if os.path.exists(f)]
df = pd.concat(parts, ignore_index=True)
df.to_csv('$OUT_CSV', index=False)
print(f'  Guardado: $OUT_CSV ({len(df)} filas)')
"
rm -f generated_samples/_tmp_target.csv generated_samples/_tmp_nontarget.csv
