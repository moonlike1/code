#!/bin/sh
set -eu

ROOT_DIR="/ai/0309/cloud"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python3}"
RUNNER="${ROOT_DIR}/leakage_detection_torch/run_segmentation_ablation.py"

ABLATION_IDS="${ABLATION_IDS:-A0 A1 A2 A3 A4 A5}"
SEEDS="${SEEDS:-42 52 62}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/output/ablation_exec}"
TRAIN_PATH="${TRAIN_PATH:-${ROOT_DIR}/full_area_train_v2.h5}"
TEST_PATH="${TEST_PATH:-${ROOT_DIR}/full_area_val_v2.h5}"
NUM_POINTS="${NUM_POINTS:-4096}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-200}"
LR="${LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
VAL_INTERVAL="${VAL_INTERVAL:-5}"
AUGMENT="${AUGMENT:-1}"
AUGMENT_MODE="${AUGMENT_MODE:-strong}"
DEVICE="${DEVICE:-cuda}"
TRAIN_SAMPLER="${TRAIN_SAMPLER:-small_positive}"
SEG_LOSS="${SEG_LOSS:-dice_ce}"
SEG_CLASS_WEIGHT_MODE="${SEG_CLASS_WEIGHT_MODE:-sqrt_inv}"
BEST_METRIC="${BEST_METRIC:-val_global_leak_iou}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

echo "[seg_ablation] python=${PYTHON_BIN}"
echo "[seg_ablation] ablation_ids=${ABLATION_IDS}"
echo "[seg_ablation] seeds=${SEEDS}"
echo "[seg_ablation] output_root=${OUTPUT_ROOT}"
echo "[seg_ablation] best_metric=${BEST_METRIC}"

set -- \
  --ablation_ids ${ABLATION_IDS} \
  --seeds ${SEEDS} \
  --output_root "${OUTPUT_ROOT}" \
  --train_path "${TRAIN_PATH}" \
  --test_path "${TEST_PATH}" \
  --num_points "${NUM_POINTS}" \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --val_interval "${VAL_INTERVAL}" \
  --augment_mode "${AUGMENT_MODE}" \
  --device "${DEVICE}" \
  --train_sampler "${TRAIN_SAMPLER}" \
  --seg_loss "${SEG_LOSS}" \
  --seg_class_weight_mode "${SEG_CLASS_WEIGHT_MODE}" \
  --best_metric "${BEST_METRIC}" \
  --python_bin "${PYTHON_BIN}"

if [ "${AUGMENT}" = "1" ]; then
  set -- "$@" --augment
else
  set -- "$@" --no_augment
fi

if [ "${DRY_RUN}" = "1" ]; then
  set -- "$@" --dry_run
fi

if [ "${SKIP_EXISTING}" = "1" ]; then
  set -- "$@" --skip_existing
fi

cd "${ROOT_DIR}"
exec "${PYTHON_BIN}" "${RUNNER}" "$@"
