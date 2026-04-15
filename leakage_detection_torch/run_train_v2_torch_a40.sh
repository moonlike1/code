#!/bin/sh
set -eu

ROOT_DIR="/ai/0309/cloud"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python3}"
TRAIN_SCRIPT="${ROOT_DIR}/leakage_detection_torch/train_v2_torch.py"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/output/v2_training_torch_dataset_v2}"

TRAIN_PATH="${TRAIN_PATH:-/ai/0309/cloud/full_area_train_v2.h5}"
TEST_PATH="${TEST_PATH:-/ai/0309/cloud/full_area_val_v2.h5}"
NUM_POINTS="${NUM_POINTS:-4096}"
AUGMENT="${AUGMENT:-1}"
AUGMENT_MODE="${AUGMENT_MODE:-strong}"
IN_CHANNELS="${IN_CHANNELS:-4}"
SEG_CLASSES="${SEG_CLASSES:-2}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-300}"
LR="${LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
TRAIN_SAMPLER="${TRAIN_SAMPLER:-shuffle}"
SAMPLER_ALPHA="${SAMPLER_ALPHA:-0.5}"
SAMPLER_MIN_POSITIVE_RATIO="${SAMPLER_MIN_POSITIVE_RATIO:-0.005}"
SAMPLER_MAX_WEIGHT="${SAMPLER_MAX_WEIGHT:-4.0}"
SAMPLER_EMPTY_WEIGHT="${SAMPLER_EMPTY_WEIGHT:-1.0}"
SEG_LOSS="${SEG_LOSS:-dice_ce}"
DICE_WEIGHT="${DICE_WEIGHT:-1.0}"
DICE_SMOOTH="${DICE_SMOOTH:-1.0}"
DICE_TARGET="${DICE_TARGET:-leak}"
BOUNDARY_INPUT_MODE="${BOUNDARY_INPUT_MODE:-features_fine_probs}"
K_SCALES="${K_SCALES:-10 20 40}"
USE_NOISE_GUIDANCE="${USE_NOISE_GUIDANCE:-1}"
USE_PROGRESSIVE="${USE_PROGRESSIVE:-1}"
USE_NOISE_LEAK_CORR="${USE_NOISE_LEAK_CORR:-1}"
USE_MULTI_SCALE="${USE_MULTI_SCALE:-1}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
VAL_INTERVAL="${VAL_INTERVAL:-5}"
BEST_METRIC="${BEST_METRIC:-val_iou}"
SCHEDULER="${SCHEDULER:-none}"
MIN_LR="${MIN_LR:-1e-5}"
SCHEDULER_MILESTONES="${SCHEDULER_MILESTONES:-0.6 0.8}"
SCHEDULER_GAMMA="${SCHEDULER_GAMMA:-0.1}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-0}"
WARMUP_START_FACTOR="${WARMUP_START_FACTOR:-0.2}"
TOPK_CHECKPOINTS="${TOPK_CHECKPOINTS:-3}"
SEG_USE_CLASS_WEIGHTS="${SEG_USE_CLASS_WEIGHTS:-0}"
SEG_CLASS_WEIGHT_MODE="${SEG_CLASS_WEIGHT_MODE:-sqrt_inv}"
SEG_CLASS_WEIGHT_BETA="${SEG_CLASS_WEIGHT_BETA:-0.999999}"
SEG_ONLY="${SEG_ONLY:-0}"

export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

echo "[train_v2_torch] root_dir=${ROOT_DIR}"
echo "[train_v2_torch] python=${PYTHON_BIN}"
echo "[train_v2_torch] output_dir=${OUTPUT_DIR}"
echo "[train_v2_torch] train_path=${TRAIN_PATH}"
echo "[train_v2_torch] test_path=${TEST_PATH}"
echo "[train_v2_torch] train_sampler=${TRAIN_SAMPLER} seg_use_class_weights=${SEG_USE_CLASS_WEIGHTS}"
echo "[train_v2_torch] k_scales=${K_SCALES} use_noise_guidance=${USE_NOISE_GUIDANCE} use_progressive=${USE_PROGRESSIVE} use_noise_leak_corr=${USE_NOISE_LEAK_CORR} use_multi_scale=${USE_MULTI_SCALE}"
echo "[train_v2_torch] scheduler=${SCHEDULER} min_lr=${MIN_LR} warmup_epochs=${WARMUP_EPOCHS} topk_checkpoints=${TOPK_CHECKPOINTS}"

cd "${ROOT_DIR}"
mkdir -p "${OUTPUT_DIR}"

set -- \
  --train_path "${TRAIN_PATH}" \
  --test_path "${TEST_PATH}" \
  --num_points "${NUM_POINTS}" \
  --augment_mode "${AUGMENT_MODE}" \
  --in_channels "${IN_CHANNELS}" \
  --seg_classes "${SEG_CLASSES}" \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --train_sampler "${TRAIN_SAMPLER}" \
  --sampler_alpha "${SAMPLER_ALPHA}" \
  --sampler_min_positive_ratio "${SAMPLER_MIN_POSITIVE_RATIO}" \
  --sampler_max_weight "${SAMPLER_MAX_WEIGHT}" \
  --sampler_empty_weight "${SAMPLER_EMPTY_WEIGHT}" \
  --seg_loss "${SEG_LOSS}" \
  --dice_weight "${DICE_WEIGHT}" \
  --dice_smooth "${DICE_SMOOTH}" \
  --dice_target "${DICE_TARGET}" \
  --boundary_input_mode "${BOUNDARY_INPUT_MODE}" \
  --output_dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --val_interval "${VAL_INTERVAL}" \
  --best_metric "${BEST_METRIC}" \
  --scheduler "${SCHEDULER}" \
  --min_lr "${MIN_LR}" \
  --scheduler_gamma "${SCHEDULER_GAMMA}" \
  --warmup_epochs "${WARMUP_EPOCHS}" \
  --warmup_start_factor "${WARMUP_START_FACTOR}" \
  --topk_checkpoints "${TOPK_CHECKPOINTS}"

set -- "$@" --k_scales
for k in ${K_SCALES}; do
  set -- "$@" "$k"
done

set -- "$@" --scheduler_milestones
for milestone in ${SCHEDULER_MILESTONES}; do
  set -- "$@" "$milestone"
done

if [ "${USE_NOISE_GUIDANCE}" = "1" ]; then
  set -- "$@" --use_noise_guidance
else
  set -- "$@" --no_noise_guidance
fi

if [ "${USE_PROGRESSIVE}" = "1" ]; then
  set -- "$@" --use_progressive
else
  set -- "$@" --no_progressive
fi

if [ "${USE_NOISE_LEAK_CORR}" = "1" ]; then
  set -- "$@" --use_noise_leak_corr
else
  set -- "$@" --no_noise_leak_corr
fi

if [ "${USE_MULTI_SCALE}" = "1" ]; then
  set -- "$@" --use_multi_scale
else
  set -- "$@" --no_multi_scale
fi

if [ "${AUGMENT}" = "1" ]; then
  set -- "$@" --augment
fi

if [ "${SEG_USE_CLASS_WEIGHTS}" = "1" ]; then
  set -- "$@" --seg_use_class_weights \
    --seg_class_weight_mode "${SEG_CLASS_WEIGHT_MODE}" \
    --seg_class_weight_beta "${SEG_CLASS_WEIGHT_BETA}"
fi

if [ "${SEG_ONLY}" = "1" ]; then
  set -- "$@" --seg_only
fi

exec "${PYTHON_BIN}" "${TRAIN_SCRIPT}" "$@"
