#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

export TRAIN_PATH="${TRAIN_PATH:-/ai/0309/cloud/full_area_train_v2.h5}"
export TEST_PATH="${TEST_PATH:-/ai/0309/cloud/full_area_val_v2.h5}"
export OUTPUT_DIR="${OUTPUT_DIR:-/ai/0309/cloud/output/seg_recall}"
export BOUNDARY_INPUT_MODE="${BOUNDARY_INPUT_MODE:-features_fine_probs}"
export SEG_ONLY="${SEG_ONLY:-1}"

export TRAIN_SAMPLER="${TRAIN_SAMPLER:-small_positive}"
export SAMPLER_ALPHA="${SAMPLER_ALPHA:-0.8}"
export SAMPLER_MIN_POSITIVE_RATIO="${SAMPLER_MIN_POSITIVE_RATIO:-0.003}"
export SAMPLER_MAX_WEIGHT="${SAMPLER_MAX_WEIGHT:-6.0}"
export SAMPLER_EMPTY_WEIGHT="${SAMPLER_EMPTY_WEIGHT:-0.8}"

export SEG_USE_CLASS_WEIGHTS="${SEG_USE_CLASS_WEIGHTS:-1}"
export SEG_CLASS_WEIGHT_MODE="${SEG_CLASS_WEIGHT_MODE:-effective_num}"
export SEG_CLASS_WEIGHT_BETA="${SEG_CLASS_WEIGHT_BETA:-0.999999}"

exec sh "${SCRIPT_DIR}/run_train_v2_torch_a40.sh"
