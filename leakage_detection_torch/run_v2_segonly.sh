#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

export TRAIN_PATH="${TRAIN_PATH:-/ai/0309/cloud/full_area_train_v2.h5}"
export TEST_PATH="${TEST_PATH:-/ai/0309/cloud/full_area_val_v2.h5}"
export OUTPUT_DIR="${OUTPUT_DIR:-/ai/0309/cloud/output/v2_training_torch_dataset_v2_segonly}"
export BOUNDARY_INPUT_MODE="${BOUNDARY_INPUT_MODE:-features_fine_probs}"
export SEG_ONLY="${SEG_ONLY:-1}"

exec sh "${SCRIPT_DIR}/run_train_v2_torch_a40.sh"
