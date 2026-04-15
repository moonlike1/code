#!/bin/sh
[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"
set -euo pipefail

ROOT_DIR="/ai/0309/cloud"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_PATH="${DATA_PATH:-${ROOT_DIR}/liquid_leakage_dataset_with_groups.h5}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/classification_feasibility/generalization_runs_mtl}"
GROUP_OUTPUT_DIR="${GROUP_OUTPUT_DIR:-${ROOT_DIR}/classification_feasibility/generalization_groups}"
STRATEGY="${STRATEGY:-aligned_chrono_block}"
GROUP_NUM_BLOCKS="${GROUP_NUM_BLOCKS:-${N_SPLITS:-5}}"
SPLIT_MODE="${SPLIT_MODE:-group_kfold}"
N_SPLITS="${N_SPLITS:-5}"
BEST_METRIC="${BEST_METRIC:-val_cls_macro_f1}"
EPOCHS="${EPOCHS:-300}"
PRINT_ONLY=0

REQUESTED_BASELINES=()

usage() {
  cat <<'EOF'
用法：
  bash classification_feasibility/run_classification_generalization_mtl.sh [--print-only] [BASELINE...]

用途：
  为你自己的多任务分类模型生成更严格的泛化分组标签，并直接跑 grouped CV。

默认：
  - strategy: aligned_chrono_block
  - baseline: B3_MTL_Full
  - split_mode: group_kfold

可选 baseline：
  B0_ClsOnly
  B1_MTL_NoUF
  B2_MTL_NoMS
  B3_MTL_Full

可覆盖环境变量：
  PYTHON_BIN DATA_PATH OUTPUT_ROOT GROUP_OUTPUT_DIR
  STRATEGY GROUP_NUM_BLOCKS SPLIT_MODE N_SPLITS BEST_METRIC EPOCHS

示例：
  bash classification_feasibility/run_classification_generalization_mtl.sh
  bash classification_feasibility/run_classification_generalization_mtl.sh --print-only B3_MTL_Full
  STRATEGY=session bash classification_feasibility/run_classification_generalization_mtl.sh B3_MTL_Full
EOF
}

while (($#)); do
  case "$1" in
    --print-only|--dry-run)
      PRINT_ONLY=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    B0_ClsOnly|B1_MTL_NoUF|B2_MTL_NoMS|B3_MTL_Full)
      REQUESTED_BASELINES+=("$1")
      shift
      ;;
    *)
      echo "未知参数：$1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if ((${#REQUESTED_BASELINES[@]} == 0)); then
  REQUESTED_BASELINES=("B3_MTL_Full")
fi

DATA_STEM="$(basename "${DATA_PATH}")"
DATA_STEM="${DATA_STEM%.*}"
GROUP_LABELS_PATH="${GROUP_LABELS_PATH:-${GROUP_OUTPUT_DIR}/${DATA_STEM}_${STRATEGY}_k${GROUP_NUM_BLOCKS}.json}"

group_cmd=(
  "${PYTHON_BIN}"
  "${ROOT_DIR}/classification_feasibility/build_generalization_group_labels.py"
  --data-path "${DATA_PATH}"
  --strategy "${STRATEGY}"
  --num-blocks "${GROUP_NUM_BLOCKS}"
  --output-json "${GROUP_LABELS_PATH}"
)

echo
echo "============================================================"
echo "Generalization grouping"
echo "Data:          ${DATA_PATH}"
echo "Strategy:      ${STRATEGY}"
echo "Num blocks:    ${GROUP_NUM_BLOCKS}"
echo "Group labels:  ${GROUP_LABELS_PATH}"
echo "Command:"
printf '  %q' "${group_cmd[@]}"
printf '\n'

if ((PRINT_ONLY)); then
  :
else
  mkdir -p "${GROUP_OUTPUT_DIR}"
  (
    cd "${ROOT_DIR}"
    export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
    "${group_cmd[@]}"
  )
fi

export DATA_PATH
export OUTPUT_ROOT
export SPLIT_MODE
export N_SPLITS
export BEST_METRIC
export EPOCHS
export GROUP_LABELS_PATH

exec bash "${ROOT_DIR}/classification_feasibility/run_classification_cv_v2.sh" \
  $( ((PRINT_ONLY)) && printf '%s ' --print-only ) \
  "${REQUESTED_BASELINES[@]}"
