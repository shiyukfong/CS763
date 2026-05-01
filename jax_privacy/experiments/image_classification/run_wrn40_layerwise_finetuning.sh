#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "${LOG_DIR}"

# "${PYTHON_BIN}" "${ROOT_DIR}/configs/generate_wrn40_finetune_configs.py"

CONFIGS=(
  "configs/cifar10_wrn_40_4_eps1_finetune_head_only.py"
  "configs/cifar10_wrn_40_4_eps1_finetune_last4.py"
  "configs/cifar10_wrn_40_4_eps1_finetune_last13.py"
  "configs/cifar10_wrn_40_4_eps1_finetune_last26.py"
  "configs/cifar10_wrn_40_4_eps1_finetune_full.py"
  "configs/cifar10_wrn_40_4_eps1_finetune_head_only_nodp.py"
  "configs/cifar10_wrn_40_4_eps1_finetune_last4_nodp.py"
  "configs/cifar10_wrn_40_4_eps1_finetune_last13_nodp.py"
  "configs/cifar10_wrn_40_4_eps1_finetune_last26_nodp.py"
  "configs/cifar10_wrn_40_4_eps1_finetune_full_nodp.py"
)
for cfg in "${CONFIGS[@]}"; do
  run_name="$(basename "${cfg}" .py)"
  timestamp="$(date +%Y%m%d_%H%M%S)"
  log_path="${LOG_DIR}/${run_name}_${timestamp}.log"
  echo "Running ${run_name}..."
  (
    cd "${ROOT_DIR}"
    "${PYTHON_BIN}" run_experiment.py \
      --config="${cfg}" \
      --jaxline_mode=train_eval_multithreaded
  ) 2>&1 | tee "${log_path}"
done
