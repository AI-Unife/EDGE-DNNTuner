#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-edge-dnntuner-bananas}"
PYTHON_BIN="${PYTHON_BIN:-conda run --no-capture-output -n ${CONDA_ENV} python}"
JOB_SETUP="${JOB_SETUP:-}"
# Valid GPU partitions on the UNIFE cluster: gpu_H100, gpu_H100_partitioned, gpu_L40S.
PARTITION="${PARTITION:-gpu_L40S}"
GPUS="${GPUS:-1}"
TIME_LIMIT="${TIME_LIMIT:-1-00:00:00}"
EVALS="${EVALS:-1000}"
EPOCHS="${EPOCHS:-100}"
RESULTS_DIR="${RESULTS_DIR:-results_BANANAS}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-slurm_logs}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HOME}/.cache/huggingface/datasets}"

DATASETS=(cifar10 cifar100)
SEEDS=(42 123 96 7 84)

if [[ ! -f bananas_runner.py ]]; then
  echo "Error: bananas_runner.py not found. Run this script from the EDGE-DNNTuner repo root." >&2
  exit 1
fi

mkdir -p "$RESULTS_DIR" "$SLURM_LOG_DIR"

for DATA in "${DATASETS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    NAME_EXP="bananas_${DATA}_seed${SEED}"
    RUN_CMD="export HF_DATASETS_CACHE=${HF_DATASETS_CACHE}; ${PYTHON_BIN} bananas_runner.py \
        --name ${RESULTS_DIR}/${NAME_EXP} \
        --dataset ${DATA} \
        --seed ${SEED} \
        --eval ${EVALS} \
        --epochs ${EPOCHS} \
        --mod_list flops_module"
    if [[ -n "$JOB_SETUP" ]]; then
      RUN_CMD="${JOB_SETUP} && ${RUN_CMD}"
    fi

    sbatch \
      --job-name="bananas_${DATA}_${SEED}" \
      --partition="$PARTITION" \
      --gres="gpu:${GPUS}" \
      --time="$TIME_LIMIT" \
      --output="${SLURM_LOG_DIR}/%x_%j.out" \
      --error="${SLURM_LOG_DIR}/%x_%j.err" \
      --wrap="$RUN_CMD"
  done
done
