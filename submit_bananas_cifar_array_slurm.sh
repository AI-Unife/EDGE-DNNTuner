#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-edge-dnntuner-bananas}"
PYTHON_BIN="${PYTHON_BIN:-conda run --no-capture-output -n ${CONDA_ENV} python}"
JOB_SETUP="${JOB_SETUP:-module load cuda/12.2}"

# Current UNIFE GPU partitions from sinfo:
# gpu_H100, gpu_H100_partitioned, gpu_L40S
# A comma-separated list is accepted, for example:
# PARTITION=gpu_H100,gpu_H100_partitioned
PARTITION="${PARTITION:-gpu_H100_partitioned}"
GPUS="${GPUS:-1}"
MEMORY="${MEMORY:-64G}"
QOS="${QOS:-}"
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"

# Smaller defaults than the full 1000 x 100 run.
EVALS="${EVALS:-50}"
EPOCHS="${EPOCHS:-20}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"

RESULTS_DIR="${RESULTS_DIR:-results_BANANAS_array}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-slurm_logs}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HOME}/.cache/huggingface/datasets}"

read -r -a DATASETS <<< "${DATASETS_LIST:-cifar10 cifar100}"
read -r -a SEEDS <<< "${SEEDS_LIST:-42 123 96 7 84}"

if [[ ! -f bananas_runner.py ]]; then
  echo "Error: bananas_runner.py not found. Run this script from the EDGE-DNNTuner repo root." >&2
  exit 1
fi

validate_partitions() {
  local part
  IFS=',' read -r -a requested_partitions <<< "$PARTITION"
  for part in "${requested_partitions[@]}"; do
    case "$part" in
      gpu_H100|gpu_H100_partitioned|gpu_L40S)
        ;;
      cpu_amd_default|cpu_amd_zen4|cpu_amd_zen4_2x96|cpu_amd_zen4_2x48)
        echo "Error: PARTITION contains CPU partition '$part', but BANANAS training requires GPU." >&2
        exit 1
        ;;
      *)
        echo "Error: unknown partition '$part'." >&2
        echo "Valid GPU partitions: gpu_H100, gpu_H100_partitioned, gpu_L40S." >&2
        exit 1
        ;;
    esac
  done
}

validate_partitions
mkdir -p "$RESULTS_DIR" "$SLURM_LOG_DIR"

TASK_COUNT=$((${#DATASETS[@]} * ${#SEEDS[@]}))
if (( TASK_COUNT == 0 )); then
  echo "Error: empty DATASETS_LIST or SEEDS_LIST." >&2
  exit 1
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  ARRAY_SPEC="0-$((TASK_COUNT - 1))%${MAX_PARALLEL}"
  SBATCH_ARGS=(
    --job-name="bananas_array"
    --partition="$PARTITION"
    --array="$ARRAY_SPEC"
    --gres="gpu:${GPUS}"
    --mem="$MEMORY"
    --time="$TIME_LIMIT"
    --output="${SLURM_LOG_DIR}/%x_%A_%a.out"
    --error="${SLURM_LOG_DIR}/%x_%A_%a.err"
  )
  if [[ -n "$QOS" ]]; then
    SBATCH_ARGS+=(--qos="$QOS")
  fi

  echo "Submitting ${TASK_COUNT} BANANAS tasks as array ${ARRAY_SPEC}"
  echo "Datasets: ${DATASETS[*]}"
  echo "Seeds: ${SEEDS[*]}"
  echo "Budget per task: evals=${EVALS}, epochs=${EPOCHS}"
  sbatch "${SBATCH_ARGS[@]}" "$0"
  exit 0
fi

TASK_ID="$SLURM_ARRAY_TASK_ID"
DATASET_INDEX=$((TASK_ID / ${#SEEDS[@]}))
SEED_INDEX=$((TASK_ID % ${#SEEDS[@]}))

DATA="${DATASETS[$DATASET_INDEX]}"
SEED="${SEEDS[$SEED_INDEX]}"
NAME_EXP="bananas_${DATA}_seed${SEED}_e${EVALS}_ep${EPOCHS}"

echo "Running BANANAS array task ${TASK_ID}/${TASK_COUNT}"
echo "Dataset=${DATA} Seed=${SEED} Evals=${EVALS} Epochs=${EPOCHS}"
echo "Result dir=${RESULTS_DIR}/${NAME_EXP}"

export HF_DATASETS_CACHE

if [[ -n "$JOB_SETUP" ]]; then
  eval "$JOB_SETUP"
fi

${PYTHON_BIN} bananas_runner.py \
  --name "${RESULTS_DIR}/${NAME_EXP}" \
  --dataset "$DATA" \
  --seed "$SEED" \
  --eval "$EVALS" \
  --epochs "$EPOCHS" \
  --mod_list flops_module
