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
TIME_LIMIT="${TIME_LIMIT:-08:00:00}"

TOTAL_EVALS="${TOTAL_EVALS:-1000}"
CHUNK_EVALS="${CHUNK_EVALS:-25}"
EPOCHS="${EPOCHS:-100}"
MAX_PARALLEL="${MAX_PARALLEL:-10}"

RESULTS_DIR="${RESULTS_DIR:-results_BANANAS_rolling}"
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

history_count() {
  local history_path="$1"
  if [[ ! -f "$history_path" ]]; then
    echo 0
    return
  fi
  local lines
  lines=$(wc -l < "$history_path")
  if (( lines <= 1 )); then
    echo 0
  else
    echo $((lines - 1))
  fi
}

validate_partitions
mkdir -p "$RESULTS_DIR" "$SLURM_LOG_DIR"

if (( TOTAL_EVALS < 1 )); then
  echo "Error: TOTAL_EVALS must be >= 1." >&2
  exit 1
fi
if (( CHUNK_EVALS < 1 )); then
  echo "Error: CHUNK_EVALS must be >= 1." >&2
  exit 1
fi

TASK_COUNT=$((${#DATASETS[@]} * ${#SEEDS[@]}))
if (( TASK_COUNT == 0 )); then
  echo "Error: empty DATASETS_LIST or SEEDS_LIST." >&2
  exit 1
fi

CHUNKS=$(((TOTAL_EVALS + CHUNK_EVALS - 1) / CHUNK_EVALS))

if [[ "${BANANAS_ROLLING_WORKER:-0}" == "1" ]]; then
  TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required in worker mode}"
  CHUNK="${BANANAS_ROLLING_CHUNK:?BANANAS_ROLLING_CHUNK is required in worker mode}"

  DATASET_INDEX=$((TASK_ID / ${#SEEDS[@]}))
  SEED_INDEX=$((TASK_ID % ${#SEEDS[@]}))

  DATA="${DATASETS[$DATASET_INDEX]}"
  SEED="${SEEDS[$SEED_INDEX]}"
  NAME_EXP="bananas_${DATA}_seed${SEED}_e${TOTAL_EVALS}_ep${EPOCHS}"
  HISTORY_PATH="${RESULTS_DIR}/${NAME_EXP}/algorithm_logs/bananas_history.csv"
  COMPLETED_BEFORE=$(history_count "$HISTORY_PATH")

  echo "Running rolling chunk ${CHUNK}/${CHUNKS}"
  echo "Array task ${TASK_ID}/${TASK_COUNT}: dataset=${DATA}, seed=${SEED}"
  echo "Completed before chunk: ${COMPLETED_BEFORE}/${TOTAL_EVALS}"
  echo "Budget: chunk_evals=${CHUNK_EVALS}, epochs=${EPOCHS}"
  echo "Result dir=${RESULTS_DIR}/${NAME_EXP}"

  CHUNK_TARGET=$((CHUNK * CHUNK_EVALS))
  if (( CHUNK_TARGET > TOTAL_EVALS )); then
    CHUNK_TARGET="$TOTAL_EVALS"
  fi
  REMAINING_FOR_CHUNK=$((CHUNK_TARGET - COMPLETED_BEFORE))

  if (( COMPLETED_BEFORE >= TOTAL_EVALS )); then
    echo "Run already complete. Nothing to do."
    exit 0
  fi
  if (( REMAINING_FOR_CHUNK <= 0 )); then
    echo "Chunk target already reached: ${COMPLETED_BEFORE}/${CHUNK_TARGET}. Nothing to do."
    exit 0
  fi
  if (( REMAINING_FOR_CHUNK > CHUNK_EVALS )); then
    REMAINING_FOR_CHUNK="$CHUNK_EVALS"
  fi
  echo "This task will run ${REMAINING_FOR_CHUNK} new eval(s) to reach chunk target ${CHUNK_TARGET}/${TOTAL_EVALS}."

  export HF_DATASETS_CACHE

  if [[ -n "$JOB_SETUP" ]]; then
    eval "$JOB_SETUP"
  fi

  ${PYTHON_BIN} bananas_runner.py \
    --name "${RESULTS_DIR}/${NAME_EXP}" \
    --dataset "$DATA" \
    --seed "$SEED" \
    --eval "$TOTAL_EVALS" \
    --epochs "$EPOCHS" \
    --mod_list flops_module \
    --resume \
    --max_new_evals "$REMAINING_FOR_CHUNK"

  COMPLETED_AFTER=$(history_count "$HISTORY_PATH")
  echo "Completed after chunk: ${COMPLETED_AFTER}/${TOTAL_EVALS}"
  exit 0
fi

DATASETS_LIST="${DATASETS[*]}"
SEEDS_LIST="${SEEDS[*]}"
export CONDA_ENV PYTHON_BIN JOB_SETUP PARTITION GPUS MEMORY QOS TIME_LIMIT
export TOTAL_EVALS CHUNK_EVALS EPOCHS MAX_PARALLEL RESULTS_DIR SLURM_LOG_DIR HF_DATASETS_CACHE
export DATASETS_LIST SEEDS_LIST

base_sbatch_args=(
  --partition="$PARTITION"
  --array="0-$((TASK_COUNT - 1))%${MAX_PARALLEL}"
  --gres="gpu:${GPUS}"
  --mem="$MEMORY"
  --time="$TIME_LIMIT"
  --output="${SLURM_LOG_DIR}/%x_%A_%a.out"
  --error="${SLURM_LOG_DIR}/%x_%A_%a.err"
)
if [[ -n "$QOS" ]]; then
  base_sbatch_args+=(--qos="$QOS")
fi

echo "Starting rolling BANANAS array submission"
echo "Datasets: ${DATASETS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Array tasks per chunk: ${TASK_COUNT}, max parallel tasks: ${MAX_PARALLEL}"
echo "Budget per run: total_evals=${TOTAL_EVALS}, chunk_evals=${CHUNK_EVALS}, epochs=${EPOCHS}"
echo "Chunks: ${CHUNKS}"
echo "Only one chunk array is submitted at a time."

SCRIPT_PATH="$0"
for CHUNK in $(seq 1 "$CHUNKS"); do
  echo "Submitting chunk ${CHUNK}/${CHUNKS}"
  set +e
  sbatch --wait \
    "${base_sbatch_args[@]}" \
    --job-name="bananas_roll_c${CHUNK}" \
    --export="ALL,BANANAS_ROLLING_WORKER=1,BANANAS_ROLLING_CHUNK=${CHUNK}" \
    "$SCRIPT_PATH"
  status=$?
  set -e

  if (( status != 0 )); then
    echo "Error: chunk ${CHUNK}/${CHUNKS} failed with sbatch status ${status}." >&2
    echo "Fix the failed task, then rerun this script with the same RESULTS_DIR to resume." >&2
    exit "$status"
  fi

  echo "Chunk ${CHUNK}/${CHUNKS} completed."
done

echo "All rolling chunks completed."
