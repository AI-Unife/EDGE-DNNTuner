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

TOTAL_EVALS="${TOTAL_EVALS:-1000}"
CHUNK_EVALS="${CHUNK_EVALS:-50}"
EPOCHS="${EPOCHS:-100}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
# aftercorr lets array task N in chunk K+1 start as soon as array task N in
# chunk K succeeds. Use afterok to make each chunk wait for the whole previous
# array.
DEPENDENCY_TYPE="${DEPENDENCY_TYPE:-aftercorr}"

RESULTS_DIR="${RESULTS_DIR:-results_BANANAS_resume}"
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

if (( TOTAL_EVALS < 1 )); then
  echo "Error: TOTAL_EVALS must be >= 1." >&2
  exit 1
fi
if (( CHUNK_EVALS < 1 )); then
  echo "Error: CHUNK_EVALS must be >= 1." >&2
  exit 1
fi
case "$DEPENDENCY_TYPE" in
  aftercorr|afterok)
    ;;
  *)
    echo "Error: DEPENDENCY_TYPE must be aftercorr or afterok." >&2
    exit 1
    ;;
esac

CHUNKS=$(((TOTAL_EVALS + CHUNK_EVALS - 1) / CHUNK_EVALS))
TASK_COUNT=$((${#DATASETS[@]} * ${#SEEDS[@]}))

if (( TASK_COUNT == 0 )); then
  echo "Error: empty DATASETS_LIST or SEEDS_LIST." >&2
  exit 1
fi

if [[ "${BANANAS_CHAIN_WORKER:-0}" == "1" ]]; then
  TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required in worker mode}"
  CHUNK="${BANANAS_CHAIN_CHUNK:?BANANAS_CHAIN_CHUNK is required in worker mode}"
  DATASET_INDEX=$((TASK_ID / ${#SEEDS[@]}))
  SEED_INDEX=$((TASK_ID % ${#SEEDS[@]}))

  DATA="${DATASETS[$DATASET_INDEX]}"
  SEED="${SEEDS[$SEED_INDEX]}"
  NAME_EXP="bananas_${DATA}_seed${SEED}_e${TOTAL_EVALS}_ep${EPOCHS}"

  echo "Running BANANAS resume chunk ${CHUNK}/${CHUNKS}"
  echo "Array task ${TASK_ID}/${TASK_COUNT}: dataset=${DATA}, seed=${SEED}"
  echo "Budget: total_evals=${TOTAL_EVALS}, chunk_evals=${CHUNK_EVALS}, epochs=${EPOCHS}"
  echo "Result dir=${RESULTS_DIR}/${NAME_EXP}"

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
    --max_new_evals "$CHUNK_EVALS"
  exit 0
fi

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

echo "Submitting BANANAS resume chains"
echo "Datasets: ${DATASETS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Budget per run: total_evals=${TOTAL_EVALS}, chunk_evals=${CHUNK_EVALS}, epochs=${EPOCHS}"
echo "Chunks per dataset/seed: ${CHUNKS}"
echo "Array tasks per chunk: ${TASK_COUNT}, max parallel tasks per chunk: ${MAX_PARALLEL}"
echo "Dependency between chunk arrays: ${DEPENDENCY_TYPE}"

DATASETS_LIST="${DATASETS[*]}"
SEEDS_LIST="${SEEDS[*]}"
export CONDA_ENV PYTHON_BIN JOB_SETUP PARTITION GPUS MEMORY QOS TIME_LIMIT
export TOTAL_EVALS CHUNK_EVALS EPOCHS MAX_PARALLEL DEPENDENCY_TYPE RESULTS_DIR SLURM_LOG_DIR HF_DATASETS_CACHE
export DATASETS_LIST SEEDS_LIST

SCRIPT_PATH="$0"
previous_array_job=""
for CHUNK in $(seq 1 "$CHUNKS"); do
  sbatch_args=(
    "${base_sbatch_args[@]}"
    --job-name="bananas_c${CHUNK}"
    --export="ALL,BANANAS_CHAIN_WORKER=1,BANANAS_CHAIN_CHUNK=${CHUNK}"
  )
  if [[ -n "$previous_array_job" ]]; then
    sbatch_args+=(--dependency="${DEPENDENCY_TYPE}:${previous_array_job}")
  fi

  job_id=$(sbatch --parsable "${sbatch_args[@]}" "$SCRIPT_PATH")
  echo "Submitted chunk ${CHUNK}/${CHUNKS} as array job ${job_id}"
  previous_array_job="$job_id"
done
