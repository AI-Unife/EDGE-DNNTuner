#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-edge-dnntuner-bananas}"
PYTHON_BIN="${PYTHON_BIN:-conda run --no-capture-output -n ${CONDA_ENV} python}"
JOB_SETUP="${JOB_SETUP:-module load cuda/12.2}"

# Lightweight controller job. It only submits the next GPU array and exits.
CONTROLLER_PARTITION="${CONTROLLER_PARTITION:-cpu_amd_default}"
CONTROLLER_TIME="${CONTROLLER_TIME:-00:10:00}"
CONTROLLER_MEMORY="${CONTROLLER_MEMORY:-1G}"
CONTROLLER_DEPENDENCY="${CONTROLLER_DEPENDENCY:-afterany}"

# GPU worker arrays.
PARTITION="${PARTITION:-gpu_H100_partitioned}"
GPUS="${GPUS:-1}"
MEMORY="${MEMORY:-64G}"
QOS="${QOS:-}"
TIME_LIMIT="${TIME_LIMIT:-08:00:00}"

TOTAL_EVALS="${TOTAL_EVALS:-1000}"
CHUNK_EVALS="${CHUNK_EVALS:-25}"
EPOCHS="${EPOCHS:-100}"
MAX_PARALLEL="${MAX_PARALLEL:-10}"
START_CHUNK="${START_CHUNK:-1}"
MUTATION_PARENTS="${MUTATION_PARENTS:-10}"
MUTATION_ATTEMPTS="${MUTATION_ATTEMPTS:-100}"
RANDOM_CANDIDATE_FRACTION="${RANDOM_CANDIDATE_FRACTION:-0.10}"

RESULTS_DIR="${RESULTS_DIR:-results_BANANAS_controller}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-slurm_logs}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HOME}/.cache/huggingface/datasets}"

read -r -a DATASETS <<< "${DATASETS_LIST:-cifar10 cifar100}"
read -r -a SEEDS <<< "${SEEDS_LIST:-42 123 96 7 84}"

if [[ ! -f bananas_runner.py ]]; then
  echo "Error: bananas_runner.py not found. Run this script from the EDGE-DNNTuner repo root." >&2
  exit 1
fi
if [[ ! -f submit_bananas_cifar_rolling_array_slurm.sh ]]; then
  echo "Error: submit_bananas_cifar_rolling_array_slurm.sh not found." >&2
  exit 1
fi

validate_gpu_partitions() {
  local part
  IFS=',' read -r -a requested_partitions <<< "$PARTITION"
  for part in "${requested_partitions[@]}"; do
    case "$part" in
      gpu_H100|gpu_H100_partitioned|gpu_L40S)
        ;;
      cpu_amd_default|cpu_amd_zen4|cpu_amd_zen4_2x96|cpu_amd_zen4_2x48)
        echo "Error: PARTITION contains CPU partition '$part', but BANANAS workers require GPU." >&2
        exit 1
        ;;
      *)
        echo "Error: unknown GPU partition '$part'." >&2
        exit 1
        ;;
    esac
  done
}

validate_controller_partition() {
  case "$CONTROLLER_PARTITION" in
    cpu_amd_default|cpu_amd_zen4|cpu_amd_zen4_2x96|cpu_amd_zen4_2x48)
      ;;
    *)
      echo "Error: CONTROLLER_PARTITION must be a CPU partition." >&2
      echo "Valid CPU partitions: cpu_amd_default, cpu_amd_zen4, cpu_amd_zen4_2x96, cpu_amd_zen4_2x48." >&2
      exit 1
      ;;
  esac
}

validate_gpu_partitions
validate_controller_partition
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
if (( START_CHUNK < 1 || START_CHUNK > CHUNKS )); then
  echo "Error: START_CHUNK must be in [1, ${CHUNKS}]." >&2
  exit 1
fi

DATASETS_LIST="${DATASETS[*]}"
SEEDS_LIST="${SEEDS[*]}"
export CONDA_ENV PYTHON_BIN JOB_SETUP PARTITION GPUS MEMORY QOS TIME_LIMIT
export TOTAL_EVALS CHUNK_EVALS EPOCHS MAX_PARALLEL RESULTS_DIR SLURM_LOG_DIR HF_DATASETS_CACHE
export DATASETS_LIST SEEDS_LIST CONTROLLER_PARTITION CONTROLLER_TIME CONTROLLER_MEMORY CONTROLLER_DEPENDENCY START_CHUNK
export MUTATION_PARENTS MUTATION_ATTEMPTS RANDOM_CANDIDATE_FRACTION

if [[ "${BANANAS_CONTROLLER_WORKER:-0}" != "1" ]]; then
  controller_job_id=$(sbatch --parsable \
    --job-name="bananas_ctrl_c${START_CHUNK}" \
    --partition="$CONTROLLER_PARTITION" \
    --mem="$CONTROLLER_MEMORY" \
    --time="$CONTROLLER_TIME" \
    --output="${SLURM_LOG_DIR}/%x_%j.out" \
    --error="${SLURM_LOG_DIR}/%x_%j.err" \
    --export="ALL,BANANAS_CONTROLLER_WORKER=1" \
    "$0")

  echo "Submitted BANANAS CPU controller job ${controller_job_id}"
  echo "It will submit one GPU array at a time."
  echo "Chunks: ${CHUNKS}, array tasks per chunk: ${TASK_COUNT}, max parallel GPU tasks: ${MAX_PARALLEL}"
  exit 0
fi

echo "BANANAS CPU controller running chunk ${START_CHUNK}/${CHUNKS}"
echo "Submitting one GPU array with ${TASK_COUNT} tasks and max parallel ${MAX_PARALLEL}"

gpu_sbatch_args=(
  --job-name="bananas_gpu_c${START_CHUNK}"
  --partition="$PARTITION"
  --array="0-$((TASK_COUNT - 1))%${MAX_PARALLEL}"
  --gres="gpu:${GPUS}"
  --mem="$MEMORY"
  --time="$TIME_LIMIT"
  --output="${SLURM_LOG_DIR}/%x_%A_%a.out"
  --error="${SLURM_LOG_DIR}/%x_%A_%a.err"
  --export="ALL,BANANAS_ROLLING_WORKER=1,BANANAS_ROLLING_CHUNK=${START_CHUNK}"
)
if [[ -n "$QOS" ]]; then
  gpu_sbatch_args+=(--qos="$QOS")
fi

gpu_job_id=$(sbatch --parsable "${gpu_sbatch_args[@]}" submit_bananas_cifar_rolling_array_slurm.sh)
echo "Submitted GPU chunk array ${gpu_job_id} for chunk ${START_CHUNK}/${CHUNKS}"

next_chunk=$((START_CHUNK + 1))
if (( next_chunk <= CHUNKS )); then
  next_controller_job_id=$(START_CHUNK="$next_chunk" sbatch --parsable \
    --job-name="bananas_ctrl_c${next_chunk}" \
    --partition="$CONTROLLER_PARTITION" \
    --dependency="${CONTROLLER_DEPENDENCY}:${gpu_job_id}" \
    --mem="$CONTROLLER_MEMORY" \
    --time="$CONTROLLER_TIME" \
    --output="${SLURM_LOG_DIR}/%x_%j.out" \
    --error="${SLURM_LOG_DIR}/%x_%j.err" \
    --export="ALL,BANANAS_CONTROLLER_WORKER=1,START_CHUNK=${next_chunk}" \
    "$0")
  echo "Submitted next CPU controller ${next_controller_job_id}, dependent on GPU array ${gpu_job_id} with ${CONTROLLER_DEPENDENCY}"
else
  echo "Last chunk submitted. No next controller needed."
fi
