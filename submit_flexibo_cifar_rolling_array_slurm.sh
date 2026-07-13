#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-edge-dnntuner-bananas}"
PYTHON_BIN="${PYTHON_BIN:-conda run --no-capture-output -n ${CONDA_ENV} python}"
JOB_SETUP="${JOB_SETUP:-module load cuda/12.2}"

PARTITION="${PARTITION:-gpu_H100_partitioned}"
GPUS="${GPUS:-1}"
MEMORY="${MEMORY:-64G}"
QOS="${QOS:-}"
TIME_LIMIT="${TIME_LIMIT:-08:00:00}"

TOTAL_EVALS="${TOTAL_EVALS:-1000}"
CHUNK_EVALS="${CHUNK_EVALS:-25}"
EPOCHS="${EPOCHS:-100}"
MAX_PARALLEL="${MAX_PARALLEL:-10}"
SURROGATE="${SURROGATE:-GP}"
BETA="${BETA:-1.0}"
CANDIDATE_POOL="${CANDIDATE_POOL:-512}"
INIT_RANDOM="${INIT_RANDOM:-10}"
ACCURACY_COST="${ACCURACY_COST:-1.0}"
FLOPS_COST="${FLOPS_COST:-0.05}"
FLOPS_SCALE="${FLOPS_SCALE:-}"

RESULTS_DIR="${RESULTS_DIR:-results_FLEXIBO_rolling}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-slurm_logs}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HOME}/.cache/huggingface/datasets}"

read -r -a DATASETS <<< "${DATASETS_LIST:-cifar10 cifar100}"
read -r -a SEEDS <<< "${SEEDS_LIST:-42 123 96 7 84}"

if [[ ! -f flexibo_runner.py ]]; then
  echo "Error: flexibo_runner.py not found. Run this script from the EDGE-DNNTuner repo root." >&2
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
        echo "Error: PARTITION contains CPU partition '$part', but FlexiBO training requires GPU." >&2
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
  awk -F, 'NR > 1 && ($3 == "accuracy" || $3 == "both") {count++} END {print count + 0}' "$history_path"
}

configure_tensorflow_xla() {
  if [[ -n "${XLA_FLAGS:-}" ]]; then
    echo "Using existing XLA_FLAGS=${XLA_FLAGS}"
    return
  fi

  local cuda_data_dir
  cuda_data_dir="$(${PYTHON_BIN} - <<'PY'
from pathlib import Path

try:
    import nvidia.cuda_nvcc as cuda_nvcc
except Exception:
    raise SystemExit(0)

cuda_data_dir = Path(cuda_nvcc.__file__).resolve().parent
libdevice = cuda_data_dir / "nvvm" / "libdevice" / "libdevice.10.bc"
if libdevice.exists():
    print(cuda_data_dir)
PY
)"

  if [[ -n "$cuda_data_dir" ]]; then
    export XLA_FLAGS="--xla_gpu_cuda_data_dir=${cuda_data_dir}"
    echo "Configured XLA_FLAGS=${XLA_FLAGS}"
  else
    echo "Warning: could not find nvidia-cuda-nvcc libdevice. TensorFlow XLA JIT may fail on H100." >&2
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

if [[ "${FLEXIBO_ROLLING_WORKER:-0}" == "1" ]]; then
  TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required in worker mode}"
  CHUNK="${FLEXIBO_ROLLING_CHUNK:?FLEXIBO_ROLLING_CHUNK is required in worker mode}"

  DATASET_INDEX=$((TASK_ID / ${#SEEDS[@]}))
  SEED_INDEX=$((TASK_ID % ${#SEEDS[@]}))

  DATA="${DATASETS[$DATASET_INDEX]}"
  SEED="${SEEDS[$SEED_INDEX]}"
  NAME_EXP="flexibo_${DATA}_seed${SEED}_e${TOTAL_EVALS}_ep${EPOCHS}"
  HISTORY_PATH="${RESULTS_DIR}/${NAME_EXP}/algorithm_logs/flexibo_history.csv"
  COMPLETED_BEFORE=$(history_count "$HISTORY_PATH")

  echo "Running FlexiBO rolling chunk ${CHUNK}/${CHUNKS}"
  echo "Array task ${TASK_ID}/${TASK_COUNT}: dataset=${DATA}, seed=${SEED}"
  echo "Completed before chunk: ${COMPLETED_BEFORE}/${TOTAL_EVALS}"
  echo "Budget: chunk_accuracy_evals=${CHUNK_EVALS}, epochs=${EPOCHS}, surrogate=${SURROGATE}"
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
  configure_tensorflow_xla

  flops_scale_args=()
  if [[ -n "$FLOPS_SCALE" ]]; then
    flops_scale_args=(--flops_scale "$FLOPS_SCALE")
  fi

  ${PYTHON_BIN} flexibo_runner.py \
    --name "${RESULTS_DIR}/${NAME_EXP}" \
    --dataset "$DATA" \
    --seed "$SEED" \
    --eval "$TOTAL_EVALS" \
    --epochs "$EPOCHS" \
    --mod_list flops_module \
    --resume \
    --max_new_evals "$REMAINING_FOR_CHUNK" \
    --surrogate "$SURROGATE" \
    --beta "$BETA" \
    --candidate_pool "$CANDIDATE_POOL" \
    --init_random "$INIT_RANDOM" \
    --accuracy_cost "$ACCURACY_COST" \
    --flops_cost "$FLOPS_COST" \
    "${flops_scale_args[@]}"

  COMPLETED_AFTER=$(history_count "$HISTORY_PATH")
  echo "Completed after chunk: ${COMPLETED_AFTER}/${TOTAL_EVALS}"
  exit 0
fi

DATASETS_LIST="${DATASETS[*]}"
SEEDS_LIST="${SEEDS[*]}"
export CONDA_ENV PYTHON_BIN JOB_SETUP PARTITION GPUS MEMORY QOS TIME_LIMIT
export TOTAL_EVALS CHUNK_EVALS EPOCHS MAX_PARALLEL RESULTS_DIR SLURM_LOG_DIR HF_DATASETS_CACHE
export DATASETS_LIST SEEDS_LIST SURROGATE BETA CANDIDATE_POOL INIT_RANDOM
export ACCURACY_COST FLOPS_COST FLOPS_SCALE

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

echo "Starting rolling FlexiBO array submission"
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
    --job-name="flexibo_roll_c${CHUNK}" \
    --export="ALL,FLEXIBO_ROLLING_WORKER=1,FLEXIBO_ROLLING_CHUNK=${CHUNK}" \
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
