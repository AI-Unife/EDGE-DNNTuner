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

CHUNKS=$(((TOTAL_EVALS + CHUNK_EVALS - 1) / CHUNK_EVALS))

base_sbatch_args=(
  --partition="$PARTITION"
  --gres="gpu:${GPUS}"
  --mem="$MEMORY"
  --time="$TIME_LIMIT"
  --output="${SLURM_LOG_DIR}/%x_%j.out"
  --error="${SLURM_LOG_DIR}/%x_%j.err"
)
if [[ -n "$QOS" ]]; then
  base_sbatch_args+=(--qos="$QOS")
fi

echo "Submitting BANANAS resume chains"
echo "Datasets: ${DATASETS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Budget per run: total_evals=${TOTAL_EVALS}, chunk_evals=${CHUNK_EVALS}, epochs=${EPOCHS}"
echo "Chunks per dataset/seed: ${CHUNKS}"

for DATA in "${DATASETS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    NAME_EXP="bananas_${DATA}_seed${SEED}_e${TOTAL_EVALS}_ep${EPOCHS}"
    previous_job=""

    for CHUNK in $(seq 1 "$CHUNKS"); do
      RUN_CMD="export HF_DATASETS_CACHE=${HF_DATASETS_CACHE}; ${PYTHON_BIN} bananas_runner.py \
        --name ${RESULTS_DIR}/${NAME_EXP} \
        --dataset ${DATA} \
        --seed ${SEED} \
        --eval ${TOTAL_EVALS} \
        --epochs ${EPOCHS} \
        --mod_list flops_module \
        --resume \
        --max_new_evals ${CHUNK_EVALS}"

      if [[ -n "$JOB_SETUP" ]]; then
        RUN_CMD="${JOB_SETUP} && ${RUN_CMD}"
      fi

      sbatch_args=(
        "${base_sbatch_args[@]}"
        --job-name="bananas_${DATA}_${SEED}_c${CHUNK}"
      )
      if [[ -n "$previous_job" ]]; then
        sbatch_args+=(--dependency="afterok:${previous_job}")
      fi

      job_id=$(sbatch --parsable "${sbatch_args[@]}" --wrap="$RUN_CMD")
      echo "Submitted ${NAME_EXP} chunk ${CHUNK}/${CHUNKS}: job ${job_id}"
      previous_job="$job_id"
    done
  done
done
