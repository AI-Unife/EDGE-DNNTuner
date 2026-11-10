#!/bin/bash
# Symbolic DNN-Tuner on the K-sub-window ROI variants, one array task per line
# of params_roi_k.txt (dataset,mode,delta_t,roi_k,seed). Submitted by
# slurm/submit_roi_k.sh after the cache array has finished.
#SBATCH --job-name=roi_k_tuner
#SBATCH --account=IscrC_NESY
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=logs/roi_k_tuner_%A_%a.out
#SBATCH --error=logs/roi_k_tuner_%A_%a.err

source "$SLURM_SUBMIT_DIR/slurm/env_leonardo.sh"

OPT="filtered"
NPARAMS_TH=300000
EPOCHS=${EPOCHS:-100}      # as the roigesture_matrix runs; override with: EPOCHS=... bash slurm/submit_roi_k.sh
EVAL=${EVAL:-300}          # symbolic_tuner.py default

# RESULTS_DIR = $WORK/$USER/tuner/results (slurm/leonardo_config.sh)
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" params_roi_k.txt)
IFS=',' read -r DATASET MODE DELTA_T ROI_K SEED <<< "$LINE"
# date + array job id prefix (as the old launchers): every submission gets its
# own folders, a re-run never appends to an older run's logs.
NAME="${RESULTS_DIR}/$(date +%y_%m_%d_%H)_${SLURM_ARRAY_JOB_ID}_${DATASET}_${MODE}_dt${DELTA_T}_k${ROI_K}_300K_${OPT}_s${SEED}"
echo "[$(date)] task ${SLURM_ARRAY_TASK_ID}: ${NAME}"

# analyze_results.py reads hyperparameters / parameter counts from a *.out file
# INSIDE the experiment folder: link this job's SLURM log there as tuner.out.
mkdir -p "$NAME"
JOB_OUT="$LOGS_DIR/roi_k_tuner_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
if [[ -f "$JOB_OUT" ]]; then
    ln -sfn "$JOB_OUT" "$NAME/tuner.out"
else
    echo "WARNING: $JOB_OUT not found, $NAME/tuner.out not linked" >&2
fi

python symbolic_tuner.py \
    --dataset "$DATASET" --mode "$MODE" \
    --delta_t "$DELTA_T" --roi_k "$ROI_K" \
    --train_framing delta_t --test_framing delta_t \
    --opt "$OPT" --seed "$SEED" \
    --nparams_th "$NPARAMS_TH" \
    --epochs "$EPOCHS" --eval "$EVAL" \
    --name "$NAME" \
    --mod_list flops_module
