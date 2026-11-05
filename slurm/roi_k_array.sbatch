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
NAME="${RESULTS_DIR}/${DATASET}_${MODE}_dt${DELTA_T}_k${ROI_K}_300K_${OPT}_s${SEED}"
echo "[$(date)] task ${SLURM_ARRAY_TASK_ID}: ${NAME}"

python symbolic_tuner.py \
    --dataset "$DATASET" --mode "$MODE" \
    --delta_t "$DELTA_T" --roi_k "$ROI_K" \
    --train_framing delta_t --test_framing delta_t \
    --opt "$OPT" --seed "$SEED" \
    --nparams_th "$NPARAMS_TH" \
    --epochs "$EPOCHS" --eval "$EVAL" \
    --name "$NAME"
