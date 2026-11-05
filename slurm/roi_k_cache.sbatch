#!/bin/bash
# Builds the ROI sub-window cache, one array task per line of params_roi_k_cache.txt
# (delta_t,roi_k). Submitted by slurm/submit_roi_k.sh before the tuner array.
#SBATCH --job-name=roi_k_cache
#SBATCH --account=IscrC_NESY
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=logs/roi_k_cache_%A_%a.out
#SBATCH --error=logs/roi_k_cache_%A_%a.err

source "$SLURM_SUBMIT_DIR/slurm/env_leonardo.sh"

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" params_roi_k_cache.txt)
IFS=',' read -r DELTA_T ROI_K <<< "$LINE"
echo "[$(date)] task ${SLURM_ARRAY_TASK_ID}: delta_t=${DELTA_T} roi_k=${ROI_K}"

python build_roi_k_cache.py --delta_t "$DELTA_T" --roi_k "$ROI_K"
