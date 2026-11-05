#!/bin/bash
# Submit the K-sub-window ROI experiments on Leonardo (account IscrC_NESY):
#   1) cache array  (one task per delta_t,roi_k)
#   2) tuner array  (one task per dataset,delta_t,roi_k,seed), starts only if
#      every cache task succeeded.
# Run from the repository root:  bash slurm/submit_roi_k.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source slurm/leonardo_config.sh
module load "$PY_MODULE"
echo "results -> $RESULTS_DIR"

"$VENV_DIR/bin/python" multi_sbatch.py roi_k   # writes params_roi_k.txt + params_roi_k_cache.txt
mkdir -p logs

N_CACHE=$(grep -c . params_roi_k_cache.txt)
N_JOBS=$(grep -c . params_roi_k.txt)

CACHE_ID=$(sbatch --parsable --array=1-"$N_CACHE" slurm/roi_k_cache.sbatch)
echo "cache array: $CACHE_ID (1-$N_CACHE)"

TUNER_ID=$(sbatch --parsable --array=1-"$N_JOBS" --dependency=afterok:"$CACHE_ID" slurm/roi_k_array.sbatch)
echo "tuner array: $TUNER_ID (1-$N_JOBS), after $CACHE_ID"
