#!/bin/bash
# Re-run every ROI experiment on Leonardo (account IscrC_NESY) at 5, 10, 100 and
# 200 ms, fixed delta_t framing on train and test, 3 seeds:
#   roigesture_coords, roigesture_matrix, roigesture_matrix_fc   (roi_k = 0)
#   roigesture_conv1d, roigesture_channels                       (roi_k = 5 at 5 ms, 10 otherwise)
# No symbolic module (no flops_module): networks above 300K params are cut with
# --params_cut (not trained, penalty score), see slurm/roi_all_array.sbatch.
# Steps (as slurm/submit_roi_k.sh):
#   1) cache array  (one task per delta_t,roi_k whose cache is not complete yet:
#      a complete cache has <cache folder>/.complete, written by build_roi_k_cache.py)
#   2) tuner array  (one task per dataset,delta_t,roi_k,seed), starts only if
#      every cache task succeeded (immediately when no cache has to be built).
#
# ROI positions are now the ROI centre: caches built before that change still hold
# the old (top-left) positions. Delete them BEFORE the first submission:
#   rm -rf "$CACHE_DIR"/DVS_ROI_*        (CACHE_DIR from slurm/leonardo_config.sh)
# Results and logs go to $WORK/$USER/tuner (slurm/leonardo_config.sh).
# Run from the repository root:  bash slurm/submit_roi_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source slurm/leonardo_config.sh
module load "$PY_MODULE"
mkdir -p "$LOGS_DIR" "$RESULTS_DIR"
echo "results -> $RESULTS_DIR"
echo "logs    -> $LOGS_DIR"

"$VENV_DIR/bin/python" multi_sbatch.py roi_all   # writes params_roi_all.txt + params_roi_all_cache.txt

N_JOBS=$(grep -c . params_roi_all.txt)

# --- caches still to build ------------------------------------------------------
CACHE_TODO=params_roi_all_cache_todo.txt
: > "$CACHE_TODO"
while IFS=',' read -r DT K; do
    [[ -z "$DT" ]] && continue
    if (( K > 0 )); then
        DIR="cache/DVS_ROI_32_fwdPass_trte${DT}k${K}_tete${DT}k${K}_2"
    else
        DIR="cache/DVS_ROI_32_fwdPass_trte${DT}_tete${DT}_2"
    fi
    if [[ -f "$DIR/.complete" ]]; then
        echo "cache ok      : dt=$DT k=$K"
    else
        echo "cache to build: dt=$DT k=$K"
        echo "$DT,$K" >> "$CACHE_TODO"
    fi
done < params_roi_all_cache.txt
N_CACHE=$(grep -c . "$CACHE_TODO" || true)

DEPENDENCY=()
CACHE_ID=""
if (( N_CACHE > 0 )); then
    CACHE_ID=$(sbatch --parsable --array=1-"$N_CACHE" --export=ALL,CACHE_PARAMS="$CACHE_TODO" \
        --output="$LOGS_DIR/roi_all_cache_%A_%a.out" --error="$LOGS_DIR/roi_all_cache_%A_%a.err" \
        slurm/roi_k_cache.sbatch)
    echo "cache array: $CACHE_ID (1-$N_CACHE)"
    DEPENDENCY=(--dependency=afterok:"$CACHE_ID")
else
    echo "cache array: nothing to build"
fi

TUNER_ID=$(sbatch --parsable --array=1-"$N_JOBS" ${DEPENDENCY[@]+"${DEPENDENCY[@]}"} \
    --output="$LOGS_DIR/roi_all_tuner_%A_%a.out" --error="$LOGS_DIR/roi_all_tuner_%A_%a.err" \
    slurm/roi_all_array.sbatch)
echo "tuner array: $TUNER_ID (1-$N_JOBS)${CACHE_ID:+, after $CACHE_ID}"
