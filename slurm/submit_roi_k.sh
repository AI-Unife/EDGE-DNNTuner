#!/bin/bash
# Submit the K-sub-window ROI experiments on Leonardo (account IscrC_NESY):
#   1) cache array  (one task per delta_t,roi_k whose cache is not complete yet:
#      a complete cache has <cache folder>/.complete, written by build_roi_k_cache.py)
#   2) tuner array  (one task per dataset,delta_t,roi_k,seed), starts only if
#      every cache task succeeded (immediately when no cache has to be built).
# SKIP_CACHE=1 trusts caches built before the .complete marker existed: every
# existing, non-empty cache folder is marked complete and not rebuilt.
# Results and logs go to $WORK/$USER/tuner (slurm/leonardo_config.sh).
# Run from the repository root:  bash slurm/submit_roi_k.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source slurm/leonardo_config.sh
module load "$PY_MODULE"
mkdir -p "$LOGS_DIR" "$RESULTS_DIR"
echo "results -> $RESULTS_DIR"
echo "logs    -> $LOGS_DIR"

"$VENV_DIR/bin/python" multi_sbatch.py roi_k   # writes params_roi_k.txt + params_roi_k_cache.txt

N_JOBS=$(grep -c . params_roi_k.txt)

# --- caches still to build ------------------------------------------------------
CACHE_TODO=params_roi_k_cache_todo.txt
: > "$CACHE_TODO"
while IFS=',' read -r DT K; do
    [[ -z "$DT" ]] && continue
    DIR="cache/DVS_ROI_32_fwdPass_trte${DT}k${K}_tete${DT}k${K}_2"
    if [[ "${SKIP_CACHE:-0}" == 1 && ! -f "$DIR/.complete" \
          && -n "$(ls -A "$DIR/train" 2>/dev/null)" && -n "$(ls -A "$DIR/test" 2>/dev/null)" ]]; then
        echo "existing (trusted)" > "$DIR/.complete"
    fi
    if [[ -f "$DIR/.complete" ]]; then
        echo "cache ok      : dt=$DT k=$K"
    else
        echo "cache to build: dt=$DT k=$K"
        echo "$DT,$K" >> "$CACHE_TODO"
    fi
done < params_roi_k_cache.txt
N_CACHE=$(grep -c . "$CACHE_TODO" || true)

DEPENDENCY=()
CACHE_ID=""
if (( N_CACHE > 0 )); then
    CACHE_ID=$(sbatch --parsable --array=1-"$N_CACHE" --export=ALL,CACHE_PARAMS="$CACHE_TODO" \
        --output="$LOGS_DIR/roi_k_cache_%A_%a.out" --error="$LOGS_DIR/roi_k_cache_%A_%a.err" \
        slurm/roi_k_cache.sbatch)
    echo "cache array: $CACHE_ID (1-$N_CACHE)"
    DEPENDENCY=(--dependency=afterok:"$CACHE_ID")
else
    echo "cache array: nothing to build"
fi

TUNER_ID=$(sbatch --parsable --array=1-"$N_JOBS" ${DEPENDENCY[@]+"${DEPENDENCY[@]}"} \
    --output="$LOGS_DIR/roi_k_tuner_%A_%a.out" --error="$LOGS_DIR/roi_k_tuner_%A_%a.err" \
    slurm/roi_k_array.sbatch)
echo "tuner array: $TUNER_ID (1-$N_JOBS)${CACHE_ID:+, after $CACHE_ID}"
