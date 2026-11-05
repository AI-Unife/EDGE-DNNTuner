# Job environment on Leonardo (CINECA) — sourced by the *.sbatch scripts.
# Paths come from slurm/leonardo_config.sh ($WORK/$USER/tuner/...); run
# slurm/setup_env_leonardo.sh once on a login node first.

# shellcheck disable=SC1091
source "$SLURM_SUBMIT_DIR/slurm/leonardo_config.sh"

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "ERROR: no venv in $VENV_DIR — run 'bash slurm/setup_env_leonardo.sh' on a login node first" >&2
    exit 1
fi
if [[ ! -d "$REPO_DIR/rois_and_coordinates/datasets" ]]; then
    echo "ERROR: ROI dataset not reachable at $REPO_DIR/rois_and_coordinates/datasets" \
         "(expected link to $ROI_DIR) — run 'bash slurm/setup_env_leonardo.sh links'" >&2
    exit 1
fi
module load "$PY_MODULE"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
mkdir -p "$RESULTS_DIR"
cd "$REPO_DIR" || exit 1
export PYTHONUNBUFFERED=1
