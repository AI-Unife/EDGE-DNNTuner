# Shared Leonardo (CINECA) settings, sourced by setup_env_leonardo.sh, env_leonardo.sh
# and submit_roi_k.sh. The repository stays in $HOME; everything heavy lives under
# $WORK/$USER/tuner. Each value can be overridden from the environment.
#
#   $WORK/$USER/tuner/
#     venv/                         Python environment
#     datasets/rois_fs_32_16/       ROI dataset   (<repo>/rois_and_coordinates links here)
#     datasets/data/                DVSGesture    (<repo>/data links here)
#     cache/                        tonic caches  (<repo>/cache links here)
#     results/                      one sub-folder per experiment
#     logs/                         SLURM .out / .err

BASE_DIR="${BASE_DIR:-${WORK:?WORK is not set (Leonardo project work area)}/${USER}/tuner}"

# Python module the venv is built on (check with: module avail python).
# The same module must be loaded when the venv is used.
PY_MODULE="${PY_MODULE:-python/3.11.6--gcc--8.5.0}"

VENV_DIR="${VENV_DIR:-$BASE_DIR/venv}"
DATASETS_DIR="${DATASETS_DIR:-$BASE_DIR/datasets}"
CACHE_DIR="${CACHE_DIR:-$BASE_DIR/cache}"
RESULTS_DIR="${RESULTS_DIR:-$BASE_DIR/results}"
LOGS_DIR="${LOGS_DIR:-$BASE_DIR/logs}"

ROI_DIR="$DATASETS_DIR/rois_fs_32_16"   # contains datasets/DVSGestureROI_fs32
DATA_DIR="$DATASETS_DIR/data"           # contains DVSGesture

# Repository root: the folder containing slurm/ (jobs are submitted from it).
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
