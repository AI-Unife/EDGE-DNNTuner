# Shared Leonardo (CINECA) settings, sourced by setup_env_leonardo.sh and env_leonardo.sh.
# Everything lives under $WORK/$USER; each value can be overridden from the environment.

BASE_DIR="${BASE_DIR:-${WORK:?WORK is not set (Leonardo project work area)}/${USER}/tuner}"

# Python module the venv is built on (check with: module avail python).
# The same module must be loaded when the venv is used.
PY_MODULE="${PY_MODULE:-python/3.11.7}"

VENV_DIR="${VENV_DIR:-$BASE_DIR/venv_tuner}"          # Python environment
RESULTS_DIR="${RESULTS_DIR:-$BASE_DIR/tuner_results}"  # one sub-folder per experiment
CACHE_DIR="${CACHE_DIR:-$BASE_DIR/tuner_cache}"        # tonic caches (<repo>/cache links here)

# Repository root: the folder containing slurm/ (jobs are submitted from it).
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
