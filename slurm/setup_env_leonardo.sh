#!/bin/bash
# Setup on a Leonardo LOGIN node (compute nodes have no internet). The repo stays
# in $HOME, everything else goes to $WORK/$USER/tuner (see slurm/leonardo_config.sh):
#   1) folders under $WORK/$USER/tuner
#   2) datasets: any rois_fs_32_16.zip / data.zip copied into $WORK/$USER/tuner/datasets
#      is extracted there (zip removed afterwards)
#   3) links from the repo to $WORK, so the code finds its relative paths:
#        <repo>/rois_and_coordinates -> datasets/rois_fs_32_16   (training)
#        <repo>/../rois_fs_32_16     -> datasets/rois_fs_32_16   (robustness_eval.py)
#        <repo>/data                 -> datasets/data
#        <repo>/cache                -> cache
#        <repo>/logs                 -> logs   (also catches direct `sbatch slurm/*.sbatch`)
#   4) Python venv in $WORK/$USER/tuner/venv (TensorFlow 2.15 with CUDA wheels)
#
#   bash slurm/setup_env_leonardo.sh          # everything
#   bash slurm/setup_env_leonardo.sh links    # only 1-3 (e.g. after uploading a zip)
set -euo pipefail
source "$(dirname "$0")/leonardo_config.sh"

echo "repo     : $REPO_DIR"
echo "base dir : $BASE_DIR"
mkdir -p "$VENV_DIR" "$DATASETS_DIR" "$CACHE_DIR" "$RESULTS_DIR" "$LOGS_DIR"

# --- 2) datasets --------------------------------------------------------------
for zip in "$DATASETS_DIR/rois_fs_32_16.zip" "$DATASETS_DIR/data.zip"; do
    if [[ -f "$zip" ]]; then
        echo "extracting $zip"
        unzip -q -o "$zip" -d "$DATASETS_DIR" -x '__MACOSX/*' '*.DS_Store'
        rm "$zip"
    fi
done
[[ -d "$ROI_DIR/datasets" ]] || echo "WARNING: ROI dataset missing: copy rois_fs_32_16.zip to $DATASETS_DIR and rerun" >&2

# --- 3) links -------------------------------------------------------------------
link() {   # link <target> <link path>
    local target="$1" path="$2"
    if [[ -L "$path" ]]; then
        ln -sfn "$target" "$path"
    elif [[ -e "$path" ]]; then
        echo "WARNING: $path is a real folder/file, not linked to $target (move its content there by hand)" >&2
        return
    else
        ln -s "$target" "$path"
    fi
    echo "linked $path -> $target"
}
link "$ROI_DIR"   "$REPO_DIR/rois_and_coordinates"
link "$ROI_DIR"   "$(dirname "$REPO_DIR")/rois_fs_32_16"
link "$DATA_DIR"  "$REPO_DIR/data"
link "$CACHE_DIR" "$REPO_DIR/cache"
link "$LOGS_DIR"  "$REPO_DIR/logs"

[[ "${1:-}" == "links" ]] && exit 0

# --- 4) venv --------------------------------------------------------------------
echo "venv     : $VENV_DIR  (module $PY_MODULE)"
module load "$PY_MODULE"
[[ -f "$VENV_DIR/bin/activate" ]] || python3 -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
# Versions aligned with the local development venv (tensorflow/keras 2.15, numpy < 2).
python -m pip install \
    "tensorflow[and-cuda]==2.15.*" \
    "numpy==1.26.4" \
    "scikit-optimize==0.10.2" \
    "tonic==1.6.0" \
    "problog==2.2.10" \
    h5py matplotlib "pandas<3" tqdm "PyYAML==6.0.3"

python -c "import tensorflow as tf, tonic, skopt, problog; print('tensorflow', tf.__version__, '| tonic', tonic.__version__, '| skopt', skopt.__version__)"
echo "Done. GPUs are visible only on compute nodes (check inside a job with: python -c 'import tensorflow as tf; print(tf.config.list_physical_devices(\"GPU\"))')."
