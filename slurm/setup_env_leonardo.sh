#!/bin/bash
# One-time setup on a Leonardo LOGIN node (compute nodes have no internet):
#   - Python venv in $WORK/$USER/venv_tuner (TensorFlow 2.15 with CUDA wheels)
#   - $WORK/$USER/tuner_results and $WORK/$USER/tuner_cache
#   - <repo>/cache -> $WORK/$USER/tuner_cache (the code writes caches to ./cache;
#     the 1 ms ROI sub-window caches are too big for $HOME)
# Run from anywhere:  bash slurm/setup_env_leonardo.sh
set -euo pipefail
source "$(dirname "$0")/leonardo_config.sh"

echo "base dir : $BASE_DIR"
echo "venv     : $VENV_DIR  (module $PY_MODULE)"
echo "results  : $RESULTS_DIR"
echo "cache    : $CACHE_DIR"
echo "repo     : $REPO_DIR"

module load "$PY_MODULE"
mkdir -p "$RESULTS_DIR" "$CACHE_DIR"

if [[ ! -d "$VENV_DIR" ]]; then
    python3 -m venv "$VENV_DIR"
fi
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

# <repo>/cache -> $CACHE_DIR
if [[ -L "$REPO_DIR/cache" ]]; then
    echo "cache link already present: $(readlink "$REPO_DIR/cache")"
elif [[ -e "$REPO_DIR/cache" ]]; then
    echo "WARNING: $REPO_DIR/cache is a real folder; move its content to $CACHE_DIR and link it by hand" >&2
else
    ln -s "$CACHE_DIR" "$REPO_DIR/cache"
    echo "linked $REPO_DIR/cache -> $CACHE_DIR"
fi

python -c "import tensorflow as tf, tonic, skopt, problog; print('tensorflow', tf.__version__, '| tonic', tonic.__version__, '| skopt', skopt.__version__)"
echo "Done. GPUs are visible only on compute nodes (check inside a job with: python -c 'import tensorflow as tf; print(tf.config.list_physical_devices(\"GPU\"))')."
