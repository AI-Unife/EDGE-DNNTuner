#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 {remote|lan}" >&2
  echo "  remote -> fresca@copernico.unife.it" >&2
  echo "  lan    -> fresca@copernico.endif.man" >&2
}

if [[ $# -ne 1 ]]; then
  usage
  exit 2
fi

case "$1" in
  remote)
    HOST="fresca@copernico.unife.it"
    ;;
  lan)
    HOST="fresca@copernico.endif.man"
    ;;
  *)
    usage
    exit 2
    ;;
esac

REMOTE_BASE="/hpc/home/fresca/EDGE-DNNTuner"

mkdir -p \
  results_BANANAS_naszilla_controller_cluster \
  results_FLEXIBO_controller_cluster \
  slurm_logs_cluster

echo "Downloading BANANAS results from ${HOST}..."
rsync -avz --progress \
  "${HOST}:${REMOTE_BASE}/results_BANANAS_naszilla_controller/" \
  ./results_BANANAS_naszilla_controller_cluster/

echo "Downloading FlexiBO results from ${HOST}..."
rsync -avz --progress \
  "${HOST}:${REMOTE_BASE}/results_FLEXIBO_controller/" \
  ./results_FLEXIBO_controller_cluster/

echo "Downloading Slurm logs from ${HOST}..."
rsync -avz --progress \
  "${HOST}:${REMOTE_BASE}/slurm_logs/" \
  ./slurm_logs_cluster/

echo "Done."
