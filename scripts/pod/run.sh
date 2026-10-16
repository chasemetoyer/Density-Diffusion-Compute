#!/usr/bin/env bash
# One experiment session on the pod: this machine's canary, then every ready job of the run (density.run).
# Usage: scripts/pod/run.sh h100|slice RUN   (run inside tmux; sync.sh run-exp starts it). RUN is a folder under /workspace/runs.
# The status line goes to /workspace/logs/run_<machine>.status.
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$PROJECT"
machine=$1
run=/workspace/runs/$2
status=/workspace/logs/run_$machine.status
mkdir -p /workspace/logs
source /workspace/smoke_scratch/pod.env  # POD_CREATED_UTC, POD_GPU and POD_IMAGE, written from the Mac when the pod was created
export POD_CREATED_UTC POD_GPU POD_IMAGE
echo "running since $(date -u +%FT%TZ)" > "$status"
case $machine in
  h100) python -m density.canary || { echo "failed: the canary (exit $?) $(date -u +%FT%TZ)" > "$status"; exit 1; } ;;
  slice) python -m density.mask_canary || { echo "failed: the mask canary (exit $?) $(date -u +%FT%TZ)" > "$status"; exit 1; } ;;
  *) echo "usage: $0 h100|slice RUN" >&2; exit 2 ;;
esac
# the metric canary (registered 2026-10-07): it records at the machine's first session, and checks at every later one
python -m density.metric_canary --machine "$machine" || { echo "failed: the metric canary (exit $?) $(date -u +%FT%TZ)" > "$status"; exit 1; }
code=0
python -m density.run session "$run" --machine "$machine" || code=$?
if [[ $code -ne 0 ]]; then
  echo "failed in the session (exit $code) $(date -u +%FT%TZ)" > "$status"
  exit "$code"
fi
echo "done $(date -u +%FT%TZ)" > "$status"
