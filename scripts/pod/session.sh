#!/usr/bin/env bash
# One smoke-test session on the pod, in order, stopping at the first failing step.
# Usage: scripts/pod/session.sh A|B|C|D|E|F   (run inside tmux; sync.sh run starts it)
# The status line goes to /workspace/logs/session_<S>.status.
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$PROJECT"
session=$1
status=/workspace/logs/session_$session.status
mkdir -p /workspace/logs
source /workspace/smoke_scratch/pod.env  # POD_CREATED_UTC and POD_GPU, written from the Mac when the pod was created
export SMOKE_SESSION=$session POD_CREATED_UTC POD_GPU
echo "running since $(date -u +%FT%TZ)" > "$status"
args=()
after=()
case $session in
  A) bash scripts/pod/bootstrap.sh || { echo "failed in bootstrap (exit $?)" > "$status"; exit 1; }
     steps=(weights noise encode env) ;;
  B) steps=(env weights pytest milestones generate sdpa hsa_dense t3 perf vae_precision sheet) ;;
  C) steps=(env weights masks mask_check mask_compare) ;;
  D) # flash attention registered: its own folders, so Session B's records (PyTorch's default, cuDNN) stay as they are
     flash=results/checks/smoke_test/flash
     mkdir -p "$flash" /workspace/smoke_scratch/flash
     for f in noise.json encode.json; do [[ -e "$flash/$f" ]] || cp "results/checks/smoke_test/$f" "$flash/$f"; done
     [[ -e "$flash/inputs" ]] || cp -r results/checks/smoke_test/inputs "$flash/inputs"
     args=(--out "$flash" --scratch /workspace/smoke_scratch/flash)
     steps=(env weights milestones generate sdpa hsa_dense t3 perf sheet crossmachine) ;;
  E) # the mask-speed diagnosis on the slice (approved 2026-10-07): thread configurations, one process each
     steps=(env weights)
     after=("python scripts/pod/diagnostics/mask_threads.py all") ;;
  F) # the mask-check rerun with the thread cap and the registered ROIAlign kernel (approved 2026-10-07), into its own
     # folder so Session C's records stay; then the mask canary is recorded and checked from a fresh process
     rerun=results/checks/smoke_test/rerun
     args=(--out "$rerun")
     steps=(env weights mask_check mask_compare)
     after=("python scripts/pod/rerun_gate.py $rerun"
            "python -m density.mask_canary --record $rerun/mask_canary/registered.json --out $rerun/mask_canary"
            "python -m density.mask_canary --registered $rerun/mask_canary/registered.json --out $rerun/mask_canary") ;;
  *) echo "usage: $0 A|B|C|D|E|F" >&2; exit 2 ;;
esac
for step in "${steps[@]}"; do
  code=0
  python -m density.checks.smoke "$step" "${args[@]}" || code=$?
  if [[ $code -ne 0 ]]; then
    echo "failed at $step (exit $code) $(date -u +%FT%TZ)" > "$status"
    exit "$code"
  fi
done
for cmd in "${after[@]}"; do  # commands after the steps, each a string, in order; the first failure stops the session
  code=0
  bash -c "$cmd" || code=$?
  if [[ $code -ne 0 ]]; then
    echo "failed in $cmd (exit $code) $(date -u +%FT%TZ)" > "$status"
    exit "$code"
  fi
done
echo "done $(date -u +%FT%TZ)" > "$status"
