#!/usr/bin/env bash
# Mac-side transfers to and from a pod over SSH; rsync only. Usage: scripts/pod/sync.sh <command> <host> <port> [args]
#   prepare        install rsync, tmux and git on the pod if missing (approved apt packages); needed before any rsync
#   code           push the tracked files, .git and results/checks/wan_real_mps/ (refuses a dirty working tree)
#   mask-weights   push weights/masks/
#   clips          push the stock clips' conformed frames, and the Mac's stock-clip masks to /workspace/smoke_scratch/mac_masks
#   metric-weights push weights/metrics/
#   prompt-sources push data/prompt_sources/ (the pinned prompt lists, the derived Kinetics list included), which any
#                  validation of prompts.csv on the pod reads; push it with every code push
#   pod-env UTC GPU [IMAGE]  record the pod's creation time (ISO 8601 with +00:00), GPU (slice or h100) and container image
#   run S          start session S (A to E) in tmux
#   status S       the session's status line and the last lines of its log
#   results        pull results/checks/smoke_test/ and the pod's logs back to the Mac (never data/ or weights/)
#   run-exp M RUN  start an experiment session of run RUN (under /workspace/runs) on machine M (h100 or slice) in tmux
#   exp-status M   that session's status line and the last lines of its log
#   run-pull RUN   pull the run's manifest, records, logs, masks, sheets, scores, audit and MP4s to results/runs/RUN
#                  (never lossless frames, PNG or FFV1, noise or embeddings)
#   run-push-audit RUN  push results/runs/RUN/audit/audit.csv, with Chase's reviews, to the pod
#   run-cmd ARGS   run "python -m density.run ARGS" on the pod (env.sh sourced), e.g. run-cmd audit /workspace/runs/pilot
#   registrations-pull  pull the canary registrations a pod made (results/checks/metric_canary/registered_*.json, and
#                  configs/encode_canary.json if it was re-recorded there) to results/checks/pod_registrations/ for review
#   shell          open a shell on the pod
set -euo pipefail
cmd=$1 host=$2 port=$3
shift 3
cd "$(git rev-parse --show-toplevel)"
known=${TMPDIR:-/tmp}/density_pod_known_hosts
ssh_opts=(-p "$port" -i "$HOME/.ssh/id_ed25519" -o "UserKnownHostsFile=$known" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30)
remote=root@$host
dest=/workspace/Density_Project
on_pod() { ssh "${ssh_opts[@]}" "$remote" "$@"; }
rs() { rsync -az --no-owner --no-group -e "ssh ${ssh_opts[*]}" "$@"; }
case $cmd in
  prepare)
    on_pod 'missing=""; for t in rsync tmux git; do command -v $t >/dev/null || missing="$missing $t"; done
            if [ -n "$missing" ]; then apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq $missing; fi
            echo "apt installed:${missing:- nothing}"; mkdir -p /workspace/Density_Project /workspace/smoke_scratch /workspace/logs' ;;
  code)
    # the same rule as density.checks.euler.git_state: reports under results/checks don't make the code dirty
    [[ -z "$(git status --porcelain -- . ':(exclude)results/checks')" ]] || { echo "commit first: the working tree is dirty" >&2; exit 1; }
    git ls-files -z | rs --from0 --files-from=- ./ "$remote:$dest/"
    rs --delete .git/ "$remote:$dest/.git/"
    rs results/checks/wan_real_mps/ "$remote:$dest/results/checks/wan_real_mps/"
    on_pod "cd $dest && git rev-parse HEAD && git status --porcelain | head -5" ;;
  mask-weights)
    on_pod "mkdir -p $dest/weights/masks"
    rs weights/masks/ "$remote:$dest/weights/masks/" ;;
  clips)
    on_pod "mkdir -p $dest/data/sample_clips /workspace/smoke_scratch/mac_masks"
    rs --include='*/' --include='/*/frames/frame_*.png' --exclude='*' --prune-empty-dirs data/sample_clips/ "$remote:$dest/data/sample_clips/"
    rs --include='*/' --include='/*/masks/masks.json' --exclude='*' --prune-empty-dirs data/sample_clips/ "$remote:/workspace/smoke_scratch/mac_masks/" ;;
  metric-weights)
    on_pod "mkdir -p $dest/weights/metrics"
    rs weights/metrics/ "$remote:$dest/weights/metrics/" ;;
  prompt-sources)
    on_pod "mkdir -p $dest/data/prompt_sources"
    rs data/prompt_sources/ "$remote:$dest/data/prompt_sources/" ;;
  pod-env)
    on_pod "mkdir -p /workspace/smoke_scratch && printf 'POD_CREATED_UTC=%s\nPOD_GPU=%s\nPOD_IMAGE=%s\n' '$1' '$2' '${3:-}' > /workspace/smoke_scratch/pod.env && cat /workspace/smoke_scratch/pod.env" ;;
  run)
    on_pod "tmux new-session -d -s smoke_$1 'bash $dest/scripts/pod/session.sh $1 2>&1 | tee -a /workspace/logs/session_$1.log'" && echo "session $1 started" ;;
  status)
    on_pod "cat /workspace/logs/session_$1.status 2>/dev/null; tail -n ${2:-15} /workspace/logs/session_$1.log 2>/dev/null" ;;
  results)
    mkdir -p results/checks/smoke_test/pod_logs
    rs "$remote:$dest/results/checks/smoke_test/" results/checks/smoke_test/
    rs "$remote:/workspace/logs/" results/checks/smoke_test/pod_logs/logs/
    rs "$remote:/workspace/setup/" results/checks/smoke_test/pod_logs/setup/ || true ;;
  run-exp)
    on_pod "mkdir -p /workspace/runs && tmux new-session -d -s run_$1 'bash $dest/scripts/pod/run.sh $1 $2 2>&1 | tee -a /workspace/logs/run_$1.log'" && echo "run $2 on $1 started" ;;
  exp-status)
    on_pod "cat /workspace/logs/run_$1.status 2>/dev/null; tail -n ${2:-15} /workspace/logs/run_$1.log 2>/dev/null" ;;
  run-pull)
    mkdir -p "results/runs/$1"
    rs --exclude='frames/' --exclude='frames.tmp/' --exclude='*.mkv' --exclude='inputs/noise/' --exclude='inputs/embeddings/' "$remote:/workspace/runs/$1/" "results/runs/$1/" ;;
  run-push-audit)
    rs "results/runs/$1/audit/audit.csv" "$remote:/workspace/runs/$1/audit/audit.csv" ;;
  run-cmd)
    on_pod "source $dest/scripts/pod/env.sh && cd $dest && python -m density.run $*" ;;
  registrations-pull)
    mkdir -p results/checks/pod_registrations
    rs "$remote:$dest/results/checks/metric_canary/" results/checks/pod_registrations/metric_canary/ || true
    rs "$remote:$dest/configs/encode_canary.json" results/checks/pod_registrations/encode_canary.json || true ;;
  shell)
    ssh "${ssh_opts[@]}" "$remote" ;;
  *) echo "unknown command $cmd" >&2; exit 2 ;;
esac
