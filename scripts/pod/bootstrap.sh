#!/usr/bin/env bash
# Session A setup on the 24 GB slice, run on the pod: tools, uv, CPython, the CUDA 13.0 toolkit, the
# environment exactly as uv.lock pins it, and the Wan2.1 weights at the pinned revision.
# Every download is verified against configs/pod_installs.json or configs/wan_weights.json. Finished parts
# are skipped on a rerun. Facts for the environment record are appended to /workspace/setup/facts.txt.
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$PROJECT"
mkdir -p /workspace/setup /workspace/bin /workspace/logs
facts=/workspace/setup/facts.txt
fact() { echo "$1=$2" >> "$facts"; echo "[bootstrap] $1=$2"; }
note() { echo "[bootstrap] $(date -u +%FT%TZ) $*"; }
stop() { echo "[bootstrap] STOP: $*"; exit 2; }
cfg() { python3 -c 'import json, sys; v = json.load(open("configs/pod_installs.json"))[sys.argv[1]][sys.argv[2]]; print("" if v is None else v)' "$1" "$2"; }
wan() { python3 -c 'import json, sys; v = json.load(open("configs/wan_weights.json"))[sys.argv[1]]; print("\n".join(v) if isinstance(v, list) else v)' "$1"; }
fact bootstrap_started_utc "$(date -u +%FT%TZ)"

note "tools"
missing=()
for t in rsync tmux git; do command -v "$t" >/dev/null || missing+=("$t"); done
if ((${#missing[@]})); then
  apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${missing[@]}"
  fact apt_installed "${missing[*]}"
fi
for t in gcc g++ make curl python3 sha256sum md5sum; do
  command -v "$t" >/dev/null || stop "$t is missing; installing it is outside the approved list, so ask Chase"
done
fact gcc "$(gcc --version | head -1)"

uv_version=$(cfg uv version)
if [[ ! -x /workspace/bin/uv ]] || [[ "$(/workspace/bin/uv --version | awk '{print $2}')" != "$uv_version" ]]; then
  note "uv $uv_version"
  url=$(cfg uv url); want=$(cfg uv sha256)
  curl -fsSL --retry 3 -o /tmp/uv.tar.gz "$url"
  curl -fsSL --retry 3 -o /tmp/uv.tar.gz.sha256 "$url.sha256"
  got=$(sha256sum /tmp/uv.tar.gz | awk '{print $1}')
  published=$(awk '{print $1}' /tmp/uv.tar.gz.sha256)
  [[ "$got" == "$want" && "$published" == "$want" ]] || stop "uv tarball sha256 $got (published $published) differs from the registered $want"
  tar -xzf /tmp/uv.tar.gz -C /tmp
  install -m 0755 /tmp/uv-x86_64-unknown-linux-gnu/uv /tmp/uv-x86_64-unknown-linux-gnu/uvx /workspace/bin/
fi
fact uv "$(uv --version)"

note "CPython"
uv python install "$(cfg python version)"
fact python "$(uv python find "$(cfg python version)")"

cuda_dir=$(cfg cuda_toolkit install_dir)
if [[ ! -x "$cuda_dir/bin/nvcc" ]]; then
  note "CUDA toolkit $(cfg cuda_toolkit version)"
  url=$(cfg cuda_toolkit url); md5_want=$(cfg cuda_toolkit md5); sha_want=$(cfg cuda_toolkit sha256)
  runfile=/tmp/$(basename "$url")
  [[ -f "$runfile" ]] || curl -fL --retry 3 -o "$runfile" "$url"
  md5=$(md5sum "$runfile" | awk '{print $1}'); sha=$(sha256sum "$runfile" | awk '{print $1}')
  fact cuda_runfile_md5 "$md5"
  fact cuda_runfile_sha256 "$sha"
  if [[ -n "$sha_want" ]]; then
    [[ "$sha" == "$sha_want" ]] || stop "CUDA runfile sha256 $sha differs from the registered $sha_want"
  else
    [[ "$md5" == "$md5_want" ]] || stop "CUDA runfile md5 $md5 differs from NVIDIA's $md5_want"
    note "record cuda_toolkit.sha256 = $sha in configs/pod_installs.json"
  fi
  mkdir -p /tmp/cuda-tmp
  sh "$runfile" --silent --toolkit --toolkitpath="$cuda_dir" --no-man-page --override --tmpdir=/tmp/cuda-tmp
  rm -f "$runfile"
fi
fact nvcc "$("$cuda_dir/bin/nvcc" --version | grep release)"

# The volume (MooseFS) can return a transient "Stale file handle" (ESTALE) during uv's parallel unpacking;
# uv sync is retried only for that error.
uv_sync() {
  local log=$1; shift
  for attempt in 1 2 3; do
    uv sync --locked "$@" 2>&1 | tee -a "$log" && return 0
    grep -q "Stale file handle" "$log" || return 1
    fact uv_stale_file_handle_retry "$(basename "$log") attempt $attempt"
  done
  return 1
}
note "environment from uv.lock (torch first, then the sam-2 and Detectron2 builds against it)"
start=$(date +%s)
uv_sync /workspace/logs/uv-sync-1.log --no-install-package sam-2 --no-install-package detectron2 || stop "uv sync (torch pass) failed; see /workspace/logs/uv-sync-1.log"
uv_sync /workspace/logs/uv-sync-2.log || stop "uv sync (build pass) failed; see /workspace/logs/uv-sync-2.log"
fact uv_sync_seconds "$(( $(date +%s) - start ))"
torch_init=$(python -c 'import importlib.util; print(importlib.util.find_spec("torch").origin)')
links=$(stat -c %h "$torch_init")
if [[ "$links" -gt 1 ]] && ! grep -q "Failed to hardlink" /workspace/logs/uv-sync-*.log; then
  fact uv_link_mode hardlink
else
  fact uv_link_mode "copy (link count $links)"
  note "uv could not hardlink between its cache and the venv; cleaning the uv cache"
  uv cache clean
fi

note "build checks"
so=$(python -c 'import importlib.util; print(importlib.util.find_spec("detectron2._C").origin)')
archs=$("$cuda_dir/bin/cuobjdump" --list-elf "$so" | grep -o 'sm_[0-9]*' | sort -u | tr '\n' ' ')
fact detectron2_archs "$archs"
[[ "$archs" == *sm_90* && "$archs" == *sm_120* ]] || stop "the Detectron2 extension lacks sm_90 or sm_120: $archs"
python -c 'import importlib.util, sys; sys.exit(importlib.util.find_spec("sam2._C") is not None)' || stop "sam-2 was built with its CUDA extension"
python - <<'PY'
import torch, torchvision, detectron2._C, onnxruntime
boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=torch.float32, device="cuda")
keep = torchvision.ops.nms(boxes, torch.tensor([0.9, 0.8, 0.7], device="cuda"), 0.5).tolist()
assert keep == [0, 2], keep
print("[bootstrap] torch", torch.__version__, "CUDA", torch.version.cuda, torch.cuda.get_device_name(0), "| onnxruntime", onnxruntime.get_available_providers())
PY

note "Wan2.1 weights at the pinned revision"
repo=$(wan repo); revision=$(wan revision)
while read -r pattern; do
  hf download "$repo" --revision "$revision" --include "$pattern" --local-dir weights/Wan2.1-T2V-1.3B-Diffusers
done <<< "$(wan download_patterns)"
fact hf_home_gib "$(du -s --block-size=1G "$HF_HOME" 2>/dev/null | cut -f1)"
fact wan_weights_gib "$(du -s --block-size=1G weights/Wan2.1-T2V-1.3B-Diffusers | cut -f1)"
fact workspace_used "$(df -h /workspace | tail -1)"
fact bootstrap_finished_utc "$(date -u +%FT%TZ)"
