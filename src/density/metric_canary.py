"""The metric canary (registered by Chase on 2026-10-07, decision 8): every session on a pod scores fixed inputs before
any job and requires the registered numbers, bit for bit.

Inputs: the stock clips' lossless frames and their CRF 10 MP4s (density.video).
- h100 (what the generation jobs score): LPIPS (spatial and scalar), masked PSNR and SSIM, and crop LPIPS of the face
  clip's first 17 frames, MP4 against lossless, under a fixed box mask; and the clip's FFV1 copy decoding bit for bit.
- slice (what the score jobs compute): the five quality dimensions of those 17 lossless frames and of their MP4; and the
  detector checks, MP4 against lossless, of SCRFD with ArcFace on the face clip, WiLoR on the hands clip and Mask R-CNN
  on the body clip (the first 9 frames of each).

The first pilot session on each machine has no registration yet, so the canary records: it scores the inputs in two
fresh processes, requires them identical, writes the registration (results/checks/metric_canary/registered_<machine>.json)
and a passing record for this host and code, which lets that session go on. The registration is then copied into
configs/metric_canary.json and committed, and every later session checks against it from a fresh process. A difference
stops the session.

    python -m density.metric_canary --machine h100|slice     check, or record when the machine has no registration
"""

import argparse
import hashlib
import json
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REGISTERED_FILE = ROOT / "configs" / "metric_canary.json"
OUT_DIR = ROOT / "results" / "checks" / "metric_canary"  # not versioned; synced back with the session's results
MACHINES = ("h100", "slice", "mac")  # mac: both sets, for the Mac's check of the canary itself
QUALITY_FRAMES, DETECTOR_FRAMES = 17, 9
BOX = (slice(120, 360), slice(300, 532))  # the fixed measurement ROI on the face clip (rows, columns)


class CanaryNotPassed(RuntimeError):
    pass


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def array_sha256(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def stock(name: str, frames: int, tmp: Path):
    """A stock clip's first frames and their decoded MP4."""
    from density import video
    from density.masks.clips import load_frames

    lossless = load_frames(name)[:frames]
    video.encode(lossless, tmp / f"{name}.mp4")
    return lossless, video.decode_rgb(tmp / f"{name}.mp4"), tmp / f"{name}.mp4"


def h100_scores(device: str, tmp: Path) -> dict:
    from density import video
    from density.masks.measurement import measurement_masks
    from density.metrics import fidelity

    lossless, mp4, _ = stock("face", QUALITY_FRAMES, tmp)
    roi = np.zeros(lossless.shape[:3], bool)
    roi[:, BOX[0], BOX[1]] = True
    scores = fidelity.fidelity(fidelity.LPIPS(device), mp4, lossless, measurement_masks(roi))
    copy = video.encode_lossless(lossless, tmp / "face.mkv")
    return {"fidelity": scores, "ffv1_uint8_sha256": copy["uint8_sha256"], "ffv1_verified_bitwise": copy["verified_bitwise"]}


def slice_scores(device: str, tmp: Path) -> dict:
    from density.metrics import detection
    from density.metrics.quality import QualityModels

    lossless, mp4, mp4_path = stock("face", QUALITY_FRAMES, tmp)
    models = QualityModels(device)
    out = {"quality_lossless": models.raw(tmp / "face.lossless.mp4", lossless), "quality_mp4": models.raw(mp4_path, mp4)}
    detectors = detection.Detectors(device)
    for name, stratum in (("face", "face"), ("hands", "manipulation"), ("body", "body")):
        a, b, _ = stock(name, DETECTOR_FRAMES, tmp)
        ref, cond = detection.detect_video(detectors, a, stratum), detection.detect_video(detectors, b, stratum)
        out[f"detection_{name}"] = {
            "boxes_sha256": [array_sha256(x) for x in ref["boxes"] + cond["boxes"]],
            **({"embeddings_sha256": [array_sha256(x) for x in ref["embeddings"] + cond["embeddings"]]} if "embeddings" in ref else {}),
            "compare": detection.compare(ref, cond, stratum),
        }
    return out


def scores(machine: str, device: str) -> dict:
    """One process's canary scores, under the jobs' strict deterministic settings."""
    from density import determinism

    determinism.configure(strict=True)
    with tempfile.TemporaryDirectory() as tmp:
        out = {}
        if machine in ("h100", "mac"):
            out["h100"] = h100_scores(device, Path(tmp))
        if machine in ("slice", "mac"):
            out["slice"] = slice_scores(device, Path(tmp))
    return out


def child(machine: str, device: str) -> dict:
    """The canary's scores from a fresh Python process."""
    r = subprocess.run([sys.executable, "-m", "density.metric_canary", "--child", "--machine", machine, "--device", device],
                       cwd=ROOT, capture_output=True, text=True)
    line = next((l for l in r.stdout.splitlines() if l.startswith("{")), None)
    if r.returncode or line is None:
        raise RuntimeError(f"the canary's scoring process failed (exit {r.returncode}): {r.stderr[-1500:]}")
    return json.loads(line)


def host_info() -> dict:
    import platform

    import torch

    return {"host": socket.gethostname(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cpu": platform.processor() or platform.machine(), "platform": platform.platform()}


def run(machine: str, device: str, registered_file: Path = REGISTERED_FILE, out: Path = OUT_DIR) -> dict:
    """Check against the machine's registration, or record one (two fresh processes, identical) when it has none."""
    from density.checks import euler

    start, host, code = time.perf_counter(), host_info(), euler.git_state()
    registered = json.loads(registered_file.read_text()).get(machine) if registered_file.exists() else None
    record = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "machine": machine, "device": device, "host": host, "code": code}
    try:
        if registered is None:
            first, second = child(machine, device), child(machine, device)
            same = digest(first) == digest(second)
            record.update(recorded=True, passed=same, digest=digest(first),
                          stop=None if same else "two fresh processes scored the canary's inputs differently: nothing is registered; stop")
            if same:
                registration = {"pod": host, "code": code, "utc": record["utc"], "device": device, "digest": digest(first), "scores": first}
                out.mkdir(parents=True, exist_ok=True)
                (out / f"registered_{machine}.json").write_text(json.dumps(registration, indent=1, sort_keys=True) + "\n")
                record["registration"] = str(out / f"registered_{machine}.json")
        else:
            got = child(machine, device)
            same = digest(got) == registered["digest"]
            record.update(recorded=False, passed=same, digest=digest(got), registered={k: registered[k] for k in ("pod", "code", "utc", "digest")},
                          stop=None if same else f"the canary's scores differ from the registration made on {registered['pod']['host']}: stop and report this host")
            if not same:
                record["got"] = got
    except Exception as error:
        record.update(passed=False, stop=f"the metric canary did not finish: {type(error).__name__}: {error}")
    record["seconds"] = time.perf_counter() - start
    out.mkdir(parents=True, exist_ok=True)
    (out / f"metric_canary_{machine}.json").write_text(json.dumps(record, indent=1) + "\n")
    return record


def require_passed(machine: str, out: Path = OUT_DIR) -> dict:
    """The passing record for this machine on this host at the current code; raises CanaryNotPassed otherwise."""
    from density.checks import euler

    path = out / f"metric_canary_{machine}.json"
    if not path.exists():
        raise CanaryNotPassed(f"no metric canary record for {machine} in {out}: run python -m density.metric_canary --machine {machine}")
    rec, code = json.loads(path.read_text()), euler.git_state()
    if not rec.get("passed"):
        raise CanaryNotPassed(f"the metric canary did not pass: {rec.get('stop')}")
    if rec["host"]["host"] != socket.gethostname():
        raise CanaryNotPassed(f"the metric canary passed on host {rec['host']['host']}, not this one: run it on this pod")
    if rec["code"]["git_sha"] != code["git_sha"] or rec["code"]["git_dirty"] or code["git_dirty"]:
        raise CanaryNotPassed(f"the metric canary ran at code {rec['code']['git_sha']} (dirty: {rec['code']['git_dirty']}); this is {code['git_sha']}: run it again")
    return rec


def main(argv=None) -> int:
    import torch

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--machine", choices=MACHINES, required=True)
    parser.add_argument("--device", default=None, help="default: cuda when available, else cpu")
    parser.add_argument("--child", action="store_true", help="internal: print one process's scores")
    parser.add_argument("--registered", type=Path, default=REGISTERED_FILE)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.child:
        print(json.dumps(scores(args.machine, device), sort_keys=True, default=float))
        return 0
    record = run(args.machine, device, args.registered, args.out)
    if record["passed"]:
        print(f"[metric canary] {'recorded' if record['recorded'] else 'passed'} on {record['host']['host']} ({args.machine}) in {record['seconds']:.0f} s", flush=True)
        return 0
    print(f"[metric canary] STOP: {record['stop']}", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
