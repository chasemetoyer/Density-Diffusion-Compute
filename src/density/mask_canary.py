"""The mask canary: at the start of every mask session, reproduce a short stock-clip run bitwise.

Registered (2026-10-07, Chase). Before any mask job, every model the mask pipeline uses runs on the first 9 frames of
the stock clips, whose conformed frames are checked against their committed hashes:
- SCRFD on every face-clip frame;
- WiLoR on every hands-clip frame, and Hands23 on its frames 0 and 8;
- Mask R-CNN, through the registered ROIAlign kernel, on the body clip's frames 0 and 8;
- SAM 2.1 on each clip, prompted with that clip's frame-0 detections.

Each model's outputs are hashed separately, so a mismatch names the model that moved. Every hash must equal the
registered ones in configs/mask_canary.json, recorded on the slice in the smoke test's Session F. If any differs, the
session stops and reports this pod's host, GPU, driver and CPU beside the registered pod's. SCRFD runs on the host
CPU, and its results differ between CPUs, so the CPU model matters as much as the GPU.

The canary runs as mask jobs do: strict deterministic mode, the CPU pools at the pod's quota, the registered ROIAlign
kernel and SAM's strict fp32 convolutions. Mask jobs call ``require_passed`` before they start. It accepts only a
passing record from this host at the current code.

    python -m density.mask_canary                    on the slice, after sourcing scripts/pod/env.sh
    python -m density.mask_canary --record FILE      write a registration record (refuses if a model finds nothing)
"""

import argparse
import dataclasses
import hashlib
import json
import socket
import struct
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from density import determinism
from density.canary import host_info
from density.checks import euler, smoke

ROOT = Path(__file__).resolve().parents[2]
REGISTERED_FILE = ROOT / "configs" / "mask_canary.json"
OUT_DIR = ROOT / "results" / "checks" / "mask_canary"  # not versioned; synced back with the session's results
FRAMES = 9
KEYFRAMES = (0, 8)
CLIPS = ("face", "hands", "body")


class CanaryNotPassed(RuntimeError):
    pass


def digest(obj) -> str:
    """A hash of nested outputs (arrays, tensors, numbers, strings, dataclasses), equal only for bitwise-equal outputs."""
    h = hashlib.sha256()

    def feed(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if isinstance(x, np.ndarray):
            h.update(f"{x.dtype}{x.shape}".encode())
            h.update(np.ascontiguousarray(x).tobytes())
        elif dataclasses.is_dataclass(x):
            feed(dataclasses.asdict(x))
        elif isinstance(x, dict):
            for k in sorted(x, key=str):
                h.update(str(k).encode())
                feed(x[k])
        elif isinstance(x, (list, tuple)):
            h.update(b"[")
            for v in x:
                feed(v)
            h.update(b"]")
        elif isinstance(x, float):
            h.update(struct.pack("<d", x))
        else:
            h.update(repr(x).encode())

    feed(obj)
    return h.hexdigest()


def clip_frames() -> dict:
    """The first FRAMES frames of each stock clip, after checking the whole clip against its committed hash."""
    from density.checks.masks import verified_frames_record
    from density.masks.clips import load_frames

    out = {}
    for clip in CLIPS:
        verified_frames_record(clip)
        out[clip] = np.ascontiguousarray(load_frames(clip)[:FRAMES])
    return out


def run_models(device: str) -> tuple[dict, dict]:
    """Every model on its clip: each model's output hash, and how many detections each call returned."""
    from density.masks import roi_align
    from density.masks.pipeline import Models
    from density.masks.sam import SAM2Propagator

    clips = clip_frames()
    models = Models(device)
    start = roi_align.calls()
    scrfd = [models.get("scrfd").detect(f) for f in clips["face"]]
    wilor = [models.get("wilor").detect(f) for f in clips["hands"]]
    hands23 = [models.get("hands23").detect(np.ascontiguousarray(clips["hands"][i][:, :, ::-1])) for i in KEYFRAMES]
    mask_rcnn = [models.get("mask_rcnn").detect(clips["body"][i]) for i in KEYFRAMES]
    hashes = {"scrfd": digest(scrfd), "wilor": digest(wilor), "hands23": digest(hands23), "mask_rcnn": digest(mask_rcnn)}
    counts = {"scrfd": [len(x) for x in scrfd], "wilor": [len(x) for x in wilor], "hands23": [len(x) for x in hands23], "mask_rcnn": [len(x) for x in mask_rcnn],
              "roi_align_kernel_calls": roi_align.calls() - start, "onnxruntime_threads": models.cpu_threads}
    prompts = {"face": scrfd[0], "hands": wilor[0], "body": mask_rcnn[0]}
    for clip, found in prompts.items():
        boxes = {track: {0: np.asarray(d["box"], dtype=np.float32)} for track, d in enumerate(found)}
        if not boxes:
            counts[f"sam_{clip}"] = 0
            continue
        propagator = SAM2Propagator(clips[clip], device)
        try:
            masks = propagator.propagate(boxes, FRAMES)
        finally:
            propagator.close()
        hashes[f"sam_{clip}"] = digest({str(track): m for track, m in masks.items()})
        counts[f"sam_{clip}"] = int(sum(m.sum() for m in masks.values()))
    return hashes, counts


def exercised(counts: dict) -> list[str]:
    """Models the run did not exercise: no detection on a chosen frame, no SAM pixels, or no ROIAlign kernel call."""
    missing = [name for name in ("scrfd", "wilor", "hands23", "mask_rcnn") if not all(counts[name])]
    missing += [f"sam_{clip}" for clip in CLIPS if not counts.get(f"sam_{clip}")]
    if not counts["roi_align_kernel_calls"]:
        missing.append("the ROIAlign kernel")
    return missing


def describe(pod: dict) -> str:
    return f"host {pod.get('host')} ({pod.get('gpu')}, driver {pod.get('driver')}, CPU {pod.get('cpu')}, kernel {pod.get('kernel')})"


def judge(hashes: dict, registered: dict, host: dict) -> dict:
    differ = sorted(name for name, h in registered["hashes"].items() if hashes.get(name) != h)
    extra = sorted(set(hashes) - set(registered["hashes"]))
    passed = not differ and not extra
    stop = None if passed else (
        f"mask canary mismatch in {', '.join(differ + extra)}: this pod is {describe(host)}; the registered run is {describe(registered['pod'])}. "
        "Stop: no mask job may run on this pod; report the host, CPU and driver to Chase.")
    return {"passed": passed, "differ": differ + extra, "stop": stop}


def run(device: str, out: Path, registered: dict | None, record_path: Path | None = None) -> dict:
    """Run the canary; with ``record_path`` write a registration record there, else judge against ``registered``."""
    start, host = time.perf_counter(), host_info()
    record = {"utc": smoke.utc_now(), "device": device, "host": host, "code": euler.git_state()}
    log = determinism.WarningLog()
    try:
        record["determinism"] = determinism.settings()
        with log.capture("mask canary"):
            record["hashes"], record["counts"] = run_models(device)
        missing = exercised(record["counts"])
        if record_path is not None:
            if missing:
                raise RuntimeError(f"not recorded: the canary did not exercise {', '.join(missing)}")
            record.update(passed=True, stop=None, recorded=str(record_path))
        else:
            record.update(judge(record["hashes"], registered, host))
            record["registered"] = {k: registered[k] for k in ("pod", "code", "utc")}
    except Exception as error:
        record.update(passed=False, error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(),
                      stop=f"the mask canary did not finish on {describe(host)}: {type(error).__name__}: {error}. Stop: no mask job may run on this pod.")
    record["warnings"] = log.summary()
    record["seconds"] = time.perf_counter() - start
    if record_path is not None and record.get("passed"):
        smoke.write_json(record_path, {"pod": host, "code": record["code"], "utc": record["utc"], "device": device, "hashes": record["hashes"],
                                       "counts": record["counts"], "determinism": record["determinism"], "seconds": record["seconds"]})
    out.mkdir(parents=True, exist_ok=True)
    smoke.write_json(out / "mask_canary.json", record)
    smoke.write_json(out / f"mask_canary_{host['host']}_{record['utc'][:19].replace(':', '')}.json", record)
    return record


def require_passed(out: Path = OUT_DIR) -> dict:
    """The passing check record for this host at the current code; raises CanaryNotPassed otherwise."""
    path = out / "mask_canary.json"
    if not path.exists():
        raise CanaryNotPassed(f"no mask canary record in {out}: run python -m density.mask_canary on this pod first")
    rec = json.loads(path.read_text())
    host, code = socket.gethostname(), euler.git_state()
    if not rec.get("passed") or rec.get("recorded"):
        raise CanaryNotPassed(f"the mask canary did not pass a check: {rec.get('stop') or 'the record is a registration, not a check'}")
    if rec["host"]["host"] != host:
        raise CanaryNotPassed(f"the mask canary passed on host {rec['host']['host']}, not this one ({host}): run it on this pod")
    if rec["code"]["git_sha"] != code["git_sha"] or rec["code"]["git_dirty"] or code["git_dirty"]:
        raise CanaryNotPassed(f"the mask canary ran at code {rec['code']['git_sha']} (dirty: {rec['code']['git_dirty']}); this is {code['git_sha']} (dirty: {code['git_dirty']}): run it again")
    return rec


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--device", default=None, help="default: cuda when available, else cpu")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--registered", type=Path, default=REGISTERED_FILE, help="the registered hashes to judge against")
    parser.add_argument("--record", type=Path, help="write a registration record here instead of judging")
    args = parser.parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    determinism.configure(strict=True)
    registered = None if args.record else json.loads(args.registered.read_text())
    record = run(device, args.out, registered, args.record)
    if record.get("passed"):
        print(f"[mask canary] {'recorded' if args.record else 'passed'} on {describe(record['host'])} in {record['seconds']:.0f} s", flush=True)
        return 0
    print(f"[mask canary] STOP: {record['stop']}", flush=True)
    return 1 if "differ" in record else 2


if __name__ == "__main__":
    raise SystemExit(main())
