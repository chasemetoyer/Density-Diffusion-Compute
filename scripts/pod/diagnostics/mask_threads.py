"""Slice diagnosis (Session E, approved by Chase 2026-10-07): why the mask detectors are slow on the 24 GB slice.

Session C found the detectors dominating on the slice while SAM 2.1 was fast there: SCRFD took 82 of 99 s per face
reference, Hands23 204 of 551 s per manipulation reference, and Mask R-CNN 1,623 of the body path's 1,708 s. The first
suspect is CPU oversubscription. The pod's cgroup allows about 3.4 CPUs of the host's 128, but onnxruntime and OpenCV
size their thread pools to the host, and torch's inter-op pool had 64 threads.

    python scripts/pod/diagnostics/mask_threads.py all [--quick]   on the slice, from session.sh E (inside tmux)
    python scripts/pod/diagnostics/mask_threads.py mac [--quick]   on the Mac: per-reference time on CPU, for the 2x rule

``all`` runs each thread configuration in its own process, because OpenMP, MKL and OpenBLAS read their thread counts
when the process starts. It writes ``results/checks/smoke_test/diag_mask_threads.json``; ``mac`` writes
``diag_mask_threads_mac.json``.

- bench: each detector on its frames. Records wall and CPU time per call, the cgroup's throttling, the process's
  thread count, every pool's size, and a digest of every output, so configurations compare bitwise. It also profiles
  one Mask R-CNN and one Hands23 call, splits Detectron2's time into preprocessing, model and copy, and times a few
  calls with deterministic algorithms off (a diagnostic only).
- pipeline: ``process_reference`` on the smoke references Session C used (Session B's) and on the stock body clip,
  with the mask digests and the seconds per reference after the models are built.
- du: the volume's usage, and how long du takes, for allocated blocks and apparent size.

Nothing here changes a registered setting; any setting change goes to Chase first.
"""

import argparse
import dataclasses
import hashlib
import json
import os
import struct
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "checks" / "smoke_test"
CONFIGS = {  # None: no override (Session C: env.sh's OMP and MKL threads, the quota rounded up; other pools at their defaults)
    "session_c": None,
    "capped_3": 3,  # the quota (3.4 CPUs) rounded down
    "capped_4": 4,  # rounded up, as env.sh does
    "single": 1,
}
# The Mac's validated configuration: torch's default of 8 threads and onnxruntime's default. 8 torch threads reproduce
# the committed Milestone 4 face masks; 1 does not. ultralytics sets OMP_NUM_THREADS=1 on import when it is unset,
# and torch reads it lazily, so the count is set explicitly here.
MAC_TORCH_THREADS = 8
KEYFRAMES = list(range(0, 81, 8))  # Hands23 runs on frames 0, 8, ..., 80
BODY_SUBSET = list(range(0, 81, 10))  # Mask R-CNN on 9 of the body clip's 81 frames in the bench
HOURS_CAP = {"slice": 6.0, "h100": 4.0}


def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def child_env(threads: int | None) -> dict:
    env = dict(os.environ)
    if threads is not None:
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env[k] = str(threads)
    return env


def apply_threads(config: str) -> None:
    """Runtime pool sizes, set before any model exists; onnxruntime's is passed to SCRFD."""
    import cv2
    import torch

    if config == "mac_validated":
        torch.set_num_threads(MAC_TORCH_THREADS)
        return
    threads = CONFIGS.get(config)
    if threads is not None:
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        cv2.setNumThreads(threads)


def os_threads() -> int | None:
    task = Path("/proc/self/task")
    return len(os.listdir(task)) if task.exists() else None


def cpu_stat() -> dict | None:
    p = Path("/sys/fs/cgroup/cpu.stat")
    return {k: int(v) for k, v in (line.split() for line in p.read_text().splitlines())} if p.exists() else None


def stat_delta(before: dict | None, after: dict | None) -> dict | None:
    if before is None or after is None:
        return None
    return {k: after[k] - before[k] for k in ("usage_usec", "nr_periods", "nr_throttled", "throttled_usec") if k in after}


def digest(obj) -> str:
    """A hash of nested outputs (arrays, tensors, numbers, strings, dataclasses), equal only for bitwise-equal outputs."""
    import numpy as np
    import torch

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


def sync(device: str) -> None:
    import torch

    if device.startswith("cuda"):
        torch.cuda.synchronize()


def timed(device: str, fn, items) -> tuple[list, dict]:
    """``fn`` on each item: outputs, and wall and CPU seconds per call with the cgroup's throttling over the run."""
    outputs, wall = [], []
    before, cpu0 = cpu_stat(), time.process_time()
    for item in items:
        start = time.perf_counter()
        outputs.append(fn(item))
        sync(device)
        wall.append(time.perf_counter() - start)
    after, cpu = cpu_stat(), time.process_time() - cpu0
    s = sorted(wall)
    return outputs, {"calls": len(wall), "wall_seconds": wall, "total_seconds": sum(wall), "median_seconds": s[len(s) // 2] if s else None,
                     "process_cpu_seconds": cpu, "cgroup": stat_delta(before, after), "os_threads_after": os_threads()}


def staged_detectron2(predictor, image_bgr, device: str) -> dict:
    """One Detectron2 call split as DefaultPredictor runs it: preprocessing on CPU, the model, the copy to the host."""
    import torch

    t0 = time.perf_counter()
    image = image_bgr[:, :, ::-1] if predictor.input_format == "RGB" else image_bgr
    height, width = image.shape[:2]
    resized = predictor.aug.get_transform(image).apply_image(image)
    tensor = torch.as_tensor(resized.astype("float32").transpose(2, 0, 1))
    t1 = time.perf_counter()
    with torch.no_grad():
        out = predictor.model([{"image": tensor, "height": height, "width": width}])[0]
    sync(device)
    t2 = time.perf_counter()
    out["instances"].to("cpu")
    t3 = time.perf_counter()
    return {"preprocess_seconds": t1 - t0, "model_seconds": t2 - t1, "to_cpu_seconds": t3 - t2}


def profile_call(device: str, fn) -> dict:
    import torch
    from torch.profiler import ProfilerActivity, profile

    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device.startswith("cuda") else [])
    with profile(activities=activities) as prof:
        fn()
        sync(device)
    averages = prof.key_averages()
    out = {"by_self_cpu": averages.table(sort_by="self_cpu_time_total", row_limit=15)}
    if device.startswith("cuda"):
        for key in ("self_device_time_total", "self_cuda_time_total"):
            try:
                out["by_self_device"] = averages.table(sort_by=key, row_limit=15)
                break
            except Exception as error:  # the key's name varies across torch versions
                out["by_self_device_error"] = f"{type(error).__name__}: {error}"
    return out


def smoke_context(device: str):
    from density.checks import smoke

    return smoke.Context(OUT, OUT / "scratch_unused", device, ROOT / "weights", None, dry_run=False)


def references(device: str) -> dict:
    """RGB uint8 frames of smoke-1 (face), smoke-2 (manipulation) and the stock body clip, with each one's stratum and seed."""
    from density.checks import smoke
    from density.masks.clips import load_frames

    ctx = smoke_context(device)
    out = {}
    for p in smoke.PROMPTS:
        frames, source = smoke.reference_frames(ctx, p)
        out[p["id"]] = {"frames": frames, "stratum": p["mask_stratum"], "seed": p["seed"], "source": source}
    out["body"] = {"frames": load_frames("body"), "stratum": "body", "seed": None, "source": {"stock_clip": "body"}}
    return out


def bench(config: str, device: str, quick: bool) -> dict:
    threads = CONFIGS.get(config)
    apply_threads(config)
    import cv2
    import numpy as np
    import torch

    from density import determinism
    from density.masks.pipeline import Models

    if device.startswith("cuda"):
        determinism.configure()  # as Session C ran: warn-only
    refs = references(device)
    face, manip, body = (refs[k]["frames"] for k in ("smoke-1", "smoke-2", "body"))
    n = 3 if quick else None
    face_frames = list(face[:n] if n else face)
    keyframes = [manip[i] for i in (KEYFRAMES[:2] if quick else KEYFRAMES)]
    body_frames = [body[i] for i in (BODY_SUBSET[:2] if quick else BODY_SUBSET)]
    hand_frames = list(manip[:n] if n else manip)

    record = {"config": config, "threads": threads, "device": device, "utc": utc(), "os_threads_at_start": os_threads()}
    models = Models(device, cpu_threads=threads)
    build = {}
    for name in ("scrfd", "wilor", "hands23", "mask_rcnn"):
        start = time.perf_counter()
        models.get(name)
        build[name] = time.perf_counter() - start
    so = models.get("scrfd").model.session.get_session_options()
    record["pools"] = {
        "torch_intra_op": torch.get_num_threads(), "torch_inter_op": torch.get_num_interop_threads(), "cv2": cv2.getNumThreads(),
        "onnxruntime": {"intra_op": so.intra_op_num_threads, "inter_op": so.inter_op_num_threads, "note": "0 is onnxruntime's default, sized to the host's physical cores"},
        "env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "os_threads_after_build": os_threads(),
    }
    record["build_seconds"] = build

    detectors = {
        "scrfd": (models.get("scrfd").detect, face_frames, "smoke-1, every frame"),
        "wilor": (models.get("wilor").detect, hand_frames, "smoke-2, every frame"),
        "hands23": (lambda im: models.get("hands23").detect(np.ascontiguousarray(im[:, :, ::-1])), keyframes, "smoke-2, keyframes 0, 8, ..., 80"),
        "mask_rcnn": (models.get("mask_rcnn").detect, body_frames, "stock body clip, frames 0, 10, ..., 80"),
    }
    record["detectors"] = {}
    for name, (fn, frames, label) in detectors.items():
        fn(frames[0])  # warm-up call, not timed
        outputs, timing = timed(device, fn, frames)
        record["detectors"][name] = {"frames": label, **timing, "digest": digest(outputs), "per_frame_digests": [digest(o) for o in outputs]}
        print(f"[diag] {config} {name}: {timing['total_seconds']:.1f} s for {timing['calls']} calls", flush=True)

    rcnn, h23 = models.get("mask_rcnn"), models.get("hands23")
    record["detectron2_stages"] = {
        "mask_rcnn": [staged_detectron2(rcnn.predictor, np.ascontiguousarray(f[:, :, ::-1]), device) for f in body_frames[:3]],
        "hands23": [staged_detectron2(h23, np.ascontiguousarray(f[:, :, ::-1]), device) for f in keyframes[:3]],
    }
    if config in ("session_c", "capped_3"):
        record["profile"] = {
            "mask_rcnn": profile_call(device, lambda: rcnn.detect(body_frames[0])),
            "hands23": profile_call(device, lambda: h23.detect(np.ascontiguousarray(keyframes[0][:, :, ::-1]))),
        }
    if config == "capped_3" and device.startswith("cuda"):
        torch.use_deterministic_algorithms(False)
        torch.utils.deterministic.fill_uninitialized_memory = False
        try:
            off = {}
            for name in ("mask_rcnn", "hands23"):
                fn, frames, _ = detectors[name]
                outputs, timing = timed(device, fn, frames[:3])
                off[name] = {**timing, "same_outputs_as_deterministic": [digest(o) for o in outputs] == record["detectors"][name]["per_frame_digests"][:3]}
            record["deterministic_off"] = {"note": "a diagnostic only: deterministic algorithms and fill_uninitialized_memory off", **off}
        finally:
            determinism.configure()
    return record


def pipeline(config: str, device: str, strict: bool, refs_wanted: list[str]) -> dict:
    threads = CONFIGS.get(config)
    apply_threads(config)
    import torch

    from density import determinism
    from density.checks.masks import mask_digest
    from density.masks.pipeline import Models, process_reference

    settings = determinism.configure(strict=strict) if device.startswith("cuda") else None  # the Mac's validated CPU run configures nothing
    log = determinism.WarningLog()
    refs = references(device)
    models = Models(device, cpu_threads=threads)
    record = {"config": config, "threads": threads, "device": device, "strict": strict, "determinism": settings, "utc": utc(), "references": {}}
    start = time.perf_counter()
    for name in ("scrfd", "wilor", "hands23", "mask_rcnn"):
        models.get(name)
    record["build_seconds"] = time.perf_counter() - start
    record["pools"] = {"torch_intra_op": torch.get_num_threads(), "torch_inter_op": torch.get_num_interop_threads(), "os_threads_after_build": os_threads()}
    with log.capture(f"pipeline_{config}"):
        for rid in refs_wanted:
            r = refs[rid]
            before = cpu_stat()
            t0 = time.perf_counter()
            result = process_reference(r["frames"], r["stratum"], models, seed=r["seed"])
            wall = time.perf_counter() - t0
            record["references"][rid] = {
                "stratum": r["stratum"], "source": r["source"], "seconds": result.seconds, "wall_seconds": wall, "sam": result.sam,
                "cgroup": stat_delta(before, cpu_stat()), "digest": mask_digest(result),
            }
            print(f"[diag] pipeline {config} {rid}: {wall:.0f} s", flush=True)
    record["warnings"] = log.summary()
    return record


def child(kind: str, config: str, device: str, extra: list[str], timeout: int, record_dir: Path) -> dict:
    """Run one part in a fresh process with ``config``'s thread environment; its record, or the failure."""
    threads = CONFIGS.get(config)
    tmp = record_dir / f".diag_{kind}_{config}.json"
    tmp.unlink(missing_ok=True)
    cmd = [sys.executable, __file__, f"_{kind}", "--config", config, "--device", device, "--write", str(tmp), *extra]
    start = time.perf_counter()
    try:
        r = subprocess.run(cmd, env=child_env(threads), timeout=timeout, capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        return {"config": config, "error": f"timed out after {timeout} s"}
    print(r.stdout[-4000:], end="", flush=True)
    if not tmp.exists():
        return {"config": config, "error": f"exit {r.returncode} without a record", "stderr": r.stderr[-6000:], "seconds": time.perf_counter() - start}
    rec = json.loads(tmp.read_text())
    tmp.unlink()
    rec["process_seconds"] = time.perf_counter() - start
    if r.returncode != 0:
        rec["stderr"] = r.stderr[-6000:]
    return rec


def hour_check(estimate_hours: float) -> str | None:
    """A stop reason when the next part could take the pod past its cap (POD_CREATED_UTC and POD_GPU from session.sh)."""
    created, gpu = os.environ.get("POD_CREATED_UTC"), os.environ.get("POD_GPU")
    if not created or gpu not in HOURS_CAP:
        return None
    elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(created.replace("Z", "+00:00"))).total_seconds() / 3600
    if elapsed + estimate_hours > HOURS_CAP[gpu]:
        return f"the next part (about {estimate_hours:.2f} h) would take the {gpu} pod past its {HOURS_CAP[gpu]:g} h cap ({elapsed:.2f} h so far); ask Chase"
    return None


def write(path: Path, record: dict) -> None:
    from density.checks import smoke

    smoke.write_json(path, record)


def du_record() -> dict:
    from density import volume

    if not volume.VOLUME.is_dir():
        return {"note": "no /workspace on this machine"}
    out = {}
    for apparent in (False, True):
        try:
            u = volume.usage(apparent=apparent)
            out["apparent" if apparent else "allocated"] = {**u, "used_gb": u["used_bytes"] / 1e9}
        except Exception as error:
            out["apparent" if apparent else "allocated"] = {"error": f"{type(error).__name__}: {error}"}
    out["stop_gb"] = volume.STOP_BYTES / 1e9
    return out


def run_all(device: str, quick: bool, record_dir: Path) -> int:
    path = record_dir / "diag_mask_threads.json"
    record = {"utc": utc(), "device": device, "quick": quick, "host": os.uname().nodename, "cgroup_cpu_max": None, "cpu_count": os.cpu_count(),
              "bench": {}, "pipeline": {}, "du_start": du_record()}
    p = Path("/sys/fs/cgroup/cpu.max")
    record["cgroup_cpu_max"] = p.read_text().strip() if p.exists() else None
    write(path, record)
    refs = "smoke-1" if quick else "smoke-1,smoke-2"  # quick: the pipeline still needs all 81 frames, so it runs one reference
    plan = [("bench", c, [], 0.25) for c in CONFIGS]
    plan += [("pipeline", "session_c", ["--refs", refs], 0.3), ("pipeline", "capped_3", ["--refs", refs, "--strict"], 0.3)]
    for kind, config, extra, hours in plan:
        stop = hour_check(hours)
        if stop:
            record["stopped"] = stop
            write(path, record)
            print(f"[diag] STOP: {stop}", flush=True)
            return 3
        flags = [*extra, *(["--quick"] if quick and kind == "bench" else [])]
        record[kind][config] = child(kind, config, device, flags, timeout=5400, record_dir=record_dir)
        write(path, record)
    # The full body path only where the bench shows Mask R-CNN is fast enough (Session C: about 20 s a frame, 27 min a clip).
    for config in ("capped_3", "capped_4"):
        rcnn = (record["bench"].get(config) or {}).get("detectors", {}).get("mask_rcnn")
        per_frame = rcnn["total_seconds"] / rcnn["calls"] if rcnn else None
        if per_frame is not None and per_frame < 3.0 and not quick:
            stop = hour_check(0.3)
            if stop:
                record["stopped"] = stop
                break
            record["pipeline"][f"{config}_body"] = child("pipeline", config, device, ["--refs", "body", "--strict"], timeout=5400, record_dir=record_dir)
        else:
            record["pipeline"][f"{config}_body"] = {"skipped": "quick mode" if quick else f"Mask R-CNN takes {per_frame} s a frame at {config} in the bench; the body clip would take about {81 * (per_frame or 0) / 60:.0f} min"}
        write(path, record)
    record["du_end"] = du_record()
    record["finished_utc"] = utc()
    write(path, record)
    print(f"[diag] done; record in {path}", flush=True)
    return 0


def run_mac(quick: bool, record_dir: Path) -> int:
    """The Mac's per-reference time on CPU in its validated configuration (no thread overrides), for Chase's 2x rule."""
    path = record_dir / "diag_mask_threads_mac.json"
    record = {"utc": utc(), "device": "cpu", "host": os.uname().nodename, "cpu_count": os.cpu_count(),
              "pipeline": pipeline("mac_validated", "cpu", strict=False, refs_wanted=["smoke-1"] if quick else ["smoke-1", "smoke-2", "body"])}
    record["finished_utc"] = utc()
    write(path, record)
    print(f"[diag] done; record in {path}", flush=True)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["all", "mac", "_bench", "_pipeline"])
    parser.add_argument("--config", default="session_c")
    parser.add_argument("--device", default=None)
    parser.add_argument("--refs", default="smoke-1,smoke-2")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--quick", action="store_true", help="a few frames per detector and no profiling of the full set: to test this script")
    parser.add_argument("--write", type=Path)
    parser.add_argument("--record-dir", type=Path, default=OUT, help="where all and mac write their record (default: results/checks/smoke_test)")
    args = parser.parse_args(argv)
    sys.path.insert(0, str(ROOT / "src"))
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.command in ("all", "mac"):
        args.record_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "all":
        return run_all(device, args.quick, args.record_dir)
    if args.command == "mac":
        return run_mac(args.quick, args.record_dir)
    try:
        rec = bench(args.config, device, args.quick) if args.command == "_bench" else pipeline(args.config, device, args.strict, args.refs.split(","))
    except Exception as error:
        traceback.print_exc()
        rec = {"config": args.config, "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}
        write(args.write, rec)
        return 1
    write(args.write, rec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
