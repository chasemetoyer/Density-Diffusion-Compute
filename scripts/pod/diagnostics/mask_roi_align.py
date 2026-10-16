"""Session E, follow-up (the approved slice diagnosis): Detectron2's ROIAlign through torchvision's CUDA kernel.

mask_threads.py found Mask R-CNN and Hands23 spending about 20 s an image in one compiled region. With deterministic
algorithms on and the input on CUDA, torchvision 0.29.1 replaces its roi_align kernel with a pure-Python version,
lazily torch.compile'd, which exists to make the backward pass deterministic (torchvision/ops/roi_align.py). The masks
need only the forward pass, and the kernel's forward pass computes each output on one thread, without atomics.

This measures the option of calling that kernel for inference while every other operation stays in deterministic mode,
with the CPU pools capped at 3 threads:
- bench: Mask R-CNN on the body clip's frames 0, 10, ..., 80 and Hands23 on smoke-2's keyframes, twice each, with
  the kernel; the first three of each also with torchvision's Python version, compared numerically.
- pipeline: process_reference on smoke-1, smoke-2 and the body clip in strict mode, twice each with fresh models,
  with the mask digests and seconds per reference.

    python scripts/pod/diagnostics/mask_roi_align.py      on the slice, after mask_threads.py all (inside tmux)

It writes results/checks/smoke_test/diag_mask_roi_align.json. Nothing here changes a registered setting.
"""

import importlib.util
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("mask_threads", HERE / "mask_threads.py")
mt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mt)
THREADS = 3


CALLS = {"kernel": 0}


def use_cuda_roi_align() -> None:
    """Detectron2's ROIAlign through torchvision's kernel, for inference only.

    The module comes from importlib: ``import detectron2.layers.roi_align as m`` would bind the package attribute,
    which detectron2.layers rebinds to the roi_align function itself."""
    import importlib

    import torch

    module = importlib.import_module("detectron2.layers.roi_align")

    def roi_align_forward(input, rois, output_size, spatial_scale, sampling_ratio, aligned):
        if torch.is_grad_enabled():
            raise RuntimeError("inference only: the kernel's backward pass is not deterministic")
        CALLS["kernel"] += 1
        return torch.ops.torchvision.roi_align(input, rois, spatial_scale, output_size[0], output_size[1], sampling_ratio, aligned)

    module.roi_align = roi_align_forward


def compare_rcnn(a: list, b: list) -> dict:
    import numpy as np

    if len(a) != len(b):
        return {"same_count": False, "counts": [len(a), len(b)]}
    ious = []
    for x, y in zip(a, b):
        union = int((x["mask"] | y["mask"]).sum())
        ious.append(int((x["mask"] & y["mask"]).sum()) / union if union else 1.0)
    return {"same_count": True, "max_box_diff_px": max((float(np.abs(x["box"] - y["box"]).max()) for x, y in zip(a, b)), default=0.0),
            "max_score_diff": max((abs(x["score"] - y["score"]) for x, y in zip(a, b)), default=0.0), "min_mask_iou": min(ious, default=1.0)}


def compare_hands23(a: list, b: list) -> dict:
    import numpy as np

    if len(a) != len(b):
        return {"same_count": False, "counts": [len(a), len(b)]}
    same_labels = all((x.hand_side, x.contact_state, x.object_touch, x.object_box is None) == (y.hand_side, y.contact_state, y.object_touch, y.object_box is None) for x, y in zip(a, b))
    boxes = [float(np.abs(x.hand_box - y.hand_box).max()) for x, y in zip(a, b)]
    boxes += [float(np.abs(x.object_box - y.object_box).max()) for x, y in zip(a, b) if x.object_box is not None and y.object_box is not None]
    return {"same_count": True, "same_labels": same_labels, "max_box_diff_px": max(boxes, default=0.0),
            "max_hand_score_diff": max((abs(x.hand_score - y.hand_score) for x, y in zip(a, b)), default=0.0)}


def bench() -> dict:
    mt.apply_threads("capped_3")
    import numpy as np

    from density import determinism
    from density.masks.pipeline import Models

    determinism.configure()
    refs = mt.references("cuda")
    body, manip = refs["body"]["frames"], refs["smoke-2"]["frames"]
    body_frames = [body[i] for i in mt.BODY_SUBSET]
    keyframes = [np.ascontiguousarray(manip[i][:, :, ::-1]) for i in mt.KEYFRAMES]
    models = Models("cuda", cpu_threads=THREADS)
    rcnn, h23 = models.get("mask_rcnn"), models.get("hands23")
    out = {"threads": THREADS, "utc": mt.utc()}
    fallback = {"mask_rcnn": [rcnn.detect(f) for f in body_frames[:3]], "hands23": [h23.detect(f) for f in keyframes[:3]]}  # torchvision's Python version
    use_cuda_roi_align()
    for name, fn, frames, compare in (("mask_rcnn", rcnn.detect, body_frames, compare_rcnn), ("hands23", h23.detect, keyframes, compare_hands23)):
        fn(frames[0])  # warm-up
        first, t1 = mt.timed("cuda", fn, frames)
        second, t2 = mt.timed("cuda", fn, frames)
        if CALLS["kernel"] == 0:
            raise RuntimeError("the patched roi_align was never called")
        out[name] = {
            "kernel": t1, "kernel_repeat": t2, "per_call_seconds": t1["total_seconds"] / t1["calls"], "kernel_calls_so_far": CALLS["kernel"],
            "repeat_bitwise": [mt.digest(o) for o in first] == [mt.digest(o) for o in second],
            "per_frame_digests": [mt.digest(o) for o in first],
            "against_python_version": [compare(x, y) for x, y in zip(first[:3], fallback[name])],
            "bitwise_equal_to_python_version": [mt.digest(x) == mt.digest(y) for x, y in zip(first[:3], fallback[name])],
        }
        print(f"[diag] roi_align kernel {name}: {out[name]['per_call_seconds']:.2f} s a call, repeat bitwise {out[name]['repeat_bitwise']}", flush=True)
    return out


def pipeline(refs_wanted: list[str]) -> dict:
    mt.apply_threads("capped_3")
    from density import determinism
    from density.checks.masks import mask_digest
    from density.masks.pipeline import Models, process_reference

    settings = determinism.configure(strict=True)
    use_cuda_roi_align()
    refs = mt.references("cuda")
    out = {"threads": THREADS, "strict": True, "determinism": settings, "utc": mt.utc(), "references": {}}
    log = determinism.WarningLog()
    with log.capture("roi_align_pipeline"):
        for rid in refs_wanted:
            r = refs[rid]
            runs = []
            for _ in range(2):
                models = Models("cuda", cpu_threads=THREADS)
                for name in ("scrfd", "wilor", "hands23", "mask_rcnn"):
                    models.get(name)
                t0 = time.perf_counter()
                result = process_reference(r["frames"], r["stratum"], models, seed=r["seed"])
                runs.append({"wall_seconds": time.perf_counter() - t0, "seconds": result.seconds, "digest": mask_digest(result)})
            if CALLS["kernel"] == 0 and r["stratum"] in ("manipulation", "body"):
                raise RuntimeError("the patched roi_align was never called")
            out["references"][rid] = {"stratum": r["stratum"], "runs": runs, "kernel_calls_so_far": CALLS["kernel"], "repeat_identical": runs[0]["digest"] == runs[1]["digest"],
                                      "wall_seconds": min(x["wall_seconds"] for x in runs), "digest": runs[0]["digest"], "seconds": runs[0]["seconds"]}
            print(f"[diag] roi_align kernel pipeline {rid}: {out['references'][rid]['wall_seconds']:.0f} s, repeat identical {out['references'][rid]['repeat_identical']}", flush=True)
    out["warnings"] = log.summary()
    return out


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] in ("_bench", "_pipeline"):
        target = Path(sys.argv[2])
        try:
            rec = bench() if sys.argv[1] == "_bench" else pipeline(["smoke-1", "smoke-2", "body"])
            code = 0
        except Exception as error:
            traceback.print_exc()
            rec, code = {"error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}, 1
        mt.write(target, rec)
        return code
    path = mt.OUT / "diag_mask_roi_align.json"
    record = {"utc": mt.utc(), "note": __doc__.split("\n\n")[1].replace("\n", " ")}
    for kind, hours in (("_bench", 0.15), ("_pipeline", 0.5)):
        stop = mt.hour_check(hours)
        if stop:
            record["stopped"] = stop
            break
        tmp = mt.OUT / f".diag_roi_{kind}.json"
        tmp.unlink(missing_ok=True)
        start = time.perf_counter()
        r = subprocess.run([sys.executable, __file__, kind, str(tmp)], env=mt.child_env(THREADS), capture_output=True, text=True, timeout=5400)
        print(r.stdout[-3000:], end="", flush=True)
        rec = json.loads(tmp.read_text()) if tmp.exists() else {"error": f"exit {r.returncode} without a record", "stderr": r.stderr[-6000:]}
        tmp.unlink(missing_ok=True)
        rec["process_seconds"] = time.perf_counter() - start
        record[kind.lstrip("_")] = rec
        mt.write(path, record)
    record["finished_utc"] = mt.utc()
    mt.write(path, record)
    print(f"[diag] done; record in {path}", flush=True)
    return 0 if "stopped" not in record else 3


if __name__ == "__main__":
    sys.path.insert(0, str(mt.ROOT / "src"))
    raise SystemExit(main())
