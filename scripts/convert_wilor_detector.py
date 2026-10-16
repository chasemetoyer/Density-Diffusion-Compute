"""One-time conversion of WiLoR's detector.pt (a dill-pickled Ultralytics YOLOv8m-pose model) to data-only files.

Run this ONLY in a throwaway environment outside the project, with exactly:
    torch==2.14.1 torchvision==0.29.1 ultralytics==8.4.123 dill==0.4.1
    numpy==2.2.6 opencv-python==4.14.0.94 safetensors==0.8.0
and YOLO_OFFLINE=True YOLO_AUTOINSTALL=False YOLO_CONFIG_DIR=<scratch dir>.

It writes, next to the original file:
    detector.safetensors   the model's state dict, as stored (fp16 weights, int64 BatchNorm counters)
    detector.json          architecture dict, class names, keypoint shape, stride, the checkpoint
                           arguments Ultralytics keeps for prediction, and provenance (versions, hashes)
and a reference file of the ORIGINAL model's detections on the images given. The reference and the
five Hands23 demo images it was made from are kept under weights/masks/wilor/conversion_check/ and
registered in density.masks.weights, so they are verified and copied to the pod with the weights.

The checkpoint was saved by Ultralytics 8.1.0. Some module definitions changed
since (for example SPPF's first Conv lost its activation), so a model built
from the architecture dict alone would differ. The converter compares every
module of the original against a fresh build and records activation
differences as explicit overrides; any other structural difference aborts. The project
environment (no dill, no pickle) rebuilds the model from these files and must reproduce
those detections exactly (``density.masks.wilor``; checked by ``density.checks.masks``).

Usage:
    python scripts/convert_wilor_detector.py DETECTOR_PT REFERENCE_NPZ IMAGE [IMAGE ...]
"""

import hashlib
import json
import sys
from pathlib import Path

import cv2
import dill
import numpy as np
import torch
import ultralytics
from safetensors.torch import save_file
from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.engine.model import Model
from ultralytics.nn.tasks import PoseModel, torch_safe_load

CONF = 0.3  # registered WiLoR threshold


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def structure_overrides(original, fresh) -> dict:
    """Activation overrides that make a fresh build match the original module graph exactly."""
    a = list(original.named_modules(remove_duplicate=False))
    b = list(fresh.named_modules(remove_duplicate=False))
    if [(n, type(m).__name__) for n, m in a if not n.endswith(".act")] != [(n, type(m).__name__) for n, m in b if not n.endswith(".act")]:
        raise SystemExit("module trees differ beyond activations; refusing to convert")
    overrides = {}
    fresh_modules = dict(b)
    for name, module in a:
        if name.endswith(".act"):
            other = fresh_modules[name]
            if type(module) is not type(other) or getattr(module, "inplace", None) != getattr(other, "inplace", None):
                if not isinstance(module, (torch.nn.SiLU, torch.nn.Identity)):
                    raise SystemExit(f"unsupported activation {type(module).__name__} at {name}")
                overrides[name] = {"type": type(module).__name__, "inplace": getattr(module, "inplace", None)}
    return overrides


def main(detector_pt: Path, reference_npz: Path, images: list[str]) -> None:
    ckpt, _ = torch_safe_load(str(detector_pt))  # the one full unpickle, needs dill
    model = ckpt.get("ema") or ckpt["model"]
    out_weights = detector_pt.with_name("detector.safetensors")
    out_meta = detector_pt.with_name("detector.json")

    fresh = PoseModel(cfg=model.yaml, ch=3, nc=len(model.names), data_kpt_shape=tuple(model.kpt_shape), verbose=False)
    overrides = structure_overrides(model, fresh)
    state = {k: v.detach().clone().contiguous() for k, v in model.state_dict().items()}
    save_file(state, str(out_weights), metadata={"source": detector_pt.name, "source_sha256": sha256(detector_pt)})
    meta = {
        "architecture": model.yaml,
        "model_class": type(model).__name__,
        "names": {int(k): v for k, v in model.names.items()},
        "kpt_shape": list(model.kpt_shape) if hasattr(model, "kpt_shape") else None,
        "stride": model.stride.tolist(),
        "nc": int(model.nc),
        "activation_overrides": overrides,
        "overrides": Model._reset_ckpt_args({**DEFAULT_CFG_DICT, **ckpt.get("train_args", {})}),
        "checkpoint_ultralytics_version": ckpt.get("version"),
        "checkpoint_date": ckpt.get("date"),
        "converted_with": {"ultralytics": ultralytics.__version__, "dill": dill.__version__, "torch": torch.__version__},
        "source_sha256": sha256(detector_pt),
        "weights_sha256": sha256(out_weights),
    }
    out_meta.write_text(json.dumps(meta, indent=2, sort_keys=True))

    # Reference detections from the original checkpoint, loaded the standard way.
    yolo = YOLO(str(detector_pt))
    reference = {}
    for path in images:
        image = cv2.imread(path)
        result = yolo.predict(image, conf=CONF, device="cpu", verbose=False)[0]
        name = Path(path).name
        reference[f"{name}:boxes"] = result.boxes.xyxy.numpy()
        reference[f"{name}:conf"] = result.boxes.conf.numpy()
        reference[f"{name}:cls"] = result.boxes.cls.numpy()
        reference[f"{name}:keypoints"] = result.keypoints.data.numpy()
    np.savez(reference_npz, **reference)
    print(json.dumps({"weights": str(out_weights), "weights_sha256": meta["weights_sha256"], "meta": str(out_meta),
                      "meta_sha256": sha256(out_meta), "dill": dill.__version__, "images": len(images)}, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:])
