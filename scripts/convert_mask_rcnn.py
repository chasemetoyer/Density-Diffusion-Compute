"""One-time conversion of Detectron2's Mask R-CNN R50-FPN-3x checkpoint to safetensors.

The Model Zoo file is a plain pickle. It is read once with a restricted
unpickler that allows only the four globals a static scan found in it; any
other global aborts the conversion. Only keys in the model's state dict are
kept (Detectron2 ignores the rest), so the runtime load can be strict.

Exactness check: on every image given, the converted model (strict
safetensors load) must give bitwise-identical instances to Detectron2's own
loading of the original pickle.

Usage, from the project root:
    .venv/bin/python scripts/convert_mask_rcnn.py IMAGE [IMAGE ...]
"""

import json
import pickle
import sys
from pathlib import Path

import cv2
import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from density.masks.detectors import Detectron2Predictor, mask_rcnn_cfg  # noqa: E402
from density.masks.weights import ROOT, sha256_file, verified_path  # noqa: E402

ALLOWED_GLOBALS = {
    ("collections", "OrderedDict"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy", "dtype"),
    ("numpy", "ndarray"),
}
OUT = ROOT / "detectron2" / "model_final_f10217.safetensors"
CHECK_THRESHOLD = 0.05  # well below the registered 0.5, so many low-score instances are compared too


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in ALLOWED_GLOBALS:
            raise pickle.UnpicklingError(f"global {module}.{name} is not allowed")
        return super().find_class(module, name)


def convert() -> dict:
    source = verified_path("mask_rcnn_r50_fpn_3x")
    with open(source, "rb") as f:
        data = RestrictedUnpickler(f, encoding="latin1").load()
    if data.get("__author__") != "Detectron2 Model Zoo" or data.get("matching_heuristics", False):
        raise RuntimeError(f"unexpected checkpoint format: {sorted(data)}")

    model_keys = _model_state_keys()
    state = {k: torch.from_numpy(v.copy()) for k, v in data["model"].items() if k in model_keys}
    dropped = sorted(set(data["model"]) - model_keys)
    missing = sorted(model_keys - set(state))
    if missing:
        raise RuntimeError(f"checkpoint lacks model keys: {missing[:5]}")
    save_file(state, str(OUT), metadata={"source": source.name, "source_sha256": sha256_file(source), "dropped_keys": json.dumps(dropped)})
    return {"tensors": len(state), "dropped_keys": dropped, "sha256": sha256_file(OUT), "size": OUT.stat().st_size}


def _model_state_keys() -> set[str]:
    from detectron2.modeling import build_model

    return set(build_model(mask_rcnn_cfg("cpu")).state_dict())


def exactness_check(images: list[str]) -> list[dict]:
    from detectron2.engine import DefaultPredictor

    cfg = mask_rcnn_cfg("cpu", CHECK_THRESHOLD).clone()
    cfg.defrost()
    cfg.MODEL.WEIGHTS = str(verified_path("mask_rcnn_r50_fpn_3x"))
    original = DefaultPredictor(cfg)  # Detectron2's own pickle path, used only here
    converted = Detectron2Predictor(mask_rcnn_cfg("cpu", CHECK_THRESHOLD), load_file(str(OUT)))
    results = []
    for path in images:
        image = cv2.imread(path)
        a, b = original(image)["instances"].to("cpu"), converted(image)["instances"].to("cpu")
        fields = {
            "boxes": torch.equal(a.pred_boxes.tensor, b.pred_boxes.tensor),
            "scores": torch.equal(a.scores, b.scores),
            "classes": torch.equal(a.pred_classes, b.pred_classes),
            "masks": torch.equal(a.pred_masks, b.pred_masks),
        }
        results.append({"image": Path(path).name, "instances": len(a), "identical": all(fields.values()), **fields})
    return results


if __name__ == "__main__":
    summary = convert()
    checks = exactness_check(sys.argv[1:])
    print(json.dumps({"conversion": summary, "exactness": checks}, indent=2))
    if not checks or not all(c["identical"] for c in checks):
        raise SystemExit("converted Mask R-CNN differs from the original; do not use it")
