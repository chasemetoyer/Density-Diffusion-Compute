"""WiLoR's hand detector, an Ultralytics YOLOv8m-pose model rebuilt from data-only files.

This is the only module in the project that imports ``ultralytics`` (AGPL-3.0),
so the dependency can be swapped out in one place. Auto-install, online
checks and analytics are switched off before the import.

The original ``detector.pt`` needs a full (dill) unpickle, so it was
converted once offline by ``scripts/convert_wilor_detector.py``. Here the
model is built from the saved architecture dict, its state dict is loaded
strictly from safetensors, and prediction goes through the same
``PosePredictor`` path as ``YOLO(detector.pt).predict``.
"""

import json
import os

import numpy as np

from density.masks.registered import HAND_SIDES, WILOR_THRESHOLD
from density.masks.weights import ROOT, verified_path


def _import_ultralytics():
    os.environ["YOLO_AUTOINSTALL"] = "False"
    os.environ["YOLO_OFFLINE"] = "True"
    config_dir = ROOT / "ultralytics_config"
    config_dir.mkdir(parents=True, exist_ok=True)  # Ultralytics falls back to /tmp unless the directory exists
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)
    import ultralytics
    from ultralytics import settings

    settings.update({"sync": False})
    return ultralytics


class WiLoRHandDetector:
    """Hand boxes with handedness."""

    def __init__(self, device: str = "cpu", threshold: float = WILOR_THRESHOLD):
        _import_ultralytics()
        import torch
        from safetensors.torch import load_file
        from ultralytics.cfg import DEFAULT_CFG_DICT
        from ultralytics.nn.tasks import PoseModel
        from ultralytics.utils import callbacks

        meta = json.loads(verified_path("wilor_detector_meta").read_text())
        if meta["model_class"] != "PoseModel":
            raise RuntimeError(f"unexpected model class {meta['model_class']}")
        model = PoseModel(cfg=meta["architecture"], ch=3, nc=len(meta["names"]), data_kpt_shape=tuple(meta["kpt_shape"]), verbose=False)
        model.load_state_dict(load_file(str(verified_path("wilor_detector_safetensors"))), strict=True)
        if model.stride.tolist() != meta["stride"]:
            raise RuntimeError(f"rebuilt stride {model.stride.tolist()} differs from {meta['stride']}")
        # Module definitions changed after the checkpoint's Ultralytics 8.1.0; restore its activations.
        modules = dict(model.named_modules(remove_duplicate=False))
        for name, spec in meta["activation_overrides"].items():
            parent, attr = name.rsplit(".", 1)
            act = torch.nn.SiLU(inplace=bool(spec["inplace"])) if spec["type"] == "SiLU" else torch.nn.Identity()
            setattr(modules[parent], attr, act)
        # The checkpoint object's own attributes, then the rest of ultralytics.nn.tasks.load_checkpoint.
        model.kpt_shape = tuple(meta["kpt_shape"])
        model.nc = meta["nc"]
        model.names = {int(k): v for k, v in meta["names"].items()}
        model.args = {**DEFAULT_CFG_DICT, **meta["overrides"]}
        model.task = "pose"
        model = model.float().eval()
        for module in model.modules():
            if hasattr(module, "inplace"):
                module.inplace = True
            elif isinstance(module, torch.nn.Upsample) and not hasattr(module, "recompute_scale_factor"):
                module.recompute_scale_factor = None
        self.model = model
        self.callbacks = callbacks.get_default_callbacks()
        # Model._load: overrides = reset checkpoint args, plus the model path and task.
        self.overrides = {**meta["overrides"], "model": str(ROOT / "wilor/pretrained_models/detector.pt"), "task": "pose"}
        self.device = device
        self.threshold = threshold
        self._predictor = None

    def _predict(self, image_bgr: np.ndarray):
        from ultralytics.models.yolo.pose import PosePredictor

        if self._predictor is None:
            # Model.predict: overrides, then its fixed predict defaults, then the call's arguments.
            custom = {"conf": 0.25, "batch": 1, "save": False, "mode": "predict", "rect": True, "embed": None}
            args = {**self.overrides, **custom, "conf": self.threshold, "device": self.device, "verbose": False}
            self._predictor = PosePredictor(overrides=args, _callbacks=self.callbacks)
            self._predictor.setup_model(model=self.model, verbose=False)
        return self._predictor(source=image_bgr, stream=False)[0]

    def detect_bgr(self, image_bgr: np.ndarray) -> dict[str, np.ndarray]:
        result = self._predict(image_bgr)
        return {
            "boxes": result.boxes.xyxy.cpu().numpy(),
            "conf": result.boxes.conf.cpu().numpy(),
            "cls": result.boxes.cls.cpu().numpy(),
            "keypoints": result.keypoints.data.cpu().numpy(),
        }

    def detect(self, image_rgb: np.ndarray) -> list[dict]:
        """Boxes (x0, y0, x1, y1), scores and handedness ("left" for class 0, "right" for class 1) for one RGB frame."""
        out = self.detect_bgr(np.ascontiguousarray(image_rgb[:, :, ::-1]))
        return [
            {"box": box.astype(np.float64), "score": float(score), "side": HAND_SIDES[int(cls)]}
            for box, score, cls in zip(out["boxes"], out["conf"], out["cls"])
        ]
