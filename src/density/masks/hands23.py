"""Hands23 hand, contact and object detector.

The detector is vendored at ``third_party/hands23_detector`` (commit
1bc0f919ffa7a9f375e7e8042e1d26f3743d5819, MIT) with the recorded patch
``third_party/patches/hands23_detector.diff`` (device handling, pdb calls
replaced by errors, Pillow ``textbbox``).

Weights are loaded only from our converted file, a safetensors copy of the
tensors in ``model_hands23.pth`` read with ``torch.load(weights_only=True)``.
Nothing here unpickles a checkpoint.
"""

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from density.masks.registered import HAND_SIDES, HANDS23_THRESHOLDS
from density.masks.weights import verified_path

VENDOR = Path(__file__).resolve().parents[3] / "third_party" / "hands23_detector"
VENDOR_COMMIT = "1bc0f919ffa7a9f375e7e8042e1d26f3743d5819"
PATCH = VENDOR.parent / "patches" / "hands23_detector.diff"
CONFIG = VENDOR / "faster_rcnn_X_101_32x8d_FPN_3x_Hands23.yaml"


def _vendored_demo():
    """Import the vendored demo module (decoding helpers) without a top-level ``demo`` name."""
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    import hodetector.modeling.roi_heads  # noqa: F401  (registers hoRCNNROIHeads)

    spec = importlib.util.spec_from_file_location("hands23_vendored_demo", VENDOR / "demo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class HandObject:
    """One detected hand and, if Hands23 paired one, the object it interacts with."""

    hand_box: np.ndarray  # x0, y0, x1, y1 in pixels
    hand_side: str  # "left_hand" or "right_hand"
    hand_score: float
    contact_state: str  # no_contact, self_contact, other_person_contact, object_contact, obj_to_obj_contact
    object_box: np.ndarray | None
    object_score: float | None
    object_touch: str | None

    @property
    def side(self) -> str:
        """``left`` or ``right``, the vocabulary WiLoR's hand tracks use."""
        side = self.hand_side.removesuffix("_hand")
        if side not in HAND_SIDES:
            raise ValueError(f"unexpected Hands23 hand side {self.hand_side!r}")
        return side


class Hands23Predictor:
    """Mirrors detectron2's DefaultPredictor, but loads weights from the converted safetensors file only."""

    def __init__(self, weights: Path | None = None, device: str = "cpu", thresholds: dict = HANDS23_THRESHOLDS):
        from detectron2.config import get_cfg
        from detectron2.data import transforms as T
        from detectron2.modeling import build_model

        from density.masks import roi_align

        roi_align.install()  # the registered ROIAlign kernel, checked on every call
        self.demo = _vendored_demo()
        cfg = get_cfg()
        cfg.merge_from_file(str(CONFIG))
        cfg.MODEL.WEIGHTS = ""
        cfg.MODEL.DEVICE = device
        # Same keys and arithmetic as the vendored demo's set_cfg.
        cfg.HAND = thresholds["hand"]
        cfg.FIRSTOBJ = thresholds["first_object"]
        cfg.SECONDOBJ = thresholds["second_object"]
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = min(thresholds["hand"], thresholds["first_object"], thresholds["second_object"])
        cfg.HAND_RELA = thresholds["hand_relation"]
        cfg.OBJ_RELA = thresholds["object_relation"]
        cfg.freeze()
        self.cfg = cfg

        self.model = build_model(cfg).eval()
        state = load_file(str(weights or verified_path("hands23")))
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"Hands23 weights do not match the model: missing {missing[:5]}, unexpected {unexpected[:5]}")
        self.aug = T.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)
        self.input_format = cfg.INPUT.FORMAT

    def __call__(self, image_bgr: np.ndarray) -> dict:
        """Model output for one BGR uint8 image, as DefaultPredictor returns it."""
        from density.masks import roi_align

        with torch.no_grad(), roi_align.checked("Hands23"):
            image = image_bgr[:, :, ::-1] if self.input_format == "RGB" else image_bgr
            height, width = image.shape[:2]
            resized = self.aug.get_transform(image).apply_image(image)
            tensor = torch.as_tensor(resized.astype("float32").transpose(2, 0, 1))
            return self.model([{"image": tensor, "height": height, "width": width}])[0]

    def detect(self, image_bgr: np.ndarray) -> list[HandObject]:
        """Hands with their paired objects, decoded by the vendored ``deal_output``."""
        hands = self.demo.deal_output(image_bgr, self)
        return [
            HandObject(
                hand_box=np.asarray(h.hand_bbox, dtype=np.float32),
                hand_side=h.hand_side,
                hand_score=float(h.pred_score),
                contact_state=h.contactState,
                object_box=None if h.obj_bbox is None else np.asarray(h.obj_bbox, dtype=np.float32),
                object_score=None if h.obj_pred_score is None else float(h.obj_pred_score),
                object_touch=h.obj_touch,
            )
            for h in hands
        ]
