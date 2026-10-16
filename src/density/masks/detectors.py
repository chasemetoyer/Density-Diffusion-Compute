"""Detector wrappers that load weights as data only.

Registered rule: at runtime every weight file loads as data (safetensors,
ONNX, or ``torch.load(weights_only=True)``). Files that need a full unpickle
are converted once offline, with an exactness check (``scripts/convert_*.py``).

Detectron2 models are built from config and loaded strictly from safetensors;
no checkpointer, model-zoo lookup or download is involved.
"""

from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from density.masks.registered import MASK_RCNN_PERSON_THRESHOLD, SCRFD_INPUT_SIZE, SCRFD_THRESHOLD
from density.masks.weights import verified_path

COCO_PERSON = 0  # contiguous class id of "person" in Detectron2's COCO models


def mask_rcnn_config_path() -> Path:
    import detectron2

    return Path(detectron2.__file__).parent / "model_zoo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"


class Detectron2Predictor:
    """Detectron2's DefaultPredictor preprocessing, with a model whose weights were loaded strictly.

    ROIAlign runs through the registered kernel (density/masks/roi_align.py), checked on every call.
    """

    def __init__(self, cfg, state: dict[str, torch.Tensor]):
        from detectron2.data import transforms as T
        from detectron2.modeling import build_model

        from density.masks import roi_align

        roi_align.install()
        self.cfg = cfg
        self.model = build_model(cfg).eval()
        self.model.load_state_dict(state, strict=True)
        self.aug = T.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)
        self.input_format = cfg.INPUT.FORMAT

    def __call__(self, image_bgr: np.ndarray) -> dict:
        from density.masks import roi_align

        with torch.no_grad(), roi_align.checked("Mask R-CNN"):
            image = image_bgr[:, :, ::-1] if self.input_format == "RGB" else image_bgr
            height, width = image.shape[:2]
            resized = self.aug.get_transform(image).apply_image(image)
            tensor = torch.as_tensor(resized.astype("float32").transpose(2, 0, 1))
            return self.model([{"image": tensor, "height": height, "width": width}])[0]


def mask_rcnn_cfg(device: str = "cpu", threshold: float = MASK_RCNN_PERSON_THRESHOLD):
    from detectron2.config import get_cfg

    cfg = get_cfg()
    cfg.merge_from_file(str(mask_rcnn_config_path()))
    cfg.MODEL.WEIGHTS = ""
    cfg.MODEL.DEVICE = device
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.freeze()
    return cfg


class MaskRCNNPersonDetector:
    """Person boxes and masks from Mask R-CNN R50-FPN-3x (COCO), loaded from the converted safetensors file."""

    def __init__(self, device: str = "cpu", threshold: float = MASK_RCNN_PERSON_THRESHOLD, weights: Path | None = None):
        state = load_file(str(weights or verified_path("mask_rcnn_r50_fpn_3x_safetensors")))
        self.predictor = Detectron2Predictor(mask_rcnn_cfg(device, threshold), state)

    def instances(self, image_bgr: np.ndarray):
        return self.predictor(image_bgr)["instances"].to("cpu")

    def detect(self, image_rgb: np.ndarray) -> list[dict]:
        out = self.instances(np.ascontiguousarray(image_rgb[:, :, ::-1]))
        keep = out.pred_classes == COCO_PERSON
        return [
            {"box": box.numpy().astype(np.float64), "score": float(score), "mask": mask.numpy()}
            for box, score, mask in zip(out.pred_boxes.tensor[keep], out.scores[keep], out.pred_masks[keep])
        ]


class SCRFDFaceDetector:
    """InsightFace SCRFD-10GF face boxes, from the verified ONNX file on the CPU execution provider.

    ``threads`` (opt-in, for the slice diagnosis): onnxruntime's intra-op thread count, with one inter-op
    thread and sequential execution. None keeps onnxruntime's default, which sizes the pool to the host's cores.
    """

    def __init__(self, threshold: float = SCRFD_THRESHOLD, input_size: tuple[int, int] = SCRFD_INPUT_SIZE, threads: int | None = None):
        from insightface.model_zoo import get_model  # import insightface before onnxruntime (it disables telemetry)

        options = {}
        if threads is not None:
            import onnxruntime

            session_options = onnxruntime.SessionOptions()
            session_options.intra_op_num_threads = threads
            session_options.inter_op_num_threads = 1
            session_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
            options["sess_options"] = session_options
        self.model = get_model(str(verified_path("scrfd_10g")), providers=["CPUExecutionProvider"], **options)
        self.model.prepare(ctx_id=-1, input_size=input_size, det_thresh=threshold)
        self.input_size = input_size

    def detect(self, image_rgb: np.ndarray) -> list[dict]:
        boxes, _ = self.model.detect(np.ascontiguousarray(image_rgb[:, :, ::-1]), input_size=self.input_size, max_num=0)
        return [{"box": b[:4].astype(np.float64), "score": float(b[4])} for b in boxes]

    def detect_with_landmarks(self, image_rgb: np.ndarray) -> list[dict]:
        """``detect``'s boxes and scores from the same call, plus SCRFD's five landmarks, which align faces for ArcFace
        (Milestone 5's identity check). ``detect`` itself is unchanged."""
        boxes, landmarks = self.model.detect(np.ascontiguousarray(image_rgb[:, :, ::-1]), input_size=self.input_size, max_num=0)
        return [{"box": b[:4].astype(np.float64), "score": float(b[4]), "landmarks": k.astype(np.float64)} for b, k in zip(boxes, landmarks)]


def build_sam2_predictor(device: str = "cpu"):
    """SAM 2.1 Hiera-small video predictor; its checkpoint loads with torch.load(weights_only=True)."""
    from sam2.build_sam import build_sam2_video_predictor

    return build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_s.yaml", str(verified_path("sam2.1_hiera_small")), device=device)
