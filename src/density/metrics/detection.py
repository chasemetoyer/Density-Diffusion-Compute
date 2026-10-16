"""Descriptive detector checks (protocol section 7): face identity, hand detection and box overlap with the reference.

These explain failures and never determine GO. Each runs the mask pipeline's registered detector and threshold on every
frame of the condition and of its reference, both decoded from their MP4s on the same host (the slice), and the
definitions are registered in density.metrics.registered (FACE_IDENTITY, HAND_DETECTION, BOX_OVERLAP). A check that
cannot be computed, because the reference has no detection, is recorded as not applicable with that reason.

ArcFace (buffalo_l's w600k_r50) is loaded on its own with insightface's model_zoo.get_model, not through FaceAnalysis,
which would also load the pack's landmark model and unpickle its mean shape. It runs on onnxruntime's CPU provider
with the same thread count as SCRFD.
"""

import numpy as np

from density.masks.geometry import box_iou
from density.masks.roi import FACE, MANIPULATION
from density.metrics.registered import DETECTION_MATCH_IOU, DETECTORS
from density.metrics.weights import verified_path


class Detectors:
    """The detectors one session needs, built on first use. ``cpu_threads`` as in density.masks.pipeline.Models."""

    def __init__(self, device: str = "cpu", cpu_threads: int | str | None = "quota"):
        from density.cpu_threads import quota_threads

        self.device = device
        self.cpu_threads = quota_threads() if cpu_threads == "quota" else cpu_threads
        self._models: dict = {}

    def _session_options(self) -> dict:
        if self.cpu_threads is None:
            return {}
        import onnxruntime

        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = self.cpu_threads
        options.inter_op_num_threads = 1
        options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        return {"sess_options": options}

    def get(self, name: str):
        if name not in self._models:
            if name == "scrfd":
                from density.masks.detectors import SCRFDFaceDetector

                self._models[name] = SCRFDFaceDetector(threads=self.cpu_threads)
            elif name == "arcface":
                from insightface.model_zoo import get_model
                from insightface.model_zoo.arcface_onnx import ArcFaceONNX

                model = get_model(str(verified_path("arcface_w600k_r50")), providers=["CPUExecutionProvider"], **self._session_options())
                if not isinstance(model, ArcFaceONNX):
                    raise RuntimeError(f"w600k_r50.onnx loaded as {type(model).__name__}, not ArcFace")
                model.prepare(ctx_id=-1)
                self._models[name] = model
            elif name == "wilor":
                from density.masks.wilor import WiLoRHandDetector

                self._models[name] = WiLoRHandDetector(self.device)
            elif name == "mask_rcnn":
                from density.masks.detectors import MaskRCNNPersonDetector

                self._models[name] = MaskRCNNPersonDetector(self.device)
            else:
                raise KeyError(name)
        return self._models[name]


def embed_faces(arcface, image_rgb: np.ndarray, faces: list[dict]) -> np.ndarray:
    """(faces, 512) float64 unit embeddings of SCRFD faces, aligned on their five landmarks."""
    from insightface.utils.face_align import norm_crop

    if not faces:
        return np.zeros((0, 512))
    bgr = np.ascontiguousarray(image_rgb[:, :, ::-1])
    crops = [norm_crop(bgr, landmark=f["landmarks"], image_size=arcface.input_size[0]) for f in faces]
    features = np.asarray(arcface.get_feat(crops), dtype=np.float64)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if not np.isfinite(features).all() or (norms == 0).any():
        raise FloatingPointError("ArcFace produced a non-finite or zero embedding")
    return features / norms


def detect_video(detectors: Detectors, frames: np.ndarray, stratum: str) -> dict:
    """Per-frame boxes (and face embeddings) of the stratum's detector on every frame."""
    name = DETECTORS[stratum]
    boxes, embeddings = [], []
    for frame in frames:
        if stratum == FACE:
            found = detectors.get("scrfd").detect_with_landmarks(frame)
            embeddings.append(embed_faces(detectors.get("arcface"), frame, found))
        else:
            found = detectors.get(name).detect(frame)
        boxes.append(np.array([d["box"] for d in found], dtype=np.float64).reshape(-1, 4))
    out = {"detector": name, "boxes": boxes}
    if stratum == FACE:
        out["embeddings"] = embeddings
    return out


def best_ious(reference: np.ndarray, condition: np.ndarray) -> list[float]:
    """Each reference box's best IoU with any condition box (0 without one)."""
    return [max((box_iou(r, c) for c in condition), default=0.0) for r in reference]


def greedy_pairs(reference: np.ndarray, condition: np.ndarray) -> list[tuple[int, int]]:
    """Reference and condition boxes paired by IoU >= 0.5, highest IoU first (ties by index); each box at most once."""
    candidates = sorted(
        ((-box_iou(r, c), i, j) for i, r in enumerate(reference) for j, c in enumerate(condition) if box_iou(r, c) >= DETECTION_MATCH_IOU)
    )
    used_r, used_c, pairs = set(), set(), []
    for _, i, j in candidates:
        if i not in used_r and j not in used_c:
            pairs.append((i, j))
            used_r.add(i)
            used_c.add(j)
    return sorted(pairs)


def compare(reference: dict, condition: dict, stratum: str) -> dict:
    """The descriptive checks of one condition against its reference, from ``detect_video`` outputs."""
    if reference["detector"] != condition["detector"] or len(reference["boxes"]) != len(condition["boxes"]):
        raise ValueError("reference and condition detections differ in detector or frame count")
    ious, frames_with_reference = [], 0
    for ref, cond in zip(reference["boxes"], condition["boxes"]):
        frames_with_reference += bool(len(ref))
        ious += best_ious(ref, cond)
    out = {"detector": reference["detector"], "reference_boxes": len(ious), "reference_frames": frames_with_reference}
    if not ious:
        reason = f"the reference has no {reference['detector']} detection on any frame"
        out["box_overlap"] = {"value": None, "reason": reason}
    else:
        out["box_overlap"] = {"value": float(np.mean(ious))}
    if stratum == MANIPULATION:
        matched = sum(i >= DETECTION_MATCH_IOU for i in ious)
        out["hand_detection"] = {"value": matched / len(ious), "matched": matched} if ious else {"value": None, "reason": out["box_overlap"]["reason"]}
    if stratum == FACE:
        per_frame = []
        for ref, cond, e_ref, e_cond in zip(reference["boxes"], condition["boxes"], reference["embeddings"], condition["embeddings"]):
            pairs = greedy_pairs(ref, cond)
            if pairs:
                per_frame.append(float(np.mean([float(np.dot(e_ref[i], e_cond[j])) for i, j in pairs])))
        out["face_identity"] = (
            {"value": float(np.mean(per_frame)), "frames": len(per_frame)}
            if per_frame
            else {"value": None, "frames": 0, "reason": "no frame has a reference face matched by a condition face at IoU >= 0.5"}
        )
    return out
