"""Descriptive detector checks: pairing, box overlap, hand detection and face identity."""

import numpy as np
import pytest

from density.masks import clips
from density.masks import weights as mask_weights
from density.metrics import detection, weights


def boxes(*rows):
    return np.array(rows, dtype=np.float64).reshape(-1, 4)


def test_greedy_pairs_take_the_highest_iou_first_and_use_each_box_once():
    ref = boxes([0, 0, 10, 10], [20, 0, 30, 10])
    cond = boxes([1, 0, 11, 10], [0, 0, 10, 10], [50, 50, 60, 60])
    assert detection.greedy_pairs(ref, cond) == [(0, 1)]  # ref 0 takes its exact match; cond 0 is then unused but ref 1 overlaps nothing
    assert detection.greedy_pairs(ref, boxes()) == [] and detection.best_ious(ref, boxes()) == [0.0, 0.0]
    assert detection.best_ious(boxes([0, 0, 10, 10]), boxes([5, 0, 15, 10])) == [pytest.approx(1 / 3)]


def test_compare_hands_counts_every_reference_box():
    ref = {"detector": "wilor", "boxes": [boxes([0, 0, 10, 10], [20, 0, 30, 10]), boxes(), boxes([0, 0, 10, 10])]}
    cond = {"detector": "wilor", "boxes": [boxes([0, 0, 10, 10]), boxes([0, 0, 5, 5]), boxes([5, 0, 15, 10])]}
    out = detection.compare(ref, cond, "manipulation")
    assert out["reference_boxes"] == 3 and out["reference_frames"] == 2
    assert out["box_overlap"]["value"] == pytest.approx((1.0 + 0.0 + 1 / 3) / 3)
    assert out["hand_detection"] == {"value": pytest.approx(1 / 3), "matched": 1}


def test_compare_faces_averages_cosine_over_frames_with_pairs():
    e = np.eye(512)
    ref = {"detector": "scrfd", "boxes": [boxes([0, 0, 10, 10]), boxes([0, 0, 10, 10])], "embeddings": [e[:1], e[:1]]}
    tilted = (e[0] + e[1]) / np.sqrt(2)
    cond = {"detector": "scrfd", "boxes": [boxes([0, 0, 10, 10]), boxes([40, 40, 50, 50])], "embeddings": [tilted[None], e[:1]]}
    out = detection.compare(ref, cond, "face")
    assert out["face_identity"] == {"value": pytest.approx(1 / np.sqrt(2)), "frames": 1}  # frame 1's faces do not overlap


def test_compare_records_not_applicable_with_a_reason():
    empty = {"detector": "scrfd", "boxes": [boxes(), boxes()], "embeddings": [np.zeros((0, 512))] * 2}
    out = detection.compare(empty, empty, "face")
    assert out["box_overlap"]["value"] is None and "no scrfd detection" in out["box_overlap"]["reason"]
    assert out["face_identity"]["value"] is None and out["face_identity"]["reason"]
    with pytest.raises(ValueError):
        detection.compare(empty, {"detector": "wilor", "boxes": [boxes(), boxes()]}, "face")


HAVE = all((mask_weights.ROOT / mask_weights.WEIGHTS[n].path).exists() for n in ("scrfd_10g",)) and (weights.ROOT / weights.WEIGHTS["arcface_w600k_r50"].path).exists()


@pytest.mark.skipif(not HAVE or not (clips.CLIPS_DIR / "face" / "frames").is_dir(), reason="SCRFD/ArcFace weights or the face stock clip are missing")
def test_a_face_clip_against_itself_is_identical():
    frames = clips.load_frames("face")[:3]
    detectors = detection.Detectors("cpu")
    ref = detection.detect_video(detectors, frames, "face")
    out = detection.compare(ref, detection.detect_video(detectors, frames, "face"), "face")
    assert out["box_overlap"]["value"] == 1.0 and out["face_identity"]["value"] == pytest.approx(1.0, abs=1e-12)
    assert all(e.shape[1] == 512 for e in ref["embeddings"]) and out["reference_boxes"] >= 3
    plain = detectors.get("scrfd").detect(frames[0])
    with_landmarks = detectors.get("scrfd").detect_with_landmarks(frames[0])
    assert len(plain) == len(with_landmarks) and all(np.array_equal(a["box"], b["box"]) and a["score"] == b["score"] for a, b in zip(plain, with_landmarks))
    assert with_landmarks[0]["landmarks"].shape == (5, 2)
