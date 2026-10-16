"""Region masks for one reference clip (protocol section 4).

    F  SCRFD face boxes on every frame -> SAM 2.1 tracks (registered association rule) -> union
    H  WiLoR hand boxes with handedness on every frame -> SAM 2.1 hand tracks, one side each; a track is kept
       only if a Hands23 hand box confirms it on a keyframe (IoU >= 0.5, either side) -> union of the kept tracks
    O  Hands23 on keyframes 0, 8, ..., 80: the objects paired with hands -> SAM 2.1 tracks;
       kept only on validated contact frames
    C  dilate(H_paired, 16) & dilate(O, 16) on validated contact frames
    B  Mask R-CNN person boxes on every frame -> SAM 2.1 tracks -> union

Hands23's hand and object boxes are matched to the SAM hand and object tracks
at IoU >= 0.5 (a hand preferring the track with its own side), which gives
each contact observation its tracked identities.
"""

import time
from dataclasses import asdict, dataclass, field

import numpy as np

from density.masks.contact import (
    ContactObservation,
    check_interval_masks,
    contact_intervals,
    contact_region,
    keyframes,
)
from density.masks.geometry import box_iou, mask_box
from density.masks.registered import FRAMES, HAND_CONFIRM_IOU, OBJECT_MATCH_IOU
from density.masks.tracking import TrackResult, same_side, track


@dataclass
class Region:
    mask: np.ndarray
    tracks: TrackResult | None = None
    detections: dict = field(default_factory=dict)  # frame -> list of detection dicts
    seconds: dict = field(default_factory=dict)  # component -> seconds
    confirmations: dict[int, list[int]] | None = None  # hand tracks: keyframes where a Hands23 hand box confirms them
    dropped: tuple[int, ...] = ()  # hand tracks no Hands23 hand box confirms; kept out of the mask

    def kept(self) -> list[int]:
        return [t for t in sorted(self.tracks.masks) if t not in self.dropped] if self.tracks is not None else []

    def backfill(self) -> np.ndarray | None:
        """Pixels of this region that only kept tracks before their first detection cover."""
        if self.tracks is None or not self.kept():
            return None
        return self.tracks.backfill_mask(self.kept()) & self.mask

    def _with_detection(self, event: dict) -> dict:
        """The event with the box and score of the detection behind it, so the audit can tell a lost hand from a stray box."""
        found = self.detections.get(event["frame"], [])
        if event["detection"] >= len(found):
            return event
        d = found[event["detection"]]
        score = d.get("score")
        return {**event, "box": [round(float(v), 1) for v in d["box"]], "score": None if score is None else round(float(score), 3)}

    def summary(self) -> dict:
        """Tracks, events (with their detections' boxes and scores) and side disagreements for the metadata."""
        out = {"pixels_per_frame": self.mask.reshape(self.mask.shape[0], -1).sum(axis=1).tolist(), "seconds": self.seconds}
        if self.tracks is not None:
            t = self.tracks
            out["detections_per_frame"] = {int(f): len(d) for f, d in self.detections.items()}
            out["tracks"] = {
                int(k): {
                    "side": t.sides.get(k),
                    "first_detection_frame": t.first_prompt(k),
                    "prompt_frames": sorted(t.prompts[k]),
                    "frames_before_first_detection": t.backfill_frames(k),
                    "kept": k not in self.dropped,
                    **({"hands23_confirmed_keyframes": self.confirmations[k]} if self.confirmations is not None else {}),
                }
                for k in sorted(t.masks)
            }
            out["events"] = [self._with_detection(e) for e in t.events]
            out["side_disagreements"] = t.side_disagreements
            out["association_rounds"] = t.rounds
        return out


def union(masks: dict[int, np.ndarray], shape) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    for m in masks.values():
        out |= m
    return out


def detect(detector_fn, frames: np.ndarray, frame_ids) -> tuple[dict, float]:
    start = time.perf_counter()
    found = {f: detector_fn(frames[f]) for f in frame_ids}
    return found, time.perf_counter() - start


def tracked_region(detector_fn, frames: np.ndarray, propagator, frame_ids=None) -> Region:
    """Detections on ``frame_ids`` (every frame by default), propagated and associated by SAM, then united.

    Detections that carry a ``side`` (WiLoR's hands) give their tracks a side; others have none.
    """
    frame_ids = range(frames.shape[0]) if frame_ids is None else frame_ids
    found, det_seconds = detect(detector_fn, frames, frame_ids)
    boxes = {f: [d["box"] for d in dets] for f, dets in found.items()}
    sides = {f: [d.get("side") for d in dets] for f, dets in found.items()}
    sam_before = propagator.seconds
    result = track(propagator, boxes, frames.shape[0], sides)
    return Region(union(result.masks, frames.shape[:3]), result, found, {"detector": det_seconds, "sam": propagator.seconds - sam_before})


def hand_track_confirmations(tracks: TrackResult, hands23: dict[int, list]) -> dict[int, list[int]]:
    """For each hand track, the keyframes where a Hands23 hand box of either side overlaps its mask box at IoU >= 0.5."""
    confirmations = {}
    for t in sorted(tracks.masks):
        confirmations[t] = []
        for f in sorted(hands23):
            box = mask_box(tracks.masks[t][f])
            if box is not None and any(box_iou(box, h.hand_box) >= HAND_CONFIRM_IOU for h in hands23[f]):
                confirmations[t].append(f)
    return confirmations


def confirm_hand_tracks(hands: Region, hands23: dict[int, list]) -> Region:
    """The registered confirmation rule: H keeps only the hand tracks a Hands23 hand box confirms; nothing is merged."""
    confirmations = hand_track_confirmations(hands.tracks, hands23)
    dropped = tuple(t for t, frames in confirmations.items() if not frames)
    kept = {t: m for t, m in hands.tracks.masks.items() if t not in dropped}
    return Region(union(kept, hands.mask.shape), hands.tracks, hands.detections, hands.seconds, confirmations, dropped)


def match_track(result: TrackResult, frame: int, box, side: str | None = None, tracks=None) -> tuple[int | None, float]:
    """The track a Hands23 box joins at ``frame``: IoU >= 0.5, the same-side track first, then the highest IoU.

    ``tracks`` limits the candidates (the kept hand tracks); by default every track is one.
    """
    qualifying = []
    for track_id in sorted(result.masks) if tracks is None else tracks:
        iou = box_iou(mask_box(result.masks[track_id][frame]), box)
        if iou >= OBJECT_MATCH_IOU:
            qualifying.append((not same_side(result.sides.get(track_id), side), -iou, track_id))
    if not qualifying:
        return None, 0.0
    _, neg_iou, best = min(qualifying)
    return best, -neg_iou


def manipulation_regions(frames: np.ndarray, propagator, hand_detector, hands23, confirm=confirm_hand_tracks) -> dict:
    """H, O and C, with the contact observations and intervals behind them."""
    shape = frames.shape[:3]
    hands = tracked_region(hand_detector.detect, frames, propagator)

    kf = keyframes(frames.shape[0])
    start = time.perf_counter()
    pairs = {f: hands23.detect(np.ascontiguousarray(frames[f][:, :, ::-1])) for f in kf}
    hands23_seconds = time.perf_counter() - start
    hands = confirm(hands, pairs)
    object_detections = {f: [{"box": p.object_box, "score": p.object_score} for p in found if p.object_box is not None] for f, found in pairs.items()}
    object_boxes = {f: [d["box"] for d in dets] for f, dets in object_detections.items()}
    sam_before = propagator.seconds
    objects = track(propagator, object_boxes, frames.shape[0]) if any(object_boxes.values()) else TrackResult({}, {})
    object_sam = propagator.seconds - sam_before

    observations, labels, disagreements = [], {}, []
    for f, found in pairs.items():
        for p in found:
            hand_track, hand_iou = match_track(hands.tracks, f, p.hand_box, p.side, hands.kept())
            object_track = match_track(objects, f, p.object_box)[0] if p.object_box is not None else None
            if hand_track is not None:
                observations.append(ContactObservation(f, hand_track, p.contact_state, object_track))
                track_side = hands.tracks.sides.get(hand_track)
                if not same_side(track_side, p.side):
                    disagreements.append({"frame": f, "hand_track": hand_track, "hands23_side": p.side, "track_side": track_side, "iou": hand_iou})
            labels.setdefault(f, []).append(
                {
                    "side": p.side,
                    "hand_score": round(p.hand_score, 3),
                    "contact_state": p.contact_state,
                    "hand_track": hand_track,
                    "object_track": object_track,
                    "object_score": None if p.object_score is None else round(p.object_score, 3),
                }
            )

    intervals = contact_intervals(observations, kf)
    O = np.zeros(shape, dtype=bool)
    C = np.zeros(shape, dtype=bool)
    for interval in intervals:
        hand_mask = hands.tracks.masks[interval.hand_track]
        object_mask = objects.masks[interval.object_track]
        check_interval_masks(interval, hand_mask, object_mask)
        if interval.masks_usable:
            valid = np.zeros(frames.shape[0], dtype=bool)
            valid[interval.start : interval.end + 1] = True
            O |= object_mask & valid[:, None, None]
            C |= contact_region(hand_mask, object_mask, valid)

    return {
        "H": hands,
        "O": Region(O, objects, object_detections, {"hands23": hands23_seconds, "sam": object_sam}),
        "C": Region(C),
        "hands23": pairs,
        "hands23_labels": labels,
        "hands23_side_disagreements": disagreements,
        "observations": [asdict(o) for o in observations],
        "intervals": intervals,
    }


def assert_frames(frames: np.ndarray) -> None:
    if frames.shape[0] != FRAMES or frames.dtype != np.uint8:
        raise ValueError(f"expected {FRAMES} uint8 frames, got {frames.shape} {frames.dtype}")
