"""Milestone 4 check: the mask pipeline (protocol v2 sections 4 and 7).

Run ``python -m density.checks.masks`` to write
``results/checks/mask_pipeline.{md,json}``, ``mask_pipeline.sha256`` (hashes of
the contact sheets, which are not versioned) and the audit CSV
``mask_pipeline_audit.csv``. ``--logic-only`` runs only the model-free tests.

Logic tests T19-T25 and the T28 negative controls run on synthetic masks and
mocked detectors, so they need no models. Each check takes the code under
test as an argument, so a negative control can pass in a broken version and
show that the check fails it. The stock clips then run through the real
detectors and SAM 2.1 on CPU (T26, T27, timings); see ``configs/test_clips.json``.

T19  mask geometry (81 x 480 x 832, bool) and exact run-length round trip.
T20  contact intervals: same hand and object on adjacent keyframes, trimmed, never extrapolated.
T21  C = dilate(H, 16) & dilate(O, 16) on validated frames only; exact disk radii.
T22  SAM re-prompt rule fires exactly when IoU < 0.5; the association rule keeps identities through
     hands crossing, a stray false positive, a side flip and a lost track, and always settles.
T23  ROI assembly per stratum; controls are an explicit empty mask.
T24  eligibility edge cases, both-seeds rule, ordered replacement, exhausted reserves; audit rejections are
     mask invalidity only, each with a code and a written reason, read and checked from the audit CSV.
T25  measurement masks and frame sets come from the reference only, identical across conditions.
T28  negative controls: an extrapolated contact interval fails T20; the body mask as the
     manipulation ROI fails T23; today's ungated association, and the rule without each of its
     safeguards (the IoU >= 0.5 override, the lost-track exception, the bound-detection rule), fail T22.
T26  repeated CPU runs on the face and hands clips give identical masks; SAM 2.1 on MPS is reported.
T27  on the stock clips: F in at least 41 frames (face), H and a validated contact interval (hands),
     B in at least 41 frames (body), an explicit empty ROI (landscape); contact sheets for sign-off.
"""

import argparse
import hashlib
import json
import os
from importlib import metadata
from pathlib import Path

import numpy as np
import torch

from density.checks import euler
from density.masks import contact as contact_module
from density.masks import roi as roi_module
from density.masks.contact import ContactInterval, ContactObservation, contact_region, keyframes, restrict_to
from density.masks.eligibility import AUDIT_CODES, AuditRejection, ReplacementLog, ReserveExhausted, ReserveList, check_reference, prompt_passes
from density.masks.geometry import box_iou, dilate, disk_offsets, mask_box, rle_decode, rle_encode
from density.masks.measurement import measurement_masks
from density.masks.registered import FRAMES, HEIGHT, MIN_USABLE_PIXELS, REPROMPT_IOU, WIDTH
from density.masks.tracking import Association, Decision, needs_reprompt, track

SHAPE = (FRAMES, HEIGHT, WIDTH)


def row(test: str, passed: bool | None, **fields) -> dict:
    return {"test": test, "passed": None if passed is None else bool(passed), **fields}


def disk(shape, cy, cx, radius) -> np.ndarray:
    yy, xx = np.ogrid[: shape[-2], : shape[-1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2


def rect(shape, y0, x0, y1, x1) -> np.ndarray:
    m = np.zeros(shape[-2:], dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


# T19


def check_t19() -> list[dict]:
    rng = np.random.default_rng(19)
    video = rng.random(SHAPE) < 0.01
    cases = {
        "random": video,
        "all false": np.zeros(SHAPE, dtype=bool),
        "all true": np.ones(SHAPE, dtype=bool),
        "first pixel set": np.pad(np.ones((1, 1, 1), bool), [(0, FRAMES - 1), (0, HEIGHT - 1), (0, WIDTH - 1)]),
        "last pixel set": np.pad(np.ones((1, 1, 1), bool), [(FRAMES - 1, 0), (HEIGHT - 1, 0), (WIDTH - 1, 0)]),
    }
    exact = {name: bool(np.array_equal(rle_decode(rle_encode(m)), m)) for name, m in cases.items()}
    region = dilate(np.stack([disk(SHAPE, 240, 416, 50)] * FRAMES), 8)
    geometry = region.shape == SHAPE and region.dtype == bool
    return [
        row("T19 RLE round trip is exact", all(exact.values()), cases=exact),
        row("T19 region masks are 81 x 480 x 832 bool", geometry, shape=list(region.shape), dtype=str(region.dtype)),
    ]


# T20


T20_CASES = [
    # (name, [(keyframe, hand, contact_state, object)], expected [(hand, object, start, end)])
    ("three adjacent keyframes", [(0, 1, "object_contact", 7), (8, 1, "object_contact", 7), (16, 1, "object_contact", 7)], [(1, 7, 0, 16)]),
    ("one keyframe is not enough", [(16, 1, "object_contact", 7)], []),
    ("gap breaks adjacency", [(0, 1, "object_contact", 7), (16, 1, "object_contact", 7)], []),
    ("object must stay the same", [(0, 1, "object_contact", 7), (8, 1, "object_contact", 9)], []),
    ("hand must stay the same", [(0, 1, "object_contact", 7), (8, 2, "object_contact", 7)], []),
    ("trimmed when contact stops", [(8, 1, "object_contact", 7), (16, 1, "object_contact", 7), (24, 1, "object_contact", 7), (32, 1, "no_contact", None)], [(1, 7, 8, 24)]),
    ("self contact does not count", [(0, 1, "self_contact", 7), (8, 1, "self_contact", 7)], []),
    ("two separate runs", [(0, 1, "object_contact", 7), (8, 1, "object_contact", 7), (24, 1, "object_contact", 7), (32, 1, "object_contact", 7)], [(1, 7, 0, 8), (1, 7, 24, 32)]),
    ("ends at the last keyframe", [(72, 1, "object_contact", 7), (80, 1, "object_contact", 7)], [(1, 7, 72, 80)]),
]


def check_t20(intervals_fn=contact_module.contact_intervals) -> list[dict]:
    rows = []
    frames = keyframes()
    for name, observations, expected in T20_CASES:
        obs = [ContactObservation(f, hand, state, obj) for f, hand, state, obj in observations]
        got = [(i.hand_track, i.object_track, i.start, i.end) for i in intervals_fn(obs, frames)]
        rows.append(row(f"T20 {name}", got == expected, got=got, expected=expected))

    # Usable masks through the interval: one frame below 256 pixels invalidates it.
    hand = np.zeros(SHAPE, dtype=bool)
    hand[:, 100:120, 100:120] = True  # 400 pixels
    obj = hand.copy()
    obj[12] = False
    obj[12, 100:115, 100:117] = True  # 255 pixels
    interval = contact_module.check_interval_masks(ContactInterval(1, 7, 8, 16), hand, obj)
    full = contact_module.check_interval_masks(ContactInterval(1, 7, 8, 16), hand, hand)
    rows.append(row("T20 masks must be usable through the interval", interval.unusable_frames == [12] and full.masks_usable, unusable=interval.unusable_frames))
    return rows


def extrapolating_intervals(observations, frames):
    """Negative control for T20: extends every interval by one keyframe on each side."""
    out = []
    for i in contact_module.contact_intervals(observations, frames):
        out.append(ContactInterval(i.hand_track, i.object_track, max(0, i.start - 8), min(frames[-1], i.end + 8)))
    return out


# T21


def check_t21() -> list[dict]:
    rows = []
    counts = {r: len(disk_offsets(r)) for r in (8, 16, 32)}
    single = {}
    for r in (8, 16, 32):
        m = np.zeros((101, 101), dtype=bool)
        m[50, 50] = True
        expected = np.zeros_like(m)
        expected[50 + disk_offsets(r)[:, 0], 50 + disk_offsets(r)[:, 1]] = True
        single[r] = bool(np.array_equal(dilate(m, r), expected))
    rows.append(row("T21 dilation radii are exact (disk lattice points)", all(single.values()) and counts == {8: 197, 16: 797, 32: 3209}, disk_points=counts))

    hand = np.stack([disk(SHAPE, 240, 380, 40)] * FRAMES)
    obj = np.stack([disk(SHAPE, 240, 470, 35)] * FRAMES)
    valid = np.zeros(FRAMES, dtype=bool)
    valid[16:41] = True
    c = contact_region(hand, obj, valid)
    expected = np.zeros(SHAPE, dtype=bool)
    for f in range(16, 41):
        expected[f] = dilate(hand[f], 16) & dilate(obj[f], 16)
    o = restrict_to(obj, valid)
    ok = bool(np.array_equal(c, expected)) and not c[~valid].any() and c[valid].any() and not o[~valid].any() and np.array_equal(o[valid], obj[valid])
    rows.append(row("T21 C is the dilated intersection on validated frames only", ok, contact_pixels_per_frame=int(c[16].sum())))
    return rows


# T22


class DriftingPropagator:
    """Mock SAM: each track's mask is its latest prompt box, drifting right by ``drift`` px per frame."""

    def __init__(self, drift: int, shape=(FRAMES, 200, 400)):
        self.drift = drift
        self.shape = shape

    def propagate(self, prompts, num_frames):
        masks = {}
        for track_id, by_frame in prompts.items():
            m = np.zeros((num_frames, *self.shape[1:]), dtype=bool)
            for f in range(num_frames):
                earlier = [p for p in by_frame if p <= f]
                if not earlier:
                    continue
                start = max(earlier)
                x0, y0, x1, y1 = by_frame[start].astype(int)
                shift = self.drift * (f - start)
                m[f, y0:y1, max(0, x0 + shift) : max(0, x1 + shift)] = True
            masks[track_id] = m
        return masks


class InstancePropagator:
    """Mock SAM over a scene with several instances (two hands, or a hand and a stray false positive).

    ``truth[name][f]`` is instance ``name``'s true box at frame f (None when out of view). A prompt binds its track to
    the instance whose true box overlaps the prompt box most at that frame, or to a static phantom at the prompt box
    when none does. From each prompt until the next one, and before a track's first prompt back to frame 0, the mask
    is ``sam_box(name, prompt_frame, frame)`` (by default the true box), empty when that is None.
    """

    def __init__(self, truth: dict[str, list], sam_box=None, size=(200, 400)):
        self.truth = truth
        self.sam_box = sam_box or (lambda name, p, f: truth[name][f])
        self.size = size

    def bind(self, frame: int, box) -> str | None:
        best, best_iou = None, 0.0
        for name in sorted(self.truth):
            iou = box_iou(self.truth[name][frame], box)
            if iou > best_iou:
                best, best_iou = name, iou
        return best

    def identities(self, prompts) -> dict[int, list[str]]:
        """The instances each track's prompts were bound to; more than one is an identity switch."""
        return {t: sorted({self.bind(f, b) or "phantom" for f, b in by_frame.items()}) for t, by_frame in prompts.items()}

    def propagate(self, prompts, num_frames):
        masks = {}
        for t, by_frame in prompts.items():
            m = np.zeros((num_frames, *self.size), dtype=bool)
            starts = sorted(by_frame)
            for f in range(num_frames):
                p = max((s for s in starts if s <= f), default=starts[0])
                name = self.bind(p, by_frame[p])
                b = by_frame[p] if name is None else self.sam_box(name, p, f)
                if b is not None:
                    x0, y0, x1, y1 = (int(round(v)) for v in b)
                    m[f, max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)] = True
            masks[t] = m
        return masks


class EmptyAtPromptPropagator:
    """Mock SAM that returns an empty mask at each of a track's own prompt frames (and the prompt box elsewhere)."""

    size = (32, 64)

    def propagate(self, prompts, num_frames):
        masks = {}
        for t, by_frame in prompts.items():
            m = np.zeros((num_frames, *self.size), dtype=bool)
            x0, y0, x1, y1 = (int(v) for v in by_frame[min(by_frame)])
            m[:, y0:y1, x0:x1] = True
            m[sorted(by_frame)] = False
            masks[t] = m
        return masks


def hand_box(x0: float, y0: float = 80, size: int = 40) -> np.ndarray:
    return np.array([x0, y0, x0 + size, y0 + size], dtype=float)


def scene_detections(truth: dict[str, list], visible: dict[str, set], sides: dict[str, str], extra=None):
    """Detections of the visible instances at every frame (in name order), plus extra[frame] = [(box, side)]."""
    boxes, labels = {}, {}
    for f in range(FRAMES):
        found = [(truth[n][f], sides[n]) for n in sorted(truth) if truth[n][f] is not None and f in visible[n]]
        found += (extra or {}).get(f, [])
        boxes[f] = [b for b, _ in found]
        labels[f] = [s for _, s in found]
    return boxes, labels


def t22_scenes() -> dict:
    """Two-instance scenes for the association rule. Each gives a propagator, detections, sides and a judge."""
    every = set(range(FRAMES))
    scenes = {}

    # Hands crossing: 40 px hands move toward each other at 8 px per frame (R 2 px lower), and SAM lags 2 frames
    # behind the motion. At frame 18 the hands have just passed each other, and the left track's lagging mask
    # overlaps the right hand's detection (IoU 0.45) more than its own (IoU 0.43), both below 0.5.
    truth = {"L": [hand_box(44 + 8 * f) for f in range(FRAMES)], "R": [hand_box(330 - 8 * f, 82) for f in range(FRAMES)]}
    truth = {n: [b if b[0] >= 0 and b[2] <= 400 else None for b in boxes] for n, boxes in truth.items()}
    sam = lambda name, p, f, truth=truth: truth[name][max(p, f - 2)] if f >= p else truth[name][f]
    boxes, sides = scene_detections(truth, {"L": every, "R": every}, {"L": "left", "R": "right"})
    scenes["hands crossing"] = (
        InstancePropagator(truth, sam),
        boxes,
        sides,
        lambda r, ids: len(ids) == 2 and ids == {0: ["L"], 1: ["R"]},
    )

    # Stray false positive: the right hand is missed at frames 30 and 31, and the detector reports a "right" box far
    # away instead. It must get its own track; the hand's track must not jump to it.
    truth = {"R": [hand_box(200) for _ in range(FRAMES)]}
    fp = {f: [(np.array([20.0, 140.0, 60.0, 180.0]), "right")] for f in (30, 31)}
    boxes, sides = scene_detections(truth, {"R": every - {30, 31}}, {"R": "right"}, fp)
    scenes["stray false positive"] = (InstancePropagator(truth), boxes, sides, lambda r, ids: ids == {0: ["R"], 1: ["phantom"]})

    # Side flip: two partly overlapping hands (IoU 0.23). At frames 30-35 the right hand's detection is labelled
    # "left", and at 30-32 the left hand is missed. The flipped box overlaps the right track at IoU 1, so it belongs
    # to it: no new track, no re-prompt of the left track, and one side disagreement per flipped frame.
    flipped = set(range(30, 36))
    truth = {"L": [hand_box(150) for _ in range(FRAMES)], "R": [hand_box(175) for _ in range(FRAMES)]}
    boxes, sides = scene_detections(truth, {"L": every - {30, 31, 32}, "R": every}, {"L": "left", "R": "right"})
    for f in flipped:
        sides[f] = ["left" if np.array_equal(b, truth["R"][f]) else s for b, s in zip(boxes[f], sides[f])]

    def side_flip_judge(r, ids):
        logged = sorted((d["frame"], d["track"], d["side"]) for d in r.side_disagreements)
        return ids == {0: ["L"], 1: ["R"]} and logged == [(f, 1, "left") for f in sorted(flipped)]

    scenes["side flip"] = (InstancePropagator(truth), boxes, sides, side_flip_judge)

    # Lost track: SAM loses the right hand at frame 30 (empty mask until re-prompted); the hand is undetected at
    # 30-39 and reappears at frame 40, 100 px away. At frame 35 a stray "left" box appears near where the hand was.
    # The lost right track must not take the left box, and must be re-prompted by the hand's own detection at 40.
    truth = {"L": [hand_box(40) for _ in range(FRAMES)], "R": [hand_box(200 if f < 40 else 300) for f in range(FRAMES)]}
    sam = lambda name, p, f, truth=truth: None if name == "R" and p < 30 <= f else truth[name][f]
    fp = {35: [(hand_box(205, 140), "left")]}
    boxes, sides = scene_detections(truth, {"L": every, "R": every - set(range(30, 40))}, {"L": "left", "R": "right"}, fp)

    def lost_judge(r, ids):
        recovered = [e for e in r.events if e["frame"] == 40 and e["track"] == 1]
        return ids == {0: ["L"], 1: ["R"], 2: ["phantom"]} and [e["action"] for e in recovered] == ["reprompt_lost"]

    scenes["lost track"] = (InstancePropagator(truth, sam), boxes, sides, lost_judge)

    # Termination: SAM returns an empty mask at a track's own prompt frame. The detection that made the prompt still
    # belongs to its track, so association settles instead of starting a new track every round.
    boxes = {f: [np.array([8.0, 8.0, 40.0, 24.0])] if f == 5 else [] for f in range(FRAMES)}
    scenes["termination"] = (EmptyAtPromptPropagator(), boxes, None, lambda r, ids: len(r.prompts) == 1)
    return scenes


def check_t22_association(association=None, label: str = "T22") -> list[dict]:
    """The association cases; ``association`` defaults to the registered rule."""
    rows = []
    for name, (propagator, boxes, sides, judge) in t22_scenes().items():
        try:
            # 20 rounds is ample for the termination scene (the registered rule settles in 2) and keeps a control fast.
            result = track(propagator, boxes, FRAMES, sides, association, max_rounds=20 if name == "termination" else 100)
        except RuntimeError as error:  # did not settle
            rows.append(row(f"{label} {name}", False, error=str(error)))
            continue
        ids = propagator.identities(result.prompts) if isinstance(propagator, InstancePropagator) else {}
        rows.append(
            row(
                f"{label} {name}",
                judge(result, ids),
                identities={str(k): v for k, v in ids.items()},
                events=[(e["frame"], e["track"], e["action"]) for e in result.events if e["frame"] > 0][:12],
                side_disagreements=len(result.side_disagreements),
            )
        )
    return rows


def check_t22(association=None) -> list[dict]:
    rows = []
    det = np.array([0, 0, 90, 50], dtype=float)
    at = lambda shift: rect((200, 400), 0, shift, 50, 90 + shift)
    # Box width 90: a 30 px shift gives IoU exactly 60 / 120 = 0.5 (no re-prompt); 31 px gives just under.
    boundary = {
        "IoU 0.5 exactly": (box_iou(mask_box(at(30)), det), needs_reprompt(at(30), det)),
        "IoU just below 0.5": (box_iou(mask_box(at(31)), det), needs_reprompt(at(31), det)),
        "empty mask": (0.0, needs_reprompt(np.zeros((200, 400), bool), det)),
    }
    ok = boundary["IoU 0.5 exactly"] == (0.5, False) and boundary["IoU just below 0.5"][1] and boundary["IoU just below 0.5"][0] < REPROMPT_IOU and boundary["empty mask"][1]
    rows.append(row("T22 rule fires exactly when IoU < 0.5", ok, cases={k: [round(v[0], 6), v[1]] for k, v in boundary.items()}))

    # Drift 10 px per frame from a fresh prompt: shifts 10, 20, 30 keep IoU >= 0.5; 40 (IoU 50/130) fires.
    detections = {f: [det.copy()] for f in range(FRAMES)}
    result = track(DriftingPropagator(10), detections, FRAMES, association=association)
    fired = [entry["frame"] for entry in result.reprompts]
    expected = list(range(4, FRAMES, 4))
    rows.append(row("T22 re-prompts land exactly where IoU first drops below 0.5", fired == expected, reprompt_frames=fired[:6], count=len(fired)))

    steady = track(DriftingPropagator(0), detections, FRAMES, association=association)
    rows.append(row("T22 no re-prompt while SAM agrees with the detector", steady.reprompts == []))
    return rows + check_t22_association(association) + check_t22_confirmation()


class Hands23Hand:
    """A Hands23 hand for the confirmation tests: only its box and side matter."""

    def __init__(self, box, side: str):
        self.hand_box = np.asarray(box, dtype=float)
        self.side = side


def check_t22_confirmation(confirm=None) -> list[dict]:
    """Hand-track confirmation: a hand track stays in H only if a Hands23 hand box of either side overlaps its mask box
    at IoU >= 0.5 on a keyframe where it has a mask. Tracks are dropped, never merged."""
    from density.masks.regions import Region, confirm_hand_tracks, union
    from density.masks.tracking import TrackResult

    confirm = confirm or confirm_hand_tracks
    size = (FRAMES, 200, 400)
    masks = {t: np.zeros(size, dtype=bool) for t in range(5)}
    masks[0][10:, 50:90, 50:90] = True  # a hand from frame 10 on
    masks[1][:, 150:190, 300:340] = True  # a stray track no Hands23 box ever covers
    masks[2][:, 50:90, 200:240] = True  # confirmed only by a box with the other side label
    masks[3][:, 120:160, 100:140] = True  # covered only just below IoU 0.5
    masks[4][:, 120:160, 250:290] = True  # covered at IoU exactly 0.5
    hands23 = {f: [] for f in keyframes()}
    hands23[0] = [Hands23Hand([50, 50, 90, 90], "right")]  # where track 0 will be, before it has a mask
    hands23[16] = [Hands23Hand([50, 50, 90, 90], "right")]
    hands23[24] = [Hands23Hand([200, 50, 240, 90], "left")]
    hands23[32] = [Hands23Hand([100, 120, 140, 139.9], "right")]  # overlap 796 of 1,600 px: IoU 0.4975
    hands23[40] = [Hands23Hand([250, 120, 290, 140], "right")]  # overlap 800 of 1,600 px: IoU 0.5
    tracks = TrackResult(masks, {t: {10 if t == 0 else 0: None} for t in masks}, [], {t: "right" for t in masks})
    result = confirm(Region(union(masks, size), tracks), hands23)
    kept = result.kept()
    return [
        row("T22 hand tracks without a Hands23 confirmation are dropped", kept == [0, 2, 4] and result.dropped == (1, 3), kept=kept, dropped=list(result.dropped)),
        row("T22 confirmation counts only keyframes where the track has a mask", (result.confirmations or {}).get(0) == [16], track_0=(result.confirmations or {}).get(0)),
        row("T22 confirmation ignores the side label and needs IoU >= 0.5", 2 in kept and 4 in kept and 3 not in kept),
        row(
            "T22 dropped tracks leave H and are not merged",
            np.array_equal(result.mask, union({t: masks[t] for t in kept}, size)) and sorted(result.tracks.masks) == list(range(5)),
        ),
    ]


def keep_every_hand_track(hands, hands23):
    """Negative control for the confirmation rule: keeps every hand track."""
    from density.masks.regions import Region

    return Region(hands.mask, hands.tracks, hands.detections, hands.seconds, {t: [] for t in hands.tracks.masks}, ())


# Negative controls for the association rule: each replaces one step of tracking.Association.


class UngatedAssociation(Association):
    """The rule before the 2026-10-06 decision: greedy one-to-one by IoU with no minimum overlap and no sides.

    A matched track is re-prompted when IoU < 0.5; every unmatched detection starts a track.
    """

    def __call__(self, masks, state, frame, boxes, sides):
        pairs = sorted(
            ((box_iou(mask_box(masks[t][frame]), boxes[j]), t, j) for t in sorted(masks) for j in range(len(boxes))),
            key=lambda p: (-p[0], p[1], p[2]),
        )
        out, used = {}, set()
        for iou, t, j in pairs:
            if t not in used and j not in out:
                used.add(t)
                action = "reprompt" if frame not in state.prompts[t] and iou < REPROMPT_IOU else "match"
                out[j] = Decision(frame, j, action, t, iou, sides[j], state.sides.get(t))
        for j in range(len(boxes)):
            if j not in out:
                out[j] = Decision(frame, j, "new_track", None, 0.0, sides[j], sides[j])
        return [out[j] for j in sorted(out)]


class SideStrictAssociation(Association):
    """Side-anchored and gated, but without the IoU >= 0.5 override: only a same-side track can claim a detection."""

    def qualifies(self, ctx, t, j):
        return super().qualifies(ctx, t, j) and ctx.same(t, j)


class NoLostExceptionAssociation(Association):
    """Gated, but a lost track is never re-prompted by a detection it does not overlap."""

    def lost(self, ctx):
        pass


class UnboundAssociation(Association):
    """Forgets which detection made each prompt."""

    def bound(self, ctx):
        pass


ASSOCIATION_CONTROLS = {
    "today's ungated rule": UngatedAssociation,
    "side-anchored without the IoU >= 0.5 override": SideStrictAssociation,
    "gate without the lost-track exception": NoLostExceptionAssociation,
    "without the bound-detection rule": UnboundAssociation,
}


# T23


def check_t23(assemble=roi_module.assemble_roi) -> list[dict]:
    rng = np.random.default_rng(23)
    regions = {name: rng.random((9, 48, 64)) < 0.2 for name in ("F", "H", "C", "O", "B")}
    shape = (9, 48, 64)
    expected = {
        roi_module.FACE: regions["F"],
        roi_module.MANIPULATION: regions["H"] | regions["C"] | regions["O"],
        roi_module.BODY: regions["B"],
    }
    rows = [row(f"T23 {stratum} ROI", bool(np.array_equal(assemble(stratum, regions, shape), want))) for stratum, want in expected.items()]
    rows.append(row("T23 A1 mask is H | C", bool(np.array_equal(roi_module.a1_mask(regions), regions["H"] | regions["C"]))))
    control = assemble(roi_module.CONTROL, None, shape)
    try:
        assemble(roi_module.CONTROL, regions, shape)
        refuses = False
    except ValueError:
        refuses = True
    rows.append(row("T23 controls are an explicit empty mask, never detector-derived", not control.any() and control.shape == shape and refuses))
    return rows


def body_as_manipulation(stratum, regions, shape=SHAPE):
    """Negative control for T23: uses the person mask B as the manipulation ROI."""
    if stratum == roi_module.MANIPULATION:
        return regions["B"].copy()
    return roi_module.assemble_roi(stratum, regions, shape)


# T24


def frames_with_roi(count: int, pixels: int = MIN_USABLE_PIXELS) -> np.ndarray:
    roi = np.zeros(SHAPE, dtype=bool)
    side = int(np.ceil(np.sqrt(pixels)))
    for f in range(count):
        block = np.zeros(side * side, dtype=bool)
        block[:pixels] = True
        roi[f, 100 : 100 + side, 100 : 100 + side] = block.reshape(side, side)
    return roi


def check_t24() -> list[dict]:
    rows = []
    good_interval = [ContactInterval(1, 7, 8, 16)]

    def result(stratum, roi, intervals=None, audit=(), seed=1001):
        return check_reference(stratum, seed, roi, measurement_masks(roi), intervals, audit)

    r40, r41 = result("face", frames_with_roi(40)), result("face", frames_with_roi(41))
    rows.append(row("T24 40 usable frames fail, 41 pass", not r40.passed and r40.reasons == ["roi_usable_frames"] and r41.passed, usable=[r40.usable_frames, r41.usable_frames]))
    small = result("face", frames_with_roi(81, MIN_USABLE_PIXELS - 1))
    rows.append(row("T24 frames below 256 mask pixels are not usable", not small.passed and small.usable_frames == 0))

    no_contact = result("manipulation", frames_with_roi(81), [])
    unusable = ContactInterval(1, 7, 8, 16, unusable_frames=[12])
    bad_masks = result("manipulation", frames_with_roi(81), [unusable])
    with_contact = result("manipulation", frames_with_roi(81), good_interval)
    rows.append(
        row(
            "T24 manipulation needs a validated contact interval",
            no_contact.reasons == ["no_contact_interval"] and bad_masks.reasons == ["no_contact_interval"] and with_contact.passed,
        )
    )

    everywhere = np.ones(SHAPE, dtype=bool)
    rows.append(row("T24 a frame outside the ROI dilated by 32 px is required", result("face", everywhere).reasons == ["no_outside_frame"]))
    border = np.ones(SHAPE, dtype=bool)
    border[40, :, :] = False
    border[40, 0, 0] = True  # 32 px dilation of one corner pixel leaves the rest of frame 40 outside
    rows.append(row("T24 one outside frame is enough", result("face", border).passed))

    # Seeds follow the protocol's 1000 + ID and 2000 + ID scheme.
    passing, failing = result("face", frames_with_roi(41)), result("face", frames_with_roi(40))
    passing2, failing2 = result("face", frames_with_roi(41), seed=2001), result("face", frames_with_roi(40), seed=2001)
    rows.append(row("T24 both seeds must pass", prompt_passes([passing, passing2]) and not prompt_passes([passing, failing2]) and not prompt_passes([failing, passing2])))

    log = ReplacementLog()
    reserves = ReserveList("face", [501, 502], log)
    order = [reserves.replace(11, [failing, passing2]), reserves.replace(12, [passing, failing2])]
    try:
        reserves.replace(13, [failing, failing2])
        stops = False
    except ReserveExhausted:
        stops = True
    rows.append(
        row(
            "T24 ordered replacement, stop when a reserve list runs out",
            order == [501, 502] and stops and log.extra_cost_re == 4 and len(log.entries) == 2 and log.entries[0]["reasons"] == {1001: ["roi_usable_frames"], 2001: []},
            replacements=order,
            extra_cost_re=log.extra_cost_re,
        )
    )
    def refused(make) -> bool:
        try:
            make()
            return False
        except ValueError:
            return True

    rows.append(
        row(
            "T24 anatomy, image quality and appearance are never grounds for rejection",
            all(refused(lambda code=code: AuditRejection(code, "looks wrong")) for code in ("poor_anatomy", "image_quality", "appearance")),
        )
    )
    rows.append(row("T24 every audit rejection needs a written reason", refused(lambda: AuditRejection("audit_mask_invalid", "  "))))
    rejected = {code: result("face", frames_with_roi(81), audit=[AuditRejection(code, f"reviewer note for {code}")]) for code in AUDIT_CODES}
    rows.append(
        row(
            "T24 each mask-invalidity code fails the reference and records the written reason",
            all(not r.passed and r.reasons == [code] and r.written_reasons == {code: f"reviewer note for {code}"} for code, r in rejected.items()),
            codes=sorted(AUDIT_CODES),
        )
    )
    rows += check_t24_audit_csv()
    return rows


def check_t24_audit_csv() -> list[dict]:
    """The audit CSV: reviewer columns survive a format change, and a reject without a code or written reason is refused."""
    import csv
    import tempfile

    from density.masks.audit import AUDIT_COLUMNS, read_audit_csv, write_audit_csv

    rows = []
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "audit.csv"
        old_columns = [c for c in AUDIT_COLUMNS if c != "reviewer_written_reason"]  # the format before the column existed
        signed = [
            {"clip": "a", "seed": "1001", "contact_sheet": "a.png", "contact_sheet_sha256": "1", "reviewer_decision": "Accept", "reviewer": "CM", "review_date": "2026-10-06"},
            {"clip": "b", "seed": "1001", "contact_sheet": "b.png", "contact_sheet_sha256": "2"},
        ]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=old_columns, restval="")
            writer.writeheader()
            writer.writerows(signed)
        write_audit_csv([{k: v for k, v in r.items() if not k.startswith("review")} for r in signed], path)
        kept = {d.clip: (d.accepted, d.reviewer, d.review_date) for d in read_audit_csv(path)}
        rows.append(row("T24 existing sign-offs survive the written-reason column", kept == {"a": (True, "CM", "2026-10-06")}, kept=kept))

        def fill(decision, code, written):
            with open(path, newline="") as f:
                table = list(csv.DictReader(line for line in f if not line.startswith("#")))
            table[1].update(reviewer_decision=decision, reviewer_reason=code, reviewer_written_reason=written)
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=AUDIT_COLUMNS)
                writer.writeheader()
                writer.writerows(table)

        outcomes = {}
        for name, decision, code, written in (
            ("complete reject", "reject", "audit_mask_invalid", "mask spills onto the table"),
            ("reject without a written reason", "reject", "audit_mask_invalid", ""),
            ("reject without a code", "reject", "", "mask spills onto the table"),
            ("reject for anatomy", "reject", "poor_anatomy", "the hand has six fingers"),
            ("unknown decision", "maybe", "", ""),
        ):
            fill(decision, code, written)
            try:
                read = read_audit_csv(path)
                outcomes[name] = "read" if read[-1].rejection == AuditRejection(code, written) else "misread"
            except ValueError:
                outcomes[name] = "refused"
        expected = {"complete reject": "read", "reject without a written reason": "refused", "reject without a code": "refused", "reject for anatomy": "refused", "unknown decision": "refused"}
        rows.append(row("T24 the audit CSV refuses a reject without a mask-invalidity code and a written reason", outcomes == expected, outcomes=outcomes))
    return rows


# T25


class ReferenceMeasurements:
    """Measurement masks per (prompt, seed), computed once from the reference and shared by every condition."""

    def __init__(self):
        self._masks = {}

    def register(self, prompt_id: int, seed: int, reference_roi: np.ndarray):
        if (prompt_id, seed) in self._masks:
            raise RuntimeError("measurement masks are frozen once registered")
        self._masks[(prompt_id, seed)] = measurement_masks(reference_roi)

    def get(self, prompt_id: int, seed: int, condition: str):
        return self._masks[(prompt_id, seed)]


def check_t25() -> list[dict]:
    roi = np.zeros(SHAPE, dtype=bool)
    roi[10:21] = disk(SHAPE, 240, 416, 60)
    m = measurement_masks(roi)
    exact = np.array_equal(m.inside, dilate(roi, 8)) and np.array_equal(m.outside, ~dilate(roi, 32))
    frames = m.inside_frames == tuple(range(10, 21)) and m.outside_frames == tuple(range(FRAMES))
    store = ReferenceMeasurements()
    store.register(1, 1001, roi)
    per_condition = [store.get(1, 1001, c) for c in ("C1", "C2", "C3", "C4", "C5", "C6", "A1")]
    shared = all(x is per_condition[0] for x in per_condition)
    try:
        per_condition[0].inside[0, 0, 0] = True
        read_only = False
    except ValueError:
        read_only = True
    try:
        store.register(1, 1001, np.zeros(SHAPE, dtype=bool))
        frozen = False
    except RuntimeError:
        frozen = True
    return [
        row("T25 inside = ROI dilated by 8 px, outside = not ROI dilated by 32 px", exact),
        row("T25 frame sets from the reference", frames, inside_frames=[m.inside_frames[0], m.inside_frames[-1]], outside_frames=len(m.outside_frames)),
        row("T25 identical, read-only and frozen across conditions", shared and read_only and frozen),
    ]


# All logic checks, including the negative controls


def logic_checks() -> list[dict]:
    rows = check_t19() + check_t20() + check_t21() + check_t22() + check_t23() + check_t24() + check_t25()
    t20_bad = check_t20(extrapolating_intervals)
    rows.append(row("T28 extrapolated contact interval fails T20", not all(r["passed"] for r in t20_bad), failing=[r["test"] for r in t20_bad if not r["passed"]]))
    t23_bad = check_t23(body_as_manipulation)
    rows.append(row("T28 body mask as the manipulation ROI fails T23", not all(r["passed"] for r in t23_bad), failing=[r["test"] for r in t23_bad if not r["passed"]]))
    for name, control in ASSOCIATION_CONTROLS.items():
        t22_bad = check_t22_association(control(), label="T22")
        failing = [r["test"] for r in t22_bad if not r["passed"]]
        rows.append(row(f"T28 association {name} fails T22", bool(failing), failing=failing))
    t22_bad = check_t22_confirmation(keep_every_hand_track)
    failing = [r["test"] for r in t22_bad if not r["passed"]]
    rows.append(row("T28 keeping every hand track fails T22 (hand-track confirmation)", bool(failing), failing=failing))
    return rows


# Registered rule: every runtime weight file loads as data only


def check_data_only_loading(device: str = "cpu") -> dict:
    """Build every runtime model while full unpickling is disabled and torch.load is forced to weights_only."""
    import pickle

    import torch

    attempts = []
    saved = (pickle.load, pickle.loads, pickle.Unpickler, torch.load)

    def refuse(*args, **kwargs):
        attempts.append("pickle.load/loads")
        raise RuntimeError("full unpickle attempted at runtime")

    class RefusingUnpickler(pickle.Unpickler):
        def __init__(self, *args, **kwargs):
            attempts.append("pickle.Unpickler")
            raise RuntimeError("full unpickle attempted at runtime")

    def weights_only_load(*args, weights_only=None, **kwargs):
        if weights_only is False:
            attempts.append("torch.load(weights_only=False)")
            raise RuntimeError("torch.load(weights_only=False) attempted at runtime")
        return saved[3](*args, weights_only=True, **kwargs)

    built = {}
    pickle.load, pickle.loads, pickle.Unpickler, torch.load = refuse, refuse, RefusingUnpickler, weights_only_load
    try:
        from density.masks.detectors import MaskRCNNPersonDetector, SCRFDFaceDetector, build_sam2_predictor
        from density.masks.hands23 import Hands23Predictor
        from density.masks.wilor import WiLoRHandDetector

        for name, build in (
            ("SAM 2.1", lambda: build_sam2_predictor(device)),
            ("SCRFD", SCRFDFaceDetector),
            ("WiLoR", lambda: WiLoRHandDetector(device=device)),
            ("Mask R-CNN", lambda: MaskRCNNPersonDetector(device=device)),
            ("Hands23", lambda: Hands23Predictor(device=device)),
        ):
            try:
                build()
                built[name] = True
            except Exception as error:  # recorded, and fails the check
                built[name] = f"{type(error).__name__}: {error}"[:200]
        # Negative control: the original Model Zoo pickle, through Detectron2's own checkpointer, must be refused.
        from detectron2.engine import DefaultPredictor

        from density.masks.detectors import mask_rcnn_cfg
        from density.masks.weights import verified_path

        cfg = mask_rcnn_cfg(device).clone()
        cfg.defrost()
        cfg.MODEL.WEIGHTS = str(verified_path("mask_rcnn_r50_fpn_3x"))
        before = len(attempts)
        try:
            DefaultPredictor(cfg)
            control_refused = False
        except RuntimeError:
            control_refused = len(attempts) > before
        del attempts[before:]
    finally:
        pickle.load, pickle.loads, pickle.Unpickler, torch.load = saved
    return row(
        "Registered: every runtime weight file loads as data only",
        all(v is True for v in built.values()) and not attempts and control_refused,
        built=built,
        unpickle_attempts=attempts,
        original_pickle_refused=control_refused,
    )


def check_wilor_conversion(device: str = "cpu") -> dict:
    """WiLoR rebuilt from the data-only files must reproduce the original checkpoint's detections bitwise.

    The reference detections were saved by scripts/convert_wilor_detector.py from the original detector.pt
    (loaded the standard Ultralytics way) on five Hands23 demo images. Both are registered weight files
    (weights.py, under wilor/conversion_check/), so they are verified and travel to the pod with the weights.
    """
    import cv2

    from density.masks.weights import WEIGHTS, sha256_file, verified_path
    from density.masks.wilor import WiLoRHandDetector

    test = "Conversion: WiLoR rebuilt from data-only files gives the original checkpoint's detections bitwise"
    try:
        reference = verified_path("wilor_conversion_reference")
        images = {p.name: p for p in (verified_path(n) for n in WEIGHTS if n.startswith("wilor_conversion_image_"))}
    except (FileNotFoundError, RuntimeError) as error:
        return row(test, False, error=str(error))
    ref = np.load(reference, allow_pickle=False)
    detector = WiLoRHandDetector(device)
    equal, boxes = {}, 0
    for name in sorted({k.split(":")[0] for k in ref.files}):
        out = detector.detect_bgr(cv2.imread(str(images[name])))
        equal[name] = all(np.array_equal(out[field], ref[f"{name}:{field}"]) for field in ("boxes", "conf", "cls", "keypoints"))
        boxes += len(ref[f"{name}:boxes"])
    return row(test, all(equal.values()), images=equal, boxes=boxes, reference_sha256=sha256_file(reference))


# Stock clips: regions, storage, contact sheets, T26, T27, timings and the report

REPORT_DIR = euler.REPORT_DIR
SHEETS_DIR = REPORT_DIR / "mask_pipeline"
AUDIT_CSV = REPORT_DIR / "mask_pipeline_audit.csv"
HASH_FILE = REPORT_DIR / "mask_pipeline.sha256"
COMMITTED_REPORT = euler.REPORT_DIR / "mask_pipeline.json"  # the Mac's committed report: conform records and mask digests
DEVICE = "cpu"  # where the detectors and SAM run (--device); SCRFD always uses the onnxruntime CPU provider
FRAMES_ONLY = False  # --frames-only: verify the conformed frames against COMMITTED_REPORT instead of running ffmpeg
CONTACT_TEST = ("body",)  # informational Hands23 contact-transition test: kettlebell catches and releases
REPEAT_CPU = ("face", "hands")  # T26; hands exercises WiLoR, Hands23 and the object path
REPEAT_MPS = ("face",)
VERSION_PACKAGES = (
    "torch", "torchvision", "sam-2", "detectron2", "insightface", "onnxruntime", "onnx", "ultralytics", "hydra-core",
    "iopath", "fvcore", "scipy", "scikit-image", "opencv-python", "pillow", "safetensors", "numpy",
)
# Masks needed for "about 200 references" (protocol sections 6 and 12): pilot holdout 6 face + 6 manipulation prompts
# with 2 dense references each, both primary seeds of 36 + 36 prompts, and seed-1 full body and controls.
REFERENCE_COUNTS = {roi_module.FACE: 12 + 72, roi_module.MANIPULATION: 12 + 72, roi_module.BODY: 16, roi_module.CONTROL: 16}


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def package_versions() -> dict:
    out = {}
    for name in VERSION_PACKAGES:
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            out[name] = None
            continue
        version = dist.version
        direct = dist.read_text("direct_url.json")
        commit = json.loads(direct).get("vcs_info", {}).get("commit_id") if direct else None
        out[name] = f"{version} (git {commit})" if commit else version
    return out


def weight_records() -> list[dict]:
    from density.masks.weights import WEIGHTS, verified_path

    records = []
    for w in WEIGHTS.values():
        verified_path(w.name)  # size and SHA-256 checked
        records.append({"name": w.name, "path": w.path, "size": w.size, "sha256": w.sha256, "license": w.license, "source": w.source})
    return records


def registered_values() -> dict:
    from density.masks import registered

    return {k: getattr(registered, k) for k in dir(registered) if k.isupper()}


def mask_digest(result) -> dict:
    """Hashes of every mask and the decisions behind them, for exact repeat comparisons."""
    from density.masks.storage import mask_sha256

    digest = {
        "regions": {n: mask_sha256(r.mask) for n, r in sorted(result.regions.items())},
        "roi": mask_sha256(result.roi),
        "tracks": {n: {str(t): mask_sha256(m) for t, m in sorted(r.tracks.masks.items())} for n, r in sorted(result.regions.items()) if r.tracks},
        "events": {n: [(e["frame"], e["track"], e["action"]) for e in r.tracks.events] for n, r in sorted(result.regions.items()) if r.tracks},
    }
    if result.manipulation is not None:
        digest["intervals"] = [(i.hand_track, i.object_track, i.start, i.end) for i in result.manipulation["intervals"]]
    return digest


def interval_text(intervals) -> str:
    return "; ".join(f"H{i.hand_track}-O{i.object_track} frames {i.start}-{i.end}" + ("" if i.masks_usable else f" (unusable at {i.unusable_frames})") for i in intervals)


def hands23_notes(manipulation) -> dict[int, list[str]]:
    notes = {}
    for f, labels in manipulation["hands23_labels"].items():
        for l in labels:
            hand = f"H{l['hand_track']}" if l["hand_track"] is not None else "untracked"
            obj = f" O{l['object_track']}" if l["object_track"] is not None else ""
            notes.setdefault(int(f), []).append(f"Hands23 {l['side'][0].upper()} {l['hand_score']:.2f} -> {hand}: {l['contact_state']}{obj}")
    return notes


def pipeline_decision(stratum: str, eligibility) -> str:
    from density.masks.eligibility import PRIMARY

    if stratum not in PRIMARY:
        return "not gated"
    return "pass" if eligibility.passed else "fail: " + ", ".join(eligibility.reasons)


def write_sheets(clip: dict, frames: np.ndarray, result) -> dict[str, dict]:
    """The clip's contact sheet (its stratum ROI), and for a contact-test clip a second, informational sheet."""
    from density.masks.audit import contact_sheet, track_labels
    from density.masks.contact import validated_frames

    sheets = {}
    stratum, regions = clip["stratum"], result.regions
    m = result.manipulation
    valid, names, notes = None, {}, {}
    if m is not None:
        valid = validated_frames(m["intervals"])
        for i in m["intervals"]:
            if i.masks_usable:
                names.update({f: f"H{i.hand_track}-O{i.object_track}" for f in i.frames})
        notes = hands23_notes(m)
    decision = pipeline_decision(stratum, result.eligibility)
    roi_names = roi_module.REGIONS.get(stratum, ())
    plans = [(clip["id"], roi_names, result.coverage["ROI"], f"{clip['id']} ({stratum}): ROI = {' | '.join(roi_names) or 'explicit empty mask'}; pipeline eligibility: {decision}")]
    if clip["id"] in CONTACT_TEST and m is not None:
        plans.append((f"{clip['id']}_contact_test", ("H", "O", "C"), None, f"{clip['id']}: informational Hands23 contact-transition test (H, O, C are not this clip's ROI)"))
    for name, region_names, coverage, title in plans:
        shown = {n: regions[n] for n in region_names if n in regions}
        with_contact = m is not None and "H" in shown
        dropped = None
        if "H" in shown and shown["H"].dropped:
            dropped = np.zeros_like(shown["H"].mask)
            for t in shown["H"].dropped:
                dropped |= shown["H"].tracks.masks[t]
        path = contact_sheet(
            frames,
            {n: r.mask for n, r in shown.items()},
            coverage,
            SHEETS_DIR / f"{name}.png",
            title,
            backfill={n: b for n, r in shown.items() if (b := r.backfill()) is not None},
            labels=track_labels(shown),
            notes=notes if with_contact else None,
            valid=valid if with_contact else None,
            interval_names=names,
            dropped=dropped,
        )
        sheets[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return sheets


def clip_summary(clip: dict, result, conform: dict, sheets: dict, stored: dict) -> dict:
    from density.masks.pipeline import metadata as reference_metadata

    meta = reference_metadata(result)
    return {
        "id": clip["id"],
        "stratum": clip["stratum"],
        "frames_sha256": conform["frames_sha256"],
        "eligibility": meta["eligibility"],
        "pipeline_decision": pipeline_decision(clip["stratum"], result.eligibility),
        "roi_usable_frames": meta["roi_usable_frames"],
        "outside_frames": len(meta["measurement"]["outside_frames"]),
        "inside_frames": len(meta["measurement"]["inside_frames"]),
        "regions": meta["regions"],
        "contact": meta["contact"],
        "seconds": meta["seconds"],
        "sam": meta["sam"],
        "sheets": sheets,
        "stored": stored,
        "digest": mask_digest(result),
    }


def usable(mask: np.ndarray) -> int:
    from density.masks.eligibility import usable_frames

    return usable_frames(mask)


def check_t27(clips: dict) -> list[dict]:
    rows = []
    face, hands, body, landscape = (clips[k] for k in ("face", "hands", "body", "landscape"))
    rows.append(row("T27 face clip: F usable in at least 41 frames", face["eligibility"]["usable_frames"] >= 41, usable_frames=face["eligibility"]["usable_frames"]))
    intervals = [i for i in hands["contact"]["intervals"] if not i["unusable_frames"]]
    rows.append(
        row(
            "T27 hands clip: H usable in at least 41 frames and a validated contact interval",
            hands["eligibility"]["usable_frames"] >= 41 and bool(intervals),
            usable_frames=hands["eligibility"]["usable_frames"],
            intervals=[(i["hand_track"], i["object_track"], i["start"], i["end"]) for i in intervals],
        )
    )
    b = sum(1 for p in body["regions"]["B"]["pixels_per_frame"] if p >= MIN_USABLE_PIXELS)
    rows.append(row("T27 body clip: B usable in at least 41 frames", b >= 41, usable_frames=b))
    rows.append(
        row(
            "T27 landscape clip: the control ROI is the explicit empty mask, and nothing is computed",
            landscape["roi_usable_frames"] == 0 and not landscape["regions"] and not landscape["sam"],
        )
    )
    for clip in clips.values():
        rows.append(row(f"T27 contact sheet for sign-off: {clip['id']}", None, sheets=clip["sheets"]))
    return rows


def check_t26(clip_records: dict, results: dict) -> tuple[list[dict], dict]:
    """Repeat the face and hands clips on DEVICE with freshly built models; run SAM 2.1 on MPS twice on the face clip."""
    from density.masks.clips import load_frames
    from density.masks.geometry import box_iou as _iou
    from density.masks.pipeline import Models, process_reference

    rows, details = [], {}
    models = Models(DEVICE)
    for clip_id in REPEAT_CPU:
        clip = clip_records[clip_id]
        again = process_reference(load_frames(clip_id), clip["stratum"], models, contact_test=clip_id in CONTACT_TEST)
        first, second = mask_digest(results[clip_id]), mask_digest(again)
        rows.append(row(f"T26 repeated {DEVICE.upper()} run gives identical masks: {clip_id}", first == second, compared=sorted(first["regions"]), seconds=round(again.seconds["total"], 1)))
    if not torch.backends.mps.is_available():
        rows.append(row("T26 SAM 2.1 on MPS", None, error="MPS not available"))
        return rows, details
    for clip_id in REPEAT_MPS:
        clip = clip_records[clip_id]
        frames = load_frames(clip_id)
        try:
            runs = [process_reference(frames, clip["stratum"], models, sam_device="mps") for _ in range(2)]
        except Exception as error:  # reported, not judged
            rows.append(row(f"T26 SAM 2.1 on MPS: {clip_id}", None, error=f"{type(error).__name__}: {error}"[:300]))
            continue
        region = roi_module.REGIONS[clip["stratum"]][0]
        cpu = results[clip_id].regions[region].mask
        mps = [r.regions[region].mask for r in runs]
        per_frame = [
            float((cpu[f] & mps[0][f]).sum() / max(1, (cpu[f] | mps[0][f]).sum())) for f in range(cpu.shape[0])
        ]
        info = {
            "mps_runs_bitwise_equal": bool(np.array_equal(mps[0], mps[1])),
            "mps_run_to_run_differing_pixels": int((mps[0] ^ mps[1]).sum()),
            "mps_vs_cpu_differing_pixels": int((cpu ^ mps[0]).sum()),
            "mps_vs_cpu_mask_iou_min": min(per_frame),
            "mps_vs_cpu_mask_iou_mean": float(np.mean(per_frame)),
            "mps_vs_cpu_box_iou_min": min(_iou(mask_box(cpu[f]), mask_box(mps[0][f])) for f in range(cpu.shape[0]) if cpu[f].any() or mps[0][f].any()),
            "sam_seconds_mps": [round(r.sam["seconds"], 1) for r in runs],
            "sam_seconds_cpu": round(results[clip_id].sam["seconds"], 1),
            "events_equal_to_cpu": mask_digest(runs[0])["events"] == mask_digest(results[clip_id])["events"],
        }
        details[clip_id] = info
        rows.append(row(f"T26 SAM 2.1 on MPS (detectors on CPU): {clip_id}", None, **info))
    return rows, details


def verified_frames_record(clip_id: str) -> dict:
    """The committed conform record, after checking the clip's frames on disk against its frames_sha256."""
    from density.masks.clips import load_frames

    record = json.loads(COMMITTED_REPORT.read_text())["clips"][clip_id]["conform"]
    digest = hashlib.sha256(load_frames(clip_id).tobytes()).hexdigest()
    if digest != record["frames_sha256"]:
        raise RuntimeError(f"{clip_id}: frames on disk ({digest[:12]}) differ from the committed conform record ({record['frames_sha256'][:12]})")
    return {**record, "verified_against": str(COMMITTED_REPORT)}


def run_clips(clip_ids=None) -> tuple[dict, dict, dict, dict]:
    """Conform (or verify) each registered clip, run its stratum's pipeline on DEVICE, store masks and write contact sheets."""
    from density.masks.audit import write_audit_csv
    from density.masks.clips import CLIPS_DIR, conform, load_frames, registry, tool_versions
    from density.masks.pipeline import Models, process_reference, save

    reg = registry()
    clip_records = {c["id"]: c for c in reg["clips"] if clip_ids is None or c["id"] in clip_ids}
    versions = {"packages": package_versions(), "tools": tool_versions(), "git": euler.git_state(), "gpu": euler.gpu_info()}
    models = Models(DEVICE)
    conforms, results, summaries, audit_rows = {}, {}, {}, []
    for clip_id, clip in clip_records.items():
        conforms[clip_id] = verified_frames_record(clip_id) if FRAMES_ONLY else conform(clip)
        frames = load_frames(clip_id)
        result = process_reference(frames, clip["stratum"], models, contact_test=clip_id in CONTACT_TEST)
        results[clip_id] = result
        sheets = write_sheets(clip, frames, result)
        stored = save(
            result,
            CLIPS_DIR / clip_id / "masks",
            {"clip": {k: clip[k] for k in ("id", "stratum", "page", "license", "attribution", "sha256", "segment_start_s")}, "conform": conforms[clip_id], "versions": versions, "registered": jsonable(registered_values())},
        )
        summaries[clip_id] = clip_summary(clip, result, conforms[clip_id], sheets, stored)
        e = result.eligibility
        audit_rows.append(
            {
                "clip": clip_id,
                "seed": "",
                "stratum": clip["stratum"],
                "pipeline_decision": pipeline_decision(clip["stratum"], e).split(":")[0],
                "pipeline_reasons": "; ".join(e.reasons),
                "usable_roi_frames": e.usable_frames if e.usable_frames is not None else usable(result.roi),
                "contact_intervals": interval_text(result.manipulation["intervals"]) if result.manipulation and clip["stratum"] == roi_module.MANIPULATION else "",
                "outside_frames": len(result.measurement.outside_frames),
                "contact_sheet": sheets[clip_id]["path"],
                "contact_sheet_sha256": sheets[clip_id]["sha256"],
            }
        )
        print(f"{clip_id}: {result.seconds['total']:.0f} s, eligibility {e.passed} {e.reasons}", flush=True)
    write_audit_csv(audit_rows, AUDIT_CSV)
    return clip_records, results, summaries, {"conform": conforms, "versions": versions, "model_build_seconds": models.build_seconds}


def timing_table(summaries: dict, build_seconds: dict) -> dict:
    per_clip = {cid: {k: round(v, 1) for k, v in s["seconds"].items()} for cid, s in summaries.items()}
    by_stratum = {}
    for cid, s in summaries.items():
        if cid in CONTACT_TEST:  # the contact test is not part of a body reference's cost
            seconds = s["seconds"].get("B.detector", 0) + s["seconds"].get("B.sam", 0) + s["seconds"]["roi_coverage_measurement_eligibility"]
        else:
            seconds = s["seconds"]["total"]
        by_stratum[s["stratum"]] = seconds
    control_seconds = by_stratum.get(roi_module.CONTROL, 0.0)
    estimate = {
        stratum: {"references": n, "seconds_per_reference": round(by_stratum.get(stratum, 0.0), 1), "hours": round(n * by_stratum.get(stratum, 0.0) / 3600, 2)}
        for stratum, n in REFERENCE_COUNTS.items()
    }
    estimate[roi_module.CONTROL]["note"] = f"nothing computed: the explicit empty mask ({control_seconds:.1f} s for the ROI, coverage and measurement masks)"
    two_hands = {}
    for cid in (c for c in CONTACT_TEST if c in summaries):
        sec = summaries[cid]["seconds"]
        two_hands[cid] = round(sum(sec.get(k, 0.0) for k in ("H.detector", "H.sam", "O.hands23", "O.sam")), 1)
    if two_hands:  # the manipulation path on a two-hand clip bounds a manipulation reference's cost from above
        high = max(two_hands.values())
        e = estimate[roi_module.MANIPULATION]
        e["seconds_per_reference_high"] = high
        e["hours_high"] = round(e["references"] * high / 3600, 2)
        e["note"] = "low: the one-hand gyroscope clip; high: the manipulation path on the two-hand kettlebell clip"
    return {
        "per_clip_seconds": per_clip,
        "manipulation_path_on_contact_test_seconds": two_hands,
        "model_build_seconds": {k: round(v, 1) for k, v in build_seconds.items()},
        "mac_cpu_estimate": estimate,
        "mac_cpu_total_hours": round(sum(v["hours"] for k, v in estimate.items() if k != roi_module.CONTROL), 2),
        "mac_cpu_total_hours_high": round(sum(v.get("hours_high", v["hours"]) for k, v in estimate.items() if k != roi_module.CONTROL), 2),
        "basis": "Apple M4 Pro CPU, one clip at a time; the pod's GPU timings must be measured there",
    }


def run_all(logic_only: bool = False) -> dict:
    rows = logic_checks()
    report = {"milestone": "Mask pipeline: regions, contact intervals, ROI, measurement masks, eligibility, audit", "environment": euler.environment()}
    if not logic_only:
        rows.append(check_data_only_loading("cpu"))
        rows.append(check_wilor_conversion("cpu"))
        clip_records, results, summaries, extra = run_clips()
        rows += check_t27(summaries)
        t26_rows, mps = check_t26(clip_records, results)
        rows += t26_rows
        from density.masks.hands23 import PATCH, VENDOR_COMMIT

        report.update(
            {
                "versions": extra["versions"],
                "hands23_vendored": {"commit": VENDOR_COMMIT, "patch": os.path.relpath(PATCH), "patch_sha256": hashlib.sha256(PATCH.read_bytes()).hexdigest()},
                "weights": weight_records(),
                "clips": {cid: {**{k: c[k] for k in ("stratum", "title", "page", "author", "license", "attribution", "restrictions", "sha256", "segment_start_s")}, "conform": extra["conform"][cid]} for cid, c in clip_records.items()},
                "clip_results": summaries,
                "timings": timing_table(summaries, extra["model_build_seconds"]),
                "t26_mps": mps,
                "audit_csv": str(AUDIT_CSV),
            }
        )
    judged = [r for r in rows if r["passed"] is not None]
    report.update(
        {
            "passed": all(r["passed"] for r in judged) and not logic_only,
            "all_judged_passed": all(r["passed"] for r in judged),
            "logic_only": logic_only,
            "num_judged": len(judged),
            "num_failed": sum(not r["passed"] for r in judged),
            "registered": jsonable(registered_values()),
            "results": rows,
        }
    )
    return jsonable(report)


def _status(passed) -> str:
    return "info" if passed is None else ("PASS" if passed else "FAIL")


def markdown_report(report: dict) -> str:
    env = report["environment"]
    if report["logic_only"]:
        overall = "INCOMPLETE (logic tests only)" if report["all_judged_passed"] else "FAIL"
    else:
        overall = "PASS" if report["passed"] else "FAIL"
    lines = [
        "# Mask pipeline",
        "",
        f"**Overall: {overall}** ({report['num_judged'] - report['num_failed']}/{report['num_judged']} judged checks passed)",
        "",
        f"Code `{env['git_sha']}` (dirty: {env['git_dirty']}); torch {env['torch']}, {env['platform']}.",
        "",
        "| Test | Result |",
        "| --- | --- |",
    ]
    lines += [f"| {r['test']} | {_status(r['passed'])} |" for r in report["results"]]
    t28 = [r for r in report["results"] if r["test"].startswith("T28 association")]
    if t28:
        lines += ["", "Association negative controls and the T22 cases each one fails:", ""]
        lines += [f"- {r['test'][len('T28 association '):-len(' fails T22')]}: {', '.join(t[len('T22 '):] for t in r['failing'])}" for r in t28]

    reg = report["registered"]
    lines += [
        "",
        "## Registered values",
        "",
        f"- Geometry {reg['WIDTH']} x {reg['HEIGHT']}, {reg['FRAMES']} frames at {reg['FPS']} fps. Keyframes every {reg['KEYFRAME_SPACING']} frames.",
        f"- Eligibility: usable primary ROI (at least {reg['MIN_USABLE_PIXELS']} mask pixels) in at least {reg['MIN_USABLE_FRAMES']} of {reg['FRAMES']} frames; "
        f"token coverage threshold {reg['ROI_THRESHOLD']}.",
        f"- Dilations: contact {reg['CONTACT_DILATION_PX']} px, measurement {reg['MEASURE_DILATION_PX']} px, outside {reg['OUTSIDE_DILATION_PX']} px; "
        f"structuring element: {reg['STRUCTURING_ELEMENT']}.",
        f"- Re-prompt when IoU < {reg['REPROMPT_IOU']}, only from a detection with IoU > {reg['REPROMPT_GATE_IOU']}; a track is lost at "
        f"{reg['LOST_TRACK_PIXELS']} mask pixels; Hands23 boxes join tracks at IoU >= {reg['OBJECT_MATCH_IOU']}; contact state `{reg['CONTACT_STATE']}`.",
        f"- Association rule (2026-10-06): {reg['ASSOCIATION_RULE']}",
        f"- Hands23 hand matching: {reg['HANDS23_HAND_MATCH']}",
        f"- Hand-track confirmation (2026-10-06): {reg['HAND_TRACK_CONFIRMATION']}",
        f"- Regions per stratum: {reg['STRATUM_REGIONS']}",
        f"- Frames before a track's first detection: {reg['BACKFILL']}",
        f"- {reg['SUSTAINED_CONTACT']}",
        f"- SAM 2.1 precision: {reg['SAM_PRECISION']} JPEG frames at quality {reg['SAM_JPEG_QUALITY']}, 4:4:4.",
        f"- SAM 2.1 build: {reg['SAM_BUILD']}",
        f"- Detector thresholds: SCRFD {reg['SCRFD_THRESHOLD']} at {reg['SCRFD_INPUT_SIZE']} (onnxruntime CPU provider on both machines), "
        f"WiLoR {reg['WILOR_THRESHOLD']}, Mask R-CNN person {reg['MASK_RCNN_PERSON_THRESHOLD']}, Hands23 {reg['HANDS23_THRESHOLDS']}.",
        "- Every runtime weight file loads as data only (safetensors, ONNX, or torch.load(weights_only=True)).",
    ]
    if report["logic_only"]:
        return "\n".join(lines) + "\n"

    lines += ["", "## Clips", "", "| Clip | Stratum | Source | License | Segment | Source frames | Frames SHA-256 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for cid, c in report["clips"].items():
        cf = c["conform"]
        lines.append(
            f"| {cid} | {c['stratum']} | [{c['title']}]({c['page']}), {c['author']} | {c['license']} | {c['segment_start_s']:g} s | "
            f"{cf['source_frames'][0]}-{cf['source_frames'][-1]} ({cf['source_times_s'][0]:.3f}-{cf['source_times_s'][-1]:.3f} s) | `{cf['frames_sha256'][:16]}` |"
        )
    tools = report["versions"]["tools"]
    resize = next(iter(report["clips"].values()))["conform"]["ffmpeg_filter"].rsplit("',", 1)[1]
    lines += ["", f"Conformed with {tools['ffmpeg']}: the nearest source frame to each 1/16 s step (no interpolation), then `{resize}`."]

    lines += [
        "",
        "## Results per clip",
        "",
        "| Clip | Stratum | Tracks (side, first detection) | Re-prompts | New tracks after frame 0 | Side disagreements | Frames before first detection | ROI usable | Outside frames | Eligibility |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for cid, s in report["clip_results"].items():
        for name, region in s["regions"].items():
            if "tracks" not in region:
                continue
            tracks = ", ".join(
                f"{name}{t} ({v['side'] or '-'}, {v['first_detection_frame']}{'' if v.get('kept', True) else ', dropped'})" for t, v in region["tracks"].items()
            ) or "none"
            reprompts = sum(e["action"].startswith("reprompt") for e in region["events"])
            new = sum(e["action"] == "new_track" and e["frame"] > 0 for e in region["events"])
            backfill = sum(len(v["frames_before_first_detection"]) for v in region["tracks"].values())
            primary = {"face": "F", "manipulation": "H", "body": "B"}.get(s["stratum"]) == name
            stratum = s["stratum"] if name in roi_module.REGIONS.get(s["stratum"], ()) else f"{s['stratum']} (contact test)"
            eligibility = s["pipeline_decision"]
            lines.append(
                f"| {cid} | {stratum} | {tracks} | {reprompts} | {new} | {len(region['side_disagreements'])} | {backfill} | "
                + (f"{s['roi_usable_frames']} | {s['outside_frames']} | {eligibility} |" if primary else " | | |")
            )
        if not any("tracks" in r for r in s["regions"].values()):
            lines.append(f"| {cid} | {s['stratum']} | nothing computed (explicit empty mask) | | | | | {s['roi_usable_frames']} | {s['outside_frames']} | {s['pipeline_decision']} |")

    for cid, s in report["clip_results"].items():
        c = s["contact"]
        if c is None:
            continue
        label = "informational contact-transition test" if cid in CONTACT_TEST else "manipulation reference"
        lines += ["", f"### {cid}: Hands23 contact ({label})", "", "| Keyframe | Hands23 hands: side, score -> hand track: contact state, object track |", "| --- | --- |"]
        for f, labels in c["hands23_labels"].items():
            text = "; ".join(
                f"{l['side'][0].upper()} {l['hand_score']:.2f} -> {('H' + str(l['hand_track'])) if l['hand_track'] is not None else 'untracked'}: {l['contact_state']}"
                + (f" O{l['object_track']}" if l["object_track"] is not None else "")
                for l in labels
            )
            lines.append(f"| {f} | {text or 'no hands'} |")
        intervals = c["intervals"]
        hand_tracks = s["regions"]["H"]["tracks"]
        confirmed = "; ".join(f"H{t} at keyframes {', '.join(map(str, v['hands23_confirmed_keyframes']))}" for t, v in hand_tracks.items() if v["kept"])
        dropped = [f"H{t} (started at frame {v['first_detection_frame']})" for t, v in hand_tracks.items() if not v["kept"]]
        lines += [
            "",
            f"Hand tracks confirmed by a Hands23 hand box: {confirmed or 'none'}. Dropped without confirmation: {', '.join(dropped) or 'none'}.",
            "",
            "Contact intervals: "
            + ("; ".join(f"H{i['hand_track']}-O{i['object_track']} frames {i['start']}-{i['end']}" + (f" (unusable masks at {i['unusable_frames']})" if i["unusable_frames"] else "") for i in intervals) or "none")
            + f". Hands23 side disagreements with the hand tracks: {len(c['hands23_side_disagreements'])}.",
        ]
        if cid in CONTACT_TEST and not intervals:
            lines.append(
                "No interval validates: contact must be seen at two adjacent keyframes (0.5 s apart) with the same hand and object tracks, "
                "and brief catches between keyframes cannot satisfy that (registered as sustained contact)."
            )

    t = report["timings"]
    lines += ["", "## Timings (Mac CPU, per 81-frame clip)", "", "| Clip | Component seconds | SAM calls |", "| --- | --- | --- |"]
    for cid, sec in t["per_clip_seconds"].items():
        parts = ", ".join(f"{k} {v:g}" for k, v in sec.items() if k not in ("detect_track",))
        lines.append(f"| {cid} | {parts} | {report['clip_results'][cid]['sam'].get('calls', 0)} |")
    lines += [
        "",
        f"Model builds (once per session): {', '.join(f'{k} {v:g} s' for k, v in t['model_build_seconds'].items())}. "
        f"The manipulation path on the two-hand contact-test clip took {', '.join(f'{v:g} s ({k})' for k, v in t['manipulation_path_on_contact_test_seconds'].items())}.",
        "",
        "| Stratum | References | Seconds per reference | Hours | Note |",
        "| --- | --- | --- | --- | --- |",
    ]
    for stratum, e in t["mac_cpu_estimate"].items():
        seconds = f"{e['seconds_per_reference']:g}" + (f"-{e['seconds_per_reference_high']:g}" if "seconds_per_reference_high" in e else "")
        hours = f"{e['hours']:g}" + (f"-{e['hours_high']:g}" if "hours_high" in e else "")
        lines.append(f"| {stratum} | {e['references']} | {seconds} | {hours} | {e.get('note', '')} |")
    lines += [
        "",
        f"About {t['mac_cpu_total_hours']:g}-{t['mac_cpu_total_hours_high']:g} hours for the masks of about 200 references at Mac CPU "
        f"speed ({t['basis']}), depending on how many manipulation references look like the two-hand clip. SAM 2.1 dominates: every "
        "association round propagates every track again, so each extra track, re-prompt or new track adds a full pass, and passes slow "
        "down as tracks accumulate.",
    ]

    lines += ["", "## T26: repeatability", ""]
    lines += [f"- {r['test']}: {_status(r['passed'])}" for r in report["results"] if r["test"].startswith("T26 repeated")]
    for cid, info in report["t26_mps"].items():
        lines += [
            f"- SAM 2.1 on MPS, {cid} clip (detectors on CPU): two MPS runs bitwise equal: {info['mps_runs_bitwise_equal']} "
            f"({info['mps_run_to_run_differing_pixels']} differing pixels); against CPU, {info['mps_vs_cpu_differing_pixels']} differing pixels, "
            f"per-frame mask IoU min {info['mps_vs_cpu_mask_iou_min']:.6f}, mean {info['mps_vs_cpu_mask_iou_mean']:.7f}; same re-prompt events: {info['events_equal_to_cpu']}. "
            f"SAM time {info['sam_seconds_mps']} s on MPS vs {info['sam_seconds_cpu']} s on CPU."
        ]

    lines += ["", "## Versions and weights", ""]
    lines += [f"- {k}: {v}" for k, v in report["versions"]["packages"].items()]
    lines += [f"- {tools['ffprobe']}", f"- Hands23 vendored at {report['hands23_vendored']['commit']}, patch `{report['hands23_vendored']['patch']}` (SHA-256 `{report['hands23_vendored']['patch_sha256'][:16]}`)"]
    lines += ["", "| Weight file | Size | SHA-256 | License |", "| --- | --- | --- | --- |"]
    lines += [f"| {w['name']} | {w['size']:,} | `{w['sha256'][:16]}` | {w['license']} |" for w in report["weights"]]

    lines += ["", "## Contact sheets and audit", "", "Sheets are not versioned; their hashes are in the JSON and `mask_pipeline.sha256`.", ""]
    for cid, s in report["clip_results"].items():
        for name, sheet in s["sheets"].items():
            lines.append(f"- `{sheet['path']}` `{sheet['sha256'][:16]}`")
    lines += [
        "",
        f"Reviewer sign-off goes in `{report['audit_csv']}` (accept or reject; a rejection names one of the permitted audit reasons). "
        "A rerun keeps a reviewer's entries only for sheets whose hash is unchanged.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    global REPORT_DIR, SHEETS_DIR, AUDIT_CSV, HASH_FILE, DEVICE, FRAMES_ONLY
    parser = argparse.ArgumentParser(description="Mask pipeline check")
    parser.add_argument("--logic-only", action="store_true", help="run only the model-free tests")
    parser.add_argument("--device", default="cpu", help="device for SAM 2.1, WiLoR, Hands23 and Mask R-CNN (default: cpu)")
    parser.add_argument("--frames-only", action="store_true", help="verify conformed frames against the committed report instead of running ffmpeg")
    parser.add_argument("--report-dir", type=Path, default=None, help="write the report, sheets and audit CSV here (default: results/checks)")
    args = parser.parse_args()
    DEVICE, FRAMES_ONLY = args.device, args.frames_only
    if args.report_dir is not None:
        REPORT_DIR = args.report_dir
        SHEETS_DIR, AUDIT_CSV, HASH_FILE = REPORT_DIR / "mask_pipeline", REPORT_DIR / "mask_pipeline_audit.csv", REPORT_DIR / "mask_pipeline.sha256"
    determinism_settings, log = None, None
    if DEVICE.startswith("cuda"):
        from density.determinism import WarningLog, configure

        determinism_settings, log = configure(), WarningLog()
    if log is not None:
        with log.capture("mask check"):
            report = run_all(args.logic_only)
        from density.determinism import settings as current_settings

        report["determinism"] = {"settings": determinism_settings, "settings_at_end": current_settings(), "warnings": log.summary(), "deterministic_warnings": log.deterministic_warnings()}
    else:
        report = run_all(args.logic_only)
    report["device"] = DEVICE
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "mask_pipeline.json").write_text(json.dumps(report, indent=2, default=str))
    (REPORT_DIR / "mask_pipeline.md").write_text(markdown_report(report))
    if not args.logic_only:
        sheets = [sheet for s in report["clip_results"].values() for sheet in s["sheets"].values()]
        HASH_FILE.write_text("".join(f"{sheet['sha256']}  {Path(sheet['path']).relative_to(REPORT_DIR)}\n" for sheet in sheets))
    print(markdown_report(report))
    return 0 if report["all_judged_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
