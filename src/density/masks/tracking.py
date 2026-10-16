"""Box-prompted video propagation with the registered re-prompt and association rule.

A propagator (SAM 2.1 in production, a mock in tests) turns box prompts into
per-frame masks for each object track. After each propagation, frames with
detections are checked in order, and each detection is associated with the
tracks by the registered rule (``Association``). At the first frame where a
detection re-prompts a track or starts a new one, the video is propagated
again with the new prompts before later frames are checked.

Association at one frame. IoU is between a track's mask box and the
detection box. A side is a hand's "left" or "right" (H only); F, O and B
tracks and detections have no side, and then every side check passes.

  a. Bound: the detection that prompted a track at this frame belongs to it.
  b. Identity: a detection that overlaps a track at IoU >= 0.5 belongs to that
     track whatever its side label: the same-side track when several qualify,
     else the highest IoU, then the lowest track ID. It never re-prompts or
     starts a track. Different labels are logged as a side disagreement. A
     track is confirmed when a same-side detection belongs to it.
  c. Re-prompt: an unconfirmed track takes the free same-side detection that
     overlaps it most, at 0 < IoU < 0.5 (greedy by IoU), and is re-prompted
     with it. A track is re-prompted at most once per frame.
  d. Lost track: an unconfirmed track whose mask has 0 pixels at this frame
     takes a remaining free same-side detection (lowest track ID first).
  e. New track: any other detection starts a new track, which keeps that
     detection's side for good, unless it overlaps a track started earlier
     at this frame at IoU >= 0.5, in which case it belongs to that track.

So a detection with zero overlap never moves an existing track (unless the
track is lost), a hand track is never re-prompted by the other hand's
detection, and no track starts on top of another. Decisions are never undone:
a track started at one frame stays even if a later re-prompt moves another
track onto the same object, and tracks are never merged or deleted.
"""

from dataclasses import asdict, dataclass, field
from typing import Protocol

import numpy as np

from density.masks.geometry import box_iou, mask_box
from density.masks.registered import REPROMPT_GATE_IOU, REPROMPT_IOU


class Propagator(Protocol):
    def propagate(self, prompts: dict[int, dict[int, np.ndarray]], num_frames: int) -> dict[int, np.ndarray]:
        """prompts[track][frame] = box -> masks[track] with shape (frames, H, W)."""


def needs_reprompt(mask: np.ndarray, detection_box) -> bool:
    """The registered rule: re-prompt when IoU(box of SAM's mask, detection box) < 0.5."""
    return box_iou(mask_box(mask), detection_box) < REPROMPT_IOU


def same_side(a: str | None, b: str | None) -> bool:
    return a is None or b is None or a == b


@dataclass(frozen=True)
class Decision:
    """What one detection at one frame did."""

    frame: int
    detection: int  # index into that frame's detections
    action: str  # bound, match, absorbed, reprompt, reprompt_lost, new_track
    track: int | None  # the track it belongs to (a new track's ID once started)
    iou: float  # IoU between the track's mask box and the detection box
    side: str | None = None  # the detection's side label
    track_side: str | None = None

    @property
    def side_disagreement(self) -> bool:
        return not same_side(self.side, self.track_side)


@dataclass
class TrackState:
    """What association can see besides the masks: prompts, track sides, and which detection made each prompt."""

    prompts: dict[int, dict[int, np.ndarray]]
    sides: dict[int, str | None]
    sources: dict[tuple[int, int], int]  # (track, frame) -> index of the detection that prompted it


class FrameContext:
    """One frame's tracks, detections, IoUs and the decisions made so far."""

    def __init__(self, masks, state: TrackState, frame: int, boxes: list[np.ndarray], sides: list[str | None]):
        self.masks, self.state, self.frame, self.boxes, self.sides = masks, state, frame, boxes, sides
        self.tracks = sorted(masks)
        mask_boxes = {t: mask_box(masks[t][frame]) for t in self.tracks}
        self.iou = {(t, j): box_iou(mask_boxes[t], boxes[j]) for t in self.tracks for j in range(len(boxes))}
        self.decisions: dict[int, Decision] = {}
        self.reprompted: set[int] = set()

    def same(self, t: int, j: int) -> bool:
        return same_side(self.state.sides.get(t), self.sides[j])

    def free(self) -> list[int]:
        return [j for j in range(len(self.boxes)) if j not in self.decisions]

    def decide(self, j: int, action: str, t: int | None, iou: float, track_side: str | None = None) -> None:
        side = self.state.sides.get(t) if t is not None else track_side
        self.decisions[j] = Decision(self.frame, j, action, t, float(iou), self.sides[j], side)

    def confirmed(self) -> set[int]:
        return {d.track for d in self.decisions.values() if d.track is not None and not d.side_disagreement}

    def prompted_here(self, t: int) -> bool:
        return self.frame in self.state.prompts[t]


class Association:
    """The registered association rule at one frame. Negative controls override one step each."""

    def __call__(self, masks, state: TrackState, frame: int, boxes: list[np.ndarray], sides: list[str | None]) -> list[Decision]:
        ctx = FrameContext(masks, state, frame, boxes, sides)
        for step in (self.bound, self.identity, self.reprompt, self.lost, self.new_tracks):
            step(ctx)
        return [ctx.decisions[j] for j in sorted(ctx.decisions)]

    def bound(self, ctx: FrameContext) -> None:
        for (t, f), j in ctx.state.sources.items():
            if f == ctx.frame and t in ctx.masks:
                ctx.decide(j, "bound", t, ctx.iou[t, j])

    def qualifies(self, ctx: FrameContext, t: int, j: int) -> bool:
        return ctx.iou[t, j] >= REPROMPT_IOU

    def identity(self, ctx: FrameContext) -> None:
        for j in ctx.free():
            qualifying = [t for t in ctx.tracks if self.qualifies(ctx, t, j)]
            if qualifying:
                t = min(qualifying, key=lambda t: (not ctx.same(t, j), -ctx.iou[t, j], t))
                ctx.decide(j, "match", t, ctx.iou[t, j])

    def reprompt(self, ctx: FrameContext) -> None:
        confirmed = ctx.confirmed()
        pairs = sorted(
            (
                (t, j)
                for t in ctx.tracks
                if t not in confirmed and not ctx.prompted_here(t)
                for j in ctx.free()
                if ctx.same(t, j) and REPROMPT_GATE_IOU < ctx.iou[t, j] < REPROMPT_IOU
            ),
            key=lambda p: (-ctx.iou[p], p),
        )
        for t, j in pairs:
            if t not in ctx.reprompted and j not in ctx.decisions:
                ctx.decide(j, "reprompt", t, ctx.iou[t, j])
                ctx.reprompted.add(t)

    def lost(self, ctx: FrameContext) -> None:
        confirmed = ctx.confirmed()
        for t in ctx.tracks:
            if t in confirmed or t in ctx.reprompted or ctx.prompted_here(t) or ctx.masks[t][ctx.frame].any():
                continue
            j = next((j for j in ctx.free() if ctx.same(t, j)), None)
            if j is not None:
                ctx.decide(j, "reprompt_lost", t, 0.0)
                ctx.reprompted.add(t)

    def new_tracks(self, ctx: FrameContext) -> None:
        started = []
        for j in ctx.free():
            on_top = [k for k in started if box_iou(ctx.boxes[k], ctx.boxes[j]) >= REPROMPT_IOU]
            if on_top:
                ctx.decide(j, "absorbed", None, box_iou(ctx.boxes[on_top[0]], ctx.boxes[j]), track_side=ctx.sides[on_top[0]])
            else:
                ctx.decide(j, "new_track", None, max((ctx.iou[t, j] for t in ctx.tracks), default=0.0), track_side=ctx.sides[j])
                started.append(j)


@dataclass
class TrackResult:
    masks: dict[int, np.ndarray]
    prompts: dict[int, dict[int, np.ndarray]]
    events: list[dict] = field(default_factory=list)  # every new track and re-prompt, in order, with its reason
    sides: dict[int, str | None] = field(default_factory=dict)
    associations: list[Decision] = field(default_factory=list)  # every detection's decision on the settled masks
    rounds: int = 0

    @property
    def reprompts(self) -> list[dict]:
        return [e for e in self.events if e["action"] in ("reprompt", "reprompt_lost")]

    @property
    def side_disagreements(self) -> list[dict]:
        return [asdict(d) for d in self.associations if d.side_disagreement]

    def first_prompt(self, track: int) -> int:
        return min(self.prompts[track])

    def backfill_frames(self, track: int) -> dict[int, int]:
        """Frames before the track's first detection where SAM still gave it pixels: frame -> pixel count."""
        m = self.masks[track]
        return {f: int(m[f].sum()) for f in range(self.first_prompt(track)) if m[f].any()}

    def backfill_mask(self, tracks=None) -> np.ndarray | None:
        """Pixels that only tracks before their first detection cover (drawn distinctly for the audit).

        ``tracks`` limits this to some of the tracks (the kept hand tracks); by default every track counts.
        """
        tracks = sorted(self.masks) if tracks is None else list(tracks)
        if not tracks:
            return None
        frames = np.arange(self.masks[tracks[0]].shape[0])[:, None, None]
        before = np.zeros_like(self.masks[tracks[0]])
        after = np.zeros_like(before)
        for t in tracks:
            before |= self.masks[t] & (frames < self.first_prompt(t))
            after |= self.masks[t] & (frames >= self.first_prompt(t))
        return before & ~after


def track(
    propagator: Propagator,
    detections: dict[int, list[np.ndarray]],
    num_frames: int,
    sides: dict[int, list[str | None]] | None = None,
    association: Association | None = None,
    max_rounds: int = 100,
) -> TrackResult:
    """Propagate box prompts, associating detections by ``association`` until no frame changes anything."""
    association = association or Association()
    frames = sorted(f for f, boxes in detections.items() if len(boxes))
    boxes = {f: [np.asarray(b, dtype=np.float64) for b in detections[f]] for f in frames}
    sides = {f: list(sides[f]) if sides else [None] * len(boxes[f]) for f in frames}
    state = TrackState({}, {}, {})
    events = []
    masks: dict[int, np.ndarray] = {}
    for rounds in range(1, max_rounds + 1):
        if state.prompts:
            masks = propagator.propagate(state.prompts, num_frames)
        changed = False
        settled = []
        for frame in frames:
            for d in association(masks, state, frame, boxes[frame], sides[frame]):
                if d.action in ("reprompt", "reprompt_lost"):
                    state.prompts[d.track][frame] = boxes[frame][d.detection]
                    state.sources[(d.track, frame)] = d.detection
                elif d.action == "new_track":
                    new_id = max(state.prompts, default=-1) + 1
                    state.prompts[new_id] = {frame: boxes[frame][d.detection]}
                    state.sides[new_id] = d.side
                    state.sources[(new_id, frame)] = d.detection
                    d = Decision(d.frame, d.detection, d.action, new_id, d.iou, d.side, d.side)
                settled.append(d)
                if d.action in ("reprompt", "reprompt_lost", "new_track"):
                    events.append({**asdict(d), "round": rounds})
                    changed = True
            if changed:
                break  # propagate again with the new prompts before checking later frames
        if not changed:
            return TrackResult(masks, state.prompts, events, dict(state.sides), settled, rounds)
    raise RuntimeError(f"association did not settle within {max_rounds} rounds")
