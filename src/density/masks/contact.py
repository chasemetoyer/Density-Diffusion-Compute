"""Contact intervals and the contact region (protocol section 4, equation 9).

Hands23 runs on keyframes 0, 8, ..., 80. An interval is kept only when
adjacent keyframes show ``object_contact`` between the same tracked hand and
the same tracked object. It runs from the first to the last keyframe of that
run: it is trimmed when contact stops and never extended before or after.
The paired hand and object masks must be usable on every frame inside it.

O and C exist only on validated frames, where C_f = dilate(H_f, 16) & dilate(O_f, 16).
"""

from dataclasses import dataclass, field

import numpy as np

from density.masks.geometry import dilate
from density.masks.registered import CONTACT_DILATION_PX, CONTACT_STATE, FRAMES, KEYFRAME_SPACING, MIN_USABLE_PIXELS


def keyframes(num_frames: int = FRAMES, spacing: int = KEYFRAME_SPACING) -> list[int]:
    return list(range(0, num_frames, spacing))


@dataclass(frozen=True)
class ContactObservation:
    """One Hands23 hand at one keyframe, with its tracked hand and object identities."""

    frame: int
    hand_track: int
    contact_state: str
    object_track: int | None  # the SAM object track its paired object box was matched to


@dataclass
class ContactInterval:
    hand_track: int
    object_track: int
    start: int  # first contact keyframe, inclusive
    end: int  # last contact keyframe, inclusive
    unusable_frames: list[int] = field(default_factory=list)

    @property
    def frames(self) -> range:
        return range(self.start, self.end + 1)

    @property
    def masks_usable(self) -> bool:
        return not self.unusable_frames


def contact_intervals(observations: list[ContactObservation], frames_with_keyframes: list[int]) -> list[ContactInterval]:
    """Maximal runs of at least two adjacent keyframes with contact between one hand track and one object track."""
    position = {frame: i for i, frame in enumerate(frames_with_keyframes)}
    contact = {}
    for obs in observations:
        if obs.frame not in position:
            raise ValueError(f"frame {obs.frame} is not a keyframe")
        if obs.contact_state == CONTACT_STATE and obs.object_track is not None:
            contact.setdefault((obs.hand_track, obs.object_track), set()).add(position[obs.frame])

    intervals = []
    for (hand, obj), indices in sorted(contact.items()):
        run = []
        for i in sorted(indices) + [None]:
            if run and (i is None or i != run[-1] + 1):
                if len(run) >= 2:
                    intervals.append(ContactInterval(hand, obj, frames_with_keyframes[run[0]], frames_with_keyframes[run[-1]]))
                run = []
            if i is not None:
                run.append(i)
    return intervals


def check_interval_masks(interval: ContactInterval, hand_masks: np.ndarray, object_masks: np.ndarray) -> ContactInterval:
    """Record frames inside the interval where the paired hand or object mask is not usable."""
    interval.unusable_frames = [
        f for f in interval.frames if hand_masks[f].sum() < MIN_USABLE_PIXELS or object_masks[f].sum() < MIN_USABLE_PIXELS
    ]
    return interval


def validated_frames(intervals: list[ContactInterval], num_frames: int = FRAMES) -> np.ndarray:
    """bool [frames]: inside an interval whose masks are usable throughout."""
    valid = np.zeros(num_frames, dtype=bool)
    for interval in intervals:
        if interval.masks_usable:
            valid[interval.start : interval.end + 1] = True
    return valid


def contact_region(hand_masks: np.ndarray, object_masks: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Equation 9 on validated frames; empty elsewhere."""
    region = np.zeros_like(hand_masks, dtype=bool)
    for f in np.flatnonzero(valid):
        region[f] = dilate(hand_masks[f], CONTACT_DILATION_PX) & dilate(object_masks[f], CONTACT_DILATION_PX)
    return region


def restrict_to(masks: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Keep masks only on validated frames (used for O)."""
    return masks & valid[:, None, None]
