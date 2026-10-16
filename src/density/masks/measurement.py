"""Measurement masks (protocol section 7), derived from the reference ROI only.

    inside   = dilate(ROI, 8 px)            primary masked LPIPS
    outside  = not dilate(ROI, 32 px)       outside-ROI guardrail
    V        = frames with a non-empty inside mask
    V_out    = frames with a non-empty outside mask

Every condition is scored with the same object; nothing here takes a condition's output.
"""

from dataclasses import dataclass

import numpy as np

from density.masks.geometry import dilate
from density.masks.registered import MEASURE_DILATION_PX, OUTSIDE_DILATION_PX


@dataclass(frozen=True)
class MeasurementMasks:
    inside: np.ndarray
    outside: np.ndarray
    inside_frames: tuple[int, ...]
    outside_frames: tuple[int, ...]


def measurement_masks(reference_roi: np.ndarray) -> MeasurementMasks:
    inside = dilate(reference_roi, MEASURE_DILATION_PX)
    outside = ~dilate(reference_roi, OUTSIDE_DILATION_PX)
    for array in (inside, outside):
        array.setflags(write=False)
    return MeasurementMasks(
        inside=inside,
        outside=outside,
        inside_frames=tuple(np.flatnonzero(inside.any(axis=(1, 2))).tolist()),
        outside_frames=tuple(np.flatnonzero(outside.any(axis=(1, 2))).tolist()),
    )
