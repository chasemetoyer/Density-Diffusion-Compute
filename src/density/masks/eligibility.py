"""Reference eligibility and ordered replacement (protocol section 4).

A primary reference (face or manipulation) is eligible when:
  * its primary ROI (F for faces, H for manipulation) is usable, meaning at
    least 256 mask pixels, in at least 41 of the 81 frames;
  * for manipulation, at least one contact interval spans two or more
    adjacent keyframes with the same tracked hand and object and has usable
    masks throughout;
  * at least one frame has pixels outside the ROI dilated by 32 px.

A prompt passes only when both seeds pass. A failed prompt is replaced by the
next item in its stratum's ordered reserve list; an exhausted list stops the
pipeline.

The audit (protocol section 4) rejects a reference only for mask invalidity: a
target that isn't visible counts as invalid. Each rejection names one of the
codes in AUDIT_CODES, all kinds of mask invalidity, and carries the reviewer's
written reason. Anatomy, image quality and appearance are never grounds.
"""

from dataclasses import dataclass, field

import numpy as np

from density.masks.contact import ContactInterval
from density.masks.measurement import MeasurementMasks
from density.masks.registered import FRAMES, MIN_USABLE_FRAMES, MIN_USABLE_PIXELS
from density.masks.roi import FACE, MANIPULATION

PRIMARY = (FACE, MANIPULATION)
AUDIT_CODES = {  # every audit rejection is a kind of mask invalidity
    "audit_roi_not_visible": "mask invalid: the target is not visible",
    "audit_mask_invalid": "mask invalid: off target, missing key parts, major spill, or not the same object across frames",
    "audit_contact_interrupted": "mask invalid: contact is not uninterrupted through the interval",
}
REASONS = {
    "roi_usable_frames": "primary ROI usable in fewer than 41 of 81 frames",
    "no_contact_interval": "no validated contact interval (two adjacent keyframes, same hand and object, usable masks)",
    "no_outside_frame": "no frame with pixels outside the ROI dilated by 32 px",
    **AUDIT_CODES,
}
REFERENCE_COST_RE = 1  # one dense 40-step reference per seed


def usable_frames(primary_roi: np.ndarray) -> int:
    return int((primary_roi.reshape(primary_roi.shape[0], -1).sum(axis=1) >= MIN_USABLE_PIXELS).sum())


@dataclass(frozen=True)
class AuditRejection:
    """A reviewer's rejection of one seed's reference: a mask-invalidity code and the reviewer's own written reason."""

    code: str
    written_reason: str

    def __post_init__(self):
        if self.code not in AUDIT_CODES:
            raise ValueError(f"{self.code!r} is not a mask-invalidity code; anatomy, image quality and appearance are never grounds")
        if not self.written_reason.strip():
            raise ValueError(f"a rejection for {self.code} needs a written reason")


@dataclass
class SeedResult:
    seed: int
    passed: bool
    reasons: list[str]
    usable_frames: int | None = None
    written_reasons: dict[str, str] = field(default_factory=dict)  # audit code -> the reviewer's written reason


def check_reference(
    stratum: str,
    seed: int,
    primary_roi: np.ndarray,
    measurement: MeasurementMasks,
    intervals: list[ContactInterval] | None = None,
    audit_rejections: list[AuditRejection] = (),
) -> SeedResult:
    """Eligibility of one seed's reference. Non-primary strata are not gated here."""
    for rejection in audit_rejections:
        if not isinstance(rejection, AuditRejection):
            raise TypeError(f"{rejection!r}: audit rejections are AuditRejection(code, written_reason)")
    if stratum not in PRIMARY:
        return SeedResult(seed, True, [], None)
    if primary_roi.shape[0] != FRAMES:
        raise ValueError(f"expected {FRAMES} frames, got {primary_roi.shape[0]}")

    reasons = []
    count = usable_frames(primary_roi)
    if count < MIN_USABLE_FRAMES:
        reasons.append("roi_usable_frames")
    if stratum == MANIPULATION and not any(i.masks_usable for i in intervals or []):
        reasons.append("no_contact_interval")
    if not measurement.outside_frames:
        reasons.append("no_outside_frame")
    reasons += [r.code for r in audit_rejections]
    return SeedResult(seed, not reasons, reasons, count, {r.code: r.written_reason for r in audit_rejections})


def prompt_passes(seed_results: list[SeedResult]) -> bool:
    """Both planned seeds must pass."""
    if len(seed_results) != 2:
        raise ValueError(f"expected results for both seeds, got {len(seed_results)}")
    return all(r.passed for r in seed_results)


class ReserveExhausted(RuntimeError):
    """A stratum ran out of reserves; stop and amend the registration."""


@dataclass
class ReplacementLog:
    entries: list[dict] = field(default_factory=list)

    @property
    def extra_cost_re(self) -> int:
        return sum(entry["extra_cost_re"] for entry in self.entries)


class ReserveList:
    """One stratum's ordered reserve list. Replacement order is fixed; nothing is skipped or reordered."""

    def __init__(self, stratum: str, reserves: list[int], log: ReplacementLog):
        self.stratum = stratum
        self._reserves = list(reserves)
        self.log = log

    def replace(self, failed_prompt: int, seed_results: list[SeedResult]) -> int:
        if not self._reserves:
            raise ReserveExhausted(f"{self.stratum}: no reserves left to replace prompt {failed_prompt}")
        replacement = self._reserves.pop(0)
        self.log.entries.append(
            {
                "stratum": self.stratum,
                "failed_prompt": failed_prompt,
                "replacement": replacement,
                "reasons": {r.seed: r.reasons for r in seed_results},
                "written_reasons": {r.seed: r.written_reasons for r in seed_results if r.written_reasons},
                "extra_cost_re": 2 * REFERENCE_COST_RE,  # both seeds of the replacement are generated and audited
            }
        )
        return replacement

    @property
    def remaining(self) -> list[int]:
        return list(self._reserves)
