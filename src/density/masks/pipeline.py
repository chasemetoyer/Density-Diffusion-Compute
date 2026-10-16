"""One reference clip through the mask pipeline (protocol sections 4 and 7).

Regions for the clip's stratum, the stratum ROI and A1 mask, their token
coverage, the measurement masks, and eligibility. Storage and the audit sheet
are written by ``save`` and ``audit.contact_sheet``.

    face          F                    primary ROI F
    manipulation  H, O, C              primary ROI H; needs a validated contact interval
    body          B                    not gated
    control       nothing              an explicit empty mask

Each reference computes only the regions its stratum uses (registered as
``STRATUM_REGIONS``). For the stock-clip check only, ``contact_test`` also runs the
manipulation path (H, Hands23, O, C) on a clip of another stratum, for
information; its regions never enter that clip's ROI.
"""

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from density.masks import roi as roi_module
from density.masks.contact import keyframes, validated_frames
from density.masks.eligibility import SeedResult, check_reference, usable_frames
from density.masks.measurement import MeasurementMasks, measurement_masks
from density.masks.regions import Region, manipulation_regions, tracked_region
from density.masks.registered import FRAMES
from density.masks.storage import save_reference
from density.token_map import coverage_grid

PRIMARY_REGION = {roi_module.FACE: "F", roi_module.MANIPULATION: "H"}


class Models:
    """Detectors built once per session on ``device``; SCRFD always runs on the onnxruntime CPU provider.

    ``cpu_threads`` sets SCRFD's onnxruntime thread count. The default, "quota", is the pod's cgroup CPU quota rounded
    down (density/cpu_threads.py), and onnxruntime's own default where there is no quota, as on the Mac. None keeps
    onnxruntime's default everywhere.
    """

    def __init__(self, device: str = "cpu", cpu_threads: int | str | None = "quota"):
        from density.cpu_threads import quota_threads

        self.device = device
        self.cpu_threads = quota_threads() if cpu_threads == "quota" else cpu_threads
        self.build_seconds: dict[str, float] = {}
        self._models: dict = {}

    def get(self, name: str):
        if name not in self._models:
            start = time.perf_counter()
            if name == "scrfd":
                from density.masks.detectors import SCRFDFaceDetector

                self._models[name] = SCRFDFaceDetector(threads=self.cpu_threads)
            elif name == "wilor":
                from density.masks.wilor import WiLoRHandDetector

                self._models[name] = WiLoRHandDetector(self.device)
            elif name == "mask_rcnn":
                from density.masks.detectors import MaskRCNNPersonDetector

                self._models[name] = MaskRCNNPersonDetector(self.device)
            elif name == "hands23":
                from density.masks.hands23 import Hands23Predictor

                self._models[name] = Hands23Predictor(device=self.device)
            else:
                raise KeyError(name)
            self.build_seconds[name] = time.perf_counter() - start
        return self._models[name]


@dataclass
class ReferenceResult:
    stratum: str
    regions: dict[str, Region]  # every region computed, including contact-test ones
    roi: np.ndarray
    coverage: dict[str, np.ndarray]  # "ROI" (and "A1" for manipulation): m with shape (latent frames, rows, cols)
    measurement: MeasurementMasks
    eligibility: SeedResult
    manipulation: dict | None  # observations, intervals and Hands23 labels, when the manipulation path ran
    seconds: dict = field(default_factory=dict)
    sam: dict = field(default_factory=dict)  # SAM calls and seconds

    @property
    def roi_regions(self) -> tuple[str, ...]:
        return roi_module.REGIONS.get(self.stratum, ())


def _sam(frames: np.ndarray, device: str):
    from density.masks.sam import SAM2Propagator

    return SAM2Propagator(frames, device)


def process_reference(frames: np.ndarray, stratum: str, models: Models, seed: int | None = None, contact_test: bool = False, sam_device: str | None = None) -> ReferenceResult:
    if frames.shape[0] != FRAMES or frames.dtype != np.uint8:
        raise ValueError(f"expected {FRAMES} uint8 frames, got {frames.shape} {frames.dtype}")
    start = time.perf_counter()
    regions: dict[str, Region] = {}
    manipulation = None
    propagator = None
    if stratum != roi_module.CONTROL:
        propagator = _sam(frames, sam_device or models.device)
    try:
        if stratum == roi_module.FACE:
            regions["F"] = tracked_region(models.get("scrfd").detect, frames, propagator)
        elif stratum == roi_module.BODY:
            regions["B"] = tracked_region(models.get("mask_rcnn").detect, frames, propagator)
        if stratum == roi_module.MANIPULATION or (contact_test and propagator is not None):
            manipulation = manipulation_regions(frames, propagator, models.get("wilor"), models.get("hands23"))
            regions.update({k: manipulation[k] for k in ("H", "O", "C")})
    finally:
        sam = {"calls": propagator.calls, "seconds": propagator.seconds, "device": propagator.device} if propagator else {}
        if propagator is not None:
            propagator.close()
    detect_seconds = time.perf_counter() - start

    begin = time.perf_counter()
    if stratum == roi_module.CONTROL:
        roi = roi_module.assemble_roi(stratum, None)
    else:
        roi = roi_module.assemble_roi(stratum, {k: regions[k].mask for k in roi_module.REGIONS[stratum]})
    coverage = {"ROI": coverage_grid(torch.from_numpy(roi)).numpy()}
    if stratum == roi_module.MANIPULATION:
        coverage["A1"] = coverage_grid(torch.from_numpy(roi_module.a1_mask({k: r.mask for k, r in regions.items()}))).numpy()
    measurement = measurement_masks(roi)
    primary = regions[PRIMARY_REGION[stratum]].mask if stratum in PRIMARY_REGION else roi
    intervals = manipulation["intervals"] if stratum == roi_module.MANIPULATION else None
    eligibility = check_reference(stratum, seed, primary, measurement, intervals)
    post_seconds = time.perf_counter() - begin

    seconds = {"total": time.perf_counter() - start, "detect_track": detect_seconds, "roi_coverage_measurement_eligibility": post_seconds}
    for name, region in regions.items():
        for component, value in region.seconds.items():
            seconds[f"{name}.{component}"] = value
    return ReferenceResult(stratum, regions, roi, coverage, measurement, eligibility, manipulation, seconds, sam)


def metadata(result: ReferenceResult) -> dict:
    """Everything the metadata JSON records about one reference, apart from versions and hashes."""
    m = result.manipulation
    return {
        "stratum": result.stratum,
        "roi_regions": list(result.roi_regions),
        "keyframes": keyframes(),
        "regions": {name: region.summary() for name, region in result.regions.items()},
        "roi_usable_frames": usable_frames(result.roi),
        "primary_usable_frames": result.eligibility.usable_frames,
        "eligibility": asdict(result.eligibility),
        "measurement": {"inside_frames": list(result.measurement.inside_frames), "outside_frames": list(result.measurement.outside_frames)},
        "contact": None
        if m is None
        else {
            "observations": m["observations"],
            "intervals": [asdict(i) for i in m["intervals"]],
            "validated_frames": np.flatnonzero(validated_frames(m["intervals"])).tolist(),
            "hands23_labels": {int(f): v for f, v in m["hands23_labels"].items()},
            "hands23_side_disagreements": m["hands23_side_disagreements"],
            "hand_tracks_dropped": list(result.regions["H"].dropped),
        },
        "seconds": result.seconds,
        "sam": result.sam,
    }


def save(result: ReferenceResult, folder: Path, extra_metadata: dict) -> dict:
    """RLE masks (regions, the ROI, per-track masks, backfill), token coverage and metadata; returns file hashes."""
    regions = {name: region.mask for name, region in result.regions.items()}
    regions["ROI"] = result.roi
    tracks = {name: region.tracks.masks for name, region in result.regions.items() if region.tracks is not None}
    backfill = {name: region.backfill() for name, region in result.regions.items() if region.backfill() is not None}
    return save_reference(folder, regions, result.coverage, {**extra_metadata, **metadata(result)}, tracks=tracks, backfill=backfill)
