"""Reference fidelity (protocol section 7): masked, outside-ROI and full-frame LPIPS, and the descriptive pixel checks.

Every score compares a condition's lossless uint8 frames with the same seed's C1 reference, under the reference's
measurement masks (density.masks.measurement). The definitions are registered in density.metrics.registered.

LPIPS v0.1 with AlexNet is built without lpips' own loading (which would fetch AlexNet from torchvision's servers and
load the linear layers with ``strict=False``, so a missing key would leave a layer random without an error): the trunk
comes from torchvision's AlexNet file and the linear layers from the lpips wheel's v0.1 file, both verified and read
with ``torch.load(weights_only=True)``, and every parameter must be loaded.
"""

import math

import numpy as np
import torch

from density.masks.measurement import MeasurementMasks, measurement_masks
from density.metrics import registered as R
from density.metrics.precision import scoring
from density.metrics.weights import verified_path

# AlexNet's features (torchvision indices) in lpips' five slices.
SLICES = {1: range(0, 2), 2: range(2, 5), 3: range(5, 8), 4: range(8, 10), 5: range(10, 12)}


def trunk_key(torchvision_key: str) -> str:
    """torchvision's ``features.<i>.<param>`` as lpips names it, ``net.slice<k>.<i>.<param>``."""
    _, index, param = torchvision_key.split(".")
    k = next(k for k, r in SLICES.items() if int(index) in r)
    return f"net.slice{k}.{index}.{param}"


def build_lpips(device="cpu", spatial: bool = True):
    """LPIPS v0.1 (AlexNet) in fp32 eval mode, from the verified weight files; every parameter is loaded."""
    import warnings

    import lpips

    with warnings.catch_warnings():  # lpips asks torchvision for an untrained AlexNet with its deprecated 'pretrained' argument
        warnings.filterwarnings("ignore", message=r"(The parameter 'pretrained'|Arguments other than a weight enum)", category=UserWarning)
        model = lpips.LPIPS(net=R.LPIPS_NET, version=R.LPIPS_VERSION, spatial=spatial, pnet_rand=True, pretrained=False, verbose=False)
    trunk = torch.load(verified_path("alexnet"), map_location="cpu", weights_only=True)
    linear = torch.load(verified_path("lpips_alex_v0.1"), map_location="cpu", weights_only=True)
    state = {trunk_key(k): v for k, v in trunk.items() if k.startswith("features.")}
    state.update(linear)
    parameters = {name for name, _ in model.named_parameters()}
    if set(state) != parameters:
        raise RuntimeError(f"LPIPS weights do not cover the model exactly: missing {sorted(parameters - set(state))}, extra {sorted(set(state) - parameters)}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Not in the files: the scaling layer's buffers, which are lpips' constants, and ``lins.<k>``, the ModuleList that
    # holds the same modules as ``lin<k>``.
    aliases = {f"lins.{k}.model.1.weight" for k in range(len(model.lins)) if model.lins[k] is getattr(model, f"lin{k}")}
    if unexpected or set(missing) != {"scaling_layer.shift", "scaling_layer.scale"} | aliases or len(aliases) != 5:
        raise RuntimeError(f"LPIPS state dict: missing {missing}, unexpected {unexpected}")
    return model.float().eval().to(device)


def to_tensor(frames: np.ndarray) -> torch.Tensor:
    """(T, H, W, 3) uint8 to (T, 3, H, W) float32 in [-1, 1], as lpips.im2tensor computes it (in float64, then cast)."""
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"scorers take (T, H, W, 3) uint8 frames, got {frames.shape} {frames.dtype}")
    return torch.from_numpy((frames.astype(np.float64) / 127.5 - 1.0).astype(np.float32)).permute(0, 3, 1, 2)


class LPIPS:
    """Per-frame spatial maps and scalars, in fixed batches on ``device``."""

    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.model = build_lpips(self.device, spatial=True)

    def _run(self, a: np.ndarray, b: np.ndarray, spatial: bool) -> np.ndarray:
        if a.shape != b.shape:
            raise ValueError(f"frame stacks differ in shape: {a.shape} and {b.shape}")
        ta, tb = to_tensor(a), to_tensor(b)
        out = []
        self.model.spatial = spatial
        try:
            with scoring(self.device):
                for i in range(0, len(ta), R.LPIPS_FRAMES_PER_BATCH):
                    d = self.model(ta[i:i + R.LPIPS_FRAMES_PER_BATCH].to(self.device), tb[i:i + R.LPIPS_FRAMES_PER_BATCH].to(self.device))
                    out.append(d[:, 0].float().cpu())
        finally:
            self.model.spatial = True
        result = torch.cat(out).numpy()
        if not np.isfinite(result).all():
            raise FloatingPointError("LPIPS produced a non-finite value")
        return result

    def maps(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """(T, H, W) float32 spatial LPIPS maps."""
        return self._run(a, b, spatial=True)

    def scalars(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """(T,) float32 scalar LPIPS per frame (spatial=False)."""
        return self._run(a, b, spatial=False).reshape(len(a))


def masked_mean(maps: np.ndarray, mask: np.ndarray, frames) -> float | None:
    """Equations 15 and 16: the map's mean inside ``mask`` on each listed frame, then the frames' mean (float64)."""
    if not len(frames):
        return None
    values = []
    for f in frames:
        m = mask[f]
        values.append(float(np.sum(maps[f].astype(np.float64)[m]) / np.count_nonzero(m)))
    return float(np.mean(values))


def crop_box(mask: np.ndarray, min_side: int = R.CROP_MIN_SIDE) -> tuple[int, int, int, int] | None:
    """(y0, y1, x0, x1): the mask's bounding box widened to at least ``min_side`` per side and kept inside the frame."""
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return None
    box = []
    for lo, hi, size in ((int(ys.min()), int(ys.max()) + 1, mask.shape[0]), (int(xs.min()), int(xs.max()) + 1, mask.shape[1])):
        need = max(0, min(min_side, size) - (hi - lo))
        lo, hi = lo - need // 2, hi + need - need // 2
        if lo < 0:
            lo, hi = 0, hi - lo
        if hi > size:
            lo, hi = max(0, lo - (hi - size)), size
        box += [lo, hi]
    return tuple(box)


def crop_lpips(model: LPIPS, a: np.ndarray, b: np.ndarray, mask: np.ndarray, frames) -> float | None:
    if not len(frames):
        return None
    values = []
    for f in frames:
        y0, y1, x0, x1 = crop_box(mask[f])
        values.append(float(model.scalars(a[f:f + 1, y0:y1, x0:x1], b[f:f + 1, y0:y1, x0:x1])[0]))
    return float(np.mean(values))


def masked_psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray, frames) -> float | None:
    if not len(frames):
        return None
    mse = []
    for f in frames:
        m = mask[f]
        diff = a[f].astype(np.float64)[m] - b[f].astype(np.float64)[m]  # (pixels, 3)
        mse.append(float(np.mean(diff**2)))
    value = float(np.mean(mse))
    return R.PSNR_CAP_DB if value == 0 else min(R.PSNR_CAP_DB, 10.0 * math.log10(255.0**2 / value))


def ssim_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(H, W) SSIM map of two (H, W, 3) uint8 images: each channel's map (scikit-image's Gaussian SSIM), then their mean."""
    from scipy.ndimage import gaussian_filter

    def blur(x):
        return gaussian_filter(x, sigma=R.SSIM_SIGMA, truncate=R.SSIM_TRUNCATE, mode="reflect")

    c1, c2 = (R.SSIM_K1 * R.SSIM_DATA_RANGE) ** 2, (R.SSIM_K2 * R.SSIM_DATA_RANGE) ** 2
    maps = []
    for c in range(3):
        x, y = a[..., c].astype(np.float64), b[..., c].astype(np.float64)
        ux, uy = blur(x), blur(y)
        vx, vy, vxy = blur(x * x) - ux * ux, blur(y * y) - uy * uy, blur(x * y) - ux * uy
        maps.append(((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux**2 + uy**2 + c1) * (vx + vy + c2)))
    return (maps[0] + maps[1] + maps[2]) / 3.0


SSIM_RADIUS = int(R.SSIM_TRUNCATE * R.SSIM_SIGMA + 0.5)  # scipy's kernel radius: 5, an 11 x 11 window


def masked_ssim(a: np.ndarray, b: np.ndarray, mask: np.ndarray, frames) -> float | None:
    """The SSIM map's mean inside the mask on each frame, then the frames' mean. Each frame's map is computed on the
    mask's bounding box plus the window radius, which gives the full frame's values at every mask pixel."""
    if not len(frames):
        return None
    values = []
    for f in frames:
        ys, xs = np.nonzero(mask[f])
        y0, y1 = max(0, ys.min() - SSIM_RADIUS), min(mask.shape[1], ys.max() + 1 + SSIM_RADIUS)
        x0, x1 = max(0, xs.min() - SSIM_RADIUS), min(mask.shape[2], xs.max() + 1 + SSIM_RADIUS)
        m = mask[f, y0:y1, x0:x1]
        values.append(float(np.mean(ssim_map(a[f, y0:y1, x0:x1], b[f, y0:y1, x0:x1])[m])))
    return float(np.mean(values))


def fidelity(model: LPIPS, condition: np.ndarray, reference: np.ndarray, measurement: MeasurementMasks, regions: dict[str, np.ndarray] | None = None) -> dict:
    """Every in-job score of one condition against its reference, with the frame counts behind each average.

    Scores whose frame set is empty are None (a control's masked scores); ``check_required`` judges them. ``regions``
    adds masked LPIPS inside other reference regions, each dilated by 8 px like the ROI (A1's hands and contact, and
    held object, section 3).
    """
    if condition.shape != reference.shape or condition.dtype != np.uint8 or reference.dtype != np.uint8:
        raise ValueError(f"condition {condition.shape} {condition.dtype} and reference {reference.shape} {reference.dtype} must be matching uint8 stacks")
    if measurement.inside.shape != condition.shape[:3]:
        raise ValueError(f"masks {measurement.inside.shape} do not match frames {condition.shape[:3]}")
    maps = model.maps(condition, reference)
    inside, outside = measurement.inside, measurement.outside
    v, v_out = measurement.inside_frames, measurement.outside_frames
    return {
        "masked_lpips": masked_mean(maps, inside, v),
        "outside_lpips": masked_mean(maps, outside, v_out),
        "full_frame_lpips": float(np.mean(model.scalars(condition, reference).astype(np.float64))),
        "masked_psnr_db": masked_psnr(condition, reference, inside, v),
        "masked_ssim": masked_ssim(condition, reference, inside, v),
        "crop_lpips": crop_lpips(model, condition, reference, inside, v),
        "frames": {"inside": len(v), "outside": len(v_out), "all": len(condition)},
        **({"region_masked_lpips": {name: masked_mean(maps, (m := measurement_masks(mask)).inside, m.inside_frames) for name, mask in regions.items()}} if regions else {}),
    }


class MissingScore(RuntimeError):
    """A required score is missing or not finite: a technical failure that blocks the decision until the job is rerun
    or the scoring code is repaired for every affected case (section 7)."""


def required(stratum: str) -> tuple[str, ...]:
    """Required scores for a condition of ``stratum``: masked LPIPS and the seven guardrails; controls, whose mask is
    empty by design, have no masked LPIPS."""
    from density.masks.roi import CONTROL

    return R.GUARDRAILS if stratum == CONTROL else R.REQUIRED


def check_required(scores: dict, names) -> None:
    bad = {n: scores.get(n) for n in names if not isinstance(scores.get(n), (int, float)) or not math.isfinite(scores[n])}
    if bad:
        raise MissingScore(f"required scores missing or not finite: {bad}")
