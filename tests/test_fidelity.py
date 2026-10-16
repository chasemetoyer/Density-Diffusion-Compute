"""Reference fidelity: data-only LPIPS v0.1, equations 15-16, full-frame scalar LPIPS, crop LPIPS, masked PSNR and SSIM."""

import hashlib
import math
import os

import numpy as np
import pytest
import torch

from density.masks.measurement import measurement_masks
from density.metrics import fidelity, weights
from density.metrics import registered as R

HAVE_WEIGHTS = all((weights.ROOT / weights.WEIGHTS[n].path).exists() for n in ("alexnet", "lpips_alex_v0.1"))
needs_weights = pytest.mark.skipif(not HAVE_WEIGHTS, reason="the LPIPS weight files are not downloaded")


def frames(t=5, h=96, w=128, seed=0, noise=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack([(xx * 2 + 7 * i) % 256 for i in range(t)])[..., None].repeat(3, -1)
    base[..., 1] = (yy * 2)[None]
    out = base.astype(int) + (rng.integers(-noise, noise + 1, base.shape) if noise else 0)
    return np.clip(out, 0, 255).astype(np.uint8)


def roi(t=5, h=96, w=128):
    m = np.zeros((t, h, w), bool)
    m[1:4, 30:50, 40:70] = True  # frames 0 and 4 have no ROI
    return m


@pytest.fixture(scope="module")
def model():
    return fidelity.LPIPS("cpu")


@needs_weights
def test_data_only_build_equals_lpips_own_loading():
    import lpips
    import torchvision

    wheel = os.path.join(os.path.dirname(lpips.__file__), "weights", "v0.1", "alex.pth")
    assert hashlib.sha256(open(wheel, "rb").read()).hexdigest() == weights.WEIGHTS["lpips_alex_v0.1"].sha256
    reference = lpips.LPIPS(net="alex", version="0.1", pretrained=True, pnet_rand=True, verbose=False).eval()
    alexnet = torchvision.models.alexnet(weights=None)
    alexnet.load_state_dict(torch.load(weights.verified_path("alexnet"), weights_only=True))
    for k, indices in fidelity.SLICES.items():
        for i in indices:
            getattr(reference.net, f"slice{k}")._modules[str(i)].load_state_dict(alexnet.features[i].state_dict())
    ours = fidelity.build_lpips("cpu", spatial=False)
    a, b = fidelity.to_tensor(frames(2)), fidelity.to_tensor(frames(2, noise=20))
    with torch.no_grad():
        assert torch.equal(ours(a, b), reference(a, b))
    assert torch.equal(fidelity.to_tensor(frames(1)), lpips.im2tensor(frames(1)[0]))


@needs_weights
def test_identical_frames_give_zero_and_perfect_scores(model):
    a = frames()
    scores = fidelity.fidelity(model, a, a, measurement_masks(roi()))
    assert scores["masked_lpips"] == scores["outside_lpips"] == scores["full_frame_lpips"] == scores["crop_lpips"] == 0.0
    assert scores["masked_psnr_db"] == 100.0 and scores["masked_ssim"] == 1.0
    assert scores["frames"] == {"inside": 3, "outside": 5, "all": 5}


@needs_weights
def test_masked_scores_average_the_full_frame_map_with_equal_frame_weight(model):
    a, b = frames(), frames(noise=25)
    measurement = measurement_masks(roi())
    scores = fidelity.fidelity(model, a, b, measurement)
    maps = model.maps(a, b)
    expected = np.mean([maps[f][measurement.inside[f]].astype(np.float64).mean() for f in measurement.inside_frames])
    assert scores["masked_lpips"] == pytest.approx(expected, rel=1e-12)
    expected_out = np.mean([maps[f][measurement.outside[f]].astype(np.float64).mean() for f in measurement.outside_frames])
    assert scores["outside_lpips"] == pytest.approx(expected_out, rel=1e-12)
    # full-frame: the scalar per frame (registered), which is not the spatial map's mean
    scalars = model.scalars(a, b)
    assert scores["full_frame_lpips"] == pytest.approx(float(np.mean(scalars.astype(np.float64))), rel=1e-12)
    assert not np.allclose(scalars, maps.mean(axis=(1, 2)), rtol=0, atol=0)
    # complete frames: changing pixels well outside the mask changes the masked score only through LPIPS's context
    c = b.copy()
    c[:, :, :10] = 0
    assert fidelity.fidelity(model, a, c, measurement)["masked_lpips"] != scores["masked_lpips"]


def test_equations_15_and_16_on_a_known_map():
    maps = np.zeros((3, 4, 4), np.float32)
    maps[0, :2] = 1.0  # frame 0: mask covers 4 pixels with value 1 and 4 with 0 -> 0.5
    maps[2] = 3.0
    mask = np.zeros((3, 4, 4), bool)
    mask[0, 1:3, :2] = True
    mask[2, 0, 0] = True
    assert fidelity.masked_mean(maps, mask, (0, 2)) == pytest.approx((0.5 + 3.0) / 2)  # equal frame weight, not pixel weight
    assert fidelity.masked_mean(maps, mask, ()) is None


@pytest.mark.parametrize("box, shape, expected", [
    ((10, 12, 20, 30), (96, 128), (0, 64, 0, 64)),  # widened to 64 about its centre, then moved inside the frame
    ((40, 50, 10, 110), (96, 128), (13, 77, 10, 110)),  # the wide side is kept
    ((90, 96, 120, 128), (96, 128), (32, 96, 64, 128)),  # at the far corner
    ((0, 40, 0, 40), (40, 50), (0, 40, 0, 50)),  # a frame side shorter than 64 is taken whole
])
def test_crop_box(box, shape, expected):
    y0, y1, x0, x1 = box
    mask = np.zeros(shape, bool)
    mask[y0:y1, x0:x1] = True
    assert fidelity.crop_box(mask) == expected
    assert fidelity.crop_box(np.zeros(shape, bool)) is None


@needs_weights
def test_crop_lpips_works_on_a_tiny_mask(model):
    a, b = frames(), frames(noise=25)
    tiny = np.zeros((5, 96, 128), bool)
    tiny[2, 50, 60] = True
    value = fidelity.crop_lpips(model, a, b, tiny, (2,))
    y0, y1, x0, x1 = fidelity.crop_box(tiny[2])
    assert (y1 - y0, x1 - x0) == (64, 64) and value == float(model.scalars(a[2:3, y0:y1, x0:x1], b[2:3, y0:y1, x0:x1])[0]) > 0


def test_masked_psnr_known_value():
    a = np.zeros((2, 8, 8, 3), np.uint8)
    b = a.copy()
    b[0] = 2  # frame 0: MSE 4 inside the mask
    b[1, :4] = 255  # frame 1: differences only outside the mask
    mask = np.zeros((2, 8, 8), bool)
    mask[:, 4:, :] = True
    assert fidelity.masked_psnr(a, b, mask, (0, 1)) == pytest.approx(10 * math.log10(255.0**2 / 2.0))  # mean of 4 and 0
    assert fidelity.masked_psnr(a, a, mask, (0, 1)) == 100.0 and fidelity.masked_psnr(a, b, mask, ()) is None


def test_ssim_matches_scikit_image_and_its_crop_is_exact():
    from skimage.metrics import structural_similarity

    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, (40, 56, 3), dtype=np.uint8)
    b = np.clip(a.astype(int) + rng.integers(-30, 31, a.shape), 0, 255).astype(np.uint8)
    _, full = structural_similarity(a, b, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=255, channel_axis=-1, full=True)
    ours = fidelity.ssim_map(a, b)
    assert np.abs(ours - full.mean(axis=-1)).max() == 0.0
    mask = np.zeros((1, 40, 56), bool)
    mask[0, 2:9, 30:52] = True  # touches nothing; its bounding box plus 5 px stays inside
    mask[0, 35:40, 0:3] = True  # reaches the frame's corner
    assert fidelity.masked_ssim(a[None], b[None], mask, (0,)) == float(ours[mask[0]].mean())


def test_required_scores_and_missing_ones():
    assert fidelity.required("face") == R.REQUIRED and "masked_lpips" not in fidelity.required("control")
    assert len(R.GUARDRAILS) == 7
    good = {name: 0.1 for name in R.REQUIRED}
    fidelity.check_required(good, R.REQUIRED)
    for bad in (None, float("nan"), float("inf"), "0.1"):
        with pytest.raises(fidelity.MissingScore):
            fidelity.check_required({**good, "imaging_quality": bad}, R.REQUIRED)
    with pytest.raises(fidelity.MissingScore):
        fidelity.check_required({k: v for k, v in good.items() if k != "outside_lpips"}, R.REQUIRED)


@needs_weights
def test_scoring_is_deterministic_and_restores_precision(model):
    before = (torch.backends.cudnn.conv.fp32_precision, torch.backends.cuda.matmul.fp32_precision)
    a, b = frames(), frames(noise=25)
    measurement = measurement_masks(roi())
    first = fidelity.fidelity(model, a, b, measurement)
    assert fidelity.fidelity(fidelity.LPIPS("cpu"), a, b, measurement) == first  # a fresh model, the same numbers
    assert (torch.backends.cudnn.conv.fp32_precision, torch.backends.cuda.matmul.fp32_precision) == before
    with pytest.raises(ValueError):
        fidelity.fidelity(model, a.astype(np.float32), b, measurement)
