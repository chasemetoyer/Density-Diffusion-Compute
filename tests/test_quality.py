"""The five VBench quality dimensions: the 0-100 conversion, the NaN guard, and known answers on synthetic videos."""

import numpy as np
import pytest
import torch

from density import video
from density.metrics import quality, weights
from density.metrics.registered import QUALITY_DIMENSIONS, QUALITY_SCALE

QUALITY_WEIGHTS = ("dino_vitb16", "clip_vit_b32", "clip_vit_l14", "laion_aesthetic_head", "musiq_spaq", "amt_s")
HAVE_WEIGHTS = all((weights.ROOT / weights.WEIGHTS[n].path).exists() for n in QUALITY_WEIGHTS)
needs_weights = pytest.mark.skipif(not HAVE_WEIGHTS, reason="the quality-model weight files are not downloaded")

quality.import_paths()


class Constant(torch.nn.Module):
    """A stand-in model: the same output for every input."""

    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x, *args, **kwargs):
        return torch.full((x.shape[0], 4), self.value) if isinstance(self.value, float) else self.value.expand(x.shape[0], -1)


def test_the_scale_is_x100_for_four_dimensions_and_raw_musiq_for_imaging():
    assert QUALITY_SCALE == {"subject_consistency": 100.0, "background_consistency": 100.0, "motion_smoothness": 100.0, "aesthetic_quality": 100.0, "imaging_quality": 1.0}
    raw = {"subject_consistency": 0.95, "background_consistency": 0.9, "motion_smoothness": 0.99, "aesthetic_quality": 0.5, "imaging_quality": 64.0}
    assert quality.to_scale(raw) == pytest.approx({"subject_consistency": 95.0, "background_consistency": 90.0, "motion_smoothness": 99.0, "aesthetic_quality": 50.0, "imaging_quality": 64.0})
    with pytest.raises(ValueError):
        quality.to_scale({k: v for k, v in raw.items() if k != "imaging_quality"})


def test_vbench_divides_only_the_imaging_aggregate_by_100(tmp_path):
    """VBench's technical_quality: per-video values are raw MUSIQ (0-100) and only the aggregate is divided by 100."""
    from vbench.imaging_quality import technical_quality

    path = tmp_path / "v.mp4"
    frames = np.full((3, 32, 48, 3), 100, np.uint8)
    with quality.frames_for(path, frames):
        aggregate, per_video = technical_quality(lambda x: torch.tensor(42.0), [str(path)], "cpu")
    assert per_video[0]["video_results"] == 42.0 and aggregate == pytest.approx(0.42)
    assert quality.to_scale({**dict.fromkeys(QUALITY_DIMENSIONS, 0.0), "imaging_quality": per_video[0]["video_results"]})["imaging_quality"] == 42.0


def test_per_video_consistency_is_in_unit_range_before_conversion(tmp_path):
    from vbench.subject_consistency import subject_consistency

    path = tmp_path / "v.mp4"
    with quality.frames_for(path, np.full((4, 32, 48, 3), 50, np.uint8)):
        _, per_video = subject_consistency(Constant(1.0), [str(path)], "cpu", False)
    assert per_video[0]["video_results"] == pytest.approx(1.0)


def test_vbench_turns_nan_into_zero_and_the_guard_stops_it(tmp_path):
    from vbench.subject_consistency import subject_consistency

    path = tmp_path / "v.mp4"
    frames = np.full((3, 32, 48, 3), 50, np.uint8)
    model = Constant(float("nan"))
    with quality.frames_for(path, frames):
        _, per_video = subject_consistency(model, [str(path)], "cpu", False)
    assert per_video[0]["video_results"] == 0.0  # what VBench reports for a NaN
    model.register_forward_hook(quality._check_finite("stand-in"))
    with quality.frames_for(path, frames), pytest.raises(quality.NonFinite):
        subject_consistency(model, [str(path)], "cpu", False)


def test_frames_for_refuses_other_paths_and_restores_the_decoder(tmp_path):
    original = video.decode_rgb
    frames = np.zeros((2, 4, 4, 3), np.uint8)
    with quality.frames_for(tmp_path / "a.mp4", frames):
        assert np.array_equal(video.decode_rgb(str(tmp_path / "a.mp4")), frames)
        with pytest.raises(RuntimeError):
            video.decode_rgb(tmp_path / "b.mp4")
    assert video.decode_rgb is original


def synthetic(t, h=240, w=416, seed=0):
    """A smooth, textured image per seed (the quality models need natural-ish content, not noise)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w] / 40.0
    image = np.stack([np.sin(xx * rng.uniform(0.5, 2) + rng.uniform(0, 6)) + np.cos(yy * rng.uniform(0.5, 2)) for _ in range(3)], -1)
    image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
    return np.repeat(image[None], t, axis=0)


@pytest.fixture(scope="module")
def models():
    return quality.QualityModels("cpu")


@needs_weights
def test_motion_smoothness_runs_at_native_resolution(models):
    assert models.motion.anchor_resolution == 8192 * 8192  # VBench's no-resize branch, taken because the device is a torch.device
    assert quality.motion_scale(models.motion, 480, 832) == 1.0


@needs_weights
def test_a_still_video_is_fully_consistent_and_smooth(models, tmp_path):
    still = synthetic(9)
    out = models.score(tmp_path / "still.mp4", still)
    s = out["scores"]
    assert s["subject_consistency"] == pytest.approx(100.0, abs=1e-3)
    assert s["background_consistency"] == pytest.approx(100.0, abs=1e-3)
    assert s["motion_smoothness"] > 99.5
    single = models.score(tmp_path / "short.mp4", still[:3])["scores"]  # the per-frame scores of the same image (motion needs 3 frames)
    assert s["aesthetic_quality"] == pytest.approx(single["aesthetic_quality"], abs=1e-4)
    assert s["imaging_quality"] == pytest.approx(single["imaging_quality"], abs=1e-4)
    assert 0 < s["aesthetic_quality"] < 100 and 0 < s["imaging_quality"] < 100


@needs_weights
def test_alternating_frames_give_the_expected_motion_smoothness(models, tmp_path):
    a, b = synthetic(1, seed=1)[0], synthetic(1, seed=2)[0]
    frames = np.stack([a if i % 2 == 0 else b for i in range(9)])
    out = models.score(tmp_path / "alt.mp4", frames)["scores"]
    # AMT interpolates a midpoint between two copies of a; VBench compares it with b
    expected = (255.0 - np.abs(a.astype(np.float64) - b.astype(np.float64)).mean()) / 255.0 * 100.0
    assert out["motion_smoothness"] == pytest.approx(expected, abs=0.5)
    cut = models.score(tmp_path / "cut.mp4", np.concatenate([np.repeat(a[None], 5, 0), np.repeat(b[None], 4, 0)]))["scores"]
    assert cut["subject_consistency"] < 99.0 and cut["background_consistency"] < 99.0


@needs_weights
def test_scoring_twice_gives_identical_numbers(models, tmp_path):
    frames = np.concatenate([synthetic(5, seed=3), synthetic(4, seed=4)])
    path = tmp_path / "v.mp4"
    video.encode(frames, path)
    first = models.score(path)
    assert models.score(path) == first
    assert models.score(path, video.decode_rgb(path)) == first


def test_amt_leaves_no_path_that_shadows_other_utils_packages():
    import sys
    from pathlib import Path

    quality.import_paths()
    import vbench.motion_smoothness  # noqa: F401  (its build_utils appends AMT's folder to sys.path)

    quality.drop_amt_path()
    assert all(Path(p).resolve() != quality.AMT_DIR.resolve() for p in sys.path)
    assert "utils" not in sys.modules or "amt" not in str(getattr(sys.modules["utils"], "__file__", ""))


@needs_weights
def test_the_mask_pipeline_loads_after_the_quality_models(models):
    from density.masks import weights as mask_weights

    if not (mask_weights.ROOT / mask_weights.WEIGHTS["hands23"].path).exists():
        pytest.skip("Hands23 weights are not downloaded")
    from density.masks.hands23 import Hands23Predictor

    Hands23Predictor(device="cpu")  # imports Hands23's own top-level utils package
