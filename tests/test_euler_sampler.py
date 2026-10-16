"""The custom Euler loop must match diffusers' FlowMatchEulerDiscreteScheduler.

CPU comparisons are bitwise. MPS comparisons use the reference path's own
run-to-run difference as the tolerance.
"""

import pytest
import torch

from density.checks import euler as checks
from density.sampler import euler_sample, flow_match_schedule

PAIRS = checks.SHIFT_GUIDANCE_PAIRS
STEPS = checks.STEP_COUNTS
DEVICES = ["cpu", pytest.param("mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS"))]


@pytest.mark.parametrize("steps", STEPS)
@pytest.mark.parametrize("shift", [1.0, 3.0, 3.7, 5.0, 8.0])
def test_schedule_matches_diffusers(shift, steps):
    result = checks.check_schedule(shift, steps)
    assert result["passed"], result["max_abs_diff"]


def test_schedule_shape_and_endpoints():
    sigmas, timesteps = flow_match_schedule(40, 8.0)
    assert sigmas.shape == (41,) and timesteps.shape == (40,)
    assert sigmas.dtype == timesteps.dtype == torch.float32
    assert sigmas[0] == 1 and sigmas[-1] == 0
    assert torch.all(sigmas[1:] < sigmas[:-1])
    # diffusers shifts the endpoint twice: shift(shift(1/1000)) at shift 8.
    assert sigmas[-2].item() == pytest.approx(0.060207, abs=1e-6)


@pytest.mark.parametrize("steps", STEPS)
@pytest.mark.parametrize("shift", [shift for shift, _ in PAIRS])
@pytest.mark.parametrize("device", DEVICES)
def test_synthetic_trajectory_matches_scheduler_step(shift, steps, device):
    result = checks.check_synthetic(shift, steps, device)
    assert result["passed"], (result["max_abs_diff"], result["tolerance"])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("steps", STEPS)
@pytest.mark.parametrize("shift,guidance", PAIRS)
@pytest.mark.parametrize("device", DEVICES)
def test_wan_pipeline_trajectory_matches(shift, guidance, steps, dtype, device):
    result = checks.check_pipeline(shift, steps, guidance, dtype, device)
    assert result["informative"], result
    assert result["passed"], (result["max_abs_diff"], result["tolerance"])


@pytest.mark.parametrize("shift,guidance", PAIRS)
def test_wrong_schedule_is_detected(shift, guidance):
    assert checks.check_negative_control(shift, 40, guidance)["passed"]


def test_non_finite_velocity_stops_the_run():
    sigmas, timesteps = flow_match_schedule(4, 8.0)

    def velocity(x, t, n):
        return torch.full_like(x, float("nan") if n == 2 else 0.0)

    with pytest.raises(FloatingPointError, match="iteration 2"):
        euler_sample(velocity, torch.zeros(2, 3), sigmas, timesteps)
