"""HSA mechanics (Milestone 2) on the tiny random Wan Transformer and synthetic velocities.

The tiny Transformer has the real model's attention geometry (128-dim heads,
512 text tokens), so every comparison is judged bitwise in fp32 and bf16. The
real-model versions, including the final acceptance gate, run in
``python -m density.checks.hsa``.
"""

import pytest
import torch

from density.checks import hsa as checks
from density.hsa import HIGH, LOW, PRESETS, Schedule, active_iterations, patchify, unpatchify

MPS = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS")
DEVICES = ["cpu", pytest.param("mps", marks=MPS)]
CASES = [
    pytest.param((device, dtype), id=f"{device}-{name}", marks=[MPS] if device == "mps" else [])
    for device in ("cpu", "mps")
    for dtype, name in ((torch.float32, "fp32"), (torch.bfloat16, "bf16"))
]


@pytest.fixture(scope="module", params=CASES)
def case(request):
    device, dtype = request.param
    return checks.tiny_case(device, dtype)


# Schedule


@pytest.mark.parametrize("name", list(PRESETS))
def test_registered_preset_at_full_size(name):
    (row,) = [r for r in checks.check_schedule() if r["preset"] == name]
    assert row["passed"], row


def test_equation_13_by_hand():
    expected = [n for n in range(40) if n < 4 or n >= 36 or n % 4 == 0]
    assert active_iterations((4, 36), 10).nonzero().squeeze(1).tolist() == expected
    assert active_iterations((12, 36), 5).nonzero().squeeze(1).tolist() == list(range(12)) + [16, 24, 32, 36, 37, 38, 39]


def test_schedule_rejects_wrong_group_sizes():
    labels = checks.random_labels(checks.FULL_TOKENS, 5000, seed=0)
    with pytest.raises(ValueError, match="5242 high"):
        Schedule.from_preset(PRESETS["HSA-50"], labels)


def test_schedule_requires_every_token_at_iteration_0():
    iterations = torch.ones(2, 40, dtype=torch.bool)
    iterations[LOW, 0] = False
    with pytest.raises(ValueError, match="iteration 0"):
        Schedule(torch.tensor([HIGH, LOW]), iterations)


def test_patchify_round_trip():
    latents = torch.randn(1, 16, 21, 60, 104)
    tokens = patchify(latents, (1, 2, 2))
    assert tokens.shape == (1, checks.FULL_TOKENS, 64)
    assert torch.equal(unpatchify(tokens, latents.shape, (1, 2, 2)), latents)


# Dense equivalence: T0, T1, T2 and the oracle


def test_dense_equivalence(case):
    dense = checks.dense_trajectory(case)
    t0 = checks.check_token_locality(case)
    t1, _ = checks.check_t1(case, dense)
    t2 = checks.check_t2(case)
    oracle = checks.check_oracle(case, dense, checks.reference_trajectory(case, checks.hsa50_style(case.num_tokens, 0)))
    for row in [t0, t1, t2, *oracle]:
        assert row["passed"], row


# Skipping: T3, T5, T6, T7


def test_skipping_matches_oracle_and_controls_fail(case):
    rows, _ = checks.check_skipping(case)
    failures = [row for row in rows if not row["passed"]]
    assert not failures, failures


# Cached Euler update: T4


@pytest.mark.parametrize("name", list(PRESETS))
@pytest.mark.parametrize("device", DEVICES)
def test_synthetic_cached_update(device, name):
    row = checks.check_t4(device, PRESETS[name])
    assert row["passed"], row


@pytest.mark.parametrize("device", DEVICES)
def test_synthetic_control_detects_varying_low_group(device):
    assert checks.check_t4(device, PRESETS["HSA-50"], varying_low=True)["passed"]
