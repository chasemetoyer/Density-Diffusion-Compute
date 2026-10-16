"""Condition registry and run API on the real-geometry tiny Transformer (Milestone 3).

The real-model versions and the sphere demo run in ``python -m density.checks.allocation``.
"""

import pytest
import torch

from density.checks import allocation as checks
from density.checks import hsa as hsa_checks
from density.conditions import CONDITIONS, HANDS_CONTACT, ROI, run_condition
from density.token_map import token_coverage

MPS = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS")
CASES = [
    pytest.param((device, dtype), id=f"{device}-{name}", marks=[MPS] if device == "mps" else [])
    for device in ("cpu", "mps")
    for dtype, name in ((torch.float32, "fp32"), (torch.bfloat16, "bf16"))
]


@pytest.fixture(scope="module", params=CASES)
def case(request):
    return hsa_checks.tiny_case(*request.param)


def test_registry():
    table = {c.id: (c.steps, c.hsa, c.rule, c.mask) for c in CONDITIONS.values()}
    assert table == {
        "C1": (40, False, None, None),
        "C2": (20, False, None, None),
        "C3": (40, True, "need", None),
        "C4": (40, True, "importance_need", ROI),
        "C5": (40, True, "coverage", ROI),
        "C6": (40, True, "binary_then_need", ROI),
        "A1": (40, True, "importance_need", HANDS_CONTACT),
    }


def test_run_api_rejects_unregistered_inputs():
    case = hsa_checks.tiny_case("cpu", torch.float32)
    m = token_coverage(checks.disk_mask(9, 128, 128, **checks.TINY_DISK))
    common = dict(prompt_id=0, latents=case.latents, prompt_embeds=case.prompt_embeds, negative_prompt_embeds=case.negative_prompt_embeds, guidance_scale=6.0, shift=8.0)
    with pytest.raises(ValueError, match="lambda"):
        run_condition("C4", case.transformer, lam=2.0, coverage={ROI: m}, **common)
    with pytest.raises(ValueError, match="takes no lambda"):
        run_condition("C3", case.transformer, lam=1.0, **common)
    with pytest.raises(ValueError, match="mask input"):
        run_condition("C5", case.transformer, **common)


def test_model_suite(case):
    roi = token_coverage(checks.disk_mask(9, 128, 128, **checks.TINY_DISK))
    hands_contact = token_coverage(checks.disk_mask(9, 128, 128, 40, 88, 16))
    rows, _ = checks.model_suite(case, roi, hands_contact, hsa_checks.dense_trajectory(case))
    failures = [r for r in rows if r["passed"] is not True]
    assert not failures, failures
