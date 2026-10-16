"""Numerical tolerances frozen from the GPU smoke test (protocol section 11).

Registered (2026-10-07, Chase): bitwise for the four full-resolution checks below. Every observation was
bitwise in the smoke test's Session D (results/checks/smoke_test.md), and each check compares runs on one
pod, so it holds wherever the canary passes (density/canary.py).

The scope is what was measured: 832 x 480, 81 frames, on the H100 SXM with flash attention registered
(density/attention.py), comparing runs on the same device. It does not cover smaller sizes, where
Milestone 2's T3 rows (the tiny model, and the real model at 416 x 240 with 17 frames) still differ from
iteration 5 because cuBLAS gives different results for different row counts; those stay measurements.
Results across devices are reported, not judged.

Section 11 also asks that the fast path's induced LPIPS difference be small compared with the improvement
the experiment seeks. Chase recorded it as met at zero for the registered configuration: with flash
attention the fast path's frames equal the oracle's exactly for C3 and C4. LPIPS of Session B's frames
(cuDNN attention, not registered) is skipped.
"""

import torch

SCOPE = "832 x 480, 81 frames, on the H100 SXM with flash attention registered (smoke test Session D), same device"
NOT_COVERED = (
    "smaller sizes, where Milestone 2's T3 rows (the tiny model, and the real model at 416 x 240 with 17 frames) still "
    "differ from iteration 5 (cuBLAS at small row counts), and results across devices, which are reported, not judged"
)
REGISTERED = "2026-10-07, Chase"
LPIPS_CHECK = (
    "met at zero for the registered configuration (Chase, 2026-10-07): with flash attention, the fast path's frames equal "
    "the oracle's exactly for C3 and C4, so their LPIPS is zero; LPIPS of Session B's cuDNN frames is skipped"
)

# The smoke report's test names, each with its frozen rule: the latent trajectories must be identical.
FROZEN = {
    "Repeated dense run (C1 twice, same device)": {"rule": "bitwise", "latent_max_abs_diff": 0.0},
    "All HSA groups dense vs the dense loop": {"rule": "bitwise", "latent_max_abs_diff": 0.0},
    "Fast path vs masked-dense oracle at full resolution, C3 (T3)": {"rule": "bitwise", "latent_max_abs_diff": 0.0},
    "Fast path vs masked-dense oracle at full resolution, C4 (T3)": {"rule": "bitwise", "latent_max_abs_diff": 0.0},
}


def within(test: str, ours: list[torch.Tensor], reference: list[torch.Tensor]) -> bool:
    """Whether two latent trajectories meet ``test``'s frozen tolerance: every step bitwise equal."""
    if FROZEN[test]["rule"] != "bitwise":
        raise ValueError(f"{test}: unknown rule {FROZEN[test]['rule']!r}")
    return len(ours) == len(reference) and all(a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b) for a, b in zip(ours, reference))
