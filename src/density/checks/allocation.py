"""Milestone 3 check: allocation rules on the verified HSA mechanics.

Run ``python -m density.checks.allocation`` to write
``results/checks/allocation_rules.{md,json}`` and ``allocation_rules.sha256``
(hashes of the demo's ignored image files). ``--tiny-only`` skips the real
model and the sphere demo.

Unit tests run at full size (N = 32,760) with synthetic g and m; model tests run
on the real-geometry tiny Transformer (CPU and MPS, fp32 and bf16) and on the
real Transformer at the wan_real resolution (MPS, bf16).

T8   need score: hand example (floor, k = 2..K, K - 1); in a run, g equals g from
     the dense loop's guided velocities at n = 0..K-1 and nothing else.
T9   hand-built ties at the group boundary give the expected labels for C3-C6.
T10  lambda = 0: C4 equals C3 (labels, active IDs, bitwise output).
T11  explicit empty mask: C4 equals C3 for every registered lambda.
T12  lambda > 0: group sizes, unmasked order, and masked-vs-unmasked dominance.
T13  C5 ignores g; its tie permutation depends only on the prompt ID.
T14  C6 puts every b = 1 token first, ordered by g within each class.
T15  token mapping (equations 10-12).
T16  diagnostics (equations 17-20) on hand-computed examples.
T17  equal active counts and budgets across C3-C6; deterministic, frozen labels.
T18  negative controls: ascending sort (T9), conditional-only g (T8), and a
     g-seeded C5 permutation (T13) must each fail.
"""

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from density.allocation import (
    LAMBDA_GRID,
    NEED_FLOOR,
    ROI_THRESHOLD,
    AssignedSchedule,
    _lexsort,
    assign_groups,
    group_sizes,
    need_score,
    need_window,
    rank_binary_then_need,
    rank_coverage,
    rank_importance_need,
    rank_need,
    roi_binary,
    tie_permutation,
)
from density.checks import euler, wan_real
from density.checks import hsa as hsa_checks
from density.conditions import CONDITIONS, HANDS_CONTACT, ROI, run_condition, sha256
from density.diagnostics import allocation_diagnostics
from density.hsa import HIGH, LOW, PRESETS, Schedule, patchify, preset_iterations, unpatchify
from density.sampler import euler_sample, flow_match_schedule, guided_velocity
from density.token_map import cell_coverage, coverage_grid, frame_groups, token_coverage
from density.wan import WanGuidedVelocity

SHIFT, GUIDANCE, STEPS = wan_real.SHIFT, wan_real.GUIDANCE, wan_real.STEPS
FULL_TOKENS = 21 * 30 * 52
PROMPT_ID = 0  # the placeholder prompt; seeds C5's tie permutation
SEED = euler.SEED

# Hand-specified masks. The sphere circle was read off R1's frames (x 137-288, y 54-199).
SPHERE = {"x": 213, "y": 126, "radius": 74}  # pixels, in R1's 416 x 240 frames, every frame
TINY_DISK = {"x": 72, "y": 56, "radius": 20}  # pixels, in the tiny model's 128 x 128 frames

REPORT_DIR = euler.REPORT_DIR
DEMO_DIR = REPORT_DIR / "allocation_demo"
HASH_FILE = REPORT_DIR / "allocation_rules.sha256"


def row(test: str, passed: bool | None, case=None, **fields) -> dict:
    out = {"test": test, **(case.describe() if case is not None else {}), "passed": passed}
    out.update(fields)
    return out


def rank_positions(order: torch.Tensor) -> np.ndarray:
    position = np.empty(order.numel(), dtype=np.int64)
    position[order.numpy()] = np.arange(order.numel())
    return position


# Synthetic inputs at full size


def synthetic_g(seed: int, num_tokens: int = FULL_TOKENS) -> torch.Tensor:
    """fp32 need scores with many exact ties (2,000 distinct values, zero included)."""
    generator = torch.Generator().manual_seed(seed)
    return (torch.randint(0, 2000, (num_tokens,), generator=generator) / 1000).float()


def synthetic_m(seed: int, num_tokens: int = FULL_TOKENS, covered: float = 0.3) -> torch.Tensor:
    """Fractional coverage k / 256 on a random share of tokens, zero elsewhere."""
    generator = torch.Generator().manual_seed(seed)
    m = torch.randint(1, 257, (num_tokens,), generator=generator) / 256
    return torch.where(torch.rand(num_tokens, generator=generator) < covered, m, torch.zeros(num_tokens)).float()


# T8: need score


def check_t8_unit(device: str = "cpu") -> list[dict]:
    # Three tokens, two-value patches, K = 3.
    # Token 0: protocol k = 2 gives |v2 - v1| / |v2| = 2 / 2, k = 3 gives 2 / 4, so g = (1 + 0.5) / 2 = 0.75.
    #          Dividing by |v_{k-1}| would give 1.0, and dividing by K would give 0.5.
    # Token 1: |v2| = 0, so the 1e-8 floor applies: g = (1 / floor + 0) / 2.
    # Token 2: constant velocity, g = 0.
    v1 = torch.tensor([[1.0, 1.0], [1.0, 0.0], [3.0, -1.0]], device=device)
    v2 = torch.tensor([[2.0, 0.0], [0.0, 0.0], [3.0, -1.0]], device=device)
    v3 = torch.tensor([[2.0, 2.0], [0.0, 0.0], [3.0, -1.0]], device=device)
    floor_term = torch.tensor(1.0) / torch.tensor(NEED_FLOOR)
    expected = torch.stack([torch.tensor(0.75), floor_term / 2, torch.tensor(0.0)])
    g = need_score([v1, v2, v3]).cpu()
    rows = [row("T8 need score hand example", torch.equal(g, expected), device=device, g=g.tolist(), expected=expected.tolist())]

    def raises(velocities, error) -> bool:
        try:
            need_score(velocities)
        except error:
            return True
        return False

    nan, inf = v2.clone(), v2.clone()
    nan[0, 0], inf[2, 1] = float("nan"), float("inf")
    rows.append(row("T8 stops on NaN and infinity", raises([v1, nan, v3], FloatingPointError) and raises([v1, v2, inf], FloatingPointError), device=device))
    rows.append(row("T8 rejects K < 2 and non-fp32 input", raises([v1], ValueError) and raises([v1.half(), v2.half()], TypeError), device=device))
    # K is the all-active prefix: every token runs at iterations 0..K-1, and iteration K skips some.
    prefix_ok = all(
        preset_iterations(p)[:, : need_window(p)].all() and not preset_iterations(p)[:, need_window(p)].all() for p in PRESETS.values()
    )
    rows.append(
        row(
            "T8 K is the all-active prefix for every preset",
            [need_window(p) for p in PRESETS.values()] == [5, 10, 9, 12] and prefix_ok,
            K={name: need_window(p) for name, p in PRESETS.items()},
        )
    )
    return rows


class _CapturingVelocity(WanGuidedVelocity):
    """The dense guided velocity, also keeping each step's guided and conditional-only predictions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.guided, self.conditional = [], []

    @torch.no_grad()
    def __call__(self, latents, timestep, n):
        hidden_states = latents.to(self.dtype)
        timestep = timestep.expand(latents.shape[0])
        v_cond = self._predict(hidden_states, timestep, self.prompt_embeds)
        v_uncond = self._predict(hidden_states, timestep, self.negative_prompt_embeds)
        v = guided_velocity(v_uncond, v_cond, self.guidance_scale)
        patch = self.transformer.config.patch_size
        self.guided.append(patchify(v, patch)[0])
        self.conditional.append(patchify(v_cond.float(), patch)[0])
        return v


def check_t8_model(case, g_run: torch.Tensor) -> list[dict]:
    """In-run g equals g from the dense loop's guided velocities at n = 0..K-1, and no alternative."""
    K = need_window(PRESETS["HSA-50"])
    sigmas, timesteps = flow_match_schedule(STEPS, SHIFT)
    capture = _CapturingVelocity(case.transformer, case.prompt_embeds, case.negative_prompt_embeds, GUIDANCE)
    euler_sample(capture, case.latents, sigmas[: K + 2], timesteps[: K + 1])
    reference = need_score(capture.guided[:K])
    alternatives = {
        "v_2..v_K+1 (shifted)": need_score(capture.guided[1 : K + 1]),
        "v_1..v_K+1 (one extra)": need_score(capture.guided[: K + 1]),
    }
    conditional = need_score(capture.conditional[:K])
    matches = torch.equal(g_run, reference)
    others_differ = {name: not torch.equal(g_run, alt) for name, alt in alternatives.items()}
    return [
        row("T8 run uses guided v at n = 0..K-1", matches and all(others_differ.values()), case, matches_reference=matches, alternatives_differ=others_differ),
        row("T18 conditional-only g fails T8", not torch.equal(conditional, g_run), case, max_abs_diff=(conditional - g_run).abs().max().item()),
    ]


# T9: hand-built ties at the group boundary


T9_G = torch.tensor([0.5, 0.9, 0.5, 0.2, 0.5, 0.9, 0.1, 0.5])
T9_HIGH = 3


def check_t9(c3_rank=rank_need, label: str = "T9") -> list[dict]:
    """Eight tokens, three full-step slots, with a tie straddling the boundary for every rule."""
    g = T9_G
    rows = []
    # C3: 0.9 x2 (1, 5), then 0.5 x4 (0, 2, 4, 7): token 0 wins the boundary tie by ID.
    order = c3_rank(g)
    rows.append(row(f"{label} C3 ties", order.tolist() == [1, 5, 0, 2, 4, 7, 3, 6] and _high(order) == {0, 1, 5}, order=order.tolist()))

    # C4, lambda = 1: q = 0.75 for tokens 2 and 7 at the boundary; token 2 wins by ID.
    m4 = torch.tensor([0, 0, 0.5, 0, 0, 0, 0, 0.5])
    order, _, _ = rank_importance_need(g, m4, 1.0)
    rows.append(row(f"{label} C4 ties", order.tolist() == [1, 5, 2, 7, 0, 4, 3, 6] and _high(order) == {1, 2, 5}, order=order.tolist()))

    # C5: four tokens share m = 0.5 for three slots; the first three in permutation order win.
    m5 = torch.tensor([0.5, 0, 0.5, 0, 0.5, 0, 0, 0.5])
    permutation = tie_permutation(7, 8)
    expected = set([t for t in permutation.tolist() if t in (0, 2, 4, 7)][:3])
    order = rank_coverage(m5, permutation)
    rows.append(row(f"{label} C5 ties", _high(order) == expected, order=order.tolist(), permutation=permutation.tolist()))

    # C6: m = 25/256 is below the 0.1 threshold, 26/256 is above. b = 1 for tokens 2 and 7,
    # then tokens 1 and 5 tie at g = 0.9 for the last slot; token 1 wins by ID.
    m6 = torch.tensor([25 / 256, 0, 26 / 256, 0, 0, 0, 0, 0.5])
    order = rank_binary_then_need(g, m6)
    rows.append(row(f"{label} C6 ties", order.tolist() == [2, 7, 1, 5, 0, 4, 3, 6] and _high(order) == {1, 2, 7}, order=order.tolist()))

    labels = assign_groups(order, T9_HIGH)
    rows.append(row(f"{label} group sizes filled exactly", torch.bincount(labels, minlength=2).tolist() == [3, 5]))
    return rows


def _high(order: torch.Tensor, high: int = T9_HIGH) -> set:
    return set(order[:high].tolist())


def ascending_need(g: torch.Tensor) -> torch.Tensor:
    """Negative control for T9: g ascending instead of descending."""
    return _lexsort(g.double().numpy(), np.arange(g.numel()))


def check_full_size_sizes() -> list[dict]:
    rows = []
    g, m = synthetic_g(1), synthetic_m(2)
    for name, preset in PRESETS.items():
        high, low = group_sizes(preset, FULL_TOKENS)
        ok = (high, low) == (preset.high_tokens, preset.low_tokens)
        for order in (rank_need(g), rank_importance_need(g, m, 3.0)[0], rank_coverage(m, tie_permutation(PROMPT_ID, FULL_TOKENS)), rank_binary_then_need(g, m)):
            labels = assign_groups(order, high)
            Schedule.from_preset(preset, labels)  # raises if the sizes differ from the registered ones
            ok &= torch.bincount(labels, minlength=2).tolist() == [preset.high_tokens, preset.low_tokens]
        rows.append(row("T9 registered group sizes at N = 32,760", ok, preset=name, group_sizes=[high, low]))
    return rows


# T12: lambda > 0 against C3


def t12_invariants(g: torch.Tensor, m: torch.Tensor, lam: float, high: int) -> dict:
    c3, (c4, _, _) = rank_need(g), rank_importance_need(g, m, lam)
    masked = (m > 0).numpy()
    r3, r4 = rank_positions(c3), rank_positions(c4)
    unmasked_c3 = c3.numpy()[~masked[c3.numpy()]]
    unmasked_c4 = c4.numpy()[~masked[c4.numpy()]]
    # Unmasked tokens ahead of each masked token: with unmasked order preserved, "never falls behind
    # an unmasked token it led" is the same as this count never growing.
    ahead3 = np.cumsum(~masked[c3.numpy()])[r3] - (~masked)
    ahead4 = np.cumsum(~masked[c4.numpy()])[r4] - (~masked)
    labels3, labels4 = assign_groups(c3, high), assign_groups(c4, high)
    return {
        "group_sizes_equal": torch.equal(torch.bincount(labels3, minlength=2), torch.bincount(labels4, minlength=2)),
        "unmasked_order_kept": bool(np.array_equal(unmasked_c3, unmasked_c4)),
        "unmasked_only_fall": bool((r4[~masked] >= r3[~masked]).all()),
        "masked_keep_lead_over_unmasked": bool((ahead4[masked] <= ahead3[masked]).all()),
        "roi_full_steps_only_rise": int(((labels4 == HIGH) & torch.from_numpy(masked)).sum()) >= int(((labels3 == HIGH) & torch.from_numpy(masked)).sum()),
        "masked_only_rise": bool((r4[masked] <= r3[masked]).all()),
    }


def check_t12() -> list[dict]:
    high = PRESETS["HSA-50"].high_tokens
    g = synthetic_g(3)
    rows = []
    general = ["group_sizes_equal", "unmasked_order_kept", "unmasked_only_fall", "masked_keep_lead_over_unmasked", "roi_full_steps_only_rise"]
    for lam in LAMBDA_GRID:
        result = t12_invariants(g, synthetic_m(4), lam, high)
        rows.append(row("T12 fractional mask", all(result[k] for k in general), lam=lam, **result))
    uniform = torch.where(synthetic_m(5) > 0, torch.tensor(0.5), torch.tensor(0.0))
    for lam in LAMBDA_GRID:
        result = t12_invariants(g, uniform, lam, high)
        rows.append(row("T12 uniform mask: masked tokens only rise", all(result.values()), lam=lam, **result))
    return rows


def t12_counterexample() -> dict:
    """The literal "masked tokens only rise" fails with fractional coverage."""
    g = torch.tensor([1.0, 0.9])  # token A leads token B under C3
    m = torch.tensor([0.1, 1.0])  # both masked; B far more covered
    c3, (c4, _, q) = rank_need(g), rank_importance_need(g, m, 10.0)
    return {"c3_order": c3.tolist(), "c4_order": c4.tolist(), "c4_scores": q.tolist(), "masked_token_fell": c3.tolist() == [0, 1] and c4.tolist() == [1, 0]}


# T13: C5 ignores g


def boundary_tie_coverage(high: int, num_tokens: int = FULL_TOKENS) -> torch.Tensor:
    """m = 1 for high - 200 tokens and 0.5 for the next 1,000, so the boundary lies inside a tie block."""
    generator = torch.Generator().manual_seed(SEED)
    chosen = torch.randperm(num_tokens, generator=generator)
    m = torch.zeros(num_tokens)
    m[chosen[: high - 200]] = 1.0
    m[chosen[high - 200 : high + 800]] = 0.5
    return m


def check_t13(permutation_for=lambda prompt_id, g: tie_permutation(prompt_id, g.numel()), label: str = "T13") -> dict:
    high = PRESETS["HSA-50"].high_tokens
    m = boundary_tie_coverage(high)
    g1, g2 = synthetic_g(6), synthetic_g(7)
    labels = lambda prompt_id, g: assign_groups(rank_coverage(m, permutation_for(prompt_id, g)), high)
    ignores_g = torch.equal(labels(PROMPT_ID, g1), labels(PROMPT_ID, g2))
    reproducible = torch.equal(permutation_for(PROMPT_ID, g1), permutation_for(PROMPT_ID, g1))
    differs_by_prompt = not torch.equal(labels(PROMPT_ID, g1), labels(PROMPT_ID + 1, g1))
    return row(label, ignores_g and reproducible and differs_by_prompt, ignores_g=ignores_g, reproducible=reproducible, differs_across_prompt_ids=differs_by_prompt)


def g_seeded_permutation(prompt_id: int, g: torch.Tensor) -> torch.Tensor:
    """Negative control for T13: a tie permutation seeded from g."""
    seed = int(hashlib.sha256(g.numpy().tobytes()).hexdigest()[:15], 16)
    return torch.randperm(g.numel(), generator=torch.Generator().manual_seed(seed))


# T14: C6 classes


def check_t14() -> dict:
    g = synthetic_g(8)
    m = torch.tensor([0.0, 25 / 256, 26 / 256, 0.5, 1.0])[torch.randint(0, 5, (FULL_TOKENS,), generator=torch.Generator().manual_seed(9))]
    order = rank_binary_then_need(g, m).numpy()
    b, gs = roi_binary(m).numpy()[order], g.numpy()[order]
    split = int(b.sum())
    classes_ok = b[:split].all() and not b[split:].any()
    within_ok = all((np.diff(gs[s]) <= 0).all() for s in (slice(0, split), slice(split, None)))
    same = (np.diff(b.astype(int)) == 0) & (np.diff(gs) == 0)
    ties_ok = bool((np.diff(order)[same] > 0).all())
    return row("T14 C6 binary first, then g", bool(classes_ok and within_ok and ties_ok), roi_tokens=split, classes=bool(classes_ok), g_descending=bool(within_ok), ties_by_id=ties_ok)


# T15: token mapping


def check_t15() -> list[dict]:
    rows = []
    half = torch.zeros(1, 48, 48, dtype=torch.bool)
    half[0, 16:32, 16:24] = True  # the left half of the centre cell
    a = cell_coverage(half)
    m = coverage_grid(half)
    rows.append(row("T15 half-covered cell gives 0.5", a[0, 1, 1].item() == 0.5 and a.sum().item() == 0.5 and bool((m == 0.5).all()), cell=a[0].tolist(), dilated=m[0].tolist()))

    groups = frame_groups(81)
    expected = [[0]] + [[4 * j - 3, 4 * j - 2, 4 * j - 1, 4 * j] for j in range(1, 21)]
    mapped = []
    for f in range(81):
        single = torch.zeros(81, 16, 16, dtype=torch.bool)
        single[f] = True
        mapped.append(coverage_grid(single)[:, 0, 0].nonzero().squeeze(1).tolist())
    exact = all(mapped[f] == [0 if f == 0 else (f + 3) // 4] for f in range(81))
    rows.append(row("T15 frame groups map exactly", groups == expected and exact and frame_groups(17)[-1] == [13, 14, 15, 16]))

    cell = torch.zeros(81, 480, 832, dtype=torch.bool)
    cell[9:13, 80:96, 112:128] = True  # pixel frames 9-12 -> latent frame 3; cell (5, 7)
    cell[0, 0:16, 0:16] = True  # pixel frame 0 -> latent frame 0; corner cell
    m = coverage_grid(cell)
    nonzero = set(map(tuple, m.nonzero().tolist()))
    expected_cells = {(3, r, c) for r in (4, 5, 6) for c in (6, 7, 8)} | {(0, r, c) for r in (0, 1) for c in (0, 1)}
    rows.append(row("T15 dilation is spatial only, zero outside the grid", nonzero == expected_cells, nonzero_cells=len(nonzero)))

    shapes = {
        "tiny 9 x 128 x 128": (coverage_grid(torch.zeros(9, 128, 128)).shape, (3, 8, 8)),
        "small 17 x 240 x 416": (coverage_grid(torch.zeros(17, 240, 416)).shape, (5, 15, 26)),
        "full 81 x 480 x 832": (m.shape, (21, 30, 52)),
    }
    shapes_ok = all(tuple(got) == want for got, want in shapes.values()) and token_coverage(cell).numel() == FULL_TOKENS
    rows.append(row("T15 shapes at tiny, small and full size", shapes_ok, shapes={k: list(v[0]) for k, v in shapes.items()}))

    # Token IDs must follow the Transformer's patch order: token (j, r, c) is latent pixels (2r, 2c) at frame j.
    token = 3 * 30 * 52 + 5 * 52 + 7
    one_hot = torch.zeros(1, FULL_TOKENS, 64)
    one_hot[0, token] = 1
    latent = unpatchify(one_hot, torch.Size((1, 16, 21, 60, 104)), (1, 2, 2))[0, 0]
    location = latent.nonzero().tolist()
    rows.append(row("T15 token order matches the Transformer", token_coverage(cell)[token].item() == 1.0 and location == [[3, 10, 14], [3, 10, 15], [3, 11, 14], [3, 11, 15]]))
    return rows


# T16: diagnostics


def check_t16() -> list[dict]:
    rows = []
    roi = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=torch.bool)
    labels = torch.full((10,), LOW)
    labels[[0, 1, 5]] = HIGH
    active = torch.zeros(2, 10, dtype=torch.bool)
    active[0] = True
    active[1, [0, 1, 5]] = True
    reference = labels.clone()
    reference[torch.tensor([1, 7])] = torch.tensor([LOW, HIGH])  # token 1 (ROI) and token 7 (outside) differ from C3
    d = allocation_diagnostics(labels, roi, active, reference)
    expected = {"precision": Fraction(2, 3), "recall": Fraction(1, 2), "normalized_recall": Fraction(2, 3), "lift": Fraction(5, 3), "s_roi": Fraction(6, 13), "a_roi": Fraction(1, 4)}
    close = all(abs(d[k] - float(v)) < 1e-15 for k, v in expected.items())
    rows.append(row("T16 hand example (P, R, R_norm, lift, S_ROI, A_ROI)", close and d["full_step_roi_tokens"] == 2 and d["no_op"] is False, **{k: d[k] for k in expected}))

    same = allocation_diagnostics(labels, roi, active, labels.clone())
    outside_only = labels.clone()
    outside_only[9] = HIGH
    moved = allocation_diagnostics(outside_only, roi, active, labels)
    rows.append(
        row(
            "T16 no-op only when every label matches",
            same["no_op"] is True and moved["no_op"] is False and moved["a_roi"] == 0.0,
            identical=same["no_op"],
            one_token_outside_roi_moved=moved["no_op"],
            a_roi_when_only_outside_moved=moved["a_roi"],
        )
    )

    empty = allocation_diagnostics(labels, torch.zeros(10, dtype=torch.bool), active, reference)
    na = [k for k in ("precision", "recall", "normalized_recall", "lift", "s_roi", "a_roi") if empty[k] is not None]
    rows.append(row("T16 empty ROI reports N/A", not na and empty["no_op"] is False, defined_anyway=na))
    return rows


# Assignment


def check_assignment() -> list[dict]:
    schedule = AssignedSchedule(preset_iterations(PRESETS["HSA-50"]), 8)
    before = schedule.leading_iterations == 5 and torch.equal(schedule.active_ids(4), torch.arange(8))
    try:
        schedule.active_ids(5)
        blocks_unassigned = False
    except RuntimeError:
        blocks_unassigned = True
    labels = torch.tensor([HIGH, LOW, LOW, HIGH, LOW, LOW, LOW, LOW])
    schedule.assign(labels)
    labels[0] = LOW  # mutating the caller's tensor must not reach the schedule
    try:
        schedule.assign(labels)
        frozen = False
    except RuntimeError:
        frozen = True
    isolated = schedule.labels[0].item() == HIGH and torch.equal(schedule.active_ids(5), torch.tensor([0, 3]))
    return [row("T17 labels assigned once, then frozen", before and blocks_unassigned and frozen and isolated)]


# Model tests


def disk_mask(frames: int, height: int, width: int, x: int, y: int, radius: int) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    return ((xx - x) ** 2 + (yy - y) ** 2 <= radius**2).expand(frames, height, width).clone()


def run(case, condition: str, lam=None, coverage=None, keep_trajectory=True):
    return run_condition(
        condition,
        case.transformer,
        prompt_id=PROMPT_ID,
        latents=case.latents,
        prompt_embeds=case.prompt_embeds,
        negative_prompt_embeds=case.negative_prompt_embeds,
        guidance_scale=GUIDANCE,
        shift=SHIFT,
        lam=lam,
        coverage=coverage,
        keep_trajectory=keep_trajectory,
    )


def same_allocation(a, b) -> dict:
    """Labels, active IDs at every iteration and branch, and every step's latents."""
    ids = all(torch.equal(a.hsa.logged_ids[k], b.hsa.logged_ids[k]) for k in a.hsa.logged_ids) and a.hsa.logged_ids.keys() == b.hsa.logged_ids.keys()
    return {
        "labels_equal": torch.equal(a.record.labels, b.record.labels),
        "active_ids_equal": ids,
        "output_bitwise": all(torch.equal(x, y) for x, y in zip(a.trajectory, b.trajectory)),
    }


def model_suite(case, roi: torch.Tensor, hands_contact: torch.Tensor, dense: list[torch.Tensor] | None) -> tuple[list[dict], dict]:
    rows = []
    coverage = {ROI: roi, HANDS_CONTACT: hands_contact}
    runs = {"C3": run(case, "C3", coverage=coverage)}

    rows += check_t8_model(case, runs["C3"].record.g)

    c1 = run(case, "C1")
    runs["C1"] = c1
    if dense is not None:
        rows.append(row("registry: C1 equals the dense loop", all(torch.equal(x, y) for x, y in zip(c1.trajectory, dense)), case))

    runs["C4 lambda 0"] = run(case, "C4", 0.0, coverage)
    result = same_allocation(runs["C4 lambda 0"], runs["C3"])
    rows.append(row("T10 lambda = 0 equals C3", all(result.values()), case, **result))

    empty = {ROI: torch.zeros_like(roi)}
    for lam in LAMBDA_GRID:
        result = same_allocation(run(case, "C4", lam, empty), runs["C3"])
        rows.append(row("T11 empty mask equals C3", all(result.values()), case, lam=lam, **result))

    for name, condition, lam in (("C4 lambda 3", "C4", 3.0), ("C5", "C5", None), ("C6", "C6", None), ("A1 lambda 3", "A1", 3.0)):
        runs[name] = run(case, condition, lam, coverage)

    compared = ["C3", "C4 lambda 3", "C5", "C6", "A1 lambda 3"]
    counts = {name: [(n, layer, branch, count) for n, layer, branch, count in runs[name].hsa.log] for name in compared}
    fractions = {name: runs[name].record.token_step_fraction for name in compared}
    g_equal = all(torch.equal(runs[name].record.g, runs["C3"].record.g) for name in compared)
    c3_in_c4 = torch.equal(runs["C4 lambda 3"].record.labels_c3, runs["C3"].record.labels)
    repeat = run(case, "C4", 3.0, coverage)
    deterministic = torch.equal(repeat.record.labels, runs["C4 lambda 3"].record.labels) and all(
        torch.equal(x, y) for x, y in zip(repeat.trajectory, runs["C4 lambda 3"].trajectory)
    )
    try:
        runs["C4 lambda 3"].schedule.assign(runs["C3"].record.labels)
        frozen = False
    except RuntimeError:
        frozen = True
    counts_equal = all(counts[name] == counts["C3"] for name in compared)
    fractions_equal = len(set(fractions.values())) == 1
    rows.append(
        row(
            "T17 equal active counts and budgets across C3-C6 and A1",
            counts_equal and fractions_equal and g_equal and c3_in_c4 and deterministic and frozen,
            case,
            counts_equal_every_iteration_layer_branch=counts_equal,
            token_step_fractions=fractions,
            g_bitwise_equal=g_equal,
            c3_labels_inside_c4_match_c3_run=c3_in_c4,
            deterministic_repeat=deterministic,
            relabel_refused=frozen,
            log_entries=len(counts["C3"]),
        )
    )

    c2 = run(case, "C2", keep_trajectory=False)
    rows.append(row("registry: C2 runs 20 dense steps at fraction 0.5", c2.record.token_step_fraction == 0.5 and CONDITIONS["C2"].steps == 20, case))
    return rows, runs


# Sphere demo (real model, informational)


def roi_latent_mask(b: torch.Tensor, latent_shape, patch) -> torch.Tensor:
    """Latent elements belonging to binary-ROI tokens."""
    tokens = b.float()[None, :, None].expand(1, b.numel(), int(np.prod(patch)) * latent_shape[1])
    return unpatchify(tokens, latent_shape, patch) > 0.5


def sphere_demo(case, runs: dict, roi: torch.Tensor) -> dict:
    for lam in (1.0, 10.0):
        runs[f"C4 lambda {lam:g}"] = run(case, "C4", lam, {ROI: roi}, keep_trajectory=False)
    reference = runs["C1"].latents
    inside = roi_latent_mask(roi_binary(roi), reference.shape, case.transformer.config.patch_size).to(reference.device)
    results = {}
    for name in ("C3", "C4 lambda 1", "C4 lambda 3", "C4 lambda 10", "C5", "C6"):
        x = runs[name].latents
        d = runs[name].record.diagnostics[ROI]
        error = (x - reference) ** 2
        results[name] = {
            "precision": d["precision"],
            "recall": d["recall"],
            "normalized_recall": d["normalized_recall"],
            "lift": d["lift"],
            "a_roi": d["a_roi"],
            "s_roi": d["s_roi"],
            "changed_tokens": d["changed_tokens"],
            "no_op": d["no_op"],
            "mean_coverage_full_step": roi[runs[name].record.labels == HIGH].mean().item(),
            "latent_mse_inside": error[inside].mean().item(),
            "latent_mse_outside": error[~inside].mean().item(),
        }
    shares = [results[f"C4 lambda {lam:g}"]["precision"] for lam in LAMBDA_GRID]
    return {
        "circle": SPHERE,
        "roi_tokens": int(roi_binary(roi).sum()),
        "full_step_slots": int((runs["C3"].record.labels == HIGH).sum()),
        "num_tokens": roi.numel(),
        "c4_full_step_roi_share_saturated": all(share == 1.0 for share in shares),
        "inside_definition": "latent elements of tokens with b = 1[m >= 0.1], via unpatchify",
        # None when saturated: P = 1 at every lambda cannot show a rise.
        "c4_full_step_roi_share_rises_with_lambda": None if all(x == 1.0 for x in shares) else bool(all(a <= b for a, b in zip(shares, shares[1:]))),
        "conditions": results,
    }


def save_overlay(mask: torch.Tensor, roi: torch.Tensor, demo_dir: Path = DEMO_DIR) -> Path:
    """R1 frame 8 with the circle tinted and the binary-ROI token cells of its latent frame outlined."""
    demo_dir.mkdir(parents=True, exist_ok=True)
    frame = 8
    image = Image.open(wan_real.OUT_DIR / "r1_frames" / f"frame_{frame:03d}.png").convert("RGB")
    tint = Image.new("RGB", image.size, (255, 0, 0))
    image = Image.composite(Image.blend(image, tint, 0.35), image, Image.fromarray(mask[frame].numpy().astype(np.uint8) * 255))
    draw = ImageDraw.Draw(image)
    rows, cols = image.size[1] // 16, image.size[0] // 16
    latent_frame = (frame + 3) // 4
    b = roi_binary(roi).view(-1, rows, cols)[latent_frame]
    for r, c in b.nonzero().tolist():
        draw.rectangle([c * 16, r * 16, c * 16 + 15, r * 16 + 15], outline=(0, 90, 255))
    path = demo_dir / "mask_overlay.png"
    path.write_bytes(wan_real.png_bytes(np.asarray(image).astype(np.float64) / 255))
    return path


# Runs and report


def unit_tests(devices: list[str]) -> list[dict]:
    rows = [r for device in devices for r in check_t8_unit(device)] + check_t9() + check_full_size_sizes() + check_t12()
    rows.append(row("T12 literal claim fails for fractional coverage (counterexample)", None, **t12_counterexample()))
    rows += [check_t13(), check_t14()] + check_t15() + check_t16() + check_assignment()
    t9_ascending = check_t9(ascending_need, label="T18 ascending sort")
    rows.append(row("T18 ascending sort fails T9", not all(r["passed"] for r in t9_ascending), failing=[r["test"] for r in t9_ascending if not r["passed"]]))
    rows.append(row("T18 g-seeded C5 permutation fails T13", not check_t13(g_seeded_permutation, label="T13 with g-seeded permutation")["passed"]))
    return rows


def tiny_tests(devices: list[str]) -> list[dict]:
    rows = []
    disk = disk_mask(9, 128, 128, **TINY_DISK)
    roi = token_coverage(disk)
    hands_contact = token_coverage(disk_mask(9, 128, 128, 40, 88, 16))
    for device in devices:
        for dtype in (torch.float32, torch.bfloat16):
            case = hsa_checks.tiny_case(device, dtype)
            rows += model_suite(case, roi, hands_contact, hsa_checks.dense_trajectory(case))[0]
    return rows


def real_tests(weights_dir: Path, device: str = "mps", out_dir: Path = REPORT_DIR) -> tuple[list[dict], dict, dict]:
    recorded = json.loads(hsa_checks.WAN_REAL_JSON.read_text())
    case, verification = hsa_checks.load_real_case(weights_dir, recorded, device)
    mask = disk_mask(17, 240, 416, **SPHERE)
    roi = token_coverage(mask)
    # A1 needs a second mask input; the left half of the circle stands in for hands and contact.
    left = mask.clone()
    left[:, :, SPHERE["x"] :] = False
    rows, runs = model_suite(case, roi, token_coverage(left), dense=None)
    frames = hsa_checks.frame_hashes_match(runs["C1"].latents, weights_dir, recorded, device)
    judged = frames["frames_matching_record"] == frames["frames"] if frames["frames_judged"] else None  # MPS-recorded hashes
    rows.insert(0, row("registry: C1 reproduces the committed R1 frames", judged, case, **frames))
    demo = sphere_demo(case, runs, roi)
    overlay = save_overlay(mask, roi, out_dir / "allocation_demo")
    artifacts = {str(overlay.relative_to(out_dir)): wan_real.file_sha256(overlay)}
    demo["records"] = {name: r.record.summary() for name, r in runs.items() if name in demo["conditions"]}
    return rows, demo, {"verification": verification, "artifacts_sha256": artifacts}


def run_all(devices: list[str], weights_dir: Path | None, real_device: str = "mps", out_dir: Path = REPORT_DIR) -> dict:
    rows = unit_tests(devices) + tiny_tests(devices)
    demo, real = None, None
    if weights_dir is not None:
        real_rows, demo, real = real_tests(weights_dir, real_device, out_dir)
        rows += real_rows
    judged = [r for r in rows if r["passed"] is not None]
    return {
        "milestone": "Allocation rules: need score, C3-C6 ranking, token mapping, diagnostics",
        "passed": all(r["passed"] for r in judged) and real is not None,
        "all_judged_passed": all(r["passed"] for r in judged),
        "real_model_ran": real is not None,
        "num_judged": len(judged),
        "num_failed": sum(not r["passed"] for r in judged),
        "environment": euler.environment(),
        "registered": {
            "need_floor": NEED_FLOOR,
            "roi_threshold": ROI_THRESHOLD,
            "lambda_grid": LAMBDA_GRID,
            "need_window_K": {name: need_window(p) for name, p in PRESETS.items()},
            "need_window_definition": "K = number of leading iterations in which every token runs; labels assigned after iteration K - 1",
            "tie_rules": {
                "C3": "g descending, then token ID ascending",
                "C4": "(1 + lambda m) g descending (float64, exact), then token ID ascending",
                "C5": "m descending, then position in torch.randperm(N) seeded by the prompt ID",
                "C6": "b descending, then g descending, then token ID ascending",
            },
            "small_resolution_groups": "registered high-group share, rounded (31 of 192 tiny tokens, 312 of 1,950 real tokens)",
        },
        "device_scope": (
            "The need score and every model test run on CPU and MPS. Ranking (numpy lexsort in float64), token mapping and "
            "the diagnostics run on CPU by design, so their results do not depend on the device."
        ),
        "t12_note": (
            "The literal T12 statement, that masked tokens can only rise in rank relative to C3, is false when coverage is "
            "fractional: a masked token can be overtaken by a more-covered masked token. The judged invariants are the ones "
            "that hold for any coverage: equal group sizes, unmasked tokens keep their relative order and can only fall, no "
            "masked token falls behind an unmasked token it led, and the ROI's full-step count can only rise. The literal "
            "statement is judged for uniform coverage, where it holds."
        ),
        "conditions": {c.id: {"steps": c.steps, "hsa": c.hsa, "rule": c.rule, "mask": c.mask, "purpose": c.purpose} for c in CONDITIONS.values()},
        "real_model": real,
        "sphere_demo": demo,
        "results": rows,
    }


def _fmt(value) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def markdown_report(report: dict) -> str:
    env = report["environment"]
    if report["passed"]:
        overall = "PASS"
    elif report["all_judged_passed"] and not report["real_model_ran"]:
        overall = "INCOMPLETE (real model not run)"
    else:
        overall = "FAIL"
    lines = [
        "# Allocation rules",
        "",
        f"**Overall: {overall}** ({report['num_judged'] - report['num_failed']}/{report['num_judged']} judged checks passed)",
        "",
        f"Code `{env['git_sha']}` (dirty: {env['git_dirty']}); torch {env['torch']}, diffusers {env['diffusers']}, {env['platform']}.",
        "",
        "| Test | Model | Device | Dtype | Case | Result |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in report["results"]:
        case = r.get("preset") or (f"lambda {r['lam']:g}" if "lam" in r else "")
        status = "info" if r["passed"] is None else ("PASS" if r["passed"] else "FAIL")
        lines.append(f"| {r['test']} | {r.get('model', 'unit')} | {r.get('device', '')} | {r.get('dtype', '')} | {case} | {status} |")

    reg = report["registered"]
    lines += [
        "",
        "## Registered values",
        "",
        f"- Need-score floor {reg['need_floor']:g}; ROI threshold {reg['roi_threshold']:g}; lambda grid {', '.join(f'{x:g}' for x in reg['lambda_grid'])} (0 only as a control).",
        "- K (leading iterations in which every token runs): " + ", ".join(f"{k} {v}" for k, v in reg["need_window_K"].items()) + ".",
        "- Tie rules: " + "; ".join(f"{k}: {v}" for k, v in reg["tie_rules"].items()) + ".",
        f"- Small-resolution group sizes: {reg['small_resolution_groups']}.",
        f"- Devices: {report['device_scope']}",
        "",
        "## T12 note",
        "",
        report["t12_note"],
    ]

    demo = report["sphere_demo"]
    if demo:
        lines += [
            "",
            "## Sphere demo (informational, no quality claims)",
            "",
            f"Circle at x {demo['circle']['x']}, y {demo['circle']['y']}, radius {demo['circle']['radius']} px on every frame: "
            f"{demo['roi_tokens']} of {demo['num_tokens']} tokens in the binary ROI. Placeholder embeddings, synthetic mask, 416 x 240. "
            f"Inside the mask means {demo['inside_definition']}. Latent MSE is against R1 (C1).",
            "",
            "| Condition | P | R | R_norm | Lift | A_ROI | S_ROI | Changed tokens | Mean m, full-step | MSE inside | MSE outside |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for name, d in demo["conditions"].items():
            lines.append(
                f"| {name} | {_fmt(d['precision'])} | {_fmt(d['recall'])} | {_fmt(d['normalized_recall'])} | {_fmt(d['lift'])} | "
                f"{_fmt(d['a_roi'])} | {_fmt(d['s_roi'])} | {d['changed_tokens']} | {_fmt(d['mean_coverage_full_step'])} | "
                f"{_fmt(d['latent_mse_inside'])} | {_fmt(d['latent_mse_outside'])} |"
            )
        if demo["c4_full_step_roi_share_saturated"]:
            lines += [
                "",
                f"C4's ROI share of full-step slots rising with lambda is not testable at this mask size: the binary ROI "
                f"({demo['roi_tokens']} tokens) is larger than the full-step group ({demo['full_step_slots']} slots), so every "
                "full-step slot goes to the ROI from lambda = 1 on and P stays 1. The mean coverage of the full-step tokens is "
                "reported alongside, without interpretation.",
            ]
        else:
            lines += ["", f"C4's ROI share of full-step slots rises with lambda: {demo['c4_full_step_roi_share_rises_with_lambda']}."]

    if report["real_model"]:
        v = report["real_model"]["verification"]
        lines += [
            "",
            "## Real-model inputs",
            "",
            f"- Inputs match wan_real_mps.json: {v['inputs_file_matches_record'] and v['input_tensors_match_record']}; weights match: {v['weights_match_record']}.",
            f"- Demo images are not versioned; their hashes are in the JSON and `{HASH_FILE.name}`.",
        ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Allocation rules check")
    parser.add_argument("--tiny-only", action="store_true", help="skip the real Wan2.1 Transformer and the demo")
    parser.add_argument("--weights", type=Path, default=wan_real.WEIGHTS_DIR)
    parser.add_argument("--real-device", default="mps", help="device for the real Transformer (default: mps)")
    euler.device_args(parser)
    args = parser.parse_args()
    out = args.report_dir
    report = run_all(args.devices or euler.default_devices(), None if args.tiny_only else args.weights, args.real_device, out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "allocation_rules.json").write_text(json.dumps(report, indent=2, default=str))
    (out / "allocation_rules.md").write_text(markdown_report(report))
    if report["real_model"]:
        (out / HASH_FILE.name).write_text("".join(f"{h}  {p}\n" for p, h in report["real_model"]["artifacts_sha256"].items()))
    print(markdown_report(report))
    return 0 if report["all_judged_passed"] and (report["real_model_ran"] or args.tiny_only) else 1


if __name__ == "__main__":
    raise SystemExit(main())
