"""Condition registry (protocol section 3) and the run API.

``CONDITIONS`` is the one declarative table for C1-C6 and A1. ``run_condition``
runs one condition for one prompt and seed on the validated Euler loop and,
for HSA conditions, the verified HSA mechanics. It returns the output latents
and an allocation record.

For one prompt, seed and preset, every HSA condition evaluates all tokens in
the leading iterations, so g is the same for all of them. Each run therefore
also ranks by need alone and records C3's labels, which A_ROI and the no-op
test compare against.
"""

import hashlib
from dataclasses import dataclass, field

import numpy as np
import torch

from density.allocation import (
    LAMBDA_GRID,
    AssignedSchedule,
    NeedCollector,
    assign_groups,
    group_sizes,
    need_window,
    rank_binary_then_need,
    rank_coverage,
    rank_importance_need,
    rank_need,
    roi_binary,
    tie_permutation,
)
from density.diagnostics import allocation_diagnostics
from density.hsa import NUM_STEPS, PRESETS, Schedule, preset_iterations, wan_hsa_velocity
from density.sampler import euler_sample, flow_match_schedule
from density.wan import WanGuidedVelocity

# Ranking rules
NEED = "need"
IMPORTANCE_NEED = "importance_need"
COVERAGE = "coverage"
BINARY_THEN_NEED = "binary_then_need"

# Mask inputs
ROI = "roi"  # the stratum's reference ROI (equation 8)
HANDS_CONTACT = "hands_contact"  # H and C only, for A1


@dataclass(frozen=True)
class Condition:
    id: str
    steps: int
    hsa: bool  # False: dense Euler at ``steps``
    rule: str | None  # ranking rule for HSA conditions
    mask: str | None  # which mask input the rule reads
    purpose: str


EXPLORATORY = ("C7",)  # density.exploratory; kept out of CONDITIONS, which lists the registered conditions only
CONDITIONS = {
    condition.id: condition
    for condition in (
        Condition("C1", 40, False, None, None, "Dense Euler, 40 steps: full-compute reference"),
        Condition("C2", 20, False, None, None, "Dense Euler, 20 steps: naive 50 percent baseline"),
        Condition("C3", 40, True, NEED, None, "Rank by need: need-only HSA baseline"),
        Condition("C4", 40, True, IMPORTANCE_NEED, ROI, "Rank by importance times need: primary Density test"),
        Condition("C5", 40, True, COVERAGE, ROI, "Rank by fractional coverage: importance without need"),
        Condition("C6", 40, True, BINARY_THEN_NEED, ROI, "Binary ROI first, then need"),
        Condition("A1", 40, True, IMPORTANCE_NEED, HANDS_CONTACT, "C4 with hands and contact only"),
    )
}


def lookup(condition_id: str) -> Condition:
    """A registered condition, or an exploratory one (density.exploratory: C7), which never enters ``CONDITIONS``."""
    if condition_id in CONDITIONS:
        return CONDITIONS[condition_id]
    from density.exploratory import EXPLORATORY_CONDITIONS

    return EXPLORATORY_CONDITIONS[condition_id]


def sha256(value) -> str:
    """Hash of a tensor's or array's raw bytes."""
    if isinstance(value, torch.Tensor):
        data = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    else:
        data = np.ascontiguousarray(value).tobytes()
    return hashlib.sha256(data).hexdigest()


@dataclass
class AllocationRecord:
    condition: str
    prompt_id: int
    seed: int | None
    preset: str | None
    lam: float | None
    steps: int
    token_step_fraction: float
    input_sha256: dict[str, str]
    g: torch.Tensor | None = None  # need score per token
    m: torch.Tensor | None = None  # the rule's mask coverage per token
    w: np.ndarray | None = None  # importance weights (C4, A1)
    scores: object = None  # the rule's sort key: g (C3), q (C4, A1), m (C5), (b, g) (C6)
    order: torch.Tensor | None = None  # token IDs, best first
    labels: torch.Tensor | None = None
    labels_c3: torch.Tensor | None = None  # need-only labels from the same g
    tie_permutation: torch.Tensor | None = None  # C5
    group_iterations: torch.Tensor | None = None  # bool [groups, steps]
    active_counts: list[int] = field(default_factory=list)
    diagnostics: dict[str, dict] = field(default_factory=dict)  # per mask input

    def summary(self) -> dict:
        """JSON-safe summary: settings, hashes, group counts, active iterations and diagnostics."""
        arrays = {"g": self.g, "m": self.m, "w": self.w, "order": self.order, "labels": self.labels, "labels_c3": self.labels_c3, "tie_permutation": self.tie_permutation}
        hashes = {name: sha256(value) for name, value in arrays.items() if value is not None}
        out = {
            "condition": self.condition,
            "prompt_id": self.prompt_id,
            "seed": self.seed,
            "preset": self.preset,
            "lambda": self.lam,
            "steps": self.steps,
            "token_step_fraction": self.token_step_fraction,
            "input_sha256": self.input_sha256,
            "allocation_sha256": hashes,
            "diagnostics": self.diagnostics,
        }
        if self.labels is not None:
            out["group_tokens"] = torch.bincount(self.labels, minlength=2).tolist()
            out["active_iterations"] = [row.nonzero().squeeze(1).tolist() for row in self.group_iterations]
            out["active_counts"] = self.active_counts
        if self.tie_permutation is not None:
            out["tie_permutation_recipe"] = f"torch.randperm(N, generator=torch.Generator('cpu').manual_seed({self.prompt_id}))"
        return out


@dataclass
class RunResult:
    latents: torch.Tensor  # final latents, fp32
    record: AllocationRecord
    trajectory: list[torch.Tensor] | None = None  # latents after every step, if requested
    schedule: AssignedSchedule | None = None
    hsa: object = None  # the WanHSA instance, for its per-layer active-token log


def rank(condition: Condition, g: torch.Tensor, coverage: dict, lam: float | None, prompt_id: int) -> dict:
    """Apply a condition's ranking rule. C3's rule sees only g."""
    if condition.rule == NEED:
        return {"order": rank_need(g), "scores": g}
    m = coverage[condition.mask]
    if condition.rule == IMPORTANCE_NEED:
        order, w, q = rank_importance_need(g, m, lam)
        return {"order": order, "m": m, "w": w, "scores": q}
    if condition.rule == COVERAGE:
        permutation = tie_permutation(prompt_id, g.numel())
        return {"order": rank_coverage(m, permutation), "m": m, "scores": m, "tie_permutation": permutation}
    if condition.rule == BINARY_THEN_NEED:
        return {"order": rank_binary_then_need(g, m), "m": m, "scores": (roi_binary(m), g)}
    from density.exploratory import OUTSIDE_THEN_NEED, rank_outside_then_need

    if condition.rule == OUTSIDE_THEN_NEED:
        return {"order": rank_outside_then_need(g, m), "m": m, "scores": (roi_binary(m), g)}
    raise ValueError(f"unknown ranking rule {condition.rule}")


def run_condition(
    condition_id: str,
    transformer,
    *,
    prompt_id: int,
    latents: torch.Tensor,
    seed: int | None = None,
    prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor,
    guidance_scale: float,
    shift: float,
    preset: str = "HSA-50",
    lam: float | None = None,
    coverage: dict[str, torch.Tensor] | None = None,
    keep_trajectory: bool = False,
) -> RunResult:
    """Run one condition for one prompt and seed; ``latents`` is that seed's saved starting noise.

    ``coverage`` maps mask-input names (``ROI``, ``HANDS_CONTACT``) to token
    coverage m from ``density.token_map``. Diagnostics are reported for every
    mask given.
    """
    condition = lookup(condition_id)
    coverage = {name: m.detach().cpu().float() for name, m in (coverage or {}).items()}
    if condition.rule == IMPORTANCE_NEED:
        if lam is None or (lam != 0 and lam not in LAMBDA_GRID):
            raise ValueError(f"{condition_id} needs lambda from {LAMBDA_GRID} (or 0 as a control), got {lam}")
    elif lam is not None:
        raise ValueError(f"{condition_id} takes no lambda")
    if condition.mask is not None and condition.mask not in coverage:
        raise ValueError(f"{condition_id} needs the {condition.mask!r} mask input")

    pt, ph, pw = transformer.config.patch_size
    _, _, frames, height, width = latents.shape
    num_tokens = (frames // pt) * (height // ph) * (width // pw)
    for name, m in coverage.items():
        if m.shape != (num_tokens,) or m.min() < 0 or m.max() > 1:
            raise ValueError(f"coverage {name!r} must be {num_tokens} values in [0, 1]")

    sigmas, timesteps = flow_match_schedule(condition.steps, shift)
    trajectory = [] if keep_trajectory else None
    input_sha256 = {
        "latents": sha256(latents),
        "prompt_embeds": sha256(prompt_embeds),
        "negative_prompt_embeds": sha256(negative_prompt_embeds),
        **{f"coverage_{name}": sha256(m) for name, m in coverage.items()},
    }

    def record_step(n, x, v):
        if trajectory is not None:
            trajectory.append(x.clone())

    if not condition.hsa:
        velocity = WanGuidedVelocity(transformer, prompt_embeds, negative_prompt_embeds, guidance_scale)
        out = euler_sample(velocity, latents, sigmas, timesteps, on_step=record_step)
        record = AllocationRecord(condition_id, prompt_id, seed, None, None, condition.steps, condition.steps / NUM_STEPS, input_sha256)
        return RunResult(out, record, trajectory)

    registered = PRESETS[preset]
    schedule = AssignedSchedule(preset_iterations(registered), num_tokens)
    K = need_window(registered)
    if schedule.leading_iterations != K:
        raise RuntimeError(f"{preset}: schedule runs every token for {schedule.leading_iterations} iterations, K = {K}")
    high_tokens, _ = group_sizes(registered, num_tokens)
    allocation = {}

    def on_ready(g):
        ranked = rank(condition, g, coverage, lam, prompt_id)
        labels = assign_groups(ranked["order"], high_tokens)
        if condition.id in EXPLORATORY:
            from density.exploratory import check_c7

            check_c7(labels, ranked["m"])  # stops before any reduced step if the definition cannot hold
        if num_tokens == registered.num_tokens:
            Schedule.from_preset(registered, labels)  # enforces the registered group sizes
        schedule.assign(labels)
        allocation.update(ranked, g=g, labels=labels, labels_c3=assign_groups(rank_need(g), high_tokens))

    collector = NeedCollector(K, (pt, ph, pw), on_ready)

    def on_step(n, x, v):
        collector(n, x, v)
        record_step(n, x, v)

    velocity = wan_hsa_velocity(transformer, prompt_embeds, negative_prompt_embeds, guidance_scale, schedule)
    out = euler_sample(velocity, latents, sigmas, timesteps, on_step=on_step)

    record = AllocationRecord(
        condition_id,
        prompt_id,
        seed,
        preset,
        lam,
        condition.steps,
        velocity.token_step_fraction(),
        input_sha256,
        g=allocation["g"],
        m=allocation.get("m"),
        w=allocation.get("w"),
        scores=allocation["scores"],
        order=allocation["order"],
        labels=schedule.labels,
        labels_c3=allocation["labels_c3"],
        tie_permutation=allocation.get("tie_permutation"),
        group_iterations=schedule.group_iterations,
        active_counts=schedule.active_counts(),
        diagnostics={
            name: allocation_diagnostics(schedule.labels, roi_binary(m), schedule.active, allocation["labels_c3"])
            for name, m in coverage.items()
        },
    )
    if condition.id in EXPLORATORY:  # C7: the region shares of C3, C4 and C7 from this run's g
        from density.exploratory import same_g_diagnostics

        record.diagnostics["same_g"] = same_g_diagnostics(allocation["g"], allocation["m"], schedule.group_iterations, high_tokens,
                                                          schedule.labels, allocation["labels_c3"])
    return RunResult(out, record, trajectory, schedule, velocity.active_velocity_fn.hsa)
