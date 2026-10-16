"""Milestone 2 check: HSA mechanics on the Wan2.1 Transformer.

Run ``python -m density.checks.hsa`` to write
``results/checks/hsa_mechanics.{md,json}``; ``--tiny-only`` skips the real
model. The real-model tests need the weights and the inputs saved by
``density.checks.wan_real``, and verify both against ``wan_real_mps.json``.

Test oracle: ``MaskedDenseReference`` runs every token through diffusers' own
forward pass. Only the self-attention processor changes: it computes K/V for
all tokens, writes the active rows into the cache, and lets every query attend
to the cache, so inactive tokens are seen through their cached rows. Inactive
tokens' outputs are discarded and their cached velocity reused. This equals
HSA only if self-attention is the sole token-mixing operation (T0).

T0  token locality: with self-attention made token-local, perturbing one
    token's input changes only that token's output.
T1  every group at every step: the HSA path equals the dense loop bitwise.
T2  every token active: each block's output equals diffusers' dense forward.
T3  HSA-50-style skipping: the fast path equals the oracle at every step.
T4  synthetic velocity, constant for low-group tokens: HSA equals dense, and
    each skipped token moves by exactly dt times its cached velocity.
T5  negative control: one KV cache shared by both branches must fail T3.
T6  negative controls: no RoPE gather, and no cache refresh, must fail T3.
T7  active counts match the schedule at every iteration, layer and branch,
    and the measured token-step fraction equals the planned one.

The tiny model uses the real model's attention geometry (see FIXTURE_CHANGE),
and every tiny test is judged bitwise in fp32 and bf16 on CPU and MPS. The
real-model T3 is the final acceptance gate: the overall verdict is PASS only
if it ran and passed for every label seed.
"""

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import diffusers
import numpy as np
import torch
from diffusers import AutoencoderKLWan, WanTransformer3DModel
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_wan import WanAttnProcessor, _get_qkv_projections
from diffusers.video_processor import VideoProcessor

from density.attention import generation_attention
from density.checks import euler, wan_real
from density.hsa import (
    HIGH,
    LOW,
    PRESETS,
    HSAVelocity,
    KVCache,
    Schedule,
    WanHSA,
    attend,
    patchify,
    preset_iterations,
    project_qkv,
    self_attention_processors,
    unpatchify,
    wan_hsa_velocity,
)
from density.sampler import NUM_TRAIN_TIMESTEPS, euler_sample, flow_match_schedule, guided_velocity
from density.wan import WanGuidedVelocity, decode_latents

SHIFT, GUIDANCE, STEPS = wan_real.SHIFT, wan_real.GUIDANCE, wan_real.STEPS
SIGMAS, TIMESTEPS = flow_match_schedule(STEPS, SHIFT)
HSA50 = PRESETS["HSA-50"]
LABEL_SEEDS = (0, 1, 2)
SEED = euler.SEED

# Protocol section 5: low-group evaluation counts and token-step fractions at N = 32,760.
REGISTERED = {"HSA-50": (16, 0.496007), "HSA-75B": (25, 0.748752), "switch-20": (19, 0.5), "switch-30": (19, 0.5)}
FULL_TOKENS = 21 * 30 * 52

REPORT_DIR = euler.REPORT_DIR
WAN_REAL_JSON = REPORT_DIR / "wan_real_mps.json"

DIFFUSERS_WAN = "diffusers/models/transformers/transformer_wan.py"
TOKEN_MIXING_EVIDENCE = [
    ("patch_embedding", "WanTransformer3DModel.__init__ (line 598)", "Conv3d with kernel_size == stride == patch_size, no padding: each token sees only its own patch"),
    ("rope", "WanRotaryPosEmbed.forward (lines 395-416)", "depends on the latent shape only; one frequency row per token"),
    ("condition_embedder", "WanTimeTextImageEmbedding.forward (lines 330-351)", "timestep and text only; shared by every token"),
    ("norm1, norm2, norm3, norm_out", "FP32LayerNorm (normalization.py lines 85-93)", "F.layer_norm over the channel dimension of each token"),
    ("attn1 q/k/v, norm_q, norm_k, to_out", "WanAttnProcessor.__call__ (lines 93-100, 160-161)", "Linear and RMSNorm over the channel dimension of each token"),
    ("attn1 attention", "WanAttnProcessor.__call__ (lines 143-153)", "softmax(QK^T)V over all video tokens: the only token-mixing operation"),
    ("attn2 (cross-attention)", "WanTransformerBlock.forward (line 494)", "queries are video tokens, keys and values are text tokens: no video token reads another"),
    ("ffn", "FeedForward.forward (attention.py), GELU (activations.py lines 87-90)", "Linear, tanh-GELU, Linear on each token"),
    ("modulation and residuals", "WanTransformerBlock.forward (lines 483-502)", "elementwise with a per-sample timestep table"),
    ("proj_out and unpatchify", "WanTransformer3DModel.forward (lines 723-730)", "Linear on each token, then a reshape and permute that places each token's patch"),
]


# Model cases


@dataclass
class Case:
    model: str  # "tiny" or "real"
    device: str
    dtype: torch.dtype
    transformer: object
    latents: torch.Tensor
    prompt_embeds: torch.Tensor
    negative_prompt_embeds: torch.Tensor

    @property
    def num_tokens(self) -> int:
        pt, ph, pw = self.transformer.config.patch_size
        _, _, frames, height, width = self.latents.shape
        return (frames // pt) * (height // ph) * (width // pw)

    def describe(self) -> dict:
        return {"model": self.model, "device": self.device, "dtype": "bf16" if self.dtype == torch.bfloat16 else "fp32"}


# The real Transformer has 12 heads of 128 dims and 512 text tokens. Attention
# kernels pick their accumulation order by head size and key length, so the
# tiny fixture keeps that geometry and runs the same kernels.
TINY_CONFIG = dict(euler.TINY_TRANSFORMER_CONFIG, attention_head_dim=128, ffn_dim=64)
TINY_TEXT_LEN = 512

FIXTURE_CHANGE = (
    "The HSA tests first used diffusers' unit-test tiny Transformer: 2 heads of 12 dims and 8 text tokens. "
    "Its judged T3 failed once, on MPS in bf16 for label seed 0 at step 30. At that head size the attention kernel's "
    "accumulation order depends on the number of queries, and the fast path and the oracle send different query "
    "counts. With identical q, K and V the kernel's outputs differ by one rounding step; the diagnostic below "
    "recomputes this evidence on every run. The fixture now uses the real model's attention geometry, 2 heads of 128 dims "
    "and 512 text tokens, so the tiny tests exercise the same kernels as the real model. Every tiny test, including the "
    "negative controls, is judged bitwise in fp32 and bf16. The real-model T3 remains the final acceptance gate."
)


def tiny_case(device: str, dtype: torch.dtype) -> Case:
    latents, prompt_embeds, negative_prompt_embeds = euler.tiny_inputs(device, TINY_TEXT_LEN)
    return Case("tiny", device, dtype, euler.tiny_transformer(dtype, device, TINY_CONFIG), latents, prompt_embeds, negative_prompt_embeds)


def tiny_hd12_case(device: str, dtype: torch.dtype) -> Case:
    """The first fixture (diffusers' unit-test geometry), kept as a diagnostic."""
    latents, prompt_embeds, negative_prompt_embeds = euler.tiny_inputs(device)
    return Case("tiny-hd12", device, dtype, euler.tiny_transformer(dtype, device), latents, prompt_embeds, negative_prompt_embeds)


# Schedules


def random_labels(num_tokens: int, high_tokens: int, seed: int) -> torch.Tensor:
    """Fixed random group assignment: ``high_tokens`` tokens in the HIGH group."""
    labels = torch.full((num_tokens,), LOW, dtype=torch.long)
    labels[torch.randperm(num_tokens, generator=torch.Generator().manual_seed(seed))[:high_tokens]] = HIGH
    return labels


def scaled_high_tokens(num_tokens: int, preset=HSA50) -> int:
    """HSA-50-style group size at a reduced token count: the registered share, rounded."""
    return round(num_tokens * preset.high_tokens / preset.num_tokens)


def hsa50_style(num_tokens: int, seed: int) -> Schedule:
    return Schedule(random_labels(num_tokens, scaled_high_tokens(num_tokens), seed), preset_iterations(HSA50))


def all_full(num_tokens: int, seed: int = 0) -> Schedule:
    """Both groups run every iteration; the labels still exercise the two-group code path."""
    return Schedule(random_labels(num_tokens, scaled_high_tokens(num_tokens), seed), torch.ones(2, STEPS, dtype=torch.bool))


def first_skip_iteration(schedule: Schedule) -> int:
    return next(n for n in range(schedule.num_steps) if not schedule.active[n].all())


# Runs


def assert_stock_processors(transformer) -> None:
    """Dense comparisons are meaningful only if no HSA processor was left installed."""
    for layer, block in enumerate(transformer.blocks):
        if type(block.attn1.processor) is not WanAttnProcessor:
            raise RuntimeError(f"block {layer} self-attention uses {type(block.attn1.processor).__name__}")


def trajectory(velocity_fn, case: Case) -> list[torch.Tensor]:
    return euler.custom_trajectory(velocity_fn, case.latents, SIGMAS, TIMESTEPS)


def dense_trajectory(case: Case) -> list[torch.Tensor]:
    assert_stock_processors(case.transformer)
    return trajectory(WanGuidedVelocity(case.transformer, case.prompt_embeds, case.negative_prompt_embeds, GUIDANCE), case)


def hsa_trajectory(case: Case, schedule: Schedule, hsa_cls=WanHSA):
    velocity = wan_hsa_velocity(case.transformer, case.prompt_embeds, case.negative_prompt_embeds, GUIDANCE, schedule, hsa_cls)
    return trajectory(velocity, case), velocity


def first_divergence(ours: list[torch.Tensor], ref: list[torch.Tensor]) -> int | None:
    return next((n for n, (a, b) in enumerate(zip(ours, ref)) if not torch.equal(a, b)), None)


def comparison(ours: list[torch.Tensor], ref: list[torch.Tensor]) -> dict:
    stats = euler.compare_trajectories(ours, ref)
    return {
        "bitwise_equal": stats["bitwise_equal"],
        "max_abs_diff": stats["max_abs_diff"],
        "first_divergent_step": first_divergence(ours, ref),
    }


def record(test: str, case: Case | None, passed: bool | None, **fields) -> dict:
    row = {"test": test, **(case.describe() if case is not None else {}), "passed": passed}
    row.update(fields)
    return row


# Test oracle


def latent_token_mask(active: torch.Tensor, latent_shape: torch.Size, patch_size) -> torch.Tensor:
    """bool (1, 1, F, H, W): True on every latent element of an active token. Independent of patchify."""
    _, _, frames, height, width = latent_shape
    pt, ph, pw = patch_size
    grid = active.view(frames // pt, height // ph, width // pw)
    grid = grid.repeat_interleave(pt, 0).repeat_interleave(ph, 1).repeat_interleave(pw, 2)
    return grid.view(1, 1, frames, height, width)


class _MaskedCacheAttention:
    def __init__(self, reference: "MaskedDenseReference", layer: int):
        self.reference = reference
        self.layer = layer

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, rotary_emb=None):
        branch, ids = self.reference.context
        query, key, value = project_qkv(attn, hidden_states, rotary_emb)
        if (branch, self.layer) not in self.reference.caches:
            self.reference.caches[(branch, self.layer)] = KVCache(hidden_states.shape[1])
        cache = self.reference.caches[(branch, self.layer)]
        rows = ids.to(key.device)
        cache.write(ids, key[:, rows], value[:, rows])
        cached_key, cached_value = cache.read()
        return attend(attn, query, cached_key, cached_value)


class MaskedDenseReference:
    """Slow oracle for HSA: dense compute, HSA semantics imposed by masking (see module docstring)."""

    def __init__(self, transformer, prompt_embeds, negative_prompt_embeds, guidance_scale, schedule: Schedule):
        self.transformer = transformer
        self.dtype = transformer.dtype
        self.embeds = {"cond": prompt_embeds.to(self.dtype), "uncond": negative_prompt_embeds.to(self.dtype)}
        self.guidance_scale = guidance_scale
        self.schedule = schedule
        self.caches: dict[tuple[str, int], KVCache] = {}
        self.velocity: torch.Tensor | None = None
        self.context = None

    @torch.no_grad()
    def __call__(self, latents, timestep, n):
        active = self.schedule.active[n]
        ids = active.nonzero().squeeze(1)
        hidden_states = latents.to(self.dtype)
        out = {}
        for branch in ("cond", "uncond"):
            self.context = (branch, ids)
            with generation_attention(latents.device), self_attention_processors(self.transformer, lambda layer: _MaskedCacheAttention(self, layer)):
                out[branch] = self.transformer(
                    hidden_states=hidden_states,
                    timestep=timestep.expand(latents.shape[0]),
                    encoder_hidden_states=self.embeds[branch],
                    return_dict=False,
                )[0]
        fresh = guided_velocity(out["uncond"], out["cond"], self.guidance_scale)
        if self.velocity is None:
            if not active.all():
                raise RuntimeError("cached velocity read before it was written")
            self.velocity = fresh
        else:
            mask = latent_token_mask(active, latents.shape, self.transformer.config.patch_size).to(latents.device)
            self.velocity = torch.where(mask, fresh, self.velocity)
        return self.velocity


def reference_trajectory(case: Case, schedule: Schedule) -> list[torch.Tensor]:
    reference = MaskedDenseReference(case.transformer, case.prompt_embeds, case.negative_prompt_embeds, GUIDANCE, schedule)
    return trajectory(reference, case)


# Negative controls: each replaces one step of the fast path.


class SharedCacheHSA(WanHSA):
    """T5: both guidance branches read and write one cache."""

    def _cache(self, branch, layer):
        return super()._cache("shared", layer)


class NoRopeGatherHSA(WanHSA):
    """T6: RoPE rows taken in order instead of gathered by token ID."""

    def _rotary(self, rotary_emb, ids):
        return tuple(freqs[:, : ids.numel()] for freqs in rotary_emb)


class NoCacheRefreshHSA(WanHSA):
    """T6: the cache is filled once at iteration 0 and never refreshed."""

    def _write(self, cache, ids, key, value):
        if cache.key is None:
            super()._write(cache, ids, key, value)


# T0: token locality


class _TokenLocalAttention:
    """Each token attends only to itself; softmax over one key is 1, so the output is the token's own V."""

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, rotary_emb=None):
        query, key, value = project_qkv(attn, hidden_states, rotary_emb)
        return attn.to_out[1](attn.to_out[0](value.flatten(2, 3).type_as(query)))


@torch.no_grad()
def check_token_locality(case: Case) -> dict:
    tr = case.transformer
    patch = tr.config.patch_size
    probe = case.num_tokens // 2 + 1
    perturbed = patchify(case.latents, patch).clone()
    perturbed[:, probe] += 1.0
    perturbed = unpatchify(perturbed, case.latents.shape, patch)

    def changed_tokens() -> list[int]:
        out = [
            tr(hidden_states=x.to(case.dtype), timestep=TIMESTEPS[0].to(case.device).expand(1), encoder_hidden_states=case.prompt_embeds.to(case.dtype), return_dict=False)[0]
            for x in (case.latents, perturbed)
        ]
        diff = (patchify(out[0], patch) != patchify(out[1], patch)).any(dim=-1)[0]
        return diff.nonzero().squeeze(1).tolist()

    with self_attention_processors(tr, lambda layer: _TokenLocalAttention()):
        local = changed_tokens()
    assert_stock_processors(tr)
    stock = changed_tokens()
    return record(
        "T0 token locality",
        case,
        local == [probe] and len(stock) > 1,
        probe_token=probe,
        changed_with_local_attention=local,
        changed_with_stock_attention=len(stock),
        num_tokens=case.num_tokens,
    )


# T1, T2, oracle validation


def check_t1(case: Case, dense: list[torch.Tensor]) -> tuple[dict, list[torch.Tensor]]:
    ours, _ = hsa_trajectory(case, all_full(case.num_tokens))
    stats = comparison(ours, dense)
    return record("T1 all groups full vs dense", case, stats["bitwise_equal"], **stats), ours


def _block_outputs(transformer, run):
    outputs = []
    handles = [block.register_forward_hook(lambda module, args, out: outputs.append(out.clone())) for block in transformer.blocks]
    try:
        final = run()
    finally:
        for handle in handles:
            handle.remove()
    return outputs, final


@torch.no_grad()
def check_t2(case: Case) -> dict:
    tr = case.transformer
    patch = tr.config.patch_size
    hsa = WanHSA(tr)
    ids = torch.arange(case.num_tokens)
    per_layer_equal, max_diff = [], 0.0
    for n in (0, 1):  # n = 1 overwrites rows already in the cache
        for branch, embeds in (("cond", case.prompt_embeds), ("uncond", case.negative_prompt_embeds)):
            t = TIMESTEPS[n].to(case.device)
            assert_stock_processors(tr)
            dense_blocks, dense_out = _block_outputs(
                tr, lambda: tr(hidden_states=case.latents.to(case.dtype), timestep=t.expand(1), encoder_hidden_states=embeds.to(case.dtype), return_dict=False)[0]
            )
            hsa_blocks, hsa_tokens = _block_outputs(tr, lambda: hsa.forward(case.latents, t, embeds, ids, branch, n))
            pairs = list(zip(hsa_blocks, dense_blocks)) + [(unpatchify(hsa_tokens, case.latents.shape, patch), dense_out)]
            per_layer_equal.append([torch.equal(a, b) for a, b in pairs])
            max_diff = max(max_diff, max((a.float() - b.float()).abs().max().item() for a, b in pairs))
    passed = all(all(row) for row in per_layer_equal)
    return record("T2 all active, layer by layer", case, passed, bitwise_equal=passed, max_abs_diff=max_diff, layers_compared=len(per_layer_equal[0]))


def check_oracle(case: Case, dense: list[torch.Tensor], reference_hsa50: list[torch.Tensor]) -> list[dict]:
    full = comparison(reference_trajectory(case, all_full(case.num_tokens)), dense)
    skipping = comparison(reference_hsa50, dense)
    return [
        record("oracle all groups full vs dense", case, full["bitwise_equal"], **full),
        record("oracle HSA-50-style differs from dense", case, not skipping["bitwise_equal"], **skipping),
    ]


# T3, T5, T6, T7


def check_t7(case: Case, schedule: Schedule, velocity: HSAVelocity, seed: int) -> dict:
    hsa = velocity.active_velocity_fn.hsa
    counts = schedule.active_counts()
    expected_entries = schedule.num_steps * len(case.transformer.blocks) * 2
    counts_ok = len(hsa.log) == expected_entries and all(count == counts[n] for n, _, _, count in hsa.log)
    ids_ok = all(
        torch.equal(hsa.logged_ids[(n, "cond")], schedule.active_ids(n)) and torch.equal(hsa.logged_ids[(n, "uncond")], schedule.active_ids(n))
        for n in range(schedule.num_steps)
    )
    measured = velocity.token_step_fraction()
    planned = schedule.token_step_fraction()
    return record(
        "T7 active counts and budget",
        case,
        counts_ok and ids_ok and measured == planned,
        seed=seed,
        log_entries=len(hsa.log),
        counts_match=counts_ok,
        same_ids_both_branches=ids_ok,
        measured_fraction=measured,
        planned_fraction=planned,
    )


def check_skipping(case: Case) -> tuple[list[dict], list[torch.Tensor]]:
    """T3 for every label seed, T7 on the same runs, and T5/T6 against the seed-0 oracle."""
    rows, reference_seed0 = [], None
    for seed in LABEL_SEEDS:
        schedule = hsa50_style(case.num_tokens, seed)
        reference = reference_trajectory(case, schedule)
        ours, velocity = hsa_trajectory(case, schedule)
        stats = comparison(ours, reference)
        rows.append(record("T3 fast path vs oracle", case, stats["bitwise_equal"], seed=seed, **stats))
        rows.append(check_t7(case, schedule, velocity, seed))
        if seed == LABEL_SEEDS[0]:
            reference_seed0 = reference

    schedule = hsa50_style(case.num_tokens, LABEL_SEEDS[0])
    skip = first_skip_iteration(schedule)
    # Each bug should first show at the earliest iteration it can affect.
    controls = [("T5 shared KV cache", SharedCacheHSA, skip), ("T6 no RoPE gather", NoRopeGatherHSA, skip), ("T6 no cache refresh", NoCacheRefreshHSA, 1)]
    for test, bug, step in controls:
        try:
            ours, _ = hsa_trajectory(case, schedule, bug)
            stats = comparison(ours, reference_seed0)
            mode = "numerical"
        except RuntimeError as error:
            stats, mode = {"bitwise_equal": False, "max_abs_diff": None, "first_divergent_step": None}, f"assertion: {error}"
        caught = not stats["bitwise_equal"] and stats["first_divergent_step"] == step
        rows.append(record(test, case, caught, seed=LABEL_SEEDS[0], failure_mode=mode, expected_first_divergence=step, **stats))
    return rows, reference_seed0


# Head-dim-12 diagnostic: where the fast path and the oracle part ways, and why


def _module_names(transformer) -> list[str]:
    names = []
    for layer in range(len(transformer.blocks)):
        names += [f"blocks.{layer}.attn1", f"blocks.{layer}.attn2", f"blocks.{layer}.ffn"]
    return names + ["proj_out"]


def _clone(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(_clone(v) for v in value)
    return value


def _cross_qkv(attn, hidden_states, text):
    """Cross-attention q, k, v as WanAttnProcessor computes them (no RoPE)."""
    query, key, value = _get_qkv_projections(attn, hidden_states, text)
    query, key = attn.norm_q(query), attn.norm_k(key)
    return tuple(t.unflatten(2, (attn.heads, -1)) for t in (query, key, value))


def _sdpa(query, key, value):
    return dispatch_attention_fn(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, backend=None, parallel_config=None)


@torch.no_grad()
def divergence_evidence(case: Case, schedule: Schedule, step: int) -> dict:
    """At iteration ``step``, find the first module whose active-row output differs between the
    two paths, then test whether its attention kernel alone explains the difference."""
    tr = case.transformer
    names = _module_names(tr)
    active = schedule.active_ids(step).to(case.device)
    captured = {"fast": [], "oracle": []}
    state = {"n": None, "path": None}

    def hook(name):
        def fn(module, args, out):
            if state["n"] == step:
                captured[state["path"]].append((name, _clone(args), out.clone()))
        return fn

    def tracked(velocity_fn, path):
        def fn(latents, timestep, n):
            state["n"], state["path"] = n, path
            return velocity_fn(latents, timestep, n)
        return fn

    reference = MaskedDenseReference(tr, case.prompt_embeds, case.negative_prompt_embeds, GUIDANCE, schedule)
    fast = wan_hsa_velocity(tr, case.prompt_embeds, case.negative_prompt_embeds, GUIDANCE, schedule)
    handles = [tr.get_submodule(name).register_forward_hook(hook(name)) for name in names]
    try:
        for velocity_fn, path in ((reference, "oracle"), (fast, "fast")):
            euler.custom_trajectory(tracked(velocity_fn, path), case.latents, SIGMAS[: step + 2], TIMESTEPS[: step + 1])
    finally:
        for handle in handles:
            handle.remove()

    index = next(
        (i for i, ((_, _, f), (_, _, o)) in enumerate(zip(captured["fast"], captured["oracle"])) if not torch.equal(f, o[:, active])), None
    )
    if index is None:
        return {"step": step, "module": "none"}
    name, fast_args, _ = captured["fast"][index]
    _, oracle_args, _ = captured["oracle"][index]
    branch = ("cond", "uncond")[index // len(names)]
    evidence = {"step": step, "module": name, "branch": branch, "module_inputs_equal": torch.equal(fast_args[0], oracle_args[0][:, active])}

    attn = tr.get_submodule(name)
    if name.endswith("attn1"):
        layer = int(name.split(".")[1])
        q_fast = project_qkv(attn, fast_args[0], fast_args[3])[0]
        q_oracle = project_qkv(attn, oracle_args[0], oracle_args[3])[0]
        k_fast, v_fast = fast.active_velocity_fn.hsa.caches[branch][layer].read()
        k_oracle, v_oracle = reference.caches[(branch, layer)].read()
    elif name.endswith("attn2"):
        q_fast, k_fast, v_fast = _cross_qkv(attn, fast_args[0], fast_args[1])
        q_oracle, k_oracle, v_oracle = _cross_qkv(attn, oracle_args[0], oracle_args[1])
    else:
        return evidence

    evidence["qkv_equal"] = torch.equal(q_oracle[:, active], q_fast) and torch.equal(k_oracle, k_fast) and torch.equal(v_oracle, v_fast)
    all_queries, active_queries = _sdpa(q_oracle, k_oracle, v_oracle)[:, active], _sdpa(q_fast, k_fast, v_fast)
    all_fp32 = _sdpa(q_oracle.float(), k_oracle.float(), v_oracle.float())[:, active]
    active_fp32 = _sdpa(q_fast.float(), k_fast.float(), v_fast.float())
    evidence.update(
        queries=[q_oracle.shape[1], q_fast.shape[1]],
        sdpa_elements_differing=int((all_queries != active_queries).sum()),
        sdpa_elements=all_queries.numel(),
        sdpa_max_abs_diff=(all_queries.float() - active_queries.float()).abs().max().item(),
        sdpa_fp32_max_abs_diff=(all_fp32 - active_fp32).abs().max().item(),
    )
    return evidence


def hd12_diagnostic(device: str, dtype: torch.dtype) -> dict:
    case = tiny_hd12_case(device, dtype)
    runs = []
    for seed in LABEL_SEEDS:
        schedule = hsa50_style(case.num_tokens, seed)
        ours, _ = hsa_trajectory(case, schedule)
        runs.append({"seed": seed, **comparison(ours, reference_trajectory(case, schedule))})
    first = next((r for r in runs if r["first_divergent_step"] is not None), None)
    evidence = divergence_evidence(case, hsa50_style(case.num_tokens, first["seed"]), first["first_divergent_step"]) if first else None
    return {**case.describe(), "runs": runs, "evidence_seed": first["seed"] if first else None, "evidence": evidence}


# T4: synthetic velocity at the full latent size


def check_t4(device: str, preset, varying_low: bool = False) -> dict:
    patch = (1, 2, 2)
    schedule = Schedule.from_preset(preset, random_labels(FULL_TOKENS, preset.high_tokens, SEED))
    generator = torch.Generator("cpu").manual_seed(SEED)
    latents = torch.randn(euler.FULL_LATENT_SHAPE, generator=generator).to(device)
    constant = torch.randn(1, FULL_TOKENS, 64, generator=generator).to(device)
    low = (schedule.labels == LOW).to(device)

    def token_velocity(tokens, t, rows):
        s = t / NUM_TRAIN_TIMESTEPS
        varying = torch.tanh(tokens * (1 + s)) - (1 - s) * torch.sin(3 * tokens)
        return varying if varying_low else torch.where(low[rows][None, :, None], constant[:, rows], varying)

    all_rows = torch.arange(FULL_TOKENS, device=device)
    dense = euler.custom_trajectory(
        lambda x, t, n: unpatchify(token_velocity(patchify(x, patch), t, all_rows), x.shape, patch), latents, SIGMAS, TIMESTEPS
    )

    last_fresh = torch.empty(1, FULL_TOKENS, 64, device=device)

    def active_fn(x, t, n, ids):
        rows = ids.to(device)
        fresh = token_velocity(patchify(x, patch)[:, rows], t, rows)
        last_fresh[:, rows] = fresh
        return fresh

    velocity = HSAVelocity(active_fn, schedule, patch)
    sigmas = SIGMAS.to(device)
    hsa_traj = []
    state = {"x": latents, "moves_exact": True}

    def on_step(n, x, v):
        # A skipped token must receive its cached velocity and move by exactly dt times it.
        skipped = (~schedule.active[n]).nonzero().squeeze(1).to(device)
        if skipped.numel():
            cached = last_fresh[:, skipped]
            expected = patchify(state["x"], patch)[:, skipped] + (sigmas[n + 1] - sigmas[n]) * cached
            state["moves_exact"] &= torch.equal(patchify(v, patch)[:, skipped], cached) and torch.equal(patchify(x, patch)[:, skipped], expected)
        state["x"] = x.clone()
        hsa_traj.append(x.clone())

    euler_sample(velocity, latents, SIGMAS, TIMESTEPS, on_step=on_step)
    stats = comparison(hsa_traj, dense)
    low_evals, registered_fraction = REGISTERED[preset.name]
    measured = velocity.token_step_fraction()
    if varying_low:
        return record(
            "T4 control: varying low-group velocity differs", None, not stats["bitwise_equal"], device=device, preset=preset.name, **stats
        )
    return record(
        "T4 synthetic, full size",
        None,
        stats["bitwise_equal"] and state["moves_exact"] and round(measured, 6) == registered_fraction,
        device=device,
        preset=preset.name,
        skipped_moves_exact=state["moves_exact"],
        measured_fraction=measured,
        registered_fraction=registered_fraction,
        **stats,
    )


# Schedule at full size


def check_schedule() -> list[dict]:
    rows = []
    for name, preset in PRESETS.items():
        schedule = Schedule.from_preset(preset, random_labels(FULL_TOKENS, preset.high_tokens, SEED))
        low_evals, fraction = REGISTERED[name]
        evaluations = schedule.group_evaluations()
        measured = schedule.token_step_fraction()
        rows.append(
            record(
                "schedule (N = 32,760)",
                None,
                preset.num_tokens == FULL_TOKENS and evaluations == [STEPS, low_evals] and round(measured, 6) == fraction,
                preset=name,
                group_sizes=[preset.high_tokens, preset.low_tokens],
                window=list(preset.window),
                stride=STEPS // preset.low_steps,
                evaluations=evaluations,
                token_step_fraction=measured,
                registered=[low_evals, fraction],
                low_group_iterations=preset_iterations(preset)[LOW].nonzero().squeeze(1).tolist(),
            )
        )
    return rows


# Real model


def load_real_case(weights_dir: Path, recorded: dict, device: str = "mps") -> tuple[Case, dict]:
    inputs_path = wan_real.OUT_DIR / "inputs.pt"
    file_hash = wan_real.file_sha256(inputs_path)
    inputs = torch.load(inputs_path)
    tensor_hashes = {k: wan_real.tensor_sha256(v) for k, v in inputs.items()}
    weights = wan_real.weight_hashes(weights_dir)
    verification = {
        "inputs_file_sha256": file_hash,
        "inputs_file_matches_record": file_hash == recorded["artifacts_sha256"]["wan_real_mps/inputs.pt"],
        "input_tensors_match_record": tensor_hashes == recorded["settings"]["input_sha256"],
        "weights_sha256": weights,
        "weights_match_record": weights == recorded["model"]["weights_sha256"],
    }
    if not all(v for k, v in verification.items() if k.endswith("record")):
        raise RuntimeError(f"real-model inputs or weights do not match {WAN_REAL_JSON}: {verification}")

    transformer = WanTransformer3DModel.from_pretrained(weights_dir / "transformer", torch_dtype=torch.bfloat16).to(device).eval()
    case = Case("real", device, torch.bfloat16, transformer, *(inputs[k].to(device) for k in ("latents", "prompt_embeds", "negative_prompt_embeds")))
    return case, verification


def frame_hashes_match(latents: torch.Tensor, weights_dir: Path, recorded: dict, device: str = "mps") -> dict:
    """Decoded frames against the R1 frame hashes recorded on MPS; judged only on MPS (other devices report it)."""
    vae = AutoencoderKLWan.from_pretrained(weights_dir / "vae", torch_dtype=torch.float32).to(device).eval()
    frames = VideoProcessor(vae_scale_factor=8).postprocess_video(decode_latents(vae, latents.to(device)), output_type="np")[0]
    del vae
    euler.empty_cache(device)
    hashes = [hashlib.sha256(wan_real.png_bytes(frame)).hexdigest() for frame in frames]
    expected = [recorded["artifacts_sha256"][f"wan_real_mps/r1_frames/frame_{i:03d}.png"] for i in range(len(frames))]
    return {"frames": len(frames), "frames_matching_record": sum(a == b for a, b in zip(hashes, expected)), "frames_judged": device == "mps"}


def run_real(weights_dir: Path, device: str = "mps") -> tuple[list[dict], dict]:
    recorded = json.loads(WAN_REAL_JSON.read_text())
    case, verification = load_real_case(weights_dir, recorded, device)
    timings = {}

    start = time.perf_counter()
    dense = dense_trajectory(case)
    timings["dense"] = time.perf_counter() - start

    t1, hsa_full = check_t1(case, dense)
    frames = frame_hashes_match(hsa_full[-1], weights_dir, recorded, device)
    t1.update(frames)
    if frames["frames_judged"]:
        t1["passed"] = t1["passed"] and frames["frames_matching_record"] == frames["frames"]
    rows = [check_token_locality(case), t1, check_t2(case)]

    start = time.perf_counter()
    skipping, reference_seed0 = check_skipping(case)
    timings["skipping_tests"] = time.perf_counter() - start
    rows += check_oracle(case, dense, reference_seed0) + skipping
    return rows, {"verification": verification, "seconds": timings}


# Report


def run_tiny(devices: list[str]) -> tuple[list[dict], list[dict]]:
    rows, diagnostics = [], []
    for device in devices:
        for dtype in (torch.float32, torch.bfloat16):
            case = tiny_case(device, dtype)
            dense = dense_trajectory(case)
            rows += [check_token_locality(case), check_t1(case, dense)[0], check_t2(case)]
            skipping, reference_seed0 = check_skipping(case)
            rows += check_oracle(case, dense, reference_seed0) + skipping
            diagnostics.append(hd12_diagnostic(device, dtype))
        for preset in PRESETS.values():
            rows.append(check_t4(device, preset))
        rows.append(check_t4(device, HSA50, varying_low=True))
    return rows, diagnostics


def run_all(devices: list[str], weights_dir: Path | None, real_device: str = "mps") -> dict:
    rows = check_schedule()
    tiny_rows, diagnostics = run_tiny(devices)
    rows += tiny_rows
    real = None
    if weights_dir is not None:
        real_rows, real = run_real(weights_dir, real_device)
        rows += real_rows
    judged = [r for r in rows if r["passed"] is not None]
    gate = [r for r in rows if r["test"] == "T3 fast path vs oracle" and r.get("model") == "real"]
    gate_ran = len(gate) == len(LABEL_SEEDS)
    gate_passed = gate_ran and all(r["passed"] for r in gate)
    all_judged_passed = all(r["passed"] for r in judged)
    return {
        "milestone": "HSA mechanics on the Wan2.1 Transformer",
        "passed": all_judged_passed and gate_passed,
        "all_judged_passed": all_judged_passed,
        "acceptance_gate": {"test": "real-model T3: fast path vs oracle, bitwise, every label seed", "ran": gate_ran, "passed": gate_passed},
        "num_judged": len(judged),
        "num_failed": sum(not r["passed"] for r in judged),
        "environment": euler.environment(),
        "real_model": real,
        "settings": {
            "shift": SHIFT,
            "guidance": GUIDANCE,
            "steps": STEPS,
            "label_seeds": LABEL_SEEDS,
            "small_resolution_groups": "HSA-50 window and stride with the registered high-group share, rounded "
            "(31 of 192 tokens for the tiny model, 312 of 1,950 for the real model at 416 x 240)",
            "judged_precision": f"tiny model: every test in fp32 and bf16 on {', '.join(devices)}; real model: bf16 on {real_device}",
            "tiny_transformer_config": TINY_CONFIG,
            "tiny_text_len": TINY_TEXT_LEN,
        },
        "fixture_change": FIXTURE_CHANGE,
        "token_mixing_evidence": {"source": f"{DIFFUSERS_WAN} (diffusers {diffusers.__version__})", "operations": TOKEN_MIXING_EVIDENCE},
        "hd12_diagnostic": diagnostics,
        "results": rows,
    }


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return "0" if value == 0 else f"{value:.3g}"
    return str(value)


def _run_summary(run: dict) -> str:
    if run["bitwise_equal"]:
        return "bitwise"
    return f"max abs diff {_fmt(run['max_abs_diff'])} from step {run['first_divergent_step']}"


def markdown_report(report: dict) -> str:
    env = report["environment"]
    gate = report["acceptance_gate"]
    if report["passed"]:
        overall = "PASS"
    elif report["all_judged_passed"] and not gate["ran"]:
        overall = "INCOMPLETE (real-model acceptance gate not run)"
    else:
        overall = "FAIL"
    lines = [
        "# HSA mechanics on the Wan2.1 Transformer",
        "",
        f"**Overall: {overall}** "
        f"({report['num_judged'] - report['num_failed']}/{report['num_judged']} judged checks passed; "
        f"acceptance gate, {gate['test']}: {'PASS' if gate['passed'] else ('FAIL' if gate['ran'] else 'not run')})",
        "",
        f"Code `{env['git_sha']}` (dirty: {env['git_dirty']}); torch {env['torch']}, diffusers {env['diffusers']}, {env['platform']}.",
        f"Shift {SHIFT:g}, guidance {GUIDANCE:g}, {STEPS} steps. {report['settings']['small_resolution_groups']}.",
        "",
        "| Test | Model | Device | Dtype | Case | Bitwise | Max abs diff | First divergence | Result |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in report["results"]:
        case = r.get("preset") or (f"seed {r['seed']}" if "seed" in r else "")
        status = "info" if r["passed"] is None else ("PASS" if r["passed"] else "FAIL")
        bitwise = "" if "bitwise_equal" not in r else ("yes" if r["bitwise_equal"] else "no")
        lines.append(
            f"| {r['test']} | {r.get('model', 'synthetic' if 'T4' in r['test'] else '')} | {r.get('device', '')} | "
            f"{r.get('dtype', '')} | {case} | {bitwise} | {_fmt(r.get('max_abs_diff'))} | "
            f"{_fmt(r.get('first_divergent_step'))} | {status} |"
        )

    lines += ["", "## Registered schedules (N = 32,760)", "", "| Preset | High / low | Window | Stride | Evaluations (high, low) | Token-step fraction | Low-group iterations |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in report["results"]:
        if r["test"].startswith("schedule"):
            lines.append(
                f"| {r['preset']} | {r['group_sizes'][0]:,} / {r['group_sizes'][1]:,} | [{r['window'][0]}, {r['window'][1]}) | "
                f"{r['stride']} | {r['evaluations'][0]}, {r['evaluations'][1]} | {r['token_step_fraction']:.6f} | "
                f"{', '.join(map(str, r['low_group_iterations']))} |"
            )

    lines += ["", "## Why the oracle is exact: token-mixing operations", "", f"Source: `{report['token_mixing_evidence']['source']}`.", "", "| Operation | Location | Mixes video tokens? |", "| --- | --- | --- |"]
    for op, where, why in report["token_mixing_evidence"]["operations"]:
        lines.append(f"| {op} | {where} | {'**yes**: ' if op == 'attn1 attention' else 'no: '}{why} |")
    lines += ["", "T0 checks this empirically: with self-attention made token-local, perturbing one token changes only that token's output."]

    lines += ["", "## Fixture change (disclosed)", "", report["fixture_change"], "", "## Head-dim-12 diagnostic (not judged)", ""]
    for d in report["hd12_diagnostic"]:
        runs = "; ".join(f"seed {r['seed']}: {_run_summary(r)}" for r in d["runs"])
        lines.append(f"- {d['device']}, {d['dtype']}: {runs}.")
        e = d["evidence"]
        if e and "queries" in e:
            lines.append(
                f"  - Seed {d['evidence_seed']}, step {e['step']}: first differing module `{e['module']}` ({e['branch']} branch); "
                f"module inputs equal on active rows: {e['module_inputs_equal']}; q, K, V equal: {e['qkv_equal']}. "
                f"Attention with {e['queries'][0]} queries, then selecting the active rows, vs with {e['queries'][1]} queries: "
                f"{e['sdpa_elements_differing']} of {e['sdpa_elements']} elements differ, max {_fmt(e['sdpa_max_abs_diff'])}; "
                f"same inputs in fp32: max {_fmt(e['sdpa_fp32_max_abs_diff'])}."
            )
        elif e:
            lines.append(f"  - Seed {d['evidence_seed']}, step {e['step']}: first differing module `{e['module']}`.")

    if report["real_model"]:
        v = report["real_model"]["verification"]
        lines += [
            "",
            "## Real-model inputs",
            "",
            f"- `inputs.pt` sha256 `{v['inputs_file_sha256']}`; matches wan_real_mps.json: {v['inputs_file_matches_record']}; "
            f"tensor hashes match: {v['input_tensors_match_record']}.",
            f"- Weights match wan_real_mps.json: {v['weights_match_record']}.",
            "- Seconds: " + ", ".join(f"{k} {s:.0f}" for k, s in report["real_model"]["seconds"].items()),
        ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="HSA mechanics check")
    parser.add_argument("--tiny-only", action="store_true", help="skip the real Wan2.1 Transformer")
    parser.add_argument("--weights", type=Path, default=wan_real.WEIGHTS_DIR)
    parser.add_argument("--real-device", default="mps", help="device for the real Transformer (default: mps)")
    euler.device_args(parser)
    args = parser.parse_args()
    report = run_all(args.devices or euler.default_devices(), None if args.tiny_only else args.weights, args.real_device)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "hsa_mechanics.json").write_text(json.dumps(report, indent=2))
    (args.report_dir / "hsa_mechanics.md").write_text(markdown_report(report))
    print(markdown_report(report))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
