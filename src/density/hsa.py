"""HSA compute mechanics on the Wan2.1 Transformer (protocol sections 2, 5, 8, 11).

Schedule
    Protocol equation 13 decides at which iterations each group runs. The
    group label of every token is an input; how labels are chosen (need score,
    C3-C6 ranking) is a later milestone.

Active-token forward
    Only the active tokens pass through the Transformer blocks. Self-attention
    reads K and V for every token from a per-layer, per-branch cache, after the
    active tokens have refreshed their own rows. Cross-attention and the FFN see
    the active rows only.

Cached velocity
    Each token keeps its last guided velocity in fp32. Used as ``velocity_fn``
    in ``density.sampler.euler_sample``, every token advances by
    ``(sigma[n + 1] - sigma[n]) * v`` with a fresh ``v`` for active tokens and
    the cached ``v`` for skipped ones.

Restricting everything except self-attention to the active rows is exact only
because self-attention is the sole operation in ``WanTransformer3DModel`` that
mixes video tokens; ``density.checks.hsa`` records the evidence.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable

import torch
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_wan import WanAttnProcessor, _get_qkv_projections

from density.attention import generation_attention
from density.sampler import guided_velocity

NUM_STEPS = 40
HIGH, LOW = 0, 1  # group labels: HIGH runs every iteration, LOW follows equation 13


# Schedule


@dataclass(frozen=True)
class Preset:
    """A registered HSA schedule (protocol section 5), defined at the full token count."""

    name: str
    high_tokens: int
    low_tokens: int
    window: tuple[int, int]  # reduced-compute window [start, end), half-open
    low_steps: int  # nominal low-group step count s; its stride is 40 / s

    @property
    def num_tokens(self) -> int:
        return self.high_tokens + self.low_tokens


PRESETS = {
    preset.name: preset
    for preset in [
        Preset("HSA-50", 5242, 27518, (4, 36), 10),
        Preset("HSA-75B", 10811, 21949, (10, 30), 10),
        Preset("switch-20", 1560, 31200, (8, 36), 10),
        Preset("switch-30", 1560, 31200, (12, 36), 5),
    ]
}


def active_iterations(window: tuple[int, int], low_steps: int, num_steps: int = NUM_STEPS) -> torch.Tensor:
    """Protocol equation 13: active(n, s) = [n not in [start, end)] or [n mod (40 / s) == 0]."""
    if num_steps % low_steps:
        raise ValueError(f"{num_steps} steps are not divisible by {low_steps} nominal steps")
    n = torch.arange(num_steps)
    start, end = window
    return (n < start) | (n >= end) | (n % (num_steps // low_steps) == 0)


def preset_iterations(preset: Preset, num_steps: int = NUM_STEPS) -> torch.Tensor:
    """bool [2, num_steps]: the HIGH group runs every iteration, the LOW group per equation 13."""
    return torch.stack([torch.ones(num_steps, dtype=torch.bool), active_iterations(preset.window, preset.low_steps, num_steps)])


class Schedule:
    """Which tokens run at each iteration, given one group label per token.

    ``labels[i]`` is token ``i``'s group; ``group_iterations[g, n]`` says whether
    group ``g`` runs at zero-based iteration ``n``. Token IDs follow the
    Transformer's patch order: frame, then row, then column.
    """

    def __init__(self, labels: torch.Tensor, group_iterations: torch.Tensor):
        self.labels = labels.detach().cpu().long()
        self.group_iterations = group_iterations.detach().cpu().bool()
        self.num_tokens = self.labels.numel()
        self.num_steps = self.group_iterations.shape[1]
        if self.labels.min() < 0 or self.labels.max() >= self.group_iterations.shape[0]:
            raise ValueError("labels must index rows of group_iterations")
        self.active = self.group_iterations[self.labels].T.contiguous()  # bool [num_steps, num_tokens]
        if not self.active[0].all():
            raise ValueError("every token must run at iteration 0; otherwise caches are read before they are written")

    @classmethod
    def from_preset(cls, preset: Preset, labels: torch.Tensor) -> "Schedule":
        """The registered preset; the label counts must equal its registered group sizes."""
        counts = torch.bincount(labels.cpu().long(), minlength=2).tolist()
        if counts != [preset.high_tokens, preset.low_tokens]:
            raise ValueError(f"{preset.name} needs {preset.high_tokens} high / {preset.low_tokens} low tokens, got {counts}")
        return cls(labels, preset_iterations(preset))

    def active_ids(self, n: int) -> torch.Tensor:
        """Active token IDs at iteration ``n``, ascending."""
        return self.active[n].nonzero().squeeze(1)

    def active_counts(self) -> list[int]:
        return self.active.sum(dim=1).tolist()

    def group_evaluations(self) -> list[int]:
        return self.group_iterations.sum(dim=1).tolist()

    def token_step_fraction(self) -> float:
        """Protocol equation 22: sum over iterations of |A_n|, divided by 40 N."""
        return self.active.sum().item() / self.active.numel()


# Token layout


def patchify(latents: torch.Tensor, patch_size: tuple[int, int, int]) -> torch.Tensor:
    """(B, C, F, H, W) latents -> (B, tokens, pt * ph * pw * C), in the Transformer's token order."""
    batch, channels, frames, height, width = latents.shape
    pt, ph, pw = patch_size
    x = latents.reshape(batch, channels, frames // pt, pt, height // ph, ph, width // pw, pw)
    return x.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(batch, -1, pt * ph * pw * channels)


def unpatchify(tokens: torch.Tensor, latent_shape: torch.Size, patch_size: tuple[int, int, int]) -> torch.Tensor:
    """Inverse of ``patchify``; the same reshape and permute as WanTransformer3DModel's output step."""
    batch, _, frames, height, width = latent_shape
    pt, ph, pw = patch_size
    x = tokens.reshape(batch, frames // pt, height // ph, width // pw, pt, ph, pw, -1)
    return x.permute(0, 7, 1, 4, 2, 5, 3, 6).flatten(6, 7).flatten(4, 5).flatten(2, 3)


# Cached velocity and the Euler step

# (latents fp32, timestep, iteration n, active IDs on CPU) -> fp32 velocity of the active tokens, (B, A, P)
ActiveVelocityFn = Callable[[torch.Tensor, torch.Tensor, int, torch.Tensor], torch.Tensor]


class HSAVelocity:
    """``velocity_fn`` for ``euler_sample``: fresh velocity for active tokens, cached velocity for the rest.

    The cache holds each token's last guided velocity in fp32, token-major.
    """

    def __init__(self, active_velocity_fn: ActiveVelocityFn, schedule: Schedule, patch_size: tuple[int, int, int]):
        self.active_velocity_fn = active_velocity_fn
        self.schedule = schedule
        self.patch_size = tuple(patch_size)
        self.cache: torch.Tensor | None = None
        self.written = torch.zeros(schedule.num_tokens, dtype=torch.bool)
        self.active_counts: list[int] = []

    def __call__(self, latents: torch.Tensor, timestep: torch.Tensor, n: int) -> torch.Tensor:
        ids = self.schedule.active_ids(n)
        fresh = self.active_velocity_fn(latents, timestep, n, ids)
        if fresh.dtype != torch.float32 or fresh.shape[1] != ids.numel():
            raise ValueError(f"expected fp32 velocity for {ids.numel()} tokens, got {fresh.dtype} {tuple(fresh.shape)}")

        if self.cache is None:
            self.cache = torch.empty(fresh.shape[0], self.schedule.num_tokens, fresh.shape[2], dtype=torch.float32, device=fresh.device)
        self.cache[:, ids.to(fresh.device)] = fresh
        self.written[ids] = True
        if not self.written.all():
            raise RuntimeError(f"cached velocity read before it was written at iteration {n}")

        self.active_counts.append(ids.numel())
        return unpatchify(self.cache, latents.shape, self.patch_size)

    def token_step_fraction(self) -> float:
        return sum(self.active_counts) / (len(self.active_counts) * self.schedule.num_tokens)


# Wan2.1 active-token forward


class KVCache:
    """K (after q/k norm and RoPE) and V for every token, for one layer and one guidance branch."""

    def __init__(self, num_tokens: int):
        self.num_tokens = num_tokens
        self.key: torch.Tensor | None = None
        self.value: torch.Tensor | None = None
        self.written = torch.zeros(num_tokens, dtype=torch.bool)

    def write(self, ids: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> None:
        """Store rows ``ids`` (CPU) from ``key`` and ``value`` of shape (B, len(ids), heads, head_dim)."""
        if self.key is None:
            shape = (key.shape[0], self.num_tokens, *key.shape[2:])
            self.key = torch.empty(shape, dtype=key.dtype, device=key.device)
            self.value = torch.empty(shape, dtype=value.dtype, device=value.device)
        rows = ids.to(key.device)
        self.key[:, rows] = key
        self.value[:, rows] = value
        self.written[ids] = True

    def read(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.key is None or not self.written.all():
            raise RuntimeError("KV cache read before every row was written")
        return self.key, self.value


def _apply_rotary_emb(hidden_states: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
    # Verbatim from WanAttnProcessor.__call__ (diffusers 0.40.0), where it is a nested function.
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    out = torch.empty_like(hidden_states)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.type_as(hidden_states)


def project_qkv(attn, hidden_states: torch.Tensor, rotary_emb: tuple[torch.Tensor, torch.Tensor]):
    """Self-attention q, k, v as WanAttnProcessor computes them: projection, q/k norm, heads, RoPE."""
    if attn.add_k_proj is not None:
        raise NotImplementedError("image-conditioned attention is not supported")
    query, key, value = _get_qkv_projections(attn, hidden_states, None)
    query = attn.norm_q(query)
    key = attn.norm_k(key)
    query = query.unflatten(2, (attn.heads, -1))
    key = key.unflatten(2, (attn.heads, -1))
    value = value.unflatten(2, (attn.heads, -1))
    query = _apply_rotary_emb(query, *rotary_emb)
    key = _apply_rotary_emb(key, *rotary_emb)
    return query, key, value


def attend(attn, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Attention and output projection as WanAttnProcessor computes them (native backend)."""
    out = dispatch_attention_fn(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, backend=None, parallel_config=None)
    out = out.flatten(2, 3).type_as(query)
    out = attn.to_out[0](out)
    return attn.to_out[1](out)


@contextmanager
def self_attention_processors(transformer, make_processor: Callable[[int], object]):
    """Temporarily replace every block's self-attention processor; the originals are always restored."""
    original = [block.attn1.processor for block in transformer.blocks]
    for processor in original:
        if type(processor) is not WanAttnProcessor:
            raise RuntimeError(f"expected stock WanAttnProcessor, found {type(processor).__name__}")
        if processor._attention_backend is not None or processor._parallel_config is not None:
            raise RuntimeError("only the default native attention backend without context parallelism is supported")
    try:
        for layer, block in enumerate(transformer.blocks):
            block.attn1.set_processor(make_processor(layer))
        yield
    finally:
        for block, processor in zip(transformer.blocks, original):
            block.attn1.set_processor(processor)


class _CachedSelfAttention:
    """attn1 processor for the active-token forward."""

    def __init__(self, hsa: "WanHSA", layer: int):
        self.hsa = hsa
        self.layer = layer

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, rotary_emb=None):
        if encoder_hidden_states is not None or attention_mask is not None:
            raise ValueError("cached self-attention takes no encoder states or mask")
        return self.hsa._self_attention(attn, self.layer, hidden_states, rotary_emb)


class WanHSA:
    """Active-token forward of a diffusers ``WanTransformer3DModel`` (Wan2.1 text-to-video).

    Reuses the model's patch embedding, condition embedder, RoPE, blocks and
    output head without modifying them. Only each block's self-attention
    processor is swapped, for the duration of a forward, for one that keeps a
    KV cache per layer and guidance branch.
    """

    def __init__(self, transformer):
        self.transformer = transformer
        self.dtype = transformer.dtype
        self.patch_size = tuple(transformer.config.patch_size)
        self.num_tokens: int | None = None
        self.caches: dict[str, list[KVCache]] = {}
        self.log: list[tuple[int, int, str, int]] = []  # (iteration, layer, branch, active tokens)
        self.logged_ids: dict[tuple[int, str], torch.Tensor] = {}
        self._context: tuple[int, str, torch.Tensor] | None = None

    # Overridable steps; the negative controls in density.checks.hsa replace them.

    def _rotary(self, rotary_emb, ids: torch.Tensor):
        rows = ids.to(rotary_emb[0].device)
        return tuple(freqs[:, rows] for freqs in rotary_emb)

    def _cache(self, branch: str, layer: int) -> KVCache:
        if branch not in self.caches:
            self.caches[branch] = [KVCache(self.num_tokens) for _ in self.transformer.blocks]
        return self.caches[branch][layer]

    def _write(self, cache: KVCache, ids: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> None:
        cache.write(ids, key, value)

    def _self_attention(self, attn, layer: int, hidden_states, rotary_emb):
        n, branch, ids = self._context
        if hidden_states.shape[1] != ids.numel():
            raise RuntimeError(f"layer {layer} received {hidden_states.shape[1]} tokens, schedule has {ids.numel()}")
        self.log.append((n, layer, branch, hidden_states.shape[1]))

        query, key, value = project_qkv(attn, hidden_states, rotary_emb)
        cache = self._cache(branch, layer)
        self._write(cache, ids, key, value)
        cached_key, cached_value = cache.read()
        return attend(attn, query, cached_key, cached_value)

    @torch.no_grad()
    def forward(self, latents: torch.Tensor, timestep: torch.Tensor, encoder_hidden_states: torch.Tensor, ids: torch.Tensor, branch: str, n: int):
        """Transformer output for active tokens ``ids`` (CPU, ascending), token-major (B, len(ids), P).

        Mirrors WanTransformer3DModel.forward (diffusers 0.40.0) for text-to-video,
        with the token rows gathered after patch embedding.
        """
        tr = self.transformer
        hidden_states = latents.to(self.dtype)
        batch = hidden_states.shape[0]

        rotary_emb = self._rotary(tr.rope(hidden_states), ids)
        tokens = tr.patch_embedding(hidden_states).flatten(2).transpose(1, 2).contiguous()
        if self.num_tokens is None:
            self.num_tokens = tokens.shape[1]
        elif tokens.shape[1] != self.num_tokens:
            raise ValueError(f"token count changed from {self.num_tokens} to {tokens.shape[1]}")
        tokens = tokens[:, ids.to(tokens.device)]

        temb, timestep_proj, text, _ = tr.condition_embedder(
            timestep.expand(batch), encoder_hidden_states.to(self.dtype), None, timestep_seq_len=None
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        self._context = (n, branch, ids)
        self.logged_ids[(n, branch)] = ids
        try:
            with self_attention_processors(tr, lambda layer: _CachedSelfAttention(self, layer)):
                for block in tr.blocks:
                    tokens = block(tokens, text, timestep_proj, rotary_emb)
        finally:
            self._context = None

        shift, scale = (tr.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)
        tokens = (tr.norm_out(tokens.float()) * (1 + scale) + shift).type_as(tokens)
        return tr.proj_out(tokens)


class WanHSAGuidedVelocity:
    """Active-token guided velocity (protocol equation 2); an ``ActiveVelocityFn`` for ``HSAVelocity``.

    Both branches use the same active IDs and separate KV caches.
    """

    def __init__(self, transformer, prompt_embeds, negative_prompt_embeds, guidance_scale: float, hsa_cls=WanHSA):
        self.hsa = hsa_cls(transformer)
        self.prompt_embeds = prompt_embeds
        self.negative_prompt_embeds = negative_prompt_embeds
        self.guidance_scale = guidance_scale

    def __call__(self, latents, timestep, n, ids):
        with generation_attention(latents.device):  # registered: flash attention on CUDA
            v_cond = self.hsa.forward(latents, timestep, self.prompt_embeds, ids, "cond", n)
            v_uncond = self.hsa.forward(latents, timestep, self.negative_prompt_embeds, ids, "uncond", n)
        return guided_velocity(v_uncond, v_cond, self.guidance_scale)


def wan_hsa_velocity(transformer, prompt_embeds, negative_prompt_embeds, guidance_scale, schedule, hsa_cls=WanHSA) -> HSAVelocity:
    """HSA ``velocity_fn`` for ``euler_sample`` driven by a Wan2.1 Transformer."""
    active = WanHSAGuidedVelocity(transformer, prompt_embeds, negative_prompt_embeds, guidance_scale, hsa_cls)
    return HSAVelocity(active, schedule, transformer.config.patch_size)
