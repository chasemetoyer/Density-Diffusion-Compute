"""Allocation rules (protocol sections 2 and 3): need score, rankings, group assignment.

Need score (equation 3)
    g_i = 1 / (K - 1) * sum_{k=2..K} ||v_k[i] - v_{k-1}[i]||_1 / max(||v_k[i]||_1, 1e-8)

    over the fp32 guided velocities of the K leading iterations in which every
    token runs, read from the schedule (5, 10, 9 and 12 for the registered
    presets). Protocol k = n + 1, so k = 1..K are code iterations n = 0..K-1.
    Each token's vector is its latent-velocity patch (16 channels x 1 x 2 x 2).

Rankings (equations 4 and 7); each returns token IDs, best first.
    C3  g descending; ties by ascending token ID.
    C4  (1 + lambda * m) * g descending; ties by ascending token ID.
    C5  m descending; ties by a saved random permutation seeded by the prompt ID.
    C6  b = 1[m >= 0.1] first, then g descending, then ascending token ID.
    A1 is C4 with a different mask input.

Assignment fills the fixed HSA group sizes in rank order. It happens once,
after iteration K - 1 (the last in which every token runs) and before the
first iteration that skips a token; the labels are then frozen.
"""

from typing import Callable, Sequence

import numpy as np
import torch

from density.hsa import HIGH, LOW, Preset, Schedule, patchify, preset_iterations

NEED_FLOOR = 1e-8  # equation 3 denominator floor
ROI_THRESHOLD = 0.1  # equation 7: b = 1[m >= 0.1]
LAMBDA_GRID = (1.0, 3.0, 10.0)  # pilot candidates; lambda = 0 is allowed only as a control


# Need score


def leading_full_iterations(group_iterations: torch.Tensor) -> int:
    """The number of leading iterations in which every group, and so every token, runs."""
    everyone = group_iterations.bool().all(dim=0)
    return int(everyone.logical_not().nonzero()[0]) if not everyone.all() else everyone.numel()


def need_window(preset: Preset) -> int:
    """K for a registered preset, read from its schedule."""
    return leading_full_iterations(preset_iterations(preset))


def need_score(velocities: Sequence[torch.Tensor]) -> torch.Tensor:
    """Equation 3 from guided velocities ``v_1 .. v_K``, each token-major (N, P) fp32. Returns g (N,)."""
    K = len(velocities)
    if K < 2:
        raise ValueError(f"the need score needs at least two velocities, got {K}")
    for k, v in enumerate(velocities, start=1):
        if v.dtype != torch.float32 or v.ndim != 2:
            raise TypeError(f"v_{k} must be fp32 with shape (tokens, patch), got {v.dtype} {tuple(v.shape)}")
        if not torch.isfinite(v).all():
            raise FloatingPointError(f"non-finite guided velocity v_{k}")

    total = torch.zeros(velocities[0].shape[0], dtype=torch.float32, device=velocities[0].device)
    for k in range(1, K):  # zero-based index k is protocol k + 1 = 2..K
        change = (velocities[k] - velocities[k - 1]).abs().sum(dim=-1)
        size = velocities[k].abs().sum(dim=-1)
        total = total + change / size.clamp(min=NEED_FLOOR)
    g = total / (K - 1)
    if not torch.isfinite(g).all():
        raise FloatingPointError("non-finite need score")
    return g


class NeedCollector:
    """``on_step`` hook: keeps the guided velocities of iterations 0..K-1, then calls ``on_ready(g)``.

    It must run before iteration K, which is the first one that needs labels.
    """

    def __init__(self, K: int, patch_size, on_ready: Callable[[torch.Tensor], None]):
        self.K = K
        self.patch_size = tuple(patch_size)
        self.on_ready = on_ready
        self.velocities: list[torch.Tensor] = []
        self.g: torch.Tensor | None = None

    def __call__(self, n: int, latents: torch.Tensor, velocity: torch.Tensor) -> None:
        if n >= self.K:
            return
        if velocity.shape[0] != 1:
            raise ValueError("the need score is defined for one video at a time")
        self.velocities.append(patchify(velocity, self.patch_size)[0].clone())
        if n == self.K - 1:
            self.g = need_score(self.velocities)
            self.velocities = []
            self.on_ready(self.g)


# Rankings


def _lexsort(*keys) -> torch.Tensor:
    """Indices sorting ascending by ``keys[0]``, then ``keys[1]``, ...; numpy's lexsort takes the primary key last."""
    return torch.from_numpy(np.lexsort(tuple(np.asarray(k) for k in reversed(keys))))


def _float64(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().double().numpy()


def rank_need(g: torch.Tensor) -> torch.Tensor:
    """C3: g descending, ties by ascending token ID."""
    return _lexsort(-_float64(g), np.arange(g.numel()))


def importance_weights(m: torch.Tensor, lam: float) -> np.ndarray:
    """Equation 4: w = 1 + lambda * m, in float64 (exact for m = k / 256 and the registered lambdas)."""
    if lam < 0:
        raise ValueError(f"lambda must be non-negative, got {lam}")
    return 1.0 + lam * _float64(m)


def rank_importance_need(g: torch.Tensor, m: torch.Tensor, lam: float) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """C4 (and A1): q = w * g descending, ties by ascending token ID. Returns (order, w, q)."""
    w = importance_weights(m, lam)
    q = w * _float64(g)
    return _lexsort(-q, np.arange(g.numel())), w, q


def tie_permutation(prompt_id: int, num_tokens: int) -> torch.Tensor:
    """C5's saved tie-break permutation: depends only on the prompt ID, generated on CPU."""
    return torch.randperm(num_tokens, generator=torch.Generator("cpu").manual_seed(int(prompt_id)))


def rank_coverage(m: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
    """C5: m descending; ties by each token's position in ``permutation``."""
    position = np.empty(m.numel(), dtype=np.int64)
    position[permutation.cpu().numpy()] = np.arange(m.numel())
    return _lexsort(-_float64(m), position)


def roi_binary(m: torch.Tensor) -> torch.Tensor:
    """Equation 7: b = 1[m >= 0.1]."""
    return m >= ROI_THRESHOLD


def rank_binary_then_need(g: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """C6: b = 1 first, then g descending, then ascending token ID."""
    b = roi_binary(m).cpu().numpy()
    return _lexsort(-b.astype(np.int64), -_float64(g), np.arange(g.numel()))


# Group assignment


def group_sizes(preset: Preset, num_tokens: int) -> tuple[int, int]:
    """(high, low) token counts: the registered sizes at full size, the registered share (rounded) otherwise."""
    if num_tokens == preset.num_tokens:
        return preset.high_tokens, preset.low_tokens
    high = round(num_tokens * preset.high_tokens / preset.num_tokens)
    return high, num_tokens - high


def assign_groups(order: torch.Tensor, high_tokens: int) -> torch.Tensor:
    """Fill the HIGH group with the first ``high_tokens`` tokens in rank order; the rest are LOW."""
    labels = torch.full((order.numel(),), LOW, dtype=torch.long)
    labels[order[:high_tokens]] = HIGH
    return labels


class AssignedSchedule:
    """An HSA schedule whose labels are assigned once, before the first iteration that skips a token.

    Provides the interface of ``density.hsa.Schedule`` used by ``HSAVelocity``.
    Before ``assign``, the K leading iterations run every token, which is what
    every group does there. Afterwards it delegates to a ``Schedule`` built
    from the frozen labels, and checks on every use that they have not changed.
    """

    def __init__(self, group_iterations: torch.Tensor, num_tokens: int):
        self.group_iterations = group_iterations.detach().cpu().bool()
        self.num_tokens = num_tokens
        self.num_steps = self.group_iterations.shape[1]
        self.leading_iterations = leading_full_iterations(self.group_iterations)
        self._schedule: Schedule | None = None
        self._frozen: torch.Tensor | None = None

    @property
    def assigned(self) -> bool:
        return self._schedule is not None

    def assign(self, labels: torch.Tensor) -> None:
        if self._schedule is not None:
            raise RuntimeError("labels are already assigned and frozen")
        labels = labels.detach().cpu().long().clone()
        if labels.numel() != self.num_tokens:
            raise ValueError(f"expected {self.num_tokens} labels, got {labels.numel()}")
        self._schedule = Schedule(labels, self.group_iterations)
        self._frozen = labels.clone()

    def _checked(self) -> Schedule:
        if self._schedule is None:
            raise RuntimeError("labels have not been assigned")
        if not torch.equal(self._schedule.labels, self._frozen):
            raise RuntimeError("labels changed after assignment")
        return self._schedule

    def active_ids(self, n: int) -> torch.Tensor:
        if self._schedule is None:
            if n >= self.leading_iterations:
                raise RuntimeError(f"iteration {n} needs labels, which are assigned before iteration {self.leading_iterations}")
            return torch.arange(self.num_tokens)
        return self._checked().active_ids(n)

    @property
    def labels(self) -> torch.Tensor:
        return self._checked().labels.clone()

    @property
    def active(self) -> torch.Tensor:
        return self._checked().active

    def active_counts(self) -> list[int]:
        return self._checked().active_counts()

    def token_step_fraction(self) -> float:
        return self._checked().token_step_fraction()
