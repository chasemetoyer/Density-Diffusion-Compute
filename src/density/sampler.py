"""Custom FlowMatch Euler sampler.

The schedule and update reproduce diffusers' FlowMatchEulerDiscreteScheduler
with a static shift (no dynamic shifting, no stochastic sampling). Owning the
loop lets the HSA mechanics change which tokens are evaluated at each
iteration without depending on scheduler state.

Precision policy (protocol section 2): latents and velocities stay fp32 between
steps. The Transformer may run in bf16; its outputs are cast to fp32 before
guidance and the Euler update.
"""

from typing import Callable, Optional

import numpy as np
import torch

NUM_TRAIN_TIMESTEPS = 1000

# velocity_fn(latents_fp32, timestep, n) -> fp32 velocity with the latents' shape.
VelocityFn = Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]
# on_step(n, latents_after_step, velocity) is called after each update.
StepCallback = Callable[[int, torch.Tensor, torch.Tensor], None]


def shift_sigmas(sigmas, shift: float):
    """Flow-matching time shift: s * sigma / (1 + (s - 1) * sigma)."""
    return shift * sigmas / (1 + (shift - 1) * sigmas)


def flow_match_schedule(
    num_steps: int,
    shift: float,
    num_train_timesteps: int = NUM_TRAIN_TIMESTEPS,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(sigmas, timesteps)`` for ``num_steps`` Euler steps.

    ``sigmas`` has ``num_steps + 1`` fp32 entries ending at 0. ``timesteps`` has
    ``num_steps`` fp32 entries equal to ``sigmas[:-1] * num_train_timesteps``;
    these are the values passed to the Transformer.

    This follows diffusers' convention, including its precision order, so the
    result is bitwise identical to ``FlowMatchEulerDiscreteScheduler``:

    1. The training grid's endpoints, sigma = 1 and sigma = 1 / T, are shifted
       in fp32.
    2. The inference sigmas are a float64 linspace between those *shifted*
       endpoints, and the shift is then applied a second time.
    3. The result is cast to fp32, and timesteps are computed in fp32.

    Step 2 means the last nonzero sigma is shift(shift(1 / T)), about 0.0602 at
    shift 8, rather than shift(1 / T). Record this schedule with the
    experiment; other Wan samplers space their sigmas differently.
    """
    if num_steps < 1:
        raise ValueError(f"num_steps must be positive, got {num_steps}")

    train_endpoints = torch.tensor([num_train_timesteps, 1], dtype=torch.float32) / num_train_timesteps
    sigma_max, sigma_min = shift_sigmas(train_endpoints, shift).tolist()

    sigmas = np.linspace(sigma_max * num_train_timesteps, sigma_min * num_train_timesteps, num_steps)
    sigmas = shift_sigmas(sigmas / num_train_timesteps, shift)

    sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)
    timesteps = sigmas * num_train_timesteps
    sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])
    return sigmas, timesteps


def guided_velocity(v_uncond: torch.Tensor, v_cond: torch.Tensor, guidance_scale: float) -> torch.Tensor:
    """Protocol equation 2, evaluated in fp32."""
    v_uncond = v_uncond.float()
    v_cond = v_cond.float()
    return v_uncond + guidance_scale * (v_cond - v_uncond)


def euler_sample(
    velocity_fn: VelocityFn,
    latents: torch.Tensor,
    sigmas: torch.Tensor,
    timesteps: torch.Tensor,
    on_step: Optional[StepCallback] = None,
) -> torch.Tensor:
    """Integrate the flow ODE from ``sigmas[0]`` to ``sigmas[-1]`` with Euler steps.

    Iteration ``n`` (zero-based, protocol section 5) evaluates the velocity at
    ``timesteps[n]`` and applies ``x <- x + (sigmas[n + 1] - sigmas[n]) * v``.
    Raises ``FloatingPointError`` on a NaN or infinite velocity.
    """
    if sigmas.shape[0] != timesteps.shape[0] + 1:
        raise ValueError(f"expected len(sigmas) == len(timesteps) + 1, got {sigmas.shape[0]} and {timesteps.shape[0]}")

    sigmas = sigmas.to(latents.device)
    timesteps = timesteps.to(latents.device)
    x = latents.to(torch.float32)

    for n in range(timesteps.shape[0]):
        v = velocity_fn(x, timesteps[n], n)
        if v.dtype != torch.float32:
            raise TypeError(f"velocity_fn must return fp32, got {v.dtype} at iteration {n}")
        if not torch.isfinite(v).all():
            raise FloatingPointError(f"non-finite velocity at iteration {n} (timestep {timesteps[n].item()})")

        dt = sigmas[n + 1] - sigmas[n]
        x = x + dt * v

        if on_step is not None:
            on_step(n, x, v)

    return x
