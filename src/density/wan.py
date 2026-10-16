"""Wan2.1 Transformer as a guided-velocity function for the custom sampler, and the VAE decode.

Registered (2026-10-06, smoke test): the fp32 VAE decodes with TF32 convolutions, cuDNN's default on CUDA, set
explicitly in ``decode_latents``. On the H100 at full resolution, strict (IEEE) fp32 convolutions took 16.4 s per
decode against 3.3 s, adding 13.1 s per video: 18 percent of a dense C1's end-to-end time and 33 percent of an
HSA-50 run's with the registered flash attention (73.8 s and 39.4 s, smoke test Session D). They peaked at 41 GiB
against 8.7 GiB, and the decoded frames differed by 81 to 86 dB PSNR. That exceeds the 10 percent limit set for
choosing strict fp32. CPU and MPS are unaffected.
"""

import torch

from density.attention import generation_attention
from density.sampler import guided_velocity

VAE_CONV_FP32_PRECISION = "tf32"  # registered; see the module docstring


class WanGuidedVelocity:
    """Dense guided velocity (protocol equation 2) from a ``WanTransformer3DModel``.

    Both guidance branches see the same latents and timestep. Each branch's
    prediction is cast to fp32 before guidance, so the result is fp32 even when
    the Transformer runs in bf16.
    """

    def __init__(self, transformer, prompt_embeds: torch.Tensor, negative_prompt_embeds: torch.Tensor, guidance_scale: float):
        self.transformer = transformer
        self.dtype = transformer.dtype
        self.prompt_embeds = prompt_embeds.to(self.dtype)
        self.negative_prompt_embeds = negative_prompt_embeds.to(self.dtype)
        self.guidance_scale = guidance_scale

    def _predict(self, hidden_states: torch.Tensor, timestep: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )[0]

    @torch.no_grad()
    def __call__(self, latents: torch.Tensor, timestep: torch.Tensor, n: int) -> torch.Tensor:
        hidden_states = latents.to(self.dtype)
        timestep = timestep.expand(latents.shape[0])
        with generation_attention(latents.device):  # registered: flash attention on CUDA
            v_cond = self._predict(hidden_states, timestep, self.prompt_embeds)
            v_uncond = self._predict(hidden_states, timestep, self.negative_prompt_embeds)
        return guided_velocity(v_uncond, v_cond, self.guidance_scale)


@torch.no_grad()
def decode_latents(vae, latents: torch.Tensor, conv_fp32_precision: str = VAE_CONV_FP32_PRECISION) -> torch.Tensor:
    """Decode Wan latents to video in [-1, 1] with shape (B, 3, frames, H, W).

    Mirrors WanPipeline's output step, including its operation order. The VAE
    config stores per-channel mean and std; latents are de-normalized as
    ``latents * std + mean`` (written as ``latents / (1 / std) + mean``).
    cuDNN's fp32 convolutions use the registered precision (``"ieee"`` only for
    the smoke test's comparison), and attention the registered backends.
    """
    latents = latents.to(vae.dtype)
    shape = (1, vae.config.z_dim, 1, 1, 1)
    latents_mean = torch.tensor(vae.config.latents_mean).view(shape).to(latents.device, latents.dtype)
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(shape).to(latents.device, latents.dtype)
    latents = latents / latents_std + latents_mean
    previous = torch.backends.cudnn.conv.fp32_precision
    torch.backends.cudnn.conv.fp32_precision = conv_fp32_precision
    try:
        with generation_attention(latents.device):
            return vae.decode(latents, return_dict=False)[0]
    finally:
        torch.backends.cudnn.conv.fp32_precision = previous
