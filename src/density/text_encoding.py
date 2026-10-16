"""Text embeddings for Wan2.1 with the registered T5 settings.

Registered (2026-10-06, smoke test): the UMT5 text encoder runs in bf16 with
512 tokens, as Wan2.1's official config sets ``t5_dtype`` and ``text_len``
(``wan/configs/shared_config.py`` at commit 9737cba). The negative prompt is
that config's ``sample_neg_prompt``. ``ftfy`` is absent from the environment,
so Diffusers' prompt cleaning skips ``ftfy.fix_text`` and applies only its
HTML unescaping and whitespace cleanup; encoding refuses to run if ftfy is
installed, because that would change the registered cleaning.

Embeddings are computed with Diffusers' own ``WanPipeline._get_t5_prompt_embeds``,
which trims each prompt to its token count and pads with zeros to 512.
"""

import hashlib
from types import SimpleNamespace

import torch

T5_DTYPE = torch.bfloat16
MAX_SEQUENCE_LENGTH = 512
FTFY_REGISTERED = "absent"
WAN_CONFIG_SOURCE = {
    "repo": "https://github.com/Wan-Video/Wan2.1",
    "commit": "9737cba9c1c3c4d04b33fcad41c111989865d315",
    "file": "wan/configs/shared_config.py",
}
NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，"
    "残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，"
    "三条腿，背景人很多，倒着走"
)


def check_ftfy_absent() -> None:
    from diffusers.pipelines.wan import pipeline_wan

    if pipeline_wan.is_ftfy_available():
        raise RuntimeError("ftfy is installed; the registered prompt cleaning assumes it is absent")


def tensor_sha256(tensor: torch.Tensor) -> str:
    t = tensor.detach().cpu().contiguous()
    data = t.view(torch.uint16) if t.dtype == torch.bfloat16 else t
    return hashlib.sha256(data.numpy().tobytes()).hexdigest()


@torch.no_grad()
def encode(texts: dict[str, str], tokenizer, text_encoder, device: str) -> dict[str, torch.Tensor]:
    """Embeddings of shape (1, 512, 4096) in bf16 for each named text, through Diffusers' Wan encoding."""
    from diffusers import WanPipeline

    check_ftfy_absent()
    host = SimpleNamespace(tokenizer=tokenizer, text_encoder=text_encoder, _execution_device=torch.device(device))
    out = {}
    for name, text in texts.items():
        out[name] = WanPipeline._get_t5_prompt_embeds(host, prompt=text, max_sequence_length=MAX_SEQUENCE_LENGTH, device=torch.device(device), dtype=T5_DTYPE).cpu()
    return out
