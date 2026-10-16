"""Starting noise for every generation (protocol section 6).

Registered (2026-10-06, smoke test): each seed's starting noise is generated once, on CPU, with a
seeded ``torch.Generator``, then saved and identified by the SHA-256 of its tensor bytes. Every run
loads it from the saved file and checks that hash. Noise is never regenerated: the same seed gives
different noise on different CPUs (the smoke test found x86 on the pods and ARM on the Mac differ).
"""

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from density.conditions import sha256

KEY = "latents"


def generate(seed: int, shape) -> torch.Tensor:
    """The one-time generation: CPU, float32, ``torch.Generator('cpu').manual_seed(seed)``."""
    return torch.randn(shape, generator=torch.Generator("cpu").manual_seed(seed), dtype=torch.float32)


def save(noise: torch.Tensor, path: Path) -> str:
    """Write the noise to ``path`` (safetensors) and return its tensor hash, the noise's identity."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file({KEY: noise.contiguous()}, path)
    return sha256(noise)


def load(path: Path, expected_sha256: str) -> torch.Tensor:
    """The saved noise, after checking it is the noise identified by ``expected_sha256``."""
    noise = load_file(path)[KEY]
    got = sha256(noise)
    if got != expected_sha256:
        raise RuntimeError(f"{path}: starting noise hash {got} is not the registered {expected_sha256}")
    return noise
