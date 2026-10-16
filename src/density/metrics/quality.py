"""The five VBench quality dimensions (protocol section 7), scored with VBench v0.1.5's own per-video functions.

The functions are vendored verbatim in third_party/ (with the reviewed diffs in third_party/patches/); this module only
builds their models from the verified, data-only weight files (density.metrics.weights), as VBench's loaders would:

    subject consistency     DINO ViT-B/16 (hubconf ``dino_vitb16``: vit_base(patch_size=16, num_classes=0), strict load)
    background consistency  CLIP ViT-B/32 (clip.load with jit=False: build_model, then .float() on CPU)
    aesthetic quality       CLIP ViT-L/14 and the LAION linear head
    imaging quality         MUSIQ trained on SPAQ (pyiqa), VBench's default 'longer' preprocessing
    motion smoothness       AMT-S, at native resolution

Each video's MP4 is decoded once (density.video) and handed to every dimension through VBench's own loaders, which read
MP4s with ``density.video.decode_rgb``. Per-video values are converted to the registered 0-100 scale
(density.metrics.registered.QUALITY_SCALE). VBench clamps cosine similarities with ``max(0.0, x)``, which turns a NaN
into 0, so every model's outputs are checked: a non-finite value stops the job instead.
"""

import math
import sys
from contextlib import ExitStack, contextmanager
from pathlib import Path

import numpy as np
import torch

from density import video
from density.metrics.precision import scoring
from density.metrics.registered import QUALITY_DIMENSIONS, QUALITY_SCALE
from density.metrics.weights import verified_path

ROOT = Path(__file__).resolve().parents[3]
THIRD_PARTY = ROOT / "third_party"
IMPORT_PATHS = [THIRD_PARTY / name for name in ("vbench", "clip", "dino", "pyiqa")]
AMT_CONFIG = THIRD_PARTY / "vbench" / "vbench" / "third_party" / "amt" / "cfgs" / "AMT-S.yaml"


def import_paths() -> None:
    """Put the vendored packages first on sys.path: ``vbench``, ``clip``, ``dino_vit`` and ``pyiqa``."""
    for path in reversed(IMPORT_PATHS):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


class NonFinite(FloatingPointError):
    """A quality model produced NaN or infinity."""


def _check_finite(name: str):
    def hook(module, inputs, output):
        values = output.values() if isinstance(output, dict) else output if isinstance(output, (tuple, list)) else (output,)
        for value in values:
            if isinstance(value, torch.Tensor) and value.is_floating_point() and not torch.isfinite(value).all():
                raise NonFinite(f"{name} produced a non-finite value")

    return hook


def build_dino(device):
    from dino_vit.vision_transformer import vit_base

    model = vit_base(patch_size=16, num_classes=0)
    model.load_state_dict(torch.load(verified_path("dino_vitb16"), map_location="cpu", weights_only=True), strict=True)
    return model.to(device).eval()


def build_clip(name: str, device):
    """clip.load(name, device, jit=False): build_model on the archive's state dict (fp16 weights), then fp32 on CPU."""
    from clip.model import build_model
    from safetensors.torch import load_file

    model = build_model(load_file(str(verified_path(name)))).to(device)
    if torch.device(device).type == "cpu":
        model.float()
    return model.eval()


def build_aesthetic_head(device):
    head = torch.nn.Linear(768, 1)
    head.load_state_dict(torch.load(verified_path("laion_aesthetic_head"), map_location="cpu", weights_only=True))
    return head.eval().to(device)


def build_musiq(device):
    """MUSIQ with VBench's constructor defaults (one output, the SPAQ head), loaded strictly as pyiqa does."""
    from pyiqa.archs.arch_util import clean_state_dict
    from pyiqa.archs.musiq_arch import MUSIQ

    model = MUSIQ(pretrained=False)  # no download; the same architecture as MUSIQ(pretrained_model_path=...)
    model.load_state_dict(clean_state_dict(torch.load(verified_path("musiq_spaq"), map_location="cpu", weights_only=True)), strict=True)
    model.to(device)
    model.eval()  # VBench sets model.training = False; every dropout in MUSIQ has p = 0, so the outputs are the same
    return model


AMT_DIR = THIRD_PARTY / "vbench" / "vbench" / "third_party" / "amt"


def drop_amt_path() -> None:
    """AMT's build_utils appends its own folder to sys.path when imported. Our import of AMT-S uses full module names and
    never needs it, and left there it makes a later ``import utils`` find AMT's utils package instead of Hands23's."""
    sys.path[:] = [p for p in sys.path if Path(p).resolve() != AMT_DIR.resolve()]


def build_motion(device):
    from vbench.motion_smoothness import MotionSmoothness

    drop_amt_path()
    return MotionSmoothness(str(AMT_CONFIG), str(verified_path("amt_s")), torch.device(device))


def motion_scale(motion, height: int, width: int) -> float:
    """The resize factor VBench's motion_score computes for a frame size (1 means native resolution)."""
    scale = motion.anchor_resolution / (height * width) * np.sqrt((motion.vram_avail - motion.anchor_memory_bias) / motion.anchor_memory)
    scale = 1 if scale > 1 else scale
    return 1 / np.floor(1 / np.sqrt(scale) * 16) * 16


@contextmanager
def frames_for(path: Path, frames: np.ndarray):
    """VBench's loaders call density.video.decode_rgb(path); inside this context they get ``frames`` instead of a second
    decode, and any other path is an error."""
    original = video.decode_rgb

    def decode(p):
        if Path(p) != Path(path):
            raise RuntimeError(f"VBench asked for {p} while scoring {path}")
        return frames.copy()

    video.decode_rgb = decode
    try:
        yield
    finally:
        video.decode_rgb = original


class QualityModels:
    """Every quality model, built once per session on ``device``, with a finiteness check on each model's outputs."""

    def __init__(self, device="cpu"):
        import_paths()
        self.device = torch.device(device)
        with scoring(self.device):
            self.dino = build_dino(self.device)
            self.clip_b32 = build_clip("clip_vit_b32", self.device)
            self.clip_l14 = build_clip("clip_vit_l14", self.device)
            self.aesthetic = build_aesthetic_head(self.device)
            self.musiq = build_musiq(self.device)
            self.motion = build_motion(self.device)
        for name, module in (("DINO", self.dino), ("CLIP ViT-B/32", self.clip_b32.visual), ("CLIP ViT-L/14", self.clip_l14.visual),
                             ("LAION aesthetic head", self.aesthetic), ("MUSIQ", self.musiq), ("AMT-S", self.motion.model)):
            module.register_forward_hook(_check_finite(name))

    def raw(self, path: Path, frames: np.ndarray | None = None) -> dict[str, float]:
        """VBench's per-video values for the MP4 at ``path`` (decoded here unless ``frames`` is given)."""
        from vbench.aesthetic_quality import laion_aesthetic
        from vbench.background_consistency import background_consistency
        from vbench.imaging_quality import technical_quality
        from vbench.subject_consistency import subject_consistency

        path = str(path)
        frames = video.decode_rgb(path) if frames is None else frames
        with ExitStack() as stack:
            stack.enter_context(frames_for(path, frames))
            stack.enter_context(scoring(self.device))
            out = {
                "subject_consistency": subject_consistency(self.dino, [path], self.device, False)[1][0]["video_results"],
                "background_consistency": background_consistency(self.clip_b32, None, [path], self.device, False)[1][0]["video_results"],
                "motion_smoothness": self.motion.motion_score(path),
                "aesthetic_quality": laion_aesthetic(self.aesthetic, self.clip_l14, [path], self.device)[1][0]["video_results"],
                "imaging_quality": technical_quality(self.musiq, [path], self.device)[1][0]["video_results"],
            }
        out = {k: float(v) for k, v in out.items()}
        bad = {k: v for k, v in out.items() if not math.isfinite(v)}
        if bad:
            raise NonFinite(f"{path}: non-finite quality values {bad}")
        return out

    def score(self, path: Path, frames: np.ndarray | None = None) -> dict:
        raw = self.raw(path, frames)
        return {"raw": raw, "scores": to_scale(raw)}


def to_scale(raw: dict[str, float]) -> dict[str, float]:
    """The registered 0-100 scale: x100 for the four dimensions VBench reports in [0, 1], MUSIQ's per-video value as is."""
    if set(raw) != set(QUALITY_DIMENSIONS):
        raise ValueError(f"expected the five dimensions {QUALITY_DIMENSIONS}, got {sorted(raw)}")
    return {d: raw[d] * QUALITY_SCALE[d] for d in QUALITY_DIMENSIONS}
