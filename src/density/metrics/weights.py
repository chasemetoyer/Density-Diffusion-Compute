"""Metric weight files: sources, licenses and SHA-256, verified before anything loads them.

Files live under ``weights/`` (not versioned): the metric models under ``weights/metrics/``, and ArcFace beside the
SCRFD file it was unpacked with, under ``weights/masks/insightface/``. Nothing downloads on first use; a missing or
altered file is an error. Every file a scorer loads is data-only: safetensors, ONNX, or a plain state dict read with
``torch.load(weights_only=True)``. The CLIP TorchScript archives and the AMT-S pickle are kept as the verified
originals that ``scripts/convert_metric_weights.py`` converted, with exactness checks; scorers load only the
converted files. Approved by Chase on 2026-10-07.
"""

from dataclasses import dataclass
from pathlib import Path

from density.masks.weights import sha256_file

ROOT = Path(__file__).resolve().parents[3] / "weights"


@dataclass(frozen=True)
class WeightFile:
    name: str
    path: str  # relative to ROOT
    source: str
    size: int
    sha256: str
    sha256_source: str
    license: str
    loads: bool = True  # False: an original kept only as the source of a converted file


WEIGHTS = {
    w.name: w
    for w in (
        WeightFile(
            "lpips_alex_v0.1", "metrics/lpips/alex_v0.1.pth",
            "lpips 0.1.4 wheel, lpips/weights/v0.1/alex.pth (github.com/richzhang/PerceptualSimilarity tag v0.1.4)",
            6009, "df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0",
            "the wheel's RECORD file (official)", "BSD-2-Clause",
        ),
        WeightFile(
            "alexnet", "metrics/lpips/alexnet-owt-7be5be79.pth",
            "download.pytorch.org/models/alexnet-owt-7be5be79.pth (torchvision AlexNet_Weights.IMAGENET1K_V1)",
            244408911, "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02",
            "official 8-hex prefix in the file name; the full hash matches a Hugging Face mirror's LFS record",
            "no terms stated by torchvision; ImageNet-derived",
        ),
        WeightFile(
            "arcface_w600k_r50", "masks/insightface/models/buffalo_l/w600k_r50.onnx",
            "github.com/deepinsight/insightface release model-zoo, buffalo_l.zip "
            "(sha256 80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f, official release digest)",
            174383860, "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
            "computed from the verified zip; matches four third-party mirrors",
            "InsightFace pretrained models: non-commercial research only",
        ),
        WeightFile(
            "dino_vitb16", "metrics/dino/dino_vitbase16_pretrain.pth",
            "dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth",
            343242485, "bf34ad0f424b9029b593e8dc3ed553bf26e88bcba0d32bf3e62a6209cb64c85e",
            "trust on first use: recorded at download (Meta publishes none; the size matches the server's)", "Apache-2.0",
        ),
        WeightFile(
            "clip_vit_b32_original", "metrics/clip/ViT-B-32.pt",
            "openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
            353976522, "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af",
            "official (in the download URL)", "MIT (OpenAI CLIP)", loads=False,
        ),
        WeightFile(
            "clip_vit_b32", "metrics/clip/ViT-B-32.safetensors",
            "converted from clip_vit_b32_original by scripts/convert_metric_weights.py: the TorchScript archive's state dict as stored",
            353470932, "4cada1a09d8b61a98b0a852743315dc84efde62e55b8c5da173d5937157fed19",
            "computed at conversion; every tensor bitwise equal, and build_model gives identical image features", "as clip_vit_b32_original",
        ),
        WeightFile(
            "clip_vit_l14_original", "metrics/clip/ViT-L-14.pt",
            "openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
            932768134, "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836",
            "official (in the download URL)", "MIT (OpenAI CLIP)", loads=False,
        ),
        WeightFile(
            "clip_vit_l14", "metrics/clip/ViT-L-14.safetensors",
            "converted from clip_vit_l14_original by scripts/convert_metric_weights.py: the TorchScript archive's state dict as stored",
            932103532, "a8e9ea88b09f37d8fc6c225e6d6f613dcaff94172ce387e7ca3676c3ddd2831f",
            "computed at conversion; every tensor bitwise equal, and build_model gives identical image features", "as clip_vit_l14_original",
        ),
        WeightFile(
            "laion_aesthetic_head", "metrics/aesthetic/sa_0_4_vit_l_14_linear.pth",
            "github.com/LAION-AI/aesthetic-predictor, sa_0_4_vit_l_14_linear.pth (git blob d4135b63efbaac587bcb1cdfa02ac082071a5e40)",
            4071, "2cd4e60f4f24ae3bcd57b847b13c1f3ba27edc28cc1a7f9ce74ee9f421243cba",
            "computed at download after the file's git blob SHA-1 matched the repository's", "MIT",
        ),
        WeightFile(
            "musiq_spaq", "metrics/musiq/musiq_spaq_ckpt-358bb6af.pth",
            "github.com/chaofengc/IQA-PyTorch release v0.1-weights, musiq_spaq_ckpt-358bb6af.pth",
            108610983, "358bb6af275e28ea56821d44fb55c6cb83645db11f394d3ad65b2d149965ab50",
            "official 8-hex prefix in the file name; the full hash matches the Hugging Face copy (chaofengc/IQA-PyTorch-Weights)",
            "CC BY-NC-SA 4.0 (research only)",
        ),
        WeightFile(
            "amt_s_original", "metrics/amt/amt-s.pth",
            "huggingface.co/lalala125/AMT @ 31d0420ef7788f6dc22aad3445148705007141bb, amt-s.pth (linked from github.com/MCG-NKU/AMT)",
            12038359, "07e7e03405c213fe4405678db9b42d05671e48c56faf6a441b99bb0047d3cf77",
            "Hugging Face LFS", "CC BY-NC-SA 4.0 weights, CC BY-NC 4.0 code (research only)", loads=False,
        ),
        WeightFile(
            "amt_s", "metrics/amt/amt-s.safetensors",
            "converted from amt_s_original by scripts/convert_metric_weights.py: the checkpoint's state_dict (epoch 295)",
            11982684, "704f6d5063c150f23548edba03c6cd1585f78e33168c67fafe88541cb3c119f3",
            "computed at conversion; every tensor bitwise equal, and the vendored AMT-S loads it strictly", "as amt_s_original",
        ),
    )
}


def verified_path(name: str) -> Path:
    """The weight file's path, after checking its size and SHA-256; originals kept for conversion cannot be loaded."""
    weight = WEIGHTS[name]
    if not weight.loads:
        raise RuntimeError(f"{name} is kept only as the source of a converted file; load its converted copy")
    path = ROOT / weight.path
    if not path.exists():
        raise FileNotFoundError(f"{name}: {path} is missing; weights are never downloaded at run time")
    if path.stat().st_size != weight.size or sha256_file(path) != weight.sha256:
        raise RuntimeError(f"{name}: {path} does not match its registered size and SHA-256")
    return path
