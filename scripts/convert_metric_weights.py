"""One-time conversion of the metric weights that cannot load data-only, with exactness checks (approved by Chase,
2026-10-07). Run once on the Mac after the originals are downloaded and verified:

    python scripts/convert_metric_weights.py

- OpenAI CLIP ViT-B/32 and ViT-L/14 ship as TorchScript archives. VBench's clip.load(jit=False) takes the archive's
  state dict and calls clip.model.build_model on it, so the converted file holds exactly that state dict (fp16 weights,
  fp32 and int64 entries as stored). Checks: every tensor round-trips bitwise, and build_model on the converted dict
  gives the same image features as build_model on the archive's dict.
- AMT-S's checkpoint is a pickle that needs typing.OrderedDict, which torch's data-only loader refuses. It is loaded
  once with only that class allowed, and its state dict is saved. Checks: every tensor round-trips bitwise, and the
  vendored AMT-S network loads it strictly.

Each original's SHA-256 is checked before it is opened; the converted files' sizes and hashes are printed for
src/density/metrics/weights.py.
"""

import hashlib
import sys
import typing
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
W = ROOT / "weights" / "metrics"
SOURCES = {
    "clip/ViT-B-32.pt": "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af",
    "clip/ViT-L-14.pt": "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836",
    "amt/amt-s.pth": "07e7e03405c213fe4405678db9b42d05671e48c56faf6a441b99bb0047d3cf77",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified(rel: str) -> Path:
    path = W / rel
    got = sha256_file(path)
    if got != SOURCES[rel]:
        raise RuntimeError(f"{rel}: SHA-256 {got} differs from the registered {SOURCES[rel]}")
    return path


def save_and_check(state: dict, out: Path) -> dict:
    """Save ``state`` as safetensors and check that every tensor comes back bitwise identical."""
    tensors = {k: v.detach().contiguous().clone() for k, v in state.items()}
    save_file(tensors, str(out))
    back = load_file(str(out))
    assert sorted(back) == sorted(tensors), "keys differ after the round trip"
    for k, v in tensors.items():
        assert back[k].dtype == v.dtype and back[k].shape == v.shape and torch.equal(back[k], v), f"{k} differs after the round trip"
    return back


def convert_clip(name: str) -> None:
    sys.path.insert(0, str(ROOT / "third_party" / "clip"))
    from clip.model import build_model

    archive = torch.jit.load(str(verified(f"clip/{name}.pt")), map_location="cpu").eval()
    original = archive.state_dict()
    out = W / "clip" / f"{name}.safetensors"
    converted = save_and_check(original, out)
    image = torch.randn(2, 3, 224, 224, generator=torch.Generator().manual_seed(0))
    features = []
    for state in (original, converted):
        model = build_model(dict(state)).float().eval()  # clip.load(jit=False) on CPU: build_model, then .float()
        with torch.no_grad():
            features.append(model.encode_image(image))
    assert torch.equal(features[0], features[1]), "image features differ between the archive's and the converted state dict"
    print(f"{out.relative_to(ROOT)}: {len(converted)} tensors, {out.stat().st_size} bytes, sha256 {sha256_file(out)}")


def convert_amt() -> None:
    sys.path.insert(0, str(ROOT / "third_party" / "vbench"))
    import importlib

    with torch.serialization.safe_globals([typing.OrderedDict]):
        checkpoint = torch.load(verified("amt/amt-s.pth"), map_location="cpu", weights_only=True)
    out = W / "amt" / "amt-s.safetensors"
    converted = save_and_check(checkpoint["state_dict"], out)
    model = importlib.import_module("vbench.third_party.amt.networks.AMT-S").Model(corr_radius=3, corr_lvls=4, num_flows=3)
    model.load_state_dict(converted, strict=True)  # the parameters cfgs/AMT-S.yaml gives
    print(f"{out.relative_to(ROOT)}: {len(converted)} tensors (checkpoint epoch {checkpoint.get('epoch')}), {out.stat().st_size} bytes, sha256 {sha256_file(out)}")


if __name__ == "__main__":
    convert_clip("ViT-B-32")
    convert_clip("ViT-L-14")
    convert_amt()
