"""Wan2.1-T2V-1.3B-Diffusers files at the pinned revision, verified against Hugging Face's published hashes.

``configs/wan_weights.json`` lists every file the pipeline uses, with the LFS
SHA-256 for large files and the git object id (SHA-1 of the blob) for small
non-LFS files such as configs. Nothing loads a Wan file that has not been
verified here first.
"""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "configs" / "wan_weights.json"


def registry() -> dict:
    return json.loads(REGISTRY.read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    """The git object id of a file: SHA-1 of ``blob <size>\\0`` followed by its bytes."""
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def verify(weights_dir: Path, components: tuple[str, ...] | None = None) -> dict:
    """Check size and hash of every registered file (or those under ``components``); raise on any mismatch."""
    reg = registry()
    checked = {}
    for rel, expected in reg["files"].items():
        if components is not None and rel != "model_index.json" and rel.split("/")[0] not in components:
            continue
        path = Path(weights_dir) / rel
        if not path.is_file():
            raise FileNotFoundError(f"{rel} is missing from {weights_dir}")
        if path.stat().st_size != expected["size"]:
            raise RuntimeError(f"{rel}: size {path.stat().st_size} differs from the registered {expected['size']}")
        if "sha256" in expected:
            got, kind = sha256_file(path), "sha256"
        else:
            got, kind = git_blob_sha1(path), "git_blob_sha1"
        if got != expected[kind]:
            raise RuntimeError(f"{rel}: {kind} {got} differs from the registered {expected[kind]}")
        checked[rel] = {kind: got, "size": expected["size"]}
    return {"repo": reg["repo"], "revision": reg["revision"], "files": checked}
