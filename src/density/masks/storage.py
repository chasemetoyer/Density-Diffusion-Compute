"""Per-reference storage: run-length pixel masks, token-coverage arrays, and metadata.

    masks.json     {"regions": {name: RLE}, "tracks": {region: {track: RLE}}, "backfill": {region: RLE}}
                   regions include the stratum ROI; backfill marks pixels only SAM's propagation before a
                   track's first detection covers
    coverage.npz   token coverage m per mask ("ROI", and "A1" for manipulation), float32
    metadata.json  versions, thresholds, keyframes, re-prompts, contact intervals, eligibility, timings, hashes
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from density.masks.geometry import rle_decode, rle_encode


def mask_sha256(mask: np.ndarray) -> str:
    """SHA-256 of the bit-packed mask (row-major), independent of the file format."""
    return hashlib.sha256(np.packbits(np.asarray(mask, dtype=bool)).tobytes()).hexdigest()


def save_reference(
    folder: Path,
    regions: dict[str, np.ndarray],
    coverage: dict[str, np.ndarray],
    metadata: dict,
    tracks: dict[str, dict[int, np.ndarray]] | None = None,
    backfill: dict[str, np.ndarray] | None = None,
) -> dict:
    """Write masks.json, coverage.npz and metadata.json; return their SHA-256 hashes."""
    folder.mkdir(parents=True, exist_ok=True)
    masks = {
        "regions": {name: rle_encode(mask) for name, mask in regions.items()},
        "tracks": {region: {str(t): rle_encode(m) for t, m in sorted(by_track.items())} for region, by_track in (tracks or {}).items()},
        "backfill": {region: rle_encode(m) for region, m in (backfill or {}).items()},
    }
    (folder / "masks.json").write_text(json.dumps(masks))
    np.savez(folder / "coverage.npz", **{name: m.astype(np.float32) for name, m in coverage.items()})
    hashes = {
        "masks.json": hashlib.sha256((folder / "masks.json").read_bytes()).hexdigest(),
        "coverage.npz": hashlib.sha256((folder / "coverage.npz").read_bytes()).hexdigest(),
        "region_sha256": {name: mask_sha256(mask) for name, mask in regions.items()},
        "track_sha256": {region: {str(t): mask_sha256(m) for t, m in sorted(by_track.items())} for region, by_track in (tracks or {}).items()},
    }
    (folder / "metadata.json").write_text(json.dumps({**metadata, "files_sha256": hashes}, indent=2, default=str))
    hashes["metadata.json"] = hashlib.sha256((folder / "metadata.json").read_bytes()).hexdigest()
    return hashes


def load_masks(folder: Path) -> dict[str, np.ndarray]:
    return {name: rle_decode(rle) for name, rle in json.loads((folder / "masks.json").read_text())["regions"].items()}


def load_tracks(folder: Path) -> dict[str, dict[int, np.ndarray]]:
    tracks = json.loads((folder / "masks.json").read_text())["tracks"]
    return {region: {int(t): rle_decode(rle) for t, rle in by_track.items()} for region, by_track in tracks.items()}
