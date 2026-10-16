"""Conform stock test clips to the reference geometry: 81 frames, 16 fps, 832 x 480.

Frame selection is explicit nearest-frame: for target time start + i/16, the
source frame with the nearest packet timestamp is taken (no interpolation;
ties go to the earlier frame). ffmpeg then extracts exactly those frames,
scales to height 480 with area interpolation, and center-crops to 832 x 480,
writing lossless PNGs.
"""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from density.masks.registered import FPS, FRAMES, HEIGHT, WIDTH

ROOT = Path(__file__).resolve().parents[3]
REGISTRY = ROOT / "configs" / "test_clips.json"
CLIPS_DIR = ROOT / "data" / "sample_clips"


def tool_versions() -> dict:
    """ffmpeg and ffprobe versions; None where a tool is not installed (the pod verifies frames instead of conforming)."""

    def first(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.splitlines()[0]
        except (OSError, subprocess.CalledProcessError):
            return None

    return {"ffmpeg": first(["ffmpeg", "-version"]), "ffprobe": first(["ffprobe", "-version"])}


def packet_timestamps(source: Path) -> list[float]:
    """Presentation timestamps of the video packets, in decode order (VP8/VP9 have no reordering)."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time", "-of", "csv=p=0", str(source)],
        capture_output=True, text=True, check=True,
    ).stdout
    return [float(line.split(",")[0]) for line in out.split() if line.strip() and line.split(",")[0] != "N/A"]


def nearest_frames(timestamps: list[float], start: float, count: int = FRAMES, fps: int = FPS) -> list[int]:
    ts = np.asarray(timestamps)
    if np.any(np.diff(ts) <= 0):
        raise ValueError("timestamps are not strictly increasing")
    targets = start + np.arange(count) / fps
    if targets[-1] > ts[-1]:
        raise ValueError(f"segment ends at {targets[-1]:.3f} s, after the last frame at {ts[-1]:.3f} s")
    return [int(np.argmin(np.abs(ts - t))) for t in targets]  # argmin takes the first, i.e. earlier, on ties


def conform(clip: dict) -> dict:
    """Write frames/frame_000.png .. frame_080.png for one registry entry; return its conform record."""
    folder = CLIPS_DIR / clip["id"]
    source = folder / "source.webm"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != clip["sha256"]:
        raise RuntimeError(f"{clip['id']}: source does not match its registered SHA-256")
    timestamps = packet_timestamps(source)
    indices = nearest_frames(timestamps, clip["segment_start_s"])
    frames_dir = folder / "frames"
    frames_dir.mkdir(exist_ok=True)
    for old in frames_dir.glob("frame_*.png"):
        old.unlink()
    select = "+".join(f"eq(n\\,{i})" for i in sorted(set(indices)))
    vf = f"select='{select}',scale=-1:{HEIGHT}:flags=area,crop={WIDTH}:{HEIGHT}"
    command = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(source), "-vf", vf, "-fps_mode", "passthrough", "-start_number", "0", str(frames_dir / "unique_%03d.png")]
    subprocess.run(command, check=True)
    unique = sorted(set(indices))
    written = sorted(frames_dir.glob("unique_*.png"))
    if len(written) != len(unique):
        raise RuntimeError(f"{clip['id']}: ffmpeg wrote {len(written)} frames, expected {len(unique)}")
    by_index = dict(zip(unique, written))
    for i, index in enumerate(indices):  # a source frame chosen twice (upsampling) is copied, not interpolated
        (frames_dir / f"frame_{i:03d}.png").write_bytes(by_index[index].read_bytes())
    for path in written:
        path.unlink()
    frames = load_frames(clip["id"])
    return {
        "id": clip["id"],
        "segment_start_s": clip["segment_start_s"],
        "source_sha256": digest,
        "source_frames": indices,
        "source_times_s": [round(timestamps[i], 6) for i in indices],
        "ffmpeg_filter": vf,
        "command": " ".join(command[:-1] + ["frames/unique_%03d.png"]),
        "frames_sha256": hashlib.sha256(frames.tobytes()).hexdigest(),
        "frame_png_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(frames_dir.glob("frame_*.png"))},
    }


def load_frames(clip_id: str) -> np.ndarray:
    """(81, 480, 832, 3) uint8 RGB."""
    paths = sorted((CLIPS_DIR / clip_id / "frames").glob("frame_*.png"))
    if len(paths) != FRAMES:
        raise RuntimeError(f"{clip_id}: {len(paths)} conformed frames, expected {FRAMES}")
    frames = np.stack([np.asarray(Image.open(p).convert("RGB")) for p in paths])
    if frames.shape != (FRAMES, HEIGHT, WIDTH, 3):
        raise RuntimeError(f"{clip_id}: frames have shape {frames.shape}")
    return frames


def registry() -> dict:
    return json.loads(REGISTRY.read_text())
