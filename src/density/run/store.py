"""A run's folder: the manifest, job records written atomically, per-job logs, and lossless frames checked by hash.

    <run>/manifest.json            the current manifest; manifest_v<n>.json keeps every version
    <run>/inputs/                  embeddings/<prompt>.safetensors, noise/<prompt>-s<seed>.safetensors
    <run>/jobs/<job id>/           record.json (written last: its presence means done), log.txt, failed.json,
                                   frames/ (a reference's lossless PNGs) or frames.mkv (a condition's FFV1 copy),
                                   video.mp4 (playback), allocation.json
    <run>/masks/<group>/           masks.json, coverage.npz, metadata.json, sheet.png
    <run>/scores/<group>/          one JSON per scored video
    <run>/audit/audit.csv          the audit sheets' review file for Chase
    <run>/frozen.json              the freeze (density.run.freeze)
    <run>/platforms.json           the GPU, image and environment each prompt's jobs ran on
    <run>/sessions/                one record per session: gates, host, volume checks, jobs run

A dry run's folder is marked in its manifest and lives under scratch/; a real run refuses it, and a dry run refuses any
real run folder, so the two never read each other's files.
"""

import dataclasses
import enum
import hashlib
import io
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np

from density.run.manifest import Job

ROOT = Path(__file__).resolve().parents[3]
DRY_RUN_ROOT = ROOT / "scratch" / "dryrun"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, enum.Enum):
        return value.name
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return jsonable(dataclasses.asdict(value))
    if isinstance(value, (set, frozenset)):
        return sorted(jsonable(v) for v in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)  # torch dtypes and other named objects


def write_bytes_atomic(path: Path, data: bytes) -> None:
    """Write to a temporary file in the same folder, flush it to disk, then rename it over ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_json_atomic(path: Path, data) -> None:
    write_bytes_atomic(path, (json.dumps(jsonable(data), indent=2, sort_keys=False) + "\n").encode())


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def png_bytes(frame: np.ndarray) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(frame).save(buffer, format="PNG")
    return buffer.getvalue()


def save_frames(frames: np.ndarray, folder: Path) -> dict:
    """Lossless PNG frames, written to a temporary folder and renamed into place; returns their hashes."""
    if frames.dtype != np.uint8 or frames.ndim != 4:
        raise ValueError(f"frames must be (T, H, W, 3) uint8, got {frames.shape} {frames.dtype}")
    folder = Path(folder)
    tmp = folder.with_name(folder.name + ".tmp")
    if tmp.exists():
        for p in tmp.iterdir():
            p.unlink()
    tmp.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for i, frame in enumerate(frames):
        data = png_bytes(frame)
        (tmp / f"frame_{i:03d}.png").write_bytes(data)
        hashes[f"frame_{i:03d}.png"] = sha256_bytes(data)
    if folder.exists():
        for p in folder.iterdir():
            p.unlink()
        folder.rmdir()
    tmp.rename(folder)
    return {"folder": folder.name, "png_sha256": hashes, "uint8_sha256": sha256_bytes(np.ascontiguousarray(frames).tobytes()),
            "bytes": sum((folder / n).stat().st_size for n in hashes)}


def load_frames(folder: Path, record: dict) -> np.ndarray:
    """uint8 frames from PNGs, checked file by file and as a stack against the record ``save_frames`` returned."""
    from PIL import Image

    folder = Path(folder)
    paths = sorted(folder.glob("frame_*.png"))
    if [p.name for p in paths] != sorted(record["png_sha256"]):
        raise RuntimeError(f"{folder}: the frame files differ from the record")
    frames = []
    for p in paths:
        data = p.read_bytes()
        if sha256_bytes(data) != record["png_sha256"][p.name]:
            raise RuntimeError(f"{p}: hash differs from the record")
        frames.append(np.asarray(Image.open(io.BytesIO(data)).convert("RGB")))
    stack = np.stack(frames)
    if sha256_bytes(stack.tobytes()) != record["uint8_sha256"]:
        raise RuntimeError(f"{folder}: decoded frames differ from the record")
    return stack


def lossless_frames(run: "Run", job_id: str) -> np.ndarray:
    """A video's lossless frames, checked by hash: a reference's PNG frames or a condition's FFV1 copy."""
    from density import video

    rec = run.record(job_id)
    if "frames" in rec:
        return load_frames(run.job_dir(job_id) / rec["frames"]["folder"], rec["frames"])
    if "lossless" in rec:
        return video.load_lossless(run.job_dir(job_id) / rec["lossless"]["path"], rec["lossless"])
    raise RuntimeError(f"{job_id}: its record names no lossless frames")


class Run:
    def __init__(self, folder: Path):
        self.folder = Path(folder).resolve()

    # Paths
    def path(self, *parts) -> Path:
        return self.folder.joinpath(*parts)

    def job_dir(self, job: Job | str) -> Path:
        return self.path("jobs", job if isinstance(job, str) else job.id)

    def mask_dir(self, group: str) -> Path:
        return self.path("masks", group)

    def score_dir(self, group: str) -> Path:
        return self.path("scores", group)

    # Manifest
    def manifest(self) -> dict:
        return read_json(self.path("manifest.json"))

    @property
    def dry_run(self) -> bool:
        return bool(self.manifest()["dry_run"])

    def jobs(self) -> list[Job]:
        return [Job.from_dict(d) for d in self.manifest()["jobs"]]

    def job(self, job_id: str) -> Job:
        found = [j for j in self.jobs() if j.id == job_id]
        if not found:
            raise KeyError(job_id)
        return found[0]

    def write_manifest(self, manifest: dict, note: str) -> dict:
        """A new manifest version: the jobs must be unique, and earlier jobs can only be kept, never changed or removed."""
        ids = [j["id"] for j in manifest["jobs"]]
        if len(ids) != len(set(ids)):
            raise ValueError("the manifest lists a job twice")
        previous = self.manifest() if self.path("manifest.json").exists() else None
        if previous is not None:
            old = {j["id"]: j for j in previous["jobs"]}
            new = {j["id"]: j for j in manifest["jobs"]}
            changed = [i for i in old if new.get(i) != old[i]]
            if changed:
                raise ValueError(f"a new manifest version cannot change or remove jobs: {changed[:5]}")
            for key in ("run", "kind", "dry_run", "prompts_csv_sha256"):
                if manifest[key] != previous[key]:
                    raise ValueError(f"a new manifest version cannot change {key}")
        version = 1 if previous is None else previous["version"] + 1
        manifest = {**manifest, "version": version, "history": [*(previous or {}).get("history", []), {"version": version, "utc": utc_now(), "note": note, "jobs": len(ids)}]}
        write_json_atomic(self.path(f"manifest_v{version}.json"), manifest)
        write_json_atomic(self.path("manifest.json"), manifest)
        return manifest

    def add_jobs(self, jobs: list[Job], note: str) -> dict:
        manifest = self.manifest()
        known = {j["id"] for j in manifest["jobs"]}
        new = [j.to_dict() for j in jobs if j.id not in known]
        return self.write_manifest({**manifest, "jobs": manifest["jobs"] + new}, note)

    # Job state
    def record(self, job: Job | str) -> dict | None:
        path = self.job_dir(job) / "record.json"
        return read_json(path) if path.exists() else None

    def done(self, job: Job | str) -> bool:
        return (self.job_dir(job) / "record.json").exists()

    def failed(self, job: Job | str) -> dict | None:
        path = self.job_dir(job) / "failed.json"
        return read_json(path) if path.exists() and not self.done(job) else None

    def ready(self, job: Job) -> bool:
        return all(self.done(d) for d in job.depends)

    def status(self) -> dict:
        jobs = self.jobs()
        out = {}
        for j in jobs:
            state = "done" if self.done(j) else "failed" if self.failed(j) else "ready" if self.ready(j) else "waiting"
            out.setdefault(j.kind, {}).setdefault(state, 0)
            out[j.kind][state] += 1
        return out


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def check_run_folder(folder: Path, dry_run: bool) -> None:
    """A dry run only under scratch/dryrun; a real run never there."""
    folder = Path(folder).resolve()
    inside = DRY_RUN_ROOT.resolve() in folder.parents or folder == DRY_RUN_ROOT.resolve()
    if dry_run and not inside:
        raise RuntimeError(f"a dry run writes only under {DRY_RUN_ROOT}, not {folder}")
    if not dry_run and inside:
        raise RuntimeError(f"{folder} is a dry-run folder; a real run never reads it")
