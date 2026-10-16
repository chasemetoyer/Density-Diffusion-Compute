"""One session of a run on one machine: the gates, then every ready job of that machine, in order, each atomically.

Gates, before any job (the run stops if one fails):
- strict deterministic mode (density.determinism, registered);
- on the H100: the GPU is an H100, the per-pod canary passed on this host at this code (density.canary), the encode
  canary reproduces its registered decoded frames (``encode_canary``), and the metric canary passed (density.metric_canary);
- on the slice: the mask canary and the metric canary passed on this host at this code (density.mask_canary,
  density.metric_canary);
- the volume holds less than the registered 112 GB: a full du at the start, then before each job the running total of
  the files every job wrote (their allocated size), with a full du every 30 minutes (density.volume.Room).

Every prompt's jobs on one machine run on the same GPU model and image: the first job of a prompt records its platform
in platforms.json (GPU name, image, environment lock), and a job on any other platform is refused.

Jobs run in manifest order, each one only when its dependencies are done; a job's record is written last and
atomically, so an interrupted job simply runs again. A failure writes failed.json with its traceback and stops the
session; rerunning the session reruns the job under the same settings. Each job's output goes to its own log.txt.
"""

import contextlib
import io
import json
import os
import platform as platform_module
import socket
import sys
import time
import traceback
from pathlib import Path

import torch

from density import determinism, volume
from density.run import freeze as freeze_module
from density.run.manifest import Job
from density.run.store import ROOT, Run, check_run_folder, file_sha256, read_json, utc_now, write_json_atomic

ENCODE_CANARY = ROOT / "configs" / "encode_canary.json"
KIND_ORDER = {"embed": 0, "reference": 1, "mask": 2, "condition": 3, "score": 4}
GPU_NAMES = {"h100": "H100", "slice": "RTX PRO 6000"}


class GateFailed(RuntimeError):
    pass


class PlatformMismatch(RuntimeError):
    pass


def encode_canary(record: Path = ENCODE_CANARY) -> dict:
    """Encode the first 17 frames of the face stock clip and require the registered decoded frames, bit for bit."""
    import tempfile

    from density import video
    from density.masks.clips import load_frames

    registered = json.loads(record.read_text())
    frames = load_frames(registered["clip"])[: registered["frames"]]
    with tempfile.TemporaryDirectory() as tmp:
        got = video.encode(frames, Path(tmp) / "canary.mp4")
    passed = got["decoded_frames_sha256"] == registered["decoded_frames_sha256"]
    out = {"passed": passed, "decoded_frames_sha256": got["decoded_frames_sha256"], "registered": registered["decoded_frames_sha256"],
           "registered_on": registered["recorded_on"], "psnr_vs_lossless_db": got["psnr_vs_lossless_db"]}
    if not passed:
        raise GateFailed(f"the encode canary's decoded frames differ from the registration made on {registered['recorded_on']}: stop and report this host")
    return out


def record_encode_canary(record: Path = ENCODE_CANARY, clip: str = "face", frames: int = 17) -> dict:
    """Write the encode canary's registration: the decoded frames of this machine's encode of the clip's first frames."""
    import platform
    import tempfile

    from density import video
    from density.checks import euler
    from density.masks.clips import load_frames

    source = load_frames(clip)[:frames]
    with tempfile.TemporaryDirectory() as tmp:
        got = video.encode(source, Path(tmp) / "canary.mp4")
    out = {"clip": clip, "frames": frames, "source_uint8_sha256": video.frames_sha256(source), "decoded_frames_sha256": got["decoded_frames_sha256"],
           "psnr_vs_lossless_db": got["psnr_vs_lossless_db"], "encoding": got["encoding"], "x264_options": got["x264_options"],
           "recorded_on": f"{socket.gethostname()} ({platform.machine()}, {platform.platform()})", "utc": utc_now(), "code": euler.git_state()}
    record.write_text(json.dumps(out, indent=2) + "\n")
    return out


def platform_identity(machine: str, dry_run: bool) -> dict:
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {"machine": machine, "gpu": gpu or ("none (dry run on " + platform_module.machine() + ")" if dry_run else None),
            "image": os.environ.get("POD_IMAGE") or ("dry run" if dry_run else None), "uv_lock_sha256": file_sha256(ROOT / "uv.lock"),
            "python": platform_module.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda}


def check_platform(run: Run, job: Job, identity: dict) -> None:
    """The first job of a prompt on a machine fixes its platform; every later one must match it."""
    path = run.path("platforms.json")
    platforms = read_json(path) if path.exists() else {}
    key = f"{job.prompt_id:03d}/{job.machine}"
    pinned = {k: identity[k] for k in ("gpu", "image", "uv_lock_sha256")}
    if key not in platforms:
        platforms[key] = {**pinned, "first_job": job.id, "utc": utc_now()}
        write_json_atomic(path, platforms)
        return
    expected = {k: platforms[key][k] for k in pinned}
    if expected != pinned:
        raise PlatformMismatch(f"{job.id}: prompt {job.prompt_id}'s {job.machine} jobs ran on {expected}, this session is {pinned}")


def gates(machine: str, dry_run: bool) -> dict:
    out = {"determinism": determinism.configure(strict=True)}
    if dry_run:
        out["canaries"] = "not run: a dry run has no registered pod to reproduce"
        out["encode_canary"] = encode_canary() if ENCODE_CANARY.exists() else "no registration yet"
        return out
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    if GPU_NAMES[machine] not in gpu:
        raise GateFailed(f"a {machine} session needs a GPU named like {GPU_NAMES[machine]!r}; this is {gpu or 'no GPU'}")
    if not os.environ.get("POD_IMAGE"):
        raise GateFailed("POD_IMAGE is not set: record the pod's container image with scripts/pod/sync.sh pod-env, so each prompt's platform can be pinned")
    try:
        from density import metric_canary

        if machine == "h100":
            from density import canary

            out["canary"] = {k: v for k, v in canary.require_passed().items() if k in ("utc", "host", "passed", "matches")}
            out["encode_canary"] = encode_canary()
        else:
            from density import mask_canary

            out["mask_canary"] = {k: v for k, v in mask_canary.require_passed().items() if k in ("utc", "host", "passed")}
        out["metric_canary"] = {k: v for k, v in metric_canary.require_passed(machine).items() if k in ("utc", "host", "passed", "recorded", "digest")}
    except Exception as error:
        raise GateFailed(f"{type(error).__name__}: {error}") from error
    return out


class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)
        return len(text)

    def flush(self):
        for s in self.streams:
            s.flush()


class Executors:
    """The models a session needs, built on first use."""

    def __init__(self, run: Run, machine: str, device: str, backend=None, weights_dir=None):
        self.run, self.machine, self.device = run, machine, device
        self._backend, self._weights = backend, weights_dir
        self._lpips = self._mask_models = self._quality = self._detectors = None
        if run.dry_run:  # dry runs only: see density.run.masks.run_mask
            from density.run.masks import DiskCache
            from density.run.store import DRY_RUN_ROOT

            self._mask_cache = DiskCache(DRY_RUN_ROOT / "mask_cache")
        else:
            self._mask_cache = None
        self.rows = {int(r["id"]): r for r in read_json(run.path("inputs", "prompts.json"))["rows"]}

    @property
    def backend(self):
        if self._backend is None:
            from density.run.generate import WanBackend

            self._backend = WanBackend(self.device, self._weights)
        return self._backend

    def __call__(self, job: Job) -> dict:
        from density.run import generate, masks, score

        if job.kind == "embed":
            return generate.run_embed(self.run, job, self.backend, self.rows)
        if job.kind == "reference":
            return generate.run_reference(self.run, job, self.backend, self.rows)
        if job.kind == "condition":
            frozen = freeze_module.verify(self.run, job)
            if self._lpips is None:
                from density.metrics.fidelity import LPIPS

                self._lpips = LPIPS(self.device)
            return {**generate.run_condition_job(self.run, job, self.backend, self.rows, self._lpips), "freeze": frozen}
        if job.kind == "mask":
            if self._mask_models is None:
                from density.masks.pipeline import Models

                self._mask_models = Models(self.device)
            return masks.run_mask(self.run, job, self._mask_models, cache=self._mask_cache)
        if job.kind == "score":
            if self._quality is None:
                from density.metrics.detection import Detectors
                from density.metrics.quality import QualityModels

                self._quality, self._detectors = QualityModels(self.device), Detectors(self.device)
            return score.run_score(self.run, job, self._quality, self._detectors)
        raise ValueError(job.kind)


def environment(identity: dict) -> dict:
    from density.checks import euler

    return {"host": socket.gethostname(), "platform": identity, "code": euler.git_state(), "argv": sys.argv,
            "pod_env": {k: v for k, v in os.environ.items() if k.startswith("RUNPOD_") and "KEY" not in k and "TOKEN" not in k}}


def run_job(run: Run, job: Job, execute, identity: dict, env: dict | None = None) -> dict:
    """Run one job with its own log; ``env`` (the session's host, platform and code state) is computed once per session."""
    env = env or environment(identity)
    folder = run.job_dir(job)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "failed.json").unlink(missing_ok=True)
    log = determinism.WarningLog()
    start, began = time.perf_counter(), utc_now()
    with open(folder / "log.txt", "a") as log_file:
        log_file.write(f"=== {job.id} started {began} on {socket.gethostname()}\n")
        tee_out, tee_err = Tee(sys.stdout, log_file), Tee(sys.stderr, log_file)
        try:
            with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err), log.capture(job.id):
                outputs = execute(job)
        except BaseException as error:
            write_json_atomic(folder / "failed.json", {"job": job.to_dict(), "utc_start": began, "utc_end": utc_now(), "error": f"{type(error).__name__}: {error}",
                                                       "traceback": traceback.format_exc(), "warnings": log.summary(), "env": env})
            log_file.write(f"=== {job.id} FAILED: {type(error).__name__}: {error}\n")
            raise
        log_file.write(f"=== {job.id} done in {time.perf_counter() - start:.1f} s\n")
    record = {"job": job.to_dict(), "status": "done", "dry_run": run.dry_run, "utc_start": began, "utc_end": utc_now(),
              "job_seconds": time.perf_counter() - start, "warnings": log.summary(), "env": env,
              "determinism": determinism.settings(), "log": "log.txt"}
    clash = set(record) & set(outputs)
    if clash:
        raise RuntimeError(f"{job.id}: outputs use reserved record keys {sorted(clash)}")
    record.update(outputs)
    write_json_atomic(folder / "record.json", record)
    return record


def written_paths(run: Run, job: Job, session_name: str) -> list[Path]:
    """Everything a job can write, for the volume's running total: its own folder, by kind the prompt's embedding,
    the seed's noise and the group's mask or score folder, and the run's platform pins and this session's record."""
    from density.run.generate import embedding_path, noise_path

    paths = [run.job_dir(job), run.path("platforms.json"), run.path("sessions", f"{session_name}.json")]
    if job.kind == "embed":
        paths.append(embedding_path(run, job.prompt_id))
    if job.kind in ("reference", "condition"):
        paths.append(noise_path(run, job.prompt_id, job.seed))
    if job.kind == "mask":
        paths.append(run.mask_dir(job.group))
    if job.kind == "score":
        paths.append(run.score_dir(job.group))
    return paths


def session(folder: Path, machine: str, device: str | None = None, backend=None, weights_dir=None, kinds=None, max_jobs: int | None = None,
            execute=None) -> dict:
    """Run every ready job of ``machine``; ``execute`` replaces the real job executors (tests only)."""
    run = Run(folder)
    manifest = run.manifest()
    dry_run = bool(manifest["dry_run"])
    check_run_folder(run.folder, dry_run)
    if backend is not None and backend.dry_run != dry_run:
        raise RuntimeError("the backend and the run disagree about being a dry run")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    name = f"{machine}_{socket.gethostname()}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    record = {"session": name, "machine": machine, "device": device, "utc_start": utc_now(), "dry_run": dry_run, "jobs": [], "volume": []}
    identity = platform_identity(machine, dry_run)
    record["platform"] = identity
    write_json_atomic(run.path("sessions", f"{name}.json"), record)
    try:
        record["gates"] = gates(machine, dry_run)
        execute = execute or Executors(run, machine, device, backend, weights_dir)
        env = environment(identity)  # git state once per session, not per job (git status is slow on the network volume)
        record["env"] = env
        room = volume.Room(run.folder if dry_run else None)  # the registered volume stop: the session's first full du
        record["volume"] = room.log
        jobs = run.jobs()  # read once: nothing adds jobs during a session
        done = {j.id for j in jobs if run.done(j)}  # one stat per job, then kept in memory
        while True:
            pending = [j for j in jobs if j.machine == machine and j.id not in done and (kinds is None or j.kind in kinds)]
            ready = sorted((j for j in pending if all(d in done for d in j.depends)), key=lambda j: KIND_ORDER[j.kind])
            if not ready or (max_jobs is not None and len(record["jobs"]) >= max_jobs):
                break
            job = ready[0]
            room.before_job(job.id)
            check_platform(run, job, identity)
            finished = run_job(run, job, execute, identity, env)
            done.add(job.id)
            room.after_job(job.id, written_paths(run, job, name))
            record["jobs"].append({"job": job.id, "seconds": finished["job_seconds"]})
            write_json_atomic(run.path("sessions", f"{name}.json"), record)
        record["stop"] = None
    except Exception as error:
        record["stop"] = f"{type(error).__name__}: {error}"
        record["traceback"] = traceback.format_exc()
    record["utc_end"] = utc_now()
    record["remaining"] = run.status()
    write_json_atomic(run.path("sessions", f"{name}.json"), record)
    return record
