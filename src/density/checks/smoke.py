"""GPU smoke test on RunPod: exactness, determinism, memory, timing and mask cost on CUDA.

Protocol sections 8, 11, 12 and 13. The smoke test produces no experiment data, and its two prompts
(``configs/smoke_prompts.json``) are excluded from every prompt pool.

Each step is a subcommand, ``python -m density.checks.smoke <step>``, that writes
``results/checks/smoke_test/<step>.json``. A step is skipped when its JSON was written at the same code
SHA from the same inputs (``--force`` reruns it). The starting noise and the text embeddings are inputs:
they are written once and never recomputed automatically.

Failures follow the approved definition. An error, an out-of-memory error or a hash mismatch writes
``<step>.error.json`` or records a ``stop`` entry, and the step exits 1 so the session script stops.
Cross-device hash comparisons are reported, not judged. A same-device result that is not bitwise is a
measurement; it is reported with its size relative to the repeat-run noise on the same device, and for
T3, where dense repeats are bitwise on CUDA, as decoded PSNR with the frames saved losslessly. A
protocol threshold missed on the same device (repeat-run decoded PSNR below 45 dB) is recorded under
``blocks_pilot`` and does not stop the session.

    Session A (24 GB slice)  after scripts/pod/bootstrap.sh: weights, noise, encode, env
    Session B (H100 SXM)     env, weights, pytest, milestones, generate, sdpa, hsa_dense, t3, perf, vae_precision, sheet
    Session C (24 GB slice)  env, weights, masks, mask_check, mask_compare
    Session D (H100 SXM)     flash attention registered, in results/checks/smoke_test/flash: env, weights, milestones,
                             generate, sdpa, hsa_dense, t3, perf, sheet, crossmachine
    Session E (24 GB slice)  the mask-speed diagnosis: env, weights, then scripts/pod/diagnostics/mask_threads.py
    Session F (24 GB slice)  the mask-check rerun with the registered ROIAlign kernel, in results/checks/smoke_test/rerun:
                             env, weights, mask_check, mask_compare, then the gate and the mask canary
    Mac                      noise_mac, report

``--dry-run`` runs every step with tiny random models and the stock clips standing in for the references,
so the code is tested before any expensive pod time: on the Mac (CPU or MPS), and on the slice with
``--device cuda`` to rehearse the CUDA code paths. A dry run always needs its own ``--out`` and
``--scratch``; a pinned input written by a dry run stops a real run instead of being reused.
"""

import argparse
import gc
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from density import determinism, noise, text_encoding, volume, wan_weights
from density.attention import GENERATION_BACKENDS, generation_attention
from density.checks import euler, wan_real
from density.checks import hsa as hsa_checks
from density.conditions import ROI, run_condition, sha256
from density.hsa import PRESETS, Schedule, project_qkv, wan_hsa_velocity
from density.sampler import flow_match_schedule
from density.token_map import token_coverage
from density.wan import VAE_CONV_FP32_PRECISION, WanGuidedVelocity, decode_latents

ROOT = Path(__file__).resolve().parents[3]
SMOKE = json.loads((ROOT / "configs" / "smoke_prompts.json").read_text())
INSTALLS = json.loads((ROOT / "configs" / "pod_installs.json").read_text())
OUT_DIR = euler.REPORT_DIR / "smoke_test"  # not versioned; the report and its hash file are
SCRATCH_DIR = Path(os.environ.get("SMOKE_SCRATCH", "/workspace/smoke_scratch"))  # trajectories; never synced to the Mac
REPORT_JSON = euler.REPORT_DIR / "smoke_test.json"
REPORT_MD = euler.REPORT_DIR / "smoke_test.md"
HASH_FILE = euler.REPORT_DIR / "smoke_test.sha256"
COMMITTED_MASKS = euler.REPORT_DIR / "mask_pipeline.json"

SHIFT, GUIDANCE, STEPS = wan_real.SHIFT, wan_real.GUIDANCE, wan_real.STEPS
SIGMAS, TIMESTEPS = flow_match_schedule(STEPS, SHIFT)
PRESET = PRESETS["HSA-50"]
FIRST_SKIP = 5  # HSA-50 runs every token at iterations 0-4; iteration 5 is the first that skips any
REGISTERED_FRACTION, BUDGET_TOLERANCE = 0.496007, 0.005  # protocol sections 5 and 11
PSNR_THRESHOLD_DB = 45.0  # protocol section 11: repeated dense run and all HSA groups dense
C4_LAMBDA = SMOKE["c4"]["lambda"]
DISK_RADIUS = SMOKE["c4"]["synthetic_mask"]["radius_px"]  # at 480 pixels of height
REDUCED_LATENT_FRAMES = 11  # T3 fallback if the full-resolution oracle does not fit: 41 pixel frames
PROMPTS = SMOKE["prompts"]
PROMPT_NUMBERS = {p["id"]: i + 1 for i, p in enumerate(PROMPTS)}  # run_condition's prompt_id
DRY_RUN_CLIPS = {"smoke-1": "face", "smoke-2": "hands"}  # stock clips standing in for the references in a dry run
GIB = 2**30

# Stop-and-ask limits for each pod (hours since creation), and rough step durations used to check them.
POD_HOURS_CAP = {"slice": 6.0, "h100": 4.0}
STEP_HOURS = {
    "env": 0.05, "weights": 0.1, "noise": 0.01, "encode": 0.2, "pytest": 0.1, "milestones": 0.5, "generate": 0.25,
    "sdpa": 0.05, "hsa_dense": 0.15, "t3": 0.5, "perf": 0.25, "vae_precision": 0.05, "sheet": 0.02, "masks": 0.5, "mask_check": 1.0, "mask_compare": 0.02, "crossmachine": 0.05,
}
PER_SESSION = {"env", "weights"}  # written once per session: env_A.json, weights_B.json, ...
PINNED = {"noise", "encode"}  # inputs: never recomputed without --force
DEPENDS = {
    "generate": ("noise", "encode"),
    "sdpa": ("noise", "encode"),
    "hsa_dense": ("generate",),
    "t3": ("generate",),
    "perf": ("noise", "encode"),
    "sheet": ("generate",),
    "vae_precision": ("generate",),
    "masks": ("generate",),
    "mask_compare": ("mask_check",),
    "crossmachine": ("generate", "t3"),
}


class CapReached(RuntimeError):
    pass


@dataclass
class Context:
    out: Path
    scratch: Path
    device: str
    weights: Path
    session: str | None
    dry_run: bool
    force: bool = False

    @property
    def latent_shape(self) -> tuple[int, ...]:
        return euler.TINY_LATENT_SHAPE if self.dry_run else euler.FULL_LATENT_SHAPE

    @property
    def size(self) -> tuple[int, int, int]:
        """(pixel frames, height, width) of the decoded video."""
        _, _, frames, height, width = self.latent_shape
        return 4 * (frames - 1) + 1, 8 * height, 8 * width

    @property
    def num_tokens(self) -> int:
        _, _, frames, height, width = self.latent_shape
        return frames * (height // 2) * (width // 2)

    @property
    def cuda(self) -> bool:
        return self.device.startswith("cuda")

    def path(self, *parts) -> Path:
        return self.out.joinpath(*parts)


# Utilities


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, data) -> None:
    """Write atomically: a temporary file, then a rename over the target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(jsonable(data), indent=2, default=str))
    os.replace(tmp, path)


def read_result(ctx: Context, name: str) -> dict:
    path = ctx.path(f"{name}.json")
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run the {name.split('_')[0]} step first")
    return json.loads(path.read_text())["result"]


def file_sha256(path) -> str:
    return wan_real.file_sha256(Path(path))


def run_command(cmd: list[str], timeout: int = 600, cwd=None) -> dict:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd)
    except (OSError, subprocess.SubprocessError) as error:
        return {"error": f"{type(error).__name__}: {error}"}
    return {"returncode": r.returncode, "stdout": r.stdout.strip(), "stderr": r.stderr.strip()[-2000:]}


def first_line(cmd: list[str]) -> str | None:
    r = run_command(cmd)
    out = (r.get("stdout") or "").splitlines()
    return out[0] if out else None


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


class Timer:
    """Elapsed seconds: CUDA events on CUDA (protocol section 8), perf_counter after a device sync elsewhere."""

    def __init__(self, device: str):
        self.device = device
        self.seconds: float | None = None

    def __enter__(self):
        if self.device.startswith("cuda"):
            self._start, self._end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            self._start.record()
        else:
            synchronize(self.device)
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.device.startswith("cuda"):
            self._end.record()
            self._end.synchronize()
            self.seconds = self._start.elapsed_time(self._end) / 1000
        else:
            synchronize(self.device)
            self.seconds = time.perf_counter() - self._t0
        return False


def reset_peak(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()


def memory(device: str) -> dict | None:
    """Current and peak allocated and reserved memory in GiB (CUDA only)."""
    if not device.startswith("cuda"):
        return None
    return {
        "allocated_gib": torch.cuda.memory_allocated() / GIB,
        "reserved_gib": torch.cuda.memory_reserved() / GIB,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / GIB,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / GIB,
    }


def release(device: str) -> None:
    gc.collect()
    euler.empty_cache(device)


def allocated(device: str) -> float | None:
    return torch.cuda.memory_allocated() / GIB if device.startswith("cuda") else None


# Weights, models and inputs


def stat_record(path: Path) -> dict:
    s = path.stat()
    return {"size": s.st_size, "mtime_ns": s.st_mtime_ns, "inode": s.st_ino}


def wan_files(component: str) -> list[str]:
    return [rel for rel in wan_weights.registry()["files"] if rel.split("/")[0] == component]


def check_verified(ctx: Context, component: str) -> None:
    """Refuse a Wan component unless this session's weights step hashed it and its files are unchanged since."""
    if ctx.dry_run:
        return
    verified = read_result(ctx, f"weights_{ctx.session}")["wan_stat"]
    for rel in wan_files(component) + ["model_index.json"]:
        if rel not in verified:
            raise RuntimeError(f"{rel} was not verified in session {ctx.session}")
        if stat_record(ctx.weights / rel) != verified[rel]:
            raise RuntimeError(f"{rel} changed after it was verified")


def load_transformer(ctx: Context):
    if ctx.dry_run:
        return euler.tiny_transformer(torch.bfloat16, ctx.device, hsa_checks.TINY_CONFIG)
    from diffusers import WanTransformer3DModel

    check_verified(ctx, "transformer")
    return WanTransformer3DModel.from_pretrained(ctx.weights / "transformer", torch_dtype=torch.bfloat16).to(ctx.device).eval()


def load_vae(ctx: Context):
    from diffusers import AutoencoderKLWan

    if ctx.dry_run:
        torch.manual_seed(euler.SEED)
        return AutoencoderKLWan(base_dim=8, z_dim=16, num_res_blocks=1).to(ctx.device).eval()
    check_verified(ctx, "vae")
    return AutoencoderKLWan.from_pretrained(ctx.weights / "vae", torch_dtype=torch.float32).to(ctx.device).eval()


def smoke_texts() -> dict[str, str]:
    return {**{p["id"]: p["text"] for p in PROMPTS}, "negative": text_encoding.NEGATIVE_PROMPT}


def tiny_text_encoder(device: str):
    """Dry run: a word-level tokenizer over the smoke texts and a random two-layer UMT5 as wide as the tiny Transformer's text input."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, UMT5Config, UMT5EncoderModel

    words = sorted({w for text in smoke_texts().values() for w in text.split()})
    vocab = {"<pad>": 0, "</s>": 1, "<unk>": 2, **{w: i + 3 for i, w in enumerate(words)}}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", eos_token="</s>", unk_token="<unk>")
    torch.manual_seed(euler.SEED)
    config = UMT5Config(
        vocab_size=len(vocab), d_model=hsa_checks.TINY_CONFIG["text_dim"], d_kv=8, d_ff=64, num_layers=2, num_heads=4,
        relative_attention_num_buckets=8, relative_attention_max_distance=16, dropout_rate=0.0,
    )
    return tokenizer, UMT5EncoderModel(config).to(device, text_encoding.T5_DTYPE).eval()


def load_text_encoder(ctx: Context):
    if ctx.dry_run:
        return tiny_text_encoder(ctx.device)
    from transformers import AutoTokenizer, UMT5EncoderModel

    check_verified(ctx, "text_encoder")
    check_verified(ctx, "tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(ctx.weights / "tokenizer")
    encoder = UMT5EncoderModel.from_pretrained(ctx.weights / "text_encoder", dtype=text_encoding.T5_DTYPE, device_map=ctx.device)
    return tokenizer, encoder.eval()


def load_inputs(ctx: Context, prompt: dict):
    """The saved starting noise and embeddings, after checking them against the noise and encode records."""
    record = read_result(ctx, "noise")["prompts"][prompt["id"]]
    latents = noise.load(ctx.path("inputs", record["file"]), record["tensor_sha256"])  # registered: loaded by hash, never regenerated
    if tuple(latents.shape) != tuple(ctx.latent_shape):
        raise RuntimeError(f"{prompt['id']}: the saved starting noise has shape {tuple(latents.shape)}, this run needs {ctx.latent_shape}")
    encoded = read_result(ctx, "encode")
    embeds = load_file(ctx.path("inputs", encoded["file"]))
    width = hsa_checks.TINY_CONFIG["text_dim"] if ctx.dry_run else 4096
    for name in (prompt["id"], "negative"):
        if sha256(embeds[name]) != encoded["embeddings"][name]["sha256"]:
            raise RuntimeError(f"{name}: the saved embedding does not match its recorded hash")
        if tuple(embeds[name].shape) != (1, text_encoding.MAX_SEQUENCE_LENGTH, width):
            raise RuntimeError(f"{name}: the saved embedding has shape {tuple(embeds[name].shape)}, this run needs (1, {text_encoding.MAX_SEQUENCE_LENGTH}, {width})")
    return latents.to(ctx.device), embeds[prompt["id"]].to(ctx.device), embeds["negative"].to(ctx.device)


def c4_coverage(ctx: Context, pixel_frames: int | None = None) -> tuple[torch.Tensor, dict]:
    """The registered synthetic C4 mask: a centered disk on every frame, mapped to token coverage."""
    frames, height, width = ctx.size
    frames = pixel_frames or frames
    radius = DISK_RADIUS * height / 480
    cy, cx = (height - 1) / 2, (width - 1) / 2
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
    m = token_coverage(disk.expand(frames, height, width))
    info = {
        "radius_px": radius, "center_xy": [cx, cy], "pixel_frames": frames, "disk_pixels": int(disk.sum()),
        "tokens_with_coverage": int((m > 0).sum()), "coverage_sum": float(m.sum()), "coverage_sha256": sha256(m),
    }
    return m, info


# Runs and comparisons


def run(ctx: Context, transformer, condition: str, prompt: dict, inputs, *, keep_trajectory: bool, lam=None, coverage=None):
    latents, prompt_embeds, negative_embeds = inputs
    return run_condition(
        condition, transformer, prompt_id=PROMPT_NUMBERS[prompt["id"]], latents=latents, seed=prompt["seed"],
        prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, guidance_scale=GUIDANCE, shift=SHIFT,
        preset=PRESET.name, lam=lam, coverage=coverage, keep_trajectory=keep_trajectory,
    )


def to_cpu(trajectory: list[torch.Tensor]) -> list[torch.Tensor]:
    return [x.cpu() for x in trajectory]


def drop(result) -> None:
    """Release an HSA run's KV caches (about 11 GiB at full resolution) and its device trajectory."""
    if result.hsa is not None:
        result.hsa.caches.clear()
    result.hsa = None
    result.trajectory = None


def active_counts_ok(result) -> dict:
    """T7 at full resolution: every iteration, layer and branch ran the schedule's active count, with the same IDs in both branches."""
    hsa, schedule = result.hsa, result.schedule
    expected = schedule.active_counts()
    layers = len({layer for _, layer, _, _ in hsa.log})
    counts = len(hsa.log) == len(expected) * layers * 2 and all(count == expected[n] for n, _, _, count in hsa.log)
    same_ids = all(torch.equal(hsa.logged_ids[(n, "cond")], hsa.logged_ids[(n, "uncond")]) for n in range(len(expected)))
    return {"counts_match": counts, "same_ids_both_branches": same_ids, "log_entries": len(hsa.log), "active_counts": expected}


def oracle_trajectory(transformer, inputs, labels: torch.Tensor, group_iterations: torch.Tensor) -> list[torch.Tensor]:
    latents, prompt_embeds, negative_embeds = inputs
    reference = hsa_checks.MaskedDenseReference(transformer, prompt_embeds, negative_embeds, GUIDANCE, Schedule(labels, group_iterations))
    return to_cpu(euler.custom_trajectory(reference, latents, SIGMAS, TIMESTEPS))


def compare(ours: list[torch.Tensor], ref: list[torch.Tensor]) -> dict:
    stats = euler.compare_trajectories(ours, ref)
    stats["first_divergent_step"] = hsa_checks.first_divergence(ours, ref)
    return stats


def relative(diff: float, floor: float) -> dict:
    """A deviation in units of the repeat-run noise floor measured on the same device."""
    if diff == 0:
        return {"ratio": 0.0}
    if floor == 0:
        return {"ratio": None, "note": "the repeat runs were bitwise identical, so the noise floor is zero and the ratio is undefined"}
    return {"ratio": diff / floor}


def decode(ctx: Context, vae, latents: torch.Tensor, precision: str = VAE_CONV_FP32_PRECISION) -> tuple[np.ndarray, float]:
    """Frames (frames, H, W, 3) in [0, 1] as WanPipeline outputs them, and the VAE decode time."""
    from diffusers.video_processor import VideoProcessor

    with Timer(ctx.device) as t:
        video = decode_latents(vae, latents.to(ctx.device), conv_fp32_precision=precision)
    frames = VideoProcessor(vae_scale_factor=8).postprocess_video(video, output_type="np")[0]
    return frames, t.seconds


def frame_stats(a: np.ndarray, b: np.ndarray) -> dict:
    return {
        "decoded_psnr_db": wan_real.psnr(a, b),
        "decoded_max_abs_diff": float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max()),
        "uint8_identical": bool(np.array_equal(wan_real.to_uint8(a), wan_real.to_uint8(b))),
    }


def save_frames(frames: np.ndarray, folder: Path) -> dict:
    """Lossless PNG frames and their hashes; ``uint8_sha256`` is the hash of the stacked uint8 array."""
    folder.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for i, frame in enumerate(frames):
        data = wan_real.png_bytes(frame)
        (folder / f"frame_{i:03d}.png").write_bytes(data)
        hashes[f"frame_{i:03d}.png"] = hashlib.sha256(data).hexdigest()
    return {"folder": str(folder), "png_sha256": hashes, "uint8_sha256": hashlib.sha256(wan_real.to_uint8(frames).tobytes()).hexdigest()}


def load_frames(folder: Path, record: dict) -> np.ndarray:
    """uint8 frames from PNGs, checked against the record written by save_frames."""
    from PIL import Image

    paths = sorted(Path(folder).glob("frame_*.png"))
    if sorted(p.name for p in paths) != sorted(record["png_sha256"]):
        raise RuntimeError(f"{folder}: the frame files differ from the record")
    for p in paths:
        if hashlib.sha256(p.read_bytes()).hexdigest() != record["png_sha256"][p.name]:
            raise RuntimeError(f"{p}: hash differs from the record")
    frames = np.stack([np.asarray(Image.open(p).convert("RGB")) for p in paths])
    if hashlib.sha256(frames.tobytes()).hexdigest() != record["uint8_sha256"]:
        raise RuntimeError(f"{folder}: decoded frames differ from the record")
    return frames


def dense_path(ctx: Context, prompt: dict) -> Path:
    return ctx.scratch / f"dense_{prompt['id']}.safetensors"


def load_dense(ctx: Context, prompt: dict, generated: dict) -> list[torch.Tensor]:
    stacked = load_file(dense_path(ctx, prompt))["trajectory"]
    if sha256(stacked) != generated["prompts"][prompt["id"]]["dense_trajectory_sha256"]:
        raise RuntimeError(f"{prompt['id']}: the saved dense trajectory does not match its recorded hash")
    return list(stacked.unbind(0))


# Session steps


POD_ENV = ("RUNPOD_POD_ID", "RUNPOD_POD_HOSTNAME", "RUNPOD_GPU_COUNT", "RUNPOD_CPU_COUNT", "RUNPOD_DC_ID")
WORKSPACE_DIRS = (
    "/workspace/venv", "/workspace/uv-cache", "/workspace/python", "/workspace/cuda-13.0", "/workspace/hf",
    "/workspace/Density_Project/weights", "/workspace/Density_Project/results/checks/smoke_test", "/workspace/smoke_scratch",
)
FACTS = Path("/workspace/setup/facts.txt")


def version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", text)[:2])


def nvcc(path: Path) -> str | None:
    r = run_command([str(path), "--version"])
    return r.get("stdout") or r.get("error")


def nvidia() -> dict:
    full = run_command(["nvidia-smi"])
    query = run_command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap,mig.mode.current", "--format=csv,noheader"])
    listing = run_command(["nvidia-smi", "-L"])
    found = re.search(r"CUDA Version:\s*([\d.]+)", full.get("stdout", ""))
    return {
        "nvidia_smi": full.get("stdout") or full.get("error"),
        "query": query.get("stdout") or query.get("error"),
        "devices": listing.get("stdout") or listing.get("error"),
        "driver_cuda_version": found.group(1) if found else None,
    }


def torch_cuda() -> dict:
    if not torch.cuda.is_available():
        return {"available": False}
    devices = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        devices.append({"name": p.name, "capability": f"{p.major}.{p.minor}", "total_memory_gib": p.total_memory / GIB, "multiprocessors": p.multi_processor_count})
    try:
        nccl = ".".join(map(str, torch.cuda.nccl.version()))
    except Exception as error:  # informational only
        nccl = f"unavailable: {type(error).__name__}"
    return {"available": True, "devices": devices, "arch_list": torch.cuda.get_arch_list(), "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "nccl": nccl}


def installed_packages() -> dict:
    from importlib import metadata

    out = {}
    for dist in metadata.distributions():
        name = dist.metadata["Name"]
        direct = dist.read_text("direct_url.json")
        commit = json.loads(direct).get("vcs_info", {}).get("commit_id") if direct else None
        out[name] = f"{dist.version} (git {commit})" if commit else dist.version
    return dict(sorted(out.items(), key=lambda kv: kv[0].lower()))


def lock_differences(packages: dict) -> list[str]:
    """Installed packages whose version differs from uv.lock (packages the lock lists only for other platforms are absent, not different)."""
    import tomli

    locked = {p["name"]: p["version"] for p in tomli.loads((ROOT / "uv.lock").read_text())["package"] if "version" in p}
    norm = lambda name: re.sub(r"[-_.]+", "-", name).lower()
    installed = {norm(k): v.split(" ")[0] for k, v in packages.items()}
    return sorted(f"{name}: installed {installed[norm(name)]}, locked {version}" for name, version in locked.items() if norm(name) in installed and installed[norm(name)] != version)


def system_info() -> dict:
    def read(path):
        p = Path(path)
        return p.read_text().strip() if p.exists() else None

    meminfo = {}
    if Path("/proc/meminfo").exists():
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            if key in ("MemTotal", "MemAvailable"):
                meminfo[key] = int(value.split()[0]) * 1024 / GIB
    cpu_model = None
    if Path("/proc/cpuinfo").exists():
        cpu_model = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), None)
    os_release = read("/etc/os-release")
    pretty = re.search(r'PRETTY_NAME="([^"]+)"', os_release or "")
    return {
        "os": pretty.group(1) if pretty else platform.platform(),
        "kernel": platform.release(),
        "glibc": " ".join(platform.libc_ver()),
        "cpu_model": cpu_model or platform.processor(),
        "cpu_count": os.cpu_count(),
        "cpu_affinity": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "cgroup_cpu_max": read("/sys/fs/cgroup/cpu.max"),
        "memory_gib": meminfo or None,
        "cgroup_memory_max": read("/sys/fs/cgroup/memory.max"),
        "gcc": first_line(["gcc", "--version"]),
        "uv": first_line(["uv", "--version"]),
        "python_executable": sys.executable,
    }


def disk_usage(per_folder: bool) -> dict | None:
    """Usage of /workspace; per folder (du, slow on a network volume) only when asked."""
    if not Path("/workspace").exists():
        return None
    total, used, free = shutil.disk_usage("/workspace")
    mount = next((line for line in Path("/proc/mounts").read_text().splitlines() if " /workspace " in line), None) if Path("/proc/mounts").exists() else None
    du = {}
    for d in WORKSPACE_DIRS if per_folder else ():
        if Path(d).exists():
            out = run_command(["du", "-sb", d], timeout=900).get("stdout", "")
            du[d] = int(out.split()[0]) / GIB if out else None
    return {"workspace_total_gib": total / GIB, "workspace_used_gib": used / GIB, "workspace_free_gib": free / GIB, "mount": mount, "du_gib": du or "not measured in this session"}


def bootstrap_facts() -> dict:
    if not FACTS.exists():
        return {}
    return dict(line.split("=", 1) for line in FACTS.read_text().splitlines() if "=" in line)


def detectron2_build() -> dict:
    import importlib.util

    spec = importlib.util.find_spec("detectron2._C")
    if spec is None:
        return {"extension": None, "cuobjdump_archs": []}
    tool = Path(os.environ.get("CUDA_HOME", "/workspace/cuda-13.0")) / "bin" / "cuobjdump"
    elf = run_command([str(tool), "--list-elf", spec.origin])
    try:
        import detectron2._C  # noqa: F401  (after torch, which provides the CUDA runtime)

        imports = True
    except Exception as error:  # an import failure is recorded, then judged below
        imports = f"{type(error).__name__}: {error}"
    return {"extension": spec.origin, "sha256": file_sha256(spec.origin), "cuobjdump_archs": sorted(set(re.findall(r"sm_(\d+)", elf.get("stdout", "")))), "imports": imports}


def stack_checks(cuda: bool) -> dict:
    """Imports and one CUDA op from each GPU library the pipeline uses."""
    import importlib.util

    import onnxruntime
    import sam2
    import torchvision

    out = {
        "sam2": {"file": sam2.__file__, "cuda_extension": importlib.util.find_spec("sam2._C") is not None},
        "onnxruntime": {"version": onnxruntime.__version__, "providers": onnxruntime.get_available_providers()},
        "torchvision": torchvision.__version__,
    }
    if cuda:
        boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=torch.float32, device="cuda")
        out["torchvision_nms_cuda"] = torchvision.ops.nms(boxes, torch.tensor([0.9, 0.8, 0.7], device="cuda"), 0.5).tolist()
    return out


def credentials_present() -> dict:
    """Presence only, never values: RunPod injects a pod-scoped RUNPOD_API_KEY; GitHub credentials must be absent."""
    home = Path.home()
    return {
        "RUNPOD_API_KEY": "RUNPOD_API_KEY" in os.environ,
        "github_env": [k for k in ("GITHUB_TOKEN", "GH_TOKEN") if k in os.environ],
        "git_credentials_file": (home / ".git-credentials").exists(),
        "gh_config": (home / ".config" / "gh").exists(),
    }


def step_env(ctx: Context) -> dict:
    env = euler.environment()
    record = {
        "session": ctx.session,
        "utc": utc_now(),
        "hostname": socket.gethostname(),
        "pod_env": {k: os.environ.get(k) for k in POD_ENV if k in os.environ},
        "runpod_env_names": sorted(k for k in os.environ if k.startswith("RUNPOD_")),
        "credentials_present": credentials_present(),
        "environment": env,
        "nvidia": nvidia(),
        "nvcc": {"toolkit": nvcc(Path(os.environ.get("CUDA_HOME", "/workspace/cuda-13.0")) / "bin" / "nvcc"), "image": nvcc(Path("/usr/local/cuda/bin/nvcc"))},
        "torch_cuda": torch_cuda(),
        "determinism": determinism.settings(),
        "system": system_info(),
        "packages": installed_packages(),
        "detectron2": detectron2_build(),
        "stack": stack_checks(ctx.cuda),
        "bootstrap": bootstrap_facts(),
        "disk": disk_usage(per_folder=ctx.session == "A"),
        "env_vars": {k: os.environ.get(k) for k in ("CUBLAS_WORKSPACE_CONFIG", "CUDA_HOME", "TORCH_CUDA_ARCH_LIST", "UV_PROJECT_ENVIRONMENT", "UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR", "HF_HOME", "HF_XET_CHUNK_CACHE_SIZE_BYTES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "POD_GPU", "POD_CREATED_UTC")},
        "threads": {"torch_intra_op": torch.get_num_threads(), "torch_inter_op": torch.get_num_interop_threads()},
    }
    record["lock_differences"] = lock_differences(record["packages"])
    stop = []
    if not ctx.dry_run:
        if env["git_sha"] is None:
            stop.append("no git SHA: git is missing or rejects the repository's ownership")
        elif env["git_dirty"]:
            stop.append("the working tree differs from the commit")
        if record["lock_differences"]:
            stop.append(f"packages differ from uv.lock: {record['lock_differences']}")
        if not torch.cuda.is_available():
            stop.append("CUDA is not available")
        else:
            driver = record["nvidia"]["driver_cuda_version"]
            if driver is None or version_tuple(driver) < (13, 0):
                stop.append(f"the host driver supports CUDA {driver}; torch 2.14.1's CUDA 13.0 build needs 13.0 or newer")
            if not (torch.version.cuda or "").startswith("13.0"):
                stop.append(f"torch was built for CUDA {torch.version.cuda}, not 13.0")
            if record["stack"].get("torchvision_nms_cuda") != [0, 2]:
                stop.append(f"torchvision NMS on CUDA returned {record['stack'].get('torchvision_nms_cuda')}")
            build = record["detectron2"]
            if not {"90", "120"} <= set(build["cuobjdump_archs"]) or build.get("imports") is not True:
                stop.append(f"the Detectron2 extension is not built for sm_90 and sm_120 or does not import: {build}")
        if record["stack"]["sam2"]["cuda_extension"]:
            stop.append("sam-2 has a CUDA extension; the registered build (SAM2_BUILD_CUDA=0) has none")
        if record["credentials_present"]["github_env"] or record["credentials_present"]["git_credentials_file"] or record["credentials_present"]["gh_config"]:
            stop.append("GitHub credentials are present on the pod")
    record["stop"] = stop
    return record


SESSION_WAN = {"A": None, "B": ("transformer", "vae"), "C": (), "D": ("transformer", "vae"), "E": (), "F": ()}  # None: every registered file


def step_weights(ctx: Context) -> dict:
    """Hash every file this session loads: Wan2.1 at the pinned revision, the mask weights, and the Mac's real-model inputs."""
    if ctx.dry_run:
        return {"dry_run": "tiny random models; no weight files are loaded", "wan_stat": {}}
    from density.masks.weights import WEIGHTS, verified_path

    out = {"wan": None, "wan_stat": {}, "masks": {}, "wan_real_inputs": None}
    components = SESSION_WAN[ctx.session]
    if components != ():
        out["wan"] = wan_weights.verify(ctx.weights, components)
        out["wan_stat"] = {rel: stat_record(ctx.weights / rel) for rel in out["wan"]["files"]}
    start = time.perf_counter()
    for name, weight in WEIGHTS.items():
        verified_path(name)  # raises on a missing file or a size or hash mismatch
        out["masks"][name] = {"path": weight.path, "size": weight.size, "sha256": weight.sha256}
    out["masks_seconds"] = time.perf_counter() - start
    if ctx.session in ("A", "B", "D"):
        recorded = json.loads(hsa_checks.WAN_REAL_JSON.read_text())["artifacts_sha256"]
        checked = {}
        for rel in ("wan_real_mps/inputs.pt", "wan_real_mps/r1_frames/frame_008.png"):
            got = file_sha256(euler.REPORT_DIR / rel)
            if got != recorded[rel]:
                raise RuntimeError(f"{rel} does not match wan_real_mps.json")
            checked[rel] = got
        out["wan_real_inputs"] = checked
    return out


def step_noise(ctx: Context) -> dict:
    """The one-time generation of the starting noise (density.noise): saved, and identified from then on by its hash."""
    out = {"recipe": SMOKE["starting_noise"], "machine": platform.machine(), "prompts": {}}
    for p in PROMPTS:
        latents = noise.generate(p["seed"], ctx.latent_shape)
        name = f"noise_{p['id']}.safetensors"
        digest = noise.save(latents, ctx.path("inputs", name))
        out["prompts"][p["id"]] = {"seed": p["seed"], "shape": list(latents.shape), "file": name, "tensor_sha256": digest, "file_sha256": file_sha256(ctx.path("inputs", name))}
    return out


def step_noise_mac(ctx: Context) -> dict:
    """The same seeds generated on this machine, for the cross-device comparison (reported, not judged)."""
    return {
        "machine": platform.machine(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "prompts": {p["id"]: {"seed": p["seed"], "tensor_sha256": sha256(noise.generate(p["seed"], ctx.latent_shape))} for p in PROMPTS},
    }


def step_encode(ctx: Context) -> dict:
    meminfo = system_info()["memory_gib"]
    start = time.perf_counter()
    tokenizer, encoder = load_text_encoder(ctx)
    load_seconds = time.perf_counter() - start
    texts = smoke_texts()
    reset_peak(ctx.device)
    with Timer(ctx.device) as t:
        embeds = text_encoding.encode(texts, tokenizer, encoder, ctx.device)
    peak = memory(ctx.device)
    lengths = {name: int(tokenizer(text, padding="max_length", max_length=text_encoding.MAX_SEQUENCE_LENGTH, truncation=True, add_special_tokens=True, return_attention_mask=True, return_tensors="pt").attention_mask.sum()) for name, text in texts.items()}
    params: dict[str, int] = {}
    for p in encoder.parameters():
        params[str(p.dtype).removeprefix("torch.")] = params.get(str(p.dtype).removeprefix("torch."), 0) + p.numel()
    ctx.path("inputs").mkdir(parents=True, exist_ok=True)
    name = "embeddings.safetensors"
    save_file({k: v.contiguous() for k, v in embeds.items()}, ctx.path("inputs", name))
    del encoder
    release(ctx.device)
    records = {}
    for key, e in embeds.items():
        rows = int(e[0].abs().amax(dim=-1).gt(0).sum())
        records[key] = {"shape": list(e.shape), "dtype": str(e.dtype).removeprefix("torch."), "sha256": sha256(e), "tokens": lengths[key], "nonzero_rows": rows, "text": texts[key]}
    stop = [f"{k}: {r['nonzero_rows']} nonzero rows for {r['tokens']} tokens" for k, r in records.items() if r["nonzero_rows"] != r["tokens"]]
    return {
        "file": name,
        "file_sha256": file_sha256(ctx.path("inputs", name)),
        "embeddings": records,
        "settings": {"t5_dtype": "bf16", "max_sequence_length": text_encoding.MAX_SEQUENCE_LENGTH, "ftfy": text_encoding.FTFY_REGISTERED, "wan_config": text_encoding.WAN_CONFIG_SOURCE, "method": "diffusers WanPipeline._get_t5_prompt_embeds"},
        "encoder_parameters": params,
        "seconds": {"load": load_seconds, "encode_three_texts": t.seconds},
        "memory": peak,
        "host_memory_gib_before_load": meminfo,
        "stop": stop,
    }


def step_pytest(ctx: Context) -> dict:
    args = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"] + (["tests/test_environment.py"] if ctx.dry_run else [])
    start = time.perf_counter()
    r = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    log = ctx.path("logs", f"pytest_{ctx.session}.txt")
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(r.stdout + "\n" + r.stderr)
    lines = [line for line in r.stdout.strip().splitlines() if line.strip()]
    summary = lines[-1] if lines else ""
    return {"returncode": r.returncode, "summary": summary, "seconds": time.perf_counter() - start, "log": str(log), "log_sha256": file_sha256(log), "stop": [] if r.returncode == 0 else [f"pytest failed: {summary}"]}


# Milestones 1-3 on CUDA: which failed rows are same-device measurements and which are failures

COMPARISON_ROWS = {
    "m1": {"synthetic", "pipeline"},
    "m2": {"T1 all groups full vs dense", "T2 all active, layer by layer", "T3 fast path vs oracle", "oracle all groups full vs dense", "T4 synthetic, full size"},
    "m3": {"registry: C1 equals the dense loop", "T8 run uses guided v at n = 0..K-1", "T10 lambda = 0 equals C3", "T11 empty mask equals C3", "T17 equal active counts and budgets across C3-C6 and A1"},
}


def numeric_only(row: dict) -> bool:
    """True when a comparison row failed only on numeric equality, not on counts, labels, budgets or controls."""
    name = row.get("test") or row.get("check")
    if name == "pipeline":
        return bool(row.get("informative"))
    if name == "T4 synthetic, full size":
        return round(row["measured_fraction"], 6) == row["registered_fraction"]
    if name == "T8 run uses guided v at n = 0..K-1":
        return all(row["alternatives_differ"].values())
    if name in ("T10 lambda = 0 equals C3", "T11 empty mask equals C3"):
        return bool(row["labels_equal"] and row["active_ids_equal"])
    if name.startswith("T17"):
        return bool(row["counts_equal_every_iteration_layer_branch"] and len(set(row["token_step_fractions"].values())) == 1 and row["relabel_refused"])
    return True


BRIEF_FIELDS = ("test", "check", "model", "device", "dtype", "seed", "preset", "lam", "bitwise_equal", "max_abs_diff", "first_divergent_step", "tolerance", "frames_judged", "frames_matching_record")


def classify(milestone: str, report: dict, device: str) -> dict:
    """Failed rows split into measurements (same-device numeric differences) and failures (everything else)."""
    measurements, failures, unjudged = [], [], []
    for row in report["results"]:
        brief = {k: row[k] for k in BRIEF_FIELDS if k in row}
        if row.get("passed") is None:
            unjudged.append(brief)
            continue
        if row["passed"]:
            continue
        name = row.get("test") or row.get("check")
        if name in COMPARISON_ROWS[milestone] and row.get("device") == device and numeric_only(row):
            measurements.append(brief)
        else:
            failures.append(brief)
    return {"measurements": measurements, "failures": failures, "unjudged": unjudged}


def step_milestones(ctx: Context) -> dict:
    from density.checks import allocation as allocation_checks

    devices, weights = [ctx.device], (None if ctx.dry_run else ctx.weights)
    checks = {
        "m1": (lambda: euler.run_all(devices), euler.markdown_report, "euler_vs_diffusers"),
        "m2": (lambda: hsa_checks.run_all(devices, weights, ctx.device), hsa_checks.markdown_report, "hsa_mechanics"),
        "m3": (lambda: allocation_checks.run_all(devices, weights, ctx.device, ctx.path("m3")), allocation_checks.markdown_report, "allocation_rules"),
    }
    out, stop = {"device": ctx.device}, []
    for name, (fn, markdown, stem) in checks.items():
        start = time.perf_counter()
        report = fn()
        folder = ctx.path(name)
        write_json(folder / f"{stem}.json", report)
        (folder / f"{stem}.md").write_text(markdown(report))
        if name == "m3" and report.get("real_model"):
            (folder / "allocation_rules.sha256").write_text("".join(f"{h}  {p}\n" for p, h in report["real_model"]["artifacts_sha256"].items()))
        summary = classify(name, report, ctx.device)
        out[name] = {
            "passed": report["passed"], "num_judged": report["num_judged"], "num_failed": report["num_failed"],
            "acceptance_gate": report.get("acceptance_gate"), "seconds": time.perf_counter() - start,
            "report": str(folder / f"{stem}.json"), "report_sha256": file_sha256(folder / f"{stem}.json"), **summary,
        }
        if summary["failures"]:
            stop += [f"{name}: {r.get('test') or r.get('check')} ({r.get('device')}, {r.get('dtype', '')})" for r in summary["failures"]]
            break  # stop at the first failure
    out["unjudged_note"] = "T1's frame-hash row and allocation's R1-frames row compare against hashes recorded on MPS; on another device they are reported, not judged."
    out["stop"] = stop
    return out


def step_generate(ctx: Context) -> dict:
    transformer, vae = load_transformer(ctx), load_vae(ctx)
    resident = memory(ctx.device)
    first = load_inputs(ctx, PROMPTS[0])
    euler.custom_trajectory(WanGuidedVelocity(transformer, first[1], first[2], GUIDANCE), first[0], SIGMAS[:3], TIMESTEPS[:2])
    decode(ctx, vae, first[0][:, :, :2])
    out = {"resident_after_load": resident, "warmup": "2 dense steps and one decode, discarded", "text_encoder_loaded": False, "prompts": {}}
    blocks = []
    for p in PROMPTS:
        inputs = load_inputs(ctx, p)
        velocity = WanGuidedVelocity(transformer, inputs[1], inputs[2], GUIDANCE)
        runs = []
        for _ in range(2):
            release(ctx.device)
            reset_peak(ctx.device)
            with Timer(ctx.device) as t:
                trajectory = euler.custom_trajectory(velocity, inputs[0], SIGMAS, TIMESTEPS)
            peak = memory(ctx.device)
            trajectory = to_cpu(trajectory)
            frames, decode_seconds = decode(ctx, vae, trajectory[-1])
            runs.append({"trajectory": trajectory, "frames": frames, "denoise_seconds": t.seconds, "decode_seconds": decode_seconds, "memory_denoise": peak})
        frames_again, _ = decode(ctx, vae, runs[0]["trajectory"][-1])
        repeat = compare(runs[1]["trajectory"], runs[0]["trajectory"])
        frames = frame_stats(runs[1]["frames"], runs[0]["frames"])
        stacked = torch.stack(runs[0]["trajectory"])
        ctx.scratch.mkdir(parents=True, exist_ok=True)
        save_file({"trajectory": stacked}, dense_path(ctx, p))
        reference = save_frames(runs[0]["frames"], ctx.path("references", p["id"]))
        record = {
            "seed": p["seed"],
            "repeat": {k: repeat[k] for k in ("bitwise_equal", "max_abs_diff", "final_max_abs_diff", "final_rel_diff", "first_divergent_step", "per_step_max_abs_diff")},
            "repeat_frames": frames,
            "vae_repeat_max_abs_diff": float(np.abs(frames_again - runs[0]["frames"]).max()),
            "runs": [{k: r[k] for k in ("denoise_seconds", "decode_seconds", "memory_denoise")} for r in runs],
            "timing_note": "trajectory captured at every step (one device-to-host copy per step); the perf step times runs without capture",
            "final_latents_sha256": [sha256(r["trajectory"][-1]) for r in runs],
            "dense_trajectory_sha256": sha256(stacked),
            "reference": reference,
            "structure": wan_real.structure_stats(runs[0]["frames"]),
            "finite": bool(all(torch.isfinite(x).all() for x in runs[0]["trajectory"]) and np.isfinite(runs[0]["frames"]).all()),
        }
        if frames["decoded_psnr_db"] < PSNR_THRESHOLD_DB:
            blocks.append(f"{p['id']}: repeated dense run decoded PSNR {frames['decoded_psnr_db']:.2f} dB is below {PSNR_THRESHOLD_DB:g} dB")
        out["prompts"][p["id"]] = record
        del runs
    out["blocks_pilot"] = blocks
    out["stop"] = [f"{pid}: non-finite output" for pid, r in out["prompts"].items() if not r["finite"]]
    return out


def step_sdpa(ctx: Context) -> dict:
    """The attention backend PyTorch selects for the real shapes: dense (every query) and an HSA-50 iteration (high group only)."""
    transformer = load_transformer(ctx)
    latents, prompt_embeds, negative_embeds = load_inputs(ctx, PROMPTS[0])
    t0, t5 = TIMESTEPS[0].to(ctx.device), TIMESTEPS[FIRST_SKIP].to(ctx.device)
    dense = WanGuidedVelocity(transformer, prompt_embeds, negative_embeds, GUIDANCE)
    with determinism.record_sdpa_choices() as calls:
        dense(latents, t0, 0)
    out = {"dense": {"choices": determinism.summarize_sdpa(calls)}}
    schedule = hsa_checks.hsa50_style(ctx.num_tokens, 0)
    velocity = wan_hsa_velocity(transformer, prompt_embeds, negative_embeds, GUIDANCE, schedule)
    velocity(latents, t0, 0)  # every token: fills each cache row
    with determinism.record_sdpa_choices() as calls:
        velocity(latents, t5, FIRST_SKIP)
    out["hsa_subset"] = {"iteration": FIRST_SKIP, "active_tokens": int(schedule.active[FIRST_SKIP].sum()), "choices": determinism.summarize_sdpa(calls)}
    for key, fn in (("dense", lambda: dense(latents, t0, 0)), ("hsa_subset", lambda: velocity(latents, t5, FIRST_SKIP))):
        if not ctx.cuda:
            out[key]["kernels"] = "the profiler probe needs CUDA"
            continue
        try:
            names = determinism.cuda_kernel_names(fn)
            out[key]["kernel_backends"] = determinism.classify_kernels(names)
            out[key]["attention_kernels"] = [n for n in names if any(pattern.search(n) for pattern in determinism.SDPA_KERNEL_PATTERNS.values())]
        except Exception as error:  # profiling is corroboration only
            out[key]["kernels"] = f"profiler failed: {type(error).__name__}: {error}"
    vae = load_vae(ctx)
    with determinism.record_sdpa_choices() as calls:
        decode_latents(vae, latents[:, :, :2])
    out["vae"] = {"choices": determinism.summarize_sdpa(calls), "note": "a 2-latent-frame decode; the VAE's attention runs in fp32"}
    out["registered_backends"] = [b.name for b in GENERATION_BACKENDS]
    backends = lambda key: sorted({c["backend"] for c in out[key]["choices"]})
    out["same_backend_dense_and_subset"] = backends("dense") == backends("hsa_subset")
    out["backends"] = {"dense": backends("dense"), "hsa_subset": backends("hsa_subset")}
    return out


def step_hsa_dense(ctx: Context) -> dict:
    """Protocol section 11, all HSA groups dense: HSA with both groups at every step against the dense loop."""
    generated = read_result(ctx, "generate")
    transformer, vae = load_transformer(ctx), load_vae(ctx)
    out, blocks, stop = {"prompts": {}}, [], []
    for p in PROMPTS:
        inputs = load_inputs(ctx, p)
        dense = load_dense(ctx, p, generated)
        schedule = hsa_checks.all_full(ctx.num_tokens)
        velocity = wan_hsa_velocity(transformer, inputs[1], inputs[2], GUIDANCE, schedule)
        with Timer(ctx.device) as t:
            trajectory = to_cpu(euler.custom_trajectory(velocity, inputs[0], SIGMAS, TIMESTEPS))
        hsa = velocity.active_velocity_fn.hsa
        counts = all(count == ctx.num_tokens for _, _, _, count in hsa.log)
        fraction = velocity.token_step_fraction()
        hsa.caches.clear()
        del velocity, hsa
        release(ctx.device)
        frames, _ = decode(ctx, vae, trajectory[-1])
        dense_frames, _ = decode(ctx, vae, dense[-1])
        stats = compare(trajectory, dense)
        floor = generated["prompts"][p["id"]]["repeat"]["max_abs_diff"]
        record = {
            **{k: stats[k] for k in ("bitwise_equal", "max_abs_diff", "final_max_abs_diff", "final_rel_diff", "first_divergent_step", "per_step_max_abs_diff")},
            **frame_stats(frames, dense_frames),
            "relative_to_dense_repeat_noise": relative(stats["max_abs_diff"], floor),
            "dense_repeat_max_abs_diff": floor,
            "every_token_every_iteration": counts,
            "token_step_fraction": fraction,
            "denoise_seconds": t.seconds,
        }
        if record["decoded_psnr_db"] < PSNR_THRESHOLD_DB:
            blocks.append(f"{p['id']}: all HSA groups dense, decoded PSNR {record['decoded_psnr_db']:.2f} dB is below {PSNR_THRESHOLD_DB:g} dB")
        if not counts or fraction != 1.0:
            stop.append(f"{p['id']}: with every group at full steps the active counts or the token-step fraction ({fraction}) are wrong")
        out["prompts"][p["id"]] = record
    out["blocks_pilot"], out["stop"] = blocks, stop
    return out


# T3 at full resolution, with an investigation when the fast path and the oracle differ

MODULE_KINDS = ("attn1", "attn2", "ffn")


def module_names(transformer) -> list[str]:
    return [f"blocks.{i}.{kind}" for i in range(len(transformer.blocks)) for kind in MODULE_KINDS] + ["proj_out"]


def _clone_args(args):
    return tuple(tuple(t.clone() for t in a) if isinstance(a, tuple) else (a.clone() if isinstance(a, torch.Tensor) else a) for a in args)


FINGERPRINT_SEEDS = (20261006, 20261007)


def fingerprint(x: torch.Tensor) -> str:
    """A 128-bit fingerprint of a tensor's bytes, computed on its own device.

    Two sums of the bytes, read as int64 words, against seeded random int64 weights. Integer arithmetic
    wraps and is associative, so the result does not depend on reduction order. Equal tensors always give
    equal fingerprints; different ones collide with negligible probability. It replaces a CPU SHA-256 in the
    T3 investigation, which had to copy every full-size module output to the host.
    """
    dense = torch.empty(x.numel(), dtype=x.dtype, device=x.device)
    dense.copy_(x.detach().reshape(-1))  # a fresh stride-1 buffer: expanded (stride-0) inputs cannot be viewed as bytes
    flat = dense.view(torch.uint8)
    nbytes = flat.numel()
    if flat.numel() % 8:
        flat = torch.cat([flat, flat.new_zeros(8 - flat.numel() % 8)])
    words = flat.view(torch.int64)
    parts = []
    for seed in FINGERPRINT_SEEDS:
        g = torch.Generator(device=words.device).manual_seed(seed)
        weights = torch.randint(-(2**62), 2**62, words.shape, generator=g, device=words.device, dtype=torch.int64)
        parts.append(int((words * weights).sum().item()) & (2**64 - 1))
    return f"{nbytes:x}:{parts[0]:016x}{parts[1]:016x}"


def _capture(ctx, transformer, make_velocity, latents, step: int, active: torch.Tensor, keep=None):
    """Run to iteration ``step``; there, hash every module call's active-row input and output, and keep one call's full arguments."""
    names = module_names(transformer)
    state = {"n": None}
    calls, kept = [], {}

    def hook(name):
        def fn(module, args, out):
            if state["n"] != step:
                return
            branch = ("cond", "uncond")[len(calls) // len(names)]
            x, y = args[0], out
            if x.shape[1] != active.numel():  # the oracle passes every token
                x, y = x[:, active], y[:, active]
            calls.append({"module": name, "branch": branch, "input": fingerprint(x), "output": fingerprint(y)})
            if keep == (name, branch):
                kept["args"] = _clone_args(args)
        return fn

    velocity = make_velocity()

    def tracked(x, t, n):
        state["n"] = n
        return velocity(x, t, n)

    handles = [transformer.get_submodule(n).register_forward_hook(hook(n)) for n in names]
    try:
        trajectory = euler.custom_trajectory(tracked, latents, SIGMAS[: step + 2], TIMESTEPS[: step + 1])
    finally:
        for h in handles:
            h.remove()
    return calls, kept.get("args"), velocity, to_cpu(trajectory)


def _first_linear(module):
    return next(m for m in module.modules() if isinstance(m, torch.nn.Linear))


def _diff(a: torch.Tensor, b: torch.Tensor) -> dict:
    return {"equal": torch.equal(a, b), "elements_differing": int((a != b).sum()), "elements": a.numel(), "max_abs_diff": (a.float() - b.float()).abs().max().item()}


@torch.no_grad()
def investigate(ctx: Context, transformer, inputs, labels: torch.Tensor, group_iterations: torch.Tensor, step: int, fast_trajectory) -> dict:
    """Find the first module whose active-row output differs between the fast path and the oracle at iteration ``step``,
    then test whether that module's kernels alone reproduce the difference on identical inputs (query-count dependence)."""
    latents, prompt_embeds, negative_embeds = inputs
    schedule = Schedule(labels, group_iterations)
    active = schedule.active_ids(step).to(ctx.device)
    make_oracle = lambda: hsa_checks.MaskedDenseReference(transformer, prompt_embeds, negative_embeds, GUIDANCE, schedule)
    make_fast = lambda: wan_hsa_velocity(transformer, prompt_embeds, negative_embeds, GUIDANCE, schedule)

    oracle_calls, _, oracle, _ = _capture(ctx, transformer, make_oracle, latents, step, active)
    del oracle
    release(ctx.device)
    fast_calls, _, fast, replay = _capture(ctx, transformer, make_fast, latents, step, active)
    fast.active_velocity_fn.hsa.caches.clear()
    del fast
    release(ctx.device)
    evidence = {
        "step": step,
        "active_tokens": int(active.numel()),
        "replay_matches_run": all(torch.equal(a, b) for a, b in zip(replay, fast_trajectory[: step + 1])),
        "module_calls": len(fast_calls),
    }
    index = next((i for i, (f, o) in enumerate(zip(fast_calls, oracle_calls)) if f["output"] != o["output"]), None)
    if index is None:
        evidence["module"] = "none: every module output matches on the active rows"
        return evidence
    name, branch = fast_calls[index]["module"], fast_calls[index]["branch"]
    evidence.update(module=name, branch=branch, call_index=index, module_inputs_equal=fast_calls[index]["input"] == oracle_calls[index]["input"])

    _, oracle_args, oracle, _ = _capture(ctx, transformer, make_oracle, latents, step, active, keep=(name, branch))
    _, fast_args, fast, _ = _capture(ctx, transformer, make_fast, latents, step, active, keep=(name, branch))
    module = transformer.get_submodule(name)
    if name.endswith("attn1"):
        layer = int(name.split(".")[1])
        q_oracle = project_qkv(module, oracle_args[0], oracle_args[3])[0]
        q_fast = project_qkv(module, fast_args[0], fast_args[3])[0]
        k_oracle, v_oracle = oracle.caches[(branch, layer)].read()
        k_fast, v_fast = fast.active_velocity_fn.hsa._cache(branch, layer).read()
        evidence["q_equal_on_active_rows"] = torch.equal(q_oracle[:, active], q_fast)
        evidence["kv_equal"] = torch.equal(k_oracle, k_fast) and torch.equal(v_oracle, v_fast)
        # The fast path's own logic, checked without arithmetic: the RoPE rows it gathers, and the cache rows of skipped tokens.
        skipped = torch.ones(k_oracle.shape[1], dtype=torch.bool, device=active.device)
        skipped[active] = False
        evidence["rope_rows_equal"] = all(torch.equal(f, o[:, active]) for f, o in zip(fast_args[3], oracle_args[3]))
        evidence["skipped_cache_rows_equal"] = torch.equal(k_oracle[:, skipped], k_fast[:, skipped]) and torch.equal(v_oracle[:, skipped], v_fast[:, skipped])
        with generation_attention(ctx.device), determinism.record_sdpa_choices() as calls:
            every = hsa_checks._sdpa(q_oracle, k_oracle, v_oracle)[:, active]
            only = hsa_checks._sdpa(q_fast, k_fast, v_fast)
        evidence["sdpa_all_queries_vs_active_queries"] = _diff(every, only)
        evidence["sdpa_choices"] = determinism.summarize_sdpa(calls)
        sdpa_alone = evidence["q_equal_on_active_rows"] and evidence["kv_equal"] and not evidence["sdpa_all_queries_vs_active_queries"]["equal"]
        structure_ok = evidence["rope_rows_equal"] and evidence["skipped_cache_rows_equal"]
    else:
        with generation_attention(ctx.device), determinism.record_sdpa_choices() as calls:
            every = module(*oracle_args)[:, active]
            only = module(*fast_args)
        evidence["module_all_rows_vs_active_rows"] = _diff(every, only)
        evidence["sdpa_choices"] = determinism.summarize_sdpa(calls)
        sdpa_alone = False
        structure_ok = True  # a token-local stock module: no HSA logic of its own beyond the rows it is given
    evidence["inputs_equal_on_active_rows"] = torch.equal(oracle_args[0][:, active], fast_args[0])
    linear = _first_linear(module)
    evidence["gemm_all_rows_vs_active_rows"] = {"linear": f"{type(linear).__name__}({linear.in_features}, {linear.out_features})", **_diff(linear(oracle_args[0])[:, active], linear(fast_args[0]))}
    # On identical inputs, a kernel whose result depends on how many rows (queries) it is given explains the difference:
    # the module's first GEMM, its attention kernel with identical q, K and V, or (for a token-local module) the module itself.
    gemm_alone = not evidence["gemm_all_rows_vs_active_rows"]["equal"]
    module_alone = "module_all_rows_vs_active_rows" in evidence and not evidence["module_all_rows_vs_active_rows"]["equal"]
    evidence["hsa_structure_ok"] = bool(structure_ok)
    explained = evidence["inputs_equal_on_active_rows"] and structure_ok
    evidence["kernel_alone_reproduces"] = bool(explained and (gemm_alone or sdpa_alone or module_alone))
    evidence["explained_by"] = [name for name, hit in (("GEMM row-count dependence", gemm_alone), ("attention query-count dependence", sdpa_alone), ("module row-count dependence", module_alone)) if hit] if explained else []
    fast.active_velocity_fn.hsa.caches.clear()
    del oracle, fast
    release(ctx.device)
    return evidence


def verdict(stats: dict, evidence: dict | None) -> str:
    if stats["bitwise_equal"]:
        return "bitwise"
    if stats["first_divergent_step"] < FIRST_SKIP:
        return "bug or nondeterminism: the paths differ before the first skipped iteration"
    if evidence and evidence.get("kernel_alone_reproduces") and evidence.get("replay_matches_run"):
        return f"measurement: on identical inputs, {evidence['module']} gives different results for different row counts ({', '.join(evidence.get('explained_by') or ['kernel'])})"
    if evidence and evidence.get("hsa_structure_ok") is False:
        return "bug to investigate: the fast path's gathered RoPE rows or its skipped tokens' cache rows differ from the oracle's"
    return "bug to investigate: the difference is not explained by the kernels' row-count dependence"


def t3_prompt(ctx: Context, transformer, vae, p: dict, inputs, coverage: torch.Tensor) -> dict:
    runs = {}
    for name, condition, kwargs in (("C3", "C3", {}), ("C3 repeat", "C3", {}), ("C4", "C4", {"lam": C4_LAMBDA, "coverage": {ROI: coverage}})):
        result = run(ctx, transformer, condition, p, inputs, keep_trajectory=True, **kwargs)
        runs[name] = {
            "trajectory": to_cpu(result.trajectory), "labels": result.record.labels, "group_iterations": result.record.group_iterations,
            "fraction": result.record.token_step_fraction, "counts": active_counts_ok(result), "record": result.record.summary(),
        }
        drop(result)
        del result
        release(ctx.device)
        if name in ("C3", "C4"):
            runs[f"{name} oracle"] = {"trajectory": oracle_trajectory(transformer, inputs, runs[name]["labels"], runs[name]["group_iterations"])}
            release(ctx.device)
    frames = {name: decode(ctx, vae, r["trajectory"][-1])[0] for name, r in runs.items()}
    # Lossless frames of the fast path and the oracle, for the LPIPS check in Milestone 5 (protocol section 11).
    saved = {name: save_frames(frames[name], ctx.path("t3_frames", p["id"], name.replace(" ", "_"))) for name in ("C3", "C3 oracle", "C4", "C4 oracle")}
    repeat = compare(runs["C3 repeat"]["trajectory"], runs["C3"]["trajectory"])
    same_counts = runs["C3"]["counts"]["active_counts"] == runs["C4"]["counts"]["active_counts"] and all(runs[c]["counts"]["counts_match"] for c in ("C3", "C4"))
    out = {
        "fast_path_repeat": {**{k: repeat[k] for k in ("bitwise_equal", "max_abs_diff", "first_divergent_step")}, "labels_equal": torch.equal(runs["C3 repeat"]["labels"], runs["C3"]["labels"]), **frame_stats(frames["C3 repeat"], frames["C3"])},
        "c4_labels_changed_vs_c3": int((runs["C4"]["labels"] != runs["C3"]["labels"]).sum()),
        "c3_c4_identical_active_counts": same_counts,
        "frames": saved,
        "conditions": {},
    }
    stop, bugs = [], []
    if not same_counts:
        stop.append(f"{p['id']}: C3 and C4 active-token counts differ (protocol section 8 requires them equal)")
    for name in ("C3", "C4"):
        stats = compare(runs[name]["trajectory"], runs[f"{name} oracle"]["trajectory"])
        evidence = None
        if not stats["bitwise_equal"]:
            evidence = investigate(ctx, transformer, inputs, runs[name]["labels"], runs[name]["group_iterations"], stats["first_divergent_step"], runs[name]["trajectory"])
        fraction = runs[name]["fraction"]
        fast_u8, oracle_u8 = (wan_real.to_uint8(frames[k]).astype(np.float64) / 255 for k in (name, f"{name} oracle"))
        record = {
            **{k: stats[k] for k in ("bitwise_equal", "max_abs_diff", "final_max_abs_diff", "final_rel_diff", "first_divergent_step", "per_step_max_abs_diff")},
            **frame_stats(frames[name], frames[f"{name} oracle"]),
            "decoded_psnr_db_uint8": wan_real.psnr(fast_u8, oracle_u8),
            "per_frame_psnr_db": [wan_real.psnr(a, b) for a, b in zip(frames[name], frames[f"{name} oracle"])],
            "token_step_fraction": fraction,
            "active_counts": runs[name]["counts"],
            "allocation": runs[name]["record"],
            "evidence": evidence,
            "verdict": verdict(stats, evidence),
        }
        out["conditions"][name] = record
        if not runs[name]["counts"]["counts_match"] or not runs[name]["counts"]["same_ids_both_branches"]:
            stop.append(f"{p['id']} {name}: active counts or IDs do not match the schedule")
        if ctx.num_tokens == PRESET.num_tokens and abs(fraction - REGISTERED_FRACTION) > BUDGET_TOLERANCE:
            stop.append(f"{p['id']} {name}: token-step fraction {fraction:.6f} is not within {BUDGET_TOLERANCE} of {REGISTERED_FRACTION}")
        if record["verdict"].startswith("bug"):
            bugs.append(f"{p['id']} {name}: {record['verdict']}")
    out["stop"], out["bugs"] = stop, bugs
    return out


def step_t3(ctx: Context) -> dict:
    """Protocol T3 at full resolution on HSA-50: the fast path against the masked-dense oracle, for C3 and for C4 with the synthetic mask.

    Dense repeats are bitwise on CUDA, so repeat-run noise is zero and is no yardstick here: the deviation
    is reported as decoded PSNR, and the frames are saved losslessly for the LPIPS check in Milestone 5.
    The structural checks in ``investigate`` are the bug test.
    """
    transformer, vae = load_transformer(ctx), load_vae(ctx)
    coverage, mask_info = c4_coverage(ctx)
    out = {"preset": PRESET.name, "lambda": C4_LAMBDA, "c4_mask": mask_info, "oracle_labels": "each oracle uses the labels its fast-path run assigned", "prompts": {}}
    stop, bugs = [], []
    for p in PROMPTS:
        inputs = load_inputs(ctx, p)
        try:
            record = t3_prompt(ctx, transformer, vae, p, inputs, coverage)
        except torch.cuda.OutOfMemoryError:
            record = None  # retried below, once the exception (and the caches its frames hold) is gone
        if record is None:
            release(ctx.device)
            frames = 4 * (REDUCED_LATENT_FRAMES - 1) + 1
            reduced = (inputs[0][:, :, :REDUCED_LATENT_FRAMES].contiguous(), inputs[1], inputs[2])
            small, _ = c4_coverage(ctx, frames)
            record = t3_prompt(ctx, transformer, vae, p, reduced, small)
            record["reduced"] = {"latent_frames": REDUCED_LATENT_FRAMES, "pixel_frames": frames, "reason": "the full-resolution oracle did not fit in memory"}
        stop += record.pop("stop")
        bugs += record.pop("bugs")
        out["prompts"][p["id"]] = record
    out["stop"], out["bugs"] = stop, bugs
    return out


def step_perf(ctx: Context) -> dict:
    """Peak memory and CUDA-event timing for C1, C3 and C4, twice each, with T5 never loaded and the caches released before decoding."""
    transformer, vae = load_transformer(ctx), load_vae(ctx)
    coverage, _ = c4_coverage(ctx)
    p = PROMPTS[0]
    inputs = load_inputs(ctx, p)
    resident = memory(ctx.device)
    euler.custom_trajectory(WanGuidedVelocity(transformer, inputs[1], inputs[2], GUIDANCE), inputs[0], SIGMAS[:3], TIMESTEPS[:2])
    warm = wan_hsa_velocity(transformer, inputs[1], inputs[2], GUIDANCE, hsa_checks.hsa50_style(ctx.num_tokens, 0))
    euler.custom_trajectory(warm, inputs[0], SIGMAS[: FIRST_SKIP + 2], TIMESTEPS[: FIRST_SKIP + 1])
    warm.active_velocity_fn.hsa.caches.clear()
    del warm
    release(ctx.device)
    decode(ctx, vae, inputs[0][:, :, :2])
    rows = []
    for repeat in (1, 2):
        for condition in ("C1", "C3", "C4"):
            release(ctx.device)
            base = allocated(ctx.device)
            reset_peak(ctx.device)
            kwargs = {"lam": C4_LAMBDA, "coverage": {ROI: coverage}} if condition == "C4" else {}
            start = time.perf_counter()
            with Timer(ctx.device) as t:
                result = run(ctx, transformer, condition, p, inputs, keep_trajectory=False, **kwargs)
            wall = time.perf_counter() - start
            denoise_memory = memory(ctx.device)
            latents, fraction = result.latents, result.record.token_step_fraction
            drop(result)
            del result
            release(ctx.device)
            after_release = allocated(ctx.device)
            reset_peak(ctx.device)
            with Timer(ctx.device) as td:
                video = decode_latents(vae, latents)
            decode_memory = memory(ctx.device)
            del video, latents
            release(ctx.device)
            rows.append({
                "condition": condition, "repeat": repeat, "token_step_fraction": fraction,
                "denoise_seconds": t.seconds, "denoise_wall_seconds": wall, "decode_seconds": td.seconds, "end_to_end_seconds": t.seconds + td.seconds,
                "memory_before_gib": base, "memory_denoise": denoise_memory, "memory_after_release_gib": after_release, "memory_decode": decode_memory,
                "caches_released": None if base is None else after_release - base < 0.05,
            })
    summary = {}
    for condition in ("C1", "C3", "C4"):
        mine = [r for r in rows if r["condition"] == condition]
        mean = lambda key: sum(r[key] for r in mine) / len(mine)
        summary[condition] = {
            "denoise_seconds_mean": mean("denoise_seconds"), "decode_seconds_mean": mean("decode_seconds"), "end_to_end_seconds_mean": mean("end_to_end_seconds"),
            "denoise_seconds_spread": max(r["denoise_seconds"] for r in mine) - min(r["denoise_seconds"] for r in mine),
            "peak_allocated_denoise_gib": None if not ctx.cuda else max(r["memory_denoise"]["peak_allocated_gib"] for r in mine),
            "peak_reserved_denoise_gib": None if not ctx.cuda else max(r["memory_denoise"]["peak_reserved_gib"] for r in mine),
            "peak_allocated_decode_gib": None if not ctx.cuda else max(r["memory_decode"]["peak_allocated_gib"] for r in mine),
            "peak_reserved_decode_gib": None if not ctx.cuda else max(r["memory_decode"]["peak_reserved_gib"] for r in mine),
            "token_step_fraction": mine[0]["token_step_fraction"],
        }
    c3, c4 = summary["C3"]["denoise_seconds_mean"], summary["C4"]["denoise_seconds_mean"]
    stop = [] if all(r["caches_released"] in (True, None) for r in rows) else ["memory was not back to its pre-run level before decoding: caches were not released"]
    return {
        "prompt": p["id"],
        "resident_after_load": resident,
        "text_encoder_loaded": False,
        "warmup": "2 dense steps, 6 HSA-50 steps and one decode, discarded",
        "runs": rows,
        "summary": summary,
        "c4_overhead_vs_c3": {"seconds": c4 - c3, "relative": (c4 - c3) / c3, "timing_spread_seconds": max(summary["C3"]["denoise_seconds_spread"], summary["C4"]["denoise_seconds_spread"])},
        "re_seconds": summary["C1"]["end_to_end_seconds_mean"],
        "re_definition": "one dense 40-step generation (protocol section 12): CUDA-event denoising plus VAE decoding, with precomputed text embeddings",
        "stop": stop,
    }


def step_vae_precision(ctx: Context) -> dict:
    """The VAE decode with cuDNN's default TF32 convolutions against IEEE fp32: the time cost and the size of the difference."""
    generated = read_result(ctx, "generate")
    vae = load_vae(ctx)
    out = {"default_conv_fp32_precision": VAE_CONV_FP32_PRECISION, "prompts": {}}
    for i, p in enumerate(PROMPTS):
        latents = load_dense(ctx, p, generated)[-1]
        if i == 0:  # warm up both modes on two latent frames
            decode(ctx, vae, latents[:, :, :2])
            decode(ctx, vae, latents[:, :, :2], precision="ieee")
        reset_peak(ctx.device)
        default_frames, default_seconds = decode(ctx, vae, latents)
        default_memory = memory(ctx.device)
        reset_peak(ctx.device)
        ieee_frames, ieee_seconds = decode(ctx, vae, latents, precision="ieee")
        ieee_memory = memory(ctx.device)
        out["prompts"][p["id"]] = {"tf32_seconds": default_seconds, "ieee_seconds": ieee_seconds, "tf32_memory": default_memory, "ieee_memory": ieee_memory, **frame_stats(ieee_frames, default_frames)}
    out["note"] = "a measurement for the registered choice: the VAE decodes with TF32 convolutions (density.wan.VAE_CONV_FP32_PRECISION)"
    return out


def frame_luma(frames: np.ndarray) -> np.ndarray:
    """Mean Rec. 709 luma of each uint8 frame, on [0, 1]."""
    rgb = frames.astype(np.float64) / 255
    return (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]).mean(axis=(1, 2))


def frame0_luma(frames: np.ndarray) -> dict:
    y = frame_luma(frames)
    rest = y[1:]
    std = float(rest.std())
    return {
        "frame0": float(y[0]), "frames_1_to_4_mean": float(y[1:5].mean()), "other_frames_mean": float(rest.mean()), "other_frames_std": std,
        "other_frames_max": float(rest.max()), "difference": float(y[0] - rest.mean()), "z": float((y[0] - rest.mean()) / std) if std > 0 else None,
        "brighter_than_mean": bool(y[0] > rest.mean()), "brighter_than_every_other_frame": bool(y[0] > rest.max()), "per_frame": y.tolist(),
    }


def contact_sheet(rows: dict[str, np.ndarray], path: Path) -> dict:
    from PIL import Image, ImageDraw

    any_frames = next(iter(rows.values()))
    picks = list(range(0, any_frames.shape[0], 8))
    height, width = any_frames.shape[1] // 4, any_frames.shape[2] // 4
    label = 90
    sheet = Image.new("RGB", (label + width * len(picks), (height + 14) * len(rows)), "white")
    draw = ImageDraw.Draw(sheet)
    for r, (name, frames) in enumerate(rows.items()):
        top = r * (height + 14)
        draw.text((6, top + height // 2), name, fill="black")
        for c, f in enumerate(picks):
            sheet.paste(Image.fromarray(frames[f]).resize((width, height), Image.BOX), (label + c * width, top + 14))
            if r == 0:
                draw.text((label + c * width + 4, 0), f"frame {f}", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return {"path": str(path), "sha256": file_sha256(path), "frames": picks}


def step_sheet(ctx: Context) -> dict:
    generated = read_result(ctx, "generate")
    rows, luma = {}, {}
    for p in PROMPTS:
        frames = load_frames(ctx.path("references", p["id"]), generated["prompts"][p["id"]]["reference"])
        rows[p["id"]] = frames
        luma[p["id"]] = frame0_luma(frames)
    return {"contact_sheet": contact_sheet(rows, ctx.path("references", "contact_sheet.png")), "frame0_luma": luma}


# Session C: masks on the slice


def reference_frames(ctx: Context, p: dict) -> tuple[np.ndarray, dict]:
    if ctx.dry_run:
        from density.checks.masks import verified_frames_record
        from density.masks.clips import load_frames as load_clip

        clip = DRY_RUN_CLIPS[p["id"]]
        return load_clip(clip), {"dry_run_stand_in": clip, "frames_sha256": verified_frames_record(clip)["frames_sha256"]}
    reference = read_result(ctx, "generate")["prompts"][p["id"]]["reference"]
    return load_frames(ctx.path("references", p["id"]), reference), {"frames_uint8_sha256": reference["uint8_sha256"]}


def region_differences(a, b) -> dict:
    out = {}
    for name in sorted(set(a.regions) | set(b.regions)):
        if name not in a.regions or name not in b.regions:
            out[name] = "computed in only one run"
            continue
        x, y = a.regions[name].mask, b.regions[name].mask
        union = int((x | y).sum())
        out[name] = {"pixels_differing": int((x ^ y).sum()), "frames_differing": int((x ^ y).reshape(x.shape[0], -1).any(axis=1).sum()), "iou": (int((x & y).sum()) / union) if union else 1.0}
    return out


def sam_probe(ctx: Context, frames: np.ndarray) -> dict:
    """SAM 2.1's attention backend on its fp32 path, from a short propagation over the first 9 frames. Never fatal."""
    out = {}
    try:
        from density.masks.sam import SAM2Propagator

        clip = np.ascontiguousarray(frames[:9])
        h, w = clip.shape[1:3]
        prompts = {0: {0: np.array([0.3 * w, 0.3 * h, 0.7 * w, 0.7 * h], dtype=np.float32)}}
        propagator = SAM2Propagator(clip, ctx.device)
        try:
            with determinism.record_sdpa_choices() as calls:
                propagator.propagate(prompts, clip.shape[0])
            out["choices"] = determinism.summarize_sdpa(calls)
            if ctx.cuda:
                try:
                    out["kernel_backends"] = determinism.classify_kernels(determinism.cuda_kernel_names(lambda: propagator.propagate(prompts, clip.shape[0])))
                except Exception as error:
                    out["kernels"] = f"profiler failed: {type(error).__name__}: {error}"
        finally:
            propagator.close()
    except Exception as error:
        out["error"] = f"{type(error).__name__}: {error}"
    return out


def step_masks(ctx: Context) -> dict:
    """The mask pipeline on both smoke references on the slice: per-component timings, a repeat run with fresh models, and sheets."""
    from density.checks import masks as masks_checks
    from density.masks.pipeline import Models, metadata, process_reference, save

    masks_checks.SHEETS_DIR = ctx.path("mask_sheets")
    out, stop = {"device": ctx.device, "references": {}}, []
    frames = None
    for p in PROMPTS:
        frames, source = reference_frames(ctx, p)
        stratum = p["mask_stratum"]
        models = Models(ctx.device)
        first = process_reference(frames, stratum, models, seed=p["seed"])
        fresh = Models(ctx.device)
        second = process_reference(frames, stratum, fresh, seed=p["seed"])
        same = masks_checks.mask_digest(first) == masks_checks.mask_digest(second)
        with determinism.ieee_fp32_convolutions():  # for the TF32 decision; the same models as the repeat run, already built
            ieee = process_reference(frames, stratum, fresh, seed=p["seed"])
        ieee_same = masks_checks.mask_digest(ieee) == masks_checks.mask_digest(first)
        sheets = masks_checks.write_sheets({"id": p["id"], "stratum": stratum}, frames, first)
        stored = save(first, ctx.path("masks", p["id"]), {"reference": p["id"], "source": source, "device": ctx.device, "versions": {"packages": masks_checks.package_versions(), "git": euler.git_state(), "gpu": euler.gpu_info()}, "registered": masks_checks.jsonable(masks_checks.registered_values())})
        meta = metadata(first)
        out["references"][p["id"]] = {
            "stratum": stratum, "seed": p["seed"], "source": source,
            "seconds": first.seconds, "repeat_seconds": second.seconds, "sam": first.sam,
            "model_build_seconds": models.build_seconds, "repeat_model_build_seconds": fresh.build_seconds,
            "eligibility": meta["eligibility"], "pipeline_decision": masks_checks.pipeline_decision(stratum, first.eligibility),
            "roi_usable_frames": meta["roi_usable_frames"], "regions": meta["regions"], "contact": meta["contact"],
            "repeat_identical": same, "repeat_differences": {} if same else region_differences(first, second),
            "ieee_fp32_convolutions": {
                "seconds": ieee.seconds, "sam": ieee.sam, "identical_to_default": ieee_same, "differences": {} if ieee_same else region_differences(first, ieee),
                "default_seconds_without_build": second.seconds["total"] - sum(fresh.build_seconds.values()),
                "note": "run with the repeat run's models, already built; compare it with default_seconds_without_build (the repeat run minus its model builds)",
            },
            "digest": masks_checks.mask_digest(first), "sheets": sheets, "stored": stored,
        }
        print(f"[smoke] masks {p['id']}: {first.seconds['total']:.0f} s, repeat identical {same}", flush=True)
    out["sam_attention"] = sam_probe(ctx, frames)
    out["stop"] = stop
    return out


def step_mask_check(ctx: Context) -> dict:
    """The stock-clip mask check on this device from the verified conformed frames (no ffmpeg on the pod)."""
    folder = ctx.path("mask_check")
    args = [sys.executable, "-m", "density.checks.masks", "--device", ctx.device, "--frames-only", "--report-dir", str(folder)] + (["--logic-only"] if ctx.dry_run else [])
    report_path = folder / "mask_pipeline.json"
    report_path.unlink(missing_ok=True)
    start = time.perf_counter()
    r = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    log = ctx.path("logs", "mask_check.txt")
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(r.stdout + "\n" + r.stderr)
    if not report_path.exists():
        raise RuntimeError(f"the mask check wrote no report (exit {r.returncode}); see {log}")
    report = json.loads(report_path.read_text())
    return {
        "command": " ".join(args[1:]), "returncode": r.returncode, "seconds": time.perf_counter() - start, "log": str(log),
        "report": str(report_path), "report_sha256": file_sha256(report_path),
        "passed": report["passed"], "num_judged": report["num_judged"], "num_failed": report["num_failed"],
        "failed_rows": [row for row in report["results"] if row["passed"] is False],
        "t26_rows": [row for row in report["results"] if row["test"].startswith("T26")],
        "timings": report.get("timings"),
        "determinism": report.get("determinism"),
        "note": "judged rows that fail on this device are differences to record, not session failures: the masks are expected to differ from the Mac's",
    }


def decoded_regions(path: Path) -> dict | None:
    from density.masks.storage import load_masks

    return load_masks(path.parent) if path.exists() else None


POD_B_FLASH = Path("/workspace/smoke_scratch/dense_smoke-2_flash.safetensors")  # pod B's flash-pinned dense run (scripts/pod/diagnostics/cross_process.py)


def step_crossmachine(ctx: Context) -> dict:
    """Session D: this pod's flash results against pod B's flash diagnostic, bitwise across machines, with both hosts."""
    def host(env_path: Path) -> dict | None:
        if not env_path.exists():
            return None
        env = json.loads(env_path.read_text())["result"]
        return {"hostname": env["hostname"], "gpu": env["torch_cuda"].get("devices"), "driver": env["nvidia"]["query"], "driver_cuda": env["nvidia"]["driver_cuda_version"], "cpu": env["system"]["cpu_model"], "kernel": env["system"]["kernel"]}

    out = {"pod_b": host(OUT_DIR / "env_B.json"), "this_pod": host(ctx.path(f"env_{ctx.session}.json")), "checks": {}}
    prompt = next(p for p in PROMPTS if p["id"] == "smoke-2")
    if POD_B_FLASH.exists() and not ctx.dry_run:
        theirs = list(load_file(POD_B_FLASH)["trajectory"].unbind(0))
        mine = load_dense(ctx, prompt, read_result(ctx, "generate"))
        stats = compare(mine, theirs)
        out["checks"]["dense_smoke_2_flash"] = {k: stats[k] for k in ("bitwise_equal", "max_abs_diff", "first_divergent_step")} | {"pod_b_file": str(POD_B_FLASH), "pod_b_trajectory_sha256": sha256(torch.stack(theirs))}
    else:
        out["checks"]["dense_smoke_2_flash"] = "pod B's flash trajectory is not available here"
    diag = OUT_DIR / "diag_hsa_dense.json"
    if diag.exists() and not ctx.dry_run:
        theirs = json.loads(diag.read_text())["flash_pinned_smoke_2"]["c3_labels_sha256"]
        mine = read_result(ctx, "t3")["prompts"]["smoke-2"]["conditions"]["C3"]["allocation"]["allocation_sha256"]["labels"]
        out["checks"]["c3_labels_smoke_2"] = {"identical": mine == theirs, "this_pod": mine, "pod_b": theirs}
    bitwise = [c.get("bitwise_equal", c.get("identical")) for c in out["checks"].values() if isinstance(c, dict)]
    out["stop"] = [] if all(bitwise) else ["flash results differ between pod B and this pod"]
    return out


def step_mask_compare(ctx: Context) -> dict:
    """The stock-clip masks on this device against the Mac's committed report. Cross-device: reported, not judged."""
    from density.masks.clips import CLIPS_DIR

    pod = json.loads(ctx.path("mask_check", "mask_pipeline.json").read_text())
    mac = json.loads(COMMITTED_MASKS.read_text())
    if not pod.get("clip_results"):
        return {"note": "the mask check ran only its logic tests (dry run); there are no clip masks to compare"}
    clips = {}
    for cid, mine in mac["clip_results"].items():
        theirs = pod["clip_results"].get(cid)
        if theirs is None:
            clips[cid] = "missing from the pod's report"
            continue
        mac_regions = decoded_regions(ctx.scratch / "mac_masks" / cid / "masks" / "masks.json")  # pushed by sync.sh clips
        pod_regions = decoded_regions(CLIPS_DIR / cid / "masks" / "masks.json")
        regions = {}
        for name, digest in mine["stored"]["region_sha256"].items():
            entry = {"identical": theirs["stored"]["region_sha256"].get(name) == digest}
            if not entry["identical"] and mac_regions is not None and pod_regions is not None and name in mac_regions and name in pod_regions:
                x, y = mac_regions[name], pod_regions[name]
                union = int((x | y).sum())
                entry.update(pixels_differing=int((x ^ y).sum()), frames_differing=int((x ^ y).reshape(x.shape[0], -1).any(axis=1).sum()), iou=(int((x & y).sum()) / union) if union else 1.0, mac_pixels=int(x.sum()), pod_pixels=int(y.sum()))
            regions[name] = entry
        clips[cid] = {
            "frames_identical": theirs["frames_sha256"] == mine["frames_sha256"],
            "regions": regions,
            "tracks_identical": theirs["stored"]["track_sha256"] == mine["stored"]["track_sha256"],
            "events_identical": theirs["digest"]["events"] == mine["digest"]["events"],
            "intervals": {"mac": mine["digest"].get("intervals"), "pod": theirs["digest"].get("intervals")},
            "pipeline_decision": {"mac": mine["pipeline_decision"], "pod": theirs["pipeline_decision"]},
            "roi_usable_frames": {"mac": mine["roi_usable_frames"], "pod": theirs["roi_usable_frames"]},
            "seconds_total": {"mac_cpu": mine["seconds"]["total"], "pod": theirs["seconds"]["total"]},
            "pixel_comparison": "per-region pixel differences need both masks.json files" if mac_regions is None or pod_regions is None else "computed",
        }
    rows = {}
    for row in pod["results"]:
        if row["test"].startswith(("T27", "T26")):
            mac_name = re.sub(r"repeated \w+ run", "repeated CPU run", row["test"])  # the Mac's repeat ran on CPU
            committed = next((r for r in mac["results"] if r["test"] == mac_name), None)
            rows[row["test"]] = {"pod": row["passed"], "mac": committed["passed"] if committed else None}
    return {"clips": clips, "judged_rows": rows, "note": "cross-device comparisons are reported, not judged; differences are expected"}


STEPS = {
    "env": step_env, "weights": step_weights, "noise": step_noise, "noise_mac": step_noise_mac, "encode": step_encode,
    "pytest": step_pytest, "milestones": step_milestones, "generate": step_generate, "sdpa": step_sdpa, "hsa_dense": step_hsa_dense,
    "t3": step_t3, "perf": step_perf, "vae_precision": step_vae_precision, "sheet": step_sheet, "crossmachine": step_crossmachine, "masks": step_masks, "mask_check": step_mask_check, "mask_compare": step_mask_compare,
}


def output_name(ctx: Context, name: str) -> str:
    return f"{name}_{ctx.session}" if name in PER_SESSION else name


def step_key(ctx: Context, name: str) -> dict:
    state = euler.git_state()
    deps = {dep: file_sha256(ctx.path(f"{dep}.json")) for dep in DEPENDS.get(name, ()) if ctx.path(f"{dep}.json").exists()}
    return {"git_sha": state["git_sha"], "git_dirty": state["git_dirty"], "device": ctx.device, "dry_run": ctx.dry_run, "deps": deps}


def check_hours(ctx: Context, name: str) -> None:
    """Stop and ask before a step would take the pod past its cap (POD_CREATED_UTC and POD_GPU come from the session script)."""
    created, gpu = os.environ.get("POD_CREATED_UTC"), os.environ.get("POD_GPU")
    if ctx.dry_run or not created or not gpu:
        return
    try:
        created_at = datetime.fromisoformat(created.replace("Z", "+00:00"))
        cap = POD_HOURS_CAP[gpu]
    except (ValueError, KeyError) as error:
        raise CapReached(f"cannot check the hour cap from POD_CREATED_UTC={created!r} and POD_GPU={gpu!r} ({type(error).__name__}: {error})")
    if created_at.tzinfo is None:
        raise CapReached(f"POD_CREATED_UTC={created!r} has no time zone")
    elapsed = (datetime.now(timezone.utc) - created_at).total_seconds() / 3600
    if elapsed + STEP_HOURS.get(name, 0.1) > cap:
        raise CapReached(f"{name} (about {STEP_HOURS.get(name, 0.1):.2f} h) would take the {gpu} pod past its {POD_HOURS_CAP[gpu]:g} h cap ({elapsed:.2f} h so far); ask Chase")


def check_volume(ctx: Context) -> dict | None:
    """Stop before a step once the density volume holds the registered stop (du; df reports the whole storage cluster). Pods only."""
    if ctx.dry_run or not volume.VOLUME.is_dir():
        return None
    return volume.require_room()


def run_step(ctx: Context, name: str) -> int:
    output = output_name(ctx, name)
    path = ctx.path(f"{output}.json")
    key = step_key(ctx, name)
    if path.exists() and not ctx.force:
        previous = json.loads(path.read_text())
        if name in PINNED and previous["key"].get("dry_run") != ctx.dry_run:
            print(f"[smoke] STOP: {path} was written by a {'dry' if previous['key'].get('dry_run') else 'real'} run; use a separate --out", flush=True)
            return 1
        if name in PINNED or previous.get("key") == key:
            print(f"[smoke] {output}: done at {previous['key']['git_sha']}, skipped", flush=True)
            return 0
    try:
        check_hours(ctx, name)
        room = check_volume(ctx)
    except (CapReached, volume.VolumeFull) as error:
        print(f"[smoke] STOP: {error}", flush=True)
        return 3
    log = determinism.WarningLog()
    settings_start = determinism.settings()
    started, start = utc_now(), time.perf_counter()
    print(f"[smoke] {output}: started {started}", flush=True)
    try:
        with log.capture(output):
            result = STEPS[name](ctx)
    except Exception as error:
        oom = isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()
        write_json(ctx.path(f"{output}.error.json"), {
            "step": output, "key": key, "started_utc": started, "failed_utc": utc_now(), "error": f"{type(error).__name__}: {error}",
            "oom": oom, "traceback": traceback.format_exc(), "memory": memory(ctx.device), "warnings": log.summary(),
        })
        traceback.print_exc()
        print(f"[smoke] {output}: FAILED ({'out of memory' if oom else type(error).__name__}); see {output}.error.json", flush=True)
        return 1
    settings_end = determinism.settings()
    changed = {k: [settings_start[k], settings_end[k]] for k in settings_start if settings_start[k] != settings_end[k]}
    if changed:
        print(f"[smoke] {output}: DETERMINISTIC SETTINGS CHANGED during the step: {changed}", flush=True)
    record = {
        "step": output, "key": key, "session": ctx.session, "device": ctx.device, "dry_run": ctx.dry_run,
        "started_utc": started, "finished_utc": utc_now(), "seconds": time.perf_counter() - start,
        "gpu": euler.gpu_info(), "determinism": settings_end, "determinism_at_start": settings_start, "determinism_changed": changed,
        "torch_threads": torch.get_num_threads(), "volume_before": room,
        "warnings": log.summary(), "deterministic_warnings": log.deterministic_warnings(), "result": result,
    }
    write_json(path, record)
    ctx.path(f"{output}.error.json").unlink(missing_ok=True)
    for flag in result.get("blocks_pilot") or []:
        print(f"[smoke] {output}: BLOCKS THE PILOT: {flag}", flush=True)
    for bug in result.get("bugs") or []:
        print(f"[smoke] {output}: BUG TO INVESTIGATE: {bug}", flush=True)
    if result.get("stop"):
        for reason in result["stop"]:
            print(f"[smoke] {output}: STOP: {reason}", flush=True)
        return 1
    print(f"[smoke] {output}: done in {record['seconds']:.0f} s", flush=True)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("step", choices=[*STEPS, "report"])
    parser.add_argument("--session", choices=["A", "B", "C", "D", "E", "F"], default=os.environ.get("SMOKE_SESSION"))
    parser.add_argument("--device", default=None, help="default: cuda when available, else mps, else cpu")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--scratch", type=Path, default=SCRATCH_DIR)
    parser.add_argument("--weights", type=Path, default=wan_real.WEIGHTS_DIR)
    parser.add_argument("--dry-run", action="store_true", help="tiny random models; needs its own --out and --scratch")
    parser.add_argument("--force", action="store_true", help="rerun a step whose JSON is current")
    args = parser.parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    ctx = Context(args.out, args.scratch, device, args.weights, args.session, args.dry_run, args.force)
    if args.step == "report":
        from density.checks.smoke_report import write_report

        return write_report(ctx)
    if args.step in PER_SESSION and ctx.session is None:
        parser.error(f"{args.step} needs --session (or SMOKE_SESSION)")
    determinism.configure()
    return run_step(ctx, args.step)


if __name__ == "__main__":
    raise SystemExit(main())
