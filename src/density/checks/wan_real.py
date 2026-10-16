"""Real-model check on the Mac: dense Wan2.1-T2V-1.3B runs through the custom Euler loop.

Run ``python -m density.checks.wan_real`` after downloading the Transformer and
VAE at ``REVISION`` into ``weights/Wan2.1-T2V-1.3B-Diffusers``. Writes
``results/checks/wan_real_mps.{json,md}`` and frames under
``results/checks/wan_real_mps/``.

The text encoder is not used. Placeholder embeddings stand in for T5: the
conditional branch has small seeded values in its first tokens and zero
padding after, the unconditional branch is all zeros, and both use the full
512-token length. Video content is therefore arbitrary; the check is about
numerics, not prompt fidelity.

Runs, all on MPS with a bf16 Transformer and fp32 VAE:

R1  custom loop.
R2  custom loop again. Its difference from R1 is the MPS run-to-run tolerance.
R3  stock WanPipeline in bf16 (guidance and latents in bf16). Informational.
R4  stock WanPipeline with the fp32-output hook, decoded by the pipeline. Its
    latents must match R1, and its frames must match ``decode_latents`` of R1.

MPS repeatability says nothing about CUDA; the GPU smoke test re-measures it.
"""

import argparse
import hashlib
import io
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKLWan, WanTransformer3DModel
from diffusers.video_processor import VideoProcessor
from PIL import Image, ImageDraw

from density.checks.euler import (
    REPORT_DIR,
    compare_trajectories,
    custom_trajectory,
    environment,
    pipeline_trajectory,
    velocity_is_informative,
)
from density.sampler import flow_match_schedule
from density.wan import WanGuidedVelocity, decode_latents

REPO_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
REVISION = "0fad780a534b6463e45facd96134c9f345acfa5b"
WEIGHTS_DIR = Path("weights") / "Wan2.1-T2V-1.3B-Diffusers"

# Protocol section 5: shift 8 / guidance 6 is the tie-break candidate pair.
SHIFT = 8.0
GUIDANCE = 6.0
STEPS = 40
# Half the protocol's 832 x 480 in each dimension; 17 frames -> 5 latent frames.
VIDEO = {"num_frames": 17, "height": 240, "width": 416}
LATENT_SHAPE = (1, 16, 5, 30, 52)
TEXT_LEN = 512
TEXT_DIM = 4096
PLACEHOLDER_TOKENS = 16
PLACEHOLDER_STD = 0.1
SEED = 20261005
FPS = 16

OUT_DIR = REPORT_DIR / "wan_real_mps"
HASH_FILE = REPORT_DIR / "wan_real_mps.sha256"


def placeholder_inputs() -> dict[str, torch.Tensor]:
    generator = torch.Generator("cpu").manual_seed(SEED)
    latents = torch.randn(LATENT_SHAPE, generator=generator, dtype=torch.float32)
    prompt_embeds = torch.zeros(1, TEXT_LEN, TEXT_DIM)
    prompt_embeds[:, :PLACEHOLDER_TOKENS] = PLACEHOLDER_STD * torch.randn(
        1, PLACEHOLDER_TOKENS, TEXT_DIM, generator=generator
    )
    negative_prompt_embeds = torch.zeros(1, TEXT_LEN, TEXT_DIM)
    return {"latents": latents, "prompt_embeds": prompt_embeds, "negative_prompt_embeds": negative_prompt_embeds}


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def weight_hashes(weights_dir: Path, components: tuple[str, ...] = ("transformer", "vae")) -> dict[str, str]:
    """SHA-256 of the safetensors files of the components the checks load (the Transformer and the VAE)."""
    paths = sorted(p for component in components for p in (weights_dir / component).glob("*.safetensors"))
    return {str(p.relative_to(weights_dir)): file_sha256(p) for p in paths}


def artifact_hashes(out_dir: Path) -> dict[str, str]:
    """SHA-256 of every file the check saved, keyed by path relative to ``REPORT_DIR``."""
    return {str(p.relative_to(REPORT_DIR)): file_sha256(p) for p in sorted(out_dir.rglob("*")) if p.is_file()}


def param_dtypes(model: torch.nn.Module) -> dict[str, int]:
    counts: dict[str, int] = {}
    for p in model.parameters():
        key = str(p.dtype).removeprefix("torch.")
        counts[key] = counts.get(key, 0) + p.numel()
    return counts


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """PSNR in dB for frames in [0, 1]; infinite when identical."""
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)


def structure_stats(frames: np.ndarray) -> dict[str, float]:
    """Mean absolute neighbor differences. Natural video is smooth in space and time; noise is not."""
    return {
        "spatial_mad": float(np.abs(np.diff(frames, axis=-2)).mean()),
        "temporal_mad": float(np.abs(np.diff(frames, axis=-4)).mean()),
    }


def timed(fn):
    torch.mps.synchronize()
    start = time.perf_counter()
    out = fn()
    torch.mps.synchronize()
    return out, time.perf_counter() - start


def to_uint8(frame: np.ndarray) -> np.ndarray:
    return (np.clip(frame, 0, 1) * 255).round().astype(np.uint8)


def png_bytes(frame: np.ndarray) -> bytes:
    """Lossless PNG encoding of one frame in [0, 1]; the saved frames and their recorded hashes use this."""
    buffer = io.BytesIO()
    Image.fromarray(to_uint8(frame)).save(buffer, format="PNG")
    return buffer.getvalue()


def save_media(rows: dict[str, np.ndarray], clip: np.ndarray, out_dir: Path) -> dict[str, str]:
    """Contact sheet (one row per label, every fourth frame), lossless PNG frames, and a GIF preview."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "r1_frames"
    frames_dir.mkdir(exist_ok=True)
    for i, frame in enumerate(clip):
        (frames_dir / f"frame_{i:03d}.png").write_bytes(png_bytes(frame))

    gif_frames = [Image.fromarray(to_uint8(frame)) for frame in clip]
    gif_frames[0].save(out_dir / "r1.gif", save_all=True, append_images=gif_frames[1:], duration=1000 // FPS, loop=0)

    picks = list(range(0, clip.shape[0], 4))
    height, width = clip.shape[1:3]
    label_w = 150
    sheet = Image.new("RGB", (label_w + width * len(picks), height * len(rows)), "white")
    draw = ImageDraw.Draw(sheet)
    for r, (label, frames) in enumerate(rows.items()):
        draw.text((8, r * height + height // 2), label, fill="black")
        for c, idx in enumerate(picks):
            sheet.paste(Image.fromarray(to_uint8(frames[idx])), (label_w + c * width, r * height))
    sheet.save(out_dir / "contact_sheet.png")

    return {
        "contact_sheet": str(out_dir / "contact_sheet.png"),
        "gif": str(out_dir / "r1.gif"),
        "frames": str(frames_dir),
    }


def run(weights_dir: Path = WEIGHTS_DIR, device: str = "mps") -> dict:
    inputs = placeholder_inputs()
    # Start clean so the artifact hashes cover exactly this run's files.
    shutil.rmtree(OUT_DIR, ignore_errors=True)
    OUT_DIR.mkdir(parents=True)
    torch.save(inputs, OUT_DIR / "inputs.pt")
    latents, prompt_embeds, negative_prompt_embeds = (inputs[k].to(device) for k in inputs)

    transformer = WanTransformer3DModel.from_pretrained(weights_dir / "transformer", torch_dtype=torch.bfloat16)
    transformer = transformer.to(device).eval()
    vae = AutoencoderKLWan.from_pretrained(weights_dir / "vae", torch_dtype=torch.float32).to(device).eval()
    processor = VideoProcessor(vae_scale_factor=8)
    to_frames = lambda video: processor.postprocess_video(video, output_type="np")[0]  # (frames, H, W, 3) in [0, 1]

    sigmas, timesteps = flow_match_schedule(STEPS, SHIFT)
    velocity = WanGuidedVelocity(transformer, prompt_embeds, negative_prompt_embeds, GUIDANCE)
    pipe_args = (transformer, SHIFT, STEPS, GUIDANCE, latents, prompt_embeds, negative_prompt_embeds)

    r1, t_r1 = timed(lambda: custom_trajectory(velocity, latents, sigmas, timesteps))
    r2, t_r2 = timed(lambda: custom_trajectory(velocity, latents, sigmas, timesteps))
    (r4, r4_frames), t_r4 = timed(lambda: pipeline_trajectory(*pipe_args, True, video=VIDEO, vae=vae, output_type="np"))
    (r3, r3_frames), t_r3 = timed(lambda: pipeline_trajectory(*pipe_args, False, video=VIDEO, vae=vae, output_type="np"))
    r4_frames, r3_frames = r4_frames[0], r3_frames[0]

    r1_frames = to_frames(decode_latents(vae, r1[-1]))
    r1_frames_again = to_frames(decode_latents(vae, r1[-1]))
    r2_frames = to_frames(decode_latents(vae, r2[-1]))
    noise_frames = to_frames(decode_latents(vae, latents))

    repeat = compare_trajectories(r2, r1)
    tolerance = repeat["max_abs_diff"]
    r4_vs_r1 = compare_trajectories(r4, r1)
    r3_vs_r1 = compare_trajectories(r3, r1)
    vae_repeat = float(np.abs(r1_frames_again - r1_frames).max())
    decode_diff = float(np.abs(r4_frames - r1_frames).max())

    finite = all(torch.isfinite(x).all().item() for x in r1) and bool(np.isfinite(r1_frames).all())
    checks = {
        "finite": {"passed": finite},
        "repeat_run": {
            "passed": None,
            "bitwise_equal": repeat["bitwise_equal"],
            "max_abs_diff": repeat["max_abs_diff"],
            "decoded_psnr_db": psnr(r2_frames, r1_frames),
            "note": "Not required to be bitwise; a nonzero difference becomes the tolerance below.",
        },
        "pipeline_fp32_velocity_latents": {
            "passed": r4_vs_r1["bitwise_equal"] if tolerance == 0 else r4_vs_r1["max_abs_diff"] <= tolerance,
            "bitwise_equal": r4_vs_r1["bitwise_equal"],
            "max_abs_diff": r4_vs_r1["max_abs_diff"],
            "tolerance": tolerance,
        },
        "decode_helper_vs_pipeline": {
            "passed": decode_diff <= vae_repeat,
            "max_abs_diff": decode_diff,
            "tolerance": vae_repeat,
            "decoded_psnr_db": psnr(r4_frames, r1_frames),
        },
    }

    info = {
        "velocity_effects_at_first_steps": velocity_is_informative(velocity, latents, timesteps.to(device)),
        "stock_bf16_drift": {
            "final_max_abs_diff": r3_vs_r1["final_max_abs_diff"],
            "final_rel_diff": r3_vs_r1["final_rel_diff"],
            "decoded_psnr_db": psnr(r3_frames, r1_frames),
            "protocol_threshold_db": 45.0,
        },
        "structure": {
            "r1": structure_stats(r1_frames),
            "r3_stock_bf16": structure_stats(r3_frames),
            "decoded_noise_latent": structure_stats(noise_frames),
        },
        "seconds": {"r1_custom": t_r1, "r2_custom": t_r2, "r3_stock_bf16": t_r3, "r4_pipeline_fp32_velocity": t_r4},
    }

    media = save_media({"R1 custom": r1_frames, "R3 stock bf16": r3_frames, "noise latent": noise_frames}, r1_frames, OUT_DIR)

    judged = [c for c in checks.values() if c["passed"] is not None]
    return {
        "milestone": "Real Wan2.1-T2V-1.3B dense runs through the custom Euler loop (Mac, MPS)",
        "passed": all(c["passed"] for c in judged),
        "checks": checks,
        "info": info,
        "media": media,
        "artifacts_sha256": artifact_hashes(OUT_DIR),
        "environment": environment(),
        "model": {
            "repo_id": REPO_ID,
            "revision": REVISION,
            "weights_sha256": weight_hashes(weights_dir),
            "transformer_param_dtypes": param_dtypes(transformer),
            "vae_dtype": str(vae.dtype).removeprefix("torch."),
        },
        "settings": {
            "device": device,
            "shift": SHIFT,
            "guidance": GUIDANCE,
            "steps": STEPS,
            "video": VIDEO,
            "latent_shape": LATENT_SHAPE,
            "seed": SEED,
            "placeholder_embeddings": {
                "text_len": TEXT_LEN,
                "cond": f"first {PLACEHOLDER_TOKENS} tokens ~ N(0, {PLACEHOLDER_STD}^2), rest zero",
                "uncond": "all zeros",
            },
            "input_sha256": {k: tensor_sha256(v) for k, v in inputs.items()},
            "sigmas": sigmas.tolist(),
        },
    }


def _fmt(value) -> str:
    if isinstance(value, float):
        return "inf" if value == float("inf") else ("0" if value == 0 else f"{value:.4g}")
    return str(value)


def markdown_report(report: dict) -> str:
    env, model, info = report["environment"], report["model"], report["info"]
    lines = [
        "# Real Wan2.1-T2V-1.3B through the custom Euler loop (MPS)",
        "",
        f"**Overall: {'PASS' if report['passed'] else 'FAIL'}**",
        "",
        f"{model['repo_id']} @ `{model['revision']}`; torch {env['torch']}, diffusers {env['diffusers']}, {env['platform']}",
        f"Settings: shift {SHIFT:g}, guidance {GUIDANCE:g}, {STEPS} steps, {VIDEO['width']}x{VIDEO['height']}, "
        f"{VIDEO['num_frames']} frames, placeholder text embeddings.",
        "",
        "| Check | Result | Details |",
        "| --- | --- | --- |",
    ]
    for name, check in report["checks"].items():
        status = "measured" if check["passed"] is None else ("PASS" if check["passed"] else "FAIL")
        details = ", ".join(f"{k} {_fmt(v)}" for k, v in check.items() if k not in ("passed", "note"))
        lines.append(f"| {name} | {status} | {details} |")

    drift = info["stock_bf16_drift"]
    lines += [
        "",
        "## Information only",
        "",
        f"- Stock bf16 pipeline vs custom loop: final latent max abs diff {_fmt(drift['final_max_abs_diff'])} "
        f"(relative {_fmt(drift['final_rel_diff'])}), decoded PSNR {_fmt(drift['decoded_psnr_db'])} dB "
        f"(protocol repeat-run threshold {drift['protocol_threshold_db']:g} dB).",
        "- Neighbor differences (lower is smoother): "
        + "; ".join(
            f"{k}: spatial {_fmt(v['spatial_mad'])}, temporal {_fmt(v['temporal_mad'])}" for k, v in info["structure"].items()
        ),
        "- Velocity at the first two timesteps: "
        + ", ".join(f"{k} {_fmt(v)}" for k, v in info["velocity_effects_at_first_steps"].items()),
        "- Seconds per run: " + ", ".join(f"{k} {v:.1f}" for k, v in info["seconds"].items()),
        f"- Transformer parameter dtypes: {model['transformer_param_dtypes']}",
        "",
        f"Contact sheet: `{report['media']['contact_sheet']}`. MPS repeatability does not carry over to CUDA.",
        "",
        f"Saved inputs, frames, and previews are not versioned; their SHA-256 hashes are in the JSON and in "
        f"`{HASH_FILE.name}` (check with `shasum -a 256 -c {HASH_FILE.name}` from `{REPORT_DIR}`).",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--weights", type=Path, default=WEIGHTS_DIR)
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available")

    report = run(args.weights)
    (REPORT_DIR / "wan_real_mps.json").write_text(json.dumps(report, indent=2))
    (REPORT_DIR / "wan_real_mps.md").write_text(markdown_report(report))
    HASH_FILE.write_text("".join(f"{digest}  {path}\n" for path, digest in report["artifacts_sha256"].items()))
    print(markdown_report(report))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
