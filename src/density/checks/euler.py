"""Milestone check: the custom Euler loop matches diffusers' FlowMatchEulerDiscreteScheduler.

Run ``python -m density.checks.euler`` to write
``results/checks/euler_vs_diffusers.{json,md}``.

Checks, each at the protocol's candidate shift/guidance pairs and both dense
step counts (C1 = 40, C2 = 20):

1. schedule   sigmas and timesteps equal ``set_timesteps`` bitwise.
2. synthetic  a fixed nonlinear velocity field at the full Wan latent shape;
              every step equals ``scheduler.step`` bitwise.
3. pipeline   a tiny, randomly initialized ``WanTransformer3DModel`` driven by
              the stock ``WanPipeline`` versus the custom loop; every step is
              compared. The bf16 case adds a forward hook that casts the
              Transformer output to fp32, so the stock pipeline follows the
              protocol's fp32 guidance policy.
4. control    a plausible schedule bug (shifting once instead of twice) must be
              detected, so the comparisons above are known to have teeth.

The stock bf16 pipeline does guidance in bf16 and stores latents in bf16
between steps. The protocol deliberately keeps both in fp32, so that
difference is reported for information, not judged.

On CPU every judged comparison must be bitwise. On MPS the tolerance is the
pipeline's own run-to-run difference, measured first.
"""

import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import diffusers
import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, WanPipeline, WanTransformer3DModel

from density.attention import generation_attention
from density.sampler import NUM_TRAIN_TIMESTEPS, euler_sample, flow_match_schedule, shift_sigmas
from density.wan import WanGuidedVelocity

# Protocol section 5: candidate (shift, guidance) pairs tested on holdout prompts.
SHIFT_GUIDANCE_PAIRS = [(8.0, 6.0), (5.0, 5.0)]
# Protocol section 3: C1 runs 40 dense steps, C2 runs 20.
STEP_COUNTS = [40, 20]

# Protocol section 2: 21 latent frames x 60 x 104 latent pixels (832 x 480 video).
FULL_LATENT_SHAPE = (1, 16, 21, 60, 104)
# Tiny pipeline: 9 frames at 128 x 128 pixels -> 3 latent frames of 16 x 16.
TINY_VIDEO = {"num_frames": 9, "height": 128, "width": 128}
TINY_LATENT_SHAPE = (1, 16, 3, 16, 16)
TINY_TEXT_LEN = 8
TINY_TRANSFORMER_CONFIG = {
    "patch_size": (1, 2, 2),
    "num_attention_heads": 2,
    "attention_head_dim": 12,
    "in_channels": 16,
    "out_channels": 16,
    "text_dim": 32,
    "freq_dim": 256,
    "ffn_dim": 32,
    "num_layers": 2,
    "cross_attn_norm": True,
    "qk_norm": "rms_norm_across_heads",
    "rope_max_seq_len": 32,
}
SEED = 20261005

REPORT_DIR = Path("results") / "checks"  # relative to the project root, where the check is run


def make_scheduler(shift: float) -> FlowMatchEulerDiscreteScheduler:
    # Construct directly: the Wan checkpoint ships a UniPC config whose
    # `flow_shift` key would be ignored by from_config, leaving shift = 1.
    # set_shift() is also unsafe, because __init__ fixes sigma_min from the
    # shift it was constructed with.
    scheduler = FlowMatchEulerDiscreteScheduler(shift=shift)
    assert scheduler.config.shift == shift
    assert not scheduler.config.use_dynamic_shifting
    assert not scheduler.config.stochastic_sampling
    return scheduler


def compare_trajectories(ours: list[torch.Tensor], ref: list[torch.Tensor]) -> dict:
    if len(ours) != len(ref):
        raise ValueError(f"trajectory lengths differ: {len(ours)} vs {len(ref)}")
    per_step = [(a.float() - b.float()).abs().max().item() for a, b in zip(ours, ref)]
    final_scale = ref[-1].float().abs().max().item()
    return {
        "bitwise_equal": all(a.dtype == b.dtype and torch.equal(a, b) for a, b in zip(ours, ref)),
        "max_abs_diff": max(per_step),
        "final_max_abs_diff": per_step[-1],
        "final_rel_diff": per_step[-1] / final_scale if final_scale > 0 else float("nan"),
        "per_step_max_abs_diff": per_step,
    }


def judge(result: dict, tolerance: float) -> dict:
    result["tolerance"] = tolerance
    result["passed"] = result["bitwise_equal"] if tolerance == 0 else result["max_abs_diff"] <= tolerance
    return result


def custom_trajectory(velocity_fn, latents, sigmas, timesteps) -> list[torch.Tensor]:
    trajectory = []
    euler_sample(velocity_fn, latents, sigmas, timesteps, on_step=lambda n, x, v: trajectory.append(x.clone()))
    return trajectory


# 1. Schedule


def check_schedule(shift: float, steps: int) -> dict:
    scheduler = make_scheduler(shift)
    scheduler.set_timesteps(steps)
    sigmas, timesteps = flow_match_schedule(steps, shift)
    return {
        "check": "schedule",
        "shift": shift,
        "steps": steps,
        "bitwise_equal": torch.equal(sigmas, scheduler.sigmas) and torch.equal(timesteps, scheduler.timesteps),
        "max_abs_diff": (sigmas - scheduler.sigmas).abs().max().item(),
        "passed": torch.equal(sigmas, scheduler.sigmas) and torch.equal(timesteps, scheduler.timesteps),
        "sigmas": sigmas.tolist(),
        "timesteps": timesteps.tolist(),
    }


# 2. Synthetic velocity at the full latent shape


def synthetic_velocity(x: torch.Tensor, t: torch.Tensor, n: int) -> torch.Tensor:
    """Fixed nonlinear field that depends on both the latents and the timestep."""
    s = t / NUM_TRAIN_TIMESTEPS
    return torch.tanh(x * (1 + s)) - (1 - s) * torch.sin(3 * x)


def _diffusers_trajectory(velocity_fn, latents, scheduler, steps, device) -> list[torch.Tensor]:
    scheduler.set_timesteps(steps, device=device)
    scheduler.set_begin_index(0)
    x = latents.clone()
    trajectory = []
    for n, t in enumerate(scheduler.timesteps):
        x = scheduler.step(velocity_fn(x, t, n), t, x, return_dict=False)[0]
        trajectory.append(x.clone())
    return trajectory


def check_synthetic(shift: float, steps: int, device: str) -> dict:
    generator = torch.Generator("cpu").manual_seed(SEED)
    latents = torch.randn(FULL_LATENT_SHAPE, generator=generator, dtype=torch.float32).to(device)

    ref = _diffusers_trajectory(synthetic_velocity, latents, make_scheduler(shift), steps, device)
    tolerance = 0.0
    if device != "cpu":
        repeat = _diffusers_trajectory(synthetic_velocity, latents, make_scheduler(shift), steps, device)
        tolerance = compare_trajectories(repeat, ref)["max_abs_diff"]

    sigmas, timesteps = flow_match_schedule(steps, shift)
    ours = custom_trajectory(synthetic_velocity, latents, sigmas, timesteps)
    result = {"check": "synthetic", "device": device, "shift": shift, "steps": steps, "latent_shape": list(FULL_LATENT_SHAPE)}
    result.update(compare_trajectories(ours, ref))
    return judge(result, tolerance)


# 3. Tiny Wan Transformer through the stock WanPipeline


def tiny_inputs(device: str, text_len: int = TINY_TEXT_LEN) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator("cpu").manual_seed(SEED)
    latents = torch.randn(TINY_LATENT_SHAPE, generator=generator, dtype=torch.float32)
    text_dim = TINY_TRANSFORMER_CONFIG["text_dim"]
    prompt_embeds = torch.randn(1, text_len, text_dim, generator=generator)
    negative_prompt_embeds = torch.randn(1, text_len, text_dim, generator=generator)
    return latents.to(device), prompt_embeds.to(device), negative_prompt_embeds.to(device)


def tiny_transformer(dtype: torch.dtype, device: str, config: dict = TINY_TRANSFORMER_CONFIG) -> WanTransformer3DModel:
    """Random tiny Wan Transformer, loaded the way the real checkpoint will be.

    The save/load round trip lets ``from_pretrained(torch_dtype=...)`` keep
    Wan's fp32-pinned modules (rope, time embedder, norms) in fp32, which a
    plain ``.to(dtype)`` would not.
    """
    torch.manual_seed(SEED)
    model = WanTransformer3DModel(**config)
    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        model = WanTransformer3DModel.from_pretrained(tmp, torch_dtype=dtype)
    return model.to(device).eval()


def cast_output_to_fp32(module, args, output):
    return tuple(o.float() for o in output)


def pipeline_trajectory(
    transformer,
    shift,
    steps,
    guidance,
    latents,
    prompt_embeds,
    negative_prompt_embeds,
    fp32_velocity,
    video=TINY_VIDEO,
    vae=None,
    output_type="latent",
):
    """Run the stock WanPipeline and return ``(per-step latents, pipeline output)``."""
    pipe = WanPipeline(tokenizer=None, text_encoder=None, vae=vae, scheduler=make_scheduler(shift), transformer=transformer)
    pipe.set_progress_bar_config(disable=True)
    trajectory = []

    def record(pipeline, i, t, tensors):
        trajectory.append(tensors["latents"].clone())
        return {}

    hook = transformer.register_forward_hook(cast_output_to_fp32) if fp32_velocity else None
    try:
        with generation_attention(latents.device):  # the same attention backends as the custom loop
            output = pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                num_inference_steps=steps,
                guidance_scale=guidance,
                latents=latents.clone(),
                output_type=output_type,
                return_dict=False,
                callback_on_step_end=record,
                callback_on_step_end_tensor_inputs=["latents"],
                **video,
            )[0]
    finally:
        if hook is not None:
            hook.remove()
    return trajectory, output


def velocity_is_informative(velocity: WanGuidedVelocity, latents: torch.Tensor, timesteps: torch.Tensor) -> dict:
    """Guard against a vacuous test: the velocity must be nonzero and depend on text and timestep."""
    with torch.no_grad():
        x = latents.to(velocity.dtype)
        t0 = timesteps[0].expand(latents.shape[0])
        t1 = timesteps[1].expand(latents.shape[0])
        v_cond = velocity._predict(x, t0, velocity.prompt_embeds).float()
        v_uncond = velocity._predict(x, t0, velocity.negative_prompt_embeds).float()
        v_cond_t1 = velocity._predict(x, t1, velocity.prompt_embeds).float()
    stats = {
        "velocity_rms": v_cond.pow(2).mean().sqrt().item(),
        "text_effect_rms": (v_cond - v_uncond).pow(2).mean().sqrt().item(),
        "timestep_effect_rms": (v_cond - v_cond_t1).pow(2).mean().sqrt().item(),
    }
    stats["informative"] = all(value > 0 for value in stats.values())
    return stats


def check_pipeline(shift: float, steps: int, guidance: float, dtype: torch.dtype, device: str) -> dict:
    latents, prompt_embeds, negative_prompt_embeds = tiny_inputs(device)
    transformer = tiny_transformer(dtype, device)
    fp32_velocity = dtype != torch.float32
    args = (transformer, shift, steps, guidance, latents, prompt_embeds, negative_prompt_embeds, fp32_velocity)

    ref = pipeline_trajectory(*args)[0]
    tolerance = 0.0
    if device != "cpu":
        tolerance = compare_trajectories(pipeline_trajectory(*args)[0], ref)["max_abs_diff"]

    sigmas, timesteps = flow_match_schedule(steps, shift)
    velocity = WanGuidedVelocity(transformer, prompt_embeds, negative_prompt_embeds, guidance)
    ours = custom_trajectory(velocity, latents, sigmas, timesteps)

    result = {
        "check": "pipeline",
        "device": device,
        "transformer_dtype": str(dtype).removeprefix("torch."),
        "pipeline_velocity": "fp32 (output hook)" if fp32_velocity else "fp32 (native)",
        "shift": shift,
        "steps": steps,
        "guidance": guidance,
    }
    result.update(compare_trajectories(ours, ref))
    result.update(velocity_is_informative(velocity, latents, timesteps.to(device)))
    judge(result, tolerance)
    result["passed"] = result["passed"] and result["informative"]
    return result


def stock_bf16_deviation(shift: float, steps: int, guidance: float, device: str) -> dict:
    """Informational: protocol fp32 policy versus the stock pipeline's bf16 guidance and latents."""
    latents, prompt_embeds, negative_prompt_embeds = tiny_inputs(device)
    transformer = tiny_transformer(torch.bfloat16, device)
    ref, _ = pipeline_trajectory(transformer, shift, steps, guidance, latents, prompt_embeds, negative_prompt_embeds, False)

    sigmas, timesteps = flow_match_schedule(steps, shift)
    velocity = WanGuidedVelocity(transformer, prompt_embeds, negative_prompt_embeds, guidance)
    ours = custom_trajectory(velocity, latents, sigmas, timesteps)

    result = {
        "check": "stock_bf16_deviation",
        "device": device,
        "shift": shift,
        "steps": steps,
        "guidance": guidance,
        "stock_latent_dtype": str(ref[-1].dtype).removeprefix("torch."),
        "passed": None,
    }
    result.update(compare_trajectories(ours, ref))
    return result


# 4. Negative control


def single_shift_schedule(steps: int, shift: float) -> tuple[torch.Tensor, torch.Tensor]:
    """A plausible bug: shift linspace(1, 1/T) once, instead of diffusers' twice-shifted endpoint."""
    sigmas = shift_sigmas(np.linspace(1.0, 1.0 / NUM_TRAIN_TIMESTEPS, steps), shift)
    sigmas = torch.from_numpy(sigmas).to(torch.float32)
    return torch.cat([sigmas, torch.zeros(1)]), sigmas * NUM_TRAIN_TIMESTEPS


def check_negative_control(shift: float, steps: int, guidance: float) -> dict:
    latents, prompt_embeds, negative_prompt_embeds = tiny_inputs("cpu")
    transformer = tiny_transformer(torch.float32, "cpu")
    ref, _ = pipeline_trajectory(transformer, shift, steps, guidance, latents, prompt_embeds, negative_prompt_embeds, False)

    sigmas, timesteps = single_shift_schedule(steps, shift)
    velocity = WanGuidedVelocity(transformer, prompt_embeds, negative_prompt_embeds, guidance)
    ours = custom_trajectory(velocity, latents, sigmas, timesteps)

    result = {"check": "negative_control", "device": "cpu", "shift": shift, "steps": steps, "guidance": guidance}
    result.update(compare_trajectories(ours, ref))
    result["passed"] = result["final_max_abs_diff"] > 0
    return result


# Report


def git_state() -> dict:
    """Code SHA for the manifest; ``dirty`` means uncommitted changes outside the report folder."""
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", ".", f":(exclude){REPORT_DIR}"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"git_sha": None, "git_dirty": None}
    return {"git_sha": sha, "git_dirty": bool(status.strip())}


def gpu_info() -> dict | None:
    """GPU model and driver version for job metadata on CUDA machines; None where CUDA is unavailable."""
    if not torch.cuda.is_available():
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True, text=True, check=True
        ).stdout
        smi = [dict(zip(("name", "driver_version"), (field.strip() for field in line.split(",")))) for line in out.strip().splitlines()]
    except (OSError, subprocess.CalledProcessError) as error:
        smi = f"nvidia-smi failed: {error}"
    return {
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "nvidia_smi": smi,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }


def environment() -> dict:
    return {
        **git_state(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "diffusers": diffusers.__version__,
        "numpy": np.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "gpu": gpu_info(),
    }


def run_all(devices: list[str] | None = None) -> dict:
    if devices is None:
        devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])

    results = []
    for shift, guidance in SHIFT_GUIDANCE_PAIRS:
        for steps in STEP_COUNTS:
            results.append(check_schedule(shift, steps))
            for device in devices:
                results.append(check_synthetic(shift, steps, device))
                results.append(check_pipeline(shift, steps, guidance, torch.float32, device))
                results.append(check_pipeline(shift, steps, guidance, torch.bfloat16, device))
                results.append(stock_bf16_deviation(shift, steps, guidance, device))
            results.append(check_negative_control(shift, steps, guidance))

    judged = [r for r in results if r["passed"] is not None]
    return {
        "milestone": "Custom Euler loop matches diffusers FlowMatchEulerDiscreteScheduler",
        "passed": all(r["passed"] for r in judged),
        "num_judged": len(judged),
        "num_failed": sum(not r["passed"] for r in judged),
        "environment": environment(),
        "settings": {
            "shift_guidance_pairs": SHIFT_GUIDANCE_PAIRS,
            "step_counts": STEP_COUNTS,
            "full_latent_shape": FULL_LATENT_SHAPE,
            "tiny_latent_shape": TINY_LATENT_SHAPE,
            "tiny_transformer_config": TINY_TRANSFORMER_CONFIG,
            "seed": SEED,
        },
        "results": results,
    }


def _fmt(value: float) -> str:
    return "0" if value == 0 else f"{value:.3g}"


def markdown_report(report: dict) -> str:
    env = report["environment"]
    lines = [
        "# Euler loop vs diffusers FlowMatchEulerDiscreteScheduler",
        "",
        f"**Overall: {'PASS' if report['passed'] else 'FAIL'}** "
        f"({report['num_judged'] - report['num_failed']}/{report['num_judged']} judged checks passed)",
        "",
        f"torch {env['torch']}, diffusers {env['diffusers']}, numpy {env['numpy']}, "
        f"Python {env['python']}, {env['platform']}",
        "",
        "| Check | Device | Precision | Shift | Steps | Guidance | Bitwise | Max abs diff | Tolerance | Result |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in report["results"]:
        precision = r.get("transformer_dtype", "")
        if r["check"] == "stock_bf16_deviation":
            precision = f"bf16, stock latents {r['stock_latent_dtype']}"
        status = "info" if r["passed"] is None else ("PASS" if r["passed"] else "FAIL")
        guidance = "" if r.get("guidance") is None else format(r["guidance"], "g")
        lines.append(
            f"| {r['check']} | {r.get('device', '')} | {precision} | {r['shift']:g} | {r['steps']} | {guidance} | "
            f"{'yes' if r['bitwise_equal'] else 'no'} | {_fmt(r['max_abs_diff'])} | "
            f"{_fmt(r['tolerance']) if 'tolerance' in r else ''} | {status} |"
        )

    lines += [
        "",
        "`negative_control` passes when a deliberately wrong schedule (shift applied once) is detected.",
        "`stock_bf16_deviation` compares the protocol's fp32 guidance and latents with the stock bf16 pipeline;",
        "it is reported for information only.",
        "",
        "## Recorded schedules",
        "",
    ]
    for r in report["results"]:
        if r["check"] == "schedule":
            lines.append(f"- shift {r['shift']:g}, {r['steps']} steps, sigmas: " + ", ".join(f"{s:.6f}" for s in r["sigmas"]))
    return "\n".join(lines) + "\n"


def default_devices() -> list[str]:
    return ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


def device_args(parser) -> None:
    """--devices and --report-dir, shared by the checks so a CUDA run never overwrites the committed Mac reports."""
    parser.add_argument("--devices", type=lambda s: s.split(","), default=None, help="comma-separated devices (default: cpu, and mps when available)")
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR, help="where to write the report (default: results/checks)")


def empty_cache(device: str) -> None:
    if device == "mps":
        torch.mps.empty_cache()
    elif device.startswith("cuda"):
        torch.cuda.empty_cache()


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Euler sampler check")
    device_args(parser)
    args = parser.parse_args()
    report = run_all(args.devices or default_devices())
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "euler_vs_diffusers.json").write_text(json.dumps(report, indent=2))
    (args.report_dir / "euler_vs_diffusers.md").write_text(markdown_report(report))
    print(markdown_report(report).split("## Recorded schedules")[0])
    print(f"Report written to {args.report_dir}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
