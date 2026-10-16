"""Run on pod B (H100) after Session B, at code 7b9b39b, from /workspace/smoke_scratch; its record is in results/checks/smoke_test.

Is dense C1 bitwise across processes? mode: default (cuDNN as chosen) compared with generate's saved trajectory,
flash-save (flash pinned, saved), flash-compare (flash pinned, compared with the saved one). Prompt smoke-2."""
import json, sys, time
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file
from torch.nn.attention import SDPBackend, sdpa_kernel
from density import determinism
from density.checks import smoke

determinism.configure()
mode = sys.argv[1]
OUT = Path("/workspace/Density_Project/results/checks/smoke_test")
ctx = smoke.Context(OUT, Path("/workspace/smoke_scratch"), "cuda", Path("/workspace/Density_Project/weights/Wan2.1-T2V-1.3B-Diffusers"), "B", dry_run=False)
tr = smoke.load_transformer(ctx)
prompt = smoke.PROMPTS[1]
latents, pe, ne = smoke.load_inputs(ctx, prompt)
velocity = smoke.WanGuidedVelocity(tr, pe, ne, smoke.GUIDANCE)
start = time.time()
with determinism.record_sdpa_choices() as calls:
    if mode == "default":
        traj = smoke.to_cpu(smoke.euler.custom_trajectory(velocity, latents, smoke.SIGMAS, smoke.TIMESTEPS))
    else:
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            traj = smoke.to_cpu(smoke.euler.custom_trajectory(velocity, latents, smoke.SIGMAS, smoke.TIMESTEPS))
backends = sorted({c["backend"] for c in calls})
flash_file = ctx.scratch / "dense_smoke-2_flash.safetensors"
if mode == "default":
    generated = smoke.read_result(ctx, "generate")
    ref = smoke.load_dense(ctx, prompt, generated)
    label = "default backend in a new process vs generate's run 1"
elif mode == "flash-save":
    save_file({"trajectory": torch.stack(traj)}, flash_file)
    ref, label = None, "flash, saved"
else:
    ref, label = list(load_file(flash_file)["trajectory"].unbind(0)), "flash in a new process vs the saved flash run"
rec = {"mode": mode, "label": label, "backends": backends, "seconds": time.time() - start}
if ref is not None:
    stats = smoke.compare(traj, ref)
    rec.update({k: stats[k] for k in ("bitwise_equal", "max_abs_diff", "first_divergent_step")}, per_step=[f"{v:.2g}" for v in stats["per_step_max_abs_diff"][:8]])
path = OUT / "diag_crossproc.json"
done = json.loads(path.read_text()) if path.exists() else {}
done[mode] = rec
smoke.write_json(path, done)
print(json.dumps(rec), flush=True)
