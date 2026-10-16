"""The per-pod canary: before any experiment job on an H100, reproduce smoke-1's dense C1 bitwise.

Registered (2026-10-07, Chase): at the start of every H100 session, before any experiment job runs,
regenerate the smoke-1 prompt's dense C1 and require it to match pod D's registered hashes bitwise
(smoke test Session D, results/checks/smoke_test.md). If it doesn't match, the session stops and
reports this pod's host and driver beside pod D's.

Two hashes are judged and reported separately: the final latents and the decoded uint8 frames. Flash
latents were reproduced across two H100 hosts in the smoke test (pod D against pod B's flash diagnostic,
for smoke-2), but decoded frames have never been compared across hosts, so the first pilot pod is also
the first cross-host test of the decode.

The run repeats the smoke test's generate step for smoke-1: the saved inputs, loaded by hash; the same
warm-up (2 dense steps and one decode, discarded); 40 dense steps; one decode. The generation code is
unchanged since pod D's run (code e948123). The canary runs in strict deterministic mode, like the
experiment jobs it guards; pod D's run used the smoke test's warn-only mode, which differs only in
raising instead of warning, and it recorded no warning about deterministic algorithms.

Experiment jobs call ``require_passed`` before they start. It accepts only a passing record written on
this host at the current code.

    python -m density.canary                                  on the pod, after sourcing scripts/pod/env.sh
    python -m density.canary --dry-run --records DIR --out D  tiny models, against a smoke dry run's records
"""

import argparse
import hashlib
import json
import socket
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from density import determinism, noise, wan_weights
from density.checks import euler, smoke, wan_real
from density.conditions import sha256
from density.wan import WanGuidedVelocity

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "checks" / "canary"  # not versioned; synced back with the session's results
INPUTS_DIR = ROOT / "results" / "checks" / "smoke_test" / "inputs"  # the smoke test's saved inputs, also on the volume
PROMPT_ID = "smoke-1"

REGISTERED = {
    "source": "smoke test Session D: pod 40skcaoggtjvcf, 2026-10-06, code e948123132f1553814423f581cb77433c33362c5",
    "pod": {
        "host": "0d7e358ab3af", "gpu": "NVIDIA H100 80GB HBM3", "driver": "580.126.09", "driver_cuda": "13.0",
        "cpu": "Intel(R) Xeon(R) Platinum 8470", "kernel": "6.8.0-90-generic",
    },
    "inputs": {
        "noise": {"file": "noise_smoke-1.safetensors", "tensor_sha256": "d0e94420359a4c4e4a8d600a7834a084cd147142db20535da6990bf448828ae5", "shape": [1, 16, 21, 60, 104]},
        "embeddings": {
            "file": "embeddings.safetensors",
            "file_sha256": "dec00f08266bf7bfd8bf41dbfec2cc14485ba6579a89a507322154d74155de0f",  # recorded only; the tensor hashes below are judged
            "sha256": {"smoke-1": "a7ba7402125af47d98da249936f862dc3de5466f9165067ae4c0ab001ec7f413", "negative": "645c3b082c4d9a8e5682bff6a427c2c8b733469b96ec9e7c74c8547999437ba5"},
        },
    },
    "expected": {
        "final_latents_sha256": "0f293e09e99f5156d3670ae2a514fac813f7286dd52af2273bb20ed3b2efa127",
        "frames_uint8_sha256": "97fe81c491960f594857397113c984c56a2f888b3791d015f265497090ea89ec",
    },
}


class CanaryNotPassed(RuntimeError):
    pass


def registered_from_records(records: Path) -> dict:
    """For a dry run: the same structure from a smoke dry run's noise, encode and generate records."""
    read = lambda name: json.loads((records / f"{name}.json").read_text())["result"]
    n, e, g = read("noise")["prompts"][PROMPT_ID], read("encode"), read("generate")["prompts"][PROMPT_ID]
    return {
        "source": f"the smoke records in {records}",
        "pod": {"host": None, "gpu": None, "driver": None, "driver_cuda": None, "cpu": None, "kernel": None},
        "inputs": {
            "noise": {"file": n["file"], "tensor_sha256": n["tensor_sha256"], "shape": n["shape"]},
            "embeddings": {"file": e["file"], "file_sha256": e["file_sha256"], "sha256": {k: e["embeddings"][k]["sha256"] for k in (PROMPT_ID, "negative")}},
        },
        "expected": {"final_latents_sha256": g["final_latents_sha256"][0], "frames_uint8_sha256": g["reference"]["uint8_sha256"]},
    }


def load_inputs(inputs_dir: Path, registered: dict, device: str):
    """Starting noise and embeddings from the saved files, each checked against its registered tensor hash."""
    reg = registered["inputs"]
    latents = noise.load(inputs_dir / reg["noise"]["file"], reg["noise"]["tensor_sha256"])
    if list(latents.shape) != list(reg["noise"]["shape"]):
        raise RuntimeError(f"the saved starting noise has shape {list(latents.shape)}, not {reg['noise']['shape']}")
    embeds = load_file(inputs_dir / reg["embeddings"]["file"])
    for name, expected in reg["embeddings"]["sha256"].items():
        if sha256(embeds[name]) != expected:
            raise RuntimeError(f"{name}: the saved embedding does not match its registered hash")
    return latents.to(device), embeds[PROMPT_ID].to(device), embeds["negative"].to(device)


def load_models(ctx: smoke.Context):
    """The Transformer (bf16) and VAE (fp32) as the smoke test loads them, after hashing their files."""
    if ctx.dry_run:
        return smoke.load_transformer(ctx), smoke.load_vae(ctx), None
    from diffusers import AutoencoderKLWan, WanTransformer3DModel

    verified = wan_weights.verify(ctx.weights, ("transformer", "vae"))  # smoke's loaders check a session record instead
    transformer = WanTransformer3DModel.from_pretrained(ctx.weights / "transformer", torch_dtype=torch.bfloat16).to(ctx.device).eval()
    vae = AutoencoderKLWan.from_pretrained(ctx.weights / "vae", torch_dtype=torch.float32).to(ctx.device).eval()
    return transformer, vae, verified


def dense_c1(ctx: smoke.Context, transformer, vae, inputs_dir: Path, registered: dict) -> dict:
    """smoke-1's dense C1 as the generate step runs it, with the hashes of its final latents and uint8 frames."""
    first = load_inputs(inputs_dir, registered, ctx.device)
    euler.custom_trajectory(WanGuidedVelocity(transformer, first[1], first[2], smoke.GUIDANCE), first[0], smoke.SIGMAS[:3], smoke.TIMESTEPS[:2])
    smoke.decode(ctx, vae, first[0][:, :, :2])  # warm-up, as in the generate step: discarded
    inputs = load_inputs(inputs_dir, registered, ctx.device)
    velocity = WanGuidedVelocity(transformer, inputs[1], inputs[2], smoke.GUIDANCE)
    smoke.release(ctx.device)
    smoke.reset_peak(ctx.device)
    with smoke.Timer(ctx.device) as t:
        trajectory = euler.custom_trajectory(velocity, inputs[0], smoke.SIGMAS, smoke.TIMESTEPS)
    peak = smoke.memory(ctx.device)
    trajectory = smoke.to_cpu(trajectory)
    frames, decode_seconds = smoke.decode(ctx, vae, trajectory[-1])
    return {
        "final_latents_sha256": sha256(trajectory[-1]),
        "frames_uint8_sha256": hashlib.sha256(wan_real.to_uint8(frames).tobytes()).hexdigest(),
        "finite": bool(torch.isfinite(trajectory[-1]).all() and np.isfinite(frames).all()),
        "denoise_seconds": t.seconds, "decode_seconds": decode_seconds, "memory_denoise": peak,
    }


def host_info() -> dict:
    n, s = smoke.nvidia(), smoke.system_info()
    query = [x.strip() for x in (n.get("query") or "").split(",")]
    return {
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "driver": query[1] if len(query) > 1 else None,
        "driver_cuda": n.get("driver_cuda_version"),
        "cpu": s["cpu_model"], "kernel": s["kernel"], "nvidia_smi_query": n.get("query"),
    }


def describe(pod: dict) -> str:
    return f"host {pod['host']} ({pod['gpu']}, driver {pod['driver']}, CUDA {pod['driver_cuda']}, {pod['cpu']}, kernel {pod['kernel']})"


def judge(got: dict, registered: dict, host: dict) -> dict:
    """Pass only if both hashes equal the registered ones; a failure names what differed and both pods."""
    expected = registered["expected"]
    matches = {"final_latents": got["final_latents_sha256"] == expected["final_latents_sha256"], "decoded_frames": got["frames_uint8_sha256"] == expected["frames_uint8_sha256"]}
    passed = all(matches.values()) and got["finite"]
    stop = None
    if not passed:
        differ = [name.replace("_", " ") for name, ok in matches.items() if not ok] + ([] if got["finite"] else ["finiteness"])
        stop = (f"canary mismatch in the {' and '.join(differ)}: this pod is {describe(host)}; the registered run is pod D, {describe(registered['pod'])}. "
                "Stop: no experiment job may run on this pod; report the host and driver to Chase.")
    return {"passed": passed, "matches": matches, "stop": stop}


def run(ctx: smoke.Context, registered: dict, inputs_dir: Path) -> dict:
    """Run the canary and write ``canary.json`` (the latest) and a dated copy in ``ctx.out``."""
    start, host = time.perf_counter(), host_info()
    record = {
        "utc": smoke.utc_now(), "dry_run": ctx.dry_run, "device": ctx.device, "host": host, "code": euler.git_state(),
        "registered": {k: registered[k] for k in ("source", "pod", "expected")}, "inputs_dir": str(inputs_dir),
    }
    log = determinism.WarningLog()
    try:
        record["determinism"] = determinism.settings()
        with log.capture("canary"):
            transformer, vae, verified = load_models(ctx)
            record["weights_verified"] = None if verified is None else {"revision": verified["revision"], "files": len(verified["files"])}
            record["got"] = dense_c1(ctx, transformer, vae, inputs_dir, registered)
        record.update(judge(record["got"], registered, host))
    except Exception as error:
        record.update({"passed": False, "matches": None, "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc(),
                       "stop": f"the canary did not finish on {describe(host)}: {type(error).__name__}: {error}. Stop: no experiment job may run on this pod."})
    record["warnings"] = log.summary()
    record["seconds"] = time.perf_counter() - start
    ctx.out.mkdir(parents=True, exist_ok=True)
    smoke.write_json(ctx.out / "canary.json", record)
    smoke.write_json(ctx.out / f"canary_{host['host']}_{record['utc'][:19].replace(':', '')}.json", record)
    return record


def require_passed(out: Path = OUT_DIR) -> dict:
    """The passing canary record for this host at the current code; raises CanaryNotPassed otherwise."""
    path = out / "canary.json"
    if not path.exists():
        raise CanaryNotPassed(f"no canary record in {out}: run python -m density.canary on this pod first")
    rec = json.loads(path.read_text())
    host, code = socket.gethostname(), euler.git_state()
    if not rec.get("passed"):
        raise CanaryNotPassed(f"the canary did not pass: {rec.get('stop')}")
    if rec.get("dry_run"):
        raise CanaryNotPassed("the canary record is from a dry run")
    if rec["host"]["host"] != host:
        raise CanaryNotPassed(f"the canary passed on host {rec['host']['host']}, not this one ({host}): run it on this pod")
    if rec["code"]["git_sha"] != code["git_sha"] or rec["code"]["git_dirty"] or code["git_dirty"]:
        raise CanaryNotPassed(f"the canary ran at code {rec['code']['git_sha']} (dirty: {rec['code']['git_dirty']}); this is {code['git_sha']} (dirty: {code['git_dirty']}): run it again")
    return rec


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--inputs", type=Path, default=INPUTS_DIR)
    parser.add_argument("--weights", type=Path, default=wan_real.WEIGHTS_DIR)
    parser.add_argument("--device", default=None, help="default: cuda when available, else cpu")
    parser.add_argument("--dry-run", action="store_true", help="tiny random models, judged against --records")
    parser.add_argument("--records", type=Path, help="dry run: a smoke dry run's --out, whose noise, encode and generate records stand in for pod D's")
    args = parser.parse_args(argv)
    if args.dry_run and args.records is None:
        parser.error("--dry-run needs --records")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    determinism.configure(strict=True)
    ctx = smoke.Context(args.out, args.out, device, args.weights, None, args.dry_run)
    registered = registered_from_records(args.records) if args.dry_run else REGISTERED
    record = run(ctx, registered, args.records / "inputs" if args.dry_run else args.inputs)
    if record["passed"]:
        print(f"[canary] passed on {describe(record['host'])} in {record['seconds']:.0f} s", flush=True)
        return 0
    print(f"[canary] STOP: {record['stop']}", flush=True)
    return 1 if record.get("matches") is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
