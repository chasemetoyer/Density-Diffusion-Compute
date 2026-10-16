"""Run on pod B (H100) after Session B, at code 7b9b39b, from /workspace/smoke_scratch; its record is in results/checks/smoke_test.

Full-resolution diagnostic: at iteration 0, compare every module call of diffusers' dense forward with the
HSA forward with every token active, find the first output that differs, and test whether its inputs or their
memory layout explain it. Writes /workspace/Density_Project/results/checks/smoke_test/diag_hsa_dense.json."""
import hashlib, json, time
from pathlib import Path
import torch
from density import determinism
from density.checks import smoke
from density.hsa import WanHSA
from density.checks.hsa import assert_stock_processors

determinism.configure()
OUT = Path("/workspace/Density_Project/results/checks/smoke_test")
ctx = smoke.Context(OUT, Path("/workspace/smoke_scratch"), "cuda", Path("/workspace/Density_Project/weights/Wan2.1-T2V-1.3B-Diffusers"), "B", dry_run=False)
tr = smoke.load_transformer(ctx)
assert_stock_processors(tr)
leaves = [(n, m) for n, m in tr.named_modules() if n and not list(m.children())]

def digest(x):
    """128-bit fingerprint computed on the GPU (as in density.checks.smoke.fingerprint at c302f80)."""
    dense = torch.empty(x.numel(), dtype=x.dtype, device=x.device)
    dense.copy_(x.detach().reshape(-1))  # a fresh stride-1 buffer: expanded (stride-0) inputs cannot be viewed as bytes
    flat = dense.view(torch.uint8)
    nbytes = flat.numel()
    if nbytes % 8:
        flat = torch.cat([flat, flat.new_zeros(8 - nbytes % 8)])
    words = flat.view(torch.int64)
    parts = []
    for seed in (20261006, 20261007):
        g = torch.Generator(device=words.device).manual_seed(seed)
        w = torch.randint(-(2**62), 2**62, words.shape, generator=g, device=words.device, dtype=torch.int64)
        parts.append(int((words * w).sum().item()) & (2**64 - 1))
    return f"{nbytes:x}:{parts[0]:016x}{parts[1]:016x}"

def run(path_fn, keep=None):
    calls, kept = [], {}
    def hook(name):
        def fn(module, args, out):
            x = next((a for a in args if isinstance(a, torch.Tensor)), None)
            y = out[0] if isinstance(out, tuple) else out
            if not isinstance(y, torch.Tensor):
                return
            calls.append({"module": name, "type": type(module).__name__, "out": digest(y), "in": digest(x) if x is not None else None,
                          "in_shape": list(x.shape) if x is not None else None, "in_strides": list(x.stride()) if x is not None else None,
                          "out_strides": list(y.stride()), "in_contiguous": bool(x.is_contiguous()) if x is not None else None})
            if keep is not None and len(calls) - 1 == keep:
                kept["module"], kept["x"], kept["y"] = module, x.detach().clone(), y.detach().clone()
        return fn
    handles = [m.register_forward_hook(hook(n)) for n, m in leaves]
    try:
        with torch.no_grad():
            path_fn()
    finally:
        for h in handles:
            h.remove()
    return calls, kept

report = {"prompts": {}}
for prompt in smoke.PROMPTS:
    latents, pe, _ = smoke.load_inputs(ctx, prompt)
    t = smoke.TIMESTEPS[0].to("cuda").expand(1)
    ids = torch.arange(ctx.num_tokens)
    dense = lambda: tr(hidden_states=latents.to(tr.dtype), timestep=t, encoder_hidden_states=pe.to(tr.dtype), return_dict=False)[0]
    hsa = lambda: WanHSA(tr).forward(latents, smoke.TIMESTEPS[0].to("cuda"), pe, ids, "cond", 0)
    start = time.time()
    dense_calls, _ = run(dense)
    hsa_calls, _ = run(hsa)
    pairs = list(zip(dense_calls, hsa_calls))
    order_ok = len(dense_calls) == len(hsa_calls) and all(a["module"] == b["module"] for a, b in pairs)
    index = next((i for i, (a, b) in enumerate(pairs) if a["out"] != b["out"]), None)
    rec = {"calls": [len(dense_calls), len(hsa_calls)], "same_module_order": order_ok, "first_differing_call": index, "seconds_hash_pass": time.time() - start}
    if index is not None:
        a, b = pairs[index]
        rec["dense"], rec["hsa"] = a, b
        rec["inputs_equal"] = a["in"] == b["in"]
        _, dk = run(dense, keep=index)
        _, hk = run(hsa, keep=index)
        module, xd, xh = dk["module"], dk["x"], hk["x"]
        first = lambda y: y[0] if isinstance(y, tuple) else y
        with torch.no_grad():
            y_dense_input = first(module(xd))
            y_dense_input_contiguous = first(module(xd.contiguous()))
            y_hsa_input = first(module(xh))
        rec["replay"] = {
            "module_on_dense_input_equals_dense_output": torch.equal(y_dense_input, dk["y"]),
            "module_on_hsa_input_equals_hsa_output": torch.equal(y_hsa_input, hk["y"]),
            "dense_input_made_contiguous_equals_hsa_output": torch.equal(y_dense_input_contiguous, hk["y"]),
            "values_equal_inputs": torch.equal(xd, xh),
            "max_abs_diff_outputs": (dk["y"].float() - hk["y"].float()).abs().max().item(),
        }
        rec["previous_calls"] = [{k: c[k] for k in ("module", "type", "in_contiguous", "in_strides")} for c, _ in pairs[max(0, index - 3): index]]
    # Is the attention backend the cause? Compare the two forwards' outputs under each SDPA backend.
    import contextlib
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from density.hsa import unpatchify
    tests = {}
    for label, backends in (("default", None), ("flash only", [SDPBackend.FLASH_ATTENTION]), ("efficient only", [SDPBackend.EFFICIENT_ATTENTION]), ("cudnn only", [SDPBackend.CUDNN_ATTENTION])):
        try:
            with (sdpa_kernel(backends) if backends else contextlib.nullcontext()), torch.no_grad(), determinism.record_sdpa_choices() as calls:
                yd = dense()
                yh = unpatchify(hsa(), latents.shape, tr.config.patch_size)
            tests[label] = {"equal": torch.equal(yd, yh), "max_abs_diff": (yd.float() - yh.float()).abs().max().item(), "backends": sorted({c["backend"] for c in calls})}
        except Exception as error:
            tests[label] = {"error": f"{type(error).__name__}: {error}"[:300]}
    rec["by_attention_backend"] = tests
    report["prompts"][prompt["id"]] = rec
    print(prompt["id"], json.dumps(rec, default=str)[:1500], flush=True)
# Which attention backends are row-invariant at these shapes? Identical random q, K, V; every query vs a subset.
import contextlib
from torch.nn.attention import SDPBackend, sdpa_kernel
from density.checks import hsa as hsa_checks
from density import conditions
g = torch.Generator(device="cuda").manual_seed(0)
N, A = ctx.num_tokens, smoke.PRESET.high_tokens
rows = torch.randperm(N, generator=torch.Generator().manual_seed(0))[:A].sort().values.cuda()
q, k, v = (torch.randn(1, N, 12, 128, generator=g, device="cuda", dtype=torch.bfloat16) for _ in range(3))
invariance = {}
for label, backend in (("flash", SDPBackend.FLASH_ATTENTION), ("efficient", SDPBackend.EFFICIENT_ATTENTION), ("cudnn", SDPBackend.CUDNN_ATTENTION)):
    try:
        with sdpa_kernel([backend]), torch.no_grad():
            every = hsa_checks._sdpa(q, k, v)[:, rows]
            subset = hsa_checks._sdpa(q[:, rows].contiguous(), k, v)
            again = hsa_checks._sdpa(q, k, v)[:, rows]
        invariance[label] = {"row_invariant": torch.equal(every, subset), "repeat_identical": torch.equal(every, again), "max_abs_diff": (every.float() - subset.float()).abs().max().item()}
    except Exception as error:
        invariance[label] = {"error": f"{type(error).__name__}: {error}"[:300]}
report["sdpa_row_invariance"] = invariance
print("row invariance:", invariance, flush=True)

# Full trajectories on the sensitive prompt with flash attention pinned: HSA all-full vs dense, and C3's fast path vs the oracle.
prompt = smoke.PROMPTS[1]
inputs = smoke.load_inputs(ctx, prompt)
pinned = {}
with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
    start = time.time()
    dense_traj = smoke.to_cpu(smoke.euler.custom_trajectory(smoke.WanGuidedVelocity(tr, inputs[1], inputs[2], smoke.GUIDANCE), inputs[0], smoke.SIGMAS, smoke.TIMESTEPS))
    full = smoke.wan_hsa_velocity(tr, inputs[1], inputs[2], smoke.GUIDANCE, hsa_checks.all_full(N))
    full_traj = smoke.to_cpu(smoke.euler.custom_trajectory(full, inputs[0], smoke.SIGMAS, smoke.TIMESTEPS))
    full.active_velocity_fn.hsa.caches.clear(); del full; smoke.release("cuda")
    pinned["hsa_all_full_vs_dense"] = {k: v for k, v in smoke.compare(full_traj, dense_traj).items() if k != "per_step_max_abs_diff"}
    c3 = smoke.run(ctx, tr, "C3", prompt, inputs, keep_trajectory=True)
    c3_traj, labels, iters = smoke.to_cpu(c3.trajectory), c3.record.labels, c3.record.group_iterations
    smoke.drop(c3); del c3; smoke.release("cuda")
    oracle_traj = smoke.oracle_trajectory(tr, inputs, labels, iters)
    pinned["c3_fast_vs_oracle"] = {k: v for k, v in smoke.compare(c3_traj, oracle_traj).items() if k != "per_step_max_abs_diff"}
    pinned["c3_labels_sha256"] = conditions.sha256(labels)
    pinned["seconds"] = time.time() - start
report["flash_pinned_smoke_2"] = pinned
print("flash pinned:", json.dumps(pinned, default=str), flush=True)
smoke.write_json(OUT / "diag_hsa_dense.json", report)
