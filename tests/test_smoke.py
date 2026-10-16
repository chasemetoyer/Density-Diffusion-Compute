"""Unit tests for the GPU smoke-test helpers: weight registry hashes, T5 encoding, determinism records,
failure classification, the step runner's skip and error handling, the frame-0 check and the projections."""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from density import determinism, text_encoding, wan_weights
from density.checks import smoke, smoke_report

ROOT = Path(__file__).resolve().parents[1]


def test_git_blob_sha1_matches_git(tmp_path):
    for content in (b"", b"hello\n", bytes(range(256)) * 3):
        path = tmp_path / "f"
        path.write_bytes(content)
        expected = subprocess.run(["git", "hash-object", str(path)], capture_output=True, text=True, check=True).stdout.strip()
        assert wan_weights.git_blob_sha1(path) == expected


def test_verify_checks_size_and_hash(tmp_path, monkeypatch):
    (tmp_path / "w" / "vae").mkdir(parents=True)
    (tmp_path / "w" / "model_index.json").write_bytes(b"{}")
    (tmp_path / "w" / "vae" / "big.safetensors").write_bytes(b"x" * 10)
    registry = {
        "repo": "r", "revision": "v",
        "files": {
            "model_index.json": {"size": 2, "git_blob_sha1": wan_weights.git_blob_sha1(tmp_path / "w" / "model_index.json")},
            "vae/big.safetensors": {"size": 10, "sha256": wan_weights.sha256_file(tmp_path / "w" / "vae" / "big.safetensors")},
            "transformer/other.safetensors": {"size": 1, "sha256": "0" * 64},
        },
    }
    (tmp_path / "registry.json").write_text(json.dumps(registry))
    monkeypatch.setattr(wan_weights, "REGISTRY", tmp_path / "registry.json")
    checked = wan_weights.verify(tmp_path / "w", ("vae",))
    assert sorted(checked["files"]) == ["model_index.json", "vae/big.safetensors"]
    with pytest.raises(FileNotFoundError):
        wan_weights.verify(tmp_path / "w")  # the transformer file is missing
    (tmp_path / "w" / "vae" / "big.safetensors").write_bytes(b"y" * 10)
    with pytest.raises(RuntimeError, match="sha256"):
        wan_weights.verify(tmp_path / "w", ("vae",))


def test_registered_wan_files_cover_the_pipeline():
    files = wan_weights.registry()["files"]
    for component in ("transformer", "vae", "text_encoder", "tokenizer"):
        assert any(rel.startswith(component + "/") for rel in files)
    assert all(("sha256" in f) != ("git_blob_sha1" in f) for f in files.values())


def test_encode_pads_with_zeros_after_the_tokens():
    tokenizer, encoder = smoke.tiny_text_encoder("cpu")
    texts = {"a": "Close-up of a woman laughing softly", "b": "two hands"}
    out = text_encoding.encode(texts, tokenizer, encoder, "cpu")
    for name, text in texts.items():
        e = out[name]
        assert e.shape == (1, text_encoding.MAX_SEQUENCE_LENGTH, 32) and e.dtype == torch.bfloat16
        tokens = int(tokenizer(text, return_attention_mask=True).attention_mask.__len__())
        rows = e[0].abs().amax(dim=-1) > 0
        assert rows[:tokens].all() and not rows[tokens:].any()


def test_negative_prompt_is_wans():
    assert len(text_encoding.NEGATIVE_PROMPT) == 137 and text_encoding.NEGATIVE_PROMPT.startswith("色调艳丽") and text_encoding.NEGATIVE_PROMPT.endswith("倒着走")
    text_encoding.check_ftfy_absent()


def test_determinism_records_precision_switches():
    keys = determinism.settings()
    for key in ("CUBLAS_WORKSPACE_CONFIG", "cudnn_benchmark", "deterministic_algorithms", "deterministic_warn_only", "fill_uninitialized_memory", "cudnn_conv_fp32_precision", "cuda_matmul_fp32_precision", "sdpa_enabled"):
        assert key in keys


def test_sdpa_choices_are_recorded_and_restored():
    original = torch.nn.functional.scaled_dot_product_attention
    q = torch.randn(1, 2, 8, 16)
    with determinism.record_sdpa_choices() as calls:
        out = torch.nn.functional.scaled_dot_product_attention(query=q, key=q, value=q)
    assert torch.nn.functional.scaled_dot_product_attention is original
    assert torch.equal(out, original(q, q, q))
    summary = determinism.summarize_sdpa(calls)
    assert summary[0]["calls"] == 1 and summary[0]["backend"] in determinism.SDPA_BACKENDS.values()


def test_classify_separates_measurements_from_failures():
    rows = [
        {"test": "T3 fast path vs oracle", "device": "cuda", "passed": False, "bitwise_equal": False, "max_abs_diff": 1e-3},
        {"test": "T7 active counts and budget", "device": "cuda", "passed": False},
        {"test": "T10 lambda = 0 equals C3", "device": "cuda", "passed": False, "labels_equal": True, "active_ids_equal": True, "output_bitwise": False},
        {"test": "T10 lambda = 0 equals C3", "device": "cuda", "passed": False, "labels_equal": False, "active_ids_equal": True, "output_bitwise": False},
        {"test": "T3 fast path vs oracle", "device": "cpu", "passed": False, "bitwise_equal": False},
        {"test": "T1 all groups full vs dense", "device": "cuda", "passed": None, "frames_judged": False},
        {"test": "T2 all active, layer by layer", "device": "cuda", "passed": True},
    ]
    m2 = smoke.classify("m2", {"results": rows[:2] + rows[4:]}, "cuda")
    assert [r["test"] for r in m2["measurements"]] == ["T3 fast path vs oracle"]
    assert [r["test"] for r in m2["failures"]] == ["T7 active counts and budget", "T3 fast path vs oracle"]
    assert len(m2["unjudged"]) == 1
    m3 = smoke.classify("m3", {"results": rows[2:4]}, "cuda")
    assert len(m3["measurements"]) == 1 and len(m3["failures"]) == 1


def test_relative_reports_a_zero_noise_floor():
    assert smoke.relative(0.0, 0.0) == {"ratio": 0.0}
    assert smoke.relative(2.0, 0.5) == {"ratio": 4.0}
    assert smoke.relative(1e-3, 0.0)["ratio"] is None


def test_verdict_classes():
    bitwise = {"bitwise_equal": True, "first_divergent_step": None}
    early = {"bitwise_equal": False, "first_divergent_step": 2}
    late = {"bitwise_equal": False, "first_divergent_step": 7}
    assert smoke.verdict(bitwise, None) == "bitwise"
    assert smoke.verdict(early, None).startswith("bug or nondeterminism")
    assert smoke.verdict(late, {"kernel_alone_reproduces": True, "replay_matches_run": True, "module": "blocks.3.attn1"}).startswith("measurement")
    assert smoke.verdict(late, {"kernel_alone_reproduces": False, "replay_matches_run": True, "module": "blocks.3.attn1"}).startswith("bug to investigate")
    gemm = {"kernel_alone_reproduces": True, "replay_matches_run": True, "module": "blocks.0.attn1", "explained_by": ["GEMM row-count dependence"]}
    assert "GEMM row-count dependence" in smoke.verdict(late, gemm)
    assert smoke.verdict(late, {**gemm, "replay_matches_run": False}).startswith("bug to investigate")
    assert "RoPE rows" in smoke.verdict(late, {"kernel_alone_reproduces": False, "replay_matches_run": True, "module": "blocks.0.attn1", "hsa_structure_ok": False})


def _ctx(tmp_path):
    return smoke.Context(tmp_path / "out", tmp_path / "scratch", "cpu", tmp_path / "w", "A", dry_run=True)


def test_run_step_skips_current_records_and_records_errors(tmp_path, monkeypatch, capsys):
    calls = []

    def ok(ctx):
        calls.append(1)
        return {"value": 1}

    def boom(ctx):
        raise ValueError("injected")

    monkeypatch.setitem(smoke.STEPS, "sheet", ok)
    monkeypatch.setitem(smoke.STEPS, "sdpa", boom)
    monkeypatch.setitem(smoke.DEPENDS, "sheet", ())
    ctx = _ctx(tmp_path)
    assert smoke.run_step(ctx, "sheet") == 0
    assert smoke.run_step(ctx, "sheet") == 0
    assert calls == [1]
    record = json.loads((ctx.out / "sheet.json").read_text())
    assert record["result"] == {"value": 1} and record["key"]["dry_run"] is True
    assert smoke.run_step(ctx, "sdpa") == 1
    error = json.loads((ctx.out / "sdpa.error.json").read_text())
    assert error["error"] == "ValueError: injected" and not error["oom"]
    ctx.force = True
    assert smoke.run_step(ctx, "sheet") == 0 and calls == [1, 1]


def test_run_step_stops_on_recorded_failures(tmp_path, monkeypatch):
    monkeypatch.setitem(smoke.STEPS, "sheet", lambda ctx: {"stop": ["a hash mismatch"], "blocks_pilot": ["PSNR"]})
    monkeypatch.setitem(smoke.DEPENDS, "sheet", ())
    assert smoke.run_step(_ctx(tmp_path), "sheet") == 1


def test_hour_cap_stops_before_a_step(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    ctx.dry_run = False
    monkeypatch.setenv("POD_CREATED_UTC", "2020-01-01T00:00:00+00:00")
    monkeypatch.setenv("POD_GPU", "h100")
    with pytest.raises(smoke.CapReached):
        smoke.check_hours(ctx, "t3")
    monkeypatch.setenv("POD_CREATED_UTC", smoke.utc_now())
    smoke.check_hours(ctx, "t3")


def test_frame0_luma_flags_a_bright_first_frame():
    frames = np.full((9, 4, 4, 3), 100, dtype=np.uint8)
    frames[1:] += np.arange(8, dtype=np.uint8)[:, None, None, None]
    frames[0] = 200
    luma = smoke.frame0_luma(frames)
    assert luma["brighter_than_every_other_frame"] and luma["difference"] > 0.3
    frames[0] = 100
    assert not smoke.frame0_luma(frames)["brighter_than_mean"]


def test_c4_mask_is_a_centered_disk():
    ctx = smoke.Context(Path("."), Path("."), "cpu", Path("."), None, dry_run=False)
    coverage, info = smoke.c4_coverage(ctx)
    assert coverage.shape == (ctx.num_tokens,) == (32760,)
    assert info["center_xy"] == [415.5, 239.5] and info["radius_px"] == 120
    grid = coverage.view(21, 30, 52)
    assert torch.equal(grid, grid.flip(1)) and torch.equal(grid, grid.flip(2))  # symmetric about the center


def test_protocol_blocks_reproduce_the_planning_table():
    for block, phase, runs, planning_re in smoke_report.PROTOCOL_BLOCKS:
        assert sum(n * smoke_report.RE_PER_RUN[c] for c, n in runs.items()) == planning_re, block
    totals = {phase: sum(b[3] for b in smoke_report.PROTOCOL_BLOCKS if b[1] == phase) for phase in ("pilot", "main")}
    assert totals == {"pilot": 54, "main": 498}  # plus the smoke test's 6 RE: 558 in total (C2 joined the pilot on 2026-10-07)


def test_starting_noise_is_seeded_on_cpu_and_loaded_by_hash(tmp_path):
    from density import noise
    from density.conditions import sha256

    a = noise.generate(9001, (1, 16, 3, 4, 4))
    assert a.dtype == torch.float32 and torch.equal(a, noise.generate(9001, (1, 16, 3, 4, 4)))
    assert not torch.equal(a, noise.generate(9002, (1, 16, 3, 4, 4)))
    digest = noise.save(a, tmp_path / "n.safetensors")
    assert digest == sha256(a) and torch.equal(noise.load(tmp_path / "n.safetensors", digest), a)
    with pytest.raises(RuntimeError, match="registered"):
        noise.load(tmp_path / "n.safetensors", "0" * 64)


def test_smoke_prompts_are_excluded_from_every_pool():
    config = json.loads((ROOT / "configs" / "smoke_prompts.json").read_text())
    assert config["excluded_from_every_pool"] is True
    assert [p["seed"] for p in config["prompts"]] == [9001, 9002]


def test_pod_scripts_parse():
    for script in sorted((ROOT / "scripts" / "pod").glob("*.sh")):
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_pinned_input_from_a_dry_run_stops_a_real_run(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setitem(smoke.STEPS, "noise", lambda ctx: calls.append(1) or {"prompts": {}})
    dry = _ctx(tmp_path)
    assert smoke.run_step(dry, "noise") == 0 and calls == [1]
    real = _ctx(tmp_path)
    real.dry_run = False
    assert smoke.run_step(real, "noise") == 1 and calls == [1]  # neither reused nor recomputed
    assert smoke.run_step(dry, "noise") == 0 and calls == [1]  # the dry run still skips its own input


def test_hour_cap_parses_utc_and_rejects_bad_values(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    ctx.dry_run = False
    monkeypatch.setenv("POD_GPU", "slice")
    monkeypatch.setenv("POD_CREATED_UTC", smoke.utc_now().replace("+00:00", "Z"))
    smoke.check_hours(ctx, "masks")
    for created, gpu in (("yesterday", "slice"), ("2026-10-06T12:00:00", "slice"), (smoke.utc_now(), "a100")):
        monkeypatch.setenv("POD_CREATED_UTC", created)
        monkeypatch.setenv("POD_GPU", gpu)
        with pytest.raises(smoke.CapReached):
            smoke.check_hours(ctx, "masks")


def test_run_step_records_settings_changed_by_a_step(tmp_path, monkeypatch):
    def flip(ctx):
        torch.backends.cudnn.benchmark = True
        return {}

    monkeypatch.setitem(smoke.STEPS, "sheet", flip)
    monkeypatch.setitem(smoke.DEPENDS, "sheet", ())
    ctx = _ctx(tmp_path)
    try:
        assert smoke.run_step(ctx, "sheet") == 0
    finally:
        torch.backends.cudnn.benchmark = False
    record = json.loads((ctx.out / "sheet.json").read_text())
    assert record["determinism_changed"] == {"cudnn_benchmark": [False, True]}


def test_tf32_decision_applies_the_ten_percent_rule():
    def rec(result):
        return {"result": result}

    records = {
        "vae_precision": rec({"prompts": {"a": {"tf32_seconds": 10.0, "ieee_seconds": 14.0, "decoded_psnr_db": 60.0}}}),
        "perf": rec({
            "summary": {c: {"end_to_end_seconds_mean": v} for c, v in (("C1", 60.0), ("C3", 35.0), ("C4", 35.0))},
            "runs": [{"condition": c, "denoise_seconds": v - 3, "end_to_end_seconds": v} for c, v in (("C1", 60.0), ("C3", 35.0), ("C4", 35.0), ("C1", 61.0), ("C3", 36.0), ("C4", 35.5))],
        }),
        "masks": rec({"references": {"smoke-1": {"stratum": "face", "ieee_fp32_convolutions": {"seconds": {"total": 52.0}, "default_seconds_without_build": 50.0, "identical_to_default": False}}}}),
    }
    d = smoke_report.tf32_decision(records)
    assert d["vae"]["added_seconds_per_video"] == [4.0]
    assert abs(d["vae"]["relative_to"]["C3"] - 4 / 35) < 1e-12 and d["vae"]["within_rule"] is False  # 11.4 percent of C3
    assert d["masks"]["smoke-1"]["within_rule"] is True and abs(d["masks"]["smoke-1"]["relative"] - 0.04) < 1e-12


def test_fingerprint_is_exact_for_equality_and_sensitive_to_one_bit():
    x = torch.randn(3, 1001, dtype=torch.bfloat16)
    assert smoke.fingerprint(x) == smoke.fingerprint(x.clone()) == smoke.fingerprint(x.t().contiguous().t())
    y = x.clone()
    y.view(torch.int16)[1, 500] ^= 1  # flip the lowest bit of one element
    assert smoke.fingerprint(y) != smoke.fingerprint(x)
    assert smoke.fingerprint(torch.zeros(5)) != smoke.fingerprint(torch.zeros(6))  # length matters through padding and weights
    t = torch.tensor([999.0])
    assert smoke.fingerprint(t.expand(1)) == smoke.fingerprint(t) and smoke.fingerprint(t.expand(4)) == smoke.fingerprint(t.repeat(4))  # stride-0 views


def test_generation_attention_is_registered_for_cuda_only():
    from contextlib import nullcontext

    from torch.nn.attention import SDPBackend

    from density import attention

    assert attention.GENERATION_BACKENDS[0] == SDPBackend.FLASH_ATTENTION
    assert isinstance(attention.generation_attention("cpu"), nullcontext)
    assert not isinstance(attention.generation_attention("cuda"), nullcontext)  # built, not entered: no GPU needed


def test_generation_paths_enter_the_registered_attention(monkeypatch):
    from contextlib import contextmanager

    from density import hsa, wan
    from density.checks import euler

    entered = []

    @contextmanager
    def record(device):
        entered.append(str(device))
        yield

    monkeypatch.setattr(wan, "generation_attention", record)
    monkeypatch.setattr(hsa, "generation_attention", record)
    tr = euler.tiny_transformer(torch.float32, "cpu")
    latents, pe, ne = euler.tiny_inputs("cpu")
    t = torch.tensor(900.0)
    wan.WanGuidedVelocity(tr, pe, ne, 6.0)(latents, t, 0)
    hsa.WanHSAGuidedVelocity(tr, pe, ne, 6.0)(latents, t, 0, torch.arange(192))
    assert entered == ["cpu", "cpu"]


def test_decode_latents_sets_and_restores_the_registered_conv_precision(monkeypatch):
    from density import wan

    seen = []

    class FakeVAE:
        dtype = torch.float32
        config = type("C", (), {"z_dim": 16, "latents_mean": [0.0] * 16, "latents_std": [1.0] * 16})()

        def decode(self, latents, return_dict=False):
            seen.append(torch.backends.cudnn.conv.fp32_precision)
            return (latents,)

    before = torch.backends.cudnn.conv.fp32_precision
    wan.decode_latents(FakeVAE(), torch.zeros(1, 16, 1, 2, 2))
    wan.decode_latents(FakeVAE(), torch.zeros(1, 16, 1, 2, 2), conv_fp32_precision="ieee")
    assert seen == [wan.VAE_CONV_FP32_PRECISION, "ieee"] and wan.VAE_CONV_FP32_PRECISION == "tf32"
    assert torch.backends.cudnn.conv.fp32_precision == before


def test_sam_conv_precision_is_registered_strict_fp32_and_restored():
    from density.masks import registered, sam

    before = torch.backends.cudnn.conv.fp32_precision
    with sam.conv_precision():
        assert torch.backends.cudnn.conv.fp32_precision == registered.SAM_CONV_FP32_PRECISION == "ieee"
    assert torch.backends.cudnn.conv.fp32_precision == before
