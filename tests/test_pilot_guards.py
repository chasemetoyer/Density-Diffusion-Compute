"""Unit tests for the guards Chase registered for the pilot on 2026-10-07: the per-pod canary, strict
deterministic mode, the du-based volume stop, and the frozen tolerances."""

import json
import socket
import warnings
from pathlib import Path

import pytest
import torch

from density import canary, determinism, volume
from density import tolerances as registered_tolerances
from density.checks import smoke, smoke_report

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def restore_determinism():
    enabled, warn_only = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    benchmark, strict = torch.backends.cudnn.benchmark, determinism._strict
    with warnings.catch_warnings():
        yield
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
    torch.backends.cudnn.benchmark = benchmark
    determinism._strict = strict


def test_strict_mode_stops_on_operations_and_warnings_about_determinism(restore_determinism):
    settings = determinism.configure(strict=True)
    assert settings["deterministic_warn_only"] is False and settings["deterministic_warnings_are_errors"] is True
    with pytest.raises(RuntimeError, match="deterministic"):
        torch.zeros(3).put_(torch.tensor([0]), torch.tensor([1.0]), accumulate=False)
    log = determinism.WarningLog()
    with pytest.raises(UserWarning, match="(?i)deterministic"):
        with log.capture("job"):  # the log's own "always" filter must not swallow it in strict mode
            warnings.warn("scatter_add does not have a Deterministic implementation")
    with log.capture("job"):
        warnings.warn("an unrelated warning")
    assert [e["message"] for e in log.summary()] == ["an unrelated warning"]


def test_warn_only_mode_records_what_strict_mode_raises(restore_determinism):
    settings = determinism.configure()
    assert settings["deterministic_warn_only"] is True and settings["deterministic_warnings_are_errors"] is False
    log = determinism.WarningLog()
    with log.capture("check"):
        torch.zeros(3).put_(torch.tensor([0]), torch.tensor([1.0]), accumulate=False)
        warnings.warn("scatter_add does not have a Deterministic implementation")
    assert len(log.deterministic_warnings()) == 2


def test_volume_usage_counts_bytes_and_stops_at_the_limit(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"x" * 300_000)
    used = volume.usage(tmp_path)["used_bytes"]
    assert used >= 300_000
    with pytest.raises(volume.VolumeFull, match="GB"):
        volume.require_room(tmp_path, stop_bytes=100_000)
    assert volume.require_room(tmp_path, stop_bytes=10**12)["used_bytes"] == used
    assert volume.STOP_BYTES == 112 * 10**9  # registered 2026-10-07, with the volume at 125 GB


def test_run_step_stops_cleanly_before_a_step_when_the_volume_is_full(tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setitem(smoke.STEPS, "sheet", lambda ctx: calls.append(1) or {})
    monkeypatch.setitem(smoke.DEPENDS, "sheet", ())
    monkeypatch.setattr(volume, "VOLUME", tmp_path)
    monkeypatch.setattr(volume, "STOP_BYTES", 1)
    (tmp_path / "data.bin").write_bytes(b"x" * 10)
    ctx = smoke.Context(tmp_path / "out", tmp_path / "scratch", "cpu", tmp_path / "w", "E", dry_run=False)
    assert smoke.run_step(ctx, "sheet") == 3
    assert calls == [] and not (ctx.out / "sheet.json").exists()
    assert "STOP" in capsys.readouterr().out
    monkeypatch.setattr(volume, "STOP_BYTES", 10**15)
    assert smoke.run_step(ctx, "sheet") == 0 and calls == [1]
    assert json.loads((ctx.out / "sheet.json").read_text())["volume_before"]["used_bytes"] > 0


def test_frozen_tolerances_are_bitwise():
    assert set(registered_tolerances.FROZEN) == {
        "Repeated dense run (C1 twice, same device)", "All HSA groups dense vs the dense loop",
        "Fast path vs masked-dense oracle at full resolution, C3 (T3)", "Fast path vs masked-dense oracle at full resolution, C4 (T3)",
    }
    test = next(iter(registered_tolerances.FROZEN))
    a = [torch.zeros(2), torch.ones(2)]
    assert registered_tolerances.within(test, a, [x.clone() for x in a])
    assert not registered_tolerances.within(test, a, [a[0], torch.nextafter(a[1], torch.full((2,), 2.0))])
    assert not registered_tolerances.within(test, a, a[:1])


@pytest.mark.skipif(not (ROOT / "results" / "checks" / "smoke_test" / "flash").is_dir(), reason="Session D's records are not on this machine")
def test_report_marks_exactly_the_frozen_rows_registered():
    flash = ROOT / "results" / "checks" / "smoke_test" / "flash"
    records, _, _, _ = smoke_report.load(smoke.Context(flash, flash, "cpu", ROOT / "weights", None, False))
    rows = smoke_report.tolerances(records, registered=True)
    assert {r["test"] for r in rows if r.get("registered")} == set(registered_tolerances.FROZEN)
    assert all(r["status"] == "proposed, not applied" for r in smoke_report.tolerances(records))


def test_protocol_blocks_include_c2_in_the_pilot():
    block = next(b for b in smoke_report.PROTOCOL_BLOCKS if b[0] == "Pilot HSA reproduction")
    assert block[1:] == ("pilot", {"C2": 12}, 6)


def test_canary_reproduces_the_generate_step_on_tiny_inputs(tmp_path):
    ctx = smoke.Context(tmp_path / "smoke", tmp_path / "scratch", "cpu", tmp_path / "w", None, dry_run=True)
    for step in ("noise", "encode", "generate"):
        assert smoke.run_step(ctx, step) == 0
    canary_ctx = smoke.Context(tmp_path / "canary", tmp_path / "canary", "cpu", tmp_path / "w", None, dry_run=True)
    record = canary.run(canary_ctx, canary.registered_from_records(ctx.out), ctx.out / "inputs")
    assert record["passed"] and record["matches"] == {"final_latents": True, "decoded_frames": True}, record.get("stop") or record.get("error")
    assert json.loads((canary_ctx.out / "canary.json").read_text())["passed"]


def test_canary_failure_names_what_differs_and_both_pods():
    got = {"final_latents_sha256": canary.REGISTERED["expected"]["final_latents_sha256"], "frames_uint8_sha256": "0" * 64, "finite": True}
    host = {"host": "abc123", "gpu": "NVIDIA H100 80GB HBM3", "driver": "999.1", "driver_cuda": "13.1", "cpu": "x86", "kernel": "6.9"}
    verdict = canary.judge(got, canary.REGISTERED, host)
    assert not verdict["passed"] and verdict["matches"] == {"final_latents": True, "decoded_frames": False}
    for text in ("decoded frames", "abc123", "999.1", "0d7e358ab3af", "580.126.09", "Stop"):
        assert text in verdict["stop"]
    got["frames_uint8_sha256"] = canary.REGISTERED["expected"]["frames_uint8_sha256"]
    assert canary.judge(got, canary.REGISTERED, host) == {"passed": True, "matches": {"final_latents": True, "decoded_frames": True}, "stop": None}


def test_require_passed_accepts_only_this_host_at_this_code(tmp_path, monkeypatch):
    code = {"git_sha": "a" * 40, "git_dirty": False}
    monkeypatch.setattr(canary.euler, "git_state", lambda: dict(code))
    record = {"passed": True, "dry_run": False, "host": {"host": socket.gethostname()}, "code": dict(code), "stop": None}

    def write(**changes):
        (tmp_path / "canary.json").write_text(json.dumps({**record, **changes}))

    with pytest.raises(canary.CanaryNotPassed, match="no canary record"):
        canary.require_passed(tmp_path)
    write()
    assert canary.require_passed(tmp_path)["passed"]
    for changes, message in (({"passed": False, "stop": "mismatch"}, "did not pass"), ({"dry_run": True}, "dry run"),
                             ({"host": {"host": "elsewhere"}}, "not this one"), ({"code": {"git_sha": "b" * 40, "git_dirty": False}}, "run it again")):
        write(**changes)
        with pytest.raises(canary.CanaryNotPassed, match=message):
            canary.require_passed(tmp_path)


@pytest.mark.skipif(not (ROOT / "results" / "checks" / "smoke_test.json").exists(), reason="the smoke report is not on this machine")
def test_canary_hashes_are_the_reports_flash_hashes():
    hashes = json.loads((ROOT / "results" / "checks" / "smoke_test.json").read_text())["hashes"]
    ref = hashes["references_flash"]["smoke-1"]
    assert canary.REGISTERED["expected"] == {"final_latents_sha256": ref["final_latents_sha256"][0], "frames_uint8_sha256": ref["uint8_sha256"]}
    assert canary.REGISTERED["inputs"]["noise"]["tensor_sha256"] == hashes["starting_noise"]["smoke-1"]["tensor_sha256"]
    for name in ("smoke-1", "negative"):
        assert canary.REGISTERED["inputs"]["embeddings"]["sha256"][name] == hashes["embeddings"][name]["sha256"]


def test_room_counts_allocated_sizes_and_checks_du_every_30_minutes(tmp_path, monkeypatch):
    """The registered running total (Chase, 2026-10-07): st_blocks, not byte length; a full du at the start, after 30
    minutes, and when the total reaches the stop, which the du then decides; each full du reports any undercount."""
    now = [0.0]
    fulls = []
    used = [1000]

    def fake_du(root=None, stop_bytes=None):
        fulls.append(now[0])
        if used[0] >= stop_bytes:
            raise volume.VolumeFull("stop")
        return {"used_bytes": used[0], "seconds": 0.0}

    monkeypatch.setattr(volume, "require_room", fake_du)
    room = volume.Room(tmp_path, stop_bytes=10**9, clock=lambda: now[0])
    one = tmp_path / "one.bin"
    one.write_bytes(b"x")  # one byte, but at least one allocated block
    assert room.after_job("a", [one, tmp_path / "missing"])["added_bytes"] == one.stat().st_blocks * 512 >= 1
    assert room.before_job("b") is None and len(fulls) == 1
    now[0] = volume.FULL_CHECK_SECONDS
    used[0] = 1000 + 10 * 1024  # the volume grew more than the running total knew: reported as an undercount
    entry = room.before_job("c")
    assert entry["kind"] == "full" and entry["undercount_bytes"] == used[0] - (1000 + one.stat().st_blocks * 512) and len(fulls) == 2
    room.used = 10**9  # an overcounted total at the stop: a full du decides, and the session goes on
    assert room.before_job("d")["kind"] == "full" and room.used == used[0]
    room.used, used[0] = 10**9, 10**9  # now the du agrees: stop
    with pytest.raises(volume.VolumeFull):
        room.before_job("e")
    assert volume.FULL_CHECK_SECONDS == 1800
