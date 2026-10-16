"""Unit tests for the mask-side guards of 2026-10-07: the CPU thread cap, the registered ROIAlign kernel, the mask
canary, the volume stop's allocated size, and Session F's gate."""

import importlib
import importlib.util
import json
import socket
from pathlib import Path

import pytest
import torch

from density import cpu_threads, mask_canary, volume
from density.masks import roi_align

ROOT = Path(__file__).resolve().parents[1]
MASK_WEIGHTS = (ROOT / "weights" / "masks").exists()


def test_quota_threads_rounds_the_cgroup_quota_down(tmp_path):
    cpu_max = tmp_path / "cpu.max"
    for text, expected in (("340000 100000", 3), ("2210000 100000", 22), ("50000 100000", 1), ("max 100000", None), ("", None)):
        cpu_max.write_text(text)
        assert cpu_threads.quota_threads(cpu_max) == expected, text
    assert cpu_threads.quota_threads(tmp_path / "missing") is None


def test_without_a_quota_nothing_changes(tmp_path):
    before = (torch.get_num_threads(), torch.get_num_interop_threads())
    assert cpu_threads.configure(tmp_path / "missing") == {"quota_threads": None}
    assert (torch.get_num_threads(), torch.get_num_interop_threads()) == before


def test_models_keep_onnxruntimes_default_without_a_quota(monkeypatch):
    from density.masks.pipeline import Models

    monkeypatch.setattr(cpu_threads, "quota_threads", lambda: None)  # the Mac: no cgroup quota
    assert Models("cpu").cpu_threads is None
    assert Models("cpu", cpu_threads=3).cpu_threads == 3
    monkeypatch.setattr(cpu_threads, "quota_threads", lambda: 3)
    assert Models("cpu").cpu_threads == 3


def test_roi_align_kernel_is_installed_on_the_module_and_checked():
    roi_align.install()
    module = importlib.import_module("detectron2.layers.roi_align")
    assert module.roi_align is roi_align._kernel
    x = torch.randn(1, 4, 16, 16)
    rois = torch.tensor([[0.0, 1.0, 1.0, 9.0, 9.0]])
    with torch.no_grad():
        with roi_align.checked("test"):
            out = module.roi_align(x, rois, (2, 2), 1.0, 0, True)
        assert torch.equal(out, torch.ops.torchvision.roi_align(x, rois, 1.0, 2, 2, 0, True))
        with pytest.raises(RuntimeError, match="not called"):
            with roi_align.checked("test"):
                pass
    with pytest.raises(RuntimeError, match="inference only"):
        module.roi_align(x, rois, (2, 2), 1.0, 0, True)


@pytest.mark.skipif(not MASK_WEIGHTS, reason="mask-pipeline weights not downloaded")
def test_detectron2_predictors_run_through_the_kernel():
    import numpy as np

    from density.masks.detectors import MaskRCNNPersonDetector

    frame = np.zeros((480, 832, 3), dtype=np.uint8)
    detector = MaskRCNNPersonDetector("cpu")
    start = roi_align.calls()
    detector.detect(frame)
    assert roi_align.calls() > start


def test_mask_canary_judges_every_model_and_names_both_pods():
    registered = {"pod": {"host": "reg", "gpu": "G", "driver": "1", "cpu": "AMD EPYC 9355 32-Core Processor", "kernel": "k"},
                  "hashes": {"scrfd": "a", "sam_face": "b", "mask_rcnn": "c"}}
    host = {"host": "new", "gpu": "G", "driver": "2", "cpu": "Intel(R) Xeon(R)", "kernel": "k"}
    verdict = mask_canary.judge({"scrfd": "a", "sam_face": "x", "mask_rcnn": "c"}, registered, host)
    assert not verdict["passed"] and verdict["differ"] == ["sam_face"]
    for text in ("sam_face", "new", "Intel(R) Xeon(R)", "AMD EPYC 9355", "Stop"):
        assert text in verdict["stop"]
    assert mask_canary.judge(dict(registered["hashes"]), registered, host) == {"passed": True, "differ": [], "stop": None}
    assert not mask_canary.judge({**registered["hashes"], "extra": "d"}, registered, host)["passed"]


def test_mask_canary_refuses_to_record_what_it_did_not_exercise():
    counts = {"scrfd": [1, 1], "wilor": [1, 0], "hands23": [1, 1], "mask_rcnn": [2, 2], "roi_align_kernel_calls": 40, "sam_face": 10, "sam_hands": 0, "sam_body": 5}
    assert mask_canary.exercised(counts) == ["wilor", "sam_hands"]
    assert mask_canary.exercised({**counts, "wilor": [1, 1], "sam_hands": 3, "roi_align_kernel_calls": 0}) == ["the ROIAlign kernel"]


def test_mask_canary_require_passed_wants_a_check_on_this_host_at_this_code(tmp_path, monkeypatch):
    code = {"git_sha": "a" * 40, "git_dirty": False}
    monkeypatch.setattr(mask_canary.euler, "git_state", lambda: dict(code))
    record = {"passed": True, "host": {"host": socket.gethostname()}, "code": dict(code), "stop": None}

    def write(**changes):
        (tmp_path / "mask_canary.json").write_text(json.dumps({**record, **changes}))

    with pytest.raises(mask_canary.CanaryNotPassed, match="no mask canary record"):
        mask_canary.require_passed(tmp_path)
    write()
    assert mask_canary.require_passed(tmp_path)["passed"]
    for changes, message in (({"recorded": "file"}, "registration"), ({"passed": False, "stop": "mismatch"}, "mismatch"),
                             ({"host": {"host": "elsewhere"}}, "not this one"), ({"code": {"git_sha": "b" * 40, "git_dirty": False}}, "run it again")):
        write(**changes)
        with pytest.raises(mask_canary.CanaryNotPassed, match=message):
            mask_canary.require_passed(tmp_path)


def test_volume_stop_uses_dus_allocated_size(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(volume, "usage", lambda root=None, apparent=False, timeout=1800: seen.append(apparent) or {"used_bytes": 1})
    volume.require_room(tmp_path, stop_bytes=10)
    assert seen == [False]


def _gate():
    spec = importlib.util.spec_from_file_location("rerun_gate", ROOT / "scripts" / "pod" / "rerun_gate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rerun(tmp_path, t26=(True, True), failed=("Conversion: WiLoR rebuilt from data-only files gives the original checkpoint's detections bitwise",), clips=("face", "hands", "body", "landscape")):
    (tmp_path / "mask_check").mkdir(parents=True, exist_ok=True)
    (tmp_path / "mask_check" / "mask_pipeline.json").write_text(json.dumps({"clip_results": {c: {} for c in clips}}))
    rows = [{"test": f"T26 repeated CUDA run gives identical masks: {c}", "passed": p} for c, p in zip(("face", "hands"), t26)] + [{"test": "T26 SAM 2.1 on MPS", "passed": None}]
    (tmp_path / "mask_check.json").write_text(json.dumps({"result": {"report": str(tmp_path / "nowhere.json"), "t26_rows": rows, "failed_rows": [{"test": t, "passed": False} for t in failed]}}))
    return tmp_path


def test_rerun_gate_passes_only_a_validated_mask_check(tmp_path):
    gate = _gate().gate
    assert gate(_rerun(tmp_path / "ok")) == []
    assert gate(_rerun(tmp_path / "t26", t26=(True, False)))
    assert gate(_rerun(tmp_path / "rows", failed=("Conversion: WiLoR x", "T24 something")))
    assert gate(_rerun(tmp_path / "clips", clips=("face", "hands", "landscape")))


def test_mask_projections_use_the_registered_configuration():
    from density.checks import smoke, smoke_report

    diag = {"mask_roi_align": {"pipeline": {"references": {"smoke-1": {"wall_seconds": 18.0}, "smoke-2": {"wall_seconds": 345.7}, "body": {"wall_seconds": 94.3}}}}}
    ctx = smoke.Context(Path("/nonexistent"), Path("/nonexistent"), "cpu", Path("w"), None, False)
    seconds = smoke_report.mask_seconds({}, ctx, diag)
    assert {k: v[0] for k, v in seconds.items()} == {"face": 18.0, "manipulation": 345.7, "body": 94.3}


def test_registered_mask_canary_covers_every_model_and_matches_session_f():
    registered = json.loads((ROOT / "configs" / "mask_canary.json").read_text())
    assert set(registered["hashes"]) == {"scrfd", "wilor", "hands23", "mask_rcnn", "sam_face", "sam_hands", "sam_body"}
    assert mask_canary.exercised(registered["counts"]) == []
    assert registered["pod"]["cpu"] and registered["pod"]["gpu"] and registered["code"]["git_dirty"] is False
    session_f = ROOT / "results" / "checks" / "smoke_test" / "rerun" / "mask_canary" / "registered.json"
    if session_f.exists():
        assert json.loads(session_f.read_text())["hashes"] == registered["hashes"]
