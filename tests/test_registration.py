"""The registration builder: every registered value with its file and commit, and the pilot's slots."""

import json

import pytest

from density import registration
from density.run.store import write_json_atomic


def test_draft_lists_registered_values_with_commits(tmp_path):
    reg = registration.build(tmp_path, None, draft=True)
    names = {r["name"] for r in reg["registered"]}
    for expected in ("HSA presets (high/low tokens, window, low steps)", "need floor", "lambda grid", "tolerances", "volume stop (bytes, du allocated)",
                     "FULL_FRAME_LPIPS", "QUALITY_SCALE", "fixed MP4 encoding", "bootstrap", "MASK_GPU", "encode canary", "visual check selection"):
        assert expected in names
    assert all(r["file_sha256"] for r in reg["registered"] if r["name"] != "analysis freeze")
    drafting = next(r for r in reg["registered"] if r["name"] == "drafting procedure")["value"]
    assert drafting["seed"] == 20261009 and set(drafting["readings"]) == {"principle", "face", "manipulation", "body"}  # Chase's readings, 2026-10-07
    assert reg["pilot"]["status"].startswith("pending") and reg["values"] == {"setting": None, "lambda": None, "reserve_sizes": None}
    assert (tmp_path / "registration.md").read_text().startswith("# Registration (draft)")
    assert json.loads((tmp_path / "registration.json").read_text())["draft"] is True


def test_final_registration_needs_a_complete_pilot(tmp_path, monkeypatch):
    from density.checks import euler

    monkeypatch.setattr(euler, "git_state", lambda: {"git_sha": "abc", "git_dirty": False})
    with pytest.raises(RuntimeError, match="pilot is not complete"):
        registration.build(tmp_path / "out", None, draft=False)
    pilot = tmp_path / "pilot_run"
    rules = {"stops": [], "frozen_sha256": "f", "setting": {"setting": "sh8g6", "scores": {}, "per_stratum": {}, "tie": True},
             "lambda": {"lambda": 3.0, "table": {}}, "no_op": {"unchanged_at_every_lambda": []}, "c2": {"ratio": 0.6}, "pass_rates": {},
             "reachability": {"unreachable": []}, "excluded": {}}
    write_json_atomic(pilot / "pilot" / "rules.json", rules)
    with pytest.raises(RuntimeError, match="reserve sizes"):
        registration.build(tmp_path / "out", pilot, draft=False)
    sizes = registration.parse_reserve_sizes("face=24,manipulation=40,body=10,control=10")
    reg = registration.build(tmp_path / "out", pilot, draft=False, reserve_sizes=sizes)
    assert reg["values"] == {"setting": "sh8g6", "lambda": 3.0, "reserve_sizes": sizes} and reg["pilot"]["status"] == "complete"
    write_json_atomic(pilot / "pilot" / "rules.json", {**rules, "stops": ["C2 ratio too high"]})
    with pytest.raises(RuntimeError, match="stopped"):
        registration.build(tmp_path / "out2", pilot, draft=False, reserve_sizes=sizes)
    monkeypatch.setattr(euler, "git_state", lambda: {"git_sha": "abc", "git_dirty": True})
    with pytest.raises(RuntimeError, match="commit first"):
        registration.build(tmp_path / "out3", pilot, draft=False, reserve_sizes=sizes)
