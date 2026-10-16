"""The metric canary: record from two identical fresh processes, check later sessions bit for bit, and gate jobs."""

import json
import socket

import pytest

from density import metric_canary as MC
from density.checks import euler


@pytest.fixture
def children(monkeypatch):
    queue = []
    monkeypatch.setattr(MC, "child", lambda machine, device: queue.pop(0))
    monkeypatch.setattr(MC, "host_info", lambda: {"host": socket.gethostname(), "gpu": None, "cpu": "test", "platform": "test"})
    monkeypatch.setattr(euler, "git_state", lambda: {"git_sha": "abc", "git_dirty": False})
    return queue


def test_first_session_records_from_two_identical_processes(tmp_path, children):
    children += [{"h100": {"x": 1.5}}, {"h100": {"x": 1.5}}]
    rec = MC.run("h100", "cuda", tmp_path / "none.json", tmp_path)
    assert rec["passed"] and rec["recorded"]
    reg = json.loads((tmp_path / "registered_h100.json").read_text())
    assert reg["digest"] == MC.digest({"h100": {"x": 1.5}}) and reg["scores"] == {"h100": {"x": 1.5}}
    assert MC.require_passed("h100", tmp_path)["recorded"]


def test_processes_that_disagree_register_nothing(tmp_path, children):
    children += [{"h100": {"x": 1.5}}, {"h100": {"x": 1.5000001}}]
    rec = MC.run("h100", "cuda", tmp_path / "none.json", tmp_path)
    assert not rec["passed"] and not (tmp_path / "registered_h100.json").exists()
    with pytest.raises(MC.CanaryNotPassed):
        MC.require_passed("h100", tmp_path)


def test_later_sessions_check_against_the_registration(tmp_path, children):
    registered = tmp_path / "metric_canary.json"
    registered.write_text(json.dumps({"slice": {"pod": {"host": "pod1"}, "code": {}, "utc": "t", "digest": MC.digest({"slice": {"q": 2.0}})}}))
    children += [{"slice": {"q": 2.0}}]
    assert MC.run("slice", "cuda", registered, tmp_path)["passed"]
    children += [{"slice": {"q": 2.0000000001}}]
    rec = MC.run("slice", "cuda", registered, tmp_path)
    assert not rec["passed"] and "differ from the registration" in rec["stop"]


def test_require_passed_needs_this_host_and_this_code(tmp_path, children, monkeypatch):
    children += [{"h100": {"x": 1}}, {"h100": {"x": 1}}]
    MC.run("h100", "cuda", tmp_path / "none.json", tmp_path)
    monkeypatch.setattr(euler, "git_state", lambda: {"git_sha": "def", "git_dirty": False})
    with pytest.raises(MC.CanaryNotPassed, match="ran at code"):
        MC.require_passed("h100", tmp_path)
    monkeypatch.setattr(socket, "gethostname", lambda: "elsewhere")
    with pytest.raises(MC.CanaryNotPassed, match="not this one"):
        MC.require_passed("h100", tmp_path)
