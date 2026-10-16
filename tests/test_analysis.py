"""The frozen analysis: equations 24-28, the bootstrap, the decision rules, Holm, the code freeze, and the synthetic
validations Milestone 5 requires."""

import json

import numpy as np
import pytest

from density import analysis
from density.checks.analysis_validation import synthetic_rows, validations


def test_seed_means_and_the_floor():
    rows = [{"prompt": 1, "seed": s, "condition": c, "masked_lpips": v} for s, c, v in ((1, "C3", 0.1), (2, "C3", 0.3), (1, "C4", 0.1), (2, "C4", 0.1))]
    means = analysis.seed_means(rows, ("C3", "C4"), ("masked_lpips",))
    assert means[1]["C3"]["masked_lpips"] == pytest.approx(0.2) and means[1]["C4"]["masked_lpips"] == pytest.approx(0.1)
    assert analysis.relative(np.array([0.10]), np.array([0.085]))[0] == pytest.approx(0.15)  # equation 26's worked example
    assert analysis.relative(np.array([0.0]), np.array([1e-7]))[0] == pytest.approx(-0.1)  # the 1e-6 floor
    with pytest.raises(analysis.Blocked):
        analysis.seed_means(rows[:-1], ("C3", "C4"), ("masked_lpips",))


def test_bootstrap_indices_are_fixed_and_stratified():
    a, b = analysis.resample_indices(resamples=1000), analysis.resample_indices(resamples=1000)
    assert np.array_equal(a, b) and a.shape == (1000, 72)
    assert a[:, :36].max() < 36 and a[:, 36:].min() >= 36
    rng = np.random.default_rng(20261005)
    assert np.array_equal(a[:, :36], rng.integers(0, 36, size=(1000, 36)))


def test_guardrail_endpoints_are_bonferroni_for_seven():
    rows, strata = synthetic_rows(0.15)
    r = analysis.analyze(rows, strata, resamples=2000)
    assert r["guardrails"]["endpoints"] == pytest.approx((0.003571, 0.996429), abs=1e-6)


@pytest.mark.parametrize("statuses, ci, median, expected", [
    ({"g": "pass"}, (0.02, 0.2), 0.12, "GO"),
    ({"g": "pass"}, (0.02, 0.2), 0.09, "UNCERTAIN"),  # median below 10 percent
    ({"g": "pass"}, (-0.01, 0.2), 0.12, "UNCERTAIN"),  # lower bound not above zero
    ({"g": "uncertain"}, (0.02, 0.2), 0.12, "UNCERTAIN"),
    ({"g": "confirmed harm"}, (0.02, 0.2), 0.12, "NO-GO"),  # harm is checked before GO
    ({"g": "pass"}, (-0.02, 0.049), 0.01, "NO-GO"),  # upper bound below 5 percent
])
def test_decision_rules(statuses, ci, median, expected):
    guard = {"guardrails": {g: {"status": s} for g, s in statuses.items()}}
    assert analysis.decide({"ci95": ci, "median_delta": median}, guard)["outcome"] == expected


def test_holm_step_down():
    adjusted = analysis.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": None})
    assert adjusted == {"a": pytest.approx(0.03), "b": pytest.approx(0.06), "c": pytest.approx(0.06), "d": None}


def test_synthetic_validations():
    out = validations(replicates=10)
    assert out["improvement_15_percent"]["outcome"] == "GO"
    assert out["no_effect"]["passed"] and out["injected_harm"]["passed"] and out["missing_metric"]["outcome"] == "BLOCKED"
    assert out["passed"]


def test_the_analysis_refuses_changed_code(tmp_path, monkeypatch):
    path = tmp_path / "freeze.json"
    state = analysis.code_state()
    path.write_text(json.dumps(state))
    assert analysis.require_frozen(path)["files"] == state["files"]
    path.write_text(json.dumps({**state, "files": {**state["files"], "src/density/analysis.py": "0" * 64}}))
    with pytest.raises(analysis.CodeChanged):
        analysis.require_frozen(path)
    path.write_text(json.dumps({**state, "numpy": "1.0"}))
    with pytest.raises(analysis.CodeChanged):
        analysis.require_frozen(path)


def test_secondary_families_and_controls():
    rows, strata = synthetic_rows(0.15)
    rng = np.random.default_rng(3)
    block = lambda conditions, prompts: [{"prompt": p, "seed": 1, "condition": c, "masked_lpips": rng.lognormal(np.log(0.1), 0.2)} for p in prompts for c in conditions]
    face8, manip8 = list(range(1, 9)), list(range(37, 45))
    secondary = {
        "full body": block(("C3", "C4"), range(101, 117)),
        "75 percent": block(("C3@HSA-75B", "C4@HSA-75B"), face8 + manip8),
        "switch points": block(("C3@switch-20", "C4@switch-20", "C3@switch-30", "C4@switch-30"), face8 + manip8),
        "C5 and C6": block(("C4", "C5", "C6"), list(range(1, 19)) + list(range(37, 55))),
        "controls": [{"prompt": p, "seed": 1, "condition": c, "full_frame_lpips": 0.05} for p in range(201, 217) for c in ("C3", "C4")],
    }
    strata.update({p: "body" for p in range(101, 117)} | {p: "control" for p in range(201, 217)})
    r = analysis.analyze(rows, strata, secondary, resamples=2000)
    fam = r["secondary"]
    assert set(fam["strata"]) == {"face", "manipulation", "body"} and set(fam["budget_switch"]) == {"75 percent", "switch-20", "switch-30"}
    assert set(fam["ablations"]) == {"C5 vs C4", "C6 vs C4"}
    assert all("holm_p" in v for f in fam.values() for v in f.values())
    assert r["controls_check"]["max_abs_c4_minus_c3_full_frame_lpips"] == 0.0


def test_collect_accepts_only_quality_scored_on_lossless_frames(tmp_path):
    from density.run import manifest as M
    from density.run import plan
    from density.run.store import Run, write_json_atomic

    run = Run(tmp_path / "run")
    ref = M.Job("reference", 1, "face", "primary", 1001, "sh8g6", "C1")
    gen = M.condition_job(ref, plan.GridItem("C3", "HSA-50"), "primary seed 1", f"mask-{ref.group}")
    score = M.score_job(ref, [gen.id], f"mask-{ref.group}")
    run.write_manifest({"run": "run", "kind": "main", "dry_run": False, "prompts_csv_sha256": "x", "jobs": [j.to_dict() for j in (ref, gen, score)]}, "test")
    write_json_atomic(run.job_dir(gen) / "record.json", {"fidelity": {"masked_lpips": 0.1, "full_frame_lpips": 0.05, "outside_lpips": 0.04}})
    q = dict.fromkeys(("subject_consistency", "background_consistency", "motion_smoothness", "aesthetic_quality", "imaging_quality"), 90.0)
    write_json_atomic(run.job_dir(score) / "record.json", {"videos": {gen.id: {"quality_input": "lossless frames", "quality": q}}})
    rows, _, _ = analysis.collect(run.folder)
    assert rows[0]["imaging_quality"] == 90.0
    write_json_atomic(run.job_dir(score) / "record.json", {"videos": {gen.id: {"quality": q}}})  # a record without the registered input
    with pytest.raises(analysis.Blocked, match="lossless frames"):
        analysis.collect(run.folder)
