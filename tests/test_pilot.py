"""The pilot's rules as code: the setting, lambda, no-op, C2 and reachability rules, and reserve sizing."""

import math

import pytest

from density import pilot


def rec(masked=0.1, outside=0.05, full=0.06, no_op=False):
    return {"fidelity": {"masked_lpips": masked, "outside_lpips": outside, "full_frame_lpips": full}, "allocation_diagnostics": {"roi": {"no_op": no_op}}}


def records(n=12, **per_condition):
    out = {}
    for p in range(1, n + 1):
        out[p] = {"C3": rec(), "C2": rec(full=0.10), **{f"C4@{lam:g}": rec() for lam in (1, 3, 10)}}
        for key, fields in per_condition.items():
            out[p][key] = rec(**(fields(p) if callable(fields) else fields))
    return out


def test_setting_weights_strata_equally_and_ties_select_8_6():
    cov = {"sh8g6": {"face": [1.0, 0.8], "manipulation": [0.2]}, "sh5g5": {"face": [0.5], "manipulation": [0.6, 0.6]}}
    t = pilot.setting_table(cov)
    assert t["scores"] == {"sh8g6": pytest.approx(0.55), "sh5g5": pytest.approx(0.55)} and t["setting"] == "sh8g6" and t["tie"]
    cov["sh5g5"]["manipulation"] = [0.7]
    assert pilot.setting_table(cov)["setting"] == "sh5g5"


def test_lambda_rule_qualifies_below_0_01_then_takes_the_lowest_median_masked_lpips():
    r = records(**{"C4@1": {"masked": 0.09, "outside": 0.055}, "C4@3": {"masked": 0.08, "outside": 0.0599}, "C4@10": {"masked": 0.05, "outside": 0.0601}})
    out = pilot.lambda_rule(r)
    assert out["table"][10.0]["qualifies"] is False and out["table"][3.0]["qualifies"] is True  # 0.0101 is not below 0.01
    assert out["lambda"] == 3.0 and out["stop"] is None
    tied = pilot.lambda_rule(records(**{"C4@1": {"masked": 0.08}, "C4@3": {"masked": 0.08}, "C4@10": {"masked": 0.09}}))
    assert tied["lambda"] == 1.0  # ties select the smaller
    none = pilot.lambda_rule(records(**{k: {"outside": 0.07} for k in ("C4@1", "C4@3", "C4@10")}))
    assert none["lambda"] is None and "stop" in none["stop"]


def test_noop_rule_stops_when_more_than_half_are_unchanged_at_every_lambda():
    half = records(**{k: (lambda p: {"no_op": p <= 6}) for k in ("C4@1", "C4@3", "C4@10")})
    assert pilot.noop_rule(half)["stop"] is None and pilot.noop_rule(half)["unchanged_at_every_lambda"] == [1, 2, 3, 4, 5, 6]
    more = records(**{k: (lambda p: {"no_op": p <= 7}) for k in ("C4@1", "C4@3", "C4@10")})
    assert pilot.noop_rule(more)["stop"]
    mixed = records(**{"C4@1": (lambda p: {"no_op": True}), "C4@3": (lambda p: {"no_op": p <= 3}), "C4@10": (lambda p: {"no_op": True})})
    assert pilot.noop_rule(mixed)["unchanged_at_every_lambda"] == [1, 2, 3] and pilot.noop_rule(mixed)["stop"] is None


def test_c2_rule_uses_the_ratio_of_medians():
    ok = pilot.c2_rule(records(C3={"full": 0.07}, C2={"full": 0.10}))
    assert ok["ratio"] == pytest.approx(0.7) and ok["stop"] is None
    bad = pilot.c2_rule(records(C3=(lambda p: {"full": 0.08 if p > 1 else 0.01}), C2={"full": 0.10}))
    assert bad["ratio"] == pytest.approx(0.8) and bad["stop"]


def test_reserves_cover_36_passing_prompts_with_95_percent_probability():
    from scipy.stats import nbinom

    assert pilot.reserves_needed(1.0) == 0
    r = pilot.reserves_needed(0.7)
    assert nbinom.cdf(r, 36, 0.7) >= 0.95 > nbinom.cdf(r - 1, 36, 0.7)
    assert pilot.reserves_needed(0.0) is None
    out = pilot.pass_rates({1: "pass", 2: "pass", 3: "fail: x", 4: "pass"}, {1: "face", 2: "face", 3: "face", 4: "manipulation"})
    face = out["strata"]["face"]
    assert face["p_seed"] == pytest.approx(2 / 3) and face["reserves_at_estimate"] == pilot.reserves_needed((2 / 3) ** 2)
    assert set(face) == {"holdout_prompts", "passed", "p_seed", "reserves_at_estimate"}  # the registered estimate only


def test_reachability_uses_bonferroni_z_and_the_main_sample_size():
    assert pilot.z_bonferroni() == pytest.approx(2.6901, abs=1e-4)
    deg = {g: [0.0, 0.002] * 6 for g in ("full_frame_lpips", "outside_lpips")}
    deg.update({d: [0.0, 4.0] * 6 for d in pilot.QUALITY_DIMENSIONS})
    out = pilot.reachability(deg)
    sd = pilot.np.std([0.0, 0.002] * 6, ddof=1)
    assert out["guardrails"]["outside_lpips"]["half_width"] == pytest.approx(2.6901 * sd / math.sqrt(72), rel=1e-4)
    assert out["guardrails"]["outside_lpips"]["reachable"] and not out["guardrails"]["imaging_quality"]["reachable"]
    assert set(out["unreachable"]) == set(pilot.QUALITY_DIMENSIONS)


def test_degradations_have_the_protocols_signs():
    r = {1: {"C3": rec(full=0.05, outside=0.04), "C4@3": rec(full=0.06, outside=0.03)}}
    q = {1: {"C3": dict.fromkeys(pilot.QUALITY_DIMENSIONS, 90.0), "C4@3": dict.fromkeys(pilot.QUALITY_DIMENSIONS, 89.0)}}
    d = pilot.degradations(r, q, 3.0)
    assert d["full_frame_lpips"] == [pytest.approx(0.01)] and d["outside_lpips"] == [pytest.approx(-0.01)]
    assert d["subject_consistency"] == [1.0]  # quality: C3 minus C4, so a drop in C4 is positive damage


def test_compression_effect_on_paired_degradation():
    mp4 = {1: {"C3": dict.fromkeys(pilot.QUALITY_DIMENSIONS, 90.0), "C4@3": dict.fromkeys(pilot.QUALITY_DIMENSIONS, 89.5)},
           2: {"C3": dict.fromkeys(pilot.QUALITY_DIMENSIONS, 80.0), "C4@3": dict.fromkeys(pilot.QUALITY_DIMENSIONS, 80.0)}}
    lossless = {1: {"C3": dict.fromkeys(pilot.QUALITY_DIMENSIONS, 91.0), "C4@3": dict.fromkeys(pilot.QUALITY_DIMENSIONS, 90.8)},
                2: {"C3": dict.fromkeys(pilot.QUALITY_DIMENSIONS, 81.0), "C4@3": dict.fromkeys(pilot.QUALITY_DIMENSIONS, 81.0)}}
    out = pilot.compression_effect(mp4, lossless, 3.0)
    d = out["dimensions"]["imaging_quality"]
    assert d["mean_on_mp4"] == pytest.approx(0.25) and d["mean_on_lossless"] == pytest.approx(0.1) and d["mean_difference"] == pytest.approx(0.15)
    assert pilot.compression_effect(mp4, {}, 3.0) is None
