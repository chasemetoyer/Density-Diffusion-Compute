"""The pilot's rules as code (protocol sections 5, 9 and 10), applied to a pilot run's records. Registered by Chase on
2026-10-07 (decisions 2, 3, 11 and 12), with the quality scores measured on lossless frames (decision 1).

1. Setting: shift 8 / guidance 6 or shift 5 / guidance 5, whichever has the higher mean face/hand detection coverage on
   the 12 holdout references, the two strata weighted equally; ties select 8 / 6. A reference's detection coverage is
   the share of its 81 frames where the stratum's registered detector (SCRFD for faces, WiLoR for hands) finds anything.
2. Lambda: among 1, 3 and 10, the candidates whose mean outside-ROI LPIPS degradation (C4 minus C3) is below 0.01; of
   those, the one with the lowest median masked LPIPS; ties select the smaller. If none qualifies, the pilot stops.
3. No-op: per lambda, the holdout prompts whose C4 allocation equals C3's for every token (equation 20). If more than
   half are unchanged at all three values, the pilot stops to investigate the score and masks.
4. HSA reproduction (C2): HSA-50's (C3's) median full-frame LPIPS over the holdout prompts should be at most about 70
   percent of C2's; otherwise the pilot stops to investigate.
   The comparison tolerates floating-point rounding only (1e-9).
5. Pass rates and reserves: each stratum's share of holdout references that pass eligibility and the audit, and the
   reserve size that covers 36 passing primary prompts (both seeds) with 95 percent probability: failures follow
   NegBin(36, p^2), p being the holdout's one-seed pass rate, the two seeds assumed to pass independently. Chase sets
   the registered sizes from this estimate after the pilot (six prompts per stratum make a lower bound unusable). Failed
   holdout prompts are excluded from the pilot's conditions and reported, never replaced; an exhausted reserve list in
   the main run stops it until the registration is amended.
6. Guardrail reachability: for each of the seven guardrails, 2.69 x SD / sqrt(72) of the pilot's paired degradations
   must be below its limit, or the guardrail could not pass even with no damage at the main run's size.
7. Descriptive: how far the CRF 10 encoding moves each quality guardrail's paired degradation (MP4 minus lossless).
"""

import math
from statistics import median

import numpy as np

from density.metrics.registered import GUARDRAILS, LPIPS_LIMIT, QUALITY_DIMENSIONS, QUALITY_LIMIT
from density.run import plan
from density.run.store import Run, read_json, utc_now, write_json_atomic

J = 7
MAIN_PROMPTS = 72
PRIMARY_PER_STRATUM = 36
RESERVE_COVERAGE = 0.95
LIMITS = {"full_frame_lpips": LPIPS_LIMIT, "outside_lpips": LPIPS_LIMIT, **{d: QUALITY_LIMIT for d in QUALITY_DIMENSIONS}}


def z_bonferroni(alpha: float = 0.05, j: int = J) -> float:
    from scipy.stats import norm

    return float(norm.ppf(1 - alpha / (2 * j)))


class PilotStop(RuntimeError):
    """A pilot rule says stop and revise, or investigate, before the main experiment."""


# 1. Setting


def setting_table(coverage: dict[str, dict[str, list[float]]]) -> dict:
    """coverage[setting][stratum] = per-reference detection coverage. Returns each setting's score and the choice."""
    scores = {s: (float(np.mean(by["face"])) + float(np.mean(by["manipulation"]))) / 2 for s, by in coverage.items()}
    best = max(scores.values())
    winners = [s for s, v in scores.items() if v == best]
    choice = plan.TIE_SETTING if len(winners) > 1 else winners[0]
    return {"scores": scores, "per_stratum": {s: {k: float(np.mean(v)) for k, v in by.items()} for s, by in coverage.items()},
            "setting": choice, "tie": len(winners) > 1, "rule": "higher mean detection coverage, strata weighted equally; ties select 8 / 6"}


def choose_setting(run: Run, rows: list[dict]) -> dict:
    coverage: dict[str, dict[str, list[float]]] = {s: {"face": [], "manipulation": []} for s in plan.SETTINGS}
    for job in run.jobs():
        if job.kind == "mask" and job.block == "holdout":
            rec = run.record(job)
            if rec is None:
                raise RuntimeError(f"{job.id} is not done; the setting rule needs every holdout reference's detections")
            coverage[job.setting][job.stratum].append(rec["detection_coverage"])
    out = {**setting_table(coverage), "coverage": coverage, "hsa_authors_settings": "none received", "utc": utc_now()}
    write_json_atomic(run.path("pilot", "setting.json"), out)
    return out


# 2-4. Lambda, no-op and C2, from the condition records


def holdout_records(run: Run) -> dict[int, dict[str, dict]]:
    """prompt -> condition key (C2, C3, C4@lambda) -> its record, for the frozen holdout prompts."""
    out: dict[int, dict[str, dict]] = {}
    for job in run.jobs():
        if job.kind == "condition":
            rec = run.record(job)
            if rec is None:
                raise RuntimeError(f"{job.id} is not done")
            key = job.condition if job.condition != "C4" else f"C4@{job.lam:g}"
            out.setdefault(job.prompt_id, {})[key] = rec
    return out


def lambda_rule(records: dict[int, dict[str, dict]]) -> dict:
    table = {}
    for lam in plan.LAMBDA_GRID:
        key = f"C4@{lam:g}"
        outside = [r[key]["fidelity"]["outside_lpips"] - r["C3"]["fidelity"]["outside_lpips"] for r in records.values()]
        masked = [r[key]["fidelity"]["masked_lpips"] for r in records.values()]
        table[lam] = {"mean_outside_degradation": float(np.mean(outside)), "median_masked_lpips": float(median(masked)), "prompts": len(masked)}
        table[lam]["qualifies"] = table[lam]["mean_outside_degradation"] < LPIPS_LIMIT
    qualifying = [lam for lam in plan.LAMBDA_GRID if table[lam]["qualifies"]]
    if not qualifying:
        return {"table": table, "lambda": None, "stop": "no lambda has mean outside-ROI LPIPS degradation below 0.01: stop and revise the pilot plan"}
    lam = min(qualifying, key=lambda x: (table[x]["median_masked_lpips"], x))
    return {"table": table, "lambda": lam, "stop": None,
            "rule": "lowest median masked LPIPS among lambdas with mean outside-ROI degradation below 0.01; ties select the smaller"}


def noop_rule(records: dict[int, dict[str, dict]]) -> dict:
    unchanged = {lam: sorted(p for p, r in records.items() if r[f"C4@{lam:g}"]["allocation_diagnostics"]["roi"]["no_op"]) for lam in plan.LAMBDA_GRID}
    at_all = sorted(set.intersection(*(set(v) for v in unchanged.values())))
    stop = len(at_all) > len(records) / 2
    return {"unchanged": {f"{lam:g}": v for lam, v in unchanged.items()}, "unchanged_at_every_lambda": at_all, "prompts": len(records),
            "stop": "more than half of the holdout prompts are unchanged at all three lambdas: investigate the score and masks" if stop else None}


def c2_rule(records: dict[int, dict[str, dict]], threshold: float = 0.70) -> dict:
    hsa = median(r["C3"]["fidelity"]["full_frame_lpips"] for r in records.values())
    c2 = median(r["C2"]["fidelity"]["full_frame_lpips"] for r in records.values())
    ratio = hsa / c2 if c2 > 0 else math.inf
    passed = ratio <= threshold + 1e-9  # the tolerance absorbs floating-point rounding only (0.07 / 0.10 is 0.7000000000000001)
    return {"median_full_frame_lpips": {"HSA-50 (C3)": hsa, "C2": c2}, "ratio": ratio, "threshold": threshold,
            "stop": None if passed else f"HSA-50's median full-frame LPIPS is {ratio:.1%} of C2's, above about 70 percent: investigate before the main run"}


# 5. Pass rates and reserve sizing


def reserves_needed(p_prompt: float, successes: int = PRIMARY_PER_STRATUM, coverage: float = RESERVE_COVERAGE) -> int | None:
    """The smallest R with P(failures before ``successes`` passes <= R) >= coverage, failures ~ NegBin(successes, p)."""
    from scipy.stats import nbinom

    if p_prompt <= 0:
        return None
    return int(nbinom.ppf(coverage, successes, p_prompt))


def pass_rates(states: dict[int, str], strata: dict[int, str]) -> dict:
    """Holdout pass rates (one seed per prompt) and the reserves each primary stratum would need for 36 passing prompts,
    whose two seeds must both pass (p_prompt = p_seed^2, assuming the seeds pass independently)."""
    out = {}
    for stratum in ("face", "manipulation"):
        ids = [p for p, s in strata.items() if s == stratum]
        passed = sum(states[p] == "pass" for p in ids)
        p_seed = passed / len(ids) if ids else 0.0
        out[stratum] = {"holdout_prompts": len(ids), "passed": passed, "p_seed": p_seed, "reserves_at_estimate": reserves_needed(p_seed**2)}
    return {"strata": out, "assumption": "both seeds must pass; seeds pass independently with the holdout's one-seed rate",
            "coverage": RESERVE_COVERAGE, "primary_per_stratum": PRIMARY_PER_STRATUM}


# 6. Guardrail reachability


def degradations(records: dict[int, dict[str, dict]], quality: dict[int, dict[str, dict]], lam: float) -> dict[str, list[float]]:
    """Paired degradation (equation 27) per guardrail: C4 minus C3 for LPIPS, C3 minus C4 for quality."""
    key = f"C4@{lam:g}"
    out = {g: [] for g in GUARDRAILS}
    for p, r in sorted(records.items()):
        for g in ("full_frame_lpips", "outside_lpips"):
            out[g].append(r[key]["fidelity"][g] - r["C3"]["fidelity"][g])
        for d in QUALITY_DIMENSIONS:
            out[d].append(quality[p]["C3"][d] - quality[p][key][d])
    return out


def reachability(deg: dict[str, list[float]], n_main: int = MAIN_PROMPTS) -> dict:
    z = z_bonferroni()
    table = {}
    for g, values in deg.items():
        sd = float(np.std(values, ddof=1)) if len(values) > 1 else math.nan
        half = z * sd / math.sqrt(n_main)
        table[g] = {"sd": sd, "half_width": half, "limit": LIMITS[g], "reachable": half < LIMITS[g], "mean": float(np.mean(values))}
    return {"z": z, "n_main": n_main, "guardrails": table, "unreachable": [g for g, t in table.items() if not t["reachable"]],
            "note": "the pilot has one seed per prompt; the main run averages two seeds, whose differences vary less, so this check is conservative"}


def quality_scores(run: Run, field: str = "quality") -> dict[int, dict[str, dict]]:
    """prompt -> condition key -> the five quality scores: the registered ones from lossless frames (``quality``), or the
    descriptive MP4 ones (``field="quality_mp4"``)."""
    out: dict[int, dict[str, dict]] = {}
    for job in run.jobs():
        if job.kind == "score":
            rec = run.record(job)
            if rec is None:
                raise RuntimeError(f"{job.id} is not done")
            for vid, entry in rec["videos"].items():
                if field not in entry:
                    continue
                gen = run.job(vid)
                key = "C1" if gen.kind == "reference" else gen.condition if gen.condition != "C4" else f"C4@{gen.lam:g}"
                out.setdefault(job.prompt_id, {})[key] = entry[field]
    return out


def compression_effect(mp4: dict, lossless: dict, lam: float) -> dict | None:
    """Descriptive: how much the CRF 10 encoding moves each quality guardrail's paired degradation (C3 minus C4 at the
    chosen lambda). The guardrails use the lossless frames; this compares them with the same degradation on the MP4s."""
    key = f"C4@{lam:g}"
    prompts = sorted(p for p in mp4 if p in lossless and {"C3", key} <= set(lossless[p]))
    if not prompts:
        return None
    out = {}
    for d in QUALITY_DIMENSIONS:
        on_mp4 = np.array([mp4[p]["C3"][d] - mp4[p][key][d] for p in prompts])
        on_lossless = np.array([lossless[p]["C3"][d] - lossless[p][key][d] for p in prompts])
        diff = on_mp4 - on_lossless
        out[d] = {"mean_on_mp4": float(on_mp4.mean()), "mean_on_lossless": float(on_lossless.mean()), "mean_difference": float(diff.mean()),
                  "sd_difference": float(diff.std(ddof=1)) if len(diff) > 1 else None, "limit": QUALITY_LIMIT}
    return {"prompts": prompts, "dimensions": out,
            "note": "the guardrails are scored from lossless frames; this is how far the playback MP4 would have moved them"}


def apply(run: Run, rows: list[dict]) -> dict:
    """Every pilot rule on a finished pilot run; writes pilot/rules.json. A stop is recorded, never skipped."""
    from density.run.masks import review

    setting = read_json(run.path("pilot", "setting.json"))
    frozen = read_json(run.path("frozen.json"))
    records = holdout_records(run)
    states = review(run, rows, setting["setting"], block="holdout")["states"]
    strata = {int(r["id"]): r["stratum"] for r in rows if r["block"] == "holdout"}
    lam = lambda_rule(records)
    quality = quality_scores(run)
    out = {
        "utc": utc_now(), "setting": setting, "frozen_sha256": run.manifest()["frozen_sha256"],
        "lambda": lam, "no_op": noop_rule(records), "c2": c2_rule(records), "pass_rates": pass_rates(states, strata),
        "reachability": reachability(degradations(records, quality, lam["lambda"])) if lam["lambda"] is not None else None,
        "compression": compression_effect(quality_scores(run, "quality_mp4"), quality, lam["lambda"]) if lam["lambda"] is not None else None,
        "excluded": frozen["excluded"],
    }
    out["stops"] = [s for s in (lam["stop"], out["no_op"]["stop"], out["c2"]["stop"]) if s]
    if out["reachability"] and out["reachability"]["unreachable"]:
        out["stops"].append(f"guardrails unreachable at 72 prompts: {out['reachability']['unreachable']}")
    write_json_atomic(run.path("pilot", "rules.json"), out)
    return out
