"""The one registered primary analysis (protocol section 9), frozen by code hash.

Primary result
- Equation 24: each condition's loss per primary prompt is the mean of its two seeds' masked LPIPS.
- Equation 25: Delta_p = (L_C3 - L_C4) / max(L_C3, 1e-6); the result is the median of Delta_p over the 72 prompts.
  Prompts at the floor stay in and are counted.
- A paired bootstrap: 100,000 resamples with numpy's default_rng(20261005); each draws 36 face and then 36 manipulation
  prompts with replacement (two integer arrays of shape (100000, 36), in that order), keeping each prompt's seeds and
  conditions together. The 95 percent interval is the 2.5th and 97.5th percentiles (numpy's default, linear).
- Supporting: the median absolute improvement and the ratio of mean losses, with intervals; the Wilcoxon signed-rank
  test of L_C3 - L_C4 (two-sided, zero_method "wilcox"); the win rate (ties count half).

Guardrails (equations 27-28): seed-averaged paired degradation, C4 minus C3 for full-frame and outside-ROI LPIPS and
C3 minus C4 for the five quality dimensions, every one measured before lossy compression (section 7; Chase, 2026-10-07):
LPIPS inside each generation job and the quality dimensions on the lossless frames, which ``collect`` requires of every
score record. The statistic is its mean over the 72 prompts, with Bonferroni bootstrap
endpoints 0.05 / 14 and 1 - 0.05 / 14 from the same resamples. Pass: upper bound below the limit. Confirmed harm: lower
bound above it. Otherwise uncertain.

Decision: a missing required score blocks it (a technical failure, not a NO-GO). Confirmed harm is checked first and
gives NO-GO, as does an upper bound below 5 percent. GO needs a median of at least 10 percent, a lower bound above zero,
and all seven guardrails passing. Everything else is UNCERTAIN.

Secondary families (registered by Chase, 2026-10-07, decision 10), each with Holm's correction over its Wilcoxon p-values: strata (face, manipulation, full body),
budget and switch (75 percent, switch-20, switch-30; C4 against C3), and allocation and ROI ablations (C5, C6 and A1,
each against C4, where a positive effect favours the ablation). Effects are medians of relative improvement with
bootstrap intervals resampled within the block. Quality differences and the controls' C4-equals-C3 check are
descriptive.

``python -m density.analysis freeze`` records the SHA-256 of this file, of the registered values it reads, and of the
modules ``collect`` runs to read a run's records, with numpy's and scipy's versions; ``run`` refuses to analyse anything
unless all of them still match.
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

from density.metrics.registered import GUARDRAILS, LPIPS_LIMIT, QUALITY_DIMENSIONS, QUALITY_INPUT, QUALITY_LIMIT

ROOT = Path(__file__).resolve().parents[2]
FREEZE_FILE = ROOT / "configs" / "analysis_freeze.json"
FROZEN_FILES = ("src/density/analysis.py", "src/density/metrics/registered.py",  # the analysis and its limits
                "src/density/run/store.py", "src/density/run/manifest.py", "src/density/prompts.py")  # what collect() runs
RESAMPLES = 100_000
BOOTSTRAP_SEED = 20261005
PROMPTS_PER_STRATUM = 36
FLOOR = 1e-6
ALPHA = 0.05
J = 7
GO_MEDIAN = 0.10
NO_GO_UPPER = 0.05
LIMITS = {"full_frame_lpips": LPIPS_LIMIT, "outside_lpips": LPIPS_LIMIT, **{d: QUALITY_LIMIT for d in QUALITY_DIMENSIONS}}
LPIPS_GUARDRAILS = ("full_frame_lpips", "outside_lpips")
PRIMARY_METRIC = "masked_lpips"


class Blocked(RuntimeError):
    """A required score is missing: the decision is blocked until the jobs are rerun or the scoring is repaired."""


class CodeChanged(RuntimeError):
    pass


# Tables


def seed_means(rows: list[dict], conditions: tuple[str, ...], metrics: tuple[str, ...], seeds: int = 2) -> dict:
    """{prompt: {condition: {metric: mean over seeds}}}; raises Blocked listing every missing or non-finite score."""
    values: dict = {}
    for r in rows:
        if r["condition"] in conditions:
            for m in metrics:
                values.setdefault(r["prompt"], {}).setdefault(r["condition"], {}).setdefault(m, {})[r["seed"]] = r.get(m)
    missing = []
    out = {}
    for p, by_condition in values.items():
        for c in conditions:
            for m in metrics:
                got = by_condition.get(c, {}).get(m, {})
                if len(got) != seeds or any(v is None or not math.isfinite(v) for v in got.values()):
                    missing.append(f"prompt {p} {c} {m}: {got}")
                else:
                    out.setdefault(p, {}).setdefault(c, {})[m] = float(np.mean([got[s] for s in sorted(got)]))
    if missing:
        raise Blocked(f"{len(missing)} required scores missing: {missing[:10]}")
    return out


def relative(l_base: np.ndarray, l_new: np.ndarray) -> np.ndarray:
    """Equation 25: (L_base - L_new) / max(L_base, 1e-6); positive favours the new condition."""
    return (l_base - l_new) / np.maximum(l_base, FLOOR)


# Bootstrap


def resample_indices(n_face: int = PROMPTS_PER_STRATUM, n_manipulation: int = PROMPTS_PER_STRATUM, resamples: int = RESAMPLES) -> np.ndarray:
    """(resamples, 72) indices into the face prompts followed by the manipulation prompts."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    face = rng.integers(0, n_face, size=(resamples, n_face))
    manipulation = rng.integers(0, n_manipulation, size=(resamples, n_manipulation)) + n_face
    return np.concatenate([face, manipulation], axis=1)


def interval(values: np.ndarray, low: float, high: float) -> tuple[float, float]:
    return float(np.quantile(values, low)), float(np.quantile(values, high))


def single_block_indices(n: int, resamples: int = RESAMPLES, salt: int = 0) -> np.ndarray:
    return np.random.default_rng([BOOTSTRAP_SEED, salt]).integers(0, n, size=(resamples, n))


def win_rate(delta: np.ndarray) -> float:
    return float((np.sum(delta > 0) + 0.5 * np.sum(delta == 0)) / len(delta))


def wilcoxon_p(differences: np.ndarray) -> float | None:
    from scipy.stats import wilcoxon

    if np.all(differences == 0):
        return None
    return float(wilcoxon(differences, zero_method="wilcox", alternative="two-sided").pvalue)


def holm(pvalues: dict[str, float | None]) -> dict[str, float | None]:
    """Holm's step-down adjusted p-values within one family (None stays None and is not counted)."""
    named = sorted(((p, k) for k, p in pvalues.items() if p is not None))
    m, out, running = len(named), {k: None for k in pvalues}, 0.0
    for i, (p, k) in enumerate(named):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


# The analysis


def primary(means: dict, strata: dict[int, str], idx: np.ndarray) -> dict:
    face = sorted(p for p in means if strata[p] == "face")
    manipulation = sorted(p for p in means if strata[p] == "manipulation")
    if len(face) != PROMPTS_PER_STRATUM or len(manipulation) != PROMPTS_PER_STRATUM:
        raise Blocked(f"the primary analysis needs 36 face and 36 manipulation prompts, got {len(face)} and {len(manipulation)}")
    order = face + manipulation
    l3 = np.array([means[p]["C3"][PRIMARY_METRIC] for p in order])
    l4 = np.array([means[p]["C4"][PRIMARY_METRIC] for p in order])
    delta, absolute = relative(l3, l4), l3 - l4
    boot_median = np.median(delta[idx], axis=1)
    boot_abs = np.median(absolute[idx], axis=1)
    boot_ratio = l4[idx].mean(axis=1) / l3[idx].mean(axis=1)
    return {
        "prompts": order, "delta": delta.tolist(), "median_delta": float(np.median(delta)), "ci95": interval(boot_median, 0.025, 0.975),
        "floor_prompts": int(np.sum(l3 < FLOOR)),
        "median_absolute_improvement": float(np.median(absolute)), "median_absolute_ci95": interval(boot_abs, 0.025, 0.975),
        "ratio_of_mean_losses": float(l4.mean() / l3.mean()), "ratio_ci95": interval(boot_ratio, 0.025, 0.975),
        "wilcoxon_p": wilcoxon_p(absolute), "win_rate": win_rate(delta),
        "mean_loss": {"C3": float(l3.mean()), "C4": float(l4.mean())},
    }


def guardrails(means: dict, order: list[int], idx: np.ndarray) -> dict:
    low, high = ALPHA / (2 * J), 1 - ALPHA / (2 * J)
    out = {}
    for g in GUARDRAILS:
        c3 = np.array([means[p]["C3"][g] for p in order])
        c4 = np.array([means[p]["C4"][g] for p in order])
        d = c4 - c3 if g in LPIPS_GUARDRAILS else c3 - c4  # equation 27: positive is damage
        lo, hi = interval(d[idx].mean(axis=1), low, high)
        status = "pass" if hi < LIMITS[g] else "confirmed harm" if lo > LIMITS[g] else "uncertain"
        out[g] = {"mean_degradation": float(d.mean()), "interval": (lo, hi), "limit": LIMITS[g], "status": status}
    return {"endpoints": (low, high), "guardrails": out}


def decide(prim: dict, guard: dict) -> dict:
    statuses = {g: v["status"] for g, v in guard["guardrails"].items()}
    lo, hi = prim["ci95"]
    if any(s == "confirmed harm" for s in statuses.values()):
        outcome, reason = "NO-GO", "a guardrail shows confirmed harm: " + ", ".join(g for g, s in statuses.items() if s == "confirmed harm")
    elif hi < NO_GO_UPPER:
        outcome, reason = "NO-GO", f"the primary 95 percent upper bound {hi:.4f} is below 5 percent"
    elif prim["median_delta"] >= GO_MEDIAN and lo > 0 and all(s == "pass" for s in statuses.values()):
        outcome, reason = "GO", "median at least 10 percent, lower bound above zero, and all seven guardrails pass"
    else:
        outcome, reason = "UNCERTAIN", "neither GO nor NO-GO holds"
    return {"outcome": outcome, "reason": reason, "guardrail_statuses": statuses}


def block_effect(means: dict, prompts: list[int], base: str, new: str, metric: str = PRIMARY_METRIC, salt: int = 0) -> dict:
    lb = np.array([means[p][base][metric] for p in prompts])
    ln = np.array([means[p][new][metric] for p in prompts])
    delta = relative(lb, ln)
    idx = single_block_indices(len(prompts), salt=salt)
    return {"prompts": len(prompts), "median_delta": float(np.median(delta)), "ci95": interval(np.median(delta[idx], axis=1), 0.025, 0.975),
            "median_absolute": float(np.median(lb - ln)), "wilcoxon_p": wilcoxon_p(lb - ln), "win_rate": win_rate(delta)}


def secondary(tables: dict, strata: dict[int, str]) -> dict:
    """``tables``: block name -> seed-averaged means (seed 1 only for the secondary blocks)."""
    fam: dict[str, dict] = {"strata": {}, "budget_switch": {}, "ablations": {}}
    prim = tables["primary"]
    for k, stratum in enumerate(("face", "manipulation")):
        fam["strata"][stratum] = block_effect(prim, sorted(p for p in prim if strata[p] == stratum), "C3", "C4", salt=1 + k)
    if "full body" in tables:
        fam["strata"]["body"] = block_effect(tables["full body"], sorted(tables["full body"]), "C3", "C4", salt=3)
    for k, (name, base, new) in enumerate((("75 percent", "C3@HSA-75B", "C4@HSA-75B"), ("switch-20", "C3@switch-20", "C4@switch-20"), ("switch-30", "C3@switch-30", "C4@switch-30"))):
        table = tables.get("75 percent" if name == "75 percent" else "switch points")
        if table:
            fam["budget_switch"][name] = block_effect(table, sorted(table), base, new, salt=10 + k)
    for k, (name, table_name, ablation) in enumerate((("C5 vs C4", "C5 and C6", "C5"), ("C6 vs C4", "C5 and C6", "C6"), ("A1 vs C4", "A1", "A1"))):
        table = tables.get(table_name)
        if table:
            fam["ablations"][name] = block_effect(table, sorted(table), "C4", ablation, salt=20 + k)
    if tables.get("A1 regions"):  # section 3: A1 and C4 on hands and contact, and on the held object, separately (descriptive)
        t = tables["A1 regions"]
        fam["ablations"]["A1 vs C4"]["regions"] = {r: block_effect(t, sorted(t), "C4", "A1", f"{PRIMARY_METRIC}_{r}", salt=30 + k) for k, r in enumerate(("HC", "O"))}
    for family in fam.values():
        adjusted = holm({k: v["wilcoxon_p"] for k, v in family.items()})
        for k, v in family.items():
            v["holm_p"] = adjusted[k]
    return fam


def quality_differences(means: dict, order: list[int]) -> dict:
    return {d: {"mean_c3_minus_c4": float(np.mean([means[p]["C3"][d] - means[p]["C4"][d] for p in order]))} for d in QUALITY_DIMENSIONS}


def analyze(rows: list[dict], strata: dict[int, str], secondary_rows: dict[str, list[dict]] | None = None, resamples: int = RESAMPLES) -> dict:
    """The registered analysis of the primary rows (72 prompts x 2 seeds x C3 and C4) and the secondary blocks."""
    try:
        means = seed_means(rows, ("C3", "C4"), (PRIMARY_METRIC, *GUARDRAILS))
        idx = resample_indices(resamples=resamples)
        prim = primary(means, strata, idx)
        guard = guardrails(means, prim["prompts"], idx)
    except Blocked as error:
        return {"outcome": "BLOCKED", "reason": str(error)}
    result = {**decide(prim, guard), "primary": prim, "guardrails": guard, "quality_differences": quality_differences(means, prim["prompts"]),
              "settings": {"resamples": resamples, "seed": BOOTSTRAP_SEED, "floor": FLOOR, "J": J, "alpha": ALPHA}}
    if secondary_rows:
        tables = {"primary": means}
        for name, block_rows in secondary_rows.items():
            conditions = tuple(sorted({r["condition"] for r in block_rows}))
            try:
                tables[name] = seed_means(block_rows, conditions, (PRIMARY_METRIC,) if name != "controls" else ("full_frame_lpips",), seeds=1)
                if name == "A1":
                    tables["A1 regions"] = seed_means(block_rows, ("C4", "A1"), tuple(f"{PRIMARY_METRIC}_{r}" for r in ("HC", "O")), seeds=1)
            except Blocked as error:
                result.setdefault("secondary_blocked", {})[name] = str(error)
        result["secondary"] = secondary(tables, strata)
        if "controls" in tables:
            t = tables["controls"]
            result["controls_check"] = {"max_abs_c4_minus_c3_full_frame_lpips": max(abs(t[p]["C4"]["full_frame_lpips"] - t[p]["C3"]["full_frame_lpips"]) for p in t),
                                        "note": "an explicit empty mask makes C4 reproduce C3; any difference is an implementation failure"}
    return result


# The freeze


def code_state() -> dict:
    import scipy

    return {"files": {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in FROZEN_FILES}, "numpy": np.__version__, "scipy": scipy.__version__}


def record_freeze(path: Path = FREEZE_FILE) -> dict:
    from density.checks import euler

    state = {**code_state(), "git": euler.git_state(), "note": "the registered analysis: density.analysis refuses to run unless these hashes and versions match"}
    path.write_text(json.dumps(state, indent=2) + "\n")
    return state


def require_frozen(path: Path = FREEZE_FILE) -> dict:
    if not path.exists():
        raise CodeChanged(f"no analysis freeze at {path}")
    frozen, now = json.loads(path.read_text()), code_state()
    changed = [f for f in FROZEN_FILES if frozen["files"].get(f) != now["files"][f]]
    if changed or frozen["numpy"] != now["numpy"] or frozen["scipy"] != now["scipy"]:
        raise CodeChanged(f"the analysis code or its libraries changed since the freeze: files {changed}, numpy {frozen['numpy']} -> {now['numpy']}, scipy {frozen['scipy']} -> {now['scipy']}")
    return frozen


# Collecting a run's scores


def collect(run_folder: Path) -> tuple[list[dict], dict[str, list[dict]], dict[int, str]]:
    """Flatten a main run's records into rows: prompt, stratum, seed index, condition key, and every required score."""
    from density import prompts as prompt_set
    from density.run.store import Run

    run = Run(run_folder)
    primary_rows, secondary_rows, strata = [], {}, {}
    quality = {}
    for job in run.jobs():
        if job.kind == "score" and (rec := run.record(job)):
            for vid, entry in rec["videos"].items():
                if entry.get("quality_input") != QUALITY_INPUT:
                    raise Blocked(f"{job.id}: {vid}'s quality scores come from {entry.get('quality_input')!r}, not the registered {QUALITY_INPUT}")
                quality[vid] = entry["quality"]
    for job in run.jobs():
        if job.kind != "condition":
            continue
        rec = run.record(job)
        strata[job.prompt_id] = job.stratum
        key = job.condition if job.grid_block in ("primary seed 1", "primary seed 2", "full body", "controls", "C5 and C6", "A1") else f"{job.condition}@{job.preset}"
        seed_index = prompt_set.seeds(job.prompt_id).index(job.seed) + 1
        row = {"prompt": job.prompt_id, "stratum": job.stratum, "seed": seed_index, "condition": key, "job": job.id}
        fid = rec["fidelity"] if rec else {}
        for m in (PRIMARY_METRIC, "full_frame_lpips", "outside_lpips"):
            row[m] = fid.get(m)
        for region, value in fid.get("region_masked_lpips", {}).items():
            row[f"{PRIMARY_METRIC}_{region}"] = value
        for d in QUALITY_DIMENSIONS:
            row[d] = quality.get(job.id, {}).get(d)
        if job.grid_block in ("primary seed 1", "primary seed 2"):
            primary_rows.append(row)
        if job.grid_block == "primary seed 1" and job.condition in ("C3", "C4"):  # C4 is the baseline of the ablation blocks
            for block in ("C5 and C6", "A1"):
                secondary_rows.setdefault(block, []).append({**row, "seed": 1})
        elif job.grid_block not in ("primary seed 1", "primary seed 2"):
            secondary_rows.setdefault(job.grid_block, []).append(row)
    ablation_prompts = {b: {r["prompt"] for r in rows if r["condition"] not in ("C3", "C4")} for b, rows in secondary_rows.items() if b in ("C5 and C6", "A1")}
    for b, keep in ablation_prompts.items():
        secondary_rows[b] = [r for r in secondary_rows[b] if r["prompt"] in keep]
    return primary_rows, secondary_rows, strata


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze")
    p = sub.add_parser("run")
    p.add_argument("run", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        print(json.dumps(record_freeze(), indent=1))
        return 0
    frozen = require_frozen()
    try:
        rows, secondary_rows, strata = collect(args.run)
        result = analyze(rows, strata, secondary_rows)
    except Blocked as error:  # a record that cannot be read as registered is a technical blockage too
        result = {"outcome": "BLOCKED", "reason": str(error)}
    result["freeze"] = frozen
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"[analysis] {result['outcome']}: {result['reason']}")
    return 0 if result["outcome"] != "BLOCKED" else 1


if __name__ == "__main__":
    sys.exit(main())
