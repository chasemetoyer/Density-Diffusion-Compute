"""Synthetic validations of the frozen analysis (Milestone 5): score tables with a known truth, fed to density.analysis.

- a 15 percent improvement with harmless guardrails gives GO;
- no effect never gives GO, over many synthetic replicates;
- an injected harm on one guardrail gives NO-GO, even with a 15 percent improvement;
- a missing required score blocks the decision.

The tables have the main run's shape: 36 face and 36 manipulation prompts, two seeds, C3 and C4, with masked LPIPS and
the seven guardrails. This module is not frozen; the analysis it exercises is.
"""

import numpy as np

from density import analysis
from density.metrics.registered import QUALITY_DIMENSIONS


def synthetic_rows(effect: float = 0.15, harm: dict | None = None, seed: int = 0, missing: tuple | None = None) -> tuple[list[dict], dict[int, str]]:
    """Rows for analysis.analyze. ``harm`` adds a mean degradation to named guardrails (LPIPS for C4 up, quality down);
    ``missing`` = (prompt, seed, condition, metric) drops one score."""
    rng = np.random.default_rng([seed, 7])
    harm = harm or {}
    rows, strata = [], {}
    for p in range(1, 73):
        strata[p] = "face" if p <= 36 else "manipulation"
        base = rng.lognormal(np.log(0.10), 0.4)
        quality = {d: rng.normal(60 if d == "imaging_quality" else 95, 3) for d in QUALITY_DIMENSIONS}
        lp_full, lp_out = rng.lognormal(np.log(0.05), 0.3), rng.lognormal(np.log(0.04), 0.3)
        for s in (1, 2):
            l3 = base * rng.lognormal(0, 0.10)
            l4 = l3 * (1 - effect) * rng.lognormal(0, 0.05)
            c3 = {"masked_lpips": l3, "full_frame_lpips": lp_full * rng.lognormal(0, 0.05), "outside_lpips": lp_out * rng.lognormal(0, 0.05),
                  **{d: v + rng.normal(0, 0.2) for d, v in quality.items()}}
            c4 = {"masked_lpips": l4,
                  "full_frame_lpips": c3["full_frame_lpips"] + harm.get("full_frame_lpips", 0.0) + rng.normal(0, 0.002),
                  "outside_lpips": c3["outside_lpips"] + harm.get("outside_lpips", 0.0) + rng.normal(0, 0.002),
                  **{d: c3[d] - harm.get(d, 0.0) + rng.normal(0, 0.1) for d in QUALITY_DIMENSIONS}}
            for condition, values in (("C3", c3), ("C4", c4)):
                row = {"prompt": p, "stratum": strata[p], "seed": s, "condition": condition, **values}
                if missing and (p, s, condition) == missing[:3]:
                    row[missing[3]] = None
                rows.append(row)
    return rows, strata


def validations(replicates: int = 20, resamples: int = analysis.RESAMPLES) -> dict:
    out = {}
    rows, strata = synthetic_rows(0.15)
    r = analysis.analyze(rows, strata, resamples=resamples)
    out["improvement_15_percent"] = {"outcome": r["outcome"], "median_delta": r["primary"]["median_delta"], "ci95": r["primary"]["ci95"],
                                     "guardrails": r["guardrail_statuses"], "expected": "GO", "passed": r["outcome"] == "GO"}
    outcomes = [analysis.analyze(*synthetic_rows(0.0, seed=k), resamples=resamples)["outcome"] for k in range(replicates)]
    out["no_effect"] = {"outcomes": {o: outcomes.count(o) for o in sorted(set(outcomes))}, "replicates": replicates, "expected": "never GO", "passed": "GO" not in outcomes}
    harmed = {}
    for g, size in (("outside_lpips", 0.02), ("imaging_quality", 1.0)):
        r = analysis.analyze(*synthetic_rows(0.15, harm={g: size}), resamples=resamples)
        harmed[g] = {"injected": size, "outcome": r["outcome"], "status": r["guardrail_statuses"][g], "passed": r["outcome"] == "NO-GO" and r["guardrail_statuses"][g] == "confirmed harm"}
    out["injected_harm"] = {"cases": harmed, "expected": "NO-GO with confirmed harm", "passed": all(c["passed"] for c in harmed.values())}
    r = analysis.analyze(*synthetic_rows(0.15, missing=(5, 2, "C4", "motion_smoothness")), resamples=resamples)
    out["missing_metric"] = {"outcome": r["outcome"], "reason": r["reason"][:200], "expected": "BLOCKED", "passed": r["outcome"] == "BLOCKED"}
    out["passed"] = all(v["passed"] for v in out.values() if isinstance(v, dict))
    return out
