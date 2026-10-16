"""An exploratory dose-response check, registered by Chase on 2026-10-08 before it ran. It never enters the
confirmatory analysis, and the main run's registration is unchanged.

Question: is step allocation a strong enough lever on the protected region's fidelity? The pilot found C4 (lambda 10)
improving masked LPIPS over C3 by a median of 0.6 percent. C7 runs the opposite allocation: if denying the region full
steps barely hurts it, moving steps into the region cannot help it much either.

C7 (inverse): the same HSA-50 group sizes and schedule as C3 and C4, but the full-step group is filled only with
tokens outside the region (b = 0, where b = 1[m >= 0.1] is the registered binary ROI), ranked by g descending, then by
ascending token ID; region tokens rank after every outside token, so no region token gets full steps. A run in which
the full-step group would need a region token stops instead of running (the definition could not hold). Tokens with
0 < m < 0.1 count as outside (b = 0), as the registered threshold says; how many of them reach the full-step group is
recorded, since they touch the measurement mask.

Run: the 12 holdout prompts of the pilot run, seed 1, at the chosen setting (shift 8 / guidance 6), HSA-50, from the
frozen references and masks; one condition job each, scored in-job exactly as C3 and C4 are (masked, outside-ROI and
full-frame LPIPS, masked PSNR and SSIM, crop LPIPS). Slice quality scores are not part of this check.

Reported per prompt: masked LPIPS for C3, C4 (lambda 10) and C7; outside-ROI LPIPS for each; and each condition's region
compute share S_ROI (equation 19), computed inside the C7 job from the same g for all three (every HSA run computes the
same g in its leading iterations). Before the shares are used, the report checks that C7's g and C3 labels hash to C3's
recorded ones, and that the C4 labels and S_ROI recomputed from that g equal the C4 (lambda 10) record's.

Decision rule (Chase's words, fixed before the run): "If C7's median masked LPIPS is within 3% of C3's, I'll treat
step allocation as too weak a lever, skip the main run, and pivot. If C7 is worse than C3 by more than 10%, the lever is
real, and I'll reconsider the main run. In between, I'll decide with outside input." As a formula, over the 12 prompts:
r = median(C7 masked LPIPS) / median(C3 masked LPIPS).
- 0.97 <= r <= 1.03: within 3 percent; step allocation is too weak a lever: skip the main run and pivot.
- r > 1.10: the lever is real: reconsider the main run.
- otherwise (1.03 < r <= 1.10, and r < 0.97, where C7 would beat C3, a case the rule does not name): decide with
  outside input.
The median of the per-prompt ratios is reported too, as description only.

    python -m density.exploratory add RUN       add the 12 C7 jobs to a frozen pilot run (on the pod)
    python -m density.exploratory report RUN    the per-prompt table, the checks and the decision (on the Mac)
"""

import argparse
import json
import sys
from pathlib import Path
from statistics import median

import numpy as np
import torch

from density.allocation import _float64, _lexsort, assign_groups, rank_importance_need, roi_binary
from density.conditions import ROI, Condition, sha256
from density.diagnostics import allocation_diagnostics
from density.hsa import HIGH, Schedule

OUTSIDE_THEN_NEED = "outside_then_need"
C7 = Condition("C7", 40, True, OUTSIDE_THEN_NEED, ROI, "Exploratory inverse (Chase, 2026-10-08): outside-region tokens first, then need")
EXPLORATORY_CONDITIONS = {"C7": C7}
GRID_BLOCK = "exploratory C7"
PRESET = "HSA-50"
COMPARE_LAMBDA = 10.0  # the pilot's chosen lambda (pilot/rules.json)
WITHIN, REAL = 0.03, 0.10  # Chase's thresholds


def rank_outside_then_need(g: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """C7: b = 0 (outside the region) first, then g descending, then ascending token ID."""
    b = roi_binary(m).cpu().numpy()
    return _lexsort(b.astype(np.int64), -_float64(g), np.arange(g.numel()))


def check_c7(labels: torch.Tensor, m: torch.Tensor) -> dict:
    """C7's definition: no region token in the full-step group. Raises if it cannot hold; returns the descriptive counts."""
    full = (labels == HIGH).cpu()
    region = roi_binary(m).cpu().bool()
    inside = int((full & region).sum())
    if inside:
        raise RuntimeError(f"C7: {inside} region tokens would get full steps: the region leaves too few outside tokens for the group")
    return {"full_step_region_tokens": 0, "full_step_tokens_with_m_above_0": int((full & (m.cpu() > 0)).sum())}


def same_g_diagnostics(g: torch.Tensor, m: torch.Tensor, group_iterations: torch.Tensor, high_tokens: int, labels_c7: torch.Tensor,
                       labels_c3: torch.Tensor, lam: float = COMPARE_LAMBDA) -> dict:
    """S_ROI and the other region diagnostics of C3, C4 (lambda) and C7, from one g, with each one's own active matrix."""
    labels_c4 = assign_groups(rank_importance_need(g, m, lam)[0], high_tokens)
    roi = roi_binary(m)
    out = {}
    for name, labels in (("C3", labels_c3), (f"C4@{lam:g}", labels_c4), ("C7", labels_c7)):
        active = Schedule(labels.cpu(), group_iterations).active
        out[name] = {**allocation_diagnostics(labels, roi, active, labels_c3), "labels_sha256": sha256(labels)}
    return {"g_sha256": sha256(g), "conditions": out, "c7": check_c7(labels_c7, m)}


# The run


def jobs(run) -> list:
    """C7 for every holdout reference at the chosen setting, seed 1, from the frozen references and masks."""
    from density.run import manifest as M
    from density.run import plan
    from density.run.store import read_json

    setting = read_json(run.path("pilot", "setting.json"))["setting"]
    frozen = read_json(run.path("frozen.json"))["references"]  # only the prompts the freeze kept (none was excluded in the pilot)
    refs = [j for j in run.jobs() if j.kind == "reference" and j.block == "holdout" and j.setting == setting and j.seed == 1000 + j.prompt_id
            and j.id in frozen]
    return [M.condition_job(ref, plan.GridItem("C7", PRESET), GRID_BLOCK, f"mask-{ref.group}") for ref in sorted(refs, key=lambda j: j.prompt_id)]


def add(run) -> dict:
    new = [j for j in jobs(run) if j.id not in {k.id for k in run.jobs()}]
    if new:
        run.add_jobs(new, "exploratory C7 (registered by Chase on 2026-10-08): never in the confirmatory analysis")
    return {"added": [j.id for j in new]}


def classify(r: float) -> str:
    if 1 - WITHIN <= r <= 1 + WITHIN:
        return "within 3 percent: step allocation is too weak a lever; skip the main run and pivot"
    if r > 1 + REAL:
        return "C7 worse than C3 by more than 10 percent: the lever is real; reconsider the main run"
    return "in between: decide with outside input"


def report(run) -> dict:
    """Per prompt: masked and outside-ROI LPIPS of C3, C4 (lambda 10) and C7, and their region compute shares, with the
    consistency checks; then the registered decision."""
    from density.run.store import read_json

    c4 = f"C4@{COMPARE_LAMBDA:g}"
    by: dict[int, dict] = {}
    for job in run.jobs():
        if job.kind != "condition" or job.condition not in ("C3", "C4", "C7") or (job.condition == "C4" and job.lam != COMPARE_LAMBDA):
            continue
        rec = run.record(job)
        if rec is None:
            raise RuntimeError(f"{job.id} is not done")
        key = c4 if job.condition == "C4" else job.condition
        alloc = read_json(run.job_dir(job) / "allocation.json")
        by.setdefault(job.prompt_id, {})[key] = {"rec": rec, "alloc": alloc}
    rows, problems = [], []
    for pid, r in sorted(by.items()):
        if set(r) != {"C3", c4, "C7"}:
            problems.append(f"prompt {pid}: missing {sorted({'C3', c4, 'C7'} - set(r))}")
            continue
        same = r["C7"]["rec"]["allocation_diagnostics"]["same_g"]
        checks = {
            "g equals C3's": same["g_sha256"] == r["C3"]["alloc"]["allocation_sha256"]["g"],
            "C3 labels equal C3's": same["conditions"]["C3"]["labels_sha256"] == r["C3"]["alloc"]["allocation_sha256"]["labels"],
            "C4 labels equal C4's": same["conditions"][c4]["labels_sha256"] == r[c4]["alloc"]["allocation_sha256"]["labels"],
            "C4 S_ROI equals C4's": same["conditions"][c4]["s_roi"] == r[c4]["rec"]["allocation_diagnostics"]["roi"]["s_roi"],
        }
        if not all(checks.values()):
            problems.append(f"prompt {pid}: {[k for k, v in checks.items() if not v]}")
        f = {k: r[k]["rec"]["fidelity"] for k in ("C3", c4, "C7")}
        rows.append({"prompt": pid, "stratum": r["C3"]["rec"]["job"]["stratum"],
                     "masked_lpips": {k: f[k]["masked_lpips"] for k in f}, "outside_lpips": {k: f[k]["outside_lpips"] for k in f},
                     "full_frame_lpips": {k: f[k]["full_frame_lpips"] for k in f},
                     "s_roi": {k: same["conditions"][k]["s_roi"] for k in ("C3", c4, "C7")},
                     "c7_full_step_tokens_with_m_above_0": same["c7"]["full_step_tokens_with_m_above_0"], "checks": checks})
    out = {"rows": rows, "problems": problems}
    if rows and not problems:
        med = {k: median(x["masked_lpips"][k] for x in rows) for k in ("C3", c4, "C7")}
        ratio = med["C7"] / med["C3"]
        out.update(medians=med, ratio=ratio, decision=classify(ratio),
                   per_prompt_ratio_median=median(x["masked_lpips"]["C7"] / x["masked_lpips"]["C3"] for x in rows),
                   note="the per-prompt ratio median is description only; the registered rule uses the ratio of medians")
    return out


def main(argv=None) -> int:
    from density.run.store import Run, write_json_atomic

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["add", "report"])
    parser.add_argument("run", type=Path)
    args = parser.parse_args(argv)
    run = Run(args.run)
    if args.command == "add":
        print(json.dumps(add(run), indent=1))
        return 0
    out = report(run)
    write_json_atomic(run.path("exploratory", "c7.json"), out)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    return 0 if not out["problems"] else 1


if __name__ == "__main__":
    sys.exit(main())
