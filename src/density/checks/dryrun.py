"""The Mac dry runs of Milestone 5: a full mock pilot and a small mock main run, in scratch/dryrun/ (never read by a
real run), with tiny random models standing in for Wan and stock clips standing in for its frames.

Prompts here are placeholders ("Dry-run placeholder 3 ..."), never candidate rewrites: Chase writes the real prompts.
Reviews here are automatic and labelled "dry run (automatic, not Chase)": every sheet the pipeline passes is accepted,
except the rejections a test injects to exercise the replacement rule. A few prompts are marked as noise prompts:
their reference is the tiny model's own output, which has no face or hands, so they fail the pipeline's eligibility
and exercise exclusion (holdout) or replacement (primary).

Every number these runs produce describes the stand-ins, not Wan: the conditions are the stock clip plus a seeded
disturbance (density.run.generate.DryRunBackend), and every reference of a stratum is the same clip.
"""

import csv
import json
import shutil
from pathlib import Path

from density import prompts as prompt_set
from density.run import manifest as M
from density.run.__main__ import init
from density.run.generate import DryRunBackend
from density.run.masks import audit_path, review, write_audit
from density.run.session import session
from density.run.store import DRY_RUN_ROOT, Run, read_json, write_json_atomic

SOURCES = {"face": "vbench_human", "manipulation": "vbench_human_action", "body": "vbench2_human_anatomy", "control": "vbench_scenery"}
PHRASES = {"face": ("a person", "smiles", "close-up shot"), "manipulation": ("a person", "lifts a cup", "medium shot"),
           "body": ("a person", "jumps", "wide shot"), "control": ("a quiet lake", "ripples", "wide shot")}
REVIEWER = "dry run (automatic, not Chase)"


def placeholder(pid: int, stratum: str, block: str, reserve_order: int | None = None) -> dict:
    subject, action, framing = PHRASES[stratum]
    return {"id": str(pid), "stratum": stratum, "block": block, "reserve_order": "" if reserve_order is None else str(reserve_order),
            "text": f"Dry-run placeholder {pid}: {subject} {action} in a {framing}.", "subject": subject, "action": action, "framing": framing,
            "source": SOURCES[stratum], "source_id": f"{SOURCES[stratum]}:{pid}", "rewrite_note": "dry-run placeholder, not a prompt"}


def write_prompts(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=prompt_set.COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return path


def pilot_prompts() -> tuple[list[dict], tuple[int, ...]]:
    """6 face (1-6) and 6 manipulation (7-12) holdout prompts; 6 and 12 are noise prompts."""
    rows = [placeholder(p, "face", "holdout") for p in range(1, 7)] + [placeholder(p, "manipulation", "holdout") for p in range(7, 13)]
    return rows, (6, 12)


def main_prompts() -> tuple[list[dict], tuple[int, ...]]:
    """Primary 21-22 (face) and 23-24 (manipulation, 24 a noise prompt); reserves 25 (face) and 26 (manipulation); body 27;
    control 28."""
    rows = [placeholder(21, "face", "primary"), placeholder(22, "face", "primary"), placeholder(23, "manipulation", "primary"),
            placeholder(24, "manipulation", "primary"), placeholder(25, "face", "reserve", 1), placeholder(26, "manipulation", "reserve", 1),
            placeholder(27, "body", "secondary"), placeholder(28, "control", "control")]
    return rows, (24,)


def auto_review(run: Run, reject: dict[str, tuple[str, str]] | None = None) -> None:
    """Fill the audit file's reviewer columns: accept what the pipeline passes, reject the injected groups."""
    path = audit_path(run)
    lines = path.read_text().splitlines(keepends=True)
    table = list(csv.DictReader(line for line in lines if not line.startswith("#")))
    for r in table:
        if r["reviewer_decision"]:
            continue
        if reject and r["clip"] in reject:
            code, reason = reject[r["clip"]]
            r.update(reviewer_decision="reject", reviewer_reason=code, reviewer_written_reason=reason, reviewer=REVIEWER, review_date="2026-10-07")
        elif r["pipeline_decision"] in ("pass", "not gated"):
            r.update(reviewer_decision="accept", reviewer=REVIEWER, review_date="2026-10-07")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
        f.writelines(line for line in lines if line.startswith("#"))


def sessions(folder: Path, backend, machine: str, **kwargs) -> dict:
    record = session(folder, machine, "cpu", backend if machine == "h100" else None, **kwargs)
    if record["stop"]:
        raise RuntimeError(f"{machine} session stopped: {record['stop']}\n{record.get('traceback', '')[-3000:]}")
    return {"session": record["session"], "jobs": len(record["jobs"]), "seconds": sum(j["seconds"] for j in record["jobs"])}


def new_run(name: str, kind: str, rows: list[dict], noise: tuple[int, ...], registration: Path | None = None) -> Run:
    folder = DRY_RUN_ROOT / name
    shutil.rmtree(folder, ignore_errors=True)
    csv_path = write_prompts(DRY_RUN_ROOT / f"{name}_prompts.csv", rows)
    init(folder, kind, True, csv_path, registration)
    run = Run(folder)
    run.write_manifest({**run.manifest(), "dry_run_noise_prompts": list(noise)}, "dry run: noise prompts")
    return run


def mock_pilot(name: str = "pilot", prompts: tuple[list[dict], tuple[int, ...]] | None = None) -> dict:
    from density import pilot as pilot_rules

    rows, noise = prompts or pilot_prompts()
    run = new_run(name, "pilot", rows, noise)
    backend = DryRunBackend("cpu", noise)
    log = {"h100 references": sessions(run.folder, backend, "h100"), "slice masks": sessions(run.folder, backend, "slice")}
    setting = pilot_rules.choose_setting(run, rows)
    write_audit(run, setting["setting"])
    auto_review(run)
    from density.run.freeze import freeze

    log["freeze"] = freeze(run, rows, setting["setting"])
    log["h100 conditions"] = sessions(run.folder, backend, "h100")
    log["slice scores"] = sessions(run.folder, backend, "slice")
    rules = pilot_rules.apply(run, rows)
    return {"run": str(run.folder), "status": run.status(), "sessions": log, "setting": setting, "rules": rules}


def mock_registration(path: Path, setting: str = "sh8g6", lam: float = 3.0) -> Path:
    """A stand-in registration for the mock main run when no mock pilot has produced one."""
    write_json_atomic(path, {"draft": True, "dry_run": True, "values": {"setting": setting, "lambda": lam}, "note": "dry-run stand-in"})
    return path


def mock_main(name: str = "main", registration: Path | None = None, max_score_jobs: int | None = 2) -> dict:
    from density.analysis import collect
    from density.run.freeze import freeze

    rows, noise = main_prompts()
    registration = registration or mock_registration(DRY_RUN_ROOT / f"{name}_registration.json")
    run = new_run(name, "main", rows, noise, registration)
    backend = DryRunBackend("cpu", noise)
    setting = run.manifest()["setting"]
    log = {"h100 references": sessions(run.folder, backend, "h100"), "slice masks": sessions(run.folder, backend, "slice")}
    write_audit(run, setting)
    reject = {f"022-s{prompt_set.seeds(22)[1]}-{setting}": ("audit_mask_invalid", "dry run: an injected rejection, to exercise the replacement rule")}
    auto_review(run, reject)
    log["review 1"] = review(run, rows, setting)
    log["h100 replacement references"] = sessions(run.folder, backend, "h100")
    log["slice replacement masks"] = sessions(run.folder, backend, "slice")
    write_audit(run, setting)
    auto_review(run)
    log["review 2"] = review(run, rows, setting)
    manifest = run.manifest()
    log["freeze"] = freeze(run, rows, setting, manifest["lambda"], manifest["registration"])
    log["h100 conditions"] = sessions(run.folder, backend, "h100")
    from density.run.score import group_checks

    log["group checks"] = {j.group: group_checks(run, j) for j in run.jobs() if j.kind == "score"}  # sections 8 and 11, every group
    log["slice scores"] = sessions(run.folder, backend, "slice", max_jobs=max_score_jobs)
    primary_rows, secondary_rows, strata = collect(run.folder)
    blocks = sorted({j.grid_block for j in run.jobs() if j.kind == "condition"})
    return {"run": str(run.folder), "status": run.status(), "sessions": log, "grid_blocks": blocks,
            "collected": {"primary_rows": len(primary_rows), "secondary_blocks": {k: len(v) for k, v in secondary_rows.items()}, "strata": strata},
            "visual_check_selection": read_json(run.path("visual_check", "selection.json"))["rule"], "frozen": read_json(run.path("frozen.json"))["subsets"]}


if __name__ == "__main__":
    import sys

    which = sys.argv[1]
    out = {"pilot": mock_pilot, "main": mock_main}[which]()
    print(json.dumps(out, indent=1, default=str)[-6000:])
