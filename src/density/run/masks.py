"""Slice jobs on Wan references, and the audit (protocol section 4): masks and contact sheets, the review file for
Chase, and the registered replacement rule.

A mask job reads the reference's lossless frames from the run folder (on the pod, the density volume), runs the mask
pipeline (density.masks.pipeline, every registered setting) and writes the masks, token coverage, metadata and the
contact sheet. ``write_audit`` gathers every sheet into audit/audit.csv; Chase fills its reviewer columns by the
registered criteria. ``review`` reads his decisions: a primary prompt passes only when both seeds pass the pipeline's
eligibility and both of his reviews accept them, for the sheets as they are now (by hash). A failed prompt is replaced
by the next reserve of its stratum, in order; both seeds of the replacement are then generated and audited. An empty
reserve list stops the run (ReserveExhausted) until the registration is amended. Registered by Chase on 2026-10-07
(decisions 3, 12 and 13): holdout prompts that fail are excluded from the pilot's conditions and reported, never
replaced; full-body sheets are produced for information and not gated, since section 4 gates only primary prompts.
"""

import shutil
import time
from pathlib import Path

import numpy as np

from density.masks.audit import read_audit_csv, write_audit_csv
from density.masks.eligibility import AuditRejection, ReplacementLog, ReserveList, SeedResult
from density.run.manifest import Job, reference_jobs
from density.run.store import Run, file_sha256, load_frames, read_json, utc_now, write_json_atomic

PRIMARY_DETECTOR_REGION = {"face": "F", "manipulation": "H"}  # SCRFD's and WiLoR's detections


def masks_code_sha256() -> str:
    """One hash over the mask pipeline's code (density/masks/*.py and the token map), for the dry run's cache key."""
    import hashlib

    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted((root / "masks").glob("*.py")) + [root / "token_map.py"]:
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


class DiskCache:
    """Dry runs only: the mask pipeline's results keyed by (frames hash, stratum) and the pipeline code's hash, pickled
    under scratch/dryrun/mask_cache/ so later sessions and runs reuse them. Never used by a real run."""

    def __init__(self, folder: Path):
        self.folder, self.code = Path(folder), masks_code_sha256()

    def path(self, key) -> Path:
        return self.folder / f"{key[0][:16]}-{key[1]}-{self.code[:12]}.pkl"

    def __contains__(self, key) -> bool:
        return self.path(key).exists()

    def __getitem__(self, key):
        import pickle

        with open(self.path(key), "rb") as f:
            data = pickle.load(f)
        return f"{data['job']} (cache {self.path(key).name}, {data['utc']})", data["result"]

    def __setitem__(self, key, value) -> None:
        import pickle

        from density.run.store import write_bytes_atomic

        write_bytes_atomic(self.path(key), pickle.dumps({"job": value[0], "result": value[1], "utc": utc_now(), "code": self.code}))


def detection_coverage(result) -> float | None:
    """The share of frames where the stratum's primary detector (SCRFD for faces, WiLoR for hands) found anything."""
    region = PRIMARY_DETECTOR_REGION.get(result.stratum)
    if region is None:
        return None
    detections = result.regions[region].detections
    return sum(1 for f in range(81) if detections.get(f)) / 81


def run_mask(run: Run, job: Job, models, sam_device: str | None = None, cache: dict | None = None) -> dict:
    """``cache`` (dry runs only): the pipeline's result for frames it has already processed, keyed by their hash, so the
    dry run's stand-in references, which repeat a few stock clips, run the pipeline once per clip. Storage, sheets and
    records are still written for every job."""
    import copy
    import dataclasses

    from density.checks import masks as mask_checks
    from density.masks.pipeline import metadata, process_reference, save

    ref_id = f"ref-{job.group}"
    ref = run.record(ref_id)
    frames = load_frames(run.job_dir(ref_id) / ref["frames"]["folder"], ref["frames"])
    start = time.perf_counter()
    key = (ref["frames"]["uint8_sha256"], job.stratum)
    cached_from = None
    if cache is not None and key in cache:
        cached_from, result = cache[key][0], copy.deepcopy(cache[key][1])
        result.eligibility = dataclasses.replace(result.eligibility, seed=job.seed)
    else:
        result = process_reference(frames, job.stratum, models, seed=job.seed, sam_device=sam_device)
        if cache is not None:
            cache[key] = (job.id, copy.deepcopy(result))
    final = run.mask_dir(job.group)
    tmp = final.with_name(final.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    stored = save(result, tmp, {"job": job.id, "reference": {"job": ref_id, "uint8_sha256": ref["frames"]["uint8_sha256"]},
                                "registered": mask_checks.jsonable(mask_checks.registered_values())})
    mask_checks.SHEETS_DIR = tmp
    sheets = mask_checks.write_sheets({"id": job.group, "stratum": job.stratum}, frames, result)
    shutil.rmtree(final, ignore_errors=True)
    tmp.rename(final)
    meta = metadata(result)
    sheet = Path(sheets[job.group]["path"]).name
    intervals = meta["contact"]["intervals"] if meta["contact"] else []
    return {
        "reference": {"job": ref_id, "uint8_sha256": ref["frames"]["uint8_sha256"]},
        "stored": stored, "sheet": sheet, "sheet_sha256": file_sha256(final / sheet),
        "eligibility": meta["eligibility"], "pipeline_decision": mask_checks.pipeline_decision(job.stratum, result.eligibility),
        "roi_usable_frames": meta["roi_usable_frames"], "inside_frames": len(meta["measurement"]["inside_frames"]),
        "outside_frames": len(meta["measurement"]["outside_frames"]),
        "contact_intervals": [(i["hand_track"], i["object_track"], i["start"], i["end"]) for i in intervals if not i["unusable_frames"]],
        "detection_coverage": detection_coverage(result), "seconds": meta["seconds"], "sam": meta["sam"],
        "model_build_seconds": dict(models.build_seconds), "total_seconds": time.perf_counter() - start,
        **({"dry_run_cached_from": cached_from} if cached_from else {}),
    }


def audit_path(run: Run) -> Path:
    return run.path("audit", "audit.csv")


def mask_jobs(run: Run, setting: str | None = None) -> list[Job]:
    return [j for j in run.jobs() if j.kind == "mask" and run.done(j) and (setting is None or j.setting == setting)]


def write_audit(run: Run, setting: str | None = None) -> Path:
    """audit/audit.csv: one row per finished mask job, with the sheet's path and hash; earlier reviews carry over only
    for sheets that are unchanged."""
    rows = []
    for job in mask_jobs(run, setting):
        rec = run.record(job)
        rows.append({
            "clip": job.group, "seed": job.seed, "stratum": job.stratum, "pipeline_decision": rec["pipeline_decision"],
            "pipeline_reasons": " ".join(rec["eligibility"]["reasons"]), "usable_roi_frames": rec["eligibility"]["usable_frames"],
            "contact_intervals": "; ".join(f"H{h}-O{o} {a}-{b}" for h, o, a, b in rec["contact_intervals"]), "outside_frames": rec["outside_frames"],
            "contact_sheet": str((run.mask_dir(job.group) / rec["sheet"]).relative_to(run.folder)), "contact_sheet_sha256": rec["sheet_sha256"],
        })
    return write_audit_csv(rows, audit_path(run))


def seed_result(run: Run, job: Job, decisions: dict) -> SeedResult | None:
    """A seed's eligibility: the pipeline's reasons plus the reviewer's rejection; None while the review is missing."""
    rec = run.record(job)
    if rec is None:
        return None
    reasons = list(rec["eligibility"]["reasons"])
    written = {}
    decision = decisions.get(job.group)
    if decision is not None and decision.contact_sheet_sha256 != rec["sheet_sha256"]:
        raise RuntimeError(f"{job.group}: the review is for sheet {decision.contact_sheet_sha256[:12]}, but the sheet is now {rec['sheet_sha256'][:12]}")
    if reasons:  # failed the pipeline's eligibility: no review is needed
        return SeedResult(job.seed, False, reasons, rec["eligibility"]["usable_frames"])
    if decision is None:
        return None
    if not decision.accepted:
        reasons.append(decision.rejection.code)
        written[decision.rejection.code] = decision.rejection.written_reason
    return SeedResult(job.seed, not reasons, reasons, rec["eligibility"]["usable_frames"], written)


def replacements_path(run: Run) -> Path:
    return run.path("audit", "replacements.json")


def load_replacements(run: Run) -> ReplacementLog:
    return ReplacementLog(read_json(replacements_path(run))["entries"] if replacements_path(run).exists() else [])


def current_prompts(rows: list[dict], log: ReplacementLog, block: str = "primary") -> dict[str, list[int]]:
    """Each stratum's prompts after the logged replacements, in order: failed prompts out, their replacements in."""
    by_stratum: dict[str, list[int]] = {}
    for r in rows:
        if r["block"] == block:
            by_stratum.setdefault(r["stratum"], []).append(int(r["id"]))
    if block == "primary":  # replacements happen only in the primary blocks
        for e in log.entries:
            ids = by_stratum[e["stratum"]]
            ids[ids.index(e["failed_prompt"])] = e["replacement"]
    return by_stratum


def reserves(rows: list[dict], stratum: str, log: ReplacementLog) -> list[int]:
    used = {e["replacement"] for e in log.entries}
    ordered = sorted((int(r["reserve_order"]), int(r["id"])) for r in rows if r["block"] == "reserve" and r["stratum"] == stratum)
    return [pid for _, pid in ordered if pid not in used]


def review(run: Run, rows: list[dict], setting: str, decisions_path: Path | None = None, block: str = "primary") -> dict:
    """Apply Chase's reviews. Primary prompts that fail are replaced in reserve order and their replacements' jobs added
    to the manifest; holdout prompts (``block="holdout"``) are judged but never replaced. Returns each prompt's state."""
    decisions = {d.clip: d for d in read_audit_csv(decisions_path or audit_path(run))}
    log = load_replacements(run)
    by_id = {int(r["id"]): r for r in rows}
    states, new_jobs = {}, []
    for stratum, ids in current_prompts(rows, log, block).items():
        for pid in list(ids):
            row = by_id[pid]
            seed_jobs = [j for j in run.jobs() if j.kind == "mask" and j.prompt_id == pid and j.setting == setting]
            if block == "primary" and len(seed_jobs) != 2:
                states[pid] = "pending: its references are not in the manifest yet"
                continue
            results = [seed_result(run, j, decisions) for j in sorted(seed_jobs, key=lambda j: j.seed)]
            failed = [r for r in results if r is not None and not r.passed]
            if not failed and any(r is None for r in results):
                states[pid] = "pending: " + ", ".join(f"seed {j.seed} {'not masked' if not run.done(j) else 'not reviewed'}" for j, r in zip(sorted(seed_jobs, key=lambda j: j.seed), results) if r is None)
                continue
            if not failed:
                states[pid] = "pass"
                continue
            if block != "primary":
                states[pid] = "fail: " + "; ".join(f"seed {r.seed}: {', '.join(r.reasons)}" for r in failed)
                continue
            replacement = ReserveList(stratum, reserves(rows, stratum, log), log).replace(pid, [r for r in results if r is not None])
            states[pid] = f"fail, replaced by {replacement}"
            for s in (1, 2):
                new_jobs += reference_jobs(by_id[replacement], s, setting)
            write_json_atomic(replacements_path(run), {"entries": log.entries})
    if new_jobs:
        run.add_jobs(new_jobs, f"replacement references after review: {', '.join(sorted({str(j.prompt_id) for j in new_jobs}))}")
    return {"states": states, "replacements": log.entries, "extra_cost_re": log.extra_cost_re,
            "audit_csv_sha256": file_sha256(decisions_path or audit_path(run))}
