"""The freeze before any reduced-budget output (protocol sections 4, 10 and 13).

``freeze`` records, in frozen.json: the accepted prompts, every reference's frame, video, noise and embedding hashes,
every mask file's hash, the audit file and its decisions, the replacement log, the setting (and for the main run the
registered lambda and the subsets for C5/C6, 75 percent and switch), and the manifest version it froze. Only then are
the condition and score jobs added to the manifest. Every condition job calls ``verify`` first, which recomputes the
hashes of what it reads and refuses to run if anything differs from the freeze.
"""

import numpy as np

from density.run import manifest as M
from density.run import plan
from density.run.masks import current_prompts, load_replacements, review
from density.run.store import Run, file_sha256, read_json, utc_now, write_json_atomic


class NotFrozen(RuntimeError):
    pass


def reference_entry(run: Run, ref: M.Job) -> dict:
    rec = run.record(ref)
    return {"uint8_sha256": rec["frames"]["uint8_sha256"], "png_sha256": rec["frames"]["png_sha256"],
            "video_decoded_sha256": rec["video"]["decoded_frames_sha256"], "noise_sha256": rec["noise"]["sha256"], "embeddings_sha256": rec["embeddings_sha256"]}


def mask_entry(run: Run, group: str) -> dict:
    folder = run.mask_dir(group)
    return {name: file_sha256(folder / name) for name in sorted(p.name for p in folder.iterdir())}


def subsets(accepted: dict[str, list[int]]) -> dict[str, dict[str, list[int]]]:
    """One seeded permutation per primary stratum: C5/C6 take its first 18, the 75 percent and switch blocks its first 8."""
    out = {name: {} for name in plan.SUBSETS}
    for k, stratum in enumerate(("face", "manipulation")):
        ids = sorted(accepted[stratum])
        order = [ids[i] for i in np.random.default_rng([plan.SUBSET_SEED, k]).permutation(len(ids))]
        for name, n in plan.SUBSETS.items():
            out[name][stratum] = sorted(order[:n])
    return out


def freeze(run: Run, rows: list[dict], setting: str, lam: float | None = None, registration: dict | None = None) -> dict:
    """Freeze a pilot (holdout prompts at ``setting``) or a main run (``lam`` and the registration required)."""
    kind = run.manifest()["kind"]
    if run.path("frozen.json").exists():
        raise RuntimeError("this run is already frozen")
    by_id = {int(r["id"]): r for r in rows}
    log = load_replacements(run)
    if kind == "pilot":
        states = review(run, rows, setting, block="holdout")["states"]
        pending = {p: s for p, s in states.items() if s.startswith("pending")}
        if pending:
            raise RuntimeError(f"holdout reviews pending: {pending}")
        accepted = {s: [p for p in ids if states[p] == "pass"] for s, ids in current_prompts(rows, log, "holdout").items()}
        excluded = {p: s for p, s in states.items() if s != "pass"}
        blocks = {"holdout": accepted}
        refs = [j for j in run.jobs() if j.kind == "reference" and j.setting == setting and j.block == "holdout"]
    else:
        if lam is None or registration is None:
            raise ValueError("the main run's freeze needs the registered lambda and the registration")
        states = review(run, rows, setting)["states"]
        not_passed = {p: s for p, s in states.items() if s != "pass"}
        if not_passed:
            raise RuntimeError(f"primary prompts not yet passed: {not_passed}")
        accepted = current_prompts(rows, log, "primary")
        excluded = {}
        blocks = {"primary": accepted, "secondary": current_prompts(rows, log, "secondary"), "control": current_prompts(rows, log, "control")}
        wanted = {p for b in blocks.values() for ids in b.values() for p in ids}
        refs = [j for j in run.jobs() if j.kind == "reference" and j.setting == setting and j.prompt_id in wanted]
    for ref in refs:
        if not run.done(ref) or not run.done(f"mask-{ref.group}"):
            raise RuntimeError(f"{ref.id} or its masks are not done")
    included = {p for b in blocks.values() for ids in b.values() for p in ids}
    refs = [r for r in refs if r.prompt_id in included]
    frozen = {
        "utc": utc_now(), "kind": kind, "setting": setting, "lambda": lam, "manifest_version": run.manifest()["version"],
        "blocks": blocks, "excluded": excluded, "replacements": log.entries,
        "references": {r.id: reference_entry(run, r) for r in refs},
        "masks": {r.group: mask_entry(run, r.group) for r in refs},
        "audit_csv_sha256": file_sha256(run.path("audit", "audit.csv")),
        "subsets": subsets(blocks["primary"]) if kind == "main" else None,
        "registration_sha256": registration.get("sha256") if registration else None,
    }
    write_json_atomic(run.path("frozen.json"), frozen)
    digest = file_sha256(run.path("frozen.json"))
    if kind == "main":  # the visual check's 30 pairs, chosen now, before any reduced-budget output exists
        from density import visual_check

        write_json_atomic(run.path("visual_check", "selection.json"), visual_check.select(frozen, digest))
    jobs = condition_jobs(run, frozen, refs)
    manifest = run.manifest()
    run.write_manifest({**manifest, "frozen_sha256": digest, "jobs": manifest["jobs"] + [j.to_dict() for j in jobs if j.id not in {x["id"] for x in manifest["jobs"]}]},
                       f"frozen ({kind}): {len(jobs)} condition and score jobs")
    return {"frozen_sha256": digest, "jobs_added": len(jobs), "excluded": excluded}


def condition_jobs(run: Run, frozen: dict, refs: list[M.Job]) -> list[M.Job]:
    by_group = {r.group: r for r in refs}
    jobs = []
    if frozen["kind"] == "pilot":
        for ref in refs:
            gens = [M.condition_job(ref, item, "pilot", f"mask-{ref.group}") for item in plan.pilot_conditions()]
            jobs += gens + [M.score_job(ref, [g.id for g in gens], f"mask-{ref.group}")]
        return jobs
    blocks, sub = frozen["blocks"], frozen["subsets"]
    per_group: dict[str, list[M.Job]] = {}
    for spec in plan.main_blocks(frozen["lambda"]):
        role = spec["prompts"]
        if role == "primary":
            ids = blocks["primary"]["face"] + blocks["primary"]["manipulation"]
        elif role == "primary:manipulation":
            ids = blocks["primary"]["manipulation"]
        elif role.startswith("subset:"):
            ids = sub[role.split(":")[1]]["face"] + sub[role.split(":")[1]]["manipulation"]
        else:
            ids = [p for v in blocks[role].values() for p in v]
        for pid in ids:
            seed = M.prompt_set.seeds(pid)[spec["seed"] - 1]
            ref = by_group[f"{pid:03d}-s{seed}-{frozen['setting']}"]
            for item in spec["items"]:
                per_group.setdefault(ref.group, []).append(M.condition_job(ref, item, spec["block"], f"mask-{ref.group}"))
    for group, gens in per_group.items():
        jobs += gens + [M.score_job(by_group[group], [g.id for g in gens], f"mask-{group}")]
    for ref in refs:  # references with no reduced-budget condition still get their quality scores
        if ref.group not in per_group:
            jobs.append(M.score_job(ref, [], f"mask-{ref.group}"))
    return jobs


def verify(run: Run, job: M.Job) -> dict:
    """Before a condition job: the freeze exists, is the one the manifest recorded, and still matches what the job reads."""
    path = run.path("frozen.json")
    manifest = run.manifest()
    if not path.exists() or "frozen_sha256" not in manifest:
        raise NotFrozen(f"{job.id}: the run is not frozen; reduced-budget jobs wait for the freeze")
    if file_sha256(path) != manifest["frozen_sha256"]:
        raise NotFrozen("frozen.json differs from the freeze the manifest recorded")
    frozen = read_json(path)
    ref_id = f"ref-{job.group}"
    if ref_id not in frozen["references"]:
        raise NotFrozen(f"{job.id}: its reference {ref_id} is not in the freeze")
    if reference_entry(run, run.job(ref_id)) != frozen["references"][ref_id]:
        raise NotFrozen(f"{job.id}: its reference's record differs from the freeze")
    if mask_entry(run, job.group) != frozen["masks"][job.group]:
        raise NotFrozen(f"{job.id}: its masks differ from the freeze")
    return {"frozen_sha256": manifest["frozen_sha256"]}
