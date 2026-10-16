"""The run manifest and runner: job IDs and costs, section 12's grid, append-only manifests, atomic records, the order
of phases, review and replacements, the freeze, the per-prompt platform pin, and the session's gates."""

import csv
import json
import os
from pathlib import Path

import numpy as np
import pytest

from density import prompts as prompt_set
from density.masks.eligibility import ReserveExhausted
from density.run import freeze as F
from density.run import manifest as M
from density.run import plan, session as S
from density.run.masks import review, write_audit
from density.run.store import DRY_RUN_ROOT, Run, check_run_folder, load_frames, save_frames, write_bytes_atomic, write_json_atomic


def rows(primary=36, holdout=6, body=16, control=16, reserves=3):
    out, pid = [], 1

    def add(stratum, block, n, reserve=False):
        nonlocal pid
        for k in range(n):
            out.append({"id": str(pid), "stratum": stratum, "block": block, "reserve_order": str(k + 1) if reserve else "", "text": f"prompt {pid}"})
            pid += 1

    for s in ("face", "manipulation"):
        add(s, "holdout", holdout)
        add(s, "primary", primary)
        add(s, "reserve", reserves, True)
    add("body", "secondary", body)
    add("control", "control", control)
    return out


def new_run(tmp_path, kind="pilot", jobs=None, **extra) -> Run:
    run = Run(tmp_path / "run")
    rs = extra.pop("rows", rows())
    write_json_atomic(run.path("inputs", "prompts.json"), {"rows": rs})
    jobs = jobs if jobs is not None else (M.pilot_initial(rs) if kind == "pilot" else M.main_initial(rs, "sh8g6"))
    run.write_manifest({"run": "run", "kind": kind, "dry_run": False, "prompts_csv_sha256": "x", "jobs": [j.to_dict() for j in jobs], **extra}, "test")
    return run


def done(run: Run, job_id: str, **outputs):
    write_json_atomic(run.job_dir(job_id) / "record.json", {"job": {"id": job_id}, "status": "done", **outputs})


def test_job_ids_groups_and_costs():
    ref, mask = M.reference_jobs({"id": "17", "stratum": "face", "block": "primary"}, 2, "sh8g6")
    assert ref.id == "ref-017-s2017-sh8g6" and mask.id == "mask-017-s2017-sh8g6" and mask.depends == (ref.id,) and ref.depends == ("embed-017",)
    gen = M.condition_job(ref, plan.GridItem("C4", "HSA-50", 3.0), "primary seed 2", mask.id)
    assert gen.id == "gen-017-s2017-sh8g6-C4-HSA-50-l3" and gen.group == ref.group and gen.planning_re == 0.5 and gen.machine == "h100"
    assert M.Job.from_dict(gen.to_dict()) == gen
    assert M.condition_job(ref, plan.GridItem("C2"), "x", mask.id).id == "gen-017-s2017-sh8g6-C2"
    assert ref.planning_re == 1.0 and mask.machine == "slice"


def pilot_frozen(rs):
    holdout = {s: [int(r["id"]) for r in rs if r["block"] == "holdout" and r["stratum"] == s] for s in ("face", "manipulation")}
    return {"kind": "pilot", "setting": "sh8g6", "blocks": {"holdout": holdout}}


def main_frozen(rs):
    blocks = {b: {} for b in ("primary", "secondary", "control")}
    for r in rs:
        if r["block"] in blocks:
            blocks[r["block"]].setdefault(r["stratum"], []).append(int(r["id"]))
    return {"kind": "main", "setting": "sh8g6", "lambda": 3.0, "blocks": blocks, "subsets": F.subsets(blocks["primary"])}


def test_section_12_grid_and_planning_costs():
    rs = rows()
    pilot = M.pilot_initial(rs)
    refs = [j for j in pilot if j.kind == "reference"]
    pilot += F.condition_jobs(None, pilot_frozen(rs), [r for r in refs if r.setting == "sh8g6"])
    assert sum(j.planning_re for j in pilot) == 24 + 24 + 6  # settings, lambda, HSA reproduction
    main = M.main_initial(rs, "sh8g6")
    refs = [j for j in main if j.kind == "reference"]
    assert sum(j.planning_re for j in refs) == 72 * 2 + 16 + 16
    main += F.condition_jobs(None, main_frozen(rs), refs)
    by_block = {}
    for j in main:
        if j.kind == "condition":
            by_block[j.grid_block] = by_block.get(j.grid_block, 0) + j.planning_re
    assert by_block == {"primary seed 1": 108, "primary seed 2": 72, "C5 and C6": 36, "A1": 18, "full body": 16, "controls": 16, "75 percent": 24, "switch points": 32}
    assert sum(j.planning_re for j in main) == 498  # 558 RE with the pilot's 54 and the smoke test's 6
    scores = [j for j in main if j.kind == "score"]
    assert len(scores) == len(refs) and all(j.depends[0].startswith("ref-") for j in scores)
    cost = M.planned_cost(main)
    assert cost["totals"]["h100"]["re"] == 498 and cost["totals"]["slice"]["hours"] > 0


def test_subsets_are_seeded_nested_and_per_stratum():
    accepted = {"face": list(range(1, 37)), "manipulation": list(range(101, 137))}
    a, b = F.subsets(accepted), F.subsets(accepted)
    assert a == b and len(a["C5_C6"]["face"]) == 18 and len(a["budget_switch"]["manipulation"]) == 8
    order = np.random.default_rng([plan.SUBSET_SEED, 0]).permutation(36)
    assert a["budget_switch"]["face"] == sorted(accepted["face"][i] for i in order[:8])
    assert set(a["budget_switch"]["face"]) <= set(a["C5_C6"]["face"])


def test_manifest_versions_only_add_jobs(tmp_path):
    run = new_run(tmp_path)
    first = run.manifest()
    assert first["version"] == 1 and (run.folder / "manifest_v1.json").exists()
    changed = {**first, "jobs": first["jobs"][1:]}
    with pytest.raises(ValueError, match="cannot change or remove"):
        run.write_manifest(changed, "drop a job")
    with pytest.raises(ValueError, match="cannot change dry_run"):
        run.write_manifest({**first, "dry_run": True}, "flip")
    extra = M.reference_jobs(rows()[0], 2, "sh8g6")
    m = run.add_jobs(extra, "more")
    assert m["version"] == 2 and len(m["jobs"]) == len(first["jobs"]) + 2 and [h["note"] for h in m["history"]] == ["test", "more"]


def test_atomic_writes_keep_the_old_file_on_failure(tmp_path, monkeypatch):
    path = tmp_path / "a.json"
    write_bytes_atomic(path, b"old")
    monkeypatch.setattr(os, "replace", lambda *a: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        write_bytes_atomic(path, b"new")
    assert path.read_bytes() == b"old" and [p.name for p in tmp_path.iterdir()] == ["a.json"]


def test_frames_are_checked_by_hash(tmp_path):
    frames = np.random.default_rng(0).integers(0, 256, (3, 8, 8, 3), dtype=np.uint8)
    record = save_frames(frames, tmp_path / "frames")
    assert np.array_equal(load_frames(tmp_path / "frames", record), frames)
    other = save_frames(frames[::-1].copy(), tmp_path / "other")
    (tmp_path / "frames" / "frame_000.png").write_bytes((tmp_path / "other" / "frame_000.png").read_bytes())
    with pytest.raises(RuntimeError, match="hash differs"):
        load_frames(tmp_path / "frames", record)
    assert other["uint8_sha256"] != record["uint8_sha256"]


def test_dry_and_real_runs_never_share_folders(tmp_path):
    check_run_folder(DRY_RUN_ROOT / "x", dry_run=True)
    check_run_folder(tmp_path / "real", dry_run=False)
    with pytest.raises(RuntimeError):
        check_run_folder(tmp_path / "real", dry_run=True)
    with pytest.raises(RuntimeError):
        check_run_folder(DRY_RUN_ROOT / "x", dry_run=False)


def test_platform_pin_per_prompt_and_machine(tmp_path):
    run = new_run(tmp_path)
    job = run.jobs()[0]
    ident = {"gpu": "NVIDIA H100 80GB HBM3", "image": "runpod/pytorch:1", "uv_lock_sha256": "abc"}
    S.check_platform(run, job, ident)
    S.check_platform(run, job, ident)
    for change in ({"gpu": "NVIDIA A100"}, {"image": "other"}, {"uv_lock_sha256": "def"}):
        with pytest.raises(S.PlatformMismatch):
            S.check_platform(run, job, {**ident, **change})


def write_mask_record(run: Run, job: M.Job, passed: bool):
    folder = run.mask_dir(job.group)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "sheet.png").write_bytes(job.id.encode())
    (folder / "masks.json").write_text("{}")
    sheet_sha = __import__("hashlib").sha256(job.id.encode()).hexdigest()
    done(run, job.id, eligibility={"reasons": [] if passed else ["roi_usable_frames"], "usable_frames": 81 if passed else 10, "passed": passed},
         pipeline_decision="pass" if passed else "fail: roi_usable_frames", contact_intervals=[], outside_frames=81, sheet="sheet.png", sheet_sha256=sheet_sha)


def fill_reviews(run: Run, decide):
    path = run.path("audit", "audit.csv")
    lines = path.read_text().splitlines(keepends=True)
    table = list(csv.DictReader(l for l in lines if not l.startswith("#")))
    for r in table:
        decision = decide(r)
        if decision:
            r.update(reviewer_decision=decision[0], reviewer_reason=decision[1], reviewer_written_reason=decision[2], reviewer="Chase", review_date="2026-10-08")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)


def test_review_replaces_failed_primary_prompts_in_reserve_order(tmp_path):
    rs = rows(primary=2, holdout=0, body=0, control=0, reserves=2)
    run = new_run(tmp_path, "main", rows=rs, setting="sh8g6")
    face = [int(r["id"]) for r in rs if r["block"] == "primary" and r["stratum"] == "face"]
    face_reserves = [int(r["id"]) for r in rs if r["block"] == "reserve" and r["stratum"] == "face"]
    for job in run.jobs():
        if job.kind == "mask":
            write_mask_record(run, job, passed=not (job.prompt_id == face[0] and job.seed % 1000 == job.prompt_id))  # seed 1 of the first face prompt fails
    write_audit(run)
    states = review(run, rs, "sh8g6")["states"]
    assert states[face[0]] == f"fail, replaced by {face_reserves[0]}"  # the pipeline's failure needs no review
    assert all(s.startswith("pending") for p, s in states.items() if p != face[0])
    assert {j.id for j in run.jobs() if j.prompt_id == face_reserves[0] and j.kind == "reference"} == {f"ref-{face_reserves[0]:03d}-s{s}-sh8g6" for s in prompt_set.seeds(face_reserves[0])}
    # Chase rejects the replacement's seed 2 for a mask-invalidity reason; the next reserve takes its place
    for job in run.jobs():
        if job.kind == "mask" and not run.done(job):
            write_mask_record(run, job, passed=True)
    write_audit(run)
    fill_reviews(run, lambda r: ("reject", "audit_mask_invalid", "spills onto the sofa") if r["clip"].startswith(f"{face_reserves[0]:03d}-s2") else ("accept", "", ""))
    out = review(run, rs, "sh8g6")
    assert out["states"][face_reserves[0]] == f"fail, replaced by {face_reserves[1]}"
    assert [e["replacement"] for e in out["replacements"]] == face_reserves and out["extra_cost_re"] == 4
    for job in run.jobs():
        if job.kind == "mask" and not run.done(job):
            write_mask_record(run, job, passed=job.prompt_id != face_reserves[1])
    write_audit(run)
    with pytest.raises(ReserveExhausted):
        review(run, rs, "sh8g6")


def test_a_review_of_a_changed_sheet_is_refused(tmp_path):
    rs = rows(primary=1, holdout=0, body=0, control=0, reserves=1)
    run = new_run(tmp_path, "main", rows=rs, setting="sh8g6")
    for job in run.jobs():
        if job.kind == "mask":
            write_mask_record(run, job, passed=True)
    write_audit(run)
    fill_reviews(run, lambda r: ("accept", "", ""))
    job = next(j for j in run.jobs() if j.kind == "mask")
    rec = run.record(job)
    write_json_atomic(run.job_dir(job) / "record.json", {**rec, "sheet_sha256": "0" * 64})
    with pytest.raises(RuntimeError, match="the review is for sheet"):
        review(run, rs, "sh8g6")


def test_freeze_verify_refuses_changed_inputs(tmp_path):
    rs = rows(primary=1, holdout=0, body=0, control=0, reserves=0)
    run = new_run(tmp_path, "main", rows=rs, setting="sh8g6")
    ref = next(j for j in run.jobs() if j.kind == "reference")
    gen = M.condition_job(ref, plan.GridItem("C3", "HSA-50"), "primary seed 1", f"mask-{ref.group}")
    with pytest.raises(F.NotFrozen):
        F.verify(run, gen)
    done(run, ref.id, frames={"uint8_sha256": "u", "png_sha256": {}}, video={"decoded_frames_sha256": "v"}, noise={"sha256": "n"}, embeddings_sha256={"prompt": "e"})
    write_mask_record(run, run.job(f"mask-{ref.group}"), passed=True)
    frozen = {"references": {ref.id: F.reference_entry(run, ref)}, "masks": {ref.group: F.mask_entry(run, ref.group)}}
    write_json_atomic(run.path("frozen.json"), frozen)
    m = run.manifest()
    run.write_manifest({**m, "frozen_sha256": __import__("density.run.store", fromlist=["x"]).file_sha256(run.path("frozen.json"))}, "frozen")
    assert "frozen_sha256" in F.verify(run, gen)
    (run.mask_dir(ref.group) / "masks.json").write_text('{"changed": true}')
    with pytest.raises(F.NotFrozen, match="masks differ"):
        F.verify(run, gen)


class FakeExecutor:
    def __init__(self, fail_on=None):
        self.ran, self.fail_on = [], fail_on

    def __call__(self, job):
        self.ran.append(job.id)
        print(f"running {job.id}")
        if job.id == self.fail_on:
            raise RuntimeError("simulated failure")
        return {"out": job.id}


@pytest.fixture
def real_gates(monkeypatch):
    calls = []
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i=0: "NVIDIA H100 80GB HBM3")
    monkeypatch.setattr(S.determinism, "configure", lambda strict: calls.append(("strict", strict)) or {"strict": strict})
    import density.canary

    monkeypatch.setattr(density.canary, "require_passed", lambda: calls.append("canary") or {"passed": True})
    monkeypatch.setattr(S, "encode_canary", lambda: calls.append("encode") or {"passed": True})
    import density.metric_canary

    monkeypatch.setattr(density.metric_canary, "require_passed", lambda machine: calls.append(f"metric {machine}") or {"passed": True})
    monkeypatch.setattr(S.volume, "require_room", lambda root=None, stop_bytes=None: calls.append("volume") or {"used_bytes": 1, "seconds": 0.0})
    monkeypatch.setattr(S, "platform_identity", lambda machine, dry: {"gpu": "NVIDIA H100 80GB HBM3", "image": "img", "uv_lock_sha256": "lock"})
    monkeypatch.setattr(S, "environment", lambda identity: {"host": "test"})
    monkeypatch.setenv("POD_IMAGE", "runpod/pytorch:test")
    return calls


def test_session_runs_ready_jobs_in_order_after_the_gates_and_resumes(tmp_path, real_gates):
    run = new_run(tmp_path, rows=rows(primary=0, holdout=1, body=0, control=0, reserves=0))
    fake = FakeExecutor(fail_on="ref-002-s1002-sh8g6")
    record = S.session(run.folder, "h100", device="cuda", execute=fake)
    assert real_gates[:4] == [("strict", True), "canary", "encode", "metric h100"] and "volume" in real_gates
    assert fake.ran == ["embed-001", "embed-002", "ref-001-s1001-sh8g6", "ref-001-s1001-sh5g5", "ref-002-s1002-sh8g6"]
    assert record["stop"].startswith("RuntimeError: simulated failure")
    failed = json.loads((run.job_dir("ref-002-s1002-sh8g6") / "failed.json").read_text())
    assert "simulated failure" in failed["traceback"] and "running ref-002-s1002-sh8g6" in (run.job_dir("ref-002-s1002-sh8g6") / "log.txt").read_text()
    assert run.done("ref-001-s1001-sh8g6") and not run.done("ref-002-s1002-sh8g6")
    again = FakeExecutor()
    record = S.session(run.folder, "h100", device="cuda", execute=again)
    assert again.ran == ["ref-002-s1002-sh8g6", "ref-002-s1002-sh5g5"] and record["stop"] is None  # masks wait for the slice
    assert not (run.job_dir("ref-002-s1002-sh8g6") / "failed.json").exists()


def test_a_failed_gate_runs_nothing(tmp_path, real_gates, monkeypatch):
    import density.canary

    monkeypatch.setattr(density.canary, "require_passed", lambda: (_ for _ in ()).throw(density.canary.CanaryNotPassed("no canary record")))
    run = new_run(tmp_path, rows=rows(primary=0, holdout=1, body=0, control=0, reserves=0))
    fake = FakeExecutor()
    record = S.session(run.folder, "h100", device="cuda", execute=fake)
    assert fake.ran == [] and "GateFailed" in record["stop"] and "no canary record" in record["stop"]


def test_the_volume_stop_runs_one_du_then_counts_what_each_job_writes(tmp_path, real_gates, monkeypatch):
    """The registered stop (Chase, 2026-10-07): one full du at the start, then the allocated size of each job's files;
    a running total that reaches the stop calls a full du, which decides."""
    from density.volume import STOP_BYTES, VolumeFull

    run = new_run(tmp_path, rows=rows(primary=0, holdout=1, body=0, control=0, reserves=0))
    fake = FakeExecutor()
    checks = []

    def room(root=None, stop_bytes=None):
        checks.append(1)
        if len(checks) > 1:
            raise VolumeFull("112 GB")
        return {"used_bytes": STOP_BYTES - 1, "seconds": 0.0}  # one byte short: the first job's files reach the stop

    monkeypatch.setattr(S.volume, "require_room", room)
    record = S.session(run.folder, "h100", device="cuda", execute=fake)
    assert fake.ran == ["embed-001"] and "VolumeFull" in record["stop"] and len(checks) == 2
    first, running = record["volume"][0], record["volume"][1]
    assert first["kind"] == "full" and first["reason"] == "session start"
    on_disk = sum(p.stat().st_blocks * 512 for p in run.job_dir("embed-001").rglob("*") if p.is_file())
    assert running["kind"] == "running" and running["after"] == "embed-001" and running["added_bytes"] >= on_disk > 0


def test_the_session_does_a_full_du_only_at_the_start_and_every_30_minutes(tmp_path, real_gates):
    run = new_run(tmp_path, rows=rows(primary=0, holdout=1, body=0, control=0, reserves=0))
    record = S.session(run.folder, "h100", device="cuda", execute=FakeExecutor())
    kinds = [v["kind"] for v in record["volume"]]
    assert kinds[0] == "full" and kinds.count("full") == 1 and kinds.count("running") == len(record["jobs"]) > 1
    assert real_gates.count("volume") == 1


def test_reduced_budget_jobs_exist_only_after_the_freeze(tmp_path):
    run = new_run(tmp_path)
    assert {j.kind for j in run.jobs()} == {"embed", "reference", "mask"}


def test_a_real_session_needs_the_pods_image(tmp_path, real_gates, monkeypatch):
    monkeypatch.delenv("POD_IMAGE")
    run = new_run(tmp_path, rows=rows(primary=0, holdout=1, body=0, control=0, reserves=0))
    fake = FakeExecutor()
    record = S.session(run.folder, "h100", device="cuda", execute=fake)
    assert fake.ran == [] and "POD_IMAGE" in record["stop"]


def test_group_checks_assert_sections_8_and_11(tmp_path):
    from density.run.score import group_checks

    rs = rows(primary=0, holdout=0, body=0, control=1, reserves=0)
    run = new_run(tmp_path, "main", rows=rs, setting="sh8g6")
    ref = next(j for j in run.jobs() if j.kind == "reference")
    gens = [M.condition_job(ref, plan.GridItem(c, "HSA-50", 3.0 if c == "C4" else None), "controls", f"mask-{ref.group}") for c in ("C3", "C4")]
    score = M.score_job(ref, [g.id for g in gens], f"mask-{ref.group}")
    run.add_jobs(gens + [score], "conditions")

    def write(c, counts, latents, frames):
        g = next(x for x in gens if x.condition == c)
        done(run, g.id, active_counts={"active_counts": counts}, latents_sha256=latents, video={"decoded_frames_sha256": frames})

    write("C3", [10, 5], "L", "F")
    write("C4", [10, 5], "L", "F")
    assert group_checks(run, score)["empty_mask_c4_equals_c3"] == {"HSA-50": "C4 equals C3"}
    write("C4", [10, 6], "L", "F")
    with pytest.raises(RuntimeError, match="active-token counts differ"):
        group_checks(run, score)
    write("C4", [10, 5], "L2", "F")
    with pytest.raises(RuntimeError, match="empty mask C4 differs from C3"):
        group_checks(run, score)


def test_replacements_apply_only_to_the_primary_blocks():
    from density.masks.eligibility import ReplacementLog
    from density.run.masks import current_prompts

    rs = rows(primary=2, holdout=1, body=1, control=1, reserves=1)
    face = [int(r["id"]) for r in rs if r["block"] == "primary" and r["stratum"] == "face"]
    reserve = next(int(r["id"]) for r in rs if r["block"] == "reserve" and r["stratum"] == "face")
    log = ReplacementLog([{"stratum": "face", "failed_prompt": face[0], "replacement": reserve, "extra_cost_re": 2}])
    assert current_prompts(rs, log)["face"] == [reserve, face[1]]
    assert list(current_prompts(rs, log, "secondary")) == ["body"] and list(current_prompts(rs, log, "control")) == ["control"]
    assert current_prompts(rs, log, "holdout")["face"] == [int(r["id"]) for r in rs if r["block"] == "holdout" and r["stratum"] == "face"]


def test_dry_run_stand_ins_follow_the_generation():
    from density.run.generate import DryRunBackend

    backend = DryRunBackend.__new__(DryRunBackend)  # frames() needs no models
    backend.noise_prompts = set()
    ref = M.Job("reference", 5, "control", "control", 1005, "sh8g6", "C1")
    c3, c4 = (M.condition_job(ref, plan.GridItem(c, "HSA-50", 3.0 if c == "C4" else None), "controls", "m") for c in ("C3", "C4"))
    reference = np.full((2, 16, 16, 3), 128, np.uint8)
    empty = np.zeros((2, 16, 16), bool)
    assert np.array_equal(backend.frames(c3, None, reference, empty, "ab" * 32), backend.frames(c4, None, reference, empty, "ab" * 32))
    roi = empty.copy()
    roi[:, 4:8, 4:8] = True
    assert not np.array_equal(backend.frames(c3, None, reference, roi, "ab" * 32), backend.frames(c4, None, reference, roi, "ab" * 32))
    assert not np.array_equal(backend.frames(c3, None, reference, empty, "ab" * 32), backend.frames(c3, None, reference, empty, "cd" * 32))


def test_score_jobs_use_lossless_frames_and_report_the_mp4_beside_them(tmp_path):
    from density import video
    from density.metrics.registered import QUALITY_DIMENSIONS, QUALITY_INPUT
    from density.run.score import run_score

    rs = rows(primary=0, holdout=0, body=0, control=1, reserves=0)
    run = new_run(tmp_path, "main", rows=rs, setting="sh8g6")
    ref = next(j for j in run.jobs() if j.kind == "reference")
    gen = M.condition_job(ref, plan.GridItem("C2"), "controls", f"mask-{ref.group}")
    score = M.score_job(ref, [gen.id], f"mask-{ref.group}")
    run.add_jobs([gen, score], "conditions")
    rng = np.random.default_rng(0)
    lossless = {}
    for job in (ref, gen):
        frames = rng.integers(0, 256, (3, 16, 16, 3), dtype=np.uint8)
        lossless[job.id] = frames
        mp4 = video.encode(frames, run.job_dir(job) / "video.mp4")
        store = {"frames": save_frames(frames, run.job_dir(job) / "frames")} if job is ref else {"lossless": video.encode_lossless(frames, run.job_dir(job) / "frames.mkv")}
        done(run, job.id, video={**mp4, "path": "video.mp4"}, latents_sha256="L", **store)

    class Quality:  # records which frames it scored
        def __init__(self):
            self.seen = []

        def score(self, path, frames):
            self.seen.append((str(path), frames))
            value = float(frames.mean())
            return {"scores": dict.fromkeys(QUALITY_DIMENSIONS, value), "raw": dict.fromkeys(QUALITY_DIMENSIONS, value)}

    q = Quality()
    out = run_score(run, score, q, detectors=None)
    entry = out["videos"][gen.id]
    assert entry["quality_input"] == QUALITY_INPUT and entry["quality"]["imaging_quality"] == pytest.approx(lossless[gen.id].mean())
    assert entry["mp4_minus_lossless"]["imaging_quality"] == pytest.approx(entry["quality_mp4"]["imaging_quality"] - entry["quality"]["imaging_quality"])
    assert any(np.array_equal(f, lossless[gen.id]) for _, f in q.seen) and any(np.array_equal(f, lossless[ref.id]) for _, f in q.seen)
