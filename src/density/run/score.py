"""Slice jobs after the conditions: the five quality dimensions and the detector checks of one reference and all its
conditions, on one host, so their paired differences share that host (protocol section 7).

Registered (Chase, 2026-10-07): every score is measured before lossy compression. The quality guardrails and the
detector checks use each video's lossless frames (a reference's PNGs, a condition's FFV1 copy), checked by hash. Each
video's CRF 10 MP4 is scored too, as a descriptive check (``quality_mp4``, and its difference from the lossless scores).

Before scoring, the group's generation records are checked against sections 8 and 11: at each preset, every HSA
condition of the prompt and seed ran the same active-token count at every iteration; and on a control prompt, whose
mask is empty by design, C4's final latents and decoded frames equal C3's.

Each video's scores are written to scores/<group>/<job>.json as they finish, with the host that scored them, so a
stopped session resumes where it was; on a different host the whole group is scored again. A missing quality score
stops the job: it is a technical failure, rerun under the same settings.
"""

import math
import socket
import time

from density import video
from density.metrics import detection
from density.metrics.registered import DETECTORS, QUALITY_DIMENSIONS, QUALITY_INPUT
from density.run.manifest import Job
from density.run.store import Run, lossless_frames, read_json, write_json_atomic



def decoded(run: Run, job_id: str):
    rec = run.record(job_id)
    path = run.job_dir(job_id) / rec["video"]["path"]
    frames = video.decode_rgb(path)
    if video.frames_sha256(frames) != rec["video"]["decoded_frames_sha256"]:
        raise RuntimeError(f"{job_id}: the MP4 decodes to frames that differ from its record")
    return path, frames


def group_checks(run: Run, job: Job) -> dict:
    """Sections 8 and 11 across a prompt-seed's conditions; raises on any failure."""
    gens = [run.job(d) for d in job.depends if d.startswith("gen-")]
    by_preset: dict[str, dict[str, list[int]]] = {}
    for g in gens:
        rec = run.record(g)
        if rec.get("active_counts"):
            by_preset.setdefault(g.preset, {})[g.id] = rec["active_counts"]["active_counts"]
    for preset, counts in by_preset.items():
        if len({tuple(v) for v in counts.values()}) > 1:
            raise RuntimeError(f"{job.group}: active-token counts differ across conditions at {preset} (section 8): {sorted(counts)}")
    empty = {}
    if job.stratum == "control":
        for preset in {g.preset for g in gens if g.condition in ("C3", "C4")}:
            pair = {g.condition: run.record(g) for g in gens if g.preset == preset and g.condition in ("C3", "C4")}
            if len(pair) == 2:
                same = pair["C3"]["latents_sha256"] == pair["C4"]["latents_sha256"] and pair["C3"]["video"]["decoded_frames_sha256"] == pair["C4"]["video"]["decoded_frames_sha256"]
                if not same:
                    raise RuntimeError(f"{job.group}: with the explicit empty mask C4 differs from C3 at {preset}, an implementation failure (section 11)")
                empty[preset] = "C4 equals C3"
    return {"active_counts_equal_across_conditions": sorted(by_preset), "empty_mask_c4_equals_c3": empty}


def run_score(run: Run, job: Job, quality_models, detectors) -> dict:
    folder = run.score_dir(job.group)
    ref_id = f"ref-{job.group}"
    videos = [ref_id] + [d for d in job.depends if d.startswith("gen-")]
    checks = group_checks(run, job)
    host = socket.gethostname()
    earlier = {read_json(p).get("host") for p in folder.glob("*.json")} if folder.exists() else set()
    rescored = None
    if earlier - {host}:  # same-host pairing: never mix hosts within a group
        rescored = f"scored before on {sorted(earlier - {host})}; the whole group is scored again on {host}"
        for p in folder.glob("*.json"):
            p.unlink()
    reference_detections = None
    out = {}
    for vid in videos:
        path = folder / f"{vid}.json"
        if path.exists():
            out[vid] = read_json(path)
            continue
        start = time.perf_counter()
        frames = lossless_frames(run, vid)  # the registered input
        q = quality_models.score(folder / f"{vid}.lossless.mp4", frames)  # the path only labels the frames VBench is given
        mp4, mp4_frames = decoded(run, vid)
        q_mp4 = quality_models.score(mp4, mp4_frames)
        bad = [d for d in QUALITY_DIMENSIONS if not (math.isfinite(q["scores"][d]) and math.isfinite(q_mp4["scores"][d]))]
        if bad:
            raise FloatingPointError(f"{vid}: non-finite quality scores {bad}")
        entry = {"host": host, "quality_input": QUALITY_INPUT, "quality": q["scores"], "quality_raw": q["raw"], "quality_mp4": q_mp4["scores"],
                 "mp4_minus_lossless": {d: q_mp4["scores"][d] - q["scores"][d] for d in QUALITY_DIMENSIONS}, "quality_seconds": time.perf_counter() - start}
        if job.stratum in DETECTORS:
            start = time.perf_counter()
            if reference_detections is None:
                ref_frames = frames if vid == ref_id else lossless_frames(run, ref_id)
                reference_detections = detection.detect_video(detectors, ref_frames, job.stratum)
            if vid != ref_id:
                entry["detection"] = detection.compare(reference_detections, detection.detect_video(detectors, frames, job.stratum), job.stratum)
            entry["detection_seconds"] = time.perf_counter() - start
        write_json_atomic(path, entry)
        out[vid] = entry
    return {"videos": {vid: {k: e[k] for k in ("quality_input", "quality", "quality_mp4", "mp4_minus_lossless", "detection") if k in e} for vid, e in out.items()},
            "files": {vid: str((folder / f"{vid}.json").relative_to(run.folder)) for vid in videos}, "scored_on": host, "group_checks": checks,
            **({"rescored": rescored} if rescored else {})}
