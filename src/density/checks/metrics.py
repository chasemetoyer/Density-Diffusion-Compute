"""Milestone 5's check: the prompt tooling, the metrics, the runner's dry runs, the pilot rules, the frozen analysis and
the registration, on the Mac. Writes results/checks/metrics.md and .json.

    python -m density.checks.metrics all          everything, about three hours on the Mac (run it at a clean commit)
    python -m density.checks.metrics PART ...     metrics | determinism-child | pilot | main | analysis | report

Timings are the Mac's and say nothing about the pods. Dry-run numbers describe stand-ins, not Wan (density.checks.dryrun).
"""

import argparse
import io
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from density.checks import euler

ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = ROOT / "results" / "checks"
OUT_DIR = REPORT_DIR / "metrics"  # per-part JSON, not versioned
REPORT_JSON, REPORT_MD = REPORT_DIR / "metrics.json", REPORT_DIR / "metrics.md"
CLIPS = ("face", "hands", "body", "landscape")
CLIP_STRATA = {"face": "face", "hands": "manipulation", "body": "body", "landscape": "control"}


def save(part: str, data: dict) -> dict:
    from density.run.store import write_json_atomic

    data = {"part": part, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "code": euler.git_state(), **data}
    write_json_atomic(OUT_DIR / f"{part}.json", data)
    return data


# Part 1


def check_prompts() -> dict:
    from density import prompts as prompt_set
    from density.checks import dryrun

    lines = prompt_set.source_lines()
    sources = prompt_set.sources()
    out = {"sources": {k: {"prompts": len(v), "registered": sources[k]["prompts"], "commit": sources[k]["commit"], "sha256_verified": True} for k, v in lines.items()}}
    with tempfile.TemporaryDirectory() as tmp:
        for name, (rows, _) in (("mock pilot", dryrun.pilot_prompts()), ("mock main", dryrun.main_prompts())):
            report = prompt_set.validate(dryrun.write_prompts(Path(tmp) / "p.csv", rows), lines=lines, tokenizer=prompt_set.load_tokenizer(), complete=False)
            out[name] = {"rows": report["rows"], "errors": report["errors"], "max_tokens": report["max_tokens"]}
    # the drafted prompt set, in both scopes; it is reported, not judged: Chase reviews every row before any run uses it
    drafted = {scope: prompt_set.validate(prompt_set.PROMPTS_CSV, lines=lines, tokenizer=prompt_set.load_tokenizer(), scope=scope)
               for scope in ("holdout", "all")}
    out["configs/prompts.csv"] = {"rows": drafted["all"]["rows"], "counts": drafted["all"].get("counts"),
                                  "holdout_errors": drafted["holdout"]["errors"], "all_errors": drafted["all"]["errors"],
                                  "note": "drafted by the registered procedure; Chase has not reviewed it yet"}
    out["passed"] = all(not out[k]["errors"] for k in ("mock pilot", "mock main"))
    return out


# Part 2


def weights_check() -> dict:
    from density.masks.weights import sha256_file
    from density.metrics import weights

    out = {}
    for name, w in weights.WEIGHTS.items():
        path = weights.ROOT / w.path
        ok = path.exists() and path.stat().st_size == w.size and sha256_file(path) == w.sha256
        out[name] = {"path": w.path, "size": w.size, "sha256": w.sha256, "verified": ok, "loads": w.loads, "license": w.license}
    return out


def lossless_bytes(frames: np.ndarray) -> dict:
    """FFV1 (bgr0) in Matroska, checked to decode back exactly."""
    import av

    buf = io.BytesIO()
    with av.open(buf, "w", format="matroska") as c:
        s = c.add_stream("ffv1", rate=16)
        s.width, s.height, s.pix_fmt = frames.shape[2], frames.shape[1], "bgr0"
        for f in frames:
            for p in s.encode(av.VideoFrame.from_ndarray(f, format="rgb24").reformat(format="bgr0")):
                c.mux(p)
        for p in s.encode(None):
            c.mux(p)
    size = buf.getbuffer().nbytes
    buf.seek(0)
    with av.open(buf) as c:
        back = np.stack([fr.to_ndarray(format="rgb24") for fr in c.decode(video=0)])
    return {"bytes": size, "exact": bool(np.array_equal(back, frames))}


def stock_roi(name: str) -> np.ndarray | None:
    from density.masks.clips import CLIPS_DIR
    from density.masks.storage import load_masks

    path = CLIPS_DIR / name / "masks"
    return load_masks(path)["ROI"] if (path / "masks.json").exists() else None


def check_metrics() -> dict:
    from density import video
    from density.masks.clips import load_frames
    from density.masks.measurement import measurement_masks
    from density.metrics import fidelity, quality
    from density.metrics.registered import QUALITY_DIMENSIONS
    from density.run.session import encode_canary
    from density.run.store import png_bytes

    out = {"weights": weights_check(), "encode_canary": encode_canary()}
    lp = fidelity.LPIPS("cpu")
    qm = quality.QualityModels("cpu")
    clips, compression = {}, {}
    with tempfile.TemporaryDirectory() as tmp:
        for name in CLIPS:
            frames = load_frames(name)
            t0 = time.perf_counter()
            rec = video.encode(frames, Path(tmp) / f"{name}.mp4")
            encode_s = time.perf_counter() - t0
            again = video.encode(frames, Path(tmp) / f"{name}_again.mp4")
            mp4 = video.decode_rgb(Path(tmp) / f"{name}.mp4")
            clips[name] = {"psnr_vs_lossless_db": rec["psnr_vs_lossless_db"], "mp4_bytes": rec["file_bytes"], "png_bytes": sum(len(png_bytes(f)) for f in frames),
                           "ffv1": lossless_bytes(frames), "decoded_identical_across_encodes": rec["decoded_frames_sha256"] == again["decoded_frames_sha256"],
                           "x264_options": rec["x264_options"], "encode_seconds_mac": encode_s}
            roi = stock_roi(name)
            m = measurement_masks(roi if roi is not None else np.zeros(frames.shape[:3], bool))
            f = fidelity.fidelity(lp, mp4, frames, m)
            t0 = time.perf_counter()
            q_lossless = qm.score(Path(tmp) / f"{name}_lossless.mp4", frames)["scores"]
            q_mp4 = qm.score(Path(tmp) / f"{name}.mp4")["scores"]
            quality_s = (time.perf_counter() - t0) / 2
            compression[name] = {"full_frame_lpips": f["full_frame_lpips"], "outside_lpips": f["outside_lpips"], "masked_lpips": f["masked_lpips"],
                                 "quality_lossless": q_lossless, "quality_mp4": q_mp4,
                                 "quality_difference": {d: q_mp4[d] - q_lossless[d] for d in QUALITY_DIMENSIONS}, "quality_seconds_mac": quality_s}
    out["encoding"] = clips
    out["compression"] = compression
    out["known_answers"] = known_answers(qm)
    out["determinism"] = determinism()
    out["metric_canary"] = metric_canary_on_the_mac()
    return out


def known_answers(qm) -> dict:
    from density.metrics.registered import QUALITY_DIMENSIONS

    def synthetic(t, seed, h=240, w=416):
        rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[0:h, 0:w] / 40.0
        image = np.stack([np.sin(xx * rng.uniform(0.5, 2) + rng.uniform(0, 6)) + np.cos(yy * rng.uniform(0.5, 2)) for _ in range(3)], -1)
        image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        return np.repeat(image[None], t, axis=0)

    still = synthetic(9, 0)
    s = qm.score("still.mp4", still)["scores"]
    a, b = synthetic(1, 1)[0], synthetic(1, 2)[0]
    alt = np.stack([a if i % 2 == 0 else b for i in range(9)])
    m = qm.score("alternating.mp4", alt)["scores"]["motion_smoothness"]
    expected = (255.0 - np.abs(a.astype(np.float64) - b.astype(np.float64)).mean()) / 255.0 * 100.0
    return {"still": {d: s[d] for d in QUALITY_DIMENSIONS}, "still_expected": "subject and background consistency 100, motion smoothness about 100",
            "alternating_motion_smoothness": m, "alternating_expected": expected, "alternating_error": m - expected}


def score_once() -> dict:
    """One process's scores of fixed inputs: the face clip's first 17 frames, as MP4 against lossless."""
    from density import determinism as det
    from density import video
    from density.masks.clips import load_frames
    from density.masks.measurement import measurement_masks
    from density.metrics import detection, fidelity, quality

    det.configure(strict=True)
    frames = load_frames("face")[:17]
    with tempfile.TemporaryDirectory() as tmp:
        video.encode(frames, Path(tmp) / "face.mp4")
        mp4 = video.decode_rgb(Path(tmp) / "face.mp4")
        q = quality.QualityModels("cpu").score(Path(tmp) / "face.mp4")
    roi = stock_roi("face")[:17]
    f = fidelity.fidelity(fidelity.LPIPS("cpu"), mp4, frames, measurement_masks(roi))
    d = detection.Detectors("cpu")
    ref = detection.detect_video(d, frames, "face")
    return {"quality": q, "fidelity": f, "detection": detection.compare(ref, detection.detect_video(d, mp4, "face"), "face")}


def metric_canary_on_the_mac() -> dict:
    """The metric canary's own mechanism, on the Mac's CPU: record from two fresh processes, then check from a third."""
    from density import metric_canary

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        record = metric_canary.run("mac", "cpu", tmp / "none.json", tmp)
        out = {"record": {k: record.get(k) for k in ("passed", "recorded", "digest", "seconds", "stop")}}
        if record["passed"]:
            (tmp / "registered.json").write_text(json.dumps({"mac": json.loads((tmp / "registered_mac.json").read_text())}))
            check = metric_canary.run("mac", "cpu", tmp / "registered.json", tmp)
            out["check"] = {k: check.get(k) for k in ("passed", "recorded", "digest", "seconds", "stop")}
        else:
            out["check"] = {"passed": False, "seconds": 0.0, "stop": "not run: the record failed"}
    return out


def determinism() -> dict:
    """The same inputs scored in two separate processes, compared exactly."""
    runs = []
    for _ in range(2):
        r = subprocess.run([sys.executable, "-m", "density.checks.metrics", "determinism-child"], cwd=ROOT, capture_output=True, text=True)
        line = next((l for l in r.stdout.splitlines() if l.startswith("{")), None)
        if r.returncode or line is None:
            return {"passed": False, "error": r.stderr[-2000:]}
        runs.append(json.loads(line))
    return {"passed": runs[0] == runs[1], "processes": 2, "scores": runs[0], "device": "cpu (the Mac); CUDA is first checked on the pods"}


# Parts 3-7


def cost_tables() -> dict:
    """Section 12's cost table for a full pilot and main run, from the registered grid and the measured times."""
    from density.run import freeze as F
    from density.run import manifest as M

    rows, pid = [], 1
    for stratum, block, n in (("face", "holdout", 6), ("manipulation", "holdout", 6), ("face", "primary", 36), ("manipulation", "primary", 36),
                              ("body", "secondary", 16), ("control", "control", 16)):
        for _ in range(n):
            rows.append({"id": str(pid), "stratum": stratum, "block": block})
            pid += 1
    pilot = M.pilot_initial(rows)
    pilot += F.condition_jobs(None, {"kind": "pilot", "setting": "sh8g6"}, [j for j in pilot if j.kind == "reference" and j.setting == "sh8g6"])
    main = M.main_initial(rows, "sh8g6")
    refs = [j for j in main if j.kind == "reference"]
    blocks = {b: {} for b in ("primary", "secondary", "control")}
    for r in rows:
        if r["block"] in blocks:
            blocks[r["block"]].setdefault(r["stratum"], []).append(int(r["id"]))
    main += F.condition_jobs(None, {"kind": "main", "setting": "sh8g6", "lambda": 3.0, "blocks": blocks, "subsets": F.subsets(blocks["primary"])}, refs)
    return {"pilot": M.planned_cost(pilot), "main": M.planned_cost(main)}


def mock_runs() -> dict:
    from density import registration
    from density.checks import dryrun
    from density.run.store import DRY_RUN_ROOT

    pilot = dryrun.mock_pilot("pilot")
    reg_dir = DRY_RUN_ROOT / "registration"
    reg = registration.build(reg_dir, Path(pilot["run"]), draft=True)
    main = dryrun.mock_main("main", reg_dir / "registration.json") if reg["values"]["lambda"] is not None else dryrun.mock_main("main")
    return {"pilot": pilot, "registration": {k: reg[k] for k in ("values", "pilot")} | {"entries": len(reg["registered"])}, "main": main,
            "cost_tables": cost_tables()}


def mock_main_analysis() -> dict:
    """A 72-prompt main run's records, synthetic and in the real format, through the analysis CLI's own path."""
    from density import analysis
    from density.checks.analysis_validation import synthetic_rows, validations

    folder = synthetic_main_run()
    out_path = OUT_DIR / "mock_main_analysis_result.json"
    code = analysis.main(["run", str(folder), "--out", str(out_path)])
    result = json.loads(out_path.read_text())
    return {"exit": code, "outcome": result["outcome"], "reason": result["reason"], "median_delta": result.get("primary", {}).get("median_delta"),
            "ci95": result.get("primary", {}).get("ci95"), "guardrails": result.get("guardrail_statuses"),
            "secondary": {f: {k: {"median_delta": v["median_delta"], "holm_p": v["holm_p"]} for k, v in fam.items()} for f, fam in result.get("secondary", {}).items()},
            "controls_check": result.get("controls_check"), "validations": validations(replicates=50)}


def synthetic_main_run() -> Path:
    """Records for 36 + 36 primary prompts (two seeds), 16 body and 16 control prompts, written as the runner writes them."""
    from density import prompts as prompt_set
    from density.checks.analysis_validation import synthetic_rows
    from density.run import freeze as F
    from density.run import manifest as M
    from density.run.store import DRY_RUN_ROOT, Run, write_json_atomic
    import shutil

    folder = DRY_RUN_ROOT / "mock_main_analysis"
    shutil.rmtree(folder, ignore_errors=True)
    run = Run(folder)
    rows = []
    for p in range(1, 73):
        rows.append({"id": str(p), "stratum": "face" if p <= 36 else "manipulation", "block": "primary"})
    rows += [{"id": str(p), "stratum": "body", "block": "secondary"} for p in range(101, 117)] + [{"id": str(p), "stratum": "control", "block": "control"} for p in range(201, 217)]
    write_json_atomic(run.path("inputs", "prompts.json"), {"rows": rows})
    jobs = M.main_initial(rows, "sh8g6")
    refs = [j for j in jobs if j.kind == "reference"]
    blocks = {"primary": {"face": list(range(1, 37)), "manipulation": list(range(37, 73))}, "secondary": {"body": list(range(101, 117))}, "control": {"control": list(range(201, 217))}}
    jobs += F.condition_jobs(run, {"kind": "main", "setting": "sh8g6", "lambda": 3.0, "blocks": blocks, "subsets": F.subsets(blocks["primary"])}, refs)
    run.write_manifest({"run": folder.name, "kind": "main", "dry_run": True, "prompts_csv_sha256": "synthetic", "jobs": [j.to_dict() for j in jobs]}, "synthetic records for the analysis")
    table, _ = synthetic_rows(0.15, seed=11)
    by = {(r["prompt"], r["seed"], r["condition"]): r for r in table}
    rng = np.random.default_rng(5)
    quality_by_job = {}
    for j in jobs:
        if j.kind != "condition":
            continue
        s = prompt_set.seeds(j.prompt_id).index(j.seed) + 1 if j.prompt_id < 100 else 1
        base = by.get((j.prompt_id, s, "C3" if j.condition in ("C2", "C3") else "C4")) or by[((j.prompt_id - 100) % 72 + 1, 1, "C3")]
        scale = {"C2": 1.3, "C5": 0.95, "C6": 0.97, "A1": 1.0}.get(j.condition, 1.0) * (1.0 if j.preset in (None, "HSA-50") else 0.8)
        fid = {"masked_lpips": base["masked_lpips"] * scale * rng.lognormal(0, 0.03), "full_frame_lpips": base["full_frame_lpips"], "outside_lpips": base["outside_lpips"],
               "region_masked_lpips": {"HC": base["masked_lpips"] * scale, "O": base["masked_lpips"] * scale * 1.1} if j.stratum == "manipulation" else None}
        if j.stratum == "control":
            fid = {"masked_lpips": None, "full_frame_lpips": by[((j.prompt_id - 200) % 72 + 1, 1, "C3")]["full_frame_lpips"], "outside_lpips": 0.03}
        write_json_atomic(run.job_dir(j) / "record.json", {"job": j.to_dict(), "status": "done", "dry_run": True, "fidelity": {k: v for k, v in fid.items() if v is not None}})
        quality_by_job[j.id] = {d: base[d] for d in ("subject_consistency", "background_consistency", "motion_smoothness", "aesthetic_quality", "imaging_quality")}
    for j in jobs:
        if j.kind == "score":
            videos = {d: {"quality_input": "lossless frames", "quality": quality_by_job[d]} for d in j.depends if d.startswith("gen-")}
            write_json_atomic(run.job_dir(j) / "record.json", {"job": j.to_dict(), "status": "done", "dry_run": True, "videos": videos})
    return folder


# The report

VOLUME_NOW_GB = 57.7  # du's allocated size of the density volume on 2026-10-07 (density/volume.py)
VOLUME_GB = 125  # Chase increased the volume from 100 GB on 2026-10-07
STOP_GB = 112  # density.volume.STOP_BYTES
METRIC_WEIGHTS_GB = 2.2
MASKS_PER_REFERENCE_BYTES = 2.6e6  # the dry run's mask folders (masks, coverage, metadata, sheet): 2.0-2.8 MB
NOISE_BYTES = 16 * 21 * 60 * 104 * 4  # one seed's starting noise, fp32
EMBEDDING_BYTES = 2 * 512 * 4096 * 2  # a prompt's and the negative prompt's T5 embeddings, bf16


def storage_projection(encoding: dict) -> dict:
    """Bytes the pilot and the main run add to the volume, from the stock clips' measured sizes (section 12). Registered
    (Chase, 2026-10-07): references keep PNG frames and every condition an FFV1 copy, besides the playback MP4s."""
    mean = lambda f: float(np.mean([f(v) for v in encoding.values()]))
    png, mp4, ffv1 = mean(lambda v: v["png_bytes"]), mean(lambda v: v["mp4_bytes"]), mean(lambda v: v["ffv1"]["bytes"])
    reference = png + mp4 + MASKS_PER_REFERENCE_BYTES
    condition = ffv1 + mp4
    pilot = 24 * reference + 60 * condition + 12 * (NOISE_BYTES + EMBEDDING_BYTES)
    refs, conditions, prompts = 176, 628, 72 + 16 + 16 + 60  # main references, reduced-budget videos, embedded prompts (reserves included)
    main = refs * (reference + NOISE_BYTES) + conditions * condition + prompts * EMBEDDING_BYTES
    after = VOLUME_NOW_GB + METRIC_WEIGHTS_GB + (pilot + main) / 1e9
    return {"per_video_mean_bytes": {"png": png, "mp4": mp4, "ffv1": ffv1}, "pilot_gb": pilot / 1e9, "main_gb": main / 1e9,
            "volume_after_main_gb": after, "volume_gb": VOLUME_GB, "stop_gb": STOP_GB,
            "assumptions": f"volume now {VOLUME_NOW_GB} GB; metric weights {METRIC_WEIGHTS_GB} GB; stock-clip sizes stand in for Wan's frames; no replacements"}


def report() -> dict:
    parts = {p.stem: json.loads(p.read_text()) for p in sorted(OUT_DIR.glob("*.json")) if p.stem != "mock_main_analysis_result"}
    data = {"code": euler.git_state(), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "parts": parts}
    from density.run.store import write_json_atomic

    write_json_atomic(REPORT_JSON, data)
    REPORT_MD.write_text(markdown(data))
    return data


def fmt(x, digits=4):
    return "n/a" if x is None else f"{x:.{digits}f}" if isinstance(x, float) else str(x)


def markdown(data: dict) -> str:
    P = data["parts"]
    L = ["# Milestone 5 check: metrics, prompt set, runner and the frozen analysis", "",
         f"Run on the Mac at commit `{data['code']['git_sha']}` (dirty: {data['code']['git_dirty']}), {data['utc']}. "
         "Timings are the Mac's. Every dry-run number describes stand-ins (stock clips with a seeded disturbance, tiny random "
         "models), not Wan; dry-run reviews were automatic and are not Chase's.", ""]
    if "prompts" in P:
        p = P["prompts"]
        L += ["## Part 1: prompt tooling", "", "| List | Prompts | Registered | Commit |", "| --- | --- | --- | --- |"]
        L += [f"| {k} | {v['prompts']} | {v['registered']} | `{v['commit'][:7]}` |" for k, v in p["sources"].items()]
        L += ["", f"Validator on the mock prompt files: pilot {len(p['mock pilot']['errors'])} errors, main {len(p['mock main']['errors'])} errors "
              f"(longest {p['mock pilot']['max_tokens']} T5 tokens).", ""]
        c = p["configs/prompts.csv"]
        L += [f"`configs/prompts.csv`: {c['rows']} rows, {c['note']}. The pilot's check (holdout rows): "
              f"{'passes' if not c['holdout_errors'] else '; '.join(c['holdout_errors'])}. The full check: "
              f"{'passes' if not c['all_errors'] else '; '.join(c['all_errors'])}.", ""]
    if "metrics" in P:
        m = P["metrics"]
        L += ["## Part 2: metrics", "", f"Weight files verified: {sum(w['verified'] for w in m['weights'].values())} of {len(m['weights'])}. "
              f"Encode canary: {'passed' if m['encode_canary']['passed'] else 'FAILED'} (registered on {m['encode_canary']['registered_on']}).", "",
              "### Encoding and storage per video (81 frames, 832 x 480)", "", "| Clip | MP4 PSNR vs lossless (dB) | MP4 | FFV1 (lossless) | PNG frames | Same decoded frames on re-encode |",
              "| --- | --- | --- | --- | --- | --- |"]
        L += [f"| {k} | {v['psnr_vs_lossless_db']:.2f} | {v['mp4_bytes'] / 1e6:.2f} MB | {v['ffv1']['bytes'] / 1e6:.2f} MB{'' if v['ffv1']['exact'] else ' (NOT exact)'} | {v['png_bytes'] / 1e6:.2f} MB | {v['decoded_identical_across_encodes']} |" for k, v in m["encoding"].items()]
        L += ["", "### Compression in the guardrails' units (MP4 against its lossless frames)", "",
              "| Clip | Full-frame LPIPS | Outside-ROI LPIPS | Masked LPIPS | Largest quality change (points) |", "| --- | --- | --- | --- | --- |"]
        for k, v in m["compression"].items():
            worst = max(v["quality_difference"].items(), key=lambda kv: abs(kv[1]))
            L.append(f"| {k} | {fmt(v['full_frame_lpips'])} | {fmt(v['outside_lpips'])} | {fmt(v['masked_lpips'])} | {worst[1]:+.3f} ({worst[0]}) |")
        L += ["", "Limits: 0.01 LPIPS and 0.5 points. In absolute terms the CRF 10 encoding is not negligible next to them, so every "
              "guardrail is measured before compression (registered by Chase on 2026-10-07): LPIPS inside each generation job and the "
              "quality dimensions on the lossless frames. The MP4's quality scores are recorded beside them as a descriptive check.", ""]
        sp = storage_projection(m["encoding"])
        L += ["### Storage", "", f"Mean per video: PNG frames {sp['per_video_mean_bytes']['png'] / 1e6:.1f} MB, FFV1 {sp['per_video_mean_bytes']['ffv1'] / 1e6:.1f} MB, "
              f"MP4 {sp['per_video_mean_bytes']['mp4'] / 1e6:.1f} MB. With references in PNG and every condition in FFV1 (registered), the pilot adds "
              f"about {sp['pilot_gb']:.1f} GB and the main run {sp['main_gb']:.1f} GB, so the volume would hold about {sp['volume_after_main_gb']:.1f} GB "
              f"after the main run (du allocated; {sp['volume_gb']} GB volume, {sp['stop_gb']} GB stop; {sp['assumptions']}).", ""]
        if "metric_canary" in m:
            mc = m["metric_canary"]
            L += ["### Metric canary (on the Mac's CPU)", "", f"Recorded from two fresh processes: {mc['record']['passed']} ({mc['record']['seconds']:.0f} s); "
                  f"checked from a third: {mc['check']['passed']} ({mc['check']['seconds']:.0f} s). The pods record their own at their first session.", ""]
        L += ["### Known answers and determinism", ""]
        ka = m["known_answers"]
        L += [f"- Still video: " + ", ".join(f"{d} {v:.4f}" for d, v in ka["still"].items()) + ".",
              f"- Alternating frames: motion smoothness {ka['alternating_motion_smoothness']:.3f} against the expected {ka['alternating_expected']:.3f}.",
              f"- Two separate processes, identical scores (quality, LPIPS, pixel and detector checks): {m['determinism']['passed']} ({m['determinism']['device']}).", ""]
    if "runs" in P:
        r = P["runs"]
        L += ["## Parts 3, 4 and 7: the mock pilot, its rules and registration, and a small mock main run", ""]
        pil = r["pilot"]
        L += [f"- Mock pilot: {pil['status']}. Setting {pil['setting']['setting']} (tie: {pil['setting']['tie']}; every mock reference of a stratum is the same clip, so the rule only takes its tie path here; the other path is unit-tested).",
              f"- Pilot rules: lambda {pil['rules']['lambda']['lambda']}, C2 ratio {pil['rules']['c2']['ratio']:.3f}, no-op at every lambda {pil['rules']['no_op']['unchanged_at_every_lambda']}, "
              f"excluded holdout prompts {list(pil['rules']['excluded'])}; stops: {pil['rules']['stops'] or 'none'} (artifacts of the stand-in disturbance, not findings).",
              "- The encoding's effect on the paired quality degradation, on stand-ins (mean over prompts, MP4 minus lossless): "
              + (", ".join(f"{d} {v['mean_difference']:+.3f}" for d, v in pil["rules"]["compression"]["dimensions"].items()) if pil["rules"].get("compression") else "not measured") + ".",
              f"- Registration draft: {r['registration']['entries']} registered values; pilot slots {r['registration']['pilot']['status']}.",
              f"- Mock main run: {r['main']['status']}; grid blocks {r['main']['grid_blocks']}; collected {r['main']['collected']['primary_rows']} primary rows and blocks {r['main']['collected']['secondary_blocks']}.", ""]
        for name, table in r["cost_tables"].items():
            L += [f"### Planned cost: {name}", "", "| Block | Jobs | RE | Machine | Hours | Dollars |", "| --- | --- | --- | --- | --- | --- |"]
            L += [f"| {b} | {v['jobs']} | {v['re']:g} | {v['machine']} | {v['hours']:.2f} | {v['dollars']:.2f} |" for b, v in table["blocks"].items()]
            L += ["", "Totals: " + "; ".join(f"{m} {t['re']:g} RE, {t['hours']:.1f} h, ${t['dollars']:.2f}" for m, t in table["totals"].items())
                  + f". Rates {table['rates_per_hour']} per hour; times from {table['seconds_source']}. Not included: {table['not_included']}.", ""]
    if "analysis" in P:
        a = P["analysis"]
        v = a["validations"]
        L += ["## Part 5: the frozen analysis", "",
              f"- Mock 72-prompt main analysis through the CLI (freeze checked, records collected): {a['outcome']} ({a['reason']}); median {fmt(a['median_delta'])}, 95% interval {a['ci95']}.",
              f"- 15 percent improvement: {v['improvement_15_percent']['outcome']} (expected GO).",
              f"- No effect, {v['no_effect']['replicates']} replicates: {v['no_effect']['outcomes']} (never GO: {v['no_effect']['passed']}).",
              f"- Injected harm: " + ", ".join(f"{g} +{c['injected']} gives {c['outcome']} ({c['status']})" for g, c in v["injected_harm"]["cases"].items()) + ".",
              f"- Missing score: {v['missing_metric']['outcome']}.", ""]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("parts", nargs="+", choices=["all", "prompts", "metrics", "determinism-child", "runs", "analysis", "report"])
    args = parser.parse_args(argv)
    if args.parts == ["determinism-child"]:
        print(json.dumps(score_once(), sort_keys=True, default=float))
        return 0
    parts = ["prompts", "metrics", "runs", "analysis", "report"] if "all" in args.parts else args.parts
    for part in parts:
        start = time.perf_counter()
        if part == "prompts":
            save("prompts", check_prompts())
        elif part == "metrics":
            save("metrics", check_metrics())
        elif part == "runs":
            save("runs", mock_runs())
        elif part == "analysis":
            save("analysis", mock_main_analysis())
        elif part == "report":
            report()
        print(f"[metrics check] {part} done in {time.perf_counter() - start:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
