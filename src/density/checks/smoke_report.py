"""The GPU smoke-test report: ``results/checks/smoke_test.{json,md,sha256}`` from the step records.

Run on the Mac after the pod results are synced back: ``python -m density.checks.smoke report``.
``results/checks/smoke_test/pods.json`` is written on the Mac from the RunPod MCP's responses: each pod's
id, session, GPU, hourly price, and creation and deletion times in UTC, plus the billing records.

Tolerances are proposed here, never applied. Projections use the protocol's compute table (section 12)
in two ways: reference equivalents times the measured RE time, and run counts times each condition's
measured time.
"""

import json
from datetime import datetime
from pathlib import Path

from density import tolerances as registered_tolerances
from density.checks import smoke
from density.checks.hsa import REGISTERED

# Protocol section 12: (block, phase, runs per condition, planning RE)
PROTOCOL_BLOCKS = [
    ("Pilot settings", "pilot", {"C1": 24}, 24),
    ("Pilot lambda", "pilot", {"C3": 12, "C4": 36}, 24),
    ("Pilot HSA reproduction", "pilot", {"C2": 12}, 6),  # Chase, 2026-10-07: C2 on the 12 holdout prompts
    ("Primary seed 1", "main", {"C1": 72, "C2": 72, "C3": 72, "C4": 72}, 180),
    ("Primary seed 2", "main", {"C1": 72, "C3": 72, "C4": 72}, 144),
    ("C5 and C6", "main", {"C5": 36, "C6": 36}, 36),
    ("A1", "main", {"A1": 36}, 18),
    ("Full body", "main", {"C1": 16, "C3": 16, "C4": 16}, 32),
    ("Controls", "main", {"C1": 16, "C3": 16, "C4": 16}, 32),
    ("75 percent budget", "main", {"C3 HSA-75B": 16, "C4 HSA-75B": 16}, 24),
    ("Switch points", "main", {"C3 switch": 32, "C4 switch": 32}, 32),
]
RE_PER_RUN = {"C1": 1.0, "C2": 0.5, "C3": 0.5, "C4": 0.5, "C5": 0.5, "C6": 0.5, "A1": 0.5, "C3 HSA-75B": 0.75, "C4 HSA-75B": 0.75, "C3 switch": 0.5, "C4 switch": 0.5}
# References that need masks: the pilot's 6 face and 6 manipulation prompts at 2 settings, the main run's 36 + 36 prompts at 2 seeds, and seed 1 of body and controls.
MASK_REFERENCES = {"pilot": {"face": 12, "manipulation": 12}, "main": {"face": 72, "manipulation": 72, "body": 16, "control": 16}}
LPIPS_NOTE = (
    "Not measured yet: no LPIPS model is pinned. The fast-path and oracle frames for C3 and C4 are saved losslessly in "
    "smoke_test/t3_frames/. In Milestone 5, compute their LPIPS and compare it with the improvement the experiment seeks "
    "(protocol sections 9 and 11): GO needs a median relative improvement of at least 10 percent of HSA's loss, and the "
    "fidelity guardrails allow at most 0.01 LPIPS degradation."
)
LPIPS_EXACT_NOTE = (
    "With flash attention registered, the fast path's frames equal the oracle's exactly for C3 and C4 (Session D), so their "
    "LPIPS is zero by construction. Session B's frames under cuDNN, which differ, stay saved losslessly in smoke_test/t3_frames/."
)
SAME_COUNTS_NOTE = (
    "C3 and C4 run identical active-token counts at every iteration, layer and guidance branch, so every kernel receives the same "
    "row counts in both conditions and any row-count effect applies to both in the same way. At full resolution on the H100 that "
    "effect comes from cuDNN attention (the GEMMs were row-invariant); in the tiny-model rehearsal on the slice it came from cuBLAS."
)


def parse_utc(text: str | None) -> datetime | None:
    return datetime.fromisoformat(text.replace("Z", "+00:00")) if text else None


def load(ctx) -> tuple[dict, dict, dict | None, dict]:
    """Step records, step errors, the pod log, and diagnostics (diag_*.json: follow-up runs outside the step list)."""
    records, errors, diagnostics = {}, {}, {}
    for path in sorted(ctx.out.glob("*.json")):
        if path.name == "pods.json":
            continue
        data = json.loads(path.read_text())
        if path.stem.startswith("diag_"):
            diagnostics[path.stem.removeprefix("diag_")] = data
        else:
            (errors if path.stem.endswith(".error") else records)[path.stem.removesuffix(".error")] = data
    pods_path = ctx.path("pods.json")
    return records, errors, json.loads(pods_path.read_text()) if pods_path.exists() else None, diagnostics


def result(records: dict, name: str) -> dict | None:
    return records[name]["result"] if name in records else None


# Sections


def pod_summary(pods: dict | None) -> dict | None:
    if not pods:
        return None
    rows, by_gpu = [], {}
    for pod in pods["pods"]:
        created, deleted = parse_utc(pod.get("created_utc")), parse_utc(pod.get("deleted_utc"))
        hours = (deleted - created).total_seconds() / 3600 if created and deleted else None
        cost = hours * pod["price_per_hour"] if hours is not None else None
        rows.append({**pod, "hours": hours, "cost_usd": cost})
        g = by_gpu.setdefault(pod["gpu"], {"pods": 0, "hours": 0.0, "cost_usd": 0.0, "price_per_hour": pod["price_per_hour"]})
        g["pods"] += 1
        g["hours"] += hours or 0.0
        g["cost_usd"] += cost or 0.0
    return {"pods": rows, "by_gpu": by_gpu, "total_cost_usd": sum(g["cost_usd"] for g in by_gpu.values()), "billing": pods.get("billing"), "volume": pods.get("volume"),
            "note": "Hours come from the creation and deletion times logged on the Mac"}


def cgroup_cpus(system: dict) -> float | None:
    """The pod's CPU quota from cgroup cpu.max ("quota period"); None when unlimited or unknown."""
    quota = (system.get("cgroup_cpu_max") or "max").split()
    return int(quota[0]) / int(quota[1]) if quota[0] != "max" else None


def environment_summary(env: dict) -> dict:
    e, n, t, s = env["environment"], env["nvidia"], env["torch_cuda"], env["system"]
    device = t["devices"][0] if t.get("available") else {}
    toolkit = next((line for line in (env["nvcc"]["toolkit"] or "").splitlines() if "release" in line), None)
    return {
        "utc": env["utc"], "hostname": env["hostname"], "pod": env["pod_env"],
        "gpu": device.get("name"), "capability": device.get("capability"), "gpu_memory_gib": device.get("total_memory_gib"),
        "nvidia_smi_query": n["query"], "driver_cuda_version": n["driver_cuda_version"],
        "torch": e["torch"], "torch_cuda": t.get("torch_cuda"), "cudnn": t.get("cudnn"), "arch_list": t.get("arch_list"), "nvcc_toolkit": toolkit,
        "python": e["python"], "diffusers": e["diffusers"], "os": s["os"], "kernel": s["kernel"], "cpu": s["cpu_model"],
        "cpus": cgroup_cpus(s) or s["cpu_affinity"] or s["cpu_count"], "host_cpus": s["cpu_count"], "memory_gib": s["memory_gib"],
        "memory_limit_gib": int(s["cgroup_memory_max"]) / 2**30 if (s.get("cgroup_memory_max") or "max").isdigit() else None,
        "git_sha": e["git_sha"], "git_dirty": e["git_dirty"],
        "determinism": env["determinism"], "detectron2_archs": env["detectron2"]["cuobjdump_archs"],
        "sam2_cuda_extension": env["stack"]["sam2"]["cuda_extension"], "onnxruntime_providers": env["stack"]["onnxruntime"]["providers"],
        "lock_differences": env["lock_differences"], "credentials_present": env["credentials_present"], "disk": env["disk"], "bootstrap": env["bootstrap"],
    }


def hashes(records: dict, records_d: dict | None = None) -> dict:
    out = {}
    weights = result(records, "weights_A")
    if weights and weights.get("wan"):
        out["wan"] = {"revision": weights["wan"]["revision"], "files": weights["wan"]["files"]}
        out["mask_weights"] = weights["masks"]
        out["wan_real_inputs"] = weights["wan_real_inputs"]
    noise, mac = result(records, "noise"), result(records, "noise_mac")
    if noise:
        out["starting_noise"] = noise["prompts"]
        if mac:
            out["starting_noise_cross_device"] = {
                pid: {"pod": rec["tensor_sha256"], "mac": mac["prompts"][pid]["tensor_sha256"], "identical": rec["tensor_sha256"] == mac["prompts"][pid]["tensor_sha256"]}
                for pid, rec in noise["prompts"].items()
            }
            out["starting_noise_cross_device_note"] = f"pod {noise['machine']} vs Mac {mac['machine']}: reported, not judged; every run loads the pod's saved noise"
    encoded = result(records, "encode")
    if encoded:
        out["embeddings"] = {"file": encoded["file"], "file_sha256": encoded["file_sha256"], **{k: {x: v[x] for x in ("sha256", "tokens", "shape", "dtype")} for k, v in encoded["embeddings"].items()}}
    generated = result(records, "generate")
    if generated:
        out["references"] = {pid: {"uint8_sha256": r["reference"]["uint8_sha256"], "final_latents_sha256": r["final_latents_sha256"]} for pid, r in generated["prompts"].items()}
    generated_d = result(records_d or {}, "generate")
    if generated_d:
        out["references_flash"] = {pid: {"uint8_sha256": r["reference"]["uint8_sha256"], "final_latents_sha256": r["final_latents_sha256"]} for pid, r in generated_d["prompts"].items()}
        out["references_note"] = "references: Session B, PyTorch's default attention (cuDNN); references_flash: Session D, flash attention registered"
    out["pod_installs"] = smoke.INSTALLS
    out["starting_noise_rule"] = "registered (density.noise): generated once, saved, identified by its SHA-256, always loaded from the saved file, never regenerated"
    return out


def steady_state(perf: dict) -> dict:
    """Best-of-two times: the first HSA run in a process pays a one-time cost the later ones do not."""
    best, excess = {}, {}
    for c in ("C1", "C3", "C4"):
        runs = [r for r in perf["runs"] if r["condition"] == c]
        best[c] = {"denoise_seconds": min(r["denoise_seconds"] for r in runs), "end_to_end_seconds": min(r["end_to_end_seconds"] for r in runs)}
        excess[c] = runs[0]["denoise_seconds"] - best[c]["denoise_seconds"]
    d3, d4 = best["C3"]["denoise_seconds"], best["C4"]["denoise_seconds"]
    return {"best_of_two": best, "first_run_excess_seconds": excess, "c4_overhead_vs_c3": {"seconds": d4 - d3, "relative": (d4 - d3) / d3}}


def run_seconds(perf: dict) -> dict:
    s = perf["summary"]
    best = steady_state(perf)["best_of_two"]  # steady state: the pilot's runs share long-lived processes
    c1, c3, c4 = (best[c]["end_to_end_seconds"] for c in ("C1", "C3", "C4"))
    d1, d3, d4 = (best[c]["denoise_seconds"] for c in ("C1", "C3", "C4"))
    decode = s["C1"]["decode_seconds_mean"]
    f3 = s["C3"]["token_step_fraction"]
    hsa75 = d3 + (d1 - d3) * (REGISTERED["HSA-75B"][1] - f3) / (1 - f3)
    same_c4 = "assumed equal to C4: the same HSA machinery with a different ranking rule"
    return {
        "C1": (c1, "measured"),
        "C2": (d1 / 2 + decode, "estimated: half of C1's denoising time plus one decode"),
        "C3": (c3, "measured"),
        "C4": (c4, "measured"),
        "C5": (c4, same_c4), "C6": (c4, same_c4), "A1": (c4, same_c4),
        "C3 HSA-75B": (hsa75 + decode, "estimated: denoising interpolated linearly in token-step fraction between C3 and C1"),
        "C4 HSA-75B": (hsa75 + (d4 - d3) + decode, "estimated as C3 HSA-75B plus C4's measured overhead"),
        "C3 switch": (c3, "assumed equal to C3 (token-step fraction 0.5)"),
        "C4 switch": (c4, "assumed equal to C4 (token-step fraction 0.5)"),
    }


def mask_seconds(records: dict, ctx, diagnostics: dict | None = None) -> dict:
    """Seconds per reference: the registered configuration (thread cap and ROIAlign kernel, Session E's follow-up) when
    it has run, else Session C's; control from Session C's stock landscape clip (the explicit empty mask)."""
    masks = result(records, "masks")
    out = {}
    kernel = (((diagnostics or {}).get("mask_roi_align") or {}).get("pipeline") or {}).get("references") or {}
    if masks and not kernel:
        refs = masks["references"]
        out["face"] = (refs["smoke-1"]["seconds"]["total"], "smoke-1 reference on the slice")
        out["manipulation"] = (refs["smoke-2"]["seconds"]["total"], "smoke-2 reference on the slice")
    path = ctx.path("mask_check", "mask_pipeline.json")
    if path.exists():
        clips = json.loads(path.read_text()).get("clip_results") or {}
        if "body" in clips and not kernel:
            sec = clips["body"]["seconds"]
            out["body"] = (sec.get("B.detector", 0) + sec.get("B.sam", 0) + sec["roi_coverage_measurement_eligibility"], "stock kettlebell clip's body path on the slice")
        if "landscape" in clips:
            out["control"] = (clips["landscape"]["seconds"]["total"], "stock landscape clip on the slice: the explicit empty mask")
    basis = "on the slice with the registered configuration: CPU pools at the quota, the ROIAlign kernel (Session E's follow-up, best of two)"
    for stratum, rid in STRATUM_REFERENCE.items():
        if rid in kernel:
            out[stratum] = (kernel[rid]["wall_seconds"], f"{rid if rid != 'body' else 'the stock body clip'} {basis}")
    return out


def projections(records: dict, pods: dict | None, ctx, perf: dict | None = None, diagnostics: dict | None = None) -> dict | None:
    perf = perf or result(records, "perf")
    if not perf:
        return None
    prices = {p["session"]: p["price_per_hour"] for p in (pods or {}).get("pods", [])}
    h100 = prices.get("D") or prices.get("B")
    slice_price = prices.get("C") or prices.get("A")
    re_seconds = perf["re_seconds"]
    seconds = run_seconds(perf)
    out = {"steady_state": steady_state(perf), "re_seconds": re_seconds, "re_definition": perf["re_definition"], "run_seconds": {k: {"seconds": v, "basis": b} for k, (v, b) in seconds.items()}, "h100_price_per_hour": h100, "blocks": [], "phases": {}}
    for block, phase, runs, planning_re in PROTOCOL_BLOCKS:
        re_check = sum(n * RE_PER_RUN[c] for c, n in runs.items())
        by_re = planning_re * re_seconds / 3600
        by_condition = sum(n * seconds[c][0] for c, n in runs.items()) / 3600
        out["blocks"].append({"block": block, "phase": phase, "runs": runs, "planning_re": planning_re, "re_from_runs": re_check, "gpu_hours_by_re": by_re, "gpu_hours_by_condition": by_condition})
    for phase in ("pilot", "main"):
        blocks = [b for b in out["blocks"] if b["phase"] == phase]
        hours_re, hours_cond = sum(b["gpu_hours_by_re"] for b in blocks), sum(b["gpu_hours_by_condition"] for b in blocks)
        out["phases"][phase] = {
            "planning_re": sum(b["planning_re"] for b in blocks),
            "gpu_hours_by_re": hours_re, "usd_by_re": hours_re * h100 if h100 else None,
            "gpu_hours_by_condition": hours_cond, "usd_by_condition": hours_cond * h100 if h100 else None,
        }
    per_reference = mask_seconds(records, ctx, diagnostics)
    out["masks"] = {"seconds_per_reference": {k: {"seconds": v, "basis": b} for k, (v, b) in per_reference.items()}, "slice_price_per_hour": slice_price, "phases": {}}
    masks = result(records, "masks")
    build = sum(masks["references"]["smoke-1"]["model_build_seconds"].values()) + sum(masks["references"]["smoke-2"]["model_build_seconds"].values()) if masks else 0.0
    for phase, counts in MASK_REFERENCES.items():
        if all(k in per_reference for k in counts):
            hours = (sum(n * per_reference[k][0] for k, n in counts.items()) + build) / 3600
            out["masks"]["phases"][phase] = {"references": counts, "slice_hours": hours, "usd": hours * slice_price if slice_price else None, "includes_model_build_seconds": build}
    out["note"] = "GPU time for generation and masks only; replacements, metrics, debugging and pod setup are extra (protocol section 12, equation 30)"
    return out


def propose(diffs: list[float]) -> dict:
    if all(d == 0 for d in diffs):
        return {"latent_max_abs_diff": 0.0, "rule": "bitwise", "basis": "every observation on this device was bitwise identical"}
    worst = max(diffs)
    return {"latent_max_abs_diff": 2 * worst, "rule": "twice the largest deviation observed", "basis": f"largest observed {worst:.3g}"}


def tolerances(records: dict, scope: str | None = None, registered: bool = False) -> list[dict]:
    """Proposed tolerances; ``scope`` names the configuration the full-resolution rows were measured in, and with
    ``registered`` the rows frozen in density/tolerances.py say so."""
    rows = []
    generated, dense, t3, milestones = (result(records, k) for k in ("generate", "hsa_dense", "t3", "milestones"))
    if generated:
        prompts = generated["prompts"].values()
        diffs = [p["repeat"]["max_abs_diff"] for p in prompts]
        rows.append({
            "test": "Repeated dense run (C1 twice, same device)", "protocol": "decoded PSNR at least 45 dB; record tensor errors and backend determinism",
            "observed": {"bitwise": all(p["repeat"]["bitwise_equal"] for p in prompts), "max_abs_latent_diff": max(diffs), "min_decoded_psnr_db": min(p["repeat_frames"]["decoded_psnr_db"] for p in prompts)},
            "proposal": propose(diffs),
        })
    if dense:
        prompts = dense["prompts"].values()
        diffs = [p["max_abs_diff"] for p in prompts]
        rows.append({
            "test": "All HSA groups dense vs the dense loop", "protocol": "match the same dense Euler path; decoded PSNR at least 45 dB",
            "observed": {"bitwise": all(p["bitwise_equal"] for p in prompts), "max_abs_latent_diff": max(diffs), "min_decoded_psnr_db": min(p["decoded_psnr_db"] for p in prompts)},
            "proposal": propose(diffs),
        })
    if t3:
        for condition in ("C3", "C4"):
            entries = [p["conditions"][condition] for p in t3["prompts"].values()]
            bug = any(e["verdict"].startswith("bug") for e in entries)
            rows.append({
                "test": f"Fast path vs masked-dense oracle at full resolution, {condition} (T3)",
                "protocol": "compare tensor error against the dense-path numerical baseline; check that its induced LPIPS difference is small compared with the improvement the experiment seeks; freeze the measured tolerance before the pilot",
                "observed": {"bitwise": all(e["bitwise_equal"] for e in entries), "min_decoded_psnr_db": min(e["decoded_psnr_db"] for e in entries),
                             "min_decoded_psnr_db_uint8": min((e.get("decoded_psnr_db_uint8", float("nan")) for e in entries), default=None), "max_abs_latent_diff": max(e["max_abs_diff"] for e in entries),
                             "verdicts": [e["verdict"] for e in entries]},
                "proposal": {"rule": "none", "basis": "a difference is unexplained: investigate it before setting any tolerance"} if bug else
                            propose([e["max_abs_diff"] for e in entries]) if all(e["bitwise_equal"] for e in entries) else
                            {"rule": "pending the Milestone 5 LPIPS check", "basis": "dense repeats are bitwise on CUDA, so the deviation is reported as decoded PSNR; LPIPS of the saved frames decides whether it is small against the effect size"},
            })
    if milestones and "m2" in milestones:
        rows.append({"test": "All tokens active in cached attention (T2, small resolution, same device)", "protocol": "match dense attention within the calibrated numerical tolerance; record relative error",
                     "observed": {"measurements": [m for m in milestones["m2"]["measurements"] if m.get("test", "").startswith("T2")], "failures": [m for m in milestones["m2"]["failures"] if m.get("test", "").startswith("T2")]}})
    if milestones and "m3" in milestones:
        for test in ("T10 lambda = 0 equals C3", "T11 empty mask equals C3"):
            rows.append({"test": f"{test} (small resolution, same device)", "protocol": "identical group labels and active-token IDs; output within the baseline numerical tolerance" if test.startswith("T10") else "C4 reproduces C3; any unexplained difference is an implementation failure",
                         "observed": {"non_bitwise_rows": [m for m in milestones["m3"]["measurements"] if m.get("test") == test], "failed_rows": [m for m in milestones["m3"]["failures"] if m.get("test") == test]}})
    for row in rows:
        frozen = registered_tolerances.FROZEN.get(row["test"]) if registered else None
        row["status"] = f"registered ({registered_tolerances.REGISTERED}): {frozen['rule']}" if frozen else "proposed, not applied"
        if frozen:
            row["registered"] = frozen
        if scope and "small resolution" not in row["test"]:
            row["scope"] = scope
    return rows


def warnings_summary(records: dict) -> dict:
    seen, deterministic = {}, {}
    for name, rec in records.items():
        for w in rec.get("warnings", []):
            key = f"{w['category']}: {w['message']}"
            entry = seen.setdefault(key, {"category": w["category"], "message": w["message"], "where": w["where"], "count": 0, "steps": []})
            entry["count"] += w["count"]
            entry["steps"].append(name)
        for w in rec.get("deterministic_warnings", []):
            deterministic.setdefault(f"{w['category']}: {w['message']}", {"message": w["message"], "where": w["where"], "steps": []})["steps"].append(name)
    return {"deterministic_algorithm_warnings": list(deterministic.values()), "all": list(seen.values())}


TF32_RULE = 0.10  # Chase, 2026-10-06: strict fp32 for the VAE and SAM convolutions unless it adds more than about 10 percent per video


def tf32_decision(records: dict) -> dict | None:
    """What IEEE fp32 convolutions cost per video, against Chase's rule; the registration must state the choice either way."""
    vae, perf, masks = (result(records, k) for k in ("vae_precision", "perf", "masks"))
    if not vae:
        return None
    out = {"rule": f"strict fp32 unless it adds more than about {TF32_RULE:.0%} to per-video time", "vae": {}, "masks": {}}
    if perf:
        best = steady_state(perf)["best_of_two"]  # the same steady-state times as the projections
        per_video = {c: best[c]["end_to_end_seconds"] for c in ("C1", "C3", "C4")}
        added = [r["ieee_seconds"] - r["tf32_seconds"] for r in vae["prompts"].values()]
        worst = max(added)
        out["vae"] = {
            "added_seconds_per_video": added, "per_video_seconds": per_video,
            "relative_to": {c: worst / v for c, v in per_video.items()},
            "decoded_psnr_ieee_vs_tf32_db": [r["decoded_psnr_db"] for r in vae["prompts"].values()],
            "within_rule": all(worst / v <= TF32_RULE for v in per_video.values()),
        }
    if masks:
        for pid, r in masks["references"].items():
            e = r.get("ieee_fp32_convolutions")
            if e and "default_seconds_without_build" in e:
                base = e["default_seconds_without_build"]
                out["masks"][pid] = {"stratum": r["stratum"], "default_seconds": base, "ieee_seconds": e["seconds"]["total"], "relative": (e["seconds"]["total"] - base) / base,
                                     "identical_masks": e["identical_to_default"], "within_rule": (e["seconds"]["total"] - base) / base <= TF32_RULE}
    return out


def detectron2_finding(report: dict) -> str | None:
    """The Detectron2 detectors' time on the slice against the Mac's CPU, from the stock body clip."""
    compare = (report.get("session_c") or {}).get("mask_compare") or {}
    pod_report = Path(report.get("_out", "")) / "mask_check" / "mask_pipeline.json" if report.get("_out") else None
    if not pod_report or not pod_report.exists():
        return None
    pod = json.loads(pod_report.read_text())["clip_results"]
    mac = json.loads((Path(report["_out"]).parent / "mask_pipeline.json").read_text())["clip_results"]
    try:
        b_pod, b_mac = pod["body"]["seconds"]["B.detector"], mac["body"]["seconds"]["B.detector"]
        h_pod, h_mac = pod["body"]["seconds"]["O.hands23"], mac["body"]["seconds"]["O.hands23"]
        s_pod, s_mac = pod["body"]["seconds"]["B.sam"], mac["body"]["seconds"]["B.sam"]
    except KeyError:
        return None
    text = (f"the Detectron2 detectors are much slower on the slice than on the Mac's CPU: Mask R-CNN took {b_pod:.0f} s for the body clip against {b_mac:.0f} s, "
            f"and Hands23 {h_pod:.0f} s against {h_mac:.0f} s, while SAM 2.1 ran faster ({s_pod:.0f} s against {s_mac:.0f} s for the body track). "
            + ("Diagnosed in Session E (below): SCRFD's time was onnxruntime's thread pool oversubscribing the CPU quota, and the Detectron2 detectors' was torchvision's deterministic Python roi_align."
               if (report.get("diagnostics") or {}).get("mask_threads") else "Not yet diagnosed."))
    body = pod["body"]["seconds"]
    body_path = body["B.detector"] + body.get("B.sam", 0) + body["roi_coverage_measurement_eligibility"]
    refs = ((report.get("session_c") or {}).get("masks") or {}).get("references") or {}
    face = next((r["seconds"] for r in refs.values() if r["stratum"] == "face"), None)
    manip = next((r["seconds"] for r in refs.values() if r["stratum"] == "manipulation"), None)
    text += f" In the costs above, Mask R-CNN is {body['B.detector']:.0f} s of the body path's {body_path:.0f} s"
    if manip and "O.hands23" in manip:
        text += f" and Hands23 {manip['O.hands23']:.0f} s of the manipulation reference's {manip['total']:.0f} s"
    if face and "F.detector" in face:
        text += f"; SCRFD on onnxruntime's CPU provider is {face['F.detector']:.0f} s of the face reference's {face['total']:.0f} s, and its thread pool is sized to the host, not the pod's CPU quota"
    return text + "."


def exact_t3(t3: dict | None) -> bool:
    return bool(t3) and all(c["bitwise_equal"] for r in t3["prompts"].values() for c in r["conditions"].values())


def attention_finding(diagnostics: dict, perf_cudnn: dict | None = None, perf_flash: dict | None = None) -> str | None:
    xp, hd = diagnostics.get("crossproc") or {}, diagnostics.get("hsa_dense") or {}
    default, flash = xp.get("default"), xp.get("flash-compare")
    if not default or not flash:
        return None
    pinned = hd.get("flash_pinned_smoke_2") or {}
    text = (
        f"PyTorch chose cuDNN attention on the H100 at full resolution. It is bitwise within a process, but a dense C1 rerun in a new process "
        f"differs from step 0 (smoke-2 max latent diff {default['max_abs_diff']:.2f}), and it is not row-invariant. The 'all HSA groups dense' miss "
        f"compared two processes, so its cause is this, not HSA. With flash attention pinned, C1 is bitwise across processes, and HSA all-full vs dense "
        f"({fmt(pinned.get('hsa_all_full_vs_dense', {}).get('bitwise_equal'))}) and C3's fast path vs the oracle ({fmt(pinned.get('c3_fast_vs_oracle', {}).get('bitwise_equal'))}) are bitwise; "
        f"the 40 dense steps took {flash['seconds']:.0f} s with flash against {default['seconds']:.0f} s with cuDNN (wall time after loading, with a copy to the host at every step)."
    )
    if perf_cudnn and perf_flash:
        b, d = steady_state(perf_cudnn)["best_of_two"], steady_state(perf_flash)["best_of_two"]
        text += (" Session D measured flash's cost with CUDA events (end to end in steady state, flash against cuDNN): "
                 + ", ".join(f"{c} {d[c]['end_to_end_seconds']:.1f} s against {b[c]['end_to_end_seconds']:.1f} s ({100 * (d[c]['end_to_end_seconds'] / b[c]['end_to_end_seconds'] - 1):+.0f} percent)" for c in ("C1", "C3", "C4"))
                 + ". The projections use Session D's times.")
    return text


def registrations(tf32: dict | None, d_done: bool = False, perf_flash: dict | None = None) -> list[str]:
    """Settings Chase registered from the smoke test's measurements."""
    from density.attention import GENERATION_BACKENDS
    from density.wan import VAE_CONV_FP32_PRECISION

    out = [
        f"Attention for generation on CUDA: {', then '.join(b.name for b in GENERATION_BACKENDS)} (density/attention.py)" + ("; Session D re-measured with it." if d_done else "; Session D will re-measure with it."),
        "Starting noise: generated once on CPU, saved, and always loaded from the saved file by its SHA-256, never regenerated (density/noise.py); the same seed gives different noise on x86 and ARM CPUs.",
    ]
    vae = (tf32 or {}).get("vae")
    if vae:
        added = max(vae["added_seconds_per_video"])
        cost = f"strict fp32 adds {min(vae['added_seconds_per_video']):.1f} to {added:.1f} s per video, at most {100 * vae['relative_to']['C1']:.0f} percent of C1's and {100 * vae['relative_to']['C3']:.0f} percent of C3's end-to-end time with Session B's times"
        if perf_flash:
            best = steady_state(perf_flash)["best_of_two"]
            cost += f" ({100 * added / best['C1']['end_to_end_seconds']:.0f} and {100 * added / best['C3']['end_to_end_seconds']:.0f} percent with Session D's)"
        out.append(f"VAE convolutions: {VAE_CONV_FP32_PRECISION.upper()}, the CUDA default (density/wan.py): {cost}, over the 10 percent limit.")
    else:
        out.append(f"VAE convolutions: {VAE_CONV_FP32_PRECISION} (density/wan.py).")
    if tf32 and tf32.get("masks"):
        within = all(m["within_rule"] for m in tf32["masks"].values())
        added = ", ".join(f"{100 * m['relative']:+.1f} percent for {pid} ({m['stratum']})" for pid, m in tf32["masks"].items())
        out.append(f"SAM 2.1 convolutions, by the same rule: {'strict fp32 (IEEE)' if within else 'TF32'} (density/masks/registered.py), since strict fp32 {'stays within' if within else 'exceeds'} 10 percent of per-reference mask time: {added}.")
    if d_done:  # Chase's decisions of 2026-10-07, from Sessions D and E
        out += [
            "Tolerances for the four full-resolution checks: bitwise, with Session D's scope (density/tolerances.py); section 11's LPIPS check is met at zero.",
            "The H100 canary: smoke-1's dense C1 must match pod D's latent and frame hashes at the start of every H100 session (density/canary.py).",
            "Strict deterministic mode for the pilot and main run: a deterministic-algorithm warning stops the job (density/determinism.py).",
            "The volume stop: du's allocated size over /workspace before each step or batch, stopping at 90 x 10^9 bytes (density/volume.py).",
        ]
    if roi_align_registered():
        out += [
            "Detectron2's ROIAlign through torchvision's kernel for inference, checked on every call (density/masks/roi_align.py).",
            "The mask canary: every mask model on the stock clips' first 9 frames must hash as recorded in Session F (density/mask_canary.py, configs/mask_canary.json).",
        ]
    return out


def roi_align_registered() -> bool:
    from density.masks import registered

    return hasattr(registered, "DETECTRON2_ROI_ALIGN")


def approved_pending(records: dict, diagnostics: dict | None = None) -> list[str]:
    """Steps Chase approved on 2026-10-07 that wait on the slice diagnosis or on his answers to it."""
    out = []
    if roi_align_registered():
        return out  # decided on 2026-10-07; the protocol's section 12 carries the measured cost
    done = bool((diagnostics or {}).get("mask_threads"))
    if not done:
        out.append("The slice diagnosis (Session E): cap the CPU thread pools to the pod's quota, check whether thread counts change any detection "
                   "or mask, and compare the per-reference mask time with the Mac's (the Mac CPU is proposed instead if the slice stays beyond about 2x).")
    if result(records, "masks"):
        out.append("Replace the protocol's 3 to 4 hour mask allowance with the measured mask cost (section 12)" + (", once the thread and ROIAlign decisions fix the configuration" if done else " after the slice diagnosis")
                   + "; metrics stay a placeholder until Milestone 5.")
    return out


def decisions(records: dict, envs: dict, diagnostics: dict | None = None, tolerances_registered: bool = False) -> list[str]:
    out = []
    if (result(records, "t3") or result(records, "generate")) and not tolerances_registered:
        out.append("Approve or revise the proposed tolerances (none is applied).")
    diag = (diagnostics or {}).get("mask_threads")
    if diag is None or roi_align_registered():
        return out
    bench = diag.get("bench") or {}
    scrfd = {c: b["detectors"]["scrfd"]["total_seconds"] / b["detectors"]["scrfd"]["calls"] for c, b in bench.items() if "detectors" in b}
    invariant = all(len({(b.get("detectors") or {}).get(n, {}).get("digest") for b in bench.values()}) == 1 for n in ("scrfd", "wilor", "hands23", "mask_rcnn"))
    if "session_c" in scrfd and "capped_3" in scrfd:
        out.append(f"CPU thread pools for the mask pipeline on the slice: cap onnxruntime, torch and the OpenMP, MKL and OpenBLAS pools at 3 threads (the 3.4-CPU quota rounded down) by default. "
                   f"SCRFD takes {scrfd['capped_3']:.3f} s a frame instead of {scrfd['session_c']:.2f} s, and "
                   + ("every detection is bitwise identical across 1, 3 and 4 threads and Session C's defaults, and the smoke-1 and smoke-2 masks are identical at the defaults and at 3 threads, so by Chase's rule the thread count is not a registered value."
                      if invariant else "some outputs change with the thread count, so the count would become a registered value."))
    roi = (diagnostics or {}).get("mask_roi_align") or {}
    rb = roi.get("bench") or {}
    if rb.get("mask_rcnn") and rb.get("hands23"):
        out.append(f"ROIAlign for the Detectron2 detectors on CUDA: keep torchvision's deterministic Python version (Mask R-CNN about 20 s a frame, Hands23 16 s a keyframe), "
                   f"or call its kernel for inference ({rb['mask_rcnn']['per_call_seconds']:.2f} s and {rb['hands23']['per_call_seconds']:.2f} s) with everything else deterministic. "
                   "The kernel's forward pass computes each output on one thread without atomics; repeats were bitwise identical, but its outputs differ from the Python version's in rounding, "
                   "so it is a registered change and the mask check would rerun on the slice with it (see the Session E follow-up).")
    out.append("Mask machine: keep the registered slice (MASK_GPU) or move to the Mac's CPU; see the per-reference comparison under Session E.")
    return out


def notes(envs: dict) -> list[str]:
    out = []
    if roi_align_registered():
        out.append("On pods the CPU pools run at the cgroup quota rounded down, with one inter-op thread (density/cpu_threads.py, scripts/pod/env.sh). Chase approved it on 2026-10-07; it is not a registered value, since no output changed on the slice.")
    if any(e["determinism"].get("fill_uninitialized_memory") for e in envs.values()):
        out.append("Deterministic mode fills newly allocated memory (torch.utils.deterministic.fill_uninitialized_memory), which costs a little time in every run; the timings include it.")
    if any(e["credentials_present"]["RUNPOD_API_KEY"] for e in envs.values()):
        out.append("RunPod injects a pod-scoped RUNPOD_API_KEY into every pod's environment. We never read or record its value and put no credentials on a pod ourselves.")
    return out


FLASH_SCOPE = ("832 x 480, 81 frames, on the H100 with flash attention registered (Session D), same device. It does not cover smaller sizes, where "
               "Milestone 2's T3 rows (the tiny model, and the real model at 416 x 240 with 17 frames) still differ from iteration 5 (cuBLAS at small "
               "row counts), or results across devices, which are reported, not judged")
RESOLUTIONS = {"hsa_dense": "with flash attention registered, HSA with every group at full steps equals the dense C1 run from another process, bitwise"}
GENERATION_STEPS = ("pytest", "milestones", "generate", "sdpa", "hsa_dense", "t3", "perf", "vae_precision", "sheet")


def build(ctx) -> dict:
    records, errors, pods, diagnostics = load(ctx)
    flash = ctx.out / "flash"  # Session D: flash attention registered
    records_d, errors_d = ({}, {})
    if flash.is_dir():
        from dataclasses import replace

        records_d, errors_d, _, _ = load(replace(ctx, out=flash))
    envs = {s: environment_summary(result(records, f"env_{s}")) for s in "ABCE" if result(records, f"env_{s}")}
    if result(records_d, "env_D"):
        envs["D"] = environment_summary(result(records_d, "env_D"))
    rerun = ctx.out / "rerun"  # Session F: the mask-check rerun with the registered configuration
    records_f, errors_f = ({}, {})
    if rerun.is_dir():
        from dataclasses import replace

        records_f, errors_f, _, _ = load(replace(ctx, out=rerun))
    if result(records_f, "env_F"):
        envs["F"] = environment_summary(result(records_f, "env_F"))
    both = {**records, **{f"D:{k}": v for k, v in records_d.items()}, **{f"F:{k}": v for k, v in records_f.items()}}
    errors = {**errors, **{f"D:{k}": v for k, v in errors_d.items()}, **{f"F:{k}": v for k, v in errors_f.items()}}
    read = lambda path: json.loads(path.read_text()) if path.exists() else None
    session_f = {"mask_check": result(records_f, "mask_check"), "mask_compare": result(records_f, "mask_compare"),
                 "mask_check_report": read(rerun / "mask_check" / "mask_pipeline.json"), "gate": read(rerun / "mask_canary" / "gate.json"),
                 "canary_registered": read(rerun / "mask_canary" / "registered.json"), "canary_check": read(rerun / "mask_canary" / "mask_canary.json")}
    stops = {name: rec["result"]["stop"] for name, rec in both.items() if rec["result"].get("stop")}
    label = lambda name: ("Session D (flash, registered): " if name.startswith("D:") else "Session B (cuDNN, not registered): ") if records_d else ""
    blocks, resolved = [], []
    for name, rec in both.items():
        redo = None if name.startswith("D:") else result(records_d, name)  # the same step rerun in Session D
        for flag in rec["result"].get("blocks_pilot") or []:
            if redo is not None and not redo.get("blocks_pilot"):
                resolved.append(f"{label(name)}{flag}. Resolved in Session D: {RESOLUTIONS.get(name, 'the same step passes with flash attention registered')}.")
            else:
                blocks.append(f"{label(name)}{flag}")
    bugs = [bug for rec in both.values() for bug in rec["result"].get("bugs") or []]
    settings_changed = {name: rec["determinism_changed"] for name, rec in both.items() if rec.get("determinism_changed")}
    session_b = {k: result(records, k) for k in GENERATION_STEPS}
    session_d = {**{k: result(records_d, k) for k in GENERATION_STEPS}, "crossmachine": result(records_d, "crossmachine")}
    registered = records_d if result(records_d, "perf") else records  # flash attention is the registered configuration
    session_c = {k: result(records, k) for k in ("masks", "mask_check", "mask_compare")}
    return {
        "_out": str(ctx.out),
        "title": "GPU smoke test (protocol sections 8, 11, 12 and 13)",
        "experiment_data": "No experiment data: the two smoke prompts are excluded from every prompt pool.",
        "generated_utc": smoke.utc_now(),
        "dry_run": any(rec.get("dry_run") for rec in records.values()),
        "code": {name: rec["key"]["git_sha"] for name, rec in both.items()},
        "steps_done": sorted(both),
        "step_errors": {name: {"error": e["error"], "oom": e["oom"], "failed_utc": e["failed_utc"]} for name, e in errors.items()},
        "stops": stops,
        "blocks_pilot": blocks,
        "resolved": resolved,
        "bugs_to_investigate": bugs,
        "determinism_changed_during_steps": settings_changed,
        "pods": pod_summary(pods),
        "environments": envs,
        "hashes": hashes(records, records_d),
        "session_b": session_b,
        "session_d": session_d,
        "session_c": session_c,
        "session_f": session_f,
        "tolerances": tolerances(registered, FLASH_SCOPE if registered is records_d else None, registered=registered is records_d),
        "tolerances_cudnn": tolerances(records, "832 x 480, 81 frames, on the H100 with PyTorch's default attention (cuDNN), Session B") if registered is not records else None,
        "lpips": f"{LPIPS_EXACT_NOTE} The section 11 check is {registered_tolerances.LPIPS_CHECK}." if registered is records_d and exact_t3(result(records_d, "t3")) else LPIPS_NOTE,
        "projections": projections(records, pods, ctx, perf=result(registered, "perf"), diagnostics=diagnostics),
        "warnings": warnings_summary(both),
        "tf32": tf32_decision(records),
        "registered": registrations(tf32_decision(records), d_done=bool(records_d), perf_flash=result(records_d, "perf")),
        "attention_finding": attention_finding(diagnostics, result(records, "perf"), result(records_d, "perf")),
        "decisions_for_chase": decisions(records, envs, diagnostics, tolerances_registered=registered is records_d),
        "approved_pending": approved_pending(records, diagnostics),
        "notes": notes(envs),
        "diagnostics": diagnostics,
        "steps": both,
    }


# Markdown


def fmt(value, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if value == float("inf"):
            return "inf"
        return "0" if value == 0 else f"{value:.{digits}g}"
    return str(value)


def gib(m: dict | None, key: str) -> str:
    return "" if not m else f"{m[key]:.2f}"


def table(header: list[str], rows: list[list]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"] + ["| " + " | ".join(fmt(c) for c in row) + " |" for row in rows]


def region_text(v: dict) -> str:
    if v["identical"]:
        return "identical"
    return f"differs ({v['pixels_differing']} px, IoU {v['iou']:.4f})" if "iou" in v else "differs"


MAIN_RUN_REFERENCES = {"face": 72, "manipulation": 72, "body": 16, "control": 16}
STRATUM_REFERENCE = {"face": "smoke-1", "manipulation": "smoke-2", "body": "body"}  # Session E's stand-in for each stratum


def digest_groups(bench: dict, detector: str) -> str:
    """Configurations grouped by bitwise-equal outputs, e.g. "session_c | capped_3, single | capped_4"."""
    groups: dict[str, list[str]] = {}
    for config, b in bench.items():
        d = (b.get("detectors") or {}).get(detector)
        if d:
            groups.setdefault(d["digest"], []).append(config)
    return " | ".join(", ".join(g) for g in groups.values())


def mask_threads_summary(diag: dict, mac: dict | None, roi: dict | None = None, body_path_seconds: float | None = None) -> dict:
    """Seconds per reference on the slice against the Mac's validated CPU run, per stratum and weighted by the main run.

    "slice" is the registered pipeline with the CPU pools capped at 3 (strict mode); where the bench showed the full body
    clip would take about 27 min, its time is Session C's body path (``body_path_seconds``), which the thread cap does not
    change. "slice_kernel" adds Detectron2's ROIAlign through torchvision's kernel (the follow-up, ``roi``)."""
    slice_refs = {}
    for key in ("capped_3", "capped_3_body"):
        slice_refs.update(((diag.get("pipeline") or {}).get(key) or {}).get("references") or {})
    kernel_refs = ((roi or {}).get("pipeline") or {}).get("references") or {}
    mac_refs = ((mac or {}).get("pipeline") or {}).get("references") or {}
    out = {"per_stratum": {}, "rule": "the slice should come within about 2x of the Mac's per-reference mask time (Chase, 2026-10-07)"}
    for stratum, rid in STRATUM_REFERENCE.items():
        a, b, k = slice_refs.get(rid), mac_refs.get(rid), kernel_refs.get(rid)
        now = a["wall_seconds"] if a else (body_path_seconds if stratum == "body" else None)
        if now is None or not b:
            continue
        row = {"reference": rid, "slice_seconds": now, "slice_basis": "measured in Session E" if a else "Session C's body path (the thread cap leaves Mask R-CNN unchanged)",
               "mac_seconds": b["wall_seconds"], "ratio": now / b["wall_seconds"]}
        if k:
            row.update(slice_kernel_seconds=k["wall_seconds"], kernel_ratio=k["wall_seconds"] / b["wall_seconds"])
        out["per_stratum"][stratum] = row
    if len(out["per_stratum"]) == len(STRATUM_REFERENCE):
        total = lambda key: sum(MAIN_RUN_REFERENCES[k] * v[key] for k, v in out["per_stratum"].items())
        out["main_run_hours"] = {"slice": total("slice_seconds") / 3600, "mac": total("mac_seconds") / 3600}
        out["main_run_ratio"] = total("slice_seconds") / total("mac_seconds")
        if all("slice_kernel_seconds" in v for v in out["per_stratum"].values()):
            out["main_run_hours"]["slice_kernel"] = total("slice_kernel_seconds") / 3600
    return out


def secs(x) -> str:
    return "" if x is None else f"{x:.0f}" if x >= 100 else f"{x:.1f}"


def roi_align_lines(roi: dict) -> list[str]:
    """The follow-up: Detectron2's ROIAlign through torchvision's kernel for inference, everything else deterministic."""
    lines = ["### Follow-up: Detectron2's ROIAlign through torchvision's kernel (Session E)", "",
             "With deterministic algorithms on and the input on CUDA, torchvision 0.29.1 replaces its roi_align kernel with a pure-Python version, "
             "lazily torch.compile'd, which exists for a deterministic backward pass; the profile above shows it taking the whole Mask R-CNN call. "
             "The masks need only the forward pass. This run called the kernel for inference, with every other operation deterministic and the CPU pools at 3.", ""]
    bench = roi.get("bench") or {}
    for name in ("mask_rcnn", "hands23"):
        b = bench.get(name)
        if b:
            diffs = "; ".join(json.dumps(c) for c in b["against_python_version"])
            lines.append(f"- {name}: {b['per_call_seconds']:.3f} s a call with the kernel; repeat bitwise {fmt(b['repeat_bitwise'])}; "
                         f"bitwise equal to the Python version on the first 3 inputs: {', '.join(fmt(x) for x in b['bitwise_equal_to_python_version'])}. Against it: {diffs}.")
    if "error" in bench:
        lines.append(f"- bench failed: {bench['error']}")
    pipe = roi.get("pipeline") or {}
    if pipe.get("references"):
        lines += ["", *table(["Reference", "Stratum", "Seconds (best of 2)", "Repeat identical", "Components (s)"],
                             [[rid, r["stratum"], secs(r["wall_seconds"]), r["repeat_identical"], ", ".join(f"{k} {v:.0f}" for k, v in r["seconds"].items() if k != "total")] for rid, r in pipe["references"].items()])]
    elif pipe.get("error"):
        lines.append(f"- pipeline failed: {pipe['error']}")
    lines.append("")
    return lines


def mask_threads_lines(diag: dict, mac: dict | None, roi: dict | None = None, body_path_seconds: float | None = None) -> list[str]:
    lines = ["### Mask detectors and CPU threads on the slice (Session E)", ""]
    lines.append(f"Host {diag.get('host')}: cgroup cpu.max `{diag.get('cgroup_cpu_max')}` of {diag.get('cpu_count')} host CPUs. "
                 "Each configuration ran in its own process. session_c is Session C's setup (env.sh's OMP and MKL threads, the quota rounded up; "
                 "onnxruntime, OpenCV and torch's inter-op pool at their defaults); capped_N sets every pool to N threads (torch's inter-op and onnxruntime's inter-op to 1).")
    lines.append("")
    bench = diag.get("bench") or {}
    rows = []
    for config, b in bench.items():
        if "detectors" not in b:
            rows.append([config, b.get("error", "no record"), "", "", "", "", "", ""])
            continue
        p, d = b["pools"], b["detectors"]
        per = lambda name: d[name]["total_seconds"] / d[name]["calls"]
        scrfd_cg = d["scrfd"]["cgroup"] or {}
        share = f"{scrfd_cg['nr_throttled']} of {scrfd_cg['nr_periods']}" if scrfd_cg else ""
        ort = p["onnxruntime"]["intra_op"] or "default"
        rows.append([config, f"torch {p['torch_intra_op']}/{p['torch_inter_op']}, onnxruntime {ort}, cv2 {p['cv2']}; {fmt(p['os_threads_after_build'])} OS threads",
                     per("scrfd"), per("wilor"), per("hands23"), per("mask_rcnn"), share, d["mask_rcnn"].get("process_cpu_seconds")])
    lines += table(["Config", "Pools (intra/inter)", "SCRFD (s/frame)", "WiLoR (s/frame)", "Hands23 (s/keyframe)", "Mask R-CNN (s/frame)", "SCRFD: throttled periods", "Mask R-CNN CPU (s)"], rows)
    lines += ["", "Outputs grouped by bitwise equality: " + "; ".join(f"{n} [{digest_groups(bench, n)}]" for n in ("scrfd", "wilor", "hands23", "mask_rcnn")) + ".", ""]
    stage_rows = []
    for config, b in bench.items():
        for model, calls in (b.get("detectron2_stages") or {}).items():
            med = lambda key: sorted(c[key] for c in calls)[len(calls) // 2]
            stage_rows.append([config, model, med("preprocess_seconds"), med("model_seconds"), med("to_cpu_seconds")])
    if stage_rows:
        lines += table(["Config", "Detectron2 model", "Preprocessing (s)", "Model (s)", "Copy to host (s)"], stage_rows) + [""]
    off = (bench.get("capped_3") or {}).get("deterministic_off")
    if off:
        lines.append("Deterministic algorithms off (a diagnostic only), capped_3: " + "; ".join(
            f"{n} {v['total_seconds'] / v['calls']:.2f} s a call, same outputs {fmt(v['same_outputs_as_deterministic'])}" for n, v in off.items() if isinstance(v, dict)) + ".")
        lines.append("")
    for config in ("session_c", "capped_3"):
        prof = (bench.get(config) or {}).get("profile") or {}
        if "mask_rcnn" in prof:
            lines += [f"Mask R-CNN, one call profiled at {config} (top operators by self CPU time):", "", "```", *prof["mask_rcnn"]["by_self_cpu"].splitlines()[:12], "```", ""]
    pipe = diag.get("pipeline") or {}
    prow = []
    base = ((pipe.get("session_c") or {}).get("references")) or {}
    for key, p in pipe.items():
        if "references" not in p:
            prow.append([key, "", "", p.get("skipped") or p.get("error"), "", ""])
            continue
        for rid, r in p["references"].items():
            same = (base.get(rid) or {}).get("digest")
            prow.append([key, rid, r["stratum"], r["wall_seconds"], ", ".join(f"{k} {v:.0f}" for k, v in r["seconds"].items() if k != "total"),
                         "" if key == "session_c" or same is None else fmt(r["digest"] == same)])
    if prow:
        lines += table(["Run", "Reference", "Stratum", "Seconds (models built)", "Components (s)", "Masks equal to session_c"], prow) + [""]
    mac_pipe = (mac or {}).get("pipeline") or {}
    if mac_pipe.get("references"):
        lines += ["The Mac's CPU in its validated configuration (torch 8 threads, its default; onnxruntime's default), the same references:", ""]
        lines += table(["Reference", "Stratum", "Seconds", "Components (s)"], [[rid, r["stratum"], r["wall_seconds"], ", ".join(f"{k} {v:.0f}" for k, v in r["seconds"].items() if k != "total")] for rid, r in mac_pipe["references"].items()]) + [""]
    if roi:
        lines += roi_align_lines(roi)
    summary = mask_threads_summary(diag, mac, roi, body_path_seconds)
    if summary["per_stratum"]:
        lines += ["#### Seconds per reference: the slice against the Mac", ""]
        lines += table(["Stratum", "Reference", "Slice, threads capped (s)", "Slice / Mac", "Slice, also the ROIAlign kernel (s)", "Slice / Mac", "Mac (s)"],
                       [[k, v["reference"], secs(v["slice_seconds"]) + ("" if v["slice_basis"].startswith("measured") else " (Session C)"), v["ratio"], secs(v.get("slice_kernel_seconds")), v.get("kernel_ratio"), secs(v["mac_seconds"])] for k, v in summary["per_stratum"].items()])
        if "main_run_ratio" in summary:
            h = summary["main_run_hours"]
            lines += ["", f"Weighted by the main run's references (72 face, 72 manipulation, 16 body, 16 control): {h['slice']:.1f} h on the slice with the threads capped"
                      + (f", {h['slice_kernel']:.1f} h with the ROIAlign kernel too," if "slice_kernel" in h else "") + f" against {h['mac']:.1f} h on the Mac. Rule: {summary['rule']}."]
        lines.append("")
    for when in ("du_start", "du_end"):
        du = diag.get(when) or {}
        if "allocated" in du:
            lines.append(f"- du at {'the start' if when == 'du_start' else 'the end'}: " + "; ".join(
                f"{k} {v['used_gb']:.1f} GB in {v['seconds']:.0f} s" if "used_gb" in v else f"{k}: {v.get('error')}" for k, v in du.items() if isinstance(v, dict)) + f" (stop at {du['stop_gb']:g} GB).")
    lines.append("")
    return lines


def mask_compare_lines(compare: dict) -> list[str]:
    rows = []
    for cid, r in compare["clips"].items():
        if isinstance(r, str):
            rows.append([cid, r, "", "", "", ""])
            continue
        regions = "; ".join(f"{n} {region_text(v)}" for n, v in r["regions"].items())
        rows.append([cid, regions, r["tracks_identical"], r["events_identical"], f"{r['pipeline_decision']['mac']} / {r['pipeline_decision']['pod']}", f"{r['seconds_total']['mac_cpu']:.0f} / {r['seconds_total']['pod']:.0f}"])
    return ["Stock clips, slice vs the Mac's committed masks (reported, not judged):", "", *table(["Clip", "Regions", "Tracks identical", "Events identical", "Decision (Mac / pod)", "Seconds (Mac CPU / pod)"], rows), ""]


def session_f_lines(report: dict) -> list[str]:
    """Session F: the mask-check rerun on the slice with the thread cap and the registered ROIAlign kernel, and the mask canary."""
    f = report.get("session_f") or {}
    m = f.get("mask_check")
    if not m:
        return []
    lines = ["## Session F (slice): the mask check with the registered configuration", "",
             "The stock-clip mask check rerun on the slice with the CPU pools at the quota (3 threads) and Detectron2's ROIAlign through "
             "torchvision's kernel (registered 2026-10-07), the body clip included; then the mask canary recorded and checked.", ""]
    cross = [r["test"] for r in m["failed_rows"] if r["test"].startswith("Conversion: WiLoR")]
    other = [r["test"] for r in m["failed_rows"] if r["test"] not in cross]
    lines.append(f"- Mask check: {m['num_judged'] - m['num_failed']}/{m['num_judged']} judged rows passed in {m['seconds'] / 60:.1f} min. "
                 + "; ".join(f"{r['test']}: {fmt(r['passed']) if r['passed'] is not None else 'not run on this device'}" for r in m["t26_rows"]) + "."
                 + (" The WiLoR conversion row is the cross-device comparison against detections recorded on the Mac's ARM CPU (reported, not judged)." if cross else "")
                 + (" Other rows that failed: " + "; ".join(other) + "." if other else ""))
    rep = f.get("mask_check_report") or {}
    clips = rep.get("clip_results") or {}
    if clips:
        lines += ["", *table(["Clip", "Seconds", "Components (s)"], [[cid, secs(r["seconds"]["total"]), ", ".join(f"{k} {v:.0f}" for k, v in r["seconds"].items() if k != "total")] for cid, r in clips.items()])]
        body = clips.get("body", {}).get("seconds") or {}
        if body:
            path = body.get("B.detector", 0) + body.get("B.sam", 0) + body.get("roi_coverage_measurement_eligibility", 0)
            kernel = ((((report.get("diagnostics") or {}).get("mask_roi_align") or {}).get("pipeline") or {}).get("references") or {}).get("body")
            lines += ["", f"The body clip's body path (B detector, B SAM and the ROI measures) took {path:.0f} s with the production code"
                      + (f", against {kernel['wall_seconds']:.0f} s in Session E's follow-up." if kernel else ".")]
    lines.append("")
    if f.get("mask_compare") and f["mask_compare"].get("clips"):
        lines += mask_compare_lines(f["mask_compare"])
    gate, reg, chk = f.get("gate"), f.get("canary_registered"), f.get("canary_check")
    if gate:
        lines.append(f"- Gate before recording the canary: {'passed' if gate['passed'] else 'STOPPED: ' + '; '.join(gate['problems'])}.")
    if reg:
        lines.append(f"- Mask canary recorded on {reg['pod']['host']} ({reg['pod']['gpu']}, driver {reg['pod']['driver']}, CPU {reg['pod']['cpu']}) at code {reg['code']['git_sha'][:7]}: "
                     f"{len(reg['hashes'])} hashes ({', '.join(sorted(reg['hashes']))}); counts {json.dumps(reg['counts'])}; {reg['seconds']:.0f} s.")
    if chk:
        lines.append(f"- Mask canary checked from a fresh process: {'passed' if chk.get('passed') else 'STOP: ' + str(chk.get('stop'))} in {chk['seconds']:.0f} s.")
    lines.append("")
    return lines


def markdown(report: dict) -> str:
    lines = [f"# {report['title']}", ""]
    if report["dry_run"]:
        lines += ["**DRY RUN on the Mac with tiny random models: these numbers say nothing about the pods.**", ""]
    lines += [f"Generated {report['generated_utc']}. {report['experiment_data']}", ""]
    shas = sorted({sha for sha in report["code"].values() if sha})
    lines += [f"Code: {', '.join(f'`{s}`' for s in shas) or 'unknown'}. Steps done: {', '.join(report['steps_done'])}.", ""]

    lines += ["## Status", ""]
    if report["step_errors"]:
        lines += [f"- **Error** in {name}: {e['error']}{' (out of memory)' if e['oom'] else ''}" for name, e in report["step_errors"].items()]
    lines += [f"- **Stop** in {name}: {'; '.join(reasons)}" for name, reasons in report["stops"].items()]
    lines += [f"- **Blocks the pilot:** {flag}" for flag in report["blocks_pilot"]]
    lines += [f"- **Resolved:** {r}" for r in report.get("resolved") or []]
    if report.get("attention_finding"):
        lines.append(f"- **Finding:** {report['attention_finding']}")
    lines += [f"- **Bug to investigate:** {bug}" for bug in report["bugs_to_investigate"]]
    lines += [f"- **Deterministic settings changed during {name}:** {json.dumps(c)}" for name, c in report["determinism_changed_during_steps"].items()]
    if not (report["step_errors"] or report["stops"] or report["blocks_pilot"] or report["bugs_to_investigate"] or report["determinism_changed_during_steps"]):
        lines.append("- No open errors, stops, pilot blockers or bugs.")
    lines += [f"- **Registered:** {r}" for r in report.get("registered") or []]
    lines += [f"- **Decision for Chase:** {d}" for d in report["decisions_for_chase"]]
    lines += [f"- **Approved, pending:** {d}" for d in report.get("approved_pending") or []]
    lines += [f"- Note: {n}" for n in report["notes"]]
    lines.append("")

    pods = report["pods"]
    if pods:
        lines += ["## Pods and spend", ""]
        lines += table(["Session", "GPU", "Pod", "Created (UTC)", "Deleted (UTC)", "Hours", "$/hr", "Cost ($)"],
                       [[p["session"], p["gpu"], p["pod_id"], p.get("created_utc"), p.get("deleted_utc"), p["hours"], p["price_per_hour"], p["cost_usd"]] for p in pods["pods"]])
        lines += ["", *table(["GPU", "Pods", "Hours", "Cost ($)"], [[g, v["pods"], v["hours"], v["cost_usd"]] for g, v in pods["by_gpu"].items()]), "", f"Total pod spend: ${pods['total_cost_usd']:.2f}. {pods['note']}.", ""]
        bill = pods.get("billing")
        if bill:
            posted = {r["podId"]: r["totalAmount"] for r in bill["records"]}
            lines += [f"RunPod's billing records, queried {bill['queried_utc']}: ${bill['total_posted_usd']:.2f} posted ("
                      + ", ".join(f"pod {p['session']} ${posted[p['pod_id']]:.2f}" if p["pod_id"] in posted else f"pod {p['session']} not posted yet" for p in pods["pods"]) + f"). {bill['note']}.", ""]

    if report["environments"]:
        lines += ["## Pod environments", ""]
        envs = report["environments"]
        fields = [
            ("GPU", lambda e: f"{e['gpu']} (sm {e['capability']}, {fmt(e['gpu_memory_gib'])} GiB)"), ("Driver", lambda e: e["nvidia_smi_query"]),
            ("Driver CUDA", lambda e: e["driver_cuda_version"]), ("torch / CUDA / cuDNN", lambda e: f"{e['torch']} / {e['torch_cuda']} / {e['cudnn']}"),
            ("CUDA toolkit", lambda e: e["nvcc_toolkit"]), ("Python, OS", lambda e: f"{e['python']}, {e['os']}, kernel {e['kernel']}"),
            ("CPU, RAM", lambda e: f"{e['cpu']}: {fmt(e['cpus'])} CPUs allowed of {e['host_cpus']} on the host; {fmt(e['memory_limit_gib'])} GiB RAM allowed"),
            ("Code", lambda e: f"{e['git_sha']} (dirty: {e['git_dirty']})"),
            ("Deterministic settings", lambda e: ", ".join(f"{k} {v}" for k, v in e["determinism"].items() if k != "sdpa_enabled")),
            ("SDPA backends enabled", lambda e: ", ".join(k for k, v in e["determinism"]["sdpa_enabled"].items() if v)),
            ("Detectron2 archs", lambda e: ", ".join(f"sm_{a}" for a in e["detectron2_archs"])), ("sam-2 CUDA extension", lambda e: fmt(e["sam2_cuda_extension"])),
            ("/workspace df (the whole network-storage cluster, not the 100 GB volume)", lambda e: "" if not e["disk"] else f"{e['disk']['workspace_used_gib']:.1f} of {e['disk']['workspace_total_gib']:.1f} GiB used"),
        ]
        lines += table(["", *envs], [[label, *(fn(e) for e in envs.values())] for label, fn in fields]) + [""]

    def generation_sections(b: dict, label: str) -> list[str]:
        lines = [f"## {label}", ""]
        if b["pytest"] or b["milestones"]:
            lines += ["### Tests and Milestones 1-3 on CUDA", ""]
            if b["pytest"]:
                lines.append(f"- pytest: {b['pytest']['summary']}")
            for name, m in (b["milestones"] or {}).items():
                if isinstance(m, dict) and "num_judged" in m:
                    lines.append(f"- {name.upper()}: {m['num_judged'] - m['num_failed']}/{m['num_judged']} judged rows passed; {len(m['measurements'])} same-device numeric differences (measurements); {len(m['failures'])} failures; {len(m['unjudged'])} unjudged.")
                    lines += [f"  - measurement: {r.get('test') or r.get('check')} (" + ", ".join(str(x) for x in (f"{r['model']} model" if r.get("model") else None, r.get("dtype"), r.get("preset"), f"seed {r['seed']}" if "seed" in r else None) if x)
                              + f"): max abs diff {fmt(r.get('max_abs_diff'))} from step {fmt(r.get('first_divergent_step'))}" for r in m["measurements"]]
                    lines += [f"  - **failure**: {r.get('test') or r.get('check')} ({r.get('device')}, {r.get('dtype', '')})" for r in m["failures"]]
            if b["milestones"]:
                lines.append(f"- {b['milestones']['unjudged_note']}")
            lines.append("")

        if b["generate"]:
            lines += ["### Dense C1 at full resolution, twice per prompt", ""]
            lines += table(["Prompt", "Bitwise", "Max abs latent diff", "First divergent step", "Decoded PSNR (dB)", "uint8 identical", "VAE repeat max diff", "Denoise (s)", "Decode (s)"],
                           [[pid, r["repeat"]["bitwise_equal"], r["repeat"]["max_abs_diff"], r["repeat"]["first_divergent_step"], r["repeat_frames"]["decoded_psnr_db"], r["repeat_frames"]["uint8_identical"], r["vae_repeat_max_abs_diff"], r["runs"][0]["denoise_seconds"], r["runs"][0]["decode_seconds"]] for pid, r in b["generate"]["prompts"].items()])
            lines.append("")
        if b["hsa_dense"]:
            lines += ["### All HSA groups dense vs dense, full resolution", ""]
            lines += table(["Prompt", "Bitwise", "Max abs latent diff", "First divergent step", "Decoded PSNR (dB)", "Relative to repeat noise"],
                           [[pid, r["bitwise_equal"], r["max_abs_diff"], r["first_divergent_step"], r["decoded_psnr_db"], r["relative_to_dense_repeat_noise"].get("ratio") if r["relative_to_dense_repeat_noise"].get("ratio") is not None else r["relative_to_dense_repeat_noise"].get("note")] for pid, r in b["hsa_dense"]["prompts"].items()])
            lines.append("")
        if b["t3"]:
            t3 = b["t3"]
            lines += ["### T3: fast path vs masked-dense oracle, full resolution, HSA-50", "", f"C4 at lambda {t3['lambda']:g} with a centered disk of radius {t3['c4_mask']['radius_px']:g} px ({t3['c4_mask']['tokens_with_coverage']} tokens with coverage). {t3['oracle_labels'].capitalize()}.", ""]
            rows = []
            for pid, rec in t3["prompts"].items():
                for name, c in rec["conditions"].items():
                    rows.append([pid, name, c["bitwise_equal"], c["max_abs_diff"], c["first_divergent_step"], c["decoded_psnr_db"], c.get("decoded_psnr_db_uint8"), c["token_step_fraction"], c["verdict"]])
                if rec.get("reduced"):
                    lines.append(f"- {pid}: **reduced to {rec['reduced']['pixel_frames']} frames**: {rec['reduced']['reason']}.")
            lines += table(["Prompt", "Condition", "Bitwise", "Max abs latent diff", "First divergent step", "Decoded PSNR (dB)", "PSNR of the saved 8-bit frames (dB)", "Token-step fraction", "Verdict"], rows)
            same = {pid: r.get("c3_c4_identical_active_counts") for pid, r in t3["prompts"].items()}
            exact = all(c["bitwise_equal"] for r in t3["prompts"].values() for c in r["conditions"].values())
            lines += [""]
            if exact:
                lines.append(f"The fast path equals the oracle bitwise at full resolution for C3 and C4. C3 and C4 run identical active-token counts at every iteration, layer and guidance branch (checked from the run logs: {', '.join(f'{pid} {fmt(v)}' for pid, v in same.items())}).")
            else:
                lines += ["Dense repeats within one process are bitwise on CUDA, so repeat-run noise is zero and is no yardstick for T3: the deviation is reported as decoded PSNR. The structural checks are the bug test. (Across processes, cuDNN attention does not reproduce: see the diagnostics.)",
                          f"{SAME_COUNTS_NOTE} Checked from the run logs: {', '.join(f'{pid} {fmt(v)}' for pid, v in same.items())}.",
                          f"Lossless frames for LPIPS: {LPIPS_NOTE}"]
            lines += [
                      "Fast-path repeat (C3 twice): " + "; ".join(f"{pid}: bitwise {fmt(r['fast_path_repeat']['bitwise_equal'])}, max abs diff {fmt(r['fast_path_repeat']['max_abs_diff'])}" for pid, r in t3["prompts"].items()), ""]
            for pid, rec in t3["prompts"].items():
                for name, c in rec["conditions"].items():
                    e = c.get("evidence")
                    if e:
                        lines.append(f"- Evidence, {pid} {name}, step {e['step']}: first differing module `{e.get('module')}` ({e.get('branch', '')}); inputs equal: {fmt(e.get('module_inputs_equal'))}; kernels alone reproduce it: {fmt(e.get('kernel_alone_reproduces'))}; replay matches the run: {fmt(e.get('replay_matches_run'))}.")
            lines.append("")
        if b["sdpa"]:
            s = b["sdpa"]
            lines += ["### Attention backend (SDPA)", ""]
            for key in ("dense", "hsa_subset"):
                lines.append(f"- {key}: " + "; ".join(f"{c['backend']} for query {c['query']} x key {c['key']} {c['dtype']} ({c['calls']} calls)" for c in s[key]["choices"]) + (f"; profiler kernels: {s[key].get('kernel_backends')}" if s[key].get("kernel_backends") else ""))
            lines += [f"- Same backend for dense and subset query counts: {fmt(s['same_backend_dense_and_subset'])}.", ""]
        if b["perf"]:
            p = b["perf"]
            lines += ["### Memory and timing (CUDA events)", "", f"Prompt {p['prompt']}; {p['warmup']}; T5 never loaded; KV caches released before each decode.", ""]
            lines += table(["Condition", "Token-step fraction", "Denoise (s)", "Spread (s)", "Decode (s)", "End to end (s)", "Peak alloc denoise (GiB)", "Peak reserved denoise (GiB)", "Peak alloc decode (GiB)", "Peak reserved decode (GiB)"],
                           [[c, v["token_step_fraction"], v["denoise_seconds_mean"], v["denoise_seconds_spread"], v["decode_seconds_mean"], v["end_to_end_seconds_mean"], v["peak_allocated_denoise_gib"], v["peak_reserved_denoise_gib"], v["peak_allocated_decode_gib"], v["peak_reserved_decode_gib"]] for c, v in p["summary"].items()])
            steady = steady_state(p)
            o = steady["c4_overhead_vs_c3"]
            lines += ["", f"C4's overhead over C3 in steady state (best of two runs each): {o['seconds']:+.2f} s ({100 * o['relative']:+.2f} percent). "
                      + "First run in excess of the best one: " + ", ".join(f"{c} {v:+.1f} s" for c, v in steady["first_run_excess_seconds"].items())
                      + ". The means above include that first-call cost.",
                      f"RE time: {p['re_seconds']:.1f} s ({p['re_definition']}). Weights resident after loading: {gib(p['resident_after_load'], 'allocated_gib')} GiB.", ""]
        if b.get("vae_precision"):
            v = b["vae_precision"]
            lines += ["### TF32 vs IEEE fp32 convolutions", "", "Comparison runs for the registered choices: TF32 for the VAE (density/wan.py) and strict fp32 for SAM 2.1 (density/masks/registered.py), by the rule below.", ""]
            lines += table(["Prompt", "VAE decode, TF32 (s)", "VAE decode, IEEE (s)", "Decoded PSNR, IEEE vs TF32 (dB)", "Max abs diff", "uint8 identical"],
                           [[pid, r["tf32_seconds"], r["ieee_seconds"], r["decoded_psnr_db"], r["decoded_max_abs_diff"], r["uint8_identical"]] for pid, r in v["prompts"].items()])
            d = report.get("tf32") or {}
            lines += ["", f"Rule (Chase): {d.get('rule', '')}."] if d else []
            if d.get("vae"):
                rel = d["vae"]["relative_to"]
                lines.append(f"- VAE: IEEE fp32 adds {min(d['vae']['added_seconds_per_video']):.1f} to {max(d['vae']['added_seconds_per_video']):.1f} s per video; the larger is {100 * rel['C1']:.1f} percent of C1's and {100 * rel['C3']:.1f} percent of C3's end-to-end time. Within 10 percent: {fmt(d['vae']['within_rule'])}.")
            for pid, m in (d.get("masks") or {}).items():
                lines.append(f"- Masks, {pid} ({m['stratum']}): IEEE fp32 takes {m['ieee_seconds']:.1f} s against {m['default_seconds']:.1f} s ({100 * m['relative']:+.1f} percent); identical masks: {fmt(m['identical_masks'])}. Within 10 percent: {fmt(m['within_rule'])}.")
            masks = (report["session_c"] or {}).get("masks")
            if masks:
                for pid, r in masks["references"].items():
                    e = r.get("ieee_fp32_convolutions")
                    if e:
                        lines.append(f"- Masks, {pid}: IEEE fp32 convolutions {'give identical masks' if e['identical_to_default'] else 'change the masks: ' + json.dumps(e['differences'])}; {e['seconds']['total']:.1f} s with the models already built.")
            lines.append("")
        if b["sheet"]:
            lines += ["### References and frame 0", "", f"Contact sheet `{b['sheet']['contact_sheet']['path']}` (sha256 `{b['sheet']['contact_sheet']['sha256'][:16]}...`), frames {b['sheet']['contact_sheet']['frames']}.", ""]
            lines += table(["Prompt", "Frame 0 luma", "Frames 1-4 mean", "Other frames mean (std)", "Other frames max", "Brighter than the mean", "Brighter than every other frame"],
                           [[pid, v["frame0"], v["frames_1_to_4_mean"], f"{v['other_frames_mean']:.4f} ({v['other_frames_std']:.4f})", v["other_frames_max"], v["brighter_than_mean"], v["brighter_than_every_other_frame"]] for pid, v in b["sheet"]["frame0_luma"].items()])
            lines.append("")

        return lines

    lines += generation_sections(report["session_b"], "Session B (H100): PyTorch's default attention, cuDNN at full resolution")
    if any(report.get("session_d", {}).values()):
        lines += generation_sections(report["session_d"], "Session D (H100): flash attention, registered")
        x = report["session_d"].get("crossmachine")
        if x:
            lines += ["### Across machines: Session D against pod B's flash diagnostic", ""]
            for name, host in (("pod B", x["pod_b"]), ("this pod", x["this_pod"])):
                if host:
                    lines.append(f"- {name}: {host['hostname']}, {host['driver']}, driver CUDA {host['driver_cuda']}, {host['cpu']}, kernel {host['kernel']}.")
            for name, c in x["checks"].items():
                lines.append(f"- {name}: {json.dumps(c, default=str)[:300]}")
            lines.append("")

    if report["tolerances"]:
        lines += ["## Tolerance calibration (protocol section 11)", ""]
        for t in report["tolerances"]:
            proposal = t.get("proposal")
            limit = f", latent max abs diff {fmt(proposal['latent_max_abs_diff'])}" if proposal and "latent_max_abs_diff" in proposal else ""
            status = f" **Registered** ({registered_tolerances.REGISTERED}, density/tolerances.py): {t['registered']['rule']}." if t.get("registered") else (f" Proposed, not applied: {proposal['rule']}{limit} ({proposal['basis']})." if proposal else "")
            lines.append(f"- **{t['test']}** (protocol: {t['protocol']}). Observed: {json.dumps(t['observed'], default=str)[:600]}." + status
                         + (f" Scope: {t['scope']}." if t.get("scope") else ""))
        lines += [f"- LPIPS: {report['lpips']}", ""]

    c = report["session_c"]
    if c["masks"]:
        lines += ["## Masks on the slice", ""]
        rows = []
        for pid, r in c["masks"]["references"].items():
            parts = ", ".join(f"{k} {v:.1f}" for k, v in r["seconds"].items())
            rows.append([pid, r["stratum"], r["seconds"]["total"], r["repeat_seconds"]["total"], r["repeat_identical"], r["pipeline_decision"], r["roi_usable_frames"], parts])
        lines += table(["Reference", "Stratum", "Seconds", "Repeat seconds", "Repeat identical", "Eligibility", "Usable ROI frames", "Components (s)"], rows)
        sam = c["masks"]["sam_attention"]
        backends = sorted({x["backend"] for x in sam.get("choices", [])})
        lines += ["", f"SAM 2.1's attention (fp32): {', '.join(backends) or sam.get('error', 'not recorded')} for all {sum(x['calls'] for x in sam.get('choices', []))} calls of a 9-frame probe.", ""]
    if c["mask_check"]:
        m = c["mask_check"]
        cross = [r["test"] for r in m["failed_rows"] if r["test"].startswith("Conversion: WiLoR")]
        other = [r["test"] for r in m["failed_rows"] if r["test"] not in cross]
        lines += [f"Stock-clip mask check on {report['environments'].get('C', {}).get('gpu', 'the slice')}: {m['num_judged'] - m['num_failed']}/{m['num_judged']} judged rows passed in {m['seconds'] / 60:.1f} min."
                  + (" The WiLoR conversion row compares against detections recorded on the Mac's ARM CPU, so on the pod's x86 CPU it is a cross-device comparison (reported, not judged); the conversion stays verified on the Mac." if cross else "")
                  + (" Rows that differ: " + "; ".join(other) if other else ""), ""]
    if c["mask_compare"] and c["mask_compare"].get("clips"):
        lines += mask_compare_lines(c["mask_compare"])
    lines += session_f_lines(report)

    proj = report["projections"]
    if proj:
        lines += ["## Projected GPU time and cost", "", f"RE time on the H100: {proj['re_seconds']:.1f} s. Price ${fmt(proj['h100_price_per_hour'])}/hr.", ""]
        lines += table(["Phase", "Planning RE", "GPU-hours (RE x RE time)", "$ (RE)", "GPU-hours (by condition)", "$ (by condition)"],
                       [[ph, v["planning_re"], v["gpu_hours_by_re"], v["usd_by_re"], v["gpu_hours_by_condition"], v["usd_by_condition"]] for ph, v in proj["phases"].items()])
        lines += ["", "Run times used: " + "; ".join(f"{k} {v['seconds']:.1f} s ({v['basis']})" for k, v in proj["run_seconds"].items()) + ".", ""]
        mk = proj["masks"]
        if mk["phases"]:
            lines += table(["Masks", "References", "Slice hours", "$"], [[ph, ", ".join(f"{k} {n}" for k, n in v["references"].items()), v["slice_hours"], v["usd"]] for ph, v in mk["phases"].items()])
            lines += ["", "Seconds per reference: " + "; ".join(f"{k} {v['seconds']:.1f} ({v['basis']})" for k, v in mk["seconds_per_reference"].items()) + ".", ""]
            finding = detectron2_finding(report)
            if finding:
                lines += [f"**Finding:** {finding}", ""]
        lines += [proj["note"] + ".", ""]

    diag = report.get("diagnostics") or {}
    if diag:
        lines += ["## Diagnostics", "", "### After Session B", ""]
        hd = diag.get("hsa_dense")
        if hd:
            for pid, r in hd["prompts"].items():
                lines.append(f"- {pid}, iteration 0 in one process: {r['calls'][0]} module calls each; first differing call: {r['first_differing_call'] if r['first_differing_call'] is not None else 'none'}. "
                             "Dense and HSA outputs by attention backend: " + "; ".join(f"{k} {'equal' if v.get('equal') else ('error' if 'error' in v else 'differ')}" for k, v in r["by_attention_backend"].items()) + ".")
            inv = hd.get("sdpa_row_invariance") or {}
            lines.append("- Attention on identical tensors, every query vs a subset: " + "; ".join(f"{k} {'row-invariant' if v.get('row_invariant') else 'not row-invariant'} (max diff {fmt(v.get('max_abs_diff'))})" for k, v in inv.items()) + ".")
            fp = hd.get("flash_pinned_smoke_2") or {}
            if fp:
                lines.append(f"- Flash pinned, smoke-2, full trajectories: HSA all-full vs dense bitwise {fmt(fp['hsa_all_full_vs_dense']['bitwise_equal'])}; C3 fast path vs oracle bitwise {fmt(fp['c3_fast_vs_oracle']['bitwise_equal'])}.")
        xp = diag.get("crossproc")
        if xp:
            for mode, r in xp.items():
                if "bitwise_equal" in r:
                    first = "none" if r["first_divergent_step"] is None else r["first_divergent_step"]
                    lines.append(f"- Across processes ({r['label']}, attention {', '.join(r['backends'])}): bitwise {fmt(r['bitwise_equal'])}, max abs diff {fmt(r['max_abs_diff'])}, first divergent step {first}; {r['seconds']:.0f} s.")
        lines.append("")
        if diag.get("mask_threads"):
            body = (((report.get("projections") or {}).get("masks") or {}).get("seconds_per_reference") or {}).get("body", {}).get("seconds")
            lines += mask_threads_lines(diag["mask_threads"], diag.get("mask_threads_mac"), diag.get("mask_roi_align"), body)
    w = report["warnings"]
    lines += ["## Warnings", "", f"Operations without a deterministic implementation (warn-only mode): {len(w['deterministic_algorithm_warnings'])}."]
    lines += [f"- {d['message']} ({d['where']}; steps {', '.join(d['steps'])})" for d in w["deterministic_algorithm_warnings"]]
    lines += [f"Other warnings: {len(w['all']) - len(w['deterministic_algorithm_warnings'])} distinct (listed in the JSON).", ""]
    lines += ["Saved inputs, references, sheets, masks and logs are not versioned; their SHA-256 hashes are in `smoke_test.sha256`.", ""]
    return "\n".join(lines)


def write_report(ctx) -> int:
    report = build(ctx)
    base = ctx.out.parent
    stem = ctx.out.name
    smoke.write_json(base / f"{stem}.json", report)
    (base / f"{stem}.md").write_text(markdown(report))
    files = sorted(p for p in ctx.out.rglob("*") if p.is_file() and not p.name.endswith(".tmp"))
    (base / f"{stem}.sha256").write_text("".join(f"{smoke.file_sha256(p)}  {p.relative_to(base)}\n" for p in files))
    print(f"[smoke] report written to {base / (stem + '.md')}")
    return 0
