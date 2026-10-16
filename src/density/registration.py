"""The dated registration (protocol section 10, step 7): every registered value, each with the file that holds it and
the commit that last changed that file, plus the slots the pilot fills.

    python -m density.registration build --out DIR [--pilot RUN] [--draft]

writes DIR/registration.json (read by ``python -m density.run init --kind main``) and DIR/registration.md for Chase.
Without --draft it refuses a dirty tree, a pilot that has not finished its rules, or any pilot stop. The pilot slots are
the setting, lambda, the reserve sizes (Chase's to set from the pass rates; recorded as pending until he does), and the
pilot's evidence: the setting table, the lambda table, the no-op counts, the C2 ratio, the pass rates and the
reachability check.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from density.run.store import file_sha256, jsonable, read_json, utc_now, write_json_atomic

ROOT = Path(__file__).resolve().parents[2]


def last_commit(path: str) -> str | None:
    out = subprocess.run(["git", "log", "-1", "--format=%H", "--", path], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return out or None


def entries() -> list[dict]:
    """(section, name, value, file) for every registered value, read from the code that enforces it."""
    from density import allocation, analysis, attention, determinism, noise, pilot, prompt_draft, prompts, text_encoding, tolerances, video, visual_check, volume, wan
    from density.hsa import NUM_STEPS, PRESETS
    from density.masks import registered as masks
    from density.metrics import registered as metrics
    from density.run import plan

    def e(section, name, value, file):
        return {"section": section, "name": name, "value": jsonable(value), "file": file}

    mask_names = [n for n in dir(masks) if n.isupper()]
    metric_names = [n for n in dir(metrics) if n.isupper()]
    return [
        e("5 schedules", "steps", NUM_STEPS, "src/density/hsa.py"),
        e("5 schedules", "HSA presets (high/low tokens, window, low steps)", {k: [p.high_tokens, p.low_tokens, list(p.window), p.low_steps] for k, p in PRESETS.items()}, "src/density/hsa.py"),
        e("5 schedules", "shift/guidance candidates and tie rule", {"candidates": plan.SETTINGS, "tie": plan.TIE_SETTING}, "src/density/run/plan.py"),
        e("2 allocation", "need floor", allocation.NEED_FLOOR, "src/density/allocation.py"),
        e("3 allocation", "ROI threshold b", allocation.ROI_THRESHOLD, "src/density/allocation.py"),
        e("10 pilot", "lambda grid", allocation.LAMBDA_GRID, "src/density/allocation.py"),
        e("6 prompts", "block counts", {f"{b} {s}": n for (b, s), n in prompts.BLOCK_COUNTS.items()}, "src/density/prompts.py"),
        e("6 prompts", "seeds", "1000 + id and 2000 + id (equation 14), ids 1-999", "src/density/prompts.py"),
        e("6 prompts", "T5 token limit", prompts.MAX_TOKENS, "src/density/prompts.py"),
        e("6 prompts", "drafting procedure", {"seed": prompt_draft.DRAW_SEED, "stratum_order": prompt_draft.STRATUM_ORDER, "counts": prompt_draft.COUNTS,
                                              "rewrite_note": prompt_draft.REWRITE_NOTE, "readings": prompt_draft.READINGS,
                                              "second_pass": prompt_draft.SECOND_PASS}, "src/density/prompt_draft.py"),
        e("6 prompts", "validator scopes", "the holdout block at the pilot's init; the whole file, with the registered reserve sizes, at the main run's", "src/density/prompts.py"),
        e("6 noise", "starting noise", noise.__doc__.strip().splitlines()[0], "src/density/noise.py"),
        e("5 model", "text encoding", {"dtype": str(text_encoding.T5_DTYPE), "tokens": text_encoding.MAX_SEQUENCE_LENGTH, "ftfy": text_encoding.FTFY_REGISTERED, "negative_prompt": text_encoding.NEGATIVE_PROMPT}, "src/density/text_encoding.py"),
        e("5 model", "generation attention", [b.name for b in attention.GENERATION_BACKENDS], "src/density/attention.py"),
        e("5 model", "VAE fp32 convolutions", wan.VAE_CONV_FP32_PRECISION, "src/density/wan.py"),
        e("11 correctness", "tolerances", tolerances.FROZEN, "src/density/tolerances.py"),
        e("13 execution", "per-pod canary", "smoke-1 dense C1 reproduced bitwise before any H100 job (density.canary.REGISTERED)", "src/density/canary.py"),
        e("13 execution", "strict deterministic mode", "determinism.configure(strict=True) in the pilot and main run", "src/density/determinism.py"),
        e("12 storage", "volume stop (bytes, du allocated)", volume.STOP_BYTES, "src/density/volume.py"),
        e("12 storage", "volume measurement in a session", {"full_du": "at the session's start, every 30 minutes, and when the running total reaches the stop",
                                                            "full_check_seconds": volume.FULL_CHECK_SECONDS,
                                                            "running_total": "after each job, the allocated size (st_blocks x 512) of every file it wrote"},
          "src/density/volume.py"),
        e("4 masks", "mask canary", "configs/mask_canary.json", "configs/mask_canary.json"),
        *[e("4 masks", n, getattr(masks, n), "src/density/masks/registered.py") for n in mask_names],
        *[e("7 metrics", n, getattr(metrics, n), "src/density/metrics/registered.py") for n in metric_names],
        e("7 metrics", "fixed MP4 encoding", video.ENCODING, "src/density/video.py"),
        e("7 metrics", "encode canary", read_json(ROOT / "configs" / "encode_canary.json")["decoded_frames_sha256"], "configs/encode_canary.json"),
        e("7 metrics", "lossless copies of conditions", video.LOSSLESS, "src/density/video.py"),
        e("13 execution", "metric canary", "recorded at each machine's first pilot session from two fresh processes, then required bit for bit", "src/density/metric_canary.py"),
        e("10 pilot", "reserve sizing", {"coverage": pilot.RESERVE_COVERAGE, "primary_per_stratum": pilot.PRIMARY_PER_STRATUM,
                                         "formula": "NegBin(36, p^2) at the holdout's one-seed pass rate; sizes set by Chase from the estimate"}, "src/density/pilot.py"),
        e("7 metrics", "metric weight files", "src/density/metrics/weights.py (size and SHA-256 of every file)", "src/density/metrics/weights.py"),
        e("9 analysis", "bootstrap", {"resamples": analysis.RESAMPLES, "seed": analysis.BOOTSTRAP_SEED, "per_stratum": analysis.PROMPTS_PER_STRATUM}, "src/density/analysis.py"),
        e("9 analysis", "decision", {"floor": analysis.FLOOR, "alpha": analysis.ALPHA, "J": analysis.J, "go_median": analysis.GO_MEDIAN, "no_go_upper": analysis.NO_GO_UPPER, "limits": analysis.LIMITS}, "src/density/analysis.py"),
        e("9 analysis", "analysis freeze", read_json(analysis.FREEZE_FILE)["files"] if analysis.FREEZE_FILE.exists() else "not yet frozen", "configs/analysis_freeze.json"),
        e("12 grid", "main blocks", [{**b, "items": [vars(i) for i in b["items"]]} for b in plan.main_blocks("registered lambda")], "src/density/run/plan.py"),
        e("12 grid", "subset rule", {"seed": plan.SUBSET_SEED, "sizes": plan.SUBSETS}, "src/density/run/plan.py"),
        e("7 metrics", "visual check selection", {"seed": visual_check.SELECTION_SEED, "per_stratum": visual_check.PAIRS_PER_STRATUM}, "src/density/visual_check.py"),
        e("13 execution", "deterministic warnings are errors", determinism.DETERMINISTIC_MESSAGE, "src/density/determinism.py"),
    ]


def pilot_slots(pilot_run: Path | None) -> dict:
    if pilot_run is None:
        return {"status": "pending: no pilot run given"}
    rules_path = Path(pilot_run) / "pilot" / "rules.json"
    if not rules_path.exists():
        return {"status": f"pending: {rules_path} does not exist"}
    rules = read_json(rules_path)
    return {
        "status": "stopped" if rules["stops"] else "complete",
        "stops": rules["stops"], "rules_sha256": file_sha256(rules_path), "pilot_frozen_sha256": rules["frozen_sha256"],
        "setting": rules["setting"]["setting"], "setting_table": {k: rules["setting"][k] for k in ("scores", "per_stratum", "tie")},
        "lambda": rules["lambda"]["lambda"], "lambda_table": rules["lambda"]["table"], "no_op": rules["no_op"], "c2": rules["c2"],
        "pass_rates": rules["pass_rates"], "reachability": rules["reachability"], "excluded_holdout": rules["excluded"],
        "reserve_sizes": "pending: Chase sets them from the pass rates' estimate",
    }


def parse_reserve_sizes(text: str | None) -> dict[str, int] | None:
    """``face=24,manipulation=40,body=10,control=10`` (Chase's sizes, set after the pilot from its estimate)."""
    if not text:
        return None
    sizes = {k.strip(): int(v) for k, v in (item.split("=") for item in text.split(","))}
    from density.prompts import STRATA

    if set(sizes) != set(STRATA) or min(sizes.values()) < 0:
        raise ValueError(f"reserve sizes need every stratum {STRATA}: {sizes}")
    return sizes


def build(out: Path, pilot_run: Path | None, draft: bool, reserve_sizes: dict[str, int] | None = None) -> dict:
    from density.checks import euler

    code = euler.git_state()
    pilot = pilot_slots(pilot_run)
    if not draft:
        if code["git_dirty"]:
            raise RuntimeError("commit first: a registration records the code it was built from")
        if pilot["status"] != "complete":
            raise RuntimeError(f"the pilot is not complete: {pilot['status']} {pilot.get('stops', '')}")
    rows = entries()
    for r in rows:
        r["commit"] = last_commit(r["file"])
        r["file_sha256"] = file_sha256(ROOT / r["file"]) if (ROOT / r["file"]).exists() else None
    if not draft and reserve_sizes is None:
        raise RuntimeError("the registration needs the reserve sizes Chase sets from the pilot's estimate (--reserve-sizes)")
    if reserve_sizes is not None:
        pilot = {**pilot, "reserve_sizes": reserve_sizes}
    registration = {"utc": utc_now(), "draft": draft, "code": code,
                    "values": {"setting": pilot.get("setting"), "lambda": pilot.get("lambda"), "reserve_sizes": reserve_sizes},
                    "pilot": pilot, "registered": rows}
    write_json_atomic(out / "registration.json", registration)
    registration["sha256"] = file_sha256(out / "registration.json")
    (out / "registration.md").write_text(markdown(registration))
    return registration


def markdown(reg: dict) -> str:
    lines = [f"# Registration{' (draft)' if reg['draft'] else ''}", "", f"Built {reg['utc']} at commit `{reg['code']['git_sha']}`"
             f"{' (dirty tree)' if reg['code']['git_dirty'] else ''}.", "", "## Pilot", ""]
    p = reg["pilot"]
    if p.get("status") in ("complete", "stopped"):
        lines += [f"- Status: {p['status']}" + (f"; stops: {'; '.join(p['stops'])}" if p["stops"] else ""),
                  f"- Setting: {p['setting']} (scores {json.dumps(p['setting_table']['scores'])}; tie {p['setting_table']['tie']})",
                  f"- Lambda: {p['lambda']}", f"- C2 ratio: {p['c2']['ratio']:.3f}", f"- No-op at every lambda: {p['no_op']['unchanged_at_every_lambda']}",
                  f"- Reserve sizes: {p['reserve_sizes']}", ""]
    else:
        lines += [f"- {p['status']}", ""]
    lines += ["## Registered values", "", "| Section | Value | File | Commit |", "| --- | --- | --- | --- |"]
    for r in reg["registered"]:
        value = json.dumps(r["value"]) if not isinstance(r["value"], str) else r["value"]
        value = value if len(value) <= 160 else value[:157] + "..."
        lines.append(f"| {r['section']} | **{r['name']}**: {value.replace('|', '/')} | `{r['file']}` | `{(r['commit'] or 'uncommitted')[:12]}` |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--pilot", type=Path)
    b.add_argument("--draft", action="store_true")
    b.add_argument("--reserve-sizes", help="Chase's sizes after the pilot, e.g. face=24,manipulation=40,body=10,control=10")
    args = parser.parse_args(argv)
    reg = build(args.out, args.pilot, args.draft, parse_reserve_sizes(args.reserve_sizes))
    print(f"[registration] {len(reg['registered'])} values; pilot {reg['pilot']['status']}; written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
