"""The blinded visual check of 30 C3/C4 pairs (protocol section 7). It looks for metric failures; it does not measure
viewer preference.

1. ``select`` runs at the main run's freeze, before any condition exists: 15 face and 15 manipulation prompt-seeds,
   drawn from the frozen primary prompts with a fixed seed (one seed per chosen prompt, also drawn). The selection
   records the freeze's hash.
2. ``build`` (on the Mac, after the conditions are done): each pair becomes one side-by-side video, labelled only A
   and B, in a shuffled order; which side is C4, and the order, come from the operating system's random source, so
   nothing in the code or the selection can reveal them. The answer sheet lists the items with their prompt text. The
   key goes to a separate folder.
3. Chase watches the items and fills responses.csv (A, B or same, and a note), then ``unblind`` joins his answers
   with the key and the masked LPIPS of both videos, and lists every pair where the metric and his judgment disagree.
"""

import csv
import json
import secrets
from pathlib import Path

import numpy as np

from density import prompts as prompt_set
from density import video
from density.run.store import Run, file_sha256, read_json, utc_now, write_json_atomic

PAIRS_PER_STRATUM = 15
SELECTION_SEED = 20261008  # registered by Chase, 2026-10-07 (decision 5)
GAP = 16  # black pixels between the two videos
RESPONSE_COLUMNS = ["item", "prompt", "choice", "note"]


def select(frozen: dict, frozen_sha256: str) -> dict:
    rng = np.random.default_rng(SELECTION_SEED)
    pairs = []
    for stratum in ("face", "manipulation"):
        ids = sorted(frozen["blocks"]["primary"][stratum])
        for pid in sorted(int(x) for x in rng.choice(ids, size=min(PAIRS_PER_STRATUM, len(ids)), replace=False)):  # 15 of 36; all of a smaller dry run
            seed = prompt_set.seeds(pid)[int(rng.integers(0, 2))]
            pairs.append({"prompt_id": pid, "stratum": stratum, "seed": seed})
    return {"pairs": pairs, "frozen_sha256": frozen_sha256, "utc": utc_now(), "seed": SELECTION_SEED,
            "rule": "15 face and 15 manipulation prompts drawn without replacement from the frozen primary prompts, then one of each prompt's two seeds"}


def pair_jobs(run: Run, pair: dict) -> dict[str, str]:
    """The C3 and C4 condition jobs of a selected prompt-seed (HSA-50, the registered lambda)."""
    found = {}
    for job in run.jobs():
        if job.kind == "condition" and job.prompt_id == pair["prompt_id"] and job.seed == pair["seed"] and job.preset == "HSA-50" and job.condition in ("C3", "C4") \
                and job.grid_block in ("primary seed 1", "primary seed 2"):
            found[job.condition] = job.id
    if set(found) != {"C3", "C4"}:
        raise RuntimeError(f"pair {pair}: found {found}")
    return found


def labelled(frames: np.ndarray, letter: str) -> np.ndarray:
    from PIL import Image, ImageDraw

    out = frames.copy()
    for i, frame in enumerate(out):
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        draw.rectangle([6, 6, 26, 24], fill=(0, 0, 0))
        draw.text((12, 8), letter, fill=(255, 255, 255))
        out[i] = np.asarray(image)
    return out


def side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    t, h, _, _ = left.shape
    gap = np.zeros((t, h, GAP, 3), np.uint8)
    return np.concatenate([labelled(left, "A"), gap, labelled(right, "B")], axis=2)


def build(run: Run, out_dir: Path, key_dir: Path) -> dict:
    """Write the items, the answer sheet and, separately, the key. Refuses to overwrite an existing key."""
    selection = read_json(run.path("visual_check", "selection.json"))
    if (key_dir / "key.json").exists():
        raise RuntimeError(f"{key_dir / 'key.json'} exists; a second build would make a second key")
    texts = {int(r["id"]): r["text"] for r in read_json(run.path("inputs", "prompts.json"))["rows"]}
    order = list(range(len(selection["pairs"])))
    secrets.SystemRandom().shuffle(order)
    key, sheet = {}, []
    for item, index in enumerate(order, 1):
        pair = selection["pairs"][index]
        jobs = pair_jobs(run, pair)
        frames = {}
        for condition, job_id in jobs.items():
            rec = run.record(job_id)
            decoded = video.decode_rgb(run.job_dir(job_id) / rec["video"]["path"])
            if video.frames_sha256(decoded) != rec["video"]["decoded_frames_sha256"]:
                raise RuntimeError(f"{job_id}: its MP4 decodes to frames that differ from its record")
            frames[condition] = decoded
        a = "C4" if secrets.randbits(1) else "C3"
        b = "C3" if a == "C4" else "C4"
        name = f"item_{item:02d}"
        video.encode(side_by_side(frames[a], frames[b]), out_dir / "items" / f"{name}.mp4")
        key[name] = {**pair, "A": a, "B": b, "jobs": jobs}
        sheet.append({"item": name, "prompt": texts[pair["prompt_id"]], "choice": "", "note": ""})
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "responses.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESPONSE_COLUMNS)
        w.writeheader()
        w.writerows(sheet)
        f.write("# choice: A, B or same, for the video that better keeps faces, hands and held objects intact; note: what you saw\n")
    write_json_atomic(key_dir / "key.json", {"items": key, "selection_sha256": file_sha256(run.path("visual_check", "selection.json")), "utc": utc_now()})
    return {"items": len(key), "out": str(out_dir), "key": str(key_dir / "key.json")}


def unblind(run: Run, responses: Path, key_path: Path) -> dict:
    key = read_json(key_path)["items"]
    with open(responses, newline="", encoding="utf-8") as f:
        answers = {r["item"]: r for r in csv.DictReader(line for line in f if not line.startswith("#"))}
    rows, disagreements = [], []
    for name, k in sorted(key.items()):
        answer = (answers.get(name, {}).get("choice") or "").strip().upper()
        if answer not in ("A", "B", "SAME"):
            raise ValueError(f"{name}: choice must be A, B or same, got {answer!r}")
        preferred = None if answer == "SAME" else k[answer]
        lpips = {c: run.record(j)["fidelity"]["masked_lpips"] for c, j in k["jobs"].items()}
        metric = "C4" if lpips["C4"] < lpips["C3"] else "C3" if lpips["C3"] < lpips["C4"] else None
        row = {"item": name, "prompt_id": k["prompt_id"], "seed": k["seed"], "stratum": k["stratum"], "preferred": preferred, "metric_prefers": metric,
               "masked_lpips": lpips, "note": answers[name].get("note", "")}
        rows.append(row)
        if preferred is not None and metric is not None and preferred != metric:
            disagreements.append(row)
    judged = [r for r in rows if r["preferred"] is not None and r["metric_prefers"] is not None]
    return {"items": rows, "agreement": sum(r["preferred"] == r["metric_prefers"] for r in judged) / len(judged) if judged else None,
            "judged": len(judged), "same": sum(r["preferred"] is None for r in rows), "disagreements": disagreements,
            "c4_preferred": sum(r["preferred"] == "C4" for r in rows), "c3_preferred": sum(r["preferred"] == "C3" for r in rows)}


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("run", type=Path)
    b.add_argument("--out", type=Path, required=True, help="the items and answer sheet, for viewing")
    b.add_argument("--key", type=Path, required=True, help="a separate folder for the key; don't open it before unblinding")
    u = sub.add_parser("unblind")
    u.add_argument("run", type=Path)
    u.add_argument("--responses", type=Path, required=True)
    u.add_argument("--key", type=Path, required=True)
    u.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    run = Run(args.run)
    if args.command == "build":
        print(json.dumps(build(run, args.out, args.key), indent=1))
    else:
        result = unblind(run, args.responses, args.key / "key.json")
        write_json_atomic(args.out, result)
        print(f"[visual check] agreement {result['agreement']}, {len(result['disagreements'])} disagreements; written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
