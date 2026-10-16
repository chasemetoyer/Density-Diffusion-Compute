"""The blinded visual check: the selection before results, hidden labels, random sides, a separate key, unblinding."""

import csv

import numpy as np
import pytest

from density import video, visual_check
from density.run import manifest as M
from density.run import plan
from density.run.store import Run, write_json_atomic


def frozen():
    return {"blocks": {"primary": {"face": list(range(1, 37)), "manipulation": list(range(101, 137))}}}


def test_selection_is_fixed_stratified_and_recorded_with_the_freeze():
    a, b = visual_check.select(frozen(), "f" * 64), visual_check.select(frozen(), "f" * 64)
    assert a["pairs"] == b["pairs"] and len(a["pairs"]) == 30 and a["frozen_sha256"] == "f" * 64
    assert sum(p["stratum"] == "face" for p in a["pairs"]) == 15 and len({p["prompt_id"] for p in a["pairs"]}) == 30
    from density import prompts

    assert all(p["seed"] in prompts.seeds(p["prompt_id"]) for p in a["pairs"])


def fake_run(tmp_path, pairs):
    run = Run(tmp_path / "run")
    rows = [{"id": str(p["prompt_id"]), "stratum": p["stratum"], "block": "primary", "text": f"text {p['prompt_id']}"} for p in pairs]
    write_json_atomic(run.path("inputs", "prompts.json"), {"rows": rows})
    jobs, rng = [], np.random.default_rng(0)
    for p in pairs:
        ref = M.Job("reference", p["prompt_id"], p["stratum"], "primary", p["seed"], "sh8g6", "C1")
        for c in ("C3", "C4"):
            job = M.condition_job(ref, plan.GridItem(c, "HSA-50", 3.0 if c == "C4" else None), "primary seed 1", "m")
            jobs.append(job)
            frames = np.full((3, 32, 48, 3), 40 if c == "C3" else 200, np.uint8) + rng.integers(0, 3, (3, 32, 48, 3), dtype=np.uint8)
            rec = video.encode(frames, run.job_dir(job) / "video.mp4")
            write_json_atomic(run.job_dir(job) / "record.json", {"video": {**rec, "path": "video.mp4"}, "fidelity": {"masked_lpips": 0.1 if c == "C4" else 0.2}})
    run.write_manifest({"run": "run", "kind": "main", "dry_run": False, "prompts_csv_sha256": "x", "jobs": [j.to_dict() for j in jobs]}, "test")
    write_json_atomic(run.path("visual_check", "selection.json"), {"pairs": pairs})
    return run


def test_build_hides_labels_randomizes_sides_and_keeps_the_key_apart(tmp_path):
    pairs = visual_check.select(frozen(), "x")["pairs"][:4]
    run = fake_run(tmp_path, pairs)
    out = visual_check.build(run, tmp_path / "view", tmp_path / "key")
    assert out["items"] == 4
    names = sorted(p.name for p in (tmp_path / "view").rglob("*"))
    assert all("C3" not in n and "C4" not in n for n in names)
    sheet = (tmp_path / "view" / "responses.csv").read_text()
    assert "C3" not in sheet and "C4" not in sheet and not (tmp_path / "view" / "key.json").exists()
    key = visual_check.read_json(tmp_path / "key" / "key.json")["items"]
    for name, k in key.items():
        frames = video.decode_rgb(tmp_path / "view" / "items" / f"{name}.mp4")
        left = frames[:, :, :48].astype(int).mean()
        assert (left > 120) == (k["A"] == "C4")  # C4's stand-in frames are bright
    with pytest.raises(RuntimeError, match="second key"):
        visual_check.build(run, tmp_path / "view2", tmp_path / "key")


def test_unblind_maps_answers_through_the_key(tmp_path):
    pairs = visual_check.select(frozen(), "x")["pairs"][:2]
    run = fake_run(tmp_path, pairs)
    visual_check.build(run, tmp_path / "view", tmp_path / "key")
    key = visual_check.read_json(tmp_path / "key" / "key.json")["items"]
    path = tmp_path / "view" / "responses.csv"
    rows = list(csv.DictReader(l for l in path.read_text().splitlines() if not l.startswith("#")))
    for r in rows:  # prefer C3 on the first item and C4 on the second
        want = "C3" if r["item"] == "item_01" else "C4"
        r["choice"] = "A" if key[r["item"]]["A"] == want else "B"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=visual_check.RESPONSE_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    result = visual_check.unblind(run, path, tmp_path / "key" / "key.json")
    assert result["c3_preferred"] == 1 and result["c4_preferred"] == 1 and result["agreement"] == 0.5
    assert [d["item"] for d in result["disagreements"]] == ["item_01"]
