"""Mask-pipeline logic (Milestone 4, T19-T25 and the T28 negative controls) on synthetic masks and mocked detectors."""

import subprocess
import sys
from pathlib import Path

import pytest

from density.checks import masks as checks

SRC = Path(__file__).resolve().parents[1] / "src"


def assert_rows(rows):
    failures = [r for r in rows if not r["passed"]]
    assert not failures, failures


@pytest.mark.parametrize(
    "check",
    [checks.check_t19, checks.check_t20, checks.check_t21, checks.check_t22, checks.check_t23, checks.check_t24, checks.check_t25],
    ids=["T19", "T20", "T21", "T22", "T23", "T24", "T25"],
)
def test_logic(check):
    assert_rows(check())


def test_extrapolated_contact_interval_fails_t20():
    assert not all(r["passed"] for r in checks.check_t20(checks.extrapolating_intervals))


def test_body_mask_as_manipulation_roi_fails_t23():
    assert not all(r["passed"] for r in checks.check_t23(checks.body_as_manipulation))


def test_only_the_wilor_module_imports_ultralytics():
    importers = sorted(
        str(p.relative_to(SRC)) for p in (SRC / "density").rglob("*.py")
        if any(line.strip().startswith(("import ultralytics", "from ultralytics")) for line in p.read_text().splitlines())
    )
    assert importers == ["density/masks/wilor.py"]
    code = (
        "import sys, density.masks.registered, density.masks.geometry, density.masks.contact, density.masks.roi, "
        "density.masks.measurement, density.masks.eligibility, density.masks.tracking, density.masks.weights, "
        "density.masks.hands23, density.checks.masks; print('ultralytics' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "False"


@pytest.mark.skipif(not (Path(__file__).resolve().parents[1] / "weights" / "masks").exists(), reason="mask-pipeline weights not downloaded")
def test_runtime_weights_load_as_data_only():
    result = checks.check_data_only_loading("cpu")
    assert result["passed"], result


EXPECTED_CONTROL_FAILURES = {
    "today's ungated rule": ["T22 hands crossing", "T22 stray false positive", "T22 lost track"],
    "side-anchored without the IoU >= 0.5 override": ["T22 side flip"],
    "gate without the lost-track exception": ["T22 lost track"],
    "without the bound-detection rule": ["T22 termination"],
}


@pytest.mark.parametrize("name", sorted(EXPECTED_CONTROL_FAILURES))
def test_association_controls_fail_their_t22_cases(name):
    rows = checks.check_t22_association(checks.ASSOCIATION_CONTROLS[name]())
    assert [r["test"] for r in rows if not r["passed"]] == EXPECTED_CONTROL_FAILURES[name]


def test_storage_round_trip(tmp_path):
    import numpy as np

    from density.masks.storage import load_masks, load_tracks, save_reference

    rng = np.random.default_rng(0)
    h = rng.random((81, 48, 64)) < 0.1
    hashes = save_reference(tmp_path, {"H": h}, {"ROI": np.zeros((21, 3, 4))}, {"note": 1}, tracks={"H": {0: h, 3: ~h}}, backfill={"H": h & ~h})
    assert np.array_equal(load_masks(tmp_path)["H"], h)
    tracks = load_tracks(tmp_path)["H"]
    assert sorted(tracks) == [0, 3] and np.array_equal(tracks[3], ~h)
    assert set(hashes) == {"masks.json", "coverage.npz", "region_sha256", "track_sha256", "metadata.json"}


def test_audit_csv_keeps_sign_off_only_for_unchanged_sheets(tmp_path):
    import csv

    from density.masks.audit import AUDIT_COLUMNS, write_audit_csv

    path = tmp_path / "audit.csv"
    rows = [{"clip": c, "seed": "", "contact_sheet": f"{c}.png", "contact_sheet_sha256": h} for c, h in (("a", "1"), ("b", "2"))]
    write_audit_csv(rows, path)
    with open(path, newline="") as f:
        signed = list(csv.DictReader(line for line in f if not line.startswith("#")))
    for r in signed:
        r.update(reviewer_decision="accept", reviewer="CM", review_date="2026-10-06")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=AUDIT_COLUMNS)
        writer.writeheader()
        writer.writerows(signed)
    rows[1]["contact_sheet_sha256"] = "3"  # sheet b changed
    write_audit_csv(rows, path)
    with open(path, newline="") as f:
        kept = {r["clip"]: r["reviewer_decision"] for r in csv.DictReader(line for line in f if not line.startswith("#"))}
    assert kept == {"a": "accept", "b": ""}


def test_contact_sheet_is_deterministic_and_hatches_backfill(tmp_path):
    import numpy as np
    from PIL import Image

    from density.masks.audit import COLORS, contact_sheet

    frames = np.full((81, 480, 832, 3), 60, dtype=np.uint8)
    h = np.zeros((81, 480, 832), dtype=bool)
    h[:, 100:200, 100:300] = True
    backfill = np.zeros_like(h)
    backfill[:16] = h[:16]
    paths = [contact_sheet(frames, {"H": h}, None, tmp_path / f"s{i}.png", "test", backfill={"H": backfill}, labels={0: [(100, 100, "H0 R*", COLORS["H"])]}) for i in range(2)]
    assert paths[0].read_bytes() == paths[1].read_bytes()
    sheet = np.asarray(Image.open(paths[0]).convert("RGB"))
    tile0 = sheet[48 + 60 : 48 + 90, 80:140]  # inside frame 0's hatched mask
    tile3 = sheet[48 + 60 : 48 + 90, 3 * 416 + 80 : 3 * 416 + 140]  # frame 24: solid fill
    green = np.all(tile0 == COLORS["H"], axis=-1).mean()
    assert 0.2 < green < 0.5 and not np.all(tile3 == COLORS["H"], axis=-1).any()


def test_region_events_record_their_detection():
    import numpy as np

    from density.masks.regions import Region
    from density.masks.tracking import TrackResult

    m = np.zeros((81, 8, 8), dtype=bool)
    m[:, 2:5, 2:5] = True
    event = {"frame": 3, "detection": 1, "action": "new_track", "track": 0, "iou": 0.0, "side": "right", "track_side": "right", "round": 1}
    result = TrackResult({0: m}, {0: {3: np.array([2.0, 2, 5, 5])}}, [event], {0: "right"})
    found = {3: [{"box": np.array([0.0, 0, 1, 1]), "score": 0.9}, {"box": np.array([2.0, 2, 5, 5]), "score": 0.31}]}
    e = Region(m, result, found).summary()["events"][0]
    assert e["box"] == [2.0, 2.0, 5.0, 5.0] and e["score"] == 0.31 and e["action"] == "new_track"


def test_keeping_every_hand_track_fails_the_confirmation_check():
    rows = checks.check_t22_confirmation(checks.keep_every_hand_track)
    assert [r["test"] for r in rows if not r["passed"]] == [
        "T22 hand tracks without a Hands23 confirmation are dropped",
        "T22 confirmation counts only keyframes where the track has a mask",
        "T22 confirmation ignores the side label and needs IoU >= 0.5",
    ]
