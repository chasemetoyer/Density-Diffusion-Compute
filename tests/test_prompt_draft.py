"""The registered prompt-drafting procedure: de-duplicated candidates, the seeded shuffle, the order-faithful walk that
fills the blocks, and the files it writes."""

import csv

import numpy as np
import pytest

from density import prompt_draft as D
from density import prompts

SOURCES = {"a": {"protocol_strata": ["face"]}, "b": {"protocol_strata": ["face", "body"]}, "c": {"protocol_strata": ["manipulation", "body"]},
           "d": {"protocol_strata": ["control"]}}
LINES = {"a": [f"face prompt {i}" for i in range(1, 9)], "b": ["Face prompt 1.", "body prompt x", "shared prompt"],
         "c": [f"hand prompt {i}" for i in range(1, 7)] + ["shared prompt"], "d": [f"scene {i}" for i in range(1, 6)]}


@pytest.fixture
def small(monkeypatch):
    monkeypatch.setattr(prompts, "sources", lambda: SOURCES)
    monkeypatch.setattr(D, "COUNTS", {"face": (("holdout", 1), ("primary", 1), ("reserve", 1)), "manipulation": (("holdout", 1), ("primary", 1), ("reserve", 1)),
                                      "body": (("secondary", 1), ("reserve", 1)), "control": (("control", 1), ("reserve", 1))})


def test_candidates_are_deduplicated_across_lists(small):
    c = D.candidates(LINES)
    face = {x["source_id"]: x for x in c["face"]}
    assert "b:1" not in face and face["a:1"]["also_in"] == ["b:1"]  # "Face prompt 1." normalizes to a's first prompt
    assert {x["source_id"] for x in c["body"]} >= {"b:2", "b:3", "c:1"} and "c:7" not in {x["source_id"] for x in c["body"]}  # c:7 repeats b:3


def test_the_shuffle_is_fixed_by_the_registered_seed(small):
    c = D.candidates(LINES)["face"]
    a, b = D.shuffled("face", c), D.shuffled("face", c)
    assert [x["source_id"] for x in a] == [x["source_id"] for x in b] and [x["rank"] for x in a] == list(range(1, len(c) + 1))
    expected = np.random.default_rng([20261009, 0]).permutation(len(c))
    assert [x["source_id"] for x in a] == [c[i]["source_id"] for i in expected]
    assert D.DRAW_SEED == 20261009


def judge_all(stratum, cands, skip_every=3):
    out = {}
    for c in D.shuffled(stratum, cands):
        yes = c["rank"] % skip_every != 0
        out[c["rank"]] = {"stratum": stratum, "rank": str(c["rank"]), "source_id": c["source_id"], "eligible": "yes" if yes else "no",
                          "reason": "fits" if yes else "two people", "text": f"Draft {stratum} {c['rank']}: a person smiles in a close-up.",
                          "subject": "a person", "action": "smiles", "framing": "close-up"}
    return out


def test_the_walk_keeps_the_first_eligible_candidates_in_block_order(small):
    c = D.candidates(LINES)
    judgments = {s: judge_all(s, c[s]) for s in D.STRATUM_ORDER}
    for s in D.STRATUM_ORDER:  # judgments only up to the last candidate each stratum needs
        result = D.walk(LINES, {k: v for k, v in judgments.items()})
        needed = result[s]["log"][-1]["rank"] if not result[s]["missing"] else None
        if needed:
            judgments[s] = {r: j for r, j in judgments[s].items() if r <= needed}
    result = D.walk(LINES, judgments)
    face = result["face"]
    assert [k["block"] for k in face["kept"]] == ["holdout", "primary", "reserve"] and face["missing"] == 0
    assert all(e["rank"] % 3 == 0 for e in face["log"] if e["decision"] == "skipped")
    assert [e["rank"] for e in face["log"]] == list(range(1, len(face["log"]) + 1))  # in order, without gaps


def test_the_walk_stops_at_the_first_unjudged_candidate_and_refuses_misordered_judgments(small):
    c = D.candidates(LINES)
    j = judge_all("face", c["face"])
    gap = {r: v for r, v in j.items() if r != 2}
    r = D.walk(LINES, {"face": gap})
    assert r["face"]["next_rank"] == 2 and len(r["face"]["kept"]) == 1
    wrong = {**j, 1: {**j[1], "source_id": "a:999"}}
    with pytest.raises(ValueError, match="shuffled order"):
        D.walk(LINES, {"face": wrong})


def test_a_source_kept_in_an_earlier_stratum_is_skipped_later(small):
    c = D.candidates(LINES)
    face = judge_all("face", c["face"], skip_every=10**6)
    order = D.shuffled("face", c["face"])
    shared_rank = next(x["rank"] for x in order if x["source_id"] == "b:3")
    for r in face:  # only the shared prompt and the two after it are eligible, so face keeps it
        face[r]["eligible"] = "yes" if shared_rank <= r <= shared_rank + 2 else "no"
    body = judge_all("body", c["body"], skip_every=10**6)
    result = D.walk(LINES, {"face": face, "body": body})
    assert "b:3" in {k["source_id"] for k in result["face"]["kept"]}
    body_log = {e["source_id"]: e for e in result["body"]["log"]}
    assert body_log.get("b:3", {"reason": "kept in the face stratum"})["reason"] == "kept in the face stratum"
    assert "b:3" not in {k["source_id"] for k in result["body"]["kept"]}


def test_assemble_writes_a_valid_prompt_file_the_log_and_the_review(small, tmp_path, monkeypatch):
    monkeypatch.setattr(prompts, "BLOCK_COUNTS", {("holdout", "face"): 1, ("holdout", "manipulation"): 1, ("primary", "face"): 1,
                                                  ("primary", "manipulation"): 1, ("secondary", "body"): 1, ("control", "control"): 1})
    monkeypatch.setattr(prompts, "RESERVE_MINIMUMS", {"face": 1, "manipulation": 1, "body": 1, "control": 1})
    c = D.candidates(LINES)
    judgments = {}
    for s in D.STRATUM_ORDER:
        js = judge_all(s, c[s])
        for r, j in js.items():
            subject, action, framing = {"face": ("a person", "smiles", "close-up"), "manipulation": ("a person", "holds a cup", "close-up"),
                                        "body": ("a person", "jumps", "wide shot"), "control": ("a lake", "ripples", "wide shot")}[s]
            j.update(text=f"Draft {s} {r}: {subject} {action} in a {framing}.", subject=subject, action=action, framing=framing)
        judgments[s] = js
        needed = D.walk(LINES, judgments)[s]
        if not needed["missing"]:
            last = needed["log"][-1]["rank"]
            judgments[s] = {r: j for r, j in js.items() if r <= last}
    out = D.assemble(LINES, judgments, tmp_path / "prompts.csv", tmp_path / "review.csv", tmp_path / "log.csv")
    assert out["rows"] == 10
    report = prompts.validate(tmp_path / "prompts.csv", lines=LINES, complete=True)
    assert report["errors"] == [], report["errors"]
    with open(tmp_path / "review.csv", newline="") as f:
        review = list(csv.DictReader(f))
    assert [r["block"] for r in review[:2]] == ["holdout", "holdout"] and review[0]["source_text"] and review[0]["rewrite_note"] == D.REWRITE_NOTE
    log = (tmp_path / "log.csv").read_text()
    assert "face prompt" not in log and "skipped" in log  # the versioned log holds no source text


def test_a_partial_assembly_needs_an_exhausted_stratum(small, tmp_path):
    c = D.candidates(LINES)
    judgments = {s: judge_all(s, c[s]) for s in D.STRATUM_ORDER}
    judgments["face"] = {r: {**j, "eligible": "no", "reason": "two people"} for r, j in judgments["face"].items()}  # face runs out
    for s in ("manipulation", "body", "control"):
        needed = D.walk(LINES, judgments)[s]
        if not needed["missing"]:
            judgments[s] = {r: j for r, j in judgments[s].items() if r <= needed["log"][-1]["rank"]}
    with pytest.raises(RuntimeError, match="not every block"):
        D.assemble(LINES, judgments, tmp_path / "p.csv", tmp_path / "r.csv", tmp_path / "l.csv")
    out = D.assemble(LINES, judgments, tmp_path / "p.csv", tmp_path / "r.csv", tmp_path / "l.csv", allow_missing=True)
    assert out["missing"] == {"face": 3} and (tmp_path / D.REVIEW_LOG.name).exists()
    partial = dict(judgments)
    partial["manipulation"] = {r: j for r, j in judgments["manipulation"].items() if r == 1}  # not exhausted, only unfinished
    with pytest.raises(RuntimeError, match="not every block"):
        D.assemble(LINES, partial, tmp_path / "p.csv", tmp_path / "r.csv", tmp_path / "l.csv", allow_missing=True)


def full_judgments(c, skip_every=3):
    judgments = {}
    for s in D.STRATUM_ORDER:
        js = judge_all(s, c[s], skip_every)
        for r, j in js.items():
            subject, action, framing = {"face": ("a person", "smiles", "close-up"), "manipulation": ("a person", "holds a cup", "close-up"),
                                        "body": ("a person", "jumps", "wide shot"), "control": ("a lake", "ripples", "wide shot")}[s]
            j.update(text=f"Draft {s} {r}: {subject} {action} in a {framing}.", subject=subject, action=action, framing=framing)
        judgments[s] = js
    return judgments


def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_ids_are_fixed_by_slot_and_reviews_carry_over(small, tmp_path):
    c = D.candidates(LINES)
    judgments = full_judgments(c)
    paths = (tmp_path / "p.csv", tmp_path / "r.csv", tmp_path / "l.csv")
    D.assemble(LINES, judgments, *paths)
    rows = {r["source_id"]: r for r in read(paths[0])}
    assert {r["block"]: r["id"] for r in rows.values() if r["stratum"] == "control"} == {"control": "6", "reserve": "10"}
    review = read(paths[1])
    first, second, third = review[0], review[1], review[2]
    first.update(chase_decision="approve")
    second.update(chase_decision="edit", chase_text="Edited: a person smiles in a close-up.", chase_note="shorter")
    third.update(chase_decision="reject", chase_note="two people after all")
    with open(paths[1], "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=D.REVIEW_COLUMNS)
        w.writeheader()
        w.writerows(review)
    D.assemble(LINES, judgments, *paths)
    rows = {r["source_id"]: r for r in read(paths[0])}
    assert rows[second["source_id"]]["text"] == "Edited: a person smiles in a close-up." and "edited by Chase: shorter" in rows[second["source_id"]]["rewrite_note"]
    assert third["source_id"] not in rows  # rejected: the next eligible candidate took its place
    carried = {r["source_id"]: r for r in read(paths[1])}
    assert carried[first["source_id"]]["chase_decision"] == "approve" and carried[second["source_id"]]["chase_note"] == "shorter"
    log = read(tmp_path / "l.csv")
    assert any(e["source_id"] == third["source_id"] and e["reason"] == "rejected by Chase: two people after all" for e in log)
    assert carried[third["source_id"]]["block"] == "rejected" and carried[third["source_id"]]["chase_decision"] == "reject"
    again = D.assemble(LINES, judgments, *paths)  # the rejection survives the next assembly
    assert third["source_id"] not in {r["source_id"] for r in read(paths[0])} and again["reviewed"] == {"approve": 1, "edit": 1, "reject": 1}
    assert again["holdout_not_yet_approved"] == []
    judgments["face"] = {r: {**j, "text": j["text"] + " Changed."} if j["source_id"] == first["source_id"] else j for r, j in judgments["face"].items()}
    with pytest.raises(RuntimeError, match="draft changed since the review"):
        D.assemble(LINES, judgments, *paths)


def test_reviews_need_their_fields(tmp_path):
    path = tmp_path / "r.csv"
    for decision, extra, message in (("maybe", {}, "must be one of"), ("edit", {}, "needs chase_text"), ("reject", {}, "needs chase_note")):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=D.REVIEW_COLUMNS)
            w.writeheader()
            w.writerow({"source_id": "a:1", "chase_decision": decision, **extra})
        with pytest.raises(ValueError, match=message):
            D.read_reviews(path)


def test_a_second_pass_follows_the_first_without_moving_it(small, monkeypatch):
    """The second pass is drawn after the first: no first-pass rank moves, its ranks continue, it has its own seed,
    and a candidate repeating a first-pass candidate is skipped."""
    lines = {**LINES, "k": ["hand prompt 1", "kinetics a", "kinetics b", "kinetics c"]}  # "hand prompt 1" repeats c:1
    before = {s: [c["source_id"] for c in D.shuffled(s, D.candidates(LINES)[s])] for s in D.STRATUM_ORDER}
    monkeypatch.setattr(prompts, "sources", lambda: {**SOURCES, "k": {"protocol_strata": ["manipulation"], "draw_pass": 2}})
    monkeypatch.setitem(D.SECOND_PASS, "list", "k")
    first = D.candidates(lines)
    assert {s: [c["source_id"] for c in D.shuffled(s, first[s])] for s in D.STRATUM_ORDER} == before
    second = D.second_pass_candidates(lines, first)
    assert [c["source_id"] for c in second["manipulation"]] == ["k:1", "k:2", "k:3", "k:4"] and second["face"] == []
    order = D.ordered("manipulation", first, second)
    n = len(first["manipulation"])
    assert [c["rank"] for c in order] == list(range(1, n + 5))
    expected = np.random.default_rng([20261009, 4]).permutation(4)
    assert [c["source_id"] for c in order[n:]] == [second["manipulation"][i]["source_id"] for i in expected]
    # every first-pass manipulation candidate skipped: the walk fills the stratum from the second pass
    judgments = {s: judge_all(s, first[s]) for s in D.STRATUM_ORDER}
    judgments["manipulation"] = {r: {**j, "eligible": "no", "reason": "two people"} for r, j in judgments["manipulation"].items()}
    for c in order[n:]:
        judgments["manipulation"][c["rank"]] = {"stratum": "manipulation", "rank": str(c["rank"]), "source_id": c["source_id"], "eligible": "yes",
                                                "reason": "fits", "text": f"Draft {c['rank']}: a person smiles in a close-up.", "subject": "a person",
                                                "action": "smiles", "framing": "close-up"}
    result = D.walk(lines, judgments)["manipulation"]
    assert sorted(k["source_id"] for k in result["kept"]) == ["k:2", "k:3", "k:4"] and not result["missing"]
    assert any(e["source_id"] == "k:1" and e["reason"] == "repeats a first-pass candidate" for e in result["log"])


def test_derive_kinetics_checks_its_inputs(tmp_path):
    val = tmp_path / "val.csv"
    val.write_text("label,youtube_id\nabseiling,x\nbaking cookies,y\ntying tie,z\nbaking cookies,w\n", encoding="utf-8")
    import hashlib

    expected = {"val_csv_sha256": hashlib.sha256(val.read_bytes()).hexdigest(), "labels": 3, "remaining": 2, "template": "A person is {label}"}
    assert D.derive_kinetics(val, ["A person is tying tie"], expected) == ["A person is abseiling", "A person is baking cookies"]
    with pytest.raises(RuntimeError, match="SHA-256"):
        D.derive_kinetics(val, ["A person is tying tie"], {**expected, "val_csv_sha256": "0" * 64})
    with pytest.raises(RuntimeError, match="Kinetics-400 label"):
        D.derive_kinetics(val, ["A person is knitting"], expected)
    assert D.KINETICS["val_csv_sha256"] == "358eaf47e7f80ebf9b17d49eb0635ad5e0fdab98a9cbd75ffdd2ee5d5e5b6944" and D.SECOND_PASS["seed"] == (20261009, 4)
