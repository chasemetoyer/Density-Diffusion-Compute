"""Unit tests for the prompt-set tooling: the prompts.csv validator, the candidate lists and the T5 token count."""

import csv
import json
from pathlib import Path

import pytest

from density import prompts

ROOT = Path(__file__).resolve().parents[1]
LINES = {"vbench_human": ["A man smiles.", "A woman laughs.", "A child waves."], "vbench_scenery": ["A quiet lake at dawn.", "Snow on pine trees."]}


def row(**fields):
    base = {"id": "1", "stratum": "face", "block": "holdout", "reserve_order": "", "text": "A woman smiles at the camera in a close-up shot.",
            "subject": "a woman", "action": "smiles", "framing": "close-up", "source": "vbench_human", "source_id": "vbench_human:1", "rewrite_note": "added framing"}
    return {**base, **fields}


def write(tmp_path, rows):
    path = tmp_path / "prompts.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=prompts.COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return path


def errors(tmp_path, rows, **kwargs):
    return prompts.validate(write(tmp_path, rows), lines=LINES, complete=False, **kwargs)["errors"]


def test_a_good_row_passes(tmp_path):
    assert errors(tmp_path, [row()]) == []


@pytest.mark.parametrize("fields, message", [
    ({"id": "F01"}, "integer from 1 to 999"),
    ({"id": "01"}, "without leading zeros"),
    ({"id": "0"}, "integer from 1 to 999"),
    ({"id": "1000"}, "integer from 1 to 999"),
    ({"stratum": "faces"}, "stratum"),
    ({"block": "secondary"}, "block secondary takes"),
    ({"text": "A woman smiles. She waves. She leaves."}, "one or two sentences"),
    ({"text": "A woman smiles at the camera in a close-up shot"}, "one or two sentences"),
    ({"framing": "wide shot"}, "framing 'wide shot' does not appear"),
    ({"action": ""}, "action is empty"),
    ({"source": "elsewhere"}, "not a list"),
    ({"source_id": "vbench_human:4"}, "source_id"),
    ({"source_id": "vbench_scenery:1"}, "source_id"),
    ({"rewrite_note": " "}, "rewrite note is empty"),
    ({"reserve_order": "1"}, "only reserves"),
    ({"block": "reserve"}, "reserve_order of 1"),
])
def test_each_rule_reports_its_error(tmp_path, fields, message):
    found = errors(tmp_path, [row(**fields)])
    assert any(message in e for e in found), found


def test_duplicates_and_smoke_prompts_are_refused(tmp_path):
    rows = [row(), row(id="2", source_id="vbench_human:2", text="A WOMAN smiles at the camera in a close-up shot!")]
    assert any("duplicates line 2" in e for e in errors(tmp_path, rows))
    assert any("id repeats" in e for e in errors(tmp_path, [row(), row(source_id="vbench_human:2", text="A man laughs in a close-up shot.", subject="a man", action="laughs")]))
    assert any("source_id repeats" in e for e in errors(tmp_path, [row(), row(id="2", text="A man laughs in a close-up shot.", subject="a man", action="laughs")]))
    smoke = json.loads(prompts.SMOKE_PROMPTS.read_text())["prompts"][0]["text"]
    assert any("smoke-test prompt" in e for e in errors(tmp_path, [row(text=smoke, subject=smoke.split()[0], action=smoke.split()[1], framing=smoke.split()[2])]))


def test_reserve_orders_run_without_gaps(tmp_path):
    reserves = [row(id=str(10 + i), block="reserve", reserve_order=str(o), source_id=f"vbench_human:{i}", text=f"Person {i} smiles in a close-up shot.", subject=f"Person {i}")
                for i, o in ((1, 1), (2, 3))]
    assert any("without gaps" in e for e in errors(tmp_path, reserves))


def test_complete_checks_registered_counts_and_reserve_minimums(tmp_path, monkeypatch):
    monkeypatch.setattr(prompts, "BLOCK_COUNTS", {("holdout", "face"): 1})
    monkeypatch.setattr(prompts, "RESERVE_MINIMUMS", {"face": 1})
    path = write(tmp_path, [row()])
    report = prompts.validate(path, lines=LINES, complete=True)
    assert report["errors"] == ["face reserves: 0, at least 1 required"]
    assert prompts.RESERVE_MINIMUMS == {"face": 1}  # the registered minimums are restored after the test


def test_registered_minimums_and_counts():
    assert prompts.BLOCK_COUNTS == {("holdout", "face"): 6, ("holdout", "manipulation"): 6, ("primary", "face"): 36, ("primary", "manipulation"): 36,
                                    ("secondary", "body"): 16, ("control", "control"): 16}
    assert prompts.RESERVE_MINIMUMS == {"face": 10, "manipulation": 30, "body": 10, "control": 10}


def test_long_prompts_fail_the_token_check(tmp_path):
    class Tokenizer:
        def __call__(self, text, add_special_tokens, truncation):
            assert add_special_tokens and truncation is False
            return {"input_ids": [0] * (len(text.split()) + 1)}  # one token per word, plus EOS

    long_text = "A woman smiles at the camera in a close-up shot " + "and waves " * 300 + "."
    found = errors(tmp_path, [row(text=long_text)], tokenizer=Tokenizer())
    assert any("T5 tokens; at most 512" in e for e in found)
    report = prompts.validate(write(tmp_path, [row()]), lines=LINES, complete=False, tokenizer=Tokenizer())
    assert report["tokens"] == {"1": 11} and report["errors"] == []


def test_candidates_list_every_prompt_with_its_duplicates(tmp_path):
    lines = {"vbench_human": ["One.", "Two."], "vbench_scenery": ["two", "Three."]}
    counts = prompts.write_candidates(lines, tmp_path)
    assert counts == {"vbench_human": 2, "vbench_scenery": 2}
    with open(tmp_path / "vbench_human.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["source_id"] for r in rows] == ["vbench_human:1", "vbench_human:2"] and rows[1]["also_in"] == "vbench_scenery:1"
    assert rows[0]["license"].startswith("Apache-2.0") and rows[0]["protocol_strata"] == "face"


@pytest.mark.skipif(not prompts.SOURCES_DIR.is_dir(), reason="the pinned prompt files are not downloaded")
def test_pinned_sources_match_their_registered_hashes_and_counts():
    lines = prompts.source_lines()
    assert {k: len(v) for k, v in lines.items()} == {k: v["prompts"] for k, v in prompts.sources().items()}


@pytest.mark.skipif(not (prompts.TOKENIZER_DIR / "tokenizer" / "tokenizer.json").exists(), reason="the UMT5 tokenizer is not downloaded")
def test_registered_tokenizer_counts_tokens_with_eos():
    tokenizer = prompts.load_tokenizer()
    ids = tokenizer(prompts.normalized("a man smiles"), add_special_tokens=True, truncation=False)["input_ids"]
    assert ids[-1] == tokenizer.eos_token_id
    assert prompts.token_count("  A man   smiles.  ", tokenizer) == len(tokenizer("A man smiles.", add_special_tokens=True)["input_ids"])


def test_seeds_follow_equation_14():
    assert prompts.seeds(1) == (1001, 2001) and prompts.seeds(999) == (1999, 2999)
    with pytest.raises(ValueError):
        prompts.seeds(1000)


def test_holdout_scope_checks_only_the_holdout_block(tmp_path, monkeypatch):
    monkeypatch.setattr(prompts, "BLOCK_COUNTS", {("holdout", "face"): 1, ("primary", "face"): 5})
    bad_primary = row(id="2", block="primary", source_id="vbench_human:2", text="A man laughs.", subject="a man", action="", framing="")
    path = write(tmp_path, [row(), bad_primary])
    holdout = prompts.validate(path, lines=LINES, complete=True, scope="holdout")
    assert holdout["errors"] == [] and holdout["checked_rows"] == 1  # the primary row's errors and count are not the pilot's
    assert prompts.validate(path, lines=LINES, complete=True)["errors"]
    dup = row(id="2", block="primary", source_id="vbench_human:2", text="A WOMAN smiles at the camera in a close-up shot!")
    found = prompts.validate(write(tmp_path, [row(), dup]), lines=LINES, complete=False, scope="holdout")["errors"]
    assert any("duplicates line 3" in e for e in found)  # a holdout row duplicating any other row is caught
    clash = row(id="1", block="primary", source_id="vbench_human:3", text="A man laughs in a close-up shot.", subject="a man", action="laughs")
    assert any("id repeats" in e for e in prompts.validate(write(tmp_path, [row(), clash]), lines=LINES, complete=False, scope="holdout")["errors"])


def test_registered_reserve_sizes_replace_the_tooling_minimums(tmp_path, monkeypatch):
    monkeypatch.setattr(prompts, "BLOCK_COUNTS", {})
    path = write(tmp_path, [row(block="reserve", reserve_order="1")])
    assert prompts.validate(path, lines=LINES, complete=True, reserve_minimums={"face": 1})["errors"] == []
    assert prompts.validate(path, lines=LINES, complete=True, reserve_minimums={"face": 2})["errors"] == ["face reserves: 1, at least 2 required"]
