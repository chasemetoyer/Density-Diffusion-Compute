"""The prompt set (protocol v2 section 6): the prompts.csv schema, its validator, and candidate lists from the pinned
sources in configs/prompt_sources.json.

The prompts are drafted by Claude under the registered procedure in density.prompt_draft (seed 20261009) and reviewed
row by row by Chase (2026-10-07; this replaced his earlier plan to write them himself). This module lists the
candidates and checks the file.

    python -m density.prompts candidates    write data/prompt_candidates/<list>.csv from the pinned source files
    python -m density.prompts template      write configs/prompts.csv with the header only, if it does not exist
    python -m density.prompts validate      check configs/prompts.csv (or --csv PATH) and print a report

Each row of prompts.csv:
- id: a unique integer from 1 to 999 across every block, reserves included. Equation 14 sets a prompt's seeds to
  1000 + id and 2000 + id, so the two seed ranges never collide, and C5's tie permutation is seeded by the id.
- stratum: face, manipulation, body or control.
- block: holdout, primary, secondary, control or reserve. reserve_order (1, 2, ...) is given for reserves only,
  one order per stratum.
- text: one or two sentences.
- subject, action, framing: the words in the text that name each, so the validator can check that all three are
  named. This is an addition to the schema Chase gave.
- source: a list name from configs/prompt_sources.json.
- source_id: the original prompt's ID, <list>:<line> (its 1-based line among the file's non-empty lines).
- rewrite_note: what changed from the original.

Registered block counts (section 6): holdout 6 face and 6 manipulation; primary 36 face and 36 manipulation;
secondary 16 full body; control 16. Section 6 registers 10 reserves per stratum. The tooling asks for at least 30
manipulation reserves, because both seeds must pass the audit; Chase sets the registered sizes after the pilot from its
pass rates' estimate (2026-10-07), and the main run's init then requires them. Raw source files and candidate lists are
not versioned. Only the rewritten prompts Chase approves, with a citation, may ever be published (T2V-CompBench has no
license).

Two scopes (Chase, 2026-10-07, decision 14): the pilot's init checks the holdout block only (its rows, its 6 + 6 count,
IDs unique across the file, and no holdout row duplicating any other row); the main run's init checks the whole file.
"""

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

from density.masks.roi import BODY, CONTROL, FACE, MANIPULATION

ROOT = Path(__file__).resolve().parents[2]
SOURCES_FILE = ROOT / "configs" / "prompt_sources.json"
SOURCES_DIR = ROOT / "data" / "prompt_sources"
CANDIDATES_DIR = ROOT / "data" / "prompt_candidates"
PROMPTS_CSV = ROOT / "configs" / "prompts.csv"
SMOKE_PROMPTS = ROOT / "configs" / "smoke_prompts.json"
TOKENIZER_DIR = ROOT / "weights" / "Wan2.1-T2V-1.3B-Diffusers"

COLUMNS = ["id", "stratum", "block", "reserve_order", "text", "subject", "action", "framing", "source", "source_id", "rewrite_note"]
STRATA = (FACE, MANIPULATION, BODY, CONTROL)
BLOCKS = ("holdout", "primary", "secondary", "control", "reserve")
BLOCK_STRATA = {"holdout": (FACE, MANIPULATION), "primary": (FACE, MANIPULATION), "secondary": (BODY,), "control": (CONTROL,), "reserve": STRATA}
BLOCK_COUNTS = {("holdout", FACE): 6, ("holdout", MANIPULATION): 6, ("primary", FACE): 36, ("primary", MANIPULATION): 36,
                ("secondary", BODY): 16, ("control", CONTROL): 16}  # protocol section 6
RESERVE_MINIMUMS = {FACE: 10, MANIPULATION: 30, BODY: 10, CONTROL: 10}  # tooling minimums, not registered values
MAX_TOKENS = 512  # text_encoding.MAX_SEQUENCE_LENGTH: longer prompts would be truncated
MAX_ID = 999  # seeds 1000 + id and 2000 + id (equation 14) stay in separate ranges, clear of the smoke seeds 9001 and 9002
SENTENCE_END = re.compile(r"[.!?]+(?:[\"')\]]*)(?=\s|$)")


def seeds(prompt_id: int) -> tuple[int, int]:
    """A prompt's two planned seeds (equation 14): 1000 + id and 2000 + id."""
    if not 1 <= prompt_id <= MAX_ID:
        raise ValueError(f"prompt id {prompt_id} is outside 1-{MAX_ID}")
    return 1000 + prompt_id, 2000 + prompt_id


def normalized(text: str) -> str:
    """Text for duplicate checks: case-folded, whitespace collapsed, trailing punctuation dropped."""
    return re.sub(r"\s+", " ", text.casefold()).strip().rstrip(".!?;:,").strip()


def sentences(text: str) -> int:
    return len(SENTENCE_END.findall(text.strip()))


def sources() -> dict:
    return json.loads(SOURCES_FILE.read_text())["lists"]


def source_lines(sources_dir: Path = SOURCES_DIR) -> dict[str, list[str]]:
    """Each pinned list's non-empty lines (stripped), after checking the file against its registered SHA-256."""
    out = {}
    for name, meta in sources().items():
        repo = meta["repository"].rstrip("/").rsplit("/", 1)[1]
        path = sources_dir / f"{repo}@{meta['commit']}" / meta["path"]
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path} is missing; download it at commit {meta['commit']}")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != meta["sha256"]:
            raise RuntimeError(f"{name}: {path} does not match its registered SHA-256")
        out[name] = [line.strip() for line in data.decode("utf-8").splitlines() if line.strip()]
        if len(out[name]) != meta["prompts"]:
            raise RuntimeError(f"{name}: {len(out[name])} prompts, registered {meta['prompts']}")
    return out


def write_candidates(lines: dict[str, list[str]] | None = None, out_dir: Path = CANDIDATES_DIR) -> dict[str, int]:
    """One CSV per list: source_id, line, text, the IDs of identical prompts in other lists, the license and section 6's strata."""
    lines = source_lines() if lines is None else lines
    meta = sources()
    seen: dict[str, list[str]] = {}
    for name, texts in lines.items():
        for i, text in enumerate(texts, 1):
            seen.setdefault(normalized(text), []).append(f"{name}:{i}")
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, texts in lines.items():
        with open(out_dir / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["source_id", "line", "text", "also_in", "license", "protocol_strata"])
            for i, text in enumerate(texts, 1):
                sid = f"{name}:{i}"
                w.writerow([sid, i, text, " ".join(x for x in seen[normalized(text)] if x != sid), meta[name]["license"], " ".join(meta[name]["protocol_strata"])])
        counts[name] = len(texts)
    return counts


def write_template(path: Path = PROMPTS_CSV) -> bool:
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(COLUMNS)
    return True


def load_tokenizer(weights: Path = TOKENIZER_DIR):
    """The registered UMT5 tokenizer, after checking its files against configs/wan_weights.json."""
    from transformers import AutoTokenizer

    from density import wan_weights

    wan_weights.verify(weights, ("tokenizer",))
    return AutoTokenizer.from_pretrained(weights / "tokenizer")


def token_count(text: str, tokenizer) -> int:
    """Tokens the Wan pipeline encodes for ``text``: its prompt cleaning, then special tokens, without truncation."""
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean

    return len(tokenizer(prompt_clean(text), add_special_tokens=True, truncation=False)["input_ids"])


def read_rows(path: Path) -> tuple[list[str], list[dict]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), [dict(r) for r in reader]


def validate(path: Path = PROMPTS_CSV, lines: dict[str, list[str]] | None = None, tokenizer=None, complete: bool = True,
             scope: str = "all", reserve_minimums: dict[str, int] | None = None) -> dict:
    """Every rule in the module docstring. ``complete=False`` skips the block and reserve counts, for checking a file
    still being written. ``scope="holdout"`` checks only the holdout rows and their counts (the pilot's init).
    ``reserve_minimums`` replaces the tooling's minimums (the main run's init passes the registered sizes). Returns the
    errors (empty when the file passes), the counts and each checked prompt's token count."""
    if scope not in ("all", "holdout"):
        raise ValueError(f"scope must be 'all' or 'holdout', not {scope!r}")
    header, rows = read_rows(path)
    errors = []
    if header != COLUMNS:
        return {"errors": [f"columns are {header}; expected {COLUMNS}"], "rows": len(rows)}
    lines = source_lines() if lines is None else lines
    smoke = {normalized(p["text"]) for p in json.loads(SMOKE_PROMPTS.read_text())["prompts"]}
    in_scope = [scope == "all" or r["block"] == "holdout" for r in rows]
    seen: dict[str, dict] = {"id": {}, "text": {}, "source_id": {}}  # every row's keys, so duplicates are found from either side
    for n, r in enumerate(rows, 2):  # line 1 is the header
        for kind, key in (("id", r["id"]), ("text", normalized(r["text"].strip())), ("source_id", r["source_id"])):
            if key:
                seen[kind].setdefault(key, []).append(n)
    counts, reserves, tokens = {}, {}, {}
    for n, (r, checked) in enumerate(zip(rows, in_scope), 2):
        counts[(r["block"], r["stratum"])] = counts.get((r["block"], r["stratum"]), 0) + 1
        if r["block"] == "reserve" and r["reserve_order"].isdigit() and int(r["reserve_order"]) >= 1:
            reserves.setdefault(r["stratum"], []).append(int(r["reserve_order"]))
        if not checked:
            continue
        where = f"line {n} ({r['id'] or 'no id'})"
        if not re.fullmatch(r"[1-9]\d{0,2}", r["id"]):
            errors.append(f"{where}: the id must be an integer from 1 to {MAX_ID}, written without leading zeros")
        elif len(seen["id"][r["id"]]) > 1:
            errors.append(f"{where}: id repeats line {[m for m in seen['id'][r['id']] if m != n][0]}")
        if r["stratum"] not in STRATA:
            errors.append(f"{where}: stratum {r['stratum']!r} is not one of {list(STRATA)}")
        if r["block"] not in BLOCKS:
            errors.append(f"{where}: block {r['block']!r} is not one of {list(BLOCKS)}")
        elif r["stratum"] in STRATA and r["stratum"] not in BLOCK_STRATA[r["block"]]:
            errors.append(f"{where}: block {r['block']} takes {list(BLOCK_STRATA[r['block']])}, not {r['stratum']}")
        if r["block"] == "reserve":
            if not r["reserve_order"].isdigit() or int(r["reserve_order"]) < 1:
                errors.append(f"{where}: a reserve needs a reserve_order of 1 or more")
        elif r["reserve_order"]:
            errors.append(f"{where}: only reserves have a reserve_order")
        text = r["text"].strip()
        if not text:
            errors.append(f"{where}: the text is empty")
            continue
        if not 1 <= sentences(text) <= 2 or not text.rstrip("\"')]").endswith((".", "!", "?")):
            errors.append(f"{where}: the text must be one or two sentences ending in . ! or ? ({sentences(text)} found)")
        for field in ("subject", "action", "framing"):
            if not r[field].strip():
                errors.append(f"{where}: {field} is empty")
            elif r[field].strip().casefold() not in text.casefold():
                errors.append(f"{where}: the {field} {r[field]!r} does not appear in the text")
        key = normalized(text)
        if key in smoke:
            errors.append(f"{where}: the text is a smoke-test prompt, which is excluded from every pool")
        if len(seen["text"][key]) > 1:
            errors.append(f"{where}: the text duplicates line {[m for m in seen['text'][key] if m != n][0]}")
        if r["source"] not in lines:
            errors.append(f"{where}: source {r['source']!r} is not a list in configs/prompt_sources.json")
        else:
            m = re.fullmatch(rf"{re.escape(r['source'])}:(\d+)", r["source_id"])
            if not m or not 1 <= int(m.group(1)) <= len(lines[r["source"]]):
                errors.append(f"{where}: source_id {r['source_id']!r} must be {r['source']}:<line> with a line from 1 to {len(lines[r['source']])}")
            elif len(seen["source_id"][r["source_id"]]) > 1:
                errors.append(f"{where}: source_id repeats line {[m for m in seen['source_id'][r['source_id']] if m != n][0]}")
        if not r["rewrite_note"].strip():
            errors.append(f"{where}: the rewrite note is empty")
        if tokenizer is not None:
            tokens[r["id"]] = token_count(text, tokenizer)
            if tokens[r["id"]] > MAX_TOKENS:
                errors.append(f"{where}: {tokens[r['id']]} T5 tokens; at most {MAX_TOKENS}")
    if scope == "all":
        for stratum, orders in reserves.items():
            if sorted(orders) != list(range(1, len(orders) + 1)):
                errors.append(f"{stratum} reserves: reserve_order must run 1, 2, ..., {len(orders)} without gaps or repeats")
    if complete:
        expected_counts = BLOCK_COUNTS if scope == "all" else {k: v for k, v in BLOCK_COUNTS.items() if k[0] == "holdout"}
        for (block, stratum), expected in expected_counts.items():
            if counts.get((block, stratum), 0) != expected:
                errors.append(f"{block} {stratum}: {counts.get((block, stratum), 0)} prompts, registered {expected}")
        if scope == "all":
            for stratum, minimum in (RESERVE_MINIMUMS if reserve_minimums is None else reserve_minimums).items():
                if len(reserves.get(stratum, [])) < minimum:
                    errors.append(f"{stratum} reserves: {len(reserves.get(stratum, []))}, at least {minimum} required")
    return {"errors": errors, "rows": len(rows), "checked_rows": sum(in_scope), "scope": scope,
            "counts": {f"{b} {s}": c for (b, s), c in sorted(counts.items())},
            "reserves": {s: len(o) for s, o in sorted(reserves.items())}, "tokens": tokens,
            "max_tokens": max(tokens.values()) if tokens else None, "complete": complete}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["candidates", "template", "validate"])
    parser.add_argument("--csv", type=Path, default=PROMPTS_CSV)
    parser.add_argument("--partial", action="store_true", help="validate: skip the block and reserve counts")
    parser.add_argument("--holdout", action="store_true", help="validate: the holdout block only (the pilot's check)")
    args = parser.parse_args(argv)
    if args.command == "candidates":
        counts = write_candidates()
        print(f"[prompts] candidates written to {CANDIDATES_DIR}: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
        return 0
    if args.command == "template":
        print(f"[prompts] {'wrote' if write_template(args.csv) else 'kept the existing'} {args.csv}")
        return 0
    report = validate(args.csv, tokenizer=load_tokenizer(), complete=not args.partial, scope="holdout" if args.holdout else "all")
    for e in report["errors"]:
        print(f"[prompts] {e}")
    print(f"[prompts] {report['rows']} rows; {'PASS' if not report['errors'] else str(len(report['errors'])) + ' errors'}; "
          f"counts {report.get('counts')}; reserves {report.get('reserves')}; longest {report.get('max_tokens')} T5 tokens")
    return 0 if not report["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
