"""The registered procedure that drafts the prompt set (Chase, 2026-10-07). Claude drafts every row by this fixed
procedure and Chase reviews every row; it replaced his earlier plan to write the prompts himself.

1. The draw's seed is 20261009, registered before any draw.
2. A stratum's candidates are the pinned lists configs/prompt_sources.json assigns it (section 6), read in that file's
   order. A prompt in several lists (the same normalized text) is one candidate, under its first list's ID, so no
   source prompt is used twice in a stratum. Strata are drawn in the order face, manipulation, body, control; a source
   prompt kept in an earlier stratum is skipped in later ones, so the prompt file never repeats a source.
3. Each stratum's candidates are shuffled with numpy's ``default_rng([20261009, k])``, k being the stratum's place in
   that order.
4. Going down the shuffled order, each candidate is judged by its stratum's eligibility rule: kept with a draft, or
   skipped with a reason. The kept candidates fill the blocks in order: holdout, then primary (secondary for body,
   control for controls), then the reserves in order. Every candidate up to the last one kept is judged.
5. Counts: face 6 holdout, 36 primary, 20 reserves; manipulation 6, 36, 40; full body 16 and 10; controls 16 and 10.
6. The second pass (registered by Chase on 2026-10-07, before its draw). Manipulation's first pass gave too few
   candidates, so its walk continues, after the last first-pass candidate (rank 500), into a second list: the 300
   Kinetics-400 classes that VBench's human_action prompts do not use, each written as VBench writes them, "A person is
   {label}", in sorted label order (``derive_kinetics``, from the CVDF's val.csv; CC BY 4.0, Kay et al. 2017). They are
   shuffled with ``default_rng([20261009, 4])`` and ranked 501 onward, so no first-pass rank moves. A second-pass
   candidate whose normalized text matches any first-pass candidate of any stratum is skipped. The same rules and
   readings apply, and one more: a class that repeats the action and the named object of a manipulation prompt kept in
   either pass is a near-duplicate and is skipped.

Eligibility:
- Face: one person whose face can be the focus throughout. No crowds or animals.
- Manipulation: one person whose hands keep sustained contact with a named object (holding, peeling, pouring, writing,
  playing an instrument). No throws, catches or claps; no contact with the body alone.
- Full body: one person in full-body motion (dancing, jumping, running).
- Controls: scenery or architecture with no people or animals.

Readings (registered by Chase on 2026-10-07, after the draw and before his review; every judgment of the draw stands).
A reading is decided by the stratum's purpose, not by which prompts it brings in.
- Face: a prompt with two successive actions is skipped, since one clear expression cannot fit both.
- Manipulation needs one person's hands as the clear focus, in sustained contact with an object the source names,
  with nothing else the close-up would have to show:
  - a second subject or an animal in the scene is skipped, even when the hands do not touch it: it conflicts with the
    close-up framing, since the rewrite rule keeps every source subject. Two sequential actions are skipped too;
  - the object must be named in the source: an action that implies an object without naming it (writing, digging,
    sweeping, cutting nails) is skipped, since the rule's examples list action types, not exemptions;
  - an animal is not an object;
  - steering or riding (a car, a bike, a horse) is skipped: the hands hold controls the source does not name;
  - flying a kite is skipped: the hands hold the line, not the kite;
  - an action that needs a second person (sword fighting, hugging, shaking hands) counts as more than one person;
  - a toss and catch is a throw, even from a pan the hands hold;
  - the action and its effect must stay at the hands and the named object: an action at the mouth (tasting, eating,
    a wind instrument) or an effect elsewhere (hair a dryer blows) is skipped. (Chase, second round, 2026-10-07.)
- Full body (Chase, second round, 2026-10-07): no animal in the action, and no vehicle motion; the motion must be the
  body's.

Rewrite rules: one or two sentences naming the subject, action and framing; keep the source's subject and object, and
add no new objects or actions.
- Face: a close-up of one face, visible the whole clip, with one clear expression that fits the action.
- Manipulation: a close-up of the hands and the object.
- Full body: the whole figure in frame, with a static camera.
- Controls: no people.
- No style or quality words ("4K", "cinematic", "masterpiece").
The rewrite note is "Drafted by Claude under the registered procedure; reviewed by Chase", plus any edit Chase makes.

IDs are fixed by slot: each block has the ID range its full count needs, in the order face holdout, manipulation holdout,
face primary, manipulation primary, full body, controls, then the reserves of each stratum, so filling one block later
never moves another block's IDs, or the seeds they set (1000 + ID, 2000 + ID).

Chase reviews data/prompt_draft/review.csv: chase_decision is approve, edit (with chase_text, and chase_subject,
chase_action or chase_framing where they change) or reject (with chase_note); ``assemble`` carries his columns over by
source ID. An edit goes into prompts.csv, and the rewrite note records it; a rejection makes that candidate ineligible
("rejected by Chase: <note>"), so the next eligible one takes its place, and the rejected row stays at the end of the
review file. ``assemble`` refuses to overwrite a reviewed row: one Chase approved or edited must still be drawn, with
the draft he reviewed.

The judgments are Claude's, in data/prompt_draft/judgments.csv (not versioned: it holds the source texts). ``assemble``
checks that the judgments follow the shuffled order without gaps, then writes configs/prompts.csv, the versioned draw
log configs/prompt_draw_log.csv (every candidate judged or skipped, with its reason, and no source text), and the
review file data/prompt_draft/review.csv (source text, reason and draft per row, holdout first; not versioned).

    python -m density.prompt_draft order STRATUM [--start N] [--count M]   the shuffled candidates, for judging
    python -m density.prompt_draft assemble                               the files, from the judgments
    python -m density.prompt_draft derive-kinetics                        write the second pass's list from val.csv
"""

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np

from density import prompts as prompt_set
from density.masks.roi import BODY, CONTROL, FACE, MANIPULATION

ROOT = Path(__file__).resolve().parents[2]
DRAW_SEED = 20261009  # registered by Chase on 2026-10-07, before any draw
STRATUM_ORDER = (FACE, MANIPULATION, BODY, CONTROL)
COUNTS = {
    FACE: (("holdout", 6), ("primary", 36), ("reserve", 20)),
    MANIPULATION: (("holdout", 6), ("primary", 36), ("reserve", 40)),
    BODY: (("secondary", 16), ("reserve", 10)),
    CONTROL: (("control", 16), ("reserve", 10)),
}
REWRITE_NOTE = "Drafted by Claude under the registered procedure; reviewed by Chase"
READINGS = {  # registered by Chase on 2026-10-07 (see the module docstring)
    "principle": "a reading is decided by the stratum's purpose, not by which prompts it brings in",
    FACE: ("two successive actions are skipped: one clear expression cannot fit both",),
    MANIPULATION: (
        "one person's hands are the clear focus, in sustained contact with an object the source names, with nothing else the close-up would have to show",
        "a second subject or an animal in the scene is skipped, even untouched: the rewrite keeps every source subject, which the close-up framing cannot show",
        "two sequential actions are skipped",
        "the object must be named in the source; an action that implies one without naming it is skipped (the rule's examples are action types, not exemptions)",
        "an animal is not an object",
        "steering or riding is skipped: the hands hold controls the source does not name",
        "flying a kite is skipped: the hands hold the line, not the kite",
        "an action that needs a second person counts as more than one person",
        "a toss and catch is a throw, even from a pan the hands hold",
        "the action and its effect must stay at the hands and the named object (a mouth action, or an effect elsewhere, is skipped)",
    ),
    BODY: ("no animal in the action", "no vehicle motion: the motion must be the body's"),
}
SECOND_PASS = {  # registered by Chase on 2026-10-07, before its draw (see the module docstring, point 6)
    "list": "kinetics400_remaining",
    "stratum": MANIPULATION,
    "seed": (DRAW_SEED, 4),
    "ranks": "after the stratum's first-pass candidates, so no first-pass rank moves",
    "candidates": "the Kinetics-400 classes VBench's human_action prompts do not use, each 'A person is {label}', in sorted label order",
    "deduplication": "a candidate whose normalized text matches any first-pass candidate of any stratum is skipped",
    "near_duplicates": "a class that repeats the action and the named object of a manipulation prompt kept in either pass is skipped",
}
KINETICS = {"val_csv_sha256": "358eaf47e7f80ebf9b17d49eb0635ad5e0fdab98a9cbd75ffdd2ee5d5e5b6944", "labels": 400, "remaining": 300,
            "vbench_list": "vbench_human_action", "template": "A person is {label}"}
DRAFT_DIR = ROOT / "data" / "prompt_draft"
JUDGMENTS = DRAFT_DIR / "judgments.csv"
REVIEW = DRAFT_DIR / "review.csv"
REVIEW_LOG = DRAFT_DIR / "draw_log_with_text.csv"  # the draw log with each candidate's source text, for review (not versioned)
DRAW_LOG = ROOT / "configs" / "prompt_draw_log.csv"
JUDGMENT_COLUMNS = ["stratum", "rank", "source_id", "eligible", "reason", "text", "subject", "action", "framing"]
REVIEWER_COLUMNS = ["chase_decision", "chase_text", "chase_subject", "chase_action", "chase_framing", "chase_note"]
REVIEW_COLUMNS = ["id", "block", "stratum", "reserve_order", "source_id", "also_in", "source_text", "eligibility", "text", "subject", "action",
                  "framing", "rewrite_note", *REVIEWER_COLUMNS]
DECISIONS = ("approve", "edit", "reject")
BLOCK_ORDER = (("face", "holdout"), ("manipulation", "holdout"), ("face", "primary"), ("manipulation", "primary"), ("body", "secondary"),
               ("control", "control"), ("face", "reserve"), ("manipulation", "reserve"), ("body", "reserve"), ("control", "reserve"))
LOG_COLUMNS = ["stratum", "rank", "source_id", "also_in", "decision", "reason"]


def first_pass_list(meta: dict) -> bool:
    return meta.get("draw_pass", 1) == 1


def candidates(lines: dict[str, list[str]] | None = None) -> dict[str, list[dict]]:
    """Each stratum's de-duplicated first-pass candidates, in configs/prompt_sources.json's list order."""
    lines = prompt_set.source_lines() if lines is None else lines
    meta = prompt_set.sources()
    out = {}
    for stratum in STRATUM_ORDER:
        seen: dict[str, dict] = {}
        for name in meta:
            if stratum not in meta[name]["protocol_strata"] or not first_pass_list(meta[name]):
                continue
            for i, text in enumerate(lines[name], 1):
                key = prompt_set.normalized(text)
                if key in seen:
                    seen[key]["also_in"].append(f"{name}:{i}")
                else:
                    seen[key] = {"source": name, "source_id": f"{name}:{i}", "text": text, "key": key, "also_in": []}
        out[stratum] = list(seen.values())
    return out


def shuffled(stratum: str, cands: list[dict]) -> list[dict]:
    order = np.random.default_rng([DRAW_SEED, STRATUM_ORDER.index(stratum)]).permutation(len(cands))
    return [{**cands[i], "rank": rank} for rank, i in enumerate(order, 1)]


def second_pass_candidates(lines: dict[str, list[str]] | None = None, first: dict | None = None) -> dict[str, list[dict]]:
    """The second pass's candidates, for its stratum only, each marked when it repeats a first-pass candidate of any
    stratum (``repeats_first_pass``)."""
    lines = prompt_set.source_lines() if lines is None else lines
    first = candidates(lines) if first is None else first
    out: dict[str, list[dict]] = {s: [] for s in STRATUM_ORDER}
    name = SECOND_PASS["list"]
    if name not in prompt_set.sources():
        return out
    first_keys = {c["key"] for cs in first.values() for c in cs}
    seen: set[str] = set()
    for i, text in enumerate(lines[name], 1):
        key = prompt_set.normalized(text)
        if key in seen:
            continue
        seen.add(key)
        out[SECOND_PASS["stratum"]].append({"source": name, "source_id": f"{name}:{i}", "text": text, "key": key, "also_in": [],
                                            "repeats_first_pass": key in first_keys})
    return out


def ordered(stratum: str, first: dict, second: dict) -> list[dict]:
    """A stratum's candidates in draw order: its first-pass shuffle, then the second pass's, ranked on from it."""
    out = shuffled(stratum, first[stratum])
    if second.get(stratum):
        order = np.random.default_rng(list(SECOND_PASS["seed"])).permutation(len(second[stratum]))
        out += [{**second[stratum][i], "rank": len(first[stratum]) + rank} for rank, i in enumerate(order, 1)]
    return out


def derive_kinetics(val_csv: Path, vbench_lines: list[str], expected: dict = KINETICS) -> list[str]:
    """The second pass's list: val.csv's labels minus the classes VBench's human_action prompts use, each written
    "A person is {label}", in sorted label order. Checks val.csv's SHA-256, the label count, that every VBench class is
    a Kinetics label, and the count left over."""
    data = Path(val_csv).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected["val_csv_sha256"]:
        raise RuntimeError(f"{val_csv} does not match its registered SHA-256")
    labels = {r["label"] for r in csv.DictReader(data.decode("utf-8").splitlines())}
    if len(labels) != expected["labels"]:
        raise RuntimeError(f"{val_csv}: {len(labels)} labels, expected {expected['labels']}")
    prefix = expected["template"].split("{label}")[0]
    used = {line[len(prefix):] for line in vbench_lines if line.startswith(prefix)}
    if len(used) != len(vbench_lines) or not used <= labels:
        raise RuntimeError(f"not every VBench human_action prompt is {prefix!r} + a Kinetics-400 label: {sorted(used - labels)[:5]}")
    remaining = sorted(labels - used)
    if len(remaining) != expected["remaining"]:
        raise RuntimeError(f"{len(remaining)} classes left over, expected {expected['remaining']}")
    return [expected["template"].format(label=label) for label in remaining]


def read_judgments(path: Path = JUDGMENTS) -> dict[str, dict[int, dict]]:
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out: dict[str, dict[int, dict]] = {}
    for r in rows:
        out.setdefault(r["stratum"], {})[int(r["rank"])] = r
    return out


DRAFT_FIELDS = ("text", "subject", "action", "framing")


def read_reviews(path: Path = REVIEW) -> dict[str, dict]:
    """Chase's decisions from review.csv, by source ID, each with the row he reviewed (``reviewed``); rows he has not
    reviewed are left out."""
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        decision = (r.get("chase_decision") or "").strip().lower()
        if not decision:
            continue
        if decision not in DECISIONS:
            raise ValueError(f"{r['source_id']}: chase_decision must be one of {DECISIONS}, not {decision!r}")
        if decision == "edit" and not (r.get("chase_text") or "").strip():
            raise ValueError(f"{r['source_id']}: an edit needs chase_text")
        if decision == "reject" and not (r.get("chase_note") or "").strip():
            raise ValueError(f"{r['source_id']}: a rejection needs chase_note")
        out[r["source_id"]] = {**{c: (r.get(c) or "").strip() for c in REVIEWER_COLUMNS}, "chase_decision": decision,
                               "reviewed": {c: r.get(c) or "" for c in REVIEW_COLUMNS}}
    return out


def with_rejections(judgments: dict, reviews: dict) -> dict:
    """The judgments with every candidate Chase rejected made ineligible."""
    out = {s: dict(js) for s, js in judgments.items()}
    for s, js in out.items():
        for rank, j in js.items():
            rv = reviews.get(j["source_id"])
            if rv and rv["chase_decision"] == "reject":
                js[rank] = {**j, "eligible": "no", "reason": f"rejected by Chase: {rv['chase_note']}"}
    return out


def slot_ids() -> dict[tuple[str, str], int]:
    """The first ID of each block, from the full counts."""
    base, out = 1, {}
    for stratum, block in BLOCK_ORDER:
        out[(stratum, block)] = base
        base += dict(COUNTS[stratum])[block]
    return out


def walk(lines: dict[str, list[str]] | None = None, judgments: dict | None = None) -> dict:
    """The procedure: each stratum's shuffled candidates, judged in order, filling its blocks. Returns, per stratum, the
    log of every candidate up to the last one kept, the kept rows, and how many more are still needed."""
    judgments = read_judgments() if judgments is None else judgments
    cands = candidates(lines)
    second = second_pass_candidates(lines, cands)
    kept_keys: dict[str, str] = {}  # normalized source text -> the stratum that kept it
    out = {}
    for stratum in STRATUM_ORDER:
        slots = [block for block, n in COUNTS[stratum] for _ in range(n)]
        log, kept = [], []
        judged = judgments.get(stratum, {})
        for c in ordered(stratum, cands, second):
            if len(kept) == len(slots):
                break
            if c.get("repeats_first_pass"):
                log.append({**c, "decision": "skipped", "reason": "repeats a first-pass candidate"})
                continue
            if c["key"] in kept_keys:
                log.append({**c, "decision": "skipped", "reason": f"kept in the {kept_keys[c['key']]} stratum"})
                continue
            j = judged.get(c["rank"])
            if j is None:
                break  # the walk stops at the first candidate not yet judged
            if j["source_id"] != c["source_id"]:
                raise ValueError(f"{stratum} rank {c['rank']}: judged {j['source_id']}, but the shuffled order has {c['source_id']}")
            if j["eligible"] == "yes":
                block = slots[len(kept)]
                kept.append({**c, "block": block, "judgment": j})
                log.append({**c, "decision": f"kept: {block}", "reason": j["reason"]})
                kept_keys[c["key"]] = stratum
            elif j["eligible"] == "no":
                log.append({**c, "decision": "skipped", "reason": j["reason"]})
            else:
                raise ValueError(f"{stratum} rank {c['rank']}: eligible must be yes or no, not {j['eligible']!r}")
        out[stratum] = {"log": log, "kept": kept, "missing": len(slots) - len(kept),
                        "next_rank": None if len(kept) == len(slots) else (log[-1]["rank"] + 1 if log else 1)}
        # judgments past the last candidate needed cannot change what is kept; they are counted, never used
        out[stratum]["unused_judgments"] = sorted(r for r in judged if r > (log[-1]["rank"] if log else 0)) if not out[stratum]["missing"] else []
    return out


def assemble(lines: dict[str, list[str]] | None = None, judgments: dict | None = None, prompts_csv: Path = prompt_set.PROMPTS_CSV,
             review: Path = REVIEW, log_path: Path = DRAW_LOG, allow_missing: bool = False) -> dict:
    """Write the files. ``allow_missing`` writes the blocks as far as they are filled, for a stratum whose candidates ran
    out before its counts were met (reported in the result as ``missing``)."""
    reviews = read_reviews(review)
    judgments = with_rejections(read_judgments() if judgments is None else judgments, reviews)
    result = walk(lines, judgments)
    missing = {s: r["missing"] for s, r in result.items() if r["missing"]}
    first = candidates(lines)
    second = second_pass_candidates(lines, first)
    exhausted = {s for s, r in result.items() if r["missing"] and r["next_rank"] is not None and r["next_rank"] > len(first[s]) + len(second[s])}
    if missing and not (allow_missing and set(missing) <= exhausted):
        raise RuntimeError(f"not every block is filled yet: {missing}; judge from rank {[(s, r['next_rank']) for s, r in result.items() if r['missing']]}")
    kept = {k["source_id"]: k for r in result.values() for k in r["kept"]}
    gone = sorted(sid for sid, rv in reviews.items() if rv["chase_decision"] != "reject" and sid not in kept)
    changed = sorted(sid for sid, rv in reviews.items() if rv["chase_decision"] != "reject" and sid in kept
                     and any(kept[sid]["judgment"][f] != rv["reviewed"][f] for f in DRAFT_FIELDS))
    if gone or changed:
        raise RuntimeError(f"refusing to overwrite reviewed rows: no longer drawn {gone}; draft changed since the review {changed}. "
                           f"Ask Chase to review them again, with the earlier review.csv kept aside")
    bases = slot_ids()
    rows, review_rows = [], []
    for stratum, block in BLOCK_ORDER:
        slot = 0
        for k in result[stratum]["kept"]:
            if k["block"] != block:
                continue
            j, rv = k["judgment"], reviews.get(k["source_id"], {})
            row = {"id": str(bases[(stratum, block)] + slot), "stratum": stratum, "block": block, "reserve_order": str(slot + 1) if block == "reserve" else "",
                   "text": j["text"], "subject": j["subject"], "action": j["action"], "framing": j["framing"],
                   "source": k["source"], "source_id": k["source_id"], "rewrite_note": REWRITE_NOTE}
            if rv.get("chase_decision") == "edit":
                row.update(text=rv["chase_text"], **{f: rv[f"chase_{f}"] for f in ("subject", "action", "framing") if rv[f"chase_{f}"]})
                row["rewrite_note"] = f"{REWRITE_NOTE}; edited by Chase: {rv['chase_note'] or 'text changed'}"
            rows.append(row)
            review_rows.append({**{c: row.get(c, "") for c in REVIEW_COLUMNS}, "text": j["text"], "subject": j["subject"], "action": j["action"],
                                "framing": j["framing"], "also_in": " ".join(k["also_in"]), "source_text": k["text"], "eligibility": j["reason"],
                                **{c: rv.get(c, "") for c in REVIEWER_COLUMNS}})
            slot += 1
    # rejected rows stay in the review file, after the drawn ones, so the next assembly keeps them rejected
    review_rows += [{**rv["reviewed"], "id": "", "block": "rejected", "reserve_order": "", **{c: rv[c] for c in REVIEWER_COLUMNS}}
                    for sid, rv in reviews.items() if rv["chase_decision"] == "reject"]
    for path, columns, data in ((prompts_csv, prompt_set.COLUMNS, rows), (review, REVIEW_COLUMNS, review_rows)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=columns)
            w.writeheader()
            w.writerows(data)
    review_log = review.with_name(REVIEW_LOG.name)
    with open(log_path, "w", newline="", encoding="utf-8") as f, open(review_log, "w", newline="", encoding="utf-8") as g:
        w = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        wt = csv.DictWriter(g, fieldnames=LOG_COLUMNS + ["source_text"])
        w.writeheader()
        wt.writeheader()
        for stratum in STRATUM_ORDER:
            for e in result[stratum]["log"]:
                entry = {"stratum": stratum, "rank": e["rank"], "source_id": e["source_id"], "also_in": " ".join(e["also_in"]),
                         "decision": e["decision"], "reason": e["reason"]}
                w.writerow(entry)
                wt.writerow({**entry, "source_text": e["text"]})
    return {"rows": len(rows), "missing": missing, "reviewed": {d: sum(r["chase_decision"] == d for r in reviews.values()) for d in DECISIONS},
            "holdout_not_yet_approved": [r["id"] for r in rows if r["block"] == "holdout" and reviews.get(r["source_id"], {}).get("chase_decision") not in ("approve", "edit")],
            "judged": {s: len(r["log"]) for s, r in result.items()},
            "skipped": {s: sum(e["decision"] == "skipped" for e in r["log"]) for s, r in result.items()},
            "unused_judgments": {s: len(r["unused_judgments"]) for s, r in result.items()}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    o = sub.add_parser("order")
    o.add_argument("stratum", choices=STRATUM_ORDER)
    o.add_argument("--start", type=int, help="the first rank to print (default: the next one to judge)")
    o.add_argument("--count", type=int, default=40)
    a = sub.add_parser("assemble")
    a.add_argument("--partial", action="store_true", help="write the blocks as far as an exhausted stratum fills them")
    sub.add_parser("derive-kinetics")
    args = parser.parse_args(argv)
    if args.command == "assemble":
        print(assemble(allow_missing=args.partial))
        return 0
    if args.command == "derive-kinetics":
        return write_kinetics()
    state = walk()
    start = args.start or state[args.stratum]["next_rank"] or 1
    earlier = STRATUM_ORDER[:STRATUM_ORDER.index(args.stratum)]
    kept_elsewhere = {e["key"] for s in earlier for e in state[s]["kept"]}
    first = candidates()
    for c in ordered(args.stratum, first, second_pass_candidates(first=first))[start - 1:start - 1 + args.count]:
        note = ("  [repeats a first-pass candidate: skipped automatically]" if c.get("repeats_first_pass")
                else "  [kept in an earlier stratum: skipped automatically]" if c["key"] in kept_elsewhere else "")
        print(f"{c['rank']}\t{c['source_id']}\t{c['text']}{note}")
    print(f"[draft] {args.stratum}: {len(state[args.stratum]['kept'])} kept, {state[args.stratum]['missing']} still needed", file=sys.stderr)
    return 0


def write_kinetics() -> int:
    """Write the second pass's list beside val.csv, after checking val.csv's labels against torchvision's copy."""
    from torchvision.models._meta import _KINETICS400_CATEGORIES

    meta = prompt_set.sources()
    base = prompt_set.SOURCES_DIR / f"annotations@{KINETICS['val_csv_sha256']}"
    vbench = meta[KINETICS["vbench_list"]]
    vbench_path = prompt_set.SOURCES_DIR / f"{vbench['repository'].rstrip('/').rsplit('/', 1)[1]}@{vbench['commit']}" / vbench["path"]
    vbench_lines = [line.strip() for line in vbench_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    labels = {r["label"] for r in csv.DictReader((base / "val.csv").read_text(encoding="utf-8").splitlines())}
    if labels != set(_KINETICS400_CATEGORIES):
        raise RuntimeError("val.csv's labels differ from torchvision's Kinetics-400 categories")
    lines = derive_kinetics(base / "val.csv", vbench_lines)
    out = base / "kinetics400_remaining_classes.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[draft] {out}: {len(lines)} candidates, sha256 {hashlib.sha256(out.read_bytes()).hexdigest()}; labels equal torchvision's")
    return 0


if __name__ == "__main__":
    sys.exit(main())
