"""Audit tooling: contact sheets for visual sign-off and the audit CSV.

A contact sheet shows every 8th frame (the Hands23 keyframes) with each region
in its own color and the stratum ROI's token coverage overlaid: the outline of
the token cells with m >= 0.1, and a dot in each, brighter for higher m. Each tracked object is labelled with
its region, track ID and side (L or R) at its mask; pixels that only SAM's
propagation before a track's first detection covers are hatched rather than
filled, and their labels carry a ``*``. Hand tracks dropped because no Hands23
hand box confirms them are outlined in grey and labelled ``dropped``. Keyframes list Hands23's hands with
their contact state and matched tracks, and say whether the frame lies inside
a validated contact interval. Sheets are PNG, so their hashes are deterministic.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from density.masks.contact import keyframes
from density.masks.eligibility import AUDIT_CODES, AuditRejection
from density.masks.geometry import mask_box
from density.masks.registered import ROI_THRESHOLD
from density.token_map import CELL, frame_groups

COLORS = {"F": (255, 140, 0), "H": (0, 200, 70), "O": (40, 110, 255), "C": (230, 0, 200), "B": (0, 200, 220)}
DROPPED = (170, 170, 170)  # hand tracks no Hands23 hand box confirms
TILE = (416, 240)  # half resolution
HEADER = 48
AUDIT_COLUMNS = [
    "clip", "seed", "stratum", "pipeline_decision", "pipeline_reasons", "usable_roi_frames", "contact_intervals",
    "outside_frames", "contact_sheet", "contact_sheet_sha256",
    "reviewer_decision", "reviewer_reason", "reviewer_written_reason", "reviewer", "review_date",
]
REVIEWER_COLUMNS = AUDIT_COLUMNS[-5:]


def _latent_frame(f: int) -> int:
    return next(j for j, group in enumerate(frame_groups(81)) if f in group)


def _edge(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1)
    inner = padded[2:, 1:-1] & padded[:-2, 1:-1] & padded[1:-1, 2:] & padded[1:-1, :-2]
    return mask & ~inner


def _draw_coverage(draw: ImageDraw.ImageDraw, m: np.ndarray, cell: int) -> None:
    """Outline the token cells with m >= 0.1 and dot each one, brighter for higher m."""
    b = m >= ROI_THRESHOLD
    padded = np.pad(b, 1)
    for r, c in zip(*np.nonzero(b)):
        x0, y0, x1, y1 = c * cell, r * cell, (c + 1) * cell - 1, (r + 1) * cell - 1
        for neighbour, line in (
            (padded[r, c + 1], (x0, y0, x1, y0)),
            (padded[r + 2, c + 1], (x0, y1, x1, y1)),
            (padded[r + 1, c], (x0, y0, x0, y1)),
            (padded[r + 1, c + 2], (x1, y0, x1, y1)),
        ):
            if not neighbour:
                draw.line(line, fill=(255, 255, 255))
        shade = int(60 + 195 * float(m[r, c]))
        cx, cy = x0 + cell // 2 - 1, y0 + cell // 2 - 1
        draw.rectangle([cx, cy, cx + 1, cy + 1], fill=(shade, shade, shade))


def track_labels(regions: dict) -> dict[int, list[tuple[int, int, str, tuple]]]:
    """frame -> (x, y, text, color) at each track's mask box, at full resolution."""
    labels: dict[int, list] = {}
    for name, region in regions.items():
        tracks = getattr(region, "tracks", None)
        if tracks is None:
            continue
        dropped = set(getattr(region, "dropped", ()))
        for t in sorted(tracks.masks):
            side = tracks.sides.get(t)
            first = tracks.first_prompt(t)
            for f in keyframes(tracks.masks[t].shape[0]):
                box = mask_box(tracks.masks[t][f] if t in dropped else tracks.masks[t][f] & region.mask[f])
                if box is None:
                    continue
                text = f"{name}{t}" + (f" {side[0].upper()}" if side else "") + ("*" if f < first else "") + (" dropped" if t in dropped else "")
                labels.setdefault(f, []).append((int(box[0]), int(box[1]), text, DROPPED if t in dropped else COLORS[name]))
    return labels


def contact_sheet(
    frames: np.ndarray,
    masks: dict[str, np.ndarray],
    coverage: np.ndarray | None,
    path: Path,
    title: str,
    backfill: dict[str, np.ndarray] | None = None,
    labels: dict[int, list[tuple[int, int, str, str]]] | None = None,
    notes: dict[int, list[str]] | None = None,
    valid: np.ndarray | None = None,
    interval_names: dict[int, str] | None = None,
    dropped: np.ndarray | None = None,
) -> Path:
    """Write the sheet. ``coverage`` is the ROI's token coverage m, shape (latent frames, rows, cols).

    ``dropped`` is the union of the dropped hand tracks, outlined in grey and never filled.
    """
    backfill, labels, notes, interval_names = backfill or {}, labels or {}, notes or {}, interval_names or {}
    sy, sx = frames.shape[1] // TILE[1], frames.shape[2] // TILE[0]
    yy, xx = np.mgrid[: TILE[1], : TILE[0]]
    hatch = (xx + yy) % 6 < 2
    tiles = []
    for f in keyframes(frames.shape[0]):
        image = np.asarray(Image.fromarray(frames[f]).resize(TILE, Image.BOX)).astype(np.float32)
        edges = []
        for name, mask in masks.items():
            m = mask[f, ::sy, ::sx]
            if not m.any():
                continue
            b = backfill[name][f, ::sy, ::sx] if name in backfill else np.zeros_like(m)
            color = np.array(COLORS[name], dtype=np.float32)
            solid = m & ~b
            image[solid] = 0.55 * image[solid] + 0.45 * color
            image[b & hatch] = color
            edges.append((_edge(m), COLORS[name]))
        if dropped is not None and dropped[f].any():
            edges.insert(0, (_edge(dropped[f, ::sy, ::sx]), DROPPED))
        for edge, color in edges:  # outlines on top of every fill
            image[edge] = color
        tile = Image.fromarray(image.clip(0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(tile)
        if coverage is not None:
            _draw_coverage(draw, coverage[_latent_frame(f)], CELL // sx)
        for x, y, text, color in labels.get(f, []):
            x, y = x // sx, max(0, y // sy - 11)
            draw.rectangle([x, y, x + 6 * len(text) + 3, y + 10], fill=(0, 0, 0))
            draw.text((x + 2, y), text, fill=color)
        caption = f"frame {f} (latent {_latent_frame(f)})"
        if valid is not None and valid[f]:
            caption += " | contact interval " + interval_names.get(f, "")
        lines = notes.get(f, [])
        draw.rectangle([0, 0, TILE[0], 13 + 11 * len(lines)], fill=(0, 0, 0))
        draw.text((3, 1), caption, fill=(255, 255, 255))
        for k, text in enumerate(lines):
            draw.text((3, 12 + 11 * k), text, fill=(255, 230, 0))
        tiles.append(tile)

    cols = 4
    rows = -(-len(tiles) // cols)
    sheet = Image.new("RGB", (cols * TILE[0], rows * TILE[1] + HEADER), (255, 255, 255))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % cols) * TILE[0], (i // cols) * TILE[1] + HEADER))
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 4), title, fill=(0, 0, 0))
    x = 6
    for name in masks:
        draw.rectangle([x, 20, x + 10, 30], fill=COLORS[name])
        draw.text((x + 14, 19), name, fill=(0, 0, 0))
        x += 40
    legend = ["labels: region, track ID, side (L/R); * and hatching: SAM before the track's first detection"]
    if dropped is not None and dropped.any():
        legend.append("grey outline: hand track dropped (no Hands23 confirmation)")
    if coverage is not None:
        legend.append("white outline: ROI token cells with m >= 0.1; dots brighter for higher m")
    draw.text((x + 8, 19), "   ".join(legend), fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, format="PNG")
    return path


def write_audit_csv(rows: list[dict], path: Path) -> Path:
    """Pipeline decisions to review. Reviewers fill the last five columns (see the notes written under the rows).

    A reviewer's entries carry over from an existing file, including one written before a column was added, only for
    rows whose contact sheet is unchanged.
    """
    previous = {}
    if path.exists():
        with open(path, newline="") as f:
            for old in csv.DictReader(line for line in f if not line.startswith("#")):
                previous[(old["clip"], old["seed"], old["contact_sheet"])] = old
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=AUDIT_COLUMNS)
        writer.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in AUDIT_COLUMNS}
            old = previous.get((str(out["clip"]), str(out["seed"]), str(out["contact_sheet"])))
            if old and old["contact_sheet_sha256"] == out["contact_sheet_sha256"]:
                out.update({k: old.get(k) or "" for k in REVIEWER_COLUMNS})
            writer.writerow(out)
        f.write("# reviewer_decision: accept or reject, by the audit criteria in protocol section 4.\n")
        f.write("# A reject needs reviewer_reason, one of these kinds of mask invalidity: " + ", ".join(AUDIT_CODES) + ",\n")
        f.write("# and reviewer_written_reason, the reviewer's own words. A target that isn't visible counts as invalid.\n")
        f.write("# Anatomy, image quality and appearance are never reasons to reject a reference.\n")
    return path


@dataclass(frozen=True)
class AuditDecision:
    """One signed row of the audit CSV."""

    clip: str
    seed: str
    accepted: bool
    rejection: AuditRejection | None
    reviewer: str
    review_date: str
    contact_sheet_sha256: str


def read_audit_csv(path: Path) -> list[AuditDecision]:
    """The reviewed rows (accept or reject, in any letter case); rows not yet reviewed are skipped.

    A reject without a mask-invalidity code or a written reason, or any other decision, is an error.
    """
    decisions = []
    with open(path, newline="") as f:
        for r in csv.DictReader(line for line in f if not line.startswith("#")):
            decision = (r.get("reviewer_decision") or "").strip().lower()
            if not decision:
                continue
            if decision not in ("accept", "reject"):
                raise ValueError(f"{r['clip']} seed {r['seed']!r}: reviewer_decision must be accept or reject, not {r['reviewer_decision']!r}")
            rejection = None
            if decision == "reject":
                rejection = AuditRejection((r.get("reviewer_reason") or "").strip(), r.get("reviewer_written_reason") or "")
            decisions.append(
                AuditDecision(r["clip"], r["seed"], decision == "accept", rejection, r.get("reviewer") or "", r.get("review_date") or "", r["contact_sheet_sha256"])
            )
    return decisions
