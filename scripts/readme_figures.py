"""The README's figures and tables, from the pilot run's own records (results/runs/pilot, pulled from the pods).

    uv run python scripts/readme_figures.py      writes docs/images/ and prints the README's tables

Charts and diagrams come in light and dark variants (the README picks one with <picture>), in a palette validated for colour
vision deficiency (blue C3, orange C4, aqua C7), with every series labelled directly. Every image is Wan2.1 output from
the pilot; no stock footage is used.
"""

import json
import statistics as st
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results" / "runs" / "pilot"
OUT = ROOT / "docs" / "images"

THEMES = {
    "light": {"surface": "#fcfcfb", "text": "#0b0b0b", "text2": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "base": "#c3c2b7",
              "C3": "#2a78d6", "C4": "#eb6834", "C7": "#1baf7a", "line": "#c3c2b7"},
    "dark": {"surface": "#1a1a19", "text": "#ffffff", "text2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a", "base": "#383835",
             "C3": "#3987e5", "C4": "#d95926", "C7": "#199e70", "line": "#52514e"},
}
NAMES = {"C3": "C3  HSA baseline (need-ranked)", "C4": "C4  Density (region-weighted, λ = 10)", "C7": "C7  inverse (exploratory)"}
SHORT = {1: "badminton", 2: "honey on rolls", 3: "kettlebell", 4: "football", 5: "posing with a flag", 6: "ping-pong",
         7: "tying a tie", 8: "sharpening knives", 9: "making a snowman", 10: "dressing a teddy bear", 11: "shining shoes",
         12: "strumming a guitar"}
C4 = "C4@10"


def style(ax, t):
    ax.set_facecolor(t["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["base"])
    ax.tick_params(colors=t["text2"], labelsize=10)
    ax.grid(True, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def figure(t, w, h):
    fig, ax = plt.subplots(figsize=(w, h), dpi=150)
    fig.patch.set_facecolor(t["surface"])
    style(ax, t)
    return fig, ax


def title(fig, t, text, sub):
    fig.text(0.012, 0.975, text, ha="left", va="top", fontsize=13, fontweight="bold", color=t["text"])
    fig.text(0.012, 0.915, sub, ha="left", va="top", fontsize=9.5, color=t["text2"])


def dose_response(rows, mode):
    t = THEMES[mode]
    fig, ax = figure(t, 9.6, 5.4)
    for r in rows:
        xs = [r["s_roi"]["C7"], r["s_roi"]["C3"], r["s_roi"][C4]]
        ys = [r["masked_lpips"]["C7"] / r["masked_lpips"]["C3"], 1.0, r["masked_lpips"][C4] / r["masked_lpips"]["C3"]]
        ax.plot(xs, ys, color=t["line"], linewidth=1.0, zorder=1)
        for x, y, c in zip(xs, ys, ("C7", "C3", "C4")):
            ax.scatter(x, y, s=46, color=t[c], edgecolors=t["surface"], linewidths=1.5, zorder=3)
    ax.axhline(1.0, color=t["base"], linewidth=1.2, zorder=0)
    for c, key in (("C3", "C3"), ("C4", C4), ("C7", "C7")):
        mx = st.median(r["s_roi"][key] for r in rows)
        my = st.median(r["masked_lpips"][key] / r["masked_lpips"]["C3"] for r in rows)
        ax.scatter(mx, my, s=190, marker="D", color=t[c], edgecolors=t["surface"], linewidths=2, zorder=4,
                   label=f"{NAMES[c]}: median share {mx:.0%}, median ratio ×{my:.3f}")
        ax.annotate(c, (mx, my), xytext=(0, -20), textcoords="offset points", ha="center", fontsize=10, fontweight="bold", color=t["text"])
    leg = ax.legend(loc="upper center", frameon=False, fontsize=9, title="Diamonds: medians over the 12 prompts", title_fontsize=9)
    leg.get_title().set_color(t["text2"])
    for text in leg.get_texts():
        text.set_color(t["text"])
    ax.set_xlabel("Share of the compute spent on the region (S_ROI)", color=t["text2"], fontsize=10)
    ax.set_ylabel("Masked LPIPS relative to C3 (lower is better)", color=t["text2"], fontsize=10)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    lo = min(r["masked_lpips"]["C7"] / r["masked_lpips"]["C3"] for r in rows)
    hi = max(max(r["masked_lpips"]["C7"], r["masked_lpips"][C4]) / r["masked_lpips"]["C3"] for r in rows)
    ax.set_ylim(min(0.93, lo - 0.02), max(1.2, hi + 0.1))
    title(fig, t, "Moving compute into or out of the region barely moves its fidelity",
          "12 holdout prompts, seed 1, one line per prompt. Same budget (HSA-50) for all three; only which tokens get the full steps differs.")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.83, bottom=0.12)
    fig.savefig(OUT / f"dose_response_{mode}.png", facecolor=t["surface"])
    plt.close(fig)


def per_prompt(rows, mode):
    t = THEMES[mode]
    fig, ax = figure(t, 9.6, 6.2)
    order = sorted(rows, key=lambda r: r["prompt"])
    offsets = {"C3": -0.2, "C4": 0.0, "C7": 0.2}
    for i, r in enumerate(order):
        y = len(order) - 1 - i
        vals = [r["masked_lpips"][k] for k in ("C3", C4, "C7")]
        ax.plot([min(vals), max(vals)], [y, y], color=t["line"], linewidth=1.0, zorder=1)
        for c, key in (("C3", "C3"), ("C4", C4), ("C7", "C7")):
            ax.scatter(r["masked_lpips"][key], y + offsets[c], s=58, color=t[c], edgecolors=t["surface"], linewidths=1.5, zorder=3,
                       marker={"C3": "o", "C4": "s", "C7": "^"}[c])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{r['prompt']}  {SHORT[r['prompt']]}  ({'face' if r['stratum'] == 'face' else 'hands'})" for r in order][::-1])
    ax.axhline(5.5, color=t["base"], linewidth=1.0)
    ax.set_xlabel("Masked LPIPS against the full-compute reference (lower is better)", color=t["text2"], fontsize=10)
    short = {"C3": "C3  HSA baseline", "C4": "C4  Density (λ = 10)", "C7": "C7  inverse"}
    for c, m in (("C3", "o"), ("C4", "s"), ("C7", "^")):
        ax.scatter([], [], s=58, color=t[c], marker=m, label=short[c])
    leg = fig.legend(loc="upper left", bbox_to_anchor=(0.29, 0.885), ncol=3, frameon=False, fontsize=8.6, handletextpad=0.3, columnspacing=1.2)
    for text in leg.get_texts():
        text.set_color(t["text"])
    title(fig, t, "Per prompt: the three allocations land close together",
          "Masked LPIPS of C3, C4 (λ = 10) and C7 on the 12 holdout prompts; each row is one prompt, seed 1.")
    fig.subplots_adjust(left=0.29, right=0.98, top=0.83, bottom=0.1)
    fig.savefig(OUT / f"per_prompt_{mode}.png", facecolor=t["surface"])
    plt.close(fig)


def c2_check(rules, mode):
    t = THEMES[mode]
    fig, ax = figure(t, 9.6, 3.2)
    med = rules["c2"]["median_full_frame_lpips"]
    items = [("C2: 20 uniform dense steps", med["C2"], t["muted"]), ("C3: HSA-50, same budget", med["HSA-50 (C3)"], t["C3"])]
    for y, (label, v, color) in enumerate(items[::-1]):
        ax.plot([0, v], [y, y], color=color, linewidth=2.5, solid_capstyle="round", zorder=2)
        ax.scatter(v, y, s=90, color=color, edgecolors=t["surface"], linewidths=2, zorder=3)
        ax.annotate(f"{v:.3f}", (v, y), xytext=(10, 0), textcoords="offset points", va="center", fontsize=10, color=t["text"])
    bar = rules["c2"]["threshold"] * med["C2"]
    ax.axvline(bar, color=t["text2"], linewidth=1.2, linestyle=(0, (4, 3)))
    ax.annotate(f"pass line: 70 % of C2 = {bar:.3f}", (bar, 1.42), xytext=(6, 0), textcoords="offset points", ha="left", fontsize=9, color=t["text2"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels([items[1][0], items[0][0]])
    ax.set_ylim(-0.6, 1.7)
    ax.set_xlim(0, max(med.values()) * 1.18)
    ax.set_xlabel("Median full-frame LPIPS vs the 40-step reference (lower is better)", color=t["text2"], fontsize=10)
    ax.grid(axis="y", visible=False)
    title(fig, t, f"HSA reproduces: at the same budget it lands at {rules['c2']['ratio']:.0%} of the naive baseline's error",
          "The pilot's reproduction check, on the HSA paper's own metric and the 12 holdout prompts; it had to come in under 70 %.")
    fig.subplots_adjust(left=0.25, right=0.95, top=0.74, bottom=0.2)
    fig.savefig(OUT / f"c2_check_{mode}.png", facecolor=t["surface"])
    plt.close(fig)


def frames(job: str) -> np.ndarray:
    from density.video import decode_rgb

    return decode_rgb(RUN / "jobs" / job / "video.mp4")


def qualitative(prompt: int, name: str, picks=(0, 40, 80)):
    g = f"{prompt:03d}-s{1000 + prompt}-sh8g6"
    rows = [("C1  full compute\n(40 dense steps)", f"ref-{g}"), ("C2  20 dense steps", f"gen-{g}-C2"), ("C3  HSA-50", f"gen-{g}-C3-HSA-50"),
            ("C4  Density\n(λ = 10)", f"gen-{g}-C4-HSA-50-l10"), ("C7  inverse\n(exploratory)", f"gen-{g}-C7-HSA-50")]
    t = THEMES["dark"]
    cell_w, cell_h = 416, 240
    fig = plt.figure(figsize=((220 + 3 * cell_w + 20) / 100, (40 + 5 * cell_h + 10) / 100), dpi=100)
    fig.patch.set_facecolor(t["surface"])
    for r, (label, job) in enumerate(rows):
        video = frames(job)
        for c, f in enumerate(picks):
            ax = fig.add_axes([(220 + c * cell_w) / fig.bbox.width, 1 - (40 + (r + 1) * cell_h) / fig.bbox.height, (cell_w - 4) / fig.bbox.width,
                               (cell_h - 4) / fig.bbox.height])
            ax.imshow(np.asarray(Image.fromarray(video[f]).resize((cell_w - 4, cell_h - 4), Image.LANCZOS)))
            ax.axis("off")
        fig.text(10 / fig.bbox.width, 1 - (40 + (r + 0.5) * cell_h) / fig.bbox.height, label, color=t["text"], fontsize=10, va="center")
    for c, f in enumerate(picks):
        fig.text((220 + (c + 0.5) * cell_w) / fig.bbox.width, 1 - 22 / fig.bbox.height, f"frame {f}", color=t["text2"], fontsize=10, ha="center")
    path = OUT / f"qualitative_{name}.jpg"
    fig.savefig(path.with_suffix(".png"), facecolor=t["surface"])
    plt.close(fig)
    Image.open(path.with_suffix(".png")).convert("RGB").save(path, quality=86, optimize=True)
    path.with_suffix(".png").unlink()


def sheets(groups):
    for group, name in groups:
        im = Image.open(RUN / "masks" / group / f"{group}.png").convert("RGB")
        im.save(OUT / f"segmentation_{name}.jpg", quality=86, optimize=True)


def gif(prompt: int, name: str, step=3, width=272):
    g = f"{prompt:03d}-s{1000 + prompt}-sh8g6"
    panels = [("C1 full compute", f"ref-{g}"), ("C4 Density", f"gen-{g}-C4-HSA-50-l10"), ("C7 inverse", f"gen-{g}-C7-HSA-50")]
    videos = [frames(job) for _, job in panels]
    h = round(width * 480 / 832)
    from PIL import ImageDraw, ImageFont

    font = ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"), 13)
    out = []
    for f in range(0, 81, step):
        canvas = Image.new("RGB", (3 * width + 8, h + 22), THEMES["dark"]["surface"])
        draw = ImageDraw.Draw(canvas)
        for i, ((label, _), v) in enumerate(zip(panels, videos)):
            canvas.paste(Image.fromarray(v[f]).resize((width, h), Image.LANCZOS), (i * (width + 4), 22))
            draw.text((i * (width + 4) + 6, 3), label, fill=THEMES["dark"]["text"], font=font)
        out.append(canvas.quantize(colors=128, method=Image.Quantize.MEDIANCUT))
    out[0].save(OUT / f"{name}.gif", save_all=True, append_images=out[1:], duration=int(1000 * step / 16), loop=0, optimize=True)


# Diagrams: drawn here rather than in mermaid, whose GitHub renderer wraps and clips long node labels. One data unit is
# 0.1 inch, and every box is sized from its measured text.

DPI, UNIT = 150, 15  # pixels per inch, pixels per data unit
HEAD, LINE, PAD = 11.5, 10, (1.3, 1.0)  # font sizes in points; box padding in units
FILL = {"light": {"node": "#ffffff", "lane": "#f1f0ea"}, "dark": {"node": "#262624", "lane": "#20201f"}}


def _size(fig, text, size, weight="normal"):
    t = fig.text(0, 0, text, fontsize=size, fontweight=weight)
    bb = t.get_window_extent(fig.canvas.get_renderer())
    t.remove()
    return bb.width / UNIT, bb.height / UNIT


def _node_size(fig, head, lines, marked=False):
    hw, hh = _size(fig, head, HEAD, "bold")
    sizes = [_size(fig, s, LINE) for s in lines]
    w = max([hw] + [lw + (1.6 if marked else 0) for lw, _ in sizes])
    h = hh + sum(lh for _, lh in sizes) + 0.45 * len(lines)
    return w + 2 * PAD[0], h + 2 * PAD[1]


def _node(ax, t, mode, cx, cy, w, h, head, lines, colors=None):
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.8",
                                fc=FILL[mode]["node"], ec=t["base"], lw=1.2, zorder=2))
    fig = ax.figure
    _, hh = _size(fig, head, HEAD, "bold")
    sizes = [_size(fig, s, LINE) for s in lines]
    y = cy + (hh + sum(lh for _, lh in sizes) + 0.45 * len(lines)) / 2
    ax.text(cx, y - hh / 2, head, ha="center", va="center", fontsize=HEAD, fontweight="bold", color=t["text"], zorder=3)
    y -= hh + 0.45
    left = cx - w / 2 + PAD[0]
    for i, (s, (_, lh)) in enumerate(zip(lines, sizes)):
        if colors:  # a series swatch, then the text in ink
            ax.scatter(left + 0.45, y - lh / 2, s=55, marker="s", color=t[colors[i]], zorder=3)
            ax.text(left + 1.6, y - lh / 2, s, ha="left", va="center", fontsize=LINE, color=t["text"], zorder=3)
        else:
            ax.text(cx, y - lh / 2, s, ha="center", va="center", fontsize=LINE, color=t["text2"], zorder=3)
        y -= lh + 0.45


def _arrow(ax, t, a, b):
    ax.annotate("", xy=b, xytext=a, zorder=1,
                arrowprops=dict(arrowstyle="-|>,head_length=0.55,head_width=0.3", color=t["text2"], lw=1.3, shrinkA=0, shrinkB=0))


def _canvas(t, w, h):
    fig = plt.figure(figsize=(w / 10, h / 10), dpi=DPI)
    fig.patch.set_facecolor(t["surface"])
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def allocation_diagram(mode):
    """How HSA-50 and the conditions choose the full-step group."""
    t = THEMES[mode]
    nodes = {
        "W": ("First 5 steps", ["every token runs"], None),
        "G": ("Need score g", ["per token"], None),
        "M": ("Region coverage m", ["per token"], None),
        "R": ("Rank tokens", ["C3: by g", "C4: by (1 + λ·m)·g", "C7: outside the region first"], ["C3", "C4", "C7"]),
        "H": ("Full-step group", ["16% of tokens, all 40 steps"], None),
        "L": ("Reduced group", ["84% of tokens, 16 of 40 steps"], None),
    }
    probe = plt.figure(dpi=DPI)
    size = {k: _node_size(probe, head, lines, colors is not None) for k, (head, lines, colors) in nodes.items()}
    plt.close(probe)
    gap, margin, vgap = 5.5, 1.5, 2.6
    cols = [["W"], ["G", "M"], ["R"], ["H", "L"]]
    col_w = [max(size[k][0] for k in c) for c in cols]
    top = max(size[k][1] for k in ("W", "G", "H"))
    bottom = max(size[k][1] for k in ("M", "L"))
    stack = top + vgap + bottom
    rank_h = max(size["R"][1], stack - 2.0)
    W = 2 * margin + sum(col_w) + gap * (len(cols) - 1)
    H = 2 * margin + max(stack, rank_h)
    yt, yb = H / 2 + stack / 2 - top / 2, H / 2 - stack / 2 + bottom / 2
    x, cx = margin, []
    for w in col_w:
        cx.append(x + w / 2)
        x += w + gap
    pos = {"W": (cx[0], yt), "G": (cx[1], yt), "M": (cx[1], yb), "R": (cx[2], H / 2), "H": (cx[3], yt), "L": (cx[3], yb)}
    fig, ax = _canvas(t, W, H)
    for k, (head, lines, colors) in nodes.items():
        w = col_w[[i for i, c in enumerate(cols) if k in c][0]] if k in ("G", "M", "H", "L") else size[k][0]
        h = rank_h if k == "R" else size[k][1]
        _node(ax, t, mode, *pos[k], w, h, head, lines, colors)
    edge = lambda k, side: pos[k][0] + (1 if side == "r" else -1) * (col_w[[i for i, c in enumerate(cols) if k in c][0]] / 2)
    _arrow(ax, t, (edge("W", "r"), yt), (edge("G", "l"), yt))
    for k, y in (("G", yt), ("M", yb)):
        _arrow(ax, t, (edge(k, "r"), y), (edge("R", "l"), y))
    for k, y in (("H", yt), ("L", yb)):
        _arrow(ax, t, (edge("R", "r"), y), (edge(k, "l"), y))
    fig.savefig(OUT / f"allocation_{mode}.png", facecolor=t["surface"])
    plt.close(fig)


def pipeline_diagram(mode):
    """The study's pipeline in three lanes, each step with the machine it runs on."""
    from matplotlib.patches import FancyBboxPatch

    t = THEMES[mode]
    lanes = [
        ("Generate", [("Prompts", ["registered draw,", "human review"]), ("Text embeddings", ["UMT5 · H100"]),
                      ("References (C1)", ["Wan2.1, 40 steps · H100"])]),
        ("Segment and check", [("Segmentation", ["SCRFD · WiLoR · Hands23", "SAM 2.1 · 24 GB slice"]),
                               ("Audit", ["every contact sheet", "reviewed and signed"]),
                               ("Freeze", ["hashes of references,", "masks and reviews"])]),
        ("Compare and decide", [("Conditions", ["C2 · C3 · C4 · C7", "in-job LPIPS · H100"]),
                                ("Quality scores", ["VBench on lossless", "frames · slice"]),
                                ("Pilot rules", ["and the frozen", "analysis · laptop"])]),
    ]
    probe = plt.figure(dpi=DPI)
    sizes = [[_node_size(probe, head, lines) for head, lines in steps] for _, steps in lanes]
    titles = [_size(probe, name, HEAD, "bold") for name, _ in lanes]
    title_h = titles[0][1]
    plt.close(probe)
    gap, margin, vgap, lane_pad = 5.0, 1.0, 2.6, 1.4
    node_w = [max(w for w, _ in s) for s in sizes]
    node_h = max(h for s in sizes for _, h in s)
    lane_w = [max(w, tw + 1.0) + 2 * lane_pad for w, (tw, _) in zip(node_w, titles)]
    lane_h = title_h + 1.2 + 3 * node_h + 2 * vgap + 2 * lane_pad
    W = 2 * margin + sum(lane_w) + gap * (len(lanes) - 1)
    H = 2 * margin + lane_h
    fig, ax = _canvas(t, W, H)
    x = margin
    for (name, steps), lw, nw in zip(lanes, lane_w, node_w):
        ax.add_patch(FancyBboxPatch((x, margin), lw, lane_h, boxstyle="round,pad=0,rounding_size=1.0",
                                    fc=FILL[mode]["lane"], ec=t["base"], lw=1.0, zorder=0))
        cx = x + lw / 2
        top = margin + lane_h - lane_pad
        ax.text(cx, top - title_h / 2, name, ha="center", va="center", fontsize=HEAD, fontweight="bold", color=t["text2"])
        y = top - title_h - 1.2 - node_h / 2
        for i, (head, lines) in enumerate(steps):
            _node(ax, t, mode, cx, y, nw, node_h, head, lines)
            if i < len(steps) - 1:
                _arrow(ax, t, (cx, y - node_h / 2), (cx, y - node_h / 2 - vgap))
            y -= node_h + vgap
        if x + lw < W - margin - 1:
            _arrow(ax, t, (x + lw, margin + lane_h / 2), (x + lw + gap, margin + lane_h / 2))
        x += lw + gap
    fig.savefig(OUT / f"pipeline_{mode}.png", facecolor=t["surface"])
    plt.close(fig)


def tables(rows, rules):
    """The README's tables, from the same records."""
    lines = ["| Prompt | LPIPS C3 | LPIPS C4 | LPIPS C7 | C7 / C3 | Share C3 | Share C4 | Share C7 |", "|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: r["prompt"]):
        m, s = r["masked_lpips"], r["s_roi"]
        lines.append(f"| {r['prompt']} {SHORT[r['prompt']]} | {m['C3']:.4f} | {m[C4]:.4f} | {m['C7']:.4f} | {m['C7'] / m['C3']:.3f} | "
                     f"{s['C3']:.0%} | {s[C4]:.0%} | {s['C7']:.0%} |")
    med = lambda k, f: st.median(f(r) for r in rows)
    lines.append(f"| **median** | **{med('C3', lambda r: r['masked_lpips']['C3']):.4f}** | **{med(C4, lambda r: r['masked_lpips'][C4]):.4f}** | "
                 f"**{med('C7', lambda r: r['masked_lpips']['C7']):.4f}** | | **{med('C3', lambda r: r['s_roi']['C3']):.0%}** | "
                 f"**{med(C4, lambda r: r['s_roi'][C4]):.0%}** | **{med('C7', lambda r: r['s_roi']['C7']):.0%}** |")
    print("\n".join(lines))
    print()
    print("| λ | Mean outside-region degradation (C4 − C3) | Median masked LPIPS | Qualifies (< 0.01) |\n|---|---|---|---|")
    for lam, v in rules["lambda"]["table"].items():
        print(f"| {float(lam):g} | {v['mean_outside_degradation']:+.4f} | {v['median_masked_lpips']:.4f} | {'yes' if v['qualifies'] else 'no'} |")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    c7 = json.loads((RUN / "exploratory" / "c7.json").read_text())
    rules = json.loads((RUN / "pilot" / "rules.json").read_text())
    rows = c7["rows"]
    for mode in THEMES:
        dose_response(rows, mode)
        per_prompt(rows, mode)
        c2_check(rules, mode)
        allocation_diagram(mode)
        pipeline_diagram(mode)
    qualitative(1, "face_badminton")
    qualitative(12, "hands_guitar")
    sheets([("001-s1001-sh8g6", "face_badminton"), ("007-s1007-sh8g6", "hands_tie"), ("012-s1012-sh8g6", "hands_guitar")])
    gif(1, "comparison_badminton")
    tables(rows, rules)
    for p in sorted(OUT.iterdir()):
        print(f"{p.name}: {p.stat().st_size / 1e6:.2f} MB", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
