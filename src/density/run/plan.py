"""The pilot's and the main run's job grids (protocol sections 5, 6, 10 and 12), their planning costs, and the
measured times behind the cost table.

Grids, as section 12's table states them (RE: one dense 40-step generation):

    pilot   settings      12 holdout prompts x 2 shift/guidance candidates, C1, seed 1          24 RE
            lambda        12 holdout prompts, C3 and C4 at lambda 1, 3, 10, HSA-50, seed 1     24 RE
            reproduction  12 holdout prompts, C2, seed 1                                        6 RE
    main    primary s1    72 primary prompts: C1, C2, C3, C4                                  180 RE
            primary s2    72 primary prompts: C1, C3, C4                                      144 RE
            C5 and C6     18 face + 18 manipulation prompts, seed 1                            36 RE
            A1            36 manipulation prompts, seed 1                                      18 RE
            full body     16 body prompts: C1, C3, C4, seed 1                                  32 RE
            controls      16 control prompts: C1, C3, C4, seed 1                               32 RE
            75 percent    8 face + 8 manipulation prompts: C3, C4 at HSA-75B, seed 1           24 RE
            switch        8 face + 8 manipulation prompts: C3, C4 at switch-20 and -30, seed 1 32 RE

The C5/C6, 75 percent and switch subsets are drawn at the freeze, before any result exists, from one seeded
permutation of each stratum's accepted primary prompts: C5 and C6 take its first 18, and the 75 percent and switch
blocks its first 8 (density.run.freeze).
"""

from dataclasses import dataclass

from density.allocation import LAMBDA_GRID  # noqa: F401  (registered in Milestone 3)
from density.masks.roi import BODY, CONTROL, FACE, MANIPULATION

SETTINGS = {"sh8g6": {"shift": 8.0, "guidance": 6.0}, "sh5g5": {"shift": 5.0, "guidance": 5.0}}  # section 5's candidates
TIE_SETTING = "sh8g6"  # ties select 8 / 6
PRIMARY_PRESET = "HSA-50"
SUBSET_SEED = 20261007  # the seeded permutation behind the C5/C6, 75 percent and switch subsets (registered by Chase, 2026-10-07)
SUBSETS = {"C5_C6": 18, "budget_switch": 8}  # per primary stratum

# Planning cost in RE (section 12 rounds HSA-50's token-step fraction 0.496 to 0.5).
PLANNING_RE = {("C1", None): 1.0, ("C2", None): 0.5, "HSA-50": 0.5, "HSA-75B": 0.75, "switch-20": 0.5, "switch-30": 0.5}

# Measured times. Generation on the H100 SXM with flash attention (smoke test Session D): dense C1 73.8 s end to end;
# HSA-50 C3 39.4 s and C4 39.6 s. Masks on the 24 GB slice with the thread cap and the ROIAlign kernel (Session E):
# 18 s per face reference, 346 s per manipulation reference, 94 s per body reference; controls compute nothing.
SECONDS = {
    "C1": 73.8, "HSA-50": 39.6, "C2": 73.8 * 0.5,  # C2: 20 dense steps, half of C1's denoising (estimate)
    "HSA-75B": 73.8 * 0.748752, "switch-20": 39.6, "switch-30": 39.6,  # by token-step fraction (estimate)
    "mask": {FACE: 18.0, MANIPULATION: 346.0, BODY: 94.0, CONTROL: 0.0},
}
SECONDS_SOURCE = "smoke test Sessions D (H100, flash attention) and E (slice, thread cap and ROIAlign kernel); C2, HSA-75B and the switches scaled by token-step fraction"
METRIC_SECONDS = None  # placeholder until measured (protocol section 12); the Mac dry run's times are not representative
# Hourly rates RunPod billed for these pods in the smoke test (2026-10-06/07) and quoted again on 2026-10-07. Use the
# rate quoted when a pod is created.
RATE_PER_HOUR = {"h100": 3.49, "slice": 0.59}


@dataclass(frozen=True)
class GridItem:
    condition: str
    preset: str | None = None
    lam: float | None = None  # None for rules without lambda; "selected" C4/A1 use the registered lambda


def planning_re(condition: str, preset: str | None) -> float:
    return PLANNING_RE[(condition, None)] if preset is None else PLANNING_RE[preset]


def gpu_seconds(condition: str, preset: str | None) -> float:
    return SECONDS[condition] if preset is None else SECONDS[preset]


def pilot_conditions() -> list[GridItem]:
    """Holdout conditions after the setting is chosen: C3 and C4 at each lambda on HSA-50, and C2."""
    return [GridItem("C3", PRIMARY_PRESET), *(GridItem("C4", PRIMARY_PRESET, lam) for lam in LAMBDA_GRID), GridItem("C2")]


def main_blocks(lam: float) -> list[dict]:
    """The main grid's reduced-budget blocks: which prompts (by role), which seed index, which conditions."""
    hsa = PRIMARY_PRESET
    return [
        {"block": "primary seed 1", "prompts": "primary", "seed": 1, "items": [GridItem("C2"), GridItem("C3", hsa), GridItem("C4", hsa, lam)]},
        {"block": "primary seed 2", "prompts": "primary", "seed": 2, "items": [GridItem("C3", hsa), GridItem("C4", hsa, lam)]},
        {"block": "C5 and C6", "prompts": "subset:C5_C6", "seed": 1, "items": [GridItem("C5", hsa), GridItem("C6", hsa)]},
        {"block": "A1", "prompts": "primary:manipulation", "seed": 1, "items": [GridItem("A1", hsa, lam)]},
        {"block": "full body", "prompts": "secondary", "seed": 1, "items": [GridItem("C3", hsa), GridItem("C4", hsa, lam)]},
        {"block": "controls", "prompts": "control", "seed": 1, "items": [GridItem("C3", hsa), GridItem("C4", hsa, lam)]},
        {"block": "75 percent", "prompts": "subset:budget_switch", "seed": 1, "items": [GridItem("C3", "HSA-75B"), GridItem("C4", "HSA-75B", lam)]},
        {"block": "switch points", "prompts": "subset:budget_switch", "seed": 1,
         "items": [GridItem(c, p, lam if c == "C4" else None) for p in ("switch-20", "switch-30") for c in ("C3", "C4")]},
    ]


REFERENCE_SEEDS = {"holdout": (1,), "primary": (1, 2), "secondary": (1,), "control": (1,), "reserve": (1, 2)}  # seed indices per block
