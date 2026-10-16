"""ROI assembly by stratum (protocol section 4, equation 8).

    face           R = F
    manipulation   R = H | C | O
    A1             R = H | C          (C4 with hands and contact only)
    full body      R = B
    controls       R = an explicit empty mask, never derived from detectors
"""

import numpy as np

from density.masks.registered import FRAMES, HEIGHT, WIDTH

FACE, MANIPULATION, BODY, CONTROL = "face", "manipulation", "body", "control"
REGIONS = {FACE: ("F",), MANIPULATION: ("H", "C", "O"), BODY: ("B",)}


def empty_mask(shape=(FRAMES, HEIGHT, WIDTH)) -> np.ndarray:
    return np.zeros(shape, dtype=bool)


def assemble_roi(stratum: str, regions: dict[str, np.ndarray] | None, shape=(FRAMES, HEIGHT, WIDTH)) -> np.ndarray:
    """The stratum's reference ROI from region masks F, H, C, O, B."""
    if stratum == CONTROL:
        if regions:
            raise ValueError("controls use an explicit empty mask; detector regions must not be passed")
        return empty_mask(shape)
    if stratum not in REGIONS:
        raise ValueError(f"unknown stratum {stratum!r}")
    roi = empty_mask(shape)
    for name in REGIONS[stratum]:
        roi |= regions[name]
    return roi


def a1_mask(regions: dict[str, np.ndarray]) -> np.ndarray:
    """A1's mask input: hands and contact, without the held object."""
    return regions["H"] | regions["C"]
