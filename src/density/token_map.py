"""Pixel masks to token coverage (protocol section 4, equations 10-12).

A token's annotation cell covers 16 x 16 pixels and a group of video frames.

    a[f, r, c] = mean of the binary mask over cell (r, c) of pixel frame f      (eq. 10)
    F_0 = {0},  F_j = {4j - 3, 4j - 2, 4j - 1, 4j}                             (eq. 11)
    u[j, r, c] = max over f in F_j of a[f, r, c]                                (eq. 12)
    m[j, r, c] = max of u over the 3 x 3 spatial neighbourhood, zero outside   (eq. 12)

There is no temporal expansion and no binarization; m keeps fractional values.
Flattened in (frame, row, column) order, m follows the Transformer's token IDs.
"""

import torch
import torch.nn.functional as F

CELL = 16  # pixels per token cell side (VAE factor 8 x patch 2)
TEMPORAL = 4  # pixel frames per latent frame after the first


def cell_coverage(mask: torch.Tensor) -> torch.Tensor:
    """Equation 10: (frames, H, W) binary mask -> (frames, H / 16, W / 16) fractional coverage, fp32."""
    frames, height, width = mask.shape
    if height % CELL or width % CELL:
        raise ValueError(f"mask size {height} x {width} is not a multiple of {CELL}")
    cells = mask.float().reshape(frames, height // CELL, CELL, width // CELL, CELL)
    return cells.mean(dim=(2, 4))


def frame_groups(num_pixel_frames: int) -> list[list[int]]:
    """Equation 11: the pixel frames behind each latent frame."""
    if (num_pixel_frames - 1) % TEMPORAL:
        raise ValueError(f"{num_pixel_frames} pixel frames do not map to whole latent frames")
    num_latent = (num_pixel_frames - 1) // TEMPORAL + 1
    return [[0]] + [list(range(TEMPORAL * j - 3, TEMPORAL * j + 1)) for j in range(1, num_latent)]


def latent_frame_max(coverage: torch.Tensor) -> torch.Tensor:
    """Equation 12, first part: the largest coverage across each latent frame's pixel-frame group."""
    return torch.stack([coverage[group].amax(dim=0) for group in frame_groups(coverage.shape[0])])


def spatial_dilate(coverage: torch.Tensor) -> torch.Tensor:
    """Equation 12, second part: 3 x 3 maximum within each latent frame, zero outside the grid."""
    padded = F.pad(coverage[:, None], (1, 1, 1, 1), value=0.0)
    return F.max_pool2d(padded, kernel_size=3, stride=1)[:, 0]


def coverage_grid(mask: torch.Tensor) -> torch.Tensor:
    """Equations 10-12: (pixel frames, H, W) binary mask -> m with shape (latent frames, H / 16, W / 16)."""
    return spatial_dilate(latent_frame_max(cell_coverage(mask)))


def token_coverage(mask: torch.Tensor) -> torch.Tensor:
    """m for every token, in the Transformer's token order."""
    return coverage_grid(mask).flatten()
