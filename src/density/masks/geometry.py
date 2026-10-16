"""Mask geometry: the registered disk dilation, boxes and IoU, and run-length encoding.

Every dilation uses one structuring element, the Euclidean disk of radius r:
pixel p is in dilate(M, r) when some mask pixel q has |p - q|^2 <= r^2. It is
computed exactly from the Euclidean distance transform of the complement.
Masks are bool arrays of shape (frames, height, width) or (height, width).
"""

import numpy as np
from scipy import ndimage


def disk_offsets(radius: int) -> np.ndarray:
    """(dy, dx) offsets of the registered structuring element: dx^2 + dy^2 <= r^2."""
    r = np.arange(-radius, radius + 1)
    dy, dx = np.meshgrid(r, r, indexing="ij")
    keep = dy**2 + dx**2 <= radius**2
    return np.stack([dy[keep], dx[keep]], axis=1)


def _dilate_frame(frame: np.ndarray, radius: int) -> np.ndarray:
    if not frame.any():
        return np.zeros_like(frame, dtype=bool)
    if radius == 0:
        return frame.astype(bool)
    # Squared integer distances, so the comparison with r^2 is exact.
    distance = ndimage.distance_transform_edt(~frame.astype(bool))
    return np.rint(distance**2) <= radius**2


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Disk dilation by ``radius`` pixels, frame by frame (never through time), clipped to the image."""
    if radius < 0:
        raise ValueError("radius must be non-negative")
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim == 2:
        return _dilate_frame(mask, radius)
    return np.stack([_dilate_frame(frame, radius) for frame in mask])


def mask_box(mask: np.ndarray) -> np.ndarray | None:
    """Tight box (x0, y0, x1, y1) of a 2-D mask, with x1 and y1 exclusive; None if empty."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float64)


def box_iou(a, b) -> float:
    """IoU of two (x0, y0, x1, y1) boxes; 0 when either is missing or empty."""
    if a is None or b is None:
        return 0.0
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def rle_encode(mask: np.ndarray) -> dict:
    """Row-major run lengths of a bool mask, starting with a run of False (possibly empty)."""
    mask = np.asarray(mask, dtype=bool)
    flat = mask.ravel()
    change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    bounds = np.concatenate([[0], change, [flat.size]])
    counts = np.diff(bounds).tolist()
    if flat.size and flat[0]:
        counts = [0] + counts
    return {"shape": list(mask.shape), "counts": counts}


def rle_decode(rle: dict) -> np.ndarray:
    shape, counts = tuple(rle["shape"]), rle["counts"]
    values = np.arange(len(counts)) % 2 == 1  # runs alternate False, True, False, ...
    flat = np.repeat(values, counts)
    if flat.size != int(np.prod(shape)):
        raise ValueError(f"run lengths cover {flat.size} pixels, shape needs {int(np.prod(shape))}")
    return flat.reshape(shape)
