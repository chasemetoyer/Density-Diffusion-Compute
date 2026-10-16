"""Detectron2's ROIAlign through torchvision's kernel, for inference only (registered 2026-10-07 by Chase).

With deterministic algorithms on and the input on CUDA, torchvision 0.29.1's ``roi_align`` swaps its kernel for a
pure-Python version, lazily ``torch.compile``'d, which exists to make the backward pass deterministic
(torchvision/ops/roi_align.py). In the smoke test's Session E it made Mask R-CNN take about 20 s a frame and Hands23
16 s a keyframe on the slice. The masks need only the forward pass, and the kernel's forward pass computes each output
on one thread, without atomics. Called directly, it took 0.08 s and 0.15 s, and repeats were bitwise identical. Its
outputs differ from the Python version's only in rounding: boxes by at most 0.00012 px, mask IoU at least 0.99997.
The smoke-1 and smoke-2 masks were identical either way.

``install`` points Detectron2 at the kernel. The module comes from importlib: ``import detectron2.layers.roi_align``
would bind the package attribute, which ``detectron2.layers`` rebinds to the function itself, so a patch through it
binds nothing. ``checked`` wraps one model call and raises unless the kernel ran during it. On CPU the kernel is the
one torchvision calls anyway, so outputs on the Mac are unchanged.
"""

import importlib
from contextlib import contextmanager

import torch

_calls = 0


def _kernel(input, rois, output_size, spatial_scale, sampling_ratio, aligned):
    global _calls
    if torch.is_grad_enabled():
        raise RuntimeError("the registered ROIAlign kernel is for inference only: its backward pass is not deterministic")
    _calls += 1
    return torch.ops.torchvision.roi_align(input, rois, spatial_scale, output_size[0], output_size[1], sampling_ratio, aligned)


def install() -> None:
    """Route Detectron2's ROIAlign layers to the kernel (idempotent)."""
    importlib.import_module("detectron2.layers.roi_align").roi_align = _kernel


def calls() -> int:
    return _calls


@contextmanager
def checked(label: str):
    """Raise after the block unless the registered kernel ran inside it."""
    before = _calls
    yield
    if _calls == before:
        raise RuntimeError(f"{label}: the registered ROIAlign kernel was not called")
