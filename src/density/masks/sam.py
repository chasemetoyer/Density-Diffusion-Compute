"""SAM 2.1 Hiera-small video propagation behind the ``tracking.Propagator`` interface."""

import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from density.masks.detectors import build_sam2_predictor
from density.masks.registered import SAM_CONV_FP32_PRECISION, SAM_JPEG_QUALITY


@contextmanager
def conv_precision(value: str = SAM_CONV_FP32_PRECISION):
    """cuDNN's fp32 convolution precision for SAM 2.1 (registered as SAM_CONV_FP32_PRECISION); restored afterwards."""
    previous = torch.backends.cudnn.conv.fp32_precision
    torch.backends.cudnn.conv.fp32_precision = value
    try:
        yield
    finally:
        torch.backends.cudnn.conv.fp32_precision = previous


def _memory_to_curr_dtype(module, args, kwargs):
    curr = kwargs["curr"]
    dtype = (curr[0] if isinstance(curr, list) else curr).dtype
    for key in ("memory", "memory_pos"):
        if kwargs.get(key) is not None:
            kwargs[key] = kwargs[key].to(dtype)
    return args, kwargs


class SAM2Propagator:
    """Box-prompted masks for every track over one clip.

    Each call resets SAM's state, adds every prompt, then propagates forward
    from the earliest prompt and, when that prompt is not on frame 0, backward
    from it. Masks are SAM's logits thresholded at 0, at video resolution.

    Precision (registered as ``SAM_PRECISION`` and ``SAM_BUILD``): SAM runs in
    fp32 without autocast, identically on the Mac and the pod. SAM 2 stores
    memory features in bfloat16; when they are concatenated with fp32 object
    pointers PyTorch promotes them to fp32, but on frames without object
    pointers they stayed bf16 and reached fp32 layers. A pre-hook casts the
    memory to fp32 at the memory-attention input on every frame, which is the
    same exact conversion the promotion performs.
    """

    def __init__(self, frames_rgb: np.ndarray, device: str = "cpu"):
        self.device = device
        self.shape = frames_rgb.shape[:3]
        self.predictor = build_sam2_predictor(device)
        self.predictor.memory_attention.register_forward_pre_hook(_memory_to_curr_dtype, with_kwargs=True)
        self._jpegs = tempfile.TemporaryDirectory(prefix="sam2_frames_")
        for i, frame in enumerate(frames_rgb):
            Image.fromarray(frame).save(Path(self._jpegs.name) / f"{i}.jpg", quality=SAM_JPEG_QUALITY, subsampling=0)
        with torch.inference_mode(), conv_precision():
            self.state = self.predictor.init_state(video_path=self._jpegs.name)
        self.calls = 0
        self.seconds = 0.0

    def propagate(self, prompts: dict[int, dict[int, np.ndarray]], num_frames: int) -> dict[int, np.ndarray]:
        if num_frames != self.shape[0]:
            raise ValueError(f"clip has {self.shape[0]} frames, asked for {num_frames}")
        start = time.perf_counter()
        masks = {track: np.zeros(self.shape, dtype=bool) for track in prompts}
        with torch.inference_mode(), conv_precision():
            self.predictor.reset_state(self.state)
            for track in sorted(prompts):
                for frame in sorted(prompts[track]):
                    self.predictor.add_new_points_or_box(self.state, frame_idx=frame, obj_id=track, box=np.asarray(prompts[track][frame], dtype=np.float32))
            first = min(frame for by_frame in prompts.values() for frame in by_frame)
            passes = [False] + ([True] if first > 0 else [])
            for reverse in passes:
                for frame, obj_ids, logits in self.predictor.propagate_in_video(self.state, start_frame_idx=first, reverse=reverse):
                    for obj_id, logit in zip(obj_ids, logits):
                        masks[int(obj_id)][frame] = (logit[0] > 0).cpu().numpy()
        self.calls += 1
        self.seconds += time.perf_counter() - start
        return masks

    def close(self):
        self._jpegs.cleanup()
