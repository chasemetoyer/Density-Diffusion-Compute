"""The registered numerics for every scorer, as one scoped context (density.metrics.registered.PRECISION).

Inside ``scoring(device)``: cuDNN's fp32 convolutions and CUDA's fp32 matmuls in IEEE fp32 (no TF32), and on CUDA
scaled-dot-product attention limited to flash attention, then memory-efficient attention. Each switch is restored on
exit, so LPIPS inside a generation job leaves the job's own settings (the VAE's TF32, generation's attention) as it
found them.
"""

from contextlib import ExitStack, contextmanager

import torch
from torch.nn.attention import sdpa_kernel

from density.metrics.registered import ATTENTION_BACKENDS, CONV_FP32_PRECISION, MATMUL_FP32_PRECISION


@contextmanager
def scoring(device):
    previous = (torch.backends.cudnn.conv.fp32_precision, torch.backends.cuda.matmul.fp32_precision)
    torch.backends.cudnn.conv.fp32_precision = CONV_FP32_PRECISION
    torch.backends.cuda.matmul.fp32_precision = MATMUL_FP32_PRECISION
    try:
        with ExitStack() as stack:
            if torch.device(device).type == "cuda":
                stack.enter_context(sdpa_kernel(list(ATTENTION_BACKENDS), set_priority=True))
            stack.enter_context(torch.no_grad())
            yield
    finally:
        torch.backends.cudnn.conv.fp32_precision, torch.backends.cuda.matmul.fp32_precision = previous


def settings() -> dict:
    return {"cudnn_conv_fp32_precision": torch.backends.cudnn.conv.fp32_precision, "cuda_matmul_fp32_precision": torch.backends.cuda.matmul.fp32_precision}
