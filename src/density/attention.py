"""The registered attention backend for generation on CUDA.

Registered (2026-10-06, smoke test): flash attention. On the H100 at full resolution, PyTorch's dispatcher picked
cuDNN attention. That was bitwise within a process, but a dense run repeated in a new process differed from the
first step, and cuDNN's result for a query depends on how many queries it is given, so the HSA fast path and its
oracle part at the first skipped iteration. Flash attention gives each query the same result whatever the query
count, and its runs are bitwise across processes; with it, the fast path equals the oracle and HSA with every
group at full steps equals dense, bitwise (results/checks/smoke_test.md).

Every forward pass of generation runs inside ``generation_attention``: the Transformer in bf16 uses flash
attention, and memory-efficient attention (also row-invariant) is allowed only where flash cannot run, which is
fp32 (the VAE's attention and the fp32 test fixtures). On CPU and MPS, PyTorch's own choice is kept.
"""

from contextlib import nullcontext

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

GENERATION_BACKENDS = (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION)  # in priority order


def generation_attention(device):
    """Context for one forward pass on ``device``: the registered backends on CUDA, no change elsewhere."""
    if torch.device(device).type != "cuda":
        return nullcontext()
    return sdpa_kernel(list(GENERATION_BACKENDS), set_priority=True)
