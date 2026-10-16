"""PyTorch's deterministic settings for generation and masks on CUDA, and records of what actually ran.

The smoke test's settings: cuDNN benchmarking off, ``CUBLAS_WORKSPACE_CONFIG``
set in the launching environment (it must be set before cuBLAS initializes, so
Python only checks it), and deterministic algorithms in warn-only mode. Every
warning is captured, so an operation without a deterministic implementation
shows up in the record instead of passing silently.

On a pod, ``configure`` also sizes the CPU thread pools to the cgroup's CPU
quota rounded down (``density/cpu_threads.py``; approved 2026-10-07, not a
registered value since no output changed). Without a quota, as on the Mac,
nothing changes.

Registered (2026-10-07, Chase): the pilot and the main run use strict mode,
``configure(strict=True)``. An operation without a deterministic implementation
raises instead of warning, and any warning about deterministic algorithms (the
same test ``WarningLog.deterministic_warnings`` applies) is raised as an error,
also inside ``WarningLog.capture``, so the job stops instead of logging it. The
smoke checks keep warn-only mode and record.

The records also cover the precision switches these settings leave alone. By
default cuDNN runs fp32 convolutions in TF32 (``cudnn.conv.fp32_precision``),
which affects the fp32 VAE and SAM 2.1 on CUDA, and deterministic mode fills
newly allocated memory (``fill_uninitialized_memory``).
"""

import os
import re
import warnings
from contextlib import contextmanager

import torch
from torch.nn.attention import SDPBackend

from density import cpu_threads

CUBLAS_WORKSPACE = ":4096:8"
DETERMINISTIC_MESSAGE = r"(?s).*deterministic"  # a warning about deterministic algorithms (matched case-insensitively)
_strict = False  # set by configure(strict=True)


def configure(strict: bool = False) -> dict:
    """Apply the deterministic settings; requires CUBLAS_WORKSPACE_CONFIG in the environment when CUDA is used.

    ``strict`` (the pilot and main run): operations without a deterministic implementation raise, and warnings
    about deterministic algorithms are errors. Otherwise (the smoke checks) they warn and are recorded.
    """
    global _strict
    if torch.cuda.is_available() and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != CUBLAS_WORKSPACE:
        raise RuntimeError(f"set CUBLAS_WORKSPACE_CONFIG={CUBLAS_WORKSPACE} in the environment that launches the job")
    cpu_threads.configure()
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=not strict)
    warnings.simplefilter("always")
    _strict = strict
    if strict:
        _deterministic_warnings_are_errors()
    return settings()


def _deterministic_warnings_are_errors() -> None:
    warnings.filterwarnings("error", message=DETERMINISTIC_MESSAGE)

def _legacy(read):
    """A legacy TF32 flag; PyTorch refuses to report it once the new API has set conv and RNN precision differently."""
    try:
        return read()
    except RuntimeError as error:
        return f"unreadable ({str(error).split(',')[0]})"


def settings() -> dict:
    """The determinism, precision and attention-backend switches as they stand."""
    cuda = torch.cuda.is_available()
    return {
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "deterministic_warnings_are_errors": _strict,
        "fill_uninitialized_memory": torch.utils.deterministic.fill_uninitialized_memory,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": _legacy(lambda: torch.backends.cuda.matmul.allow_tf32) if cuda else None,
        "cuda_matmul_fp32_precision": torch.backends.cuda.matmul.fp32_precision,
        "cudnn_allow_tf32": _legacy(lambda: torch.backends.cudnn.allow_tf32),
        "cudnn_conv_fp32_precision": torch.backends.cudnn.conv.fp32_precision,
        "cpu_threads": cpu_threads.settings(),
        "sdpa_enabled": {
            "flash": torch.backends.cuda.flash_sdp_enabled(),
            "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math": torch.backends.cuda.math_sdp_enabled(),
            "cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
        },
    }


@contextmanager
def ieee_fp32_convolutions():
    """Run cuDNN's fp32 convolutions in IEEE fp32 instead of the default TF32, for a comparison (not a registered setting)."""
    previous = torch.backends.cudnn.conv.fp32_precision
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    try:
        yield
    finally:
        torch.backends.cudnn.conv.fp32_precision = previous


class WarningLog:
    """Collects every warning raised inside ``capture()``, de-duplicated by category and message.

    After ``configure(strict=True)``, warnings about deterministic algorithms are raised inside ``capture()``
    too, instead of being recorded.
    """

    def __init__(self):
        self.seen: dict[str, dict] = {}

    @contextmanager
    def capture(self, label: str):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if _strict:
                _deterministic_warnings_are_errors()
            yield
        for w in caught:
            message = str(w.message).splitlines()[0][:300]
            key = f"{w.category.__name__}: {message}"
            entry = self.seen.setdefault(key, {"category": w.category.__name__, "message": message, "where": f"{w.filename}:{w.lineno}", "count": 0, "jobs": []})
            entry["count"] += 1
            if label not in entry["jobs"]:
                entry["jobs"].append(label)

    def deterministic_warnings(self) -> list[dict]:
        """Warnings from operations without a deterministic implementation (warn-only mode)."""
        return [e for e in self.seen.values() if re.match(DETERMINISTIC_MESSAGE, e["message"], re.I)]

    def summary(self) -> list[dict]:
        return list(self.seen.values())


SDPA_KERNEL_PATTERNS = {
    "flash": re.compile(r"flash", re.I),
    "mem_efficient": re.compile(r"fmha|efficient_attention|mem_eff", re.I),
    "cudnn": re.compile(r"cudnn.*(attn|sdpa|mha)|(attn|sdpa|mha).*cudnn", re.I),
}


def classify_kernels(names: list[str]) -> list[str]:
    """Attention backends whose kernels appear in ``names``; "math" when none of the fused kernels ran."""
    found = [backend for backend, pattern in SDPA_KERNEL_PATTERNS.items() if any(pattern.search(n) for n in names)]
    return found or ["math"]


def cuda_kernel_names(fn) -> list[str]:
    """Names of the CUDA kernels ``fn`` launches, from the PyTorch profiler; raises if the profiler sees none."""
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    names = sorted({e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA})
    if not names:
        raise RuntimeError("the profiler recorded no CUDA kernels")
    return names


SDPA_BACKENDS = {int(value): name for name, value in SDPBackend.__members__.items()}


@contextmanager
def record_sdpa_choices():
    """Record the backend PyTorch selects for every ``scaled_dot_product_attention`` call, with its shapes.

    ``torch._fused_sdp_choice`` is the dispatcher's own selection for the exact
    tensors passed, so it names the backend directly; kernel names from the
    profiler corroborate it. Diffusers and SAM 2.1 look the function up on
    ``torch.nn.functional`` at call time, so both are covered.
    """
    original = torch.nn.functional.scaled_dot_product_attention
    calls: list[dict] = []

    def wrapper(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
        try:
            choice = torch._fused_sdp_choice(query, key, value, attn_mask, dropout_p, is_causal, scale=scale, enable_gqa=enable_gqa)
            backend = SDPA_BACKENDS.get(int(choice), str(choice))
        except NotImplementedError:  # MPS has no _fused_sdp_choice; the record matters on CUDA
            backend = f"not reported on {query.device.type}"
        calls.append(
            {
                "backend": backend,
                "query": list(query.shape),
                "key": list(key.shape),
                "dtype": str(query.dtype).removeprefix("torch."),
                "mask": attn_mask is not None,
            }
        )
        return original(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)

    torch.nn.functional.scaled_dot_product_attention = wrapper
    try:
        yield calls
    finally:
        torch.nn.functional.scaled_dot_product_attention = original


def summarize_sdpa(calls: list[dict]) -> list[dict]:
    """Distinct (backend, query shape, key shape, dtype) combinations with their call counts."""
    counts: dict[tuple, int] = {}
    for c in calls:
        key = (c["backend"], tuple(c["query"]), tuple(c["key"]), c["dtype"], c["mask"])
        counts[key] = counts.get(key, 0) + 1
    return [{"backend": b, "query": list(q), "key": list(k), "dtype": d, "mask": m, "calls": n} for (b, q, k, d, m), n in counts.items()]
