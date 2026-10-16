"""CPU thread pools on a pod, sized to its cgroup CPU quota rounded down.

Pods see every host core (128 on the slice's host) while the cgroup allows a few CPUs (3.4 on the slice, 22.1 on the
H100). Pools sized to the host oversubscribe the quota. In the smoke test's Session E, onnxruntime's default pool put
SCRFD at 141 OS threads, the cgroup throttled it in every period, and a frame took 0.78 s; with 3 threads it took
0.067 s.

Chase approved (2026-10-07): every pool at the quota rounded down, with one inter-op thread for torch and onnxruntime.
scripts/pod/env.sh sizes OpenMP, MKL and OpenBLAS the same way before Python starts. It is not a registered value: on
the slice every detection was bitwise identical across 1, 3 and 4 threads and the earlier defaults, and so were the
smoke-1 and smoke-2 masks at the defaults and at 3 threads.

Without a quota (the Mac) nothing changes. There thread counts do change results: SCRFD's with onnxruntime's pool, SAM's
with torch's. The Mac's validated masks come from torch's default of 8 threads and onnxruntime's default.
"""

from pathlib import Path

import torch

CPU_MAX = Path("/sys/fs/cgroup/cpu.max")


def quota_threads(cpu_max: Path = CPU_MAX) -> int | None:
    """The cgroup CPU quota rounded down, at least 1; None without a quota (the Mac, or an unlimited cgroup)."""
    try:
        fields = Path(cpu_max).read_text().split()
    except OSError:
        return None
    if len(fields) != 2 or fields[0] == "max":
        return None
    return max(1, int(fields[0]) // int(fields[1]))


def configure(cpu_max: Path = CPU_MAX) -> dict:
    """On a pod, torch's intra-op pool at the quota and its inter-op pool at 1, and OpenCV's pool at the quota.

    onnxruntime's pool is set per session (``density.masks.pipeline.Models``). torch accepts an inter-op count only
    before any parallel work, so this runs first in every job (``density.determinism.configure`` calls it).
    """
    n = quota_threads(cpu_max)
    out = {"quota_threads": n}
    if n is None:
        return out
    torch.set_num_threads(n)
    if torch.get_num_interop_threads() != 1:
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError as error:  # parallel work already started: recorded, and visible in the counts below
            out["inter_op_error"] = str(error).splitlines()[0]
    try:
        import cv2

        cv2.setNumThreads(n)
    except ImportError:
        pass
    out.update(torch_intra_op=torch.get_num_threads(), torch_inter_op=torch.get_num_interop_threads())
    return out


def settings() -> dict:
    return {"quota_threads": quota_threads(), "torch_intra_op": torch.get_num_threads(), "torch_inter_op": torch.get_num_interop_threads()}
