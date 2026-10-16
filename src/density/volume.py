"""Space used on the density network volume, measured with du before each step or batch.

On RunPod's network volume (MooseFS), df reports the whole storage cluster, about 1 PB, not our 125 GB
volume. So usage is measured with one du over /workspace: a single invocation counts the hardlinks
between uv's cache and the venv once.

Registered (2026-10-07, Chase): a runner stops cleanly, before starting a step or job, once the volume holds 112 GB.
Chase increased the volume from 100 to 125 GB on 2026-10-07, so the lossless copies of every condition fit, and moved
the stop from 90 to 112 GB. 112 GB is read as 112 x 10^9 bytes, the conservative reading (about 104.3 GiB). The stop
uses du's allocated size (``du -sk``), the larger of du's two figures: on 2026-10-07 the volume held 57.7 GB allocated
against 48.3 GB of apparent size.

How a session measures it (registered by Chase on 2026-10-07, second round; ``Room``): one full du at the session's
start, then a running total that adds, after each job, the allocated size (st_blocks x 512, du's own measure, never the
byte length) of every file the job wrote, and a full du again every 30 minutes. A running total that reaches the stop
triggers a full du, which decides. Overwritten files count in full and deleted ones are never subtracted, so the total
can only overcount between full checks; each full du records any undercount, which would mean a write the total
missed. A du of the volume took about 52.5 s on 2026-10-07, so a du before every prompt idled the GPU for that long.
"""

import os
import subprocess
import time
from pathlib import Path

VOLUME = Path("/workspace")
STOP_BYTES = 112 * 10**9  # registered (Chase, 2026-10-07); 90 GB before the volume grew to 125 GB
FULL_CHECK_SECONDS = 30 * 60  # registered (Chase, 2026-10-07): a full du at a session's start and every 30 minutes


class VolumeFull(RuntimeError):
    pass


def usage(root: Path | None = None, apparent: bool = False, timeout: int = 1800) -> dict:
    """Bytes under ``root`` (default: the volume) from du: allocated blocks (``du -sk``), or file sizes with ``apparent`` (GNU du)."""
    root = VOLUME if root is None else root
    command = ["du", "-s", "--apparent-size", "--block-size=1", str(root)] if apparent else ["du", "-sk", str(root)]
    start = time.perf_counter()
    r = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    seconds = time.perf_counter() - start
    fields = r.stdout.split()
    if not fields or not fields[0].isdigit():
        raise RuntimeError(f"{' '.join(command)} gave no total (exit {r.returncode}): {r.stderr.strip()[:300]}")
    used = int(fields[0]) * (1 if apparent else 1024)
    # du exits 1 when a file vanishes while it walks the tree; the total still stands, and the messages are kept
    return {"root": str(root), "used_bytes": used, "measure": "apparent size" if apparent else "allocated blocks", "seconds": seconds,
            "du_exit": r.returncode, "du_messages": r.stderr.strip().splitlines()[:5]}


def require_room(root: Path | None = None, stop_bytes: int | None = None) -> dict:
    """The volume's allocated usage; raises VolumeFull once it holds ``stop_bytes`` (default: the registered 112 GB) or more."""
    root = VOLUME if root is None else root
    stop_bytes = STOP_BYTES if stop_bytes is None else stop_bytes
    u = usage(root, apparent=False)
    u["stop_bytes"] = stop_bytes
    if u["used_bytes"] >= stop_bytes:
        raise VolumeFull(f"{root} holds {u['used_bytes'] / 1e9:.1f} GB (du), at or past the {stop_bytes / 1e9:g} GB stop; free space or ask Chase before going on")
    return u


def allocated_bytes(paths) -> int:
    """The allocated size (st_blocks x 512, as du counts it) of the files at or under ``paths``; a missing path counts 0."""
    total = 0
    for path in map(Path, paths):
        if path.is_file():
            total += path.stat().st_blocks * 512
        elif path.is_dir():
            for root, _, files in os.walk(path):
                for name in files:
                    try:
                        total += os.stat(os.path.join(root, name)).st_blocks * 512
                    except FileNotFoundError:  # removed while counting (a temporary file); nothing is left to count
                        pass
    return total


class Room:
    """The registered volume stop within a session: a full du at the start, then a running total of each job's written
    files, with a full du every 30 minutes and whenever the running total reaches the stop (see the module docstring)."""

    def __init__(self, root: Path | None = None, stop_bytes: int | None = None, every: float = FULL_CHECK_SECONDS, clock=time.monotonic):
        self.root, self.every, self.clock = root, every, clock
        self.stop_bytes = STOP_BYTES if stop_bytes is None else stop_bytes
        self.used: int | None = None
        self.log: list[dict] = []
        self.full("session start")

    def full(self, reason: str) -> dict:
        """A full du; raises VolumeFull at or past the stop."""
        estimate = self.used
        u = require_room(self.root, self.stop_bytes)
        self.used, self.last_full = u["used_bytes"], self.clock()
        entry = {"kind": "full", "reason": reason, "used_bytes": u["used_bytes"], "seconds": u["seconds"]}
        if estimate is not None:
            entry.update(running_estimate_bytes=estimate, undercount_bytes=max(0, u["used_bytes"] - estimate))
        self.log.append(entry)
        return entry

    def before_job(self, job_id: str) -> dict | None:
        """Before each job: a full du every 30 minutes, or when the running total has reached the stop."""
        if self.clock() - self.last_full >= self.every:
            return self.full(f"30 minutes since the last full du, before {job_id}")
        if self.used >= self.stop_bytes:
            return self.full(f"the running total reached the stop, before {job_id}")
        return None

    def after_job(self, job_id: str, paths) -> dict:
        added = allocated_bytes(paths)
        self.used += added
        entry = {"kind": "running", "after": job_id, "added_bytes": added, "used_bytes": self.used}
        self.log.append(entry)
        return entry
