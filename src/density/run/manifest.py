"""The run manifest: every job of a pilot or main run, its dependencies, the machine it needs and its planned cost.

A job is one unit of work with a stable ID:

    embed-<prompt>                                   T5 embeddings of one prompt (H100)
    ref-<prompt>-s<seed>-<setting>                   the dense C1 reference: lossless frames, MP4 (H100)
    mask-<prompt>-s<seed>-<setting>                  masks and the audit sheet of one reference (slice)
    gen-<prompt>-s<seed>-<setting>-<cond>[-<preset>][-l<lambda>]
                                                     one reduced-budget condition, scored in-job against its reference (H100)
    score-<prompt>-s<seed>-<setting>                 quality and detector checks of a reference and all its conditions (slice)

Jobs enter the manifest in phases, which enforces the protocol's order: references and masks first; replacement
references after the audit review; reduced-budget conditions and scores only at the freeze (density.run.freeze).
Every version of the manifest is kept (manifest_v<n>.json) and the current one is manifest.json.
"""

from dataclasses import asdict, dataclass, field

from density import prompts as prompt_set
from density.run import plan

KINDS = ("embed", "reference", "mask", "condition", "score")
MACHINE = {"embed": "h100", "reference": "h100", "condition": "h100", "mask": "slice", "score": "slice"}


@dataclass(frozen=True)
class Job:
    kind: str
    prompt_id: int
    stratum: str
    block: str
    seed: int | None = None
    setting: str | None = None
    condition: str | None = None
    preset: str | None = None
    lam: float | None = None
    grid_block: str | None = None  # the section 12 block a reduced-budget job belongs to
    depends: tuple[str, ...] = field(default=())

    @property
    def id(self) -> str:
        p = f"{self.prompt_id:03d}"
        if self.kind == "embed":
            return f"embed-{p}"
        stem = f"{p}-s{self.seed}-{self.setting}"
        if self.kind == "reference":
            return f"ref-{stem}"
        if self.kind == "mask":
            return f"mask-{stem}"
        if self.kind == "score":
            return f"score-{stem}"
        return f"gen-{stem}-{self.condition}" + (f"-{self.preset}" if self.preset else "") + (f"-l{self.lam:g}" if self.lam is not None else "")

    @property
    def group(self) -> str:
        """The prompt-seed-setting a reference, its masks, conditions and scores share."""
        return f"{self.prompt_id:03d}-s{self.seed}-{self.setting}"

    @property
    def machine(self) -> str:
        return MACHINE[self.kind]

    @property
    def planning_re(self) -> float:
        if self.kind == "reference":
            return plan.planning_re("C1", None)
        if self.kind == "condition":
            return plan.planning_re(self.condition, self.preset)
        return 0.0

    def to_dict(self) -> dict:
        return {**asdict(self), "id": self.id, "depends": list(self.depends), "machine": self.machine, "planning_re": self.planning_re}

    @classmethod
    def from_dict(cls, d: dict) -> "Job":
        fields = {k: d[k] for k in ("kind", "prompt_id", "stratum", "block", "seed", "setting", "condition", "preset", "lam", "grid_block")}
        return cls(**fields, depends=tuple(d["depends"]))


def embed_job(row: dict) -> Job:
    return Job("embed", int(row["id"]), row["stratum"], row["block"])


def reference_jobs(row: dict, seed_index: int, setting: str) -> list[Job]:
    """The reference and its mask job for one prompt, seed and setting."""
    pid = int(row["id"])
    seed = prompt_set.seeds(pid)[seed_index - 1]
    ref = Job("reference", pid, row["stratum"], row["block"], seed, setting, "C1", depends=(f"embed-{pid:03d}",))
    mask = Job("mask", pid, row["stratum"], row["block"], seed, setting, depends=(ref.id,))
    return [ref, mask]


def condition_job(ref: Job, item: plan.GridItem, grid_block: str, mask_job_id: str) -> Job:
    return Job("condition", ref.prompt_id, ref.stratum, ref.block, ref.seed, ref.setting, item.condition, item.preset, item.lam, grid_block,
               depends=(ref.id, mask_job_id))


def score_job(ref: Job, condition_ids: list[str], mask_job_id: str) -> Job:
    return Job("score", ref.prompt_id, ref.stratum, ref.block, ref.seed, ref.setting, depends=(ref.id, mask_job_id, *condition_ids))


def pilot_initial(rows: list[dict]) -> list[Job]:
    """The pilot's first phase: embeddings, then both candidate settings' references and masks for every holdout prompt."""
    holdout = [r for r in rows if r["block"] == "holdout"]
    jobs = [embed_job(r) for r in holdout]
    for r in holdout:
        for setting in plan.SETTINGS:
            jobs += reference_jobs(r, 1, setting)
    return jobs


def main_initial(rows: list[dict], setting: str) -> list[Job]:
    """The main run's first phase: embeddings of every non-holdout prompt (reserves included, so a replacement never
    waits for one), and references and masks for the primary, secondary and control blocks."""
    jobs = [embed_job(r) for r in rows if r["block"] != "holdout"]
    for r in rows:
        if r["block"] in ("primary", "secondary", "control"):
            for s in plan.REFERENCE_SEEDS[r["block"]]:
                jobs += reference_jobs(r, s, setting)
    return jobs


def planned_cost(jobs: list[Job]) -> dict:
    """RE and estimated machine-hours by kind and block, from section 12's planning costs and the measured times."""
    table: dict[str, dict] = {}
    for job in jobs:
        key = job.grid_block or {"reference": f"references ({job.block})", "mask": f"masks ({job.block})", "embed": "embeddings", "score": "scores"}[job.kind]
        row = table.setdefault(key, {"jobs": 0, "re": 0.0, "machine": job.machine, "seconds": 0.0})
        row["jobs"] += 1
        row["re"] += job.planning_re
        if job.kind == "reference":
            row["seconds"] += plan.gpu_seconds("C1", None)
        elif job.kind == "condition":
            row["seconds"] += plan.gpu_seconds(job.condition, job.preset)
        elif job.kind == "mask":
            row["seconds"] += plan.SECONDS["mask"][job.stratum]
    for row in table.values():
        row["hours"] = row["seconds"] / 3600
        row["dollars"] = row["hours"] * plan.RATE_PER_HOUR[row["machine"]]
    totals = {m: {"re": sum(r["re"] for r in table.values() if r["machine"] == m), "hours": sum(r["hours"] for r in table.values() if r["machine"] == m)} for m in plan.RATE_PER_HOUR}
    for m, t in totals.items():
        t["dollars"] = t["hours"] * plan.RATE_PER_HOUR[m]
    return {"blocks": table, "totals": totals, "rates_per_hour": plan.RATE_PER_HOUR, "seconds_source": plan.SECONDS_SOURCE,
            "not_included": "metric scoring time (a placeholder until measured), embeddings, canaries, pod setup, replacements and debugging"}
