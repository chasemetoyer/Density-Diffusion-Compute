"""Command line for runs. On a pod, source scripts/pod/env.sh first; the run folder lives on the volume.

    python -m density.run init RUN --kind pilot [--dry-run]                       the first manifest from configs/prompts.csv
    python -m density.run init RUN --kind main --registration FILE                the main run at the registered setting
    python -m density.run session RUN --machine h100|slice [--max-jobs N]         every ready job of that machine
    python -m density.run status RUN                                             jobs by kind and state
    python -m density.run cost RUN                                               the cost table of the manifest's jobs
    python -m density.run audit RUN [--setting S]                                 audit/audit.csv for Chase's review
    python -m density.run choose-setting RUN                                     pilot: section 5's detection rule
    python -m density.run review RUN                                             main: apply the reviews and replacements
    python -m density.run freeze RUN                                             the freeze, then the condition and score jobs
    python -m density.run pilot-rules RUN                                        pilot: every rule of section 10, into pilot/rules.json
    python -m density.run encode-canary [--record]                               check (or register) the encode canary on this machine
"""

import argparse
import json
import sys
from pathlib import Path

from density import prompts as prompt_set
from density.run import manifest as M
from density.run.store import Run, check_run_folder, file_sha256, read_json, utc_now, write_json_atomic


def registered_reserve_sizes(reg: dict) -> dict[str, int]:
    """The reserve sizes Chase registers after the pilot (from its pass rates' estimate); the main run cannot start
    without them."""
    sizes = reg["values"].get("reserve_sizes")
    if not isinstance(sizes, dict) or set(sizes) != set(prompt_set.STRATA) or not all(isinstance(v, int) and v >= 0 for v in sizes.values()):
        raise ValueError(f"the registration has no reserve sizes for every stratum ({sizes!r}); Chase sets them after the pilot")
    return sizes


def init(folder: Path, kind: str, dry_run: bool, prompts_csv: Path, registration: Path | None) -> dict:
    """The first manifest. The pilot's init checks the holdout block of prompts.csv only; the main run's init checks
    the whole file, with the registered reserve sizes (Chase, 2026-10-07, decision 14)."""
    check_run_folder(folder, dry_run)
    run = Run(folder)
    if run.path("manifest.json").exists():
        raise RuntimeError(f"{folder} already has a manifest")
    tokenizer = None if dry_run else prompt_set.load_tokenizer()
    extra, reserve_minimums = {}, None
    if kind == "main":
        if registration is None:
            raise ValueError("the main run needs --registration (the registration built after the pilot)")
        reg = read_json(registration)
        if not dry_run:
            reserve_minimums = registered_reserve_sizes(reg)
        extra = {"registration": {"file": str(registration), "sha256": file_sha256(registration)}, "setting": reg["values"]["setting"],
                 "lambda": reg["values"]["lambda"], "reserve_sizes": reserve_minimums}
    report = prompt_set.validate(prompts_csv, tokenizer=tokenizer, complete=not dry_run, scope="holdout" if kind == "pilot" else "all",
                                 reserve_minimums=reserve_minimums)
    if report["errors"]:
        raise RuntimeError(f"{prompts_csv} does not validate: {report['errors'][:5]}")
    _, rows = prompt_set.read_rows(prompts_csv)
    write_json_atomic(run.path("inputs", "prompts.json"), {"source": str(prompts_csv), "sha256": file_sha256(prompts_csv), "rows": rows,
                                                           "validated_scope": report["scope"]})
    jobs = M.pilot_initial(rows) if kind == "pilot" else M.main_initial(rows, extra["setting"])
    manifest = {"run": folder.name, "kind": kind, "dry_run": dry_run, "created": utc_now(), "prompts_csv_sha256": file_sha256(prompts_csv),
                **extra, "jobs": [j.to_dict() for j in jobs]}
    return run.write_manifest(manifest, f"initial {kind} manifest")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("init")
    p.add_argument("run", type=Path)
    p.add_argument("--kind", choices=["pilot", "main"], required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--prompts", type=Path, default=prompt_set.PROMPTS_CSV)
    p.add_argument("--registration", type=Path)
    p = sub.add_parser("session")
    p.add_argument("run", type=Path)
    p.add_argument("--machine", choices=["h100", "slice"], required=True)
    p.add_argument("--device")
    p.add_argument("--weights", type=Path, default=Path(__file__).resolve().parents[3] / "weights" / "Wan2.1-T2V-1.3B-Diffusers")
    p.add_argument("--max-jobs", type=int)
    p.add_argument("--kinds", nargs="*")
    sub.add_parser("encode-canary").add_argument("--record", action="store_true", help="write configs/encode_canary.json from this machine")
    for name in ("status", "cost", "review", "freeze", "choose-setting", "pilot-rules"):
        sub.add_parser(name).add_argument("run", type=Path)
    p = sub.add_parser("audit")
    p.add_argument("run", type=Path)
    p.add_argument("--setting")
    args = parser.parse_args(argv)

    if args.command == "init":
        manifest = init(args.run, args.kind, args.dry_run, args.prompts, args.registration)
        print(f"[run] {args.run}: manifest v{manifest['version']} with {len(manifest['jobs'])} jobs")
        return 0
    if args.command == "encode-canary":
        from density.run.session import encode_canary, record_encode_canary

        print(json.dumps(record_encode_canary() if args.record else encode_canary(), indent=1))
        return 0
    run = Run(args.run)
    check_run_folder(run.folder, run.dry_run)
    rows = read_json(run.path("inputs", "prompts.json"))["rows"]
    if args.command == "session":
        from density.run.session import session

        backend = None
        if run.dry_run:
            from density.run.generate import DryRunBackend

            backend = DryRunBackend(args.device or "cpu", tuple(run.manifest().get("dry_run_noise_prompts", ())))
        record = session(run.folder, args.machine, args.device, backend, args.weights, args.kinds, args.max_jobs)
        print(f"[run] session {record['session']}: {len(record['jobs'])} jobs; stop: {record['stop']}; remaining {record['remaining']}")
        return 0 if record["stop"] is None else 1
    if args.command == "status":
        print(json.dumps(run.status(), indent=1))
        return 0
    if args.command == "cost":
        print(json.dumps(M.planned_cost(run.jobs()), indent=1))
        return 0
    if args.command == "audit":
        from density.run.masks import write_audit

        print(f"[run] wrote {write_audit(run, args.setting)}")
        return 0
    if args.command == "choose-setting":
        from density import pilot

        choice = pilot.choose_setting(run, rows)
        print(json.dumps(choice, indent=1))
        return 0
    if args.command == "pilot-rules":
        from density import pilot

        rules = pilot.apply(run, rows)
        print(json.dumps({k: rules[k] for k in ("stops", "excluded")} | {"lambda": rules["lambda"]["lambda"], "c2_ratio": rules["c2"]["ratio"]}, indent=1))
        return 0 if not rules["stops"] else 1
    if args.command == "review":
        from density.run.masks import review

        print(json.dumps(review(run, rows, run.manifest()["setting"]), indent=1))
        return 0
    if args.command == "freeze":
        from density.run.freeze import freeze

        manifest = run.manifest()
        if manifest["kind"] == "pilot":
            setting = read_json(run.path("pilot", "setting.json"))["setting"]
            out = freeze(run, rows, setting)
        else:
            out = freeze(run, rows, manifest["setting"], manifest["lambda"], manifest["registration"])
        print(json.dumps(out, indent=1))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
