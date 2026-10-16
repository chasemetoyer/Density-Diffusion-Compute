"""Session F's gate before the mask canary is recorded: the mask-check rerun must validate first.

A registered hash from an unvalidated state is worse than none, so the session stops here unless:
- the CUDA repeat rows (T26) ran and gave identical masks;
- the only judged row that failed is the WiLoR conversion row, a cross-device comparison against detections recorded
  on the Mac's ARM CPU (reported, not judged, as in Session C);
- every stock clip, the body clip included, went through the pipeline.

    python scripts/pod/rerun_gate.py results/checks/smoke_test/rerun
"""

import json
import sys
from pathlib import Path

CROSS_DEVICE = "Conversion: WiLoR"
CLIPS = {"face", "hands", "body", "landscape"}


def gate(folder: Path) -> list[str]:
    result = json.loads((folder / "mask_check.json").read_text())["result"]
    report = json.loads(Path(result["report"]).read_text()) if Path(result["report"]).exists() else json.loads((folder / "mask_check" / "mask_pipeline.json").read_text())
    problems = []
    cuda = [r for r in result["t26_rows"] if "CUDA" in r["test"]]
    if len(cuda) < 2 or not all(r["passed"] is True for r in cuda):
        problems.append(f"T26 CUDA repeats: {[(r['test'], r['passed']) for r in cuda]}")
    other = [r["test"] for r in result["failed_rows"] if not r["test"].startswith(CROSS_DEVICE)]
    if other:
        problems.append(f"judged rows failed besides the WiLoR cross-device row: {other}")
    missing = CLIPS - set(report["clip_results"])
    if missing:
        problems.append(f"clips not run: {sorted(missing)}")
    return problems


def main() -> int:
    folder = Path(sys.argv[1])
    problems = gate(folder)
    (folder / "mask_canary").mkdir(parents=True, exist_ok=True)
    (folder / "mask_canary" / "gate.json").write_text(json.dumps({"passed": not problems, "problems": problems}, indent=2) + "\n")
    if problems:
        print("[gate] STOP before recording the mask canary: " + "; ".join(problems), flush=True)
        return 1
    print("[gate] the mask-check rerun validates: T26 CUDA repeats identical, only the WiLoR cross-device row differs, every clip ran", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
