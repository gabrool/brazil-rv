"""Run the frozen independent GPU fits, then their causal prelude scoring."""

import argparse
from pathlib import Path
import subprocess
import sys

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.portfolio_program import freeze, plan, read


def run(root):
    if not (root / "frozen_design.json").exists():
        freeze(root)
    parents = Path(plan(root, "parents", max_parallel=6)["path"])
    fine = Path(plan(root, "forecasters", max_parallel=6)["path"])
    payload = read(parents)
    payload["jobs"].extend(read(fine)["jobs"])
    payload["dispatch_script_sha256"] = sha256_file(Path(__file__))
    payload["dependency_note"] = (
        "All F jobs inherit existing S0/TE parents; new C6 P supplies only the "
        "prelude. Run 3 P plus 102 F in one six-worker queue without idle GPU lanes."
    )
    combined = root / "forecast_fits_plan.json"
    write_json_atomic(combined, payload)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "brazil_rv.v2.run_many",
            "--plan",
            str(combined),
            "--manifest",
            str(root / "forecast_fits_result.json"),
        ],
        check=True,
    )
    prelude = plan(root, "prelude", max_parallel=6)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "brazil_rv.v2.run_many",
            "--plan",
            prelude["path"],
            "--manifest",
            str(root / "prelude_result.json"),
        ],
        check=True,
    )
    write_json_atomic(
        root / "forecast_program_result.json",
        {
            "status": "completed",
            "new_P": 3,
            "new_F": 102,
            "reused_F": 24,
            "prelude_panels": 9,
            "fits_sha256": sha256_file(root / "forecast_fits_result.json"),
            "prelude_sha256": sha256_file(root / "prelude_result.json"),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    run(parser.parse_args().root)
