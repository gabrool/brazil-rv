"""Quantify checkpoint-selection noise from saved fit histories; no new fitting.

Reads every ``history.json`` under the given roots (round-7 selection-aware
trainer output), computes the Newey-West standard error of each fit's selection
mean, compares it with the between-epoch spread, and reports what a trailing
smoothed rule would have selected. Evaluation labels are never opened.

    uv run --project research python ops/audit_selection_noise.py \
        --root D:/quant-data/b3/processed/model_runs/<run>/fits --stage F \
        --output docs/v2_selection_noise_audit.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.selection_rules import audit_histories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--stage", choices=("P", "F"))
    parser.add_argument("--smoothing", type=int, default=3)
    parser.add_argument("--lags", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paths = [p for root in args.root for p in root.rglob("history.json")]
    result = audit_histories(
        paths, smoothing=args.smoothing, lags=args.lags, stage=args.stage
    )
    if args.output is not None:
        write_json_atomic(args.output, result)
    pooled = result.get("pooled")
    print(json.dumps(pooled if pooled else result, indent=2), flush=True)


if __name__ == "__main__":
    main()
