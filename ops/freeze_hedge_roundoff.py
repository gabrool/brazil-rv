"""Preserve the observed hedge defect and freeze its bounded correction contrast."""

import json
from pathlib import Path
import shutil

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding

PROJECT = Path(__file__).resolve().parents[1]
pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
run = json.loads(pointer.read_text())
out = Path(run["scaling_investigation"]["path"]).parent / "hedge_roundoff"
out.mkdir(exist_ok=False)
# A small source snapshot keeps the unfinished baseline evaluation on exactly
# its prior account implementation while the independent correction is tested.
source = PROJECT / "research/src/brazil_rv"
snapshot = out / "baseline_runtime/brazil_rv"
shutil.copytree(source, snapshot, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
files = {str(p.relative_to(snapshot)): binding(p) for p in snapshot.rglob("*.py")}
write_json_atomic(out / "baseline_runtime.json", files)
source_root = Path(run["scaling_expanded_source_plan"]["path"]).parent
plan = dict(
    status="frozen_before_corrected_outcomes",
    original_runtime=binding(out / "baseline_runtime.json"),
    first_difference=binding(
        source_root / "account_parity/first_difference/report.json"
    ),
    first_nav_difference=binding(
        source_root / "account_parity/diagnostic/first_nav_difference.json"
    ),
    source_plan=run["scaling_expanded_source_plan"],
    source_books=binding(source_root / "replays.json"),
    matched_books=run["stage_c_refit_results"],
    change="Both accounts suppress only hedge target/current differences within eight Float64 epsilons times their maximum absolute magnitude. This relative arithmetic bound covers weight/notional round trips and a few account summation ulps; it is not a currency or quantity floor. Genuine tiny openings from zero, material partial changes, reversals and terminal/mandatory covers remain. Stock order policy, allocator, source terms, models and fits stay fixed.",
    contrasts="First qualify focused no-trade/old-minimum, tiny genuine opening, partial/reversal/terminal and gradient cases. Replay the twelve existing F3 source books with unchanged forecasts/inputs/mapping and only corrected hedge arithmetic; compare saved original books, independently reconcile NAV, and run both accounts on identical new intentions. Inspect existing four-fold corrected ensemble hedge fills before any further replay; only exposed cases need a bounded follow-up, separately frozen. No broad sensitivity rerun, model fit or silent baseline replacement.",
    baseline="The unfinished fixed eight-period evaluation uses the preserved baseline_runtime source snapshot; its existing72 books remain unchanged. Current GPU uses its existing isolated original training runtime. Correction outcomes are a separate attribution.",
    limits="First observed R10 error is real but is not established as the cause of multi-bps model reversal. Account parity and economic conclusions remain conditional until the actual exposed paths qualify. Preserve failures and all prior proofs; do not relax the opposite-direction spot guard.",
)
write_json_atomic(out / "plan.json", plan)
(out / "freeze_executed.py").write_bytes(Path(__file__).read_bytes())
run["scaling_hedge_roundoff_plan"] = binding(out / "plan.json")
write_json_atomic(pointer, run)
print(
    json.dumps(dict(plan=run["scaling_hedge_roundoff_plan"], source_files=len(files)))
)
