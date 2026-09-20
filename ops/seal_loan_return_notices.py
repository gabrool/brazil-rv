"""Bind completed lifecycle books and the narrow post-run ordering qualification."""

import difflib
import json
from pathlib import Path
import shutil

import numpy as np

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]


def main():
    pointer_path = PROJECT / "docs/v2_economic_data_scaling_run.json"
    pointer = json.loads(pointer_path.read_text())
    audit = bound_json(pointer["loan_return_notice_audit"])
    readout = bound_json(pointer["loan_return_notice_qualification"])
    source = bound_json(pointer["corporate_replay"])
    root = Path(pointer["loan_return_notice_audit"]["path"]).parent.parent
    out = root / "runtime_qualification"
    out.mkdir(exist_ok=False)
    shutil.copyfile(__file__, out / "executed.py")
    dates = np.load(Path(source["store"]["root"]) / "date_index.npy").astype(
        "datetime64[D]"
    )
    assert dates[-1] <= np.datetime64("2024-12-30")
    notice_dates = [
        str(
            dates[
                np.searchsorted(dates, np.datetime64(first))
                + audit["plan"]["recall_notice_offset"]
            ]
        )
        for first in audit["plan"]["windows"]
    ]
    deliveries = sorted(
        {
            leg["delivery_date"]
            for event in source["share_distributions"]
            for leg in event["legs"]
        }
    )
    assert not set(deliveries).intersection(notice_dates)
    versions = {}
    for name in (
        "loan_contracts",
        "portfolio_account",
        "portfolio_policy",
        "stateful_ledger",
        "allocation",
    ):
        before = root / "adaptive_bound" / f"executed_{name}.py"
        after = PROJECT / f"research/src/brazil_rv/execution/{name}.py"
        patch = "".join(
            difflib.unified_diff(
                before.read_text().splitlines(True),
                after.read_text().splitlines(True),
                fromfile="audited/" + name,
                tofile="current/" + name,
            )
        )
        path = out / f"{name}.patch"
        path.write_text(patch)
        versions[name] = {
            "audited": binding(before),
            "current": binding(after),
            "patch": binding(path),
        }
    receipt = {
        "status": "runtime_ordering_and_axis_qualification_passed_without_book_rerun",
        "notice_dates": notice_dates,
        "source_delivery_dates": deliveries,
        "source": pointer["corporate_replay"],
        "qualification": "Observe explicit recalls after known morning delivery, once per day, in both accounts. Completed books have no recall/delivery-date intersection and all recall axes were valid; no replay is required for this narrow ordering/index change. Formatting changes are bound separately by exact source patches.",
        "tests": {
            "command": "uv run --project research --group dev --no-sync pytest research/tests/test_loan_return_notices.py -q -k 'morning_delivery or out_of_axis'",
            "result": "3 passed, 7 deselected in 1.84s; exit 0",
            "scope": "Source versus successor notice on known delivery date, both independent accounts; negative/out-of-axis identity rejection.",
        },
        "versions": versions,
    }
    write_json_atomic(out / "manifest.json", receipt)
    pointer["loan_return_notice_runtime_qualification"] = binding(out / "manifest.json")
    acceptance = {
        "status": "lifecycle_implementation_and_engineering_books_verified_stage_A_incomplete",
        "schema": "BRAZIL_RV_V2_EVALUATION_V32",
        "evidence": {
            k: pointer[k]
            for k in (
                "loan_return_notice_audit",
                "loan_return_notice_qualification",
                "loan_return_notice_runtime_qualification",
            )
        },
        "scope": "24 frozen engineering scenarios, each 128 sessions/all 933 names; R$10m primary and R$1m/R$5m. Synthetic preferences use the actual adaptive allocator. No forecast fit, neural forward or model-alpha claim.",
        "contract": "Approved 63-session primary unchanged. Denial at term-minus-four after intentions retains old terms and next-decision whole-name cover; explicit two/four-session recalls arrive pre-decision after known custody delivery. Called borrowing is blocked until required physical settlement. Missing fills retain original obligations and overdue principal; no fabricated buy-in or penalty.",
        "numerical_corrections": "Actual full-cover intent and relative summation bounds eliminate roundoff loan residues; mandatory closures bypass ordinary order deadbands; only optimizer strict active faces are polished. Genuine tiny loans and free-optimum trades remain.",
        "identical_intentions_max_nav_error_brl": readout[
            "same_intentions_max_nav_error_brl"
        ],
        "independent_adaptive_path_max_nav_difference_bps_by_capital": readout[
            "adaptive_max_nav_difference_bps_by_capital"
        ],
        "independent_adaptive_path_max_target_difference": readout[
            "adaptive_max_target_difference"
        ],
        "uncertainty": readout["uncertainty"],
        "execution_cost_scope": audit["plan"]["execution_cost"],
        "primary_corporate_cost_admission": "Still pending dated separated spot/loan/custody/brokerage/shortfall/funding sensitivities on actual model books. The bundled 4bp engineering bridge is not the final corporate cost contract.",
        "timing_seconds": audit["seconds"],
        "remaining": readout["remaining"],
        "unchanged": "No source, accepted store, old fit, security axis or 60-session history changed. Stage B remains separately accepted; C/D unstarted; no final economic admission.",
        "tests": "Ten distinct lifecycle tests passed across targeted batches, eight existing allocation tests, six prior finite-renewal tests and five directly affected loan/gradient/copy tests passed. Overlaps were not added into one inflated suite count; exact failed engineering recipes and qualified outputs remain retained.",
    }
    path = PROJECT / "docs/v2_loan_return_notice_acceptance.json"
    write_json_atomic(path, acceptance)
    pointer["loan_return_notice_acceptance"] = binding(path)
    write_json_atomic(pointer_path, pointer)
    print(
        json.dumps(
            {
                "acceptance": binding(path),
                "notice_dates": notice_dates,
                "delivery_dates": deliveries,
            }
        )
    )


if __name__ == "__main__":
    main()
