"""Bind executed cost books to current code and retain metadata-only corrections."""

from difflib import unified_diff
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import torch

from brazil_rv.execution.custody_fees import CustodyAssessment, CustodyFees
from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]


def main():
    run = json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())
    root = Path(run["root"]) / "tariff_coverage"
    out = root / "runtime_qualification"
    out.mkdir(exist_ok=False)
    shutil.copyfile(__file__, out / "executed.py")
    corporate = root / "corporate_books"
    runtime = []
    for folder in ("execution", "v2"):
        for current in (PROJECT / "research/src/brazil_rv" / folder).glob("*.py"):
            executed = corporate / f"executed_{folder}_{current.name}"
            assert current.read_bytes() == executed.read_bytes(), current
            runtime.append(dict(current=binding(current), executed=binding(executed)))
    differences = []
    for folder in ("execution", "v2"):
        for current in (PROJECT / "research/src/brazil_rv" / folder).glob("*.py"):
            executed = root / "books" / f"executed_{folder}_{current.name}"
            if current.read_bytes() != executed.read_bytes():
                delta = out / f"ordinary_to_current_{folder}_{current.stem}.patch"
                delta.write_text(
                    "".join(
                        unified_diff(
                            executed.read_text().splitlines(True),
                            current.read_text().splitlines(True),
                            fromfile=str(executed),
                            tofile=str(current),
                        )
                    ),
                    encoding="utf8",
                )
                differences.append(
                    dict(
                        current=binding(current),
                        executed=binding(executed),
                        patch=binding(delta),
                    )
                )
    original_report = root / "books/qualification/report.json"
    report = json.loads(original_report.read_text())
    old_limitation = report["limitation"]
    report["limitation"] = (
        "Historical dated spot/custody interactions qualified with assessment-close payment and own-close valuation hypotheses. Earlier V41 payment3/10 bounds are reused, not rerun here. Corporate physical base was outside these18books; separately qualified corporate books now bind that interaction. Final spot invoices/calendar/source composition/integrated StageA admission remain open. Preserve larger prior discrete-fee uncertainty. No model profit."
    )
    write_json_atomic(out / "ordinary_report.json", report)
    comparisons, max_error = 0, 0.0
    calendar = np.load(bound_json(run["enat_settlement_terms"])["calendar"]["path"])
    for case in bound_json(
        bound_json(binding(root / "books/manifest.json"))["completed"]
    ):
        base = root / "books" / case["book"]
        book = json.loads((base / "book.json").read_text())
        cfg = book["provenance"]["config"]
        if not cfg["custody_assessments"]:
            continue
        assessments = tuple(CustodyAssessment(**x) for x in cfg["custody_assessments"])
        assert all(x.date == x.payment_date for x in assessments)
        rows = json.loads((base / "funding_and_costs.json").read_text())
        first = int(np.searchsorted(calendar, np.datetime64(case["start"])))
        with np.load(base / "account.npz") as a:
            for day, row in enumerate(rows):
                quantity = torch.tensor(row["loan_quantity"], dtype=torch.float64)
                loans = SimpleNamespace(
                    quantity=quantity,
                    accrual_end=np.full(934, -1),
                    _by_name=lambda x: x,
                )
                custody = CustodyFees(
                    934,
                    [
                        (x["due"], torch.tensor(x["quantity"], dtype=torch.float64))
                        for x in row["pending"]
                    ],
                )
                shares = np.r_[a["signed_shares"][day], a["hedge_signed_shares"][day]]
                date = str(calendar[first + day])
                result = custody.close(
                    day,
                    date,
                    shares,
                    loans,
                    row["custody_marks"],
                    assessments,
                    economic=cfg["custody_base"] == "economic_long",
                )
                for actual, field in zip(
                    result[:4],
                    (
                        "custody_base",
                        "custody_fee",
                        "custody_payment",
                        "physical_custody",
                    ),
                ):
                    max_error = max(
                        max_error,
                        float(np.max(np.abs(np.asarray(actual) - a[field][day]))),
                    )
                assert float(result[-1]) == 0
                comparisons += 1
    assert max_error < 1e-7
    write_json_atomic(
        out / "report.json",
        dict(
            status="qualified_without_book_replay",
            corporate_runtime=runtime,
            ordinary_runtime_differences=differences,
            ordinary_current_close_saved_state_days=comparisons,
            ordinary_current_close_max_error=max_error,
            no_corporate_rights_in_ordinary_books=True,
            ordinary_original_report=binding(original_report),
            ordinary_qualified_report=binding(out / "ordinary_report.json"),
            original_limitation=old_limitation,
            correction="Only report limitation prose changes; every original numerical field and all18books retained.",
            scope="Current corporate executed production bytes exact. Earlier ordinary-book guard admitted no exposed corporate actions; supplied saved ordinary states qualify current close branch without optimizer, replay, or source reprocessing.",
        ),
    )
    print(
        json.dumps(
            dict(
                days=comparisons,
                max_error=max_error,
                changed_production_files=len(differences),
            )
        )
    )


if __name__ == "__main__":
    main()
