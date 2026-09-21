"""Apply the frozen width-retention rule after actual exposure review."""

import json
from pathlib import Path

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]


def main():
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    plan = bound_json(run["stage_d_width_plan"])
    result = bound_json(run["stage_d_width_results"])
    books = bound_json(result["books"])
    proof = bound_json(result["qualification"])
    assert proof["status"] == "complete" and proof["qualified"] == len(books) == 48
    for source in result["sensitivities"]["sources"]:
        assert bound_json(source["qualification"])["status"] == "complete"
    scenarios = bound_json(result["sensitivities"]["books"])
    assert not any(x["loan_cash_bounds_pending"] for x in [*books, *scenarios])
    assert not any(x["maximum_overdue_principal"] for x in books)
    terminal = [
        dict(book=x["source"]["key"], claim=c)
        for x in books
        for c in x["terminal_unquoted"]
    ]
    assert all(x["claim"]["isin"] == "BRENATACNOR0" for x in terminal)
    comparisons = {
        x["candidate"]: x
        for x in bound_json(result["comparisons"])
        if x["capital"] == 10000000
    }
    root = Path(run["stage_d_width_plan"]["path"]).parent
    review_path = root / "risk_disposition.json"
    review = bound_json(binding(review_path)) if review_path.exists() else None
    retained, decisions = {}, []
    for arm, architecture in (("TE_128", "attention"), ("GRU_96", "gru")):
        comparison = comparisons[arm]
        keep = comparison["point_seed_fold_gate"]
        if keep and comparison["risk_attribution_required"]:
            assert (
                review is not None and review["results"] == run["stage_d_width_results"]
            ), (
                "Review the actual Sharpe/drawdown/exposure change before retaining or declining this width"
            )
            keep = review["candidates"][arm]["retain"]
            assert review["candidates"][arm]["reason"]
        reference = plan["parameters"][arm]["reference"]
        retained[architecture] = dict(
            arm=arm if keep else reference,
            plan=run["stage_d_width_plan"] if keep else run["stage_c_refit_plan"],
            results=run["stage_d_width_results"]
            if keep
            else run["stage_c_refit_results"],
            fit_root=str(root) if keep else run["stage_c_refit_root"],
        )
        decisions.append(
            dict(
                candidate=arm,
                reference=reference,
                retained=retained[architecture]["arm"],
                delta_bps_day=comparison["members"]["ensemble"][
                    "equal_fold_mean_bps_day"
                ],
                positive_seeds=comparison["positive_seeds"],
                positive_folds=comparison["positive_folds"],
                point_seed_fold_gate=comparison["point_seed_fold_gate"],
                risk_attribution_required=comparison["risk_attribution_required"],
            )
        )
    admission = dict(
        status="width_disposed_depth_authorized",
        technical_comparisons_valid=True,
        plan=run["stage_d_width_plan"],
        results=run["stage_d_width_results"],
        retained=retained,
        decisions=decisions,
        risk_review=binding(review_path) if review is not None else None,
        primary_terminal_claims=terminal,
        scenario_overdue_books=sum(bool(x["overdue_dates"]) for x in scenarios),
        scenario_terminal_overdue_books=sum(
            x["terminal_overdue_principal"] > 0 for x in scenarios
        ),
        limits="Existing CNPJ remuneration/brokerage, dated fees, source/custody/loan assumptions and marked unpaid ENAT fractions remain. Arithmetic-qualified recall/denial stress is not executable resolution. Preserve all adaptive uncertainty. Technical comparison and width retention do not adopt a model or claim untouched validation, live capacity or guaranteed locates. Cielo/daytrade remain exposure-conditional.",
        next="Exactly one second-layer attention and GRU contrast at retained widths. Reuse current single-layer controls, train new compatible parents. At most one final retained candidate per architecture may enter registered ten-fold confirmation; width and depth do not create extra confirmation slots.",
        recipe=binding(Path(__file__)),
    )
    output = root / "admission.json"
    assert not output.exists()
    write_json_atomic(output, admission)
    run["stage_d_width_admission"] = binding(output)
    write_json_atomic(pointer, run)
    print(json.dumps(decisions), flush=True)


if __name__ == "__main__":
    main()
