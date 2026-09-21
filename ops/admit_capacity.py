"""Apply the registered capacity screen after actual exposure and risk review."""

import argparse
import json
from pathlib import Path

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]


def main(wave):
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    prefix = f"stage_d_{wave}"
    plan = bound_json(run[prefix + "_plan"])
    result = bound_json(run[prefix + "_results"])
    books = bound_json(result["books"])
    proof = bound_json(result["qualification"])
    assert proof["status"] == "complete"
    assert proof["qualified"] == len(books) == 24 * len(plan["cells"])
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
    root = Path(run[prefix + "_plan"]["path"]).parent
    review_path = root / "risk_disposition.json"
    review = bound_json(binding(review_path)) if review_path.exists() else None
    retained = (
        dict(bound_json(run["stage_d_depth_admission"])["retained"])
        if wave == "lstm"
        else {}
    )
    decisions = []
    for arm, parameters in plan["parameters"].items():
        architecture = "attention" if arm.startswith("TE_") else "gru"
        comparison = comparisons[arm]
        keep = comparison["point_seed_fold_gate"]
        if keep and comparison["risk_attribution_required"]:
            assert (
                review is not None and review["results"] == run[prefix + "_results"]
            ), (
                "Review the actual Sharpe/drawdown/exposure change before retaining or declining this candidate"
            )
            keep = review["candidates"][arm]["retain"]
            assert review["candidates"][arm]["reason"]
        reference = parameters["reference"]
        prior = parameters.get("retained") or dict(
            arm=reference,
            plan=run["stage_c_refit_plan"],
            results=run["stage_c_refit_results"],
            fit_root=run["stage_c_refit_root"],
        )
        retained[architecture] = (
            dict(
                arm=arm,
                plan=run[prefix + "_plan"],
                results=run[prefix + "_results"],
                fit_root=str(root),
            )
            if keep
            else prior
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
        status=f"{wave}_disposed",
        technical_comparisons_valid=True,
        plan=run[prefix + "_plan"],
        results=run[prefix + "_results"],
        retained=retained,
        decisions=decisions,
        risk_review=binding(review_path) if review is not None else None,
        primary_terminal_claims=terminal,
        scenario_overdue_books=sum(bool(x["overdue_dates"]) for x in scenarios),
        scenario_terminal_overdue_books=sum(
            x["terminal_overdue_principal"] > 0 for x in scenarios
        ),
        limits="Existing CNPJ remuneration/brokerage, dated fees, source/custody/loan assumptions and marked unpaid ENAT fractions remain. Arithmetic-qualified recall/denial stress is not executable resolution. Preserve all adaptive uncertainty. Technical comparison and candidate retention do not adopt a model or claim untouched validation, live capacity or guaranteed locates. Cielo/daytrade remain exposure-conditional.",
        next={
            "width": "Exactly one second-layer attention and GRU contrast at retained widths, with fresh compatible parents and saved controls.",
            "depth": "Exactly one LSTM contrast matched to the retained GRU width/layer count, with fresh compatible parents and saved GRU controls.",
            "lstm": "Dispose of the conditional peer/context/general-capacity branch from module/fit and economic diagnostics. At most one final candidate per architecture may enter registered ten-fold confirmation; width, depth and LSTM do not create extra confirmation slots.",
        }[wave],
        recipe=binding(Path(__file__)),
    )
    output = root / "admission.json"
    assert not output.exists()
    write_json_atomic(output, admission)
    run[prefix + "_admission"] = binding(output)
    write_json_atomic(pointer, run)
    print(json.dumps(decisions), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wave", choices=("width", "depth", "lstm"), required=True)
    main(parser.parse_args().wave)
