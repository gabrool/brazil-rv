"""Matched capacity comparisons with reused controls and inherited uncertainty."""

import argparse

import json
from pathlib import Path

import numpy as np

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.foundation_readouts import paired_interval
from summarize_refit_economics import fit_readouts, sensitivity_results

PROJECT = Path(__file__).resolve().parents[1]


def main(wave):
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    prefix = f"stage_d_{wave}"
    design = bound_json(run[prefix + "_plan"])
    evaluation = bound_json(run[prefix + "_evaluation_plan"])
    local = json.loads(
        (
            Path(evaluation["context"]) / "docs/v2_economic_data_scaling_run.json"
        ).read_text()
    )
    plan = bound_json(local["stage_c_data_replay_plan"])
    root = Path(local["stage_c_data_replay_plan"]["path"]).parent
    progress = bound_json(binding(root / "replays.json"))
    proof = bound_json(local["stage_c_data_replay_qualification"])
    fits = bound_json(
        binding(Path(run[prefix + "_plan"]["path"]).parent / "refits.json")
    )
    assert fits["status"] == progress["status"] == proof["status"] == "complete"
    assert len(progress["completed"]) == proof["qualified"] == plan["planned_books"]
    references = {v["reference"] for v in design["parameters"].values()}
    reports = design.get("reference_results", {"screen": run["stage_c_refit_results"]})
    report_bindings = {v["path"]: v for v in reports.values()}
    screens = [bound_json(ref) for ref in report_bindings.values()]
    control_rows = [
        row
        for screen in screens
        for row in bound_json(screen["books"])
        if row["arm"] in references
    ]
    control_scenarios = [
        row
        for screen in screens
        for row in bound_json(screen["sensitivities"]["summary"])
        if row["arm"] in references
    ]
    out = root / "results"
    out.mkdir(exist_ok=False)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    metrics = []
    for rec in progress["completed"]:
        book = bound_json(rec["book"])
        _, capital, arm, fold, member = rec["key"].split("/")
        q = bound_json(
            binding(root / "qualification" / (rec["key"].replace("/", "_") + ".json"))
        )
        assert q["passed"] and q["book"] == rec["book"]
        spells = bound_json(rec["holding_spells"])["spells"]
        closed = [
            x["observed_close_sessions"] for x in spells if not x["censored_at_end"]
        ]
        metrics.append(
            dict(
                capital=int(capital),
                arm=arm,
                fold=fold,
                member=member,
                mean=book["summary"]["mean"],
                performance=bound_json(rec["performance"]),
                forecast=bound_json(rec["forecast"]),
                source=rec,
                economics_unresolved=rec["economics_unresolved"],
                loan_cash_bounds_pending=q["loan_cash_bounds_pending"],
                unquoted_holdings=q["unquoted_holdings"],
                terminal_unquoted=[
                    x for x in bound_json(q["unquoted_holdings"]) if x["terminal"]
                ],
                prior_debit_sessions=q["prior_debit_sessions"],
                maximum_overdue_principal=q["maximum_overdue_principal"],
                overdue_dates=[
                    d
                    for d, v in zip(
                        book["state_dates"],
                        book["daily"]["loan_overdue_principal"],
                        strict=True,
                    )
                    if v > 0
                ],
                terminal_overdue_principal=book["daily"]["loan_overdue_principal"][-1],
                holding_spells=dict(
                    completed=len(closed),
                    right_censored=sum(x["censored_at_end"] for x in spells),
                    completed_mean_sessions=float(np.mean(closed)) if closed else None,
                ),
            )
        )
    write_json_atomic(out / "books.json", metrics)
    indexed = {
        (r["capital"], r["arm"], r["fold"], r["member"]): r
        for r in [*control_rows, *metrics]
    }
    comparisons = []
    pairs = [(arm, desc["reference"]) for arm, desc in design["parameters"].items()]
    for candidate, reference in pairs:
        for capital in plan["capitals"]:
            members = {}
            for member in (
                [*map(str, plan["seeds"]), "ensemble"]
                if capital == 10000000
                else ["ensemble"]
            ):
                arrays = []
                for fold in plan["folds"]:
                    a, b = (
                        bound_json(
                            indexed[capital, arm, fold, member]["source"]["book"]
                        )
                        for arm in (candidate, reference)
                    )
                    assert a["dates"] == b["dates"]
                    for key in ("mapping", "policy_inputs"):
                        assert a["provenance"][key] == b["provenance"][key]
                    arrays.append(
                        np.asarray(a["daily"]["net_excess_bps"])
                        - b["daily"]["net_excess_bps"]
                    )
                members[member] = dict(
                    equal_fold_mean_bps_day=float(np.mean([x.mean() for x in arrays])),
                    fold_deltas={
                        f: float(x.mean())
                        for f, x in zip(plan["folds"], arrays, strict=True)
                    },
                    paired={
                        str(block): paired_interval(arrays, block)
                        for block in (20, 40, 60)
                    },
                )
            a, b = (
                [indexed[capital, arm, fold, "ensemble"] for fold in plan["folds"]]
                for arm in (candidate, reference)
            )
            sharpes = {
                k: float(
                    np.mean([x["performance"][k] for x in a])
                    - np.mean([x["performance"][k] for x in b])
                )
                for k in (
                    "sharpe_brl_minus_cdi",
                    "sharpe_brl_minus_zero",
                    "sharpe_usd_minus_us_cash",
                )
            }
            drawdown = min(x["performance"]["maximum_drawdown_brl"] for x in a) - min(
                x["performance"]["maximum_drawdown_brl"] for x in b
            )
            seeds = (
                sum(
                    members[str(s)]["equal_fold_mean_bps_day"] > 0
                    for s in plan["seeds"]
                )
                if capital == 10000000
                else None
            )
            folds = sum(v > 0 for v in members["ensemble"]["fold_deltas"].values())
            comparisons.append(
                dict(
                    candidate=candidate,
                    reference=reference,
                    capital=capital,
                    members=members,
                    positive_seeds=seeds,
                    positive_folds=folds,
                    sharpe_deltas=sharpes,
                    worst_fold_drawdown_delta=drawdown,
                    point_seed_fold_gate=capital == 10000000
                    and members["ensemble"]["equal_fold_mean_bps_day"] >= 0.25
                    and seeds >= 2
                    and folds >= 3,
                    risk_attribution_required=sharpes["sharpe_brl_minus_cdi"] < 0
                    or drawdown < 0,
                    actual_exposure_disposition_required=True,
                )
            )
    write_json_atomic(out / "comparisons.json", comparisons)
    sensitivities = sensitivity_results(
        local,
        plan,
        out,
        metrics,
        comparison_pairs=pairs,
        saved_controls=control_scenarios,
    )
    write_json_atomic(out / "fit_diagnostics.json", fit_readouts(fits))
    report = dict(
        status=f"{wave}_results_pending_registered_disposition",
        plan=run[prefix + "_plan"],
        reused_controls=list(report_bindings.values()),
        qualification=local["stage_c_data_replay_qualification"],
        books=binding(out / "books.json"),
        comparisons=binding(out / "comparisons.json"),
        fit_diagnostics=binding(out / "fit_diagnostics.json"),
        sensitivities=sensitivities,
        limits="Registered capacity contrast with fixed other configuration and unchanged final-context/trunk width; exact dimensional/layer changes are in the frozen plan. Equal-fold means and paired within-fold20/40/60 intervals,40primary; reused development, no untouched test or retrospective IC veto. Sharpes are fold means and drawdown worst individual fold. Preserve all actual unresolved claims, financing scenarios and prior adaptive numerical uncertainty. Screen flags alone do not adopt a model or resolve risk/execution assumptions.",
    )
    write_json_atomic(out / "report.json", report)
    run[prefix + "_results"] = binding(out / "report.json")
    write_json_atomic(pointer, run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wave", choices=("width", "depth", "lstm"), required=True)
    main(parser.parse_args().wave)
