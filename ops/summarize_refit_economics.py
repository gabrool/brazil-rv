"""Registered matched-data screen results, separate from old-coordinate effects."""

from collections import defaultdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.foundation_readouts import paired_interval

PROJECT = Path(__file__).resolve().parents[1]


def sensitivity_results(
    run, plan, out, primary_metrics, *, comparison_pairs=None, saved_controls=()
):
    """All tested hypotheses, with unexposed folds explicitly retaining baseline."""
    primary = {
        (r["capital"], r["arm"], r["fold"]): r
        for r in primary_metrics
        if r["member"] == "ensemble"
    }
    rows = {}
    sources = []
    for plan_key, proof_key, conditional in (
        (
            "stage_c_refit_sensitivity_plan",
            "stage_c_refit_sensitivity_qualification",
            False,
        ),
        ("stage_c_refit_debit_plan", "stage_c_refit_debit_qualification", True),
        (
            "stage_c_refit_fraction_precision_plan",
            "stage_c_refit_fraction_precision_qualification",
            False,
        ),
    ):
        specification = bound_json(run[plan_key])
        root = Path(run[plan_key]["path"]).parent
        replay = bound_json(binding(root / "replays.json"))
        proof = bound_json(run[proof_key])
        assert replay["status"] == proof["status"] == "complete"
        assert proof["qualified"] == len(replay["completed"])
        records = {r["key"]: r for r in replay["completed"]}
        skipped = {r["key"]: r for r in replay["skipped"]}
        sources.append(
            dict(
                plan=run[plan_key],
                qualification=run[proof_key],
                completed=len(records),
                skipped=len(skipped),
            )
        )
        for phase in specification["phases"]:
            variant = ("denied_" if conditional else "") + phase["name"]
            capital = phase["capital"]
            for arm in plan["arms"]:
                for fold in plan["folds"]:
                    base = primary[capital, arm, fold]
                    if conditional:
                        base = rows["denied_renewal", capital, arm, fold]
                    key = f"{variant}/{capital}/{arm}/{fold}/ensemble"
                    if key in records:
                        rec = records[key]
                        saved = bound_json(rec["book"])
                        quality = bound_json(
                            binding(
                                root
                                / "qualification"
                                / (key.replace("/", "_") + ".json")
                            )
                        )
                        assert quality["passed"] and quality["book"] == rec["book"]
                        row = dict(
                            mean=saved["summary"]["mean"],
                            performance=bound_json(rec["performance"]),
                            economics_unresolved=rec["economics_unresolved"],
                            exposed=True,
                            max_abs_path_bps=quality["contrast"]["max_abs_path_bps"],
                            loan_cash_bounds_pending=quality[
                                "loan_cash_bounds_pending"
                            ],
                            unquoted_holdings=quality["unquoted_holdings"],
                            terminal_unquoted=[
                                row
                                for row in bound_json(quality["unquoted_holdings"])
                                if row["terminal"]
                            ],
                            prior_debit_sessions=quality["prior_debit_sessions"],
                            maximum_overdue_principal=quality[
                                "maximum_overdue_principal"
                            ],
                            overdue_dates=[
                                day
                                for day, value in zip(
                                    saved["state_dates"],
                                    saved["daily"]["loan_overdue_principal"],
                                    strict=True,
                                )
                                if value > 0
                            ],
                            terminal_overdue_principal=saved["daily"][
                                "loan_overdue_principal"
                            ][-1],
                            source=rec,
                        )
                    else:
                        assert key in skipped or (
                            conditional and not base["exposed"]
                        ), key
                        row = dict(
                            mean=base["mean"],
                            performance=base["performance"],
                            economics_unresolved=base["economics_unresolved"],
                            exposed=False,
                            max_abs_path_bps=0,
                            loan_cash_bounds_pending=base["loan_cash_bounds_pending"],
                            unquoted_holdings=base["unquoted_holdings"],
                            terminal_unquoted=base["terminal_unquoted"],
                            prior_debit_sessions=base["prior_debit_sessions"],
                            maximum_overdue_principal=base["maximum_overdue_principal"],
                            overdue_dates=base["overdue_dates"],
                            terminal_overdue_principal=base[
                                "terminal_overdue_principal"
                            ],
                            source=skipped.get(key),
                            skip_reason="Unexposed hypothesis retains its baseline; not a numerical bound on an exposed book",
                        )
                    row.update(
                        variant=variant,
                        capital=capital,
                        arm=arm,
                        fold=fold,
                        baseline="denied_renewal"
                        if conditional
                        else "approved_renewal_primary",
                        conditional_delta_bps_day=row["mean"]["net_excess_bps"]
                        - base["mean"]["net_excess_bps"],
                        total_vs_primary_bps_day=row["mean"]["net_excess_bps"]
                        - primary[capital, arm, fold]["mean"]["net_excess_bps"],
                    )
                    rows[variant, capital, arm, fold] = row
    write_json_atomic(out / "sensitivity_books.json", list(rows.values()))
    groups = defaultdict(list)
    for row in rows.values():
        groups[row["variant"], row["capital"], row["arm"]].append(row)
    summaries = []
    for (variant, capital, arm), group in sorted(groups.items()):
        assert len(group) == len(plan["folds"])
        summaries.append(
            dict(
                variant=variant,
                capital=capital,
                arm=arm,
                baseline=group[0]["baseline"],
                mean_net_cdi_bps_day=float(
                    np.mean([r["mean"]["net_excess_bps"] for r in group])
                ),
                mean_conditional_delta_bps_day=float(
                    np.mean([r["conditional_delta_bps_day"] for r in group])
                ),
                mean_total_vs_primary_bps_day=float(
                    np.mean([r["total_vs_primary_bps_day"] for r in group])
                ),
                exposed_folds=sum(r["exposed"] for r in group),
                unresolved_books=sum(r["economics_unresolved"] for r in group),
                loan_cash_bounds_pending=sum(
                    r["loan_cash_bounds_pending"] for r in group
                ),
                overdue_sessions=sum(len(r["overdue_dates"]) for r in group),
                terminal_overdue_books=sum(
                    r["terminal_overdue_principal"] > 0 for r in group
                ),
                terminal_unquoted_by_fold={
                    r["fold"]: r["terminal_unquoted"] for r in group
                },
                maximum_conditional_path_bps=max(r["max_abs_path_bps"] for r in group),
                mean_fold_brl_cdi_sharpe=float(
                    np.mean([r["performance"]["sharpe_brl_minus_cdi"] for r in group])
                ),
                worst_fold_drawdown_brl=min(
                    r["performance"]["maximum_drawdown_brl"] for r in group
                ),
            )
        )
    write_json_atomic(out / "sensitivity_summary.json", summaries)
    indexed = {(r["variant"], r["capital"], r["arm"]): r for r in summaries}
    for row in saved_controls:
        key = row["variant"], row["capital"], row["arm"]
        assert key not in indexed
        indexed[key] = row
    comparisons = []
    if comparison_pairs is None:
        comparison_pairs = (
            ("TE_wide", "TE_full"),
            ("GRU_early", "TE_full"),
            ("TE_full", "C6"),
        )
    for variant, capital in sorted({(r["variant"], r["capital"]) for r in summaries}):
        for candidate, control in comparison_pairs:
            a, b = (indexed[variant, capital, arm] for arm in (candidate, control))
            comparisons.append(
                dict(
                    variant=variant,
                    capital=capital,
                    candidate=candidate,
                    reference=control,
                    net_advantage_bps_day=a["mean_net_cdi_bps_day"]
                    - b["mean_net_cdi_bps_day"],
                    mean_fold_sharpe_delta=a["mean_fold_brl_cdi_sharpe"]
                    - b["mean_fold_brl_cdi_sharpe"],
                    worst_fold_drawdown_delta=a["worst_fold_drawdown_brl"]
                    - b["worst_fold_drawdown_brl"],
                    unresolved_books=a["unresolved_books"] + b["unresolved_books"],
                    conditional_loan_cash_pending=a["loan_cash_bounds_pending"]
                    + b["loan_cash_bounds_pending"],
                )
            )
    write_json_atomic(out / "sensitivity_comparisons.json", comparisons)
    return dict(
        sources=sources,
        books=binding(out / "sensitivity_books.json"),
        summary=binding(out / "sensitivity_summary.json"),
        comparisons=binding(out / "sensitivity_comparisons.json"),
        limits="Equal four-fold means include explicitly unexposed folds at their unchanged baseline. Conditional debit spreads compare with denied renewal; their total versus approved renewal is separately labelled. These are tested adaptive point contrasts, not joint/interior extrema or an extra multiple-comparison adoption gate. Unresolved status includes any overdue loan during the window, even if later returned; dated exposure and terminal overdue principal are separate. No buy-in/penalty or executable resolution is invented. All prior adaptive numerical uncertainty remains.",
    )


def main():
    tick = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    plan = bound_json(run["stage_c_data_replay_plan"])
    root = Path(run["stage_c_data_replay_plan"]["path"]).parent
    replays = bound_json(binding(root / "replays.json"))
    quality = bound_json(run["stage_c_data_replay_qualification"])
    assert replays["status"] == quality["status"] == "complete"
    assert (
        len(replays["completed"]) == quality["qualified"] == plan["planned_books"] == 96
    )
    old = {r["key"]: r for r in bound_json(run["stage_c_event_replays"])["completed"]}
    out = root / "results"
    out.mkdir(exist_ok=False)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    books, metrics = {}, []
    for rec in replays["completed"]:
        key = rec["key"]
        phase, capital, arm, fold, member = key.split("/")
        current = bound_json(rec["book"])
        previous = bound_json(old["sources/" + key.split("/", 1)[1]]["book"])
        assert current["dates"] == previous["dates"]
        books[key] = current
        perf = bound_json(rec["performance"])
        forecast = bound_json(rec["forecast"])
        saved_quality = bound_json(
            binding(root / "qualification" / (key.replace("/", "_") + ".json"))
        )
        assert saved_quality["passed"] and saved_quality["book"] == rec["book"]
        spells = bound_json(rec["holding_spells"])["spells"]
        closed = [
            s["observed_close_sessions"] for s in spells if not s["censored_at_end"]
        ]
        metrics.append(
            dict(
                key=key,
                capital=int(capital),
                arm=arm,
                fold=fold,
                member=member,
                mean=current["summary"]["mean"],
                performance=perf,
                forecast_neutral_ic=forecast["neutral_ic_mean"],
                data_and_refit_delta_bps_day=float(
                    np.mean(
                        np.asarray(current["daily"]["net_excess_bps"])
                        - previous["daily"]["net_excess_bps"]
                    )
                ),
                old_corrected_source_mean_bps_day=float(
                    np.mean(previous["daily"]["net_excess_bps"])
                ),
                economics_unresolved=rec["economics_unresolved"],
                loan_cash_bounds_pending=saved_quality["loan_cash_bounds_pending"],
                unquoted_holdings=saved_quality["unquoted_holdings"],
                terminal_unquoted=[
                    row
                    for row in bound_json(saved_quality["unquoted_holdings"])
                    if row["terminal"]
                ],
                prior_debit_sessions=saved_quality["prior_debit_sessions"],
                maximum_overdue_principal=saved_quality["maximum_overdue_principal"],
                overdue_dates=[
                    day
                    for day, value in zip(
                        current["state_dates"],
                        current["daily"]["loan_overdue_principal"],
                        strict=True,
                    )
                    if value > 0
                ],
                terminal_overdue_principal=current["daily"]["loan_overdue_principal"][
                    -1
                ],
                holding_spells=dict(
                    completed=len(closed),
                    right_censored=sum(s["censored_at_end"] for s in spells),
                    completed_mean_sessions=float(np.mean(closed)) if closed else None,
                    completed_median_sessions=float(np.median(closed))
                    if closed
                    else None,
                ),
                source=rec,
            )
        )
    write_json_atomic(out / "books.json", metrics)
    groups = defaultdict(list)
    for row in metrics:
        if row["member"] == "ensemble":
            groups[row["capital"], row["arm"]].append(row)
    summaries = []
    for (capital, arm), rows in sorted(groups.items()):
        assert len(rows) == 4
        summaries.append(
            dict(
                capital=capital,
                arm=arm,
                net_cdi_bps_day=float(
                    np.mean([r["mean"]["net_excess_bps"] for r in rows])
                ),
                data_and_refit_delta_bps_day=float(
                    np.mean([r["data_and_refit_delta_bps_day"] for r in rows])
                ),
                mean_fold_sharpes={
                    k: float(np.mean([r["performance"][k] for r in rows]))
                    for k in (
                        "sharpe_brl_minus_cdi",
                        "sharpe_brl_minus_zero",
                        "sharpe_usd_minus_us_cash",
                    )
                },
                worst_fold_drawdown_brl=min(
                    r["performance"]["maximum_drawdown_brl"] for r in rows
                ),
                fold_mean_metrics={
                    k: float(np.mean([r["mean"][k] for r in rows]))
                    for k in rows[0]["mean"]
                },
                unresolved_books=sum(r["economics_unresolved"] for r in rows),
                terminal_unquoted_by_fold={
                    r["fold"]: r["terminal_unquoted"] for r in rows
                },
            )
        )
    comparisons = []
    for capital in plan["capitals"]:
        for candidate, reference in (
            ("TE_wide", "TE_full"),
            ("GRU_early", "TE_full"),
            ("TE_full", "C6"),
        ):
            members = {}
            for member in (
                [*map(str, plan["seeds"]), "ensemble"]
                if capital == 10000000
                else ["ensemble"]
            ):
                arrays = []
                for fold in plan["folds"]:
                    a, b = (
                        books[f"data_refit/{capital}/{arm}/{fold}/{member}"]
                        for arm in (candidate, reference)
                    )
                    assert a["dates"] == b["dates"]
                    arrays.append(
                        np.asarray(a["daily"]["net_excess_bps"])
                        - b["daily"]["net_excess_bps"]
                    )
                members[member] = dict(
                    equal_fold_mean_bps_day=float(np.mean([a.mean() for a in arrays])),
                    fold_deltas={
                        f: float(a.mean())
                        for f, a in zip(plan["folds"], arrays, strict=True)
                    },
                    paired={
                        str(block): paired_interval(arrays, block)
                        for block in (20, 40, 60)
                    },
                )
            a, b = (
                next(
                    s for s in summaries if s["capital"] == capital and s["arm"] == arm
                )
                for arm in (candidate, reference)
            )
            comparisons.append(
                dict(
                    capital=capital,
                    candidate=candidate,
                    reference=reference,
                    members=members,
                    positive_seeds=sum(
                        members[str(seed)]["equal_fold_mean_bps_day"] > 0
                        for seed in plan["seeds"]
                    )
                    if capital == 10000000
                    else None,
                    brl_cdi_sharpe_delta=a["mean_fold_sharpes"]["sharpe_brl_minus_cdi"]
                    - b["mean_fold_sharpes"]["sharpe_brl_minus_cdi"],
                    worst_fold_drawdown_delta=a["worst_fold_drawdown_brl"]
                    - b["worst_fold_drawdown_brl"],
                    nominated_ten_fold_replication_point_condition=capital == 10000000
                    and candidate in {"TE_wide", "GRU_early"}
                    and members["ensemble"]["equal_fold_mean_bps_day"] > 0,
                    technical_exposure_disposition_required=True,
                )
            )
    write_json_atomic(out / "comparisons.json", comparisons)
    sensitivities = sensitivity_results(run, plan, out, metrics)
    report = dict(
        status="corrected_data_refit_screen_results_pending_exposure_sensitivity_disposition",
        primary=run["stage_c_data_replay_plan"],
        qualification=run["stage_c_data_replay_qualification"],
        old_account_source_results=run["stage_c_account_results"],
        books=binding(out / "books.json"),
        comparisons=binding(out / "comparisons.json"),
        model_summary=summaries,
        sensitivities=sensitivities,
        seconds=perf_counter() - tick,
        limits="Corrected data plus necessary matched refits; no incompatible old-weight pure-data counterfactual. Fold means equally weighted; circular paired intervals pool days within separately preserved fold boundaries. Reported Sharpes are means of fold Sharpes, drawdown worst individual fold, not a continuous account. Both named leads were nominated from observed foundation development results. Positive four-fold point means authorize technical review for the other ten development folds, not adoption or retrospective IC-gate passage. Preserve all adaptive numerical uncertainty, cost/loan sensitivities and unresolved actual exposures.",
    )
    write_json_atomic(out / "report.json", report)
    run["stage_c_refit_results"] = binding(out / "report.json")
    write_json_atomic(pointer, run)
    print(json.dumps(dict(summary=summaries, seconds=report["seconds"])), flush=True)


if __name__ == "__main__":
    main()
