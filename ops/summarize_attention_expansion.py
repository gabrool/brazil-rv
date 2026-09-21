"""Eight fixed development periods, with separate source and stopping diagnostics."""

from collections import defaultdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.foundation_readouts import paired_interval
from summarize_refit_economics import fit_readouts

PROJECT = Path(__file__).resolve().parents[1]


def main():
    started = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    ref = run["scaling_expanded_evaluation_plan"]
    plan = bound_json(ref)
    root = Path(ref["path"]).parent
    progress = bound_json(binding(root / "replays.json"))
    quality = bound_json(run["scaling_expanded_qualification"])
    assert progress["status"] == quality["status"] == "complete"
    assert quality["qualified"] == len(progress["completed"]) == 96
    fits = bound_json(run["scaling_expanded_fits"])
    assert fits["status"] == "complete" and len(fits["completed"]) == 54
    out = root / "results"
    assert not (out / "report.json").exists()
    out.mkdir(exist_ok=True)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    proofs = {}
    for reference in quality["reports"]:
        proof = bound_json(reference)
        proofs[proof["key"]] = reference, proof
    books, metrics = {}, []
    for rec in progress["completed"]:
        key = rec["key"]
        _, capital, arm, fold, member = key.split("/")
        book = bound_json(rec["book"])
        proof_ref, proof = proofs[key]
        assert proof["passed"] and proof["book"] == rec["book"]
        books[int(capital), arm, fold, member] = book
        exposure = bound_json(proof["unquoted_holdings"])
        grouped = defaultdict(list)
        for row in exposure:
            grouped[row["isin"]].append(row)
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
                dates=[book["dates"][0], book["dates"][-1]],
                sessions=len(book["dates"]),
                mean=book["summary"]["mean"],
                performance=bound_json(rec["performance"]),
                forecast=bound_json(rec["forecast"]),
                economics_unresolved=rec["economics_unresolved"],
                prior_debit_sessions=proof["prior_debit_sessions"],
                maximum_overdue_principal=proof["maximum_overdue_principal"],
                loan_cash_bounds_pending=proof["loan_cash_bounds_pending"],
                unquoted=[
                    dict(
                        isin=isin,
                        sessions=len(rows),
                        first=rows[0]["date"],
                        last=rows[-1]["date"],
                        terminal=any(r["terminal"] for r in rows),
                        maximum_marked_brl=max(
                            (r["absolute_marked_brl"] or 0) for r in rows
                        ),
                    )
                    for isin, rows in grouped.items()
                ],
                holdings=dict(
                    completed=len(closed),
                    censored=sum(s["censored_at_end"] for s in spells),
                    mean_sessions=float(np.mean(closed)) if closed else None,
                    median_sessions=float(np.median(closed)) if closed else None,
                ),
                source=rec,
                qualification=proof_ref,
            )
        )
    summaries, comparisons = [], []
    for capital in plan["capitals"]:
        for arm in plan["arms"]:
            rows = [
                r
                for r in metrics
                if r["capital"] == capital
                and r["arm"] == arm
                and r["member"] == "ensemble"
            ]
            assert {r["fold"] for r in rows} == set(plan["folds"])
            returns = [
                np.asarray(
                    books[capital, arm, f, "ensemble"]["daily"]["net_excess_bps"]
                )
                for f in plan["folds"]
            ]
            summaries.append(
                dict(
                    capital=capital,
                    arm=arm,
                    equal_fold_net_cdi_bps_day=float(
                        np.mean([a.mean() for a in returns])
                    ),
                    pooled_day_net_cdi_bps_day=float(np.concatenate(returns).mean()),
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
                    fold_means={r["fold"]: r["mean"] for r in rows},
                    unresolved_folds=[
                        r["fold"] for r in rows if r["economics_unresolved"]
                    ],
                )
            )
        for member in (
            [*map(str, plan["seeds"]), "ensemble"]
            if capital == 10000000
            else ["ensemble"]
        ):
            arrays = []
            for fold in plan["folds"]:
                a, b = (
                    books[capital, arm, fold, member] for arm in ("TE_wide", "TE_full")
                )
                assert a["dates"] == b["dates"]
                arrays.append(
                    np.asarray(a["daily"]["net_excess_bps"])
                    - b["daily"]["net_excess_bps"]
                )
            comparisons.append(
                dict(
                    capital=capital,
                    member=member,
                    equal_fold_mean_bps_day=float(np.mean([a.mean() for a in arrays])),
                    fold_deltas=dict(
                        zip(
                            plan["folds"],
                            [float(a.mean()) for a in arrays],
                            strict=True,
                        )
                    ),
                    positive_folds=int(sum(a.mean() > 0 for a in arrays)),
                    paired={
                        str(block): paired_interval(arrays, block)
                        for block in (20, 40, 60)
                    },
                )
            )
    if (out / "books.json").exists():
        assert bound_json(binding(out / "books.json")) == metrics
    else:
        write_json_atomic(out / "books.json", metrics)
    write_json_atomic(out / "comparisons.json", comparisons)
    write_json_atomic(out / "fit_diagnostics.json", fit_readouts(fits))
    source_results = bound_json(run["scaling_expanded_event_qualification"])
    later_results = bound_json(run["scaling_expanded_later_event_qualification"])
    assert source_results["passed"] and later_results["passed"]
    source_bounds = bound_json(run["scaling_expanded_event_bounds_qualification"])
    bounds = [bound_json(r) for r in source_bounds["reports"]]
    later_bounds = bound_json(run["scaling_expanded_later_bounds_qualification"])
    bounds.extend(bound_json(r) for r in later_bounds["reports"])
    source_books = dict(books)
    source_records = [*source_results["books"], *later_results["books"]]
    for rec in source_records:
        _, capital, arm, fold, member = rec["key"].split("/")
        source_books[int(capital), arm, fold, member] = bound_json(rec["book"])
    source_comparison = []
    for capital in plan["capitals"]:
        arrays = []
        means = {arm: {} for arm in plan["arms"]}
        unresolved = {arm: [] for arm in plan["arms"]}
        for fold in plan["folds"]:
            for arm in plan["arms"]:
                book = source_books[capital, arm, fold, "ensemble"]
                means[arm][fold] = float(np.mean(book["daily"]["net_excess_bps"]))
                if book["summary"]["economics_unresolved"]:
                    unresolved[arm].append(fold)
            wide, full = (
                source_books[capital, arm, fold, "ensemble"]
                for arm in ("TE_wide", "TE_full")
            )
            arrays.append(
                np.asarray(wide["daily"]["net_excess_bps"])
                - full["daily"]["net_excess_bps"]
            )
        source_comparison.append(
            dict(
                capital=capital,
                arm_equal_fold_means={
                    arm: float(np.mean(list(values.values())))
                    for arm, values in means.items()
                },
                fold_means=means,
                wide_minus_full_equal_fold_bps_day=float(
                    np.mean([a.mean() for a in arrays])
                ),
                positive_folds=int(sum(a.mean() > 0 for a in arrays)),
                paired={
                    str(block): paired_interval(arrays, block) for block in (20, 40, 60)
                },
                unresolved_folds=unresolved,
            )
        )
    report = dict(
        status="eight_period_development_comparison_with_explicit_source_limits",
        plan=ref,
        qualification=run["scaling_expanded_qualification"],
        fits=run["scaling_expanded_fits"],
        books=binding(out / "books.json"),
        comparisons=binding(out / "comparisons.json"),
        fit_diagnostics=binding(out / "fit_diagnostics.json"),
        model_summary=summaries,
        source_only_f3=source_results,
        source_only_f7_f11=later_results,
        source_overlay_comparison=source_comparison,
        source_bounds=[
            dict(
                key=r["key"],
                contrast=r["contrast"],
                economics_unresolved=r["economics_unresolved"],
            )
            for r in bounds
        ],
        source_bounds_proof=run["scaling_expanded_event_bounds_qualification"],
        later_source_bounds_proof=run["scaling_expanded_later_bounds_qualification"],
        later_source_parity=run["scaling_expanded_later_source_account_parity"],
        stopping_diagnostic=run["scaling_parent_patience_results"],
        hedge_correction=run["scaling_hedge_roundoff_books"],
        hedge_correction_qualification=run["scaling_hedge_roundoff_qualification"],
        hedge_correction_parity=run["scaling_hedge_roundoff_account_parity"],
        timings=dict(
            new_child_fit_sum_seconds=sum(
                r["seconds"]
                for r in fits["completed"]
                if r["key"].split("/")[-2] in {"F3", "F7", "F11", "F13"}
            ),
            new_baseline_book_sum_seconds=sum(
                r["seconds"]
                for r in progress["completed"]
                if r["key"].split("/")[-2] in {"F3", "F7", "F11", "F13"}
            ),
            scope="Sum of the24 new child-fit wall times and48 new baseline book wall times. Thirty fits and48 books reused; not a future runtime estimate.",
        ),
        limits="Eight preselected disjoint development periods, not a continuous portfolio or pristine test. The six other folds are reserved from new corrected comparisons; 2025/2026 consumers remain unopened. Same accepted model data/parents/recipes for both widths. GUAR2019/QGEP2019/GPC2021/WIZ2023/RLOG2021/Smiles2021 source amendments are account-only; their model-data dependencies remain. Linx's three held June2021 dates lack an admitted BDR/final-cash contract, and Smiles fractional auction remains unknown. New unquoted holdings are listed explicitly and not certified by saved-NAV arithmetic. Source overlays, hedge correction and outcome-informed parent-patience diagnostic never replace the frozen baseline silently. The source overlay comparison uses only separate2019/2021/2023 account corrections with unchanged forecasts. Equal-fold means differ from pooled-day paired estimates; mean fold Sharpes and worst individual drawdown are not continuous-account statistics. No model adoption or broad capacity conclusion from this report.",
        seconds=perf_counter() - started,
    )
    write_json_atomic(out / "report.json", report)
    run["scaling_expanded_results"] = binding(out / "report.json")
    write_json_atomic(pointer, run)
    print(
        json.dumps(
            dict(
                report=run["scaling_expanded_results"],
                model_summary=summaries,
                comparisons=comparisons,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
