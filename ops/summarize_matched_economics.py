"""Separate saved old-coordinate account/source results from pending data refits."""

from collections import defaultdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.foundation_readouts import paired_interval

PROJECT = Path(__file__).resolve().parents[1]


def main():
    tick = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    primary = bound_json(run["stage_c_event_replays"])
    sensitivity = bound_json(run["stage_c_sensitivities"])
    assert sensitivity["passed"]
    root = Path(run["stage_c_root"])
    out = root / "account_results"
    out.mkdir(exist_ok=False)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    records = {r["key"]: r for r in primary["completed"]}
    books = {k: bound_json(r["book"]) for k, r in records.items()}
    originals = {}
    for k, r in records.items():
        originals.setdefault(r["original"]["path"], bound_json(r["original"]))
    rows = []
    metrics = (
        "net_excess_bps",
        "absolute_bps",
        "cdi_bps",
        "utility_bps",
        "turnover",
        "gross",
        "signed_net",
        "beta",
        "trading_cost_bps",
        "loan_rent_bps",
        "b3_loan_fee_bps",
        "custody_bps",
        "free_cash_income_bps",
        "short_proceeds_income_bps",
        "debit_financing_bps",
    )
    for key, book in books.items():
        phase, capital, arm, fold, member = key.split("/")
        rec = records[key]
        original = originals[rec["original"]["path"]]
        assert original["dates"] == book["dates"]
        perf = bound_json(rec["performance"])
        row = dict(
            key=key,
            phase=phase,
            capital=int(capital),
            arm=arm,
            fold=fold,
            member=member,
            sessions=len(book["dates"]),
            mean={k: float(np.mean(book["daily"][k])) for k in metrics},
            performance=perf,
            original_mean_net_excess_bps=float(
                np.mean(original["daily"]["net_excess_bps"])
            ),
            original_mean_absolute_bps=float(
                np.mean(original["daily"]["absolute_bps"])
            ),
            original_mean_cdi_bps=float(np.mean(original["daily"]["cdi_bps"])),
            unresolved=rec["economics_unresolved"],
        )
        rows.append(row)
    write_json_atomic(out / "books.json", rows)
    summary = []
    for arm in ("C6", "TE_full", "TE_wide", "GRU_early"):
        source = [
            r
            for r in rows
            if r["phase"] == "sources"
            and r["capital"] == 10000000
            and r["arm"] == arm
            and r["member"] == "ensemble"
        ]
        account = [
            r
            for r in rows
            if r["phase"] == "accounting"
            and r["arm"] == arm
            and r["member"] == "ensemble"
        ]
        assert len(source) == len(account) == 4
        before = float(np.mean([r["original_mean_net_excess_bps"] for r in source]))
        accounting = float(np.mean([r["mean"]["net_excess_bps"] for r in account]))
        sourced = float(np.mean([r["mean"]["net_excess_bps"] for r in source]))
        summary.append(
            dict(
                arm=arm,
                original_net_cdi_bps_day=before,
                corrected_account_net_cdi_bps_day=accounting,
                corrected_source_net_cdi_bps_day=sourced,
                accounting_delta_bps_day=accounting - before,
                lending_source_delta_bps_day=sourced - accounting,
                mean_by_fold={r["fold"]: r["mean"]["net_excess_bps"] for r in source},
                original_benchmark_difference_bps_day=float(
                    np.mean(
                        [
                            r["mean"]["cdi_bps"] - r["original_mean_cdi_bps"]
                            for r in source
                        ]
                    )
                ),
                mean_fold_sharpes={
                    k: float(np.mean([r["performance"][k] for r in source]))
                    for k in (
                        "sharpe_brl_minus_cdi",
                        "sharpe_brl_minus_zero",
                        "sharpe_usd_minus_us_cash",
                    )
                },
                worst_fold_drawdown_brl=min(
                    r["performance"]["maximum_drawdown_brl"] for r in source
                ),
            )
        )
    comparisons = []
    for phase in ("accounting", "sources"):
        for candidate, reference in (
            ("TE_wide", "TE_full"),
            ("GRU_early", "TE_full"),
            ("TE_full", "C6"),
        ):
            member_rows = {}
            for member in ("11", "29", "47", "ensemble"):
                arrays = []
                for fold in ("F2", "F6", "F10", "F14"):
                    a, b = (
                        books[f"{phase}/10000000/{arm}/{fold}/{member}"]
                        for arm in (candidate, reference)
                    )
                    assert a["dates"] == b["dates"]
                    arrays.append(
                        np.array(a["daily"]["net_excess_bps"])
                        - b["daily"]["net_excess_bps"]
                    )
                member_rows[member] = dict(
                    fold_mean_bps_day=float(np.mean([a.mean() for a in arrays])),
                    fold_deltas=[float(a.mean()) for a in arrays],
                    paired={
                        str(block): paired_interval(arrays, block)
                        for block in (20, 40, 60)
                    },
                )
            comparisons.append(
                dict(
                    phase=phase,
                    candidate=candidate,
                    reference=reference,
                    members=member_rows,
                    positive_seeds=sum(
                        member_rows[s]["fold_mean_bps_day"] > 0
                        for s in ("11", "29", "47")
                    ),
                    interpretation="Old-coordinate forecasts under amended accounts/sources. Pending corrected-data refits prevent a final advancement/adoption decision; reused development and conditional economic uncertainties remain.",
                )
            )
    write_json_atomic(out / "paired_comparisons.json", comparisons)
    scenarios = bound_json(sensitivity["results"])
    grouped = defaultdict(list)
    for row in scenarios:
        grouped[row["capital"], row["arm"], row["variant"]].append(row)
    bounds = [
        dict(
            capital=c,
            arm=a,
            variant=v,
            exposed_folds=len(rs),
            folds=[r["fold"] for r in rs],
            mean_exposed_fold_net_cdi_delta_bps_day=float(
                np.mean([r["mean_net_cdI_delta_bps"] for r in rs])
            ),
            min_final_path_bps=min(r["final_path_bps"] for r in rs),
            max_final_path_bps=max(r["final_path_bps"] for r in rs),
            max_abs_path_bps=max(r["max_abs_path_bps"] for r in rs),
            maximum_overdue_principal_brl=max(
                r["maximum_overdue_principal_brl"] for r in rs
            ),
            unresolved_books=sum(r["unresolved_economics"] for r in rs),
        )
        for (c, a, v), rs in sorted(grouped.items())
    ]
    write_json_atomic(out / "sensitivity_summary.json", bounds)
    report = dict(
        status="qualified_old_coordinate_account_source_results_data_refits_pending",
        primary=run["stage_c_event_replays"],
        sensitivity=run["stage_c_sensitivities"],
        model_summary=summary,
        books=binding(out / "books.json"),
        comparisons=binding(out / "paired_comparisons.json"),
        sensitivity_summary=binding(out / "sensitivity_summary.json"),
        count=dict(
            books=len(rows),
            sensitivity_books=len(scenarios),
            exposed_scenario_groups=len(bounds),
        ),
        seconds=perf_counter() - tick,
        limits="Fold means equally weighted; paired intervals pool days while preserving each fold boundary. Sharpes are explicitly mean per-fold, and drawdown is worst individual fold. Not a continuous account. Changed corporate/data histories require new compatible fits; no architecture winner or completed StageC/D. Holding-duration and detailed aggregate attribution remain for the final combined report. Preserve all measured adaptive numerical uncertainties and unexposed skips.",
    )
    write_json_atomic(out / "report.json", report)
    run = json.loads(pointer.read_text())
    run["stage_c_account_results"] = binding(out / "report.json")
    write_json_atomic(pointer, run)
    print(json.dumps(dict(summary=summary, seconds=report["seconds"])), flush=True)


if __name__ == "__main__":
    main()
