"""Common-date model, stopping and exposure effects with paired uncertainty."""

import argparse
import json
from pathlib import Path

import numpy as np

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.foundation_readouts import paired_interval

PROJECT = Path(__file__).resolve().parents[1]


def read_books(plan_ref, quality_ref, policy):
    plan = bound_json(plan_ref)
    progress = bound_json(binding(Path(plan_ref["path"]).parent / "replays.json"))
    quality = bound_json(quality_ref)
    assert progress["status"] == quality["status"] == "complete"
    assert progress["plan"] == quality["plan"] == plan_ref
    proofs = {bound_json(p)["key"]: bound_json(p) for p in quality["reports"]}
    rows = {}
    for record in progress["completed"]:
        _, capital, arm, fold, member = record["key"].split("/")
        proof = proofs[record["key"]]
        book = bound_json(record["book"])
        assert proof["passed"] and proof["book"] == record["book"]
        spells = bound_json(record["holding_spells"])["spells"]
        closed = [
            s["observed_close_sessions"] for s in spells if not s["censored_at_end"]
        ]
        rows[policy, int(capital), arm, fold, member] = dict(
            dates=book["dates"],
            daily=book["daily"],
            mean=book["summary"]["mean"],
            performance=bound_json(record["performance"]),
            forecast=bound_json(record["forecast"]),
            economics_unresolved=record["economics_unresolved"],
            unquoted_holdings=proof["unquoted_holdings"],
            loan_cash_bounds_pending=proof["loan_cash_bounds_pending"],
            prior_debit_sessions=proof["prior_debit_sessions"],
            maximum_overdue_principal=proof["maximum_overdue_principal"],
            holding_sessions_mean=float(np.mean(closed)) if closed else None,
            holding_sessions_median=float(np.median(closed)) if closed else None,
            completed_holding_spells=len(closed),
            censored_holding_spells=sum(s["censored_at_end"] for s in spells),
            source=record,
        )
    return plan, rows


def main(common=False):
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    plan, books = read_books(
        run["scaling_matched_stopping_evaluation_plan"],
        run["scaling_matched_stopping_qualification"],
        "neutral",
    )
    if common:
        for policy in ("neutral", "flexible"):
            other, rows = read_books(
                run["scaling_common_evaluation_plans"][policy],
                run[f"scaling_common_{policy}_qualification"],
                policy,
            )
            for field in ("store", "inputs", "terms", "folds", "seeds", "capitals"):
                assert other[field] == plan[field]
            assert not books.keys() & rows.keys()
            books.update(rows)
    groups = sorted({(p, c, a, m) for p, c, a, _, m in books})
    summaries = []
    for policy, capital, arm, member in groups:
        rows = [books[policy, capital, arm, f, member] for f in plan["folds"]]
        values = [np.asarray(r["daily"]["net_excess_bps"]) for r in rows]
        summaries.append(
            dict(
                policy=policy,
                capital=capital,
                arm=arm,
                member=member,
                equal_fold_net_cdi_bps_day=float(np.mean([v.mean() for v in values])),
                pooled_day_net_cdi_bps_day=float(np.concatenate(values).mean()),
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
                fold_means={
                    f: r["mean"] for f, r in zip(plan["folds"], rows, strict=True)
                },
                unresolved_folds=[
                    f
                    for f, r in zip(plan["folds"], rows, strict=True)
                    if r["economics_unresolved"]
                ],
            )
        )
    contrasts = {
        "width_at_p5": [(1, "neutral", "TE_wide_p5"), (-1, "neutral", "TE_full_p5")],
        "width_at_p20": [(1, "neutral", "TE_wide_p20"), (-1, "neutral", "TE_full_p20")],
        "stopping_full": [(1, "neutral", "TE_full_p20"), (-1, "neutral", "TE_full_p5")],
        "stopping_wide": [(1, "neutral", "TE_wide_p20"), (-1, "neutral", "TE_wide_p5")],
        "width_stopping_interaction": [
            (1, "neutral", "TE_wide_p20"),
            (-1, "neutral", "TE_wide_p5"),
            (-1, "neutral", "TE_full_p20"),
            (1, "neutral", "TE_full_p5"),
        ],
    }
    if common:
        for arm in sorted({a for _, _, a, _, _ in books}):
            contrasts[f"flexible_minus_neutral/{arm}"] = [
                (1, "flexible", arm),
                (-1, "neutral", arm),
            ]
        for policy in ("neutral", "flexible"):
            for arm in ("C6", "GRU_early"):
                contrasts[f"{arm}_minus_full_p5/{policy}"] = [
                    (1, policy, arm),
                    (-1, policy, "TE_full_p5"),
                ]
    comparisons = []
    for name, terms in contrasts.items():
        for capital in plan["capitals"]:
            for member in (
                [*map(str, plan["seeds"]), "ensemble"]
                if capital == 10000000
                else ["ensemble"]
            ):
                differences = []
                for fold in plan["folds"]:
                    rows = [books[p, capital, a, fold, member] for _, p, a in terms]
                    assert all(r["dates"] == rows[0]["dates"] for r in rows)
                    differences.append(
                        sum(
                            sign * np.asarray(r["daily"]["net_excess_bps"])
                            for (sign, _, _), r in zip(terms, rows, strict=True)
                        )
                    )
                comparisons.append(
                    dict(
                        contrast=name,
                        capital=capital,
                        member=member,
                        equal_fold_delta_bps_day=float(
                            np.mean([v.mean() for v in differences])
                        ),
                        fold_deltas=dict(
                            zip(
                                plan["folds"],
                                [float(v.mean()) for v in differences],
                                strict=True,
                            )
                        ),
                        positive_folds=int(sum(v.mean() > 0 for v in differences)),
                        paired={
                            str(b): paired_interval(differences, b)
                            for b in (20, 40, 60)
                        },
                    )
                )
    root = (
        Path(run["scaling_common_model_plan"]["path"]).parent
        if common
        else Path(plan["fit_root"])
    )
    out = root / "results"
    out.mkdir(exist_ok=True)
    assert not (out / "report.json").exists()
    write_json_atomic(
        out / "books.json", [{"key": list(k), **v} for k, v in books.items()]
    )
    write_json_atomic(out / "comparisons.json", comparisons)
    report = dict(
        status="qualified_account_comparison_pending_exposure_and_model_review",
        summaries=summaries,
        comparisons=binding(out / "comparisons.json"),
        books=binding(out / "books.json"),
        driver=binding(Path(__file__)),
        fit_plan=run["scaling_matched_stopping_plan"],
        common_fit_plan=run.get("scaling_common_model_plan") if common else None,
        limitations="Development comparisons, eight separate period accounts; mean fold Sharpes and worst fold drawdown are not continuous-account statistics. Paired20/40/60-session intervals retain period boundaries,40primary. All seed results retained; no automatic best-model adoption. Model-score qualification, actual-exposure terms and earlier measured adaptive minimum-fee uncertainty remain required before conclusions. Historical6.70751 C6 continuous result is a distinct account/data/exposure contract.",
    )
    write_json_atomic(out / "report.json", report)
    run["scaling_common_results" if common else "scaling_matched_stopping_results"] = (
        binding(out / "report.json")
    )
    write_json_atomic(pointer, run)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("limitations", "summaries")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--common", action="store_true")
    main(parser.parse_args().common)
