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
    report = dict(
        status="corrected_data_refit_screen_results_pending_exposure_sensitivity_disposition",
        primary=run["stage_c_data_replay_plan"],
        qualification=run["stage_c_data_replay_qualification"],
        old_account_source_results=run["stage_c_account_results"],
        books=binding(out / "books.json"),
        comparisons=binding(out / "comparisons.json"),
        model_summary=summaries,
        seconds=perf_counter() - tick,
        limits="Corrected data plus necessary matched refits; no incompatible old-weight pure-data counterfactual. Fold means equally weighted; circular paired intervals pool days within separately preserved fold boundaries. Reported Sharpes are means of fold Sharpes, drawdown worst individual fold, not a continuous account. Both named leads were nominated from observed foundation development results. Positive four-fold point means authorize technical review for the other ten development folds, not adoption or retrospective IC-gate passage. Preserve all adaptive numerical uncertainty, cost/loan sensitivities and unresolved actual exposures.",
    )
    write_json_atomic(out / "report.json", report)
    run["stage_c_refit_results"] = binding(out / "report.json")
    write_json_atomic(pointer, run)
    print(json.dumps(dict(summary=summaries, seconds=report["seconds"])), flush=True)


if __name__ == "__main__":
    main()
