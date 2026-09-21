"""Existing funding/custody hypotheses on the six added 2019 ensembles."""

import argparse
from copy import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import pickle
from time import perf_counter

import numpy as np
import torch

from brazil_rv.execution.custody_fees import CustodyAssessment
from brazil_rv.execution.portfolio_policy import CalibratedPolicy, exact_replay
from brazil_rv.execution.spot_costs import MonthlySpotTariff
from brazil_rv.execution.stateful_ledger import LedgerConfig
from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.foundation_readouts import new_panel
from brazil_rv.v2.objective_readouts import calibration, rank_view
from brazil_rv.v2.opportunity_research import benchmark_for
from brazil_rv.v2.performance import performance
from brazil_rv.v2.portfolio_readouts import save_book
from brazil_rv.v2.portfolio_training import windows
from brazil_rv.v2.research_rounds import _git_identity
import qualify_refit_books

PROJECT = Path(__file__).resolve().parents[1]


def freeze(run, later=False):
    source_key = (
        "scaling_expanded_later_source_plan"
        if later
        else "scaling_expanded_source_plan"
    )
    plan_key = (
        "scaling_expanded_later_bounds_plan"
        if later
        else "scaling_expanded_event_bounds_plan"
    )
    root = Path(run[source_key]["path"]).parent / "bounds"
    root.mkdir(exist_ok=False)
    variants = [
        dict(name="debit50", annual_debit_spread=0.005),
        dict(name="debit100", annual_debit_spread=0.01),
        dict(name="fibr_arrival", isin="BRFIBRACNOR9", delivery_shift=-1),
        dict(name="fibr_owned_disposal", isin="BRFIBRACNOR9", disposal_at_effect=True),
        dict(name="guar_delivery1", isin="BRGUARACNPR1", delivery_shift=1),
        dict(name="guar_delivery2", isin="BRGUARACNPR1", delivery_shift=2),
        dict(name="guar_bonus_arrival", isin="BRGUARACNOR4", bonus_shift=-1),
    ]
    if later:
        variants = [
            dict(name="debit50", annual_debit_spread=0.005),
            dict(name="debit100", annual_debit_spread=0.01),
            dict(
                name="rlog_custody",
                fold="F7",
                isin="BRRLOGACNOR4",
                disposal_at_custody=True,
            ),
            dict(
                name="rlog_precision",
                fold="F7",
                isin="BRRLOGACNOR4",
                auction_price=2572750.87 / 27449,
            ),
            dict(
                name="smiles_delivery", fold="F7", isin="BRSMLSACNOR1", delivery_shift=1
            ),
        ]
    plan = dict(
        primary=run[source_key],
        variants=variants,
        driver=binding(Path(__file__)),
        scope="Six F3 ensembles, both widths and all three capitals; seven one-factor hypotheses, at most42 books. Reuse all twelve primary books. Execute only actual funded or held exposure. No seed/model/data refit or repeated old sensitivity grid.",
        timing="Jan8 closing Fibria delivery versus next-decision primary; Jan4 prearranged owned disposal with unchanged Jan8 closing receipt. GUAR PN effect-day primary versus one/two later sessions is an unobserved delivery assumption; May6 closing split credit versus May7 primary. Existing custody and disposal implementations are reused.",
        limits="No obtained debit quote, client delivery permission, causal-DI cash-mark result, joint worst-case or all-interior adaptive bound. Fibria last-announced cash mark remains primary; its optional causal-DI contrast is unexecuted. Actual Fibria source holdings are positive, so no forced loan-fraction case. Preserve earlier adaptive numerical uncertainty.",
    )
    if later:
        plan.update(
            scope="Twelve F7/F11 ensembles at all three capitals; funding50/100annualbp and only the three F7 event contrasts. At most42 books; source/capital cases without relevant held/funded exposure are skipped. Reuse24 source and24 frozen baseline books, all existing model/data/source proofs.",
            timing="RLOG custody-first versus expressly permitted March8 owned disposal; fraction printed93.72 versus aggregate2572750.87/27449, unchanged knowledge/payment. Smiles June9 versus following-session custody, fixed June23 cash/default exchange election. F7 event hypotheses do not apply to F11.",
            limits="Same frozen pre-hedge runtime as primary and baseline. Explicit unobserved lender/debit/payment precision hypotheses, no tax exemption or optional-election hindsight. Linx BDR/final-cash gap remains. No combined/interior bound or new model result.",
        )
    write_json_atomic(root / "plan.json", plan)
    run[plan_key] = binding(root / "plan.json")
    write_json_atomic(PROJECT / "docs/v2_economic_data_scaling_run.json", run)
    print(json.dumps(dict(frozen=run[plan_key])))


def execute(run, later=False):
    torch.set_num_threads(1)
    code = _git_identity()
    ref = run[
        "scaling_expanded_later_bounds_plan"
        if later
        else "scaling_expanded_event_bounds_plan"
    ]
    plan = bound_json(ref)
    assert plan["driver"]["sha256"] == sha256_file(Path(__file__))
    root = Path(ref["path"]).parent
    (root / "executed.py").write_bytes(Path(__file__).read_bytes())
    primary = bound_json(plan["primary"])
    parent = Path(plan["primary"]["path"]).parent
    progress = bound_json(binding(parent / "replays.json"))
    assert progress["status"] == "complete"
    source = bound_json(primary["inputs"])["cache"]
    assert sha256_file(Path(source["path"])) == source["sha256"]
    with Path(source["path"]).open("rb") as f:
        data = pickle.load(f)
    names = list(data.inputs.security_ids)
    completed, skipped = [], []
    if (root / "replays.json").exists():
        saved = json.loads((root / "replays.json").read_text())
        completed, skipped = saved["completed"], saved["skipped"]
    done = {r["key"] for r in [*completed, *skipped]}
    panels = {}
    bench_root = Path(
        json.loads((PROJECT / "docs/v2_opportunity_run.json").read_text())["root"]
    )

    def save(status="running"):
        write_json_atomic(
            root / "replays.json",
            dict(status=status, plan=ref, completed=completed, skipped=skipped),
        )

    for base in progress["completed"]:
        if not base["key"].endswith("/ensemble"):
            continue
        book = bound_json(base["book"])
        proof = bound_json(
            binding(
                parent / "qualification" / (base["key"].replace("/", "_") + ".json")
            )
        )
        assert proof["passed"] and proof["book"] == base["book"]
        _, capital, arm, fold, _ = base["key"].split("/")
        rows = windows(Path(primary["prior_root"]), data, fold)["evaluation"]
        start, stop = int(rows[0]), int(rows[-1]) + 1
        if (arm, fold) not in panels:
            panels[arm, fold] = new_panel(
                Path(primary["fit_root"]), data, arm, fold, rows, "raw"
            )
        forecasts, valid, sources = panels[arm, fold]
        assert sources == book["provenance"]["forecast_sources"]
        mapping = calibration(
            bound_json(book["provenance"]["mapping"])["arms"]["TE_all"]
        )
        with np.load(Path(base["book"]["path"]).parent / "account.npz") as z:
            original = {k: z[k] for k in ("nav", "targets", "signed_shares")}
        for variant in plan["variants"]:
            key = variant["name"] + "/" + base["key"].split("/", 1)[1]
            if key in done:
                continue
            if variant.get("fold", fold) != fold:
                skipped.append(
                    dict(
                        key=key,
                        baseline=base,
                        reason="Event is outside this fold; no book run",
                    )
                )
                done.add(key)
                save()
                continue
            tick = perf_counter()
            amended = copy(data)
            cfg = book["provenance"]["config"].copy()
            cfg["monthly_spot_tariffs"] = tuple(
                MonthlySpotTariff(**x) for x in cfg["monthly_spot_tariffs"]
            )
            cfg["custody_assessments"] = tuple(
                CustodyAssessment(**x) for x in cfg["custody_assessments"]
            )
            if "annual_debit_spread" in variant:
                exposed = proof["prior_debit_sessions"] > 0
                cfg["annual_debit_spread"] = variant["annual_debit_spread"]
            else:
                axis = names.index(variant["isin"])
                field = (
                    "action_settlements"
                    if "bonus_shift" in variant
                    else "share_distributions"
                )
                events = list(getattr(data.inputs, field))
                selected = [
                    i
                    for i, event in enumerate(events)
                    if (
                        event.security_index
                        if field == "action_settlements"
                        else event.source_index
                    )
                    == axis
                    and start <= event.effective_session < stop
                ]
                assert len(selected) == 1
                position = selected[0]
                event = events[position]
                before = event.effective_session - start - 1
                held = original["signed_shares"][before, axis]
                owned = any(
                    variant.get(k)
                    for k in (
                        "disposal_at_effect",
                        "disposal_at_custody",
                        "auction_price",
                    )
                )
                exposed = held > 0 if owned else held != 0
                if field == "action_settlements":
                    events[position] = replace(
                        event,
                        bonus_delivery_session=event.bonus_delivery_session
                        + variant["bonus_shift"],
                    )
                else:
                    legs = tuple(
                        replace(leg, disposal_session=event.effective_session)
                        if variant.get("disposal_at_effect")
                        else replace(leg, disposal_session=leg.delivery_session)
                        if variant.get("disposal_at_custody")
                        else replace(
                            leg,
                            fractional_auction=replace(
                                leg.fractional_auction,
                                cash_per_share=variant["auction_price"],
                            ),
                        )
                        if "auction_price" in variant
                        else replace(
                            leg,
                            delivery_session=leg.delivery_session
                            + variant["delivery_shift"],
                        )
                        for leg in event.legs
                    )
                    events[position] = replace(event, legs=legs)
                amended.inputs = replace(data.inputs, **{field: tuple(events)})
            if not exposed:
                skipped.append(
                    dict(
                        key=key,
                        baseline=base,
                        reason="No actual baseline exposure; not a numerical bound",
                    )
                )
                done.add(key)
                save()
                continue
            path = root / "books" / key
            assert not path.exists(), "Preserve partial output and resume explicitly"
            config = LedgerConfig(**cfg)
            view = rank_view(amended, rows, forecasts["ensemble"], valid)
            provenance = dict(
                book["provenance"],
                implementation=code,
                phase=variant["name"],
                phase_hypothesis=variant,
                config=asdict(config),
                sensitivity_plan=ref,
                baseline=base["book"],
            )
            result, targets, previous = exact_replay(
                view, CalibratedPolicy(mapping), start, stop, config=config
            )
            save_book(path, view, result, targets, previous, start, start, provenance)
            bench = benchmark_for(
                bench_root, data.inputs.dates[start:stop], data.inputs.dates[start - 1]
            )
            write_json_atomic(
                path / "performance.json",
                performance(
                    result.daily_net_return, data.inputs.cdi_returns[start:stop], bench
                ),
            )
            write_json_atomic(
                path / "loan_cash_payments.json",
                [asdict(p) for p in result.loan_cash_payments],
            )
            difference = (np.asarray(result.nav) - original["nav"]) / int(capital) * 1e4
            completed.append(
                dict(
                    key=key,
                    book=binding(path / "book.json"),
                    baseline=base,
                    performance=binding(path / "performance.json"),
                    loan_cash_payments=binding(path / "loan_cash_payments.json"),
                    economics_unresolved=bool(result.economics_unresolved),
                    final_path_bps=float(difference[-1]),
                    max_abs_path_bps=float(np.max(np.abs(difference))),
                    seconds=perf_counter() - tick,
                )
            )
            done.add(key)
            save()
            print(
                json.dumps(dict(completed=key, seconds=completed[-1]["seconds"])),
                flush=True,
            )
    save("complete")
    context = root / "qualifier_context"
    (context / "docs").mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        context / "docs/v2_economic_data_scaling_run.json",
        dict(run, stage_c_refit_sensitivity_plan=ref),
    )
    qualify_refit_books.PROJECT = context
    qualify_refit_books.main(sensitivities=True)
    run = json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())
    run[
        "scaling_expanded_later_bounds" if later else "scaling_expanded_event_bounds"
    ] = binding(root / "replays.json")
    run[
        "scaling_expanded_later_bounds_qualification"
        if later
        else "scaling_expanded_event_bounds_qualification"
    ] = binding(root / "qualification/manifest.json")
    write_json_atomic(PROJECT / "docs/v2_economic_data_scaling_run.json", run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--later-source", action="store_true")
    args = parser.parse_args()
    run = json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())
    freeze(run, args.later_source) if args.freeze else execute(run, args.later_source)
