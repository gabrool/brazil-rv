"""Apply the already qualified SOMA auction precision contrast to actual claims."""

import argparse
from copy import copy
from dataclasses import replace
from decimal import Decimal
import json
from pathlib import Path

import numpy as np

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json

import qualify_refit_books
import run_refit_sensitivities

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = "BRSOMAACNOR3"


def freeze(run):
    primary = bound_json(run["stage_c_data_replay_plan"])
    terms = bound_json(primary["terms"])
    event = next(e for e in terms["share_distributions"] if e["isin"] == SOURCE)
    auction = event["legs"][0]["fractional_auction"]
    assert auction["cash_per_share"] == 50.20718
    previous = bound_json(run["opening_claim_audit"])
    prior_plan = bound_json(previous["plan"])
    assert "auction_quotient" in prior_plan["scenarios"]["SOMA"]
    root = Path(run["root"]) / "matched_refit_fraction_precision"
    root.mkdir(exist_ok=False)
    plan = dict(
        primary=run["stage_c_data_replay_plan"],
        original_qualification=run["opening_claim_audit"],
        original_plan=previous["plan"],
        source_event=event,
        phases=[
            dict(
                name="soma_auction_quotient",
                capital=c,
                cash_per_share=float(Decimal("916532.03") / Decimal("18255")),
            )
            for c in primary["capitals"]
        ],
        planned_primary_ensembles=len(primary["arms"])
        * len(primary["folds"])
        * len(primary["capitals"]),
        scope="One previously frozen printed-price versus gross-proceeds quotient contrast, only where actual prior-close SOMA residual claims remain at auction knowledge. All other current 18 distributions, dates, loan conventions, inputs, forecasts and account assumptions stay fixed. No source search or old engineering replay. Skips are unexposed, not universal bounds. Ensemble/capital scope remains the registered sensitivity scope.",
        limits="Neither printed approximate price nor the quotient is an obtained client invoice. Preserve prior adaptive accounting uncertainty; no resolved improvement from sub-bound differences.",
        driver=binding(Path(__file__)),
        producer=binding(PROJECT / "ops/run_refit_sensitivities.py"),
        qualifier=binding(PROJECT / "ops/qualify_refit_books.py"),
    )
    write_json_atomic(root / "plan.json", plan)
    run["stage_c_refit_fraction_precision_plan"] = binding(root / "plan.json")
    write_json_atomic(PROJECT / "docs/v2_economic_data_scaling_run.json", run)


def execute(run):
    ref = run["stage_c_refit_fraction_precision_plan"]
    plan = bound_json(ref)
    for key in ("driver", "producer", "qualifier"):
        bound_json_file = plan[key]
        assert sha256_file(Path(bound_json_file["path"])) == bound_json_file["sha256"]
    root = Path(ref["path"]).parent
    context = root / "context"
    (context / "docs").mkdir(parents=True, exist_ok=True)
    local = dict(run, stage_c_refit_sensitivity_plan=ref)
    write_json_atomic(context / "docs/v2_economic_data_scaling_run.json", local)
    # The unchanged producer reads only this auxiliary pointer outside the run.
    (context / "docs/v2_opportunity_run.json").write_bytes(
        (PROJECT / "docs/v2_opportunity_run.json").read_bytes()
    )
    if not (root / "executed.py").exists():
        (root / "executed.py").write_bytes(Path(__file__).read_bytes())

    def exposed(phase, book, arrays, terms, names):
        day = plan["source_event"]["legs"][0]["fractional_auction"]["available_date"]
        days = book["state_dates"]
        return (
            day in days
            and days.index(day) > 0
            and arrays["signed_shares"][days.index(day) - 1, names.index(SOURCE)] != 0
        )

    def amend(data, phase):
        amended = copy(data)
        events = []
        for event in data.inputs.share_distributions:
            if data.inputs.security_ids[event.source_index] == SOURCE:
                assert len(event.legs) == 1
                leg = event.legs[0]
                assert leg.fractional_auction.cash_per_share == 50.20718
                event = replace(
                    event,
                    legs=(
                        replace(
                            leg,
                            fractional_auction=replace(
                                leg.fractional_auction,
                                cash_per_share=phase["cash_per_share"],
                            ),
                        ),
                    ),
                )
            events.append(event)
        amended.inputs = replace(data.inputs, share_distributions=tuple(events))
        assert amended.static is data.static and amended.references is data.references
        return amended

    run_refit_sensitivities.PROJECT = context
    run_refit_sensitivities.exposed = exposed
    run_refit_sensitivities.phase_data = amend
    # __file__ stays the unchanged producer, whose hash guard uses its own bytes.
    internal = dict(plan, driver=plan["producer"])
    write_json_atomic(root / "producer_plan.json", internal)
    local["stage_c_refit_sensitivity_plan"] = binding(root / "producer_plan.json")
    write_json_atomic(context / "docs/v2_economic_data_scaling_run.json", local)
    run_refit_sensitivities.execute(local)
    qualify_refit_books.PROJECT = context
    qualify_refit_books.main(sensitivities=True)
    replay = bound_json(binding(root / "replays.json"))
    primary = bound_json(plan["primary"])
    names = np.load(Path(primary["store"]["root"]) / "isin_index.npy").tolist()
    source = names.index(SOURCE)
    event = plan["source_event"]
    known = event["legs"][0]["fractional_auction"]["available_date"]
    checks = []
    for rec in replay["completed"]:
        book = bound_json(rec["book"])
        day = book["state_dates"].index(known)
        with (
            np.load(Path(rec["book"]["path"]).parent / "account.npz") as a,
            np.load(Path(rec["baseline"]["book"]["path"]).parent / "account.npz") as b,
        ):
            assert np.array_equal(a["nav"][:day], b["nav"][:day])
            assert np.array_equal(a["targets"][: day + 1], b["targets"][: day + 1])
            quantity = Decimal(
                str(float(b["signed_shares"][day - 1, source]))
            ) * Decimal(str(event["legs"][0]["shares_per_prior_share"]))
            delta = quantity * (
                Decimal(str(plan["phases"][0]["cash_per_share"])) - Decimal("50.20718")
            )
            error = abs(float(a["nav"][day] - b["nav"][day]) - float(delta))
            assert error < 2e-8, (rec["key"], error)
            checks.append(
                dict(
                    key=rec["key"],
                    signed_fraction=str(quantity),
                    decimal_initial_cash_delta=str(delta),
                    initial_nav_error=error,
                    prefix_and_current_intentions_exact=True,
                )
            )
    write_json_atomic(
        root / "precision_arithmetic.json",
        dict(
            passed=True,
            checks=checks,
            scope="Independent initial signed-claim precision arithmetic and causal prefixes; later path differences include adaptive decisions, not just this cash delta.",
        ),
    )
    run["stage_c_refit_fraction_precision_qualification"] = binding(
        root / "qualification/manifest.json"
    )
    run["stage_c_refit_fraction_precision_arithmetic"] = binding(
        root / "precision_arithmetic.json"
    )
    write_json_atomic(PROJECT / "docs/v2_economic_data_scaling_run.json", run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    run = json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())
    freeze(run) if args.freeze else execute(run)
