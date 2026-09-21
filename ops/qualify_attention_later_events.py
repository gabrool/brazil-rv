"""Independent dated entitlement arithmetic on saved 2021/2023 source books."""

from collections import defaultdict
from decimal import Decimal, ROUND_FLOOR
import json
from pathlib import Path
import pickle
from time import perf_counter

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]


def dec(value):
    return Decimal(str(float(value)))


def main():
    started = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    reference = run["scaling_expanded_later_source_plan"]
    plan = bound_json(reference)
    root = Path(reference["path"]).parent
    out = root / "event_qualification"
    out.mkdir(exist_ok=False)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    progress = bound_json(binding(root / "replays.json"))
    assert progress["status"] == "complete" and len(progress["completed"]) == 24
    baseline = bound_json(
        binding(Path(plan["baseline"]["path"]).parent / "replays.json")
    )
    old = {r["key"]: r for r in baseline["completed"]}
    terms_ref = binding(root / "incremental_terms.json")
    terms = bound_json(terms_ref)
    with Path(bound_json(plan["inputs"])["cache"]["path"]).open("rb") as handle:
        data = pickle.load(handle)
    names = list(data.inputs.security_ids)
    all_dates = np.asarray(data.inputs.dates).astype(str)
    counts, maxima, reports = defaultdict(int), defaultdict(float), []

    def check(label, expected, actual, tolerance=1e-7):
        a, b = np.asarray(expected), np.asarray(actual)
        error = float(np.max(np.abs(a - b), initial=0))
        assert np.isfinite(error) and error < tolerance, (label, error)
        counts[label] += int(np.broadcast_arrays(a, b)[0].size)
        maxima[label] = max(maxima[label], error)

    for rec in progress["completed"]:
        book = bound_json(rec["book"])
        parent = bound_json(old[rec["key"]]["book"])
        folder = Path(rec["book"]["path"]).parent
        with np.load(folder / "account.npz") as z:
            arrays = {
                k: z[k]
                for k in ("signed_shares", "receivables", "payables", "nav", "targets")
            }
        days = book["state_dates"]
        fills = pl.read_parquet(folder / "fills.parquet").to_dicts()
        signed = defaultdict(lambda: Decimal(0))
        for fill in fills:
            if fill["purpose"] != "hedge":
                signed[fill["fill_session"], fill["security_index"]] += dec(
                    fill["quantity"]
                ) * (1 if fill["side"] == "buy" else -1)
        claim_path = folder / "share_claim_positions.json"
        claims = json.loads(claim_path.read_text()) if claim_path.exists() else []
        relevant = [
            e for e in terms["share_distributions"] if e["effective_date"] in days
        ]
        held, residuals, event_dates = {}, {}, []
        for event in relevant:
            source = names.index(event["isin"])
            effect = days.index(event["effective_date"])
            event_dates.append(effect)
            units = dec(arrays["signed_shares"][effect - 1, source])
            held[event["isin"]] = units
            leg = event["legs"][0]
            target = names.index(leg["successor_isin"])
            # Independent printed-ratio oracle, including the reciprocal RLOG
            # contract. No production term/quantity helper is used here.
            ratio = (
                Decimal(1) / Decimal("3.943112")
                if event["isin"] == "BRRLOGACNOR4"
                else Decimal("0.66010000")
            )
            incoming = units * ratio
            whole = (
                incoming.to_integral_value(rounding=ROUND_FLOOR)
                if units > 0
                else incoming
            )
            residual = incoming - whole
            residuals[event["isin"]] = residual
            arrival_date = (
                leg.get("disposal_date")
                if units > 0 and leg.get("disposal_date")
                else leg["delivery_date"]
            )
            arrival = days.index(arrival_date)
            actual = (
                dec(arrays["signed_shares"][arrival, target])
                - dec(arrays["signed_shares"][arrival - 1, target])
                - signed[arrival, target]
            )
            check("decimal_arrivals", float(whole), float(actual))
            assert not [
                f
                for f in fills
                if f["security_index"] == source
                and f["fill_session"] >= effect
                and f["purpose"] != "hedge"
            ]
            auction = leg["fractional_auction"]
            known = (
                days.index(auction["available_date"])
                if auction["available_date"] in days
                else len(days)
            )
            selected = [c for c in claims if c["source_index"] == source]
            for day in range(effect, len(days)):
                expected = (
                    incoming
                    if day < arrival
                    else residual
                    if day < known
                    else Decimal(0)
                )
                actual = sum(
                    (
                        dec(c["signed_quantity"])
                        for c in selected
                        if c["session"] == day
                    ),
                    Decimal(0),
                )
                check("decimal_retained_claims", float(expected), float(actual))
        for event in terms["identity_actions"]:
            if event["effective_date"] not in days:
                continue
            effect = days.index(event["effective_date"])
            event_dates.append(effect)
            source, target = (
                names.index(event[k]) for k in ("predecessor_isin", "successor_isin")
            )
            units = dec(arrays["signed_shares"][effect - 1, source])
            held[event["predecessor_isin"]] = units
            expected = (
                dec(arrays["signed_shares"][effect - 1, target])
                + units
                + signed[effect, target]
            )
            check(
                "identity_successor_units",
                float(expected),
                arrays["signed_shares"][effect, target],
            )
            check(
                "identity_predecessor_extinguished",
                0,
                arrays["signed_shares"][effect:, source],
            )
        # F7 has only these two distribution cash events. Recompose all scalar
        # entitlements from saved prior holdings, then separately add source cash
        # and the dated RLOG fraction. F11 changes identity only; prior corporate
        # cash proofs are reused rather than reimplementing their ledger here.
        if relevant:
            pending = []
            for day, date in enumerate(days):
                index = int(np.searchsorted(all_dates, date))
                prior = arrays["signed_shares"][day - 1] if day else np.zeros(933)
                active = (
                    data.inputs.action_has_action[index]
                    & data.inputs.action_session_resolved[index]
                )
                for axis in np.flatnonzero(
                    active
                    & (data.inputs.action_cash_per_prior_share[index] != 0)
                    & (prior != 0)
                ):
                    pending.append(
                        (
                            int(data.inputs.action_payment_session[index, axis]),
                            dec(prior[axis])
                            * dec(data.inputs.action_cash_per_prior_share[index, axis]),
                        )
                    )
                for event in relevant:
                    if (
                        date == event["effective_date"]
                        and event["cash_per_prior_share"]
                    ):
                        pending.append(
                            (
                                int(np.searchsorted(all_dates, event["payment_date"])),
                                held[event["isin"]] * Decimal("5.11719919"),
                            )
                        )
                    auction = event["legs"][0]["fractional_auction"]
                    if date == auction["available_date"]:
                        pending.append(
                            (
                                int(
                                    np.searchsorted(all_dates, auction["payment_date"])
                                ),
                                residuals[event["isin"]] * Decimal("93.72"),
                            )
                        )
                pending = [(due, amount) for due, amount in pending if due != index]
                check(
                    "decimal_receivables",
                    float(
                        sum((amount for _, amount in pending if amount > 0), Decimal(0))
                    ),
                    arrays["receivables"][day],
                )
                check(
                    "decimal_payables",
                    float(
                        -sum(
                            (amount for _, amount in pending if amount < 0), Decimal(0)
                        )
                    ),
                    arrays["payables"][day],
                )
        first = min(event_dates)
        with np.load(Path(old[rec["key"]]["book"]["path"]).parent / "account.npz") as z:
            for field in ("nav", "targets"):
                np.testing.assert_array_equal(arrays[field][:first], z[field][:first])
        for key in ("forecast_sources", "mapping", "member"):
            assert book["provenance"][key] == parent["provenance"][key]
        reports.append(
            dict(
                key=rec["key"],
                book=rec["book"],
                held={k: str(v) for k, v in held.items()},
                net_cdi_bps_day=rec["net_excess_bps"],
                source_delta_bps_day=rec["net_excess_bps"]
                - old[rec["key"]]["net_excess_bps"],
                economics_unresolved=rec["economics_unresolved"],
            )
        )
    report = dict(
        passed=True,
        plan=reference,
        terms=terms_ref,
        counts=dict(counts),
        maxima=dict(maxima),
        books=reports,
        seconds=perf_counter() - started,
        limits="Independent printed-ratio, signed arrival/identity, retained fraction and F7 cash arithmetic on24 saved books. Existing physical-custody/loan proofs and actual two-account parity remain separate. No model/data/store mutation, optional-election hindsight, Linx BDR valuation or unknown Smiles auction inference.",
    )
    write_json_atomic(out / "report.json", report)
    run = json.loads(pointer.read_text())
    run["scaling_expanded_later_event_qualification"] = binding(out / "report.json")
    write_json_atomic(pointer, run)
    print(json.dumps({k: v for k, v in report.items() if k != "books"}), flush=True)


if __name__ == "__main__":
    main()
