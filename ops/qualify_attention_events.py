"""Independent original-unit and pending-cash arithmetic for added 2019 books."""

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
    plan = bound_json(run["scaling_expanded_source_plan"])
    root = Path(run["scaling_expanded_source_plan"]["path"]).parent
    out = root / "event_qualification"
    out.mkdir(exist_ok=False)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    progress = bound_json(binding(root / "replays.json"))
    assert progress["status"] == "complete" and len(progress["completed"]) == 12
    base = bound_json(binding(Path(plan["baseline"]["path"]).parent / "replays.json"))
    old = {r["key"]: r for r in base["completed"]}
    terms = bound_json(root_receipt := binding(root / "incremental_terms.json"))
    with Path(bound_json(plan["inputs"])["cache"]["path"]).open("rb") as f:
        data = pickle.load(f)
    names = list(data.inputs.security_ids)
    all_dates = np.asarray(data.inputs.dates).astype(str)
    maxima, counts, reports = defaultdict(float), defaultdict(int), []

    def check(label, expected, actual, tolerance=1e-7):
        a, b = np.asarray(expected), np.asarray(actual)
        error = float(np.max(np.abs(a - b), initial=0))
        assert np.isfinite(error) and error < tolerance, (label, error)
        maxima[label] = max(maxima[label], error)
        counts[label] += int(np.broadcast_arrays(a, b)[0].size)

    for rec in progress["completed"]:
        book = bound_json(rec["book"])
        parent = bound_json(old[rec["key"]]["book"])
        folder = Path(rec["book"]["path"]).parent
        with np.load(folder / "account.npz") as z:
            a = {k: z[k] for k in z.files}
        days = book["state_dates"]
        indices = np.searchsorted(all_dates, days)
        fills = pl.read_parquet(folder / "fills.parquet").to_dicts()
        signed = defaultdict(lambda: Decimal(0))
        for fill in fills:
            if fill["purpose"] != "hedge":
                signed[fill["fill_session"], fill["security_index"]] += dec(
                    fill["quantity"]
                ) * (1 if fill["side"] == "buy" else -1)
        pending = []
        held = {}
        arrivals = defaultdict(lambda: Decimal(0))
        residuals = {}
        for e in terms["share_distributions"]:
            effect = days.index(e["effective_date"])
            source = names.index(e["isin"])
            units = dec(a["signed_shares"][effect - 1, source])
            held[e["isin"]] = units
            leg = e["legs"][0]
            target = names.index(leg["successor_isin"])
            incoming = units * Decimal(str(leg["shares_per_prior_share"]))
            whole = (
                incoming.to_integral_value(rounding=ROUND_FLOOR)
                if leg["fractional_auction"] and incoming > 0
                else incoming
            )
            residuals[e["isin"]] = incoming - whole
            due = days.index(leg["delivery_date"])
            arrivals[due, target] += whole
            check(
                "locked_original_units",
                float(units),
                a["signed_shares"][effect:due, source],
            )
            assert not [
                f
                for f in fills
                if f["security_index"] == source
                and f["fill_session"] >= effect
                and f["purpose"] != "hedge"
            ]
        for (day, target), arrival in arrivals.items():
            actual = (
                dec(a["signed_shares"][day, target])
                - dec(a["signed_shares"][day - 1, target])
                - signed[day, target]
            )
            check("decimal_delivered_units", float(arrival), float(actual))
        split = days.index("2019-05-02")
        guar = names.index("BRGUARACNOR4")
        check(
            "guar_eight_for_one",
            float(dec(a["signed_shares"][split - 1, guar]) * 8 + signed[split, guar]),
            a["signed_shares"][split, guar],
        )
        for day, index in enumerate(indices):
            prior_shares = a["signed_shares"][day - 1] if day else np.zeros(933)
            active = (
                data.inputs.action_has_action[index]
                & data.inputs.action_session_resolved[index]
            )
            for name in np.flatnonzero(
                active
                & (data.inputs.action_cash_per_prior_share[index] != 0)
                & (prior_shares != 0)
            ):
                amount = dec(prior_shares[name]) * dec(
                    data.inputs.action_cash_per_prior_share[index, name]
                )
                pending.append(
                    [int(data.inputs.action_payment_session[index, name]), amount, None]
                )
            for e in terms["share_distributions"]:
                units = held[e["isin"]]
                due = (
                    None
                    if e["payment_date"] is None
                    else int(np.searchsorted(all_dates, e["payment_date"]))
                )
                if days[day] == e["effective_date"] and e["cash_per_prior_share"]:
                    pending.append(
                        [
                            due,
                            units * Decimal(str(e["cash_per_prior_share"])),
                            e["isin"],
                        ]
                    )
                for value in e.get("cash_values", ()):
                    if days[day] == value["available_date"]:
                        matching = [p for p in pending if p[2] == e["isin"]]
                        assert len(matching) == 1 if units else not matching
                        for p in matching:
                            p[1] = units * Decimal(str(value["cash_per_prior_share"]))
                auction = e["legs"][0]["fractional_auction"]
                if auction and days[day] == auction["available_date"]:
                    pending.append(
                        [
                            int(np.searchsorted(all_dates, auction["payment_date"])),
                            residuals[e["isin"]]
                            * Decimal(str(auction["cash_per_share"])),
                            None,
                        ]
                    )
            pending = [p for p in pending if p[0] != index]
            positive = sum((p[1] for p in pending if p[1] > 0), Decimal(0))
            negative = -sum((p[1] for p in pending if p[1] < 0), Decimal(0))
            check("decimal_receivables", float(positive), a["receivables"][day])
            check("decimal_payables", float(negative), a["payables"][day])
        with np.load(Path(old[rec["key"]]["book"]["path"]).parent / "account.npz") as z:
            first = days.index("2019-01-04")
            np.testing.assert_array_equal(a["nav"][:first], z["nav"][:first])
            np.testing.assert_array_equal(
                a["targets"][: first + 1], z["targets"][: first + 1]
            )
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
        plan=run["scaling_expanded_source_plan"],
        terms=root_receipt,
        counts=dict(counts),
        maxima=dict(maxima),
        books=reports,
        seconds=perf_counter() - started,
        limits="All twelve saved books; independent dated source-unit/cash/actual-fill arithmetic. Existing loan/custody engine proofs reused. Model-data dependencies are still separate, and prospective account hypotheses remain unobserved client terms.",
    )
    write_json_atomic(out / "report.json", report)
    run = json.loads(pointer.read_text())
    run["scaling_expanded_event_qualification"] = binding(out / "report.json")
    write_json_atomic(pointer, run)
    print(json.dumps({k: v for k, v in report.items() if k != "books"}), flush=True)


if __name__ == "__main__":
    main()
