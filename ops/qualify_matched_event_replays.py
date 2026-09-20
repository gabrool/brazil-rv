"""Independent saved-book checks of the newly exposed corporate transitions."""

from collections import defaultdict
from decimal import Decimal, ROUND_FLOOR
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]


def dec(value):
    return Decimal(str(float(value)))


def main():
    started = perf_counter()
    run = json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())
    plan = bound_json(run["stage_c_event_replay_plan"])
    root = Path(plan["output_root"])
    out = root / "qualified"
    out.mkdir(exist_ok=False)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    completed = bound_json(binding(root / "replays.json"))
    assert completed["status"] == "complete"
    assert len(completed["completed"]) == plan["planned_new_books"] == 100
    events = bound_json(run["stage_c_event_source_admission"])["events"]
    terms = bound_json(run["stage_c_event_candidate_terms"])
    scalar_q = {
        (e["effective_date"], e["isin"]): e["shares_per_prior_share"]
        for e in terms["scalar_actions"]
    }
    old = {c["key"]: c for c in bound_json(plan["prior_replays"])["completed"]}
    foundation = bound_json(binding(PROJECT / "docs/v2_data_inputs.json"))["store"]
    store = Path(foundation["root"])
    names = np.load(store / "isin_index.npy").tolist()
    dates = np.load(store / "date_index.npy")
    q = np.load(store / "action_shares_per_prior_share.npy", mmap_mode="r")
    cash = bound_json(run["cash_calendar"])["panel"]
    assert sha256_file(Path(cash["path"])) == cash["sha256"]
    with np.load(cash["path"]) as z:
        cdi = z["cdi_returns"]
    maxima, counts = defaultdict(float), defaultdict(int)
    records = []

    def check(label, expected, actual, tolerance=1e-7):
        a, b = np.asarray(expected), np.asarray(actual)
        error = float(np.max(np.abs(a - b), initial=0))
        assert np.isfinite(error) and error < tolerance, (label, error)
        maxima[label] = max(maxima[label], error)
        counts[label] += int(np.broadcast_arrays(a, b)[0].size)

    for rec in completed["completed"]:
        book = bound_json(rec["book"])
        folder = Path(rec["book"]["path"]).parent
        for file, digest in book["files"].items():
            assert sha256_file(folder / file) == digest
        with np.load(folder / "account.npz") as z:
            a = {k: z[k] for k in z.files}
        n = len(a["nav"])
        assert a["signed_shares"].shape == (n, 933)
        counts["saved_account_cells"] += sum(v.size for v in a.values())
        nav = (
            a["free_cash"]
            + a["restricted_cash"]
            + a["hedge_restricted_cash"]
            + a["unsettled_cash"]
            + a["receivables"]
            - a["payables"]
            + (a["signed_shares"] * np.nan_to_num(a["mark_price"])).sum(1)
            + a["hedge_signed_shares"] * np.nan_to_num(a["hedge_mark_price"])
            - a["loan_liability"]
            - a["custody_liability"]
        )
        check("nav_brl", nav, a["nav"])
        cfg, daily = book["provenance"]["config"], book["daily"]
        indices = np.searchsorted(
            dates, np.array(book["state_dates"], dtype="datetime64[D]")
        )
        np.testing.assert_array_equal(dates[indices].astype(str), book["state_dates"])
        np.testing.assert_allclose(
            np.array(daily["cdi_bps"]) / 1e4, cdi[indices], rtol=0, atol=1e-18
        )
        free = np.r_[cfg["initial_capital_brl"], a["free_cash"][:-1]]
        restricted = np.r_[0, (a["restricted_cash"] + a["hedge_restricted_cash"])[:-1]]
        scale = a["start_nav"] / 1e4
        free_income = np.maximum(free, 0) * cdi[indices]
        debit = -np.minimum(free, 0) * (
            cdi[indices] + cfg["annual_debit_spread"] / cfg["annual_sessions"]
        )
        proceeds = restricted * cdi[indices] * cfg["short_proceeds_remuneration"]
        for label, expected in (
            ("free_cash_income_bps", free_income),
            ("debit_financing_bps", debit),
            ("short_proceeds_income_bps", proceeds),
            ("interest_bps", free_income - debit + proceeds),
        ):
            check(label, expected, np.array(daily[label]) * scale)
        check(
            "custody_liability_rollforward",
            np.r_[0, a["custody_liability"][:-1]]
            + a["custody_fee"]
            - a["custody_payment"],
            a["custody_liability"],
        )
        fills = pl.read_parquet(folder / "fills.parquet").to_dicts()
        signed = defaultdict(lambda: Decimal(0))
        spot_sides = defaultdict(set)
        for f in fills:
            spot_sides[f["fill_session"], f["security_index"]].add(f["side"])
            if f["purpose"] != "hedge":
                signed[f["fill_session"], f["security_index"]] += dec(f["quantity"]) * (
                    1 if f["side"] == "buy" else -1
                )
        assert all(len(v) == 1 for v in spot_sides.values())
        payments = bound_json(rec["loan_cash_payments"])
        claims_file = folder / "share_claim_positions.json"
        claims = json.loads(claims_file.read_text()) if claims_file.exists() else []
        arrivals, exposures = defaultdict(lambda: Decimal(0)), []
        first_effect = n
        for event in events:
            effect = int(np.searchsorted(book["state_dates"], event["effective_date"]))
            if effect >= n or book["state_dates"][effect] != event["effective_date"]:
                continue
            first_effect = min(first_effect, effect)
            source, target = (
                names.index(event["isin"]),
                names.index(event["successor_isin"]),
            )
            held = dec(a["signed_shares"][effect - 1, source]) if effect else Decimal(0)
            ratio = Decimal(event["shares_per_prior_share"])
            incoming = held * ratio
            delivery = int(np.searchsorted(book["state_dates"], event["delivery_date"]))
            auction = event["fractional_auction"]
            whole = (
                incoming.to_integral_value(rounding=ROUND_FLOOR)
                if auction and incoming > 0
                else incoming
            )
            residual = incoming - whole
            if delivery < n:
                destination_q = scalar_q.get(
                    (book["state_dates"][delivery], event["successor_isin"]),
                    q[indices[delivery], target],
                )
                assert destination_q == 1, (
                    event["id"],
                    "destination action requires explicit attribution",
                )
                arrivals[delivery, target] += whole
            stop = (
                min(
                    n,
                    int(
                        np.searchsorted(
                            book["state_dates"], event["source_reopens_date"]
                        )
                    ),
                )
                if event["source_reopens_date"]
                else n
            )
            assert not any(
                f["security_index"] == source
                and f["purpose"] != "hedge"
                and effect <= f["fill_session"] < stop
                for f in fills
            )
            for day in range(effect, min(delivery, n)):
                check(
                    "locked_source_units", float(held), a["signed_shares"][day, source]
                )
            recognition = (
                int(np.searchsorted(book["state_dates"], auction["available_date"]))
                if auction
                else n
            )
            for day in range(delivery, min(stop, recognition)):
                check(
                    "retained_source_fraction_units",
                    float(residual / ratio),
                    a["signed_shares"][day, source],
                )
            for c in claims:
                if (
                    c["source_index"] == source
                    and c["successor_index"] == target
                    and effect <= c["session"] < min(stop, recognition)
                ):
                    check(
                        "decimal_claim_units",
                        float(incoming if c["session"] < delivery else residual),
                        c["signed_quantity"],
                    )
            exposures.append(
                dict(
                    event=event["id"],
                    source_axis=source,
                    target_axis=target,
                    held_units=str(held),
                    incoming=str(incoming),
                    delivered=str(whole),
                    residual=str(residual),
                    effect=effect,
                    delivery=delivery,
                )
            )
        for (day, target), incoming in arrivals.items():
            before = dec(a["signed_shares"][day - 1, target]) if day else Decimal(0)
            actual = dec(a["signed_shares"][day, target]) - before - signed[day, target]
            check("decimal_arrivals_and_fills", float(incoming), float(actual))
        # The new unit/no-cash rename must also rekey an undelivered ENAT claim.
        renamed = []
        if "2024-09-09" in book["state_dates"]:
            day = book["state_dates"].index("2024-09-09")
            enat, rrrp, brav = map(
                names.index, ("BRENATACNOR0", "BRRRRPACNOR5", "BRBRAVACNOR3")
            )
            before = [
                c
                for c in claims
                if c["session"] == day - 1
                and c["source_index"] == enat
                and c["successor_index"] == rrrp
            ]
            after = [
                c for c in claims if c["session"] == day and c["source_index"] == enat
            ]
            if before:
                assert (
                    len(before) == len(after) == 1
                    and after[0]["successor_index"] == brav
                )
                check(
                    "renamed_pending_claim_quantity",
                    before[0]["signed_quantity"],
                    after[0]["signed_quantity"],
                )
                assert before[0]["delivery_session"] == after[0]["delivery_session"]
                renamed = [before[0], after[0]]
        parent = bound_json(old[rec["key"]]["book"])
        with np.load(
            Path(old[rec["key"]]["book"]["path"]).parent / "account.npz"
        ) as prior:
            np.testing.assert_array_equal(
                a["nav"][:first_effect], prior["nav"][:first_effect]
            )
            np.testing.assert_array_equal(
                a["targets"][: first_effect + 1], prior["targets"][: first_effect + 1]
            )
            contrast = (a["nav"] - prior["nav"]) / cfg["initial_capital_brl"] * 1e4
        assert (
            book["provenance"]["forecast_sources"]
            == parent["provenance"]["forecast_sources"]
        )
        records.append(
            dict(
                key=rec["key"],
                book=rec["book"],
                exposures=exposures,
                renamed_claim=renamed,
                prefix_sessions=first_effect,
                debit_sessions=int((free < 0).sum()),
                loan_cash_payments=len(payments),
                spot_groups=len(spot_sides),
                final_event_path_bps=float(contrast[-1]),
                max_event_path_bps=float(np.max(np.abs(contrast))),
            )
        )
    assert len(plan["reused"]) == 60
    for rec in plan["reused"]:
        bound_json(rec["book"])
    write_json_atomic(
        out / "report.json",
        dict(
            passed=True,
            source_plan=run["stage_c_event_replay_plan"],
            completed=binding(root / "replays.json"),
            corrected_books=len(records),
            reused_books=60,
            maxima=dict(maxima),
            counts=dict(counts),
            records=records,
            seconds=perf_counter() - started,
            limits="Saved cash/entitlement checks only. Existing loan/cost mechanics reused; no independent reconstruction of every loan cohort, full model-data admission or model-profitability gate. All contrasts are total adaptive book paths, not isolated fees. Initial90 books never executed the later pending-claim rename branch; eight resumed plus two newly identified claim-only books use the qualified branch.",
        ),
    )
    print(
        json.dumps(
            dict(
                passed=True,
                books=len(records),
                maxima=dict(maxima),
                counts=dict(counts),
                seconds=perf_counter() - started,
            )
        )
    )


if __name__ == "__main__":
    main()
