"""Admit dated source terms to a new accounting artifact, preserving model stores."""

from datetime import datetime
import json
import math
from pathlib import Path

import numpy as np

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic

PROJECT = Path(__file__).resolve().parents[1]


def main():
    pointer_path = PROJECT / "docs/v2_economic_data_scaling_run.json"
    pointer = json.loads(pointer_path.read_text())
    output = Path(pointer["root"]) / "corporate_replay"
    path = output / "copel_manifest.json"
    if path.exists():
        raise FileExistsError(path)
    binding = json.loads((PROJECT / "docs/v2_data_inputs.json").read_text())["store"]
    store = Path(binding["root"])
    if sha256_file(store / "manifest.json") != binding["manifest_sha256"]:
        raise ValueError("corporate replay store differs")
    calendar = np.load(store / "date_index.npy")
    if calendar[-1] > np.datetime64("2024-12-30"):
        raise ValueError("corporate replay is development only")
    source_record = pointer["corporate_loan_sources"]
    source_path = PROJECT / source_record["path"]
    if sha256_file(source_path) != source_record["sha256"]:
        raise ValueError("corporate source manifest differs")
    catalog = json.loads(source_path.read_text())
    sources = {row["id"]: row for row in catalog["sources"]} | catalog[
        "existing_source_receipts"
    ]
    needed = [
        "cielo_loan_b3_20240725",
        "cielo_opa_result_20240814",
        "cielo_opa_edital_20240710",
        "cielo_conversion_20240826",
        "cielo_redemption_20240923",
        "selic_sgs11_20240816_20240926",
    ]
    receipts = [sources[key] for key in needed]
    foundation = json.loads(
        (PROJECT / "docs/v2_foundation_primary_sources.json").read_text()
    )
    fraction = json.loads(
        (
            Path(pointer["root"])
            / "primary_sources/brmalls_fractions_20230125.receipt.json"
        ).read_text()
    )
    receipts += [
        sources["brmalls_loan_b3_20230103"],
        fraction,
        foundation["brmalls_schedule_20221219"],
        foundation["brmalls_delivery_20221219_mirror"],
        foundation["copel_units_20231218"],
    ]
    for record in receipts:
        if sha256_file(Path(record["path"])) != record["sha256"]:
            raise ValueError("corporate source receipt differs")
    raw = json.loads(Path(sources[needed[-1]]["path"]).read_text())
    rates = {
        datetime.strptime(x["data"], "%d/%m/%Y").date().isoformat(): float(x["valor"])
        / 100
        for x in raw
    }
    days = [
        str(d)
        for d in calendar
        if np.datetime64("2024-08-16") <= d < np.datetime64("2024-08-30")
    ]
    if any(d not in rates for d in days):
        raise ValueError("missing SELIC interval for loan settlement")
    price = 5.82 * math.prod(1 + rates[d] for d in days)
    isins = np.load(store / "isin_index.npy").tolist()
    name = isins.index("BRCIELACNOR3")
    close = np.load(store / "raw_close.npy", mmap_mode="r")[:, name]
    quoted = np.isfinite(close) & (close > 0)
    if str(calendar[np.flatnonzero(quoted)[-1]]) != "2024-08-26":
        raise ValueError("Cielo terminal quote differs from the issuer closing date")
    brml = isins.index("BRBRMLACNOR9")
    also = isins.index("BRALSOACNOR5")
    prices = np.load(store / "raw_close.npy", mmap_mode="r")
    quoted = np.isfinite(prices[:, brml]) & (prices[:, brml] > 0)
    if str(calendar[np.flatnonzero(quoted)[-1]]) != "2023-01-06":
        raise ValueError("BR Malls terminal source quote differs")
    preceding = np.searchsorted(calendar, np.datetime64("2023-01-09")) - 1
    if not np.isfinite(prices[preceding, also]):
        raise ValueError("BR Malls successor lacks a causal existing-listing mark")
    # Source announcement after the market close on Sep23: recognize at the next
    # session, never retroactively at the August conversion or Sept23 decision.
    terms = {
        "schema": "BRAZIL_RV_CORPORATE_REPLAY_V3",
        "store": binding,
        "status": "cielo_brmalls_copel_accounting_amendment_copel_loan_allocation_bounded",
        "supersedes": pointer["corporate_replay"],
        "share_distributions": [
            {
                "isin": "BRBRMLACNOR9",
                "available_date": "2023-01-04",
                "effective_date": "2023-01-09",
                "last_trade_date": "2023-01-06",
                "loan_conversion_close_date": "2023-01-10",
                "payment_date": "2023-01-20",
                "cash_per_prior_share": 1.62899410177968,
                "loan_conversion_convention": "Jan10 close represented at Jan11 opening before new decisions; unchanged principal/rate across that zero-session boundary; no daily repricing",
                "cash_basis": "announced final amount already includes projected CDI to Jan13; no additional invented Jan13-Jan20 correction",
                "legs": [
                    {
                        "successor_isin": "BRALSOACNOR5",
                        "delivery_date": "2023-01-11",
                        "shares_per_prior_share": 0.398551577675763,
                        "loan_principal_fraction": 1.0,
                        "fractional_auction": {
                            "auction_date": "2023-01-24",
                            "announcement_date": "2023-01-25",
                            "available_date": "2023-01-26",
                            "payment_date": "2023-02-02",
                            "cash_per_share": 17.694416,
                            "payment_convention": "issuer's by-Feb2 deadline; no invented earlier sweep; bound earliest known Jan26 receipt separately",
                            "basis": "whole shareholder shares delivered; residual remains marked non-tradable successor claim until next-session recognition of auction result; fractional loan quantities retained",
                        },
                    }
                ],
            }
        ],
        "calendar": {
            "path": str(store / "date_index.npy"),
            "sha256": sha256_file(store / "date_index.npy"),
        },
        "sources": [{"path": r["path"], "sha256": r["sha256"]} for r in receipts],
        "source_manifest": source_record,
        "cash_cancellations": [
            {
                "isin": "BRCIELACNOR3",
                "legal_effective_date": "2024-09-23",
                "available_date": "2024-09-24",
                "recognition_date": "2024-09-24",
                "delivery_index_timestamp": "2024-09-23T19:18:00-03:00",
                "coverage_start": "2024-08-27",
                "coverage_available_date": "2024-08-27",
                "payment_date": "2024-09-26",
                "cash_per_share": 5.89,
                "reason": "known closed-register inventory followed by compulsory redemption; no automatic voluntary tender",
            }
        ],
        "loan_cash_settlements": [
            {
                "isin": "BRCIELACNOR3",
                "settlement_date": "2024-08-30",
                "available_date": "2024-08-30",
                "cash_per_share": price,
                "price_basis": "R$5.82 times SGS11 SELIC, 2024-08-16 inclusive / 2024-08-30 exclusive",
                "price_interval_count": len(days),
                "invoice_rounding_bound_per_share_brl": 0.01,
            }
        ],
        "unchanged": "model features, scores, targets, eligibility, raw quotes and accepted store",
        "pending_cases": [
            "DMMO",
            "ALLOS/ISA identity audit",
            "other exposed events",
        ],
    }
    copel = isins.index("BRCPLECDAM13")
    quoted = np.isfinite(prices[:, copel]) & (prices[:, copel] > 0)
    if str(calendar[np.flatnonzero(quoted)[-1]]) != "2023-12-22":
        raise ValueError("Copel unit terminal quote differs from issuer schedule")
    preceding = np.searchsorted(calendar, np.datetime64("2023-12-26")) - 1
    for isin in ["BRCPLEACNOR8", "BRCPLEACNPB9"]:
        if not np.isfinite(prices[preceding, isins.index(isin)]):
            raise ValueError("Copel constituent lacks a causal existing-listing mark")
    terms["share_distributions"].append(
        {
            "isin": "BRCPLECDAM13",
            "available_date": "2023-12-19",
            "effective_date": "2023-12-26",
            "last_trade_date": "2023-12-22",
            "cash_per_prior_share": 0.0,
            "payment_date": None,
            "legs": [
                {
                    "successor_isin": "BRCPLEACNOR8",
                    "delivery_date": "2023-12-28",
                    "shares_per_prior_share": 1.0,
                    "loan_principal_fraction": 0.2,
                },
                {
                    "successor_isin": "BRCPLEACNPB9",
                    "delivery_date": "2023-12-28",
                    "shares_per_prior_share": 4.0,
                    "loan_principal_fraction": 0.8,
                },
            ],
            "loan_principal_allocation_basis": "explicit research allocation per underlying share; issuer K not recovered; NOT a source-verified contractual allocation",
            "loan_principal_allocation_sensitivity": {
                "on_fraction": [0.0, 1.0],
                "pn_fraction": "1 - on_fraction",
                "scope": "entire admissible allocation interval; original total principal/rate retained; report cost and admission sensitivity before interpreting model results",
            },
            "loan_conversion_convention": "night processing preceding Dec28 custody, represented at Dec28 opening; inferred from manual and issuer credit, not a dated loan instruction",
            "loan_conversion_timing_sensitivity": "one session either side; same total principal/rate until physically deliverable offsets",
            "custody_disposal": "reference waits for credit; prearranged Dec26-Dec27 disposal settling after credit remains a separate execution sensitivity",
            "fraction_convention": "1 ON plus 4 PN per unit creates no new entitlement fractions for integer units; regular research fills remain continuous as elsewhere",
        }
    )
    output.mkdir(exist_ok=True)
    digest = write_json_atomic(path, terms)
    pointer["corporate_replay"] = {
        "path": str(path),
        "sha256": digest,
        "status": terms["status"],
    }
    write_json_atomic(pointer_path, pointer)
    print(json.dumps(pointer["corporate_replay"]))


if __name__ == "__main__":
    main()
