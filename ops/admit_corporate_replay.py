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
    if output.exists():
        raise FileExistsError(output)
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
    # Source announcement after the market close on Sep23: recognize at the next
    # session, never retroactively at the August conversion or Sept23 decision.
    terms = {
        "schema": "BRAZIL_RV_CORPORATE_REPLAY_V1",
        "store": binding,
        "status": "cielo_admitted_accounting_only_other_cases_pending",
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
            "BRML",
            "DMMO",
            "CPLE",
            "ALLOS/ISA identity audit",
            "other exposed events",
        ],
    }
    output.mkdir()
    path = output / "manifest.json"
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
