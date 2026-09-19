"""Bind the lender-elected cash endpoint; the default PNA remains a separate case."""

from datetime import datetime
import json
import math
from pathlib import Path

import numpy as np

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.corporate_replay import load_corporate_replay

PROJECT = Path(__file__).resolve().parents[1]


def main():
    pointer_path = PROJECT / "docs/v2_economic_data_scaling_run.json"
    pointer = json.loads(pointer_path.read_text())
    root = Path(pointer["root"])
    path = root / "corporate_replay/dommo_pnb_manifest.json"
    if path.exists():
        raise FileExistsError(path)
    base = pointer["corporate_replay"]
    terms, calendar = load_corporate_replay(base["path"], base["sha256"])
    source_path = PROJECT / pointer["dommo_source_terms"]["path"]
    if sha256_file(source_path) != pointer["dommo_source_terms"]["sha256"]:
        raise ValueError("Dommo source terms changed")
    dommo = json.loads(source_path.read_text(encoding="utf-8"))
    receipts = [
        json.loads((root / "primary_sources" / name).read_text())
        for name in (
            "dommo_approval_20221024.receipt.json",
            "cdi_sgs12_20221024_20230113.receipt.json",
        )
    ]
    for record in [*dommo["sources"], *receipts]:
        if sha256_file(Path(record["path"])) != record["sha256"]:
            raise ValueError("Dommo source changed")
    rates = {
        datetime.strptime(row["data"], "%d/%m/%Y").date().isoformat(): float(
            row["valor"]
        )
        / 100
        for row in json.loads(Path(receipts[1]["path"]).read_text())
    }
    final = dommo["lender_elected_PNB"]["final_cash_per_share"]
    reconstructed = 1.85 * math.prod(
        1 + v for d, v in rates.items() if d < "2023-01-13"
    )
    if abs(reconstructed - final) > 1e-10:
        raise ValueError("published CDI does not reconcile issuer redemption")

    def accrued(day):
        # End-of-session valuation, after this day's intention. Money-market
        # dates include Dec30, when equities did not trade. Never drop that CDI.
        return 1.85 * math.prod(1 + v for d, v in rates.items() if d <= day)

    values = [
        {"available_date": str(day), "cash_per_share": accrued(str(day))}
        for day in calendar
        if np.datetime64("2022-12-26") < day < np.datetime64("2023-01-09")
    ]
    values.append({"available_date": "2023-01-09", "cash_per_share": final})
    terms["loan_cash_settlements"].append(
        {
            "isin": dommo["identity"]["source_isin"],
            "available_date": "2022-12-21",
            "settlement_date": "2022-12-26",
            "cash_per_share": accrued("2022-12-26"),
            "rent_payment_date": "2022-12-28",
            "payment_date": "2023-01-13",
            "payment_date_available_from": "2023-01-09",
            "valuations": values,
            "unreturned_only": True,
            "prohibit_new_borrow": False,
        }
    )
    terms["sources"] += [*dommo["sources"], *receipts]
    terms["schema"] = "BRAZIL_RV_CORPORATE_REPLAY_V4"
    terms["status"] = "dommo_legal_lender_election_endpoint_default_PNA_separate"
    terms["parent_manifest"] = base
    terms["dommo_election_contract"] = {
        "scenario": "all legally eligible pre-existing contracts elected by lenders; no borrower outcome selection",
        "timing": "Dec26 after actual fills, excluding whole roots with pending returns and newly registered Dec26 D+1 loans",
        "cash_valuation": "1.85 compounded with daily SGS12 from Oct24 inclusive, through each mark day inclusive; fixed final issuer amount from Jan9 only",
        "payment_timing": "Jan13 is a future cash realization, not used to discount prior marks or release proceeds early; announced Jan6, available Jan9",
        "rent": "through Dec26, paid Dec28; no further rent or fees while redemption remains payable",
        "proceeds": "retained and remunerated until Jan13; exact intermediary restriction release remains an account assumption",
        "AGE_date": "2022-10-24",
        "CDI_observations": len(rates),
        "final_reconstructed": reconstructed,
        "final_published": final,
        "default_PNA": "not admitted by this endpoint; unresolved default loan fractions must not be treated as tradable fractional loans",
        "limitations": [
            "actual lender elections and maturity records unavailable; this endpoint is a scenario, not asserted historical participation",
            "pending early-return contracts remain outside this endpoint; no hindsight cancellation of observed policy fills",
            "not a complete DMMO model replay until default PNA obligations are integrated",
            "SGS archive has observation dates, not vintage receipt timestamps; marks are realized after decisions and historical revisions remain a source-audit boundary",
        ],
    }
    digest = write_json_atomic(path, terms)
    pointer["dommo_pnb_scenario"] = {
        "path": str(path),
        "sha256": digest,
        "status": terms["status"],
    }
    write_json_atomic(pointer_path, pointer)
    print(
        json.dumps(
            {
                "path": str(path),
                "sha256": digest,
                "initial_liability_per_share": accrued("2022-12-26"),
                "final_reconstructed": reconstructed,
            }
        )
    )


if __name__ == "__main__":
    main()
