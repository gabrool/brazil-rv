"""Admit the default PNA and retain the separate lender-elected PNB endpoint."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.corporate_replay import load_corporate_replay
from brazil_rv.v2.data_repair import bound_json

PROJECT = Path(__file__).resolve().parents[1]


def main():
    pointer_path = PROJECT / "docs/v2_economic_data_scaling_run.json"
    pointer = json.loads(pointer_path.read_text())
    destination = Path(pointer["root"]) / "corporate_replay/dommo_manifest.json"
    endpoint_path = destination.with_name("dommo_pnb_with_default_manifest.json")
    if destination.exists() or endpoint_path.exists():
        raise FileExistsError("preserve admitted source manifests")
    base = pointer["corporate_replay"]
    terms, calendar = load_corporate_replay(base["path"], base["sha256"])
    dommo = bound_json(pointer["dommo_source_terms"])
    for receipt in dommo["sources"]:
        if sha256_file(Path(receipt["path"])) != receipt["sha256"]:
            raise ValueError("Dommo primary source changed")
    store = Path(terms["store"]["root"])
    isins = np.load(store / "isin_index.npy").tolist()
    source, successor = (
        isins.index(dommo["identity"][k]) for k in ("source_isin", "successor_isin")
    )
    prices = np.load(store / "raw_close.npy", mmap_mode="r")
    quoted = np.flatnonzero(np.isfinite(prices[:, source]) & (prices[:, source] > 0))
    if str(calendar[quoted[-1]]) != "2023-01-06" or not np.isfinite(
        prices[quoted[-1], successor]
    ):
        raise ValueError(
            "default conversion does not match original dated quote identities"
        )
    terms["schema"] = "BRAZIL_RV_CORPORATE_REPLAY_V5"
    terms["supersedes"] = base
    terms["status"] = (
        "default_dommo_pna_admitted_with_explicit_fraction_and_custody_bounds"
    )
    terms["share_distributions"].append(
        {
            "isin": dommo["identity"]["source_isin"],
            "available_date": "2023-01-09",
            "effective_date": "2023-01-09",
            "last_trade_date": "2023-01-06",
            "cash_per_prior_share": 0.4625,
            "payment_date": "2023-01-17",
            "loan_conversion_close_date": "2023-01-10",
            "loan_conversion_convention": "manual night processing before Jan11 custody, represented Jan11 opening; one-session timing sensitivity requires corresponding custody/return handling",
            "legs": [
                {
                    "successor_isin": dommo["identity"]["successor_isin"],
                    "shares_per_prior_share": 0.0375,
                    "delivery_date": "2023-01-11",
                    "loan_principal_fraction": 1.0,
                    "fractional_auction": {
                        "available_date": "2023-03-31",
                        "payment_date": "2023-04-06",
                        "cash_per_share": 31.94031,
                        "provision_loan_fractions": True,
                        "zero_quantity_rent_through_payment": False,
                        "basis": "B3 manual contract-by-contract truncation; whole loan quantity preserves original principal and rate; separate marked negative auction entitlement. Shareholder fractions are positive custody entitlements.",
                        "price_precision_sensitivity": [565024 / 17690, 31.94031],
                        "receipt_timing_sensitivity": ["2023-03-31", "2023-04-06"],
                        "zero_quantity_contract_assumption": "If an original contract becomes less than one whole successor share, stop rent at Jan10 conversion close and pay accrued rent Jan11; compare continued original-principal rent through Apr6 payment. No observed invoice for this small-contract case is claimed.",
                    },
                }
            ],
            "custody_boundary": "Jan6 last spot purchases/covers settle Jan10 under registered T+2, before Jan11 credit. The implementation stops if a source loan return or source purchase is still undelivered, rather than rounding net holdings. No name is excluded.",
            "presettlement_disposal": "issuer permits Jan9 trading; reference waits for Jan11 custody, with prearranged Jan9-Jan10 disposal still an explicit execution sensitivity",
            "lender_choice": "default PNA; actual lender participation unknown; separate all-eligible-lender PNB endpoint must be reported",
        }
    )
    terms["pending_cases"] = [x for x in terms.get("pending_cases", []) if x != "DMMO"]
    terms["sources"] += [
        {"path": r["path"], "sha256": r["sha256"]}
        for r in dommo["sources"]
        if r["sha256"] not in {s["sha256"] for s in terms["sources"]}
    ]
    digest = write_json_atomic(destination, terms)
    old_endpoint = pointer["dommo_pnb_scenario"]
    prior, _ = load_corporate_replay(old_endpoint["path"], old_endpoint["sha256"])
    endpoint = deepcopy(terms)
    endpoint["status"] = (
        "all_eligible_lender_pnb_endpoint_with_default_pna_for_remaining_contracts"
    )
    endpoint["supersedes"] = old_endpoint
    endpoint["loan_cash_settlements"] = prior["loan_cash_settlements"]
    endpoint["sources"] += [
        r
        for r in prior["sources"]
        if r["sha256"] not in {s["sha256"] for s in endpoint["sources"]}
    ]
    endpoint_digest = write_json_atomic(endpoint_path, endpoint)
    pointer["corporate_replay"] = {
        "path": str(destination),
        "sha256": digest,
        "status": terms["status"],
    }
    pointer["dommo_pnb_scenario"] = {
        "path": str(endpoint_path),
        "sha256": endpoint_digest,
        "status": endpoint["status"],
    }
    write_json_atomic(pointer_path, pointer)
    print(
        json.dumps({k: pointer[k] for k in ("corporate_replay", "dommo_pnb_scenario")})
    )


if __name__ == "__main__":
    main()
