"""Sparse sourced accounting amendments; never rebuild model coordinates."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from brazil_rv.execution.loan_contracts import LoanCashSettlement, LoanCashValue
from brazil_rv.execution.share_distributions import (
    FractionAuction,
    ShareDelivery,
    ShareDistribution,
)
from .artifacts import sha256_file


def load_corporate_replay(path, expected_sha256):
    path = Path(path)
    if sha256_file(path) != expected_sha256:
        raise ValueError("corporate replay terms changed")
    terms = json.loads(path.read_text(encoding="utf-8"))
    for source in [terms["calendar"], *terms["sources"]]:
        if sha256_file(Path(source["path"])) != source["sha256"]:
            raise ValueError("corporate replay source changed")
    calendar = np.load(terms["calendar"]["path"], allow_pickle=False)
    return terms, calendar


def apply_corporate_replay(inputs, terms, calendar, manifest_sha256):
    """Apply after loading frozen forecasts/policies, on the full replay calendar.

    Cash recognition follows the source's availability, not an earlier legal date.
    Arrays of signals, targets, features, eligibility and market observations retain
    their original objects. Callers retaining PolicyData must shallow-copy it and
    replace only .inputs; constructing it again would regenerate static features.
    """
    indices = np.asarray(inputs.session_indices)
    if not np.array_equal(indices, np.arange(indices[0], indices[0] + len(indices))):
        raise ValueError("corporate replay requires the full consecutive session axis")
    if not np.array_equal(
        np.asarray(inputs.dates, dtype="datetime64[D]"), calendar[indices]
    ):
        raise ValueError("corporate replay dates differ from the bound calendar")
    provenance = dict(inputs.source_artifact_hashes or {})
    if "corporate_replay" in provenance:
        raise ValueError("corporate replay already applied")
    source = f"corporate_replay:{manifest_sha256}"
    names = {isin: i for i, isin in enumerate(inputs.security_ids)}
    fields = (
        "action_shares_per_prior_share",
        "action_cash_per_prior_share",
        "action_session_resolved",
        "action_has_action",
        "action_payment_session",
    )
    changed = {key: np.array(getattr(inputs, key), copy=True) for key in fields}
    # Financial cash amounts retain issuer precision; existing values are exact
    # under this widening and no model feature changes its dtype.
    changed["action_cash_per_prior_share"] = changed[
        "action_cash_per_prior_share"
    ].astype(np.float64)
    initial = (
        np.zeros(len(names), bool)
        if inputs.initial_unresolved_action is None
        else inputs.initial_unresolved_action.copy()
    )

    def session(value):
        day = np.datetime64(value, "D")
        position = int(np.searchsorted(calendar, day))
        if position >= len(calendar) or calendar[position] != day:
            raise ValueError(f"corporate date {value} absent from bound calendar")
        return position - int(indices[0])

    distributions = list(inputs.share_distributions)
    for event in terms.get("share_distributions", ()):
        if event["isin"] not in names:
            continue
        if any(leg["successor_isin"] not in names for leg in event["legs"]):
            raise ValueError("corporate replay is missing its contractual successor")
        name = names[event["isin"]]
        effective = session(event["effective_date"])
        if effective < len(indices):
            if inputs.action_has_action[max(0, effective) :, name].any():
                raise ValueError(
                    "distribution conflicts with an existing source action"
                )
            changed["action_session_resolved"][max(0, effective) :, name] = True
            if effective < 0:
                initial[name] = False
            legs = []
            for leg in event["legs"]:
                auction = leg.get("fractional_auction")
                legs.append(
                    ShareDelivery(
                        names[leg["successor_isin"]],
                        leg["shares_per_prior_share"],
                        session(leg["delivery_date"]),
                        None
                        if auction is None
                        else FractionAuction(
                            session(auction["available_date"]),
                            auction["cash_per_share"],
                            session(auction["payment_date"]),
                            auction.get("provision_loan_fractions", False),
                            auction.get("zero_quantity_rent_through_payment", False),
                        ),
                        leg["loan_principal_fraction"],
                    )
                )
            distributions.append(
                ShareDistribution(
                    source_index=name,
                    effective_session=effective,
                    available_session=session(event["available_date"]),
                    legs=tuple(legs),
                    cash_per_prior_share=event["cash_per_prior_share"],
                    payment_session=None
                    if event["payment_date"] is None
                    else session(event["payment_date"]),
                    source=source,
                )
            )

    for event in terms["cash_cancellations"]:
        if event["isin"] not in names:
            continue
        name = names[event["isin"]]
        known = session(event["recognition_date"])
        if event["recognition_date"] < event["available_date"]:
            raise ValueError("corporate cash cannot precede source availability")
        coverage = session(event["coverage_start"])
        if event["coverage_start"] < event["coverage_available_date"]:
            raise ValueError("corporate coverage cannot precede source availability")
        if coverage < len(indices):
            if inputs.action_has_action[max(0, coverage) :, name].any():
                raise ValueError("corporate coverage conflicts with an existing action")
            changed["action_session_resolved"][max(0, coverage) :, name] = True
            if coverage < 0:
                initial[name] = False
        if 0 <= known < len(indices):
            changed["action_shares_per_prior_share"][known, name] = 0
            changed["action_cash_per_prior_share"][known, name] = event[
                "cash_per_share"
            ]
            changed["action_has_action"][known, name] = True
            changed["action_payment_session"][known, name] = session(
                event["payment_date"]
            )

    loans = list(inputs.loan_cash_settlements)
    for event in terms["loan_cash_settlements"]:
        if event["isin"] not in names:
            continue
        effective = session(event["settlement_date"])
        if effective < len(indices):
            loans.append(
                LoanCashSettlement(
                    names[event["isin"]],
                    effective,
                    session(event["available_date"]),
                    event["cash_per_share"],
                    source,
                    rent_payment_session=None
                    if event.get("rent_payment_date") is None
                    else session(event["rent_payment_date"]),
                    payment_session=None
                    if event.get("payment_date") is None
                    else session(event["payment_date"]),
                    valuations=tuple(
                        LoanCashValue(
                            session(value["available_date"]), value["cash_per_share"]
                        )
                        for value in event.get("valuations", ())
                    ),
                    unreturned_only=event.get("unreturned_only", False),
                    prohibit_new_borrow=event.get("prohibit_new_borrow", True),
                )
            )
    provenance["corporate_replay"] = manifest_sha256
    return replace(
        inputs,
        **changed,
        initial_unresolved_action=initial,
        loan_cash_settlements=tuple(loans),
        share_distributions=tuple(distributions),
        source_artifact_hashes=provenance,
        action_terms_source=f"{inputs.action_terms_source};{source}",
    )
