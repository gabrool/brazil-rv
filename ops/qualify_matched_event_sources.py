"""Bind dated originals for the corporate transitions exposed by Stage C."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

import numpy as np

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.decision_clock import load_session_schedule

PROJECT = Path(__file__).resolve().parents[1]
# These pages were rendered and read visually. Other saved pages are not claimed.
PAGES = {
    "1022908": [1],
    "1277563": [1],
    "793185": [1],
    "792562": [1],
    "797479": [1],
    "796144": [1, 2],
    "779297": [3],
    "1011593": [1, 2],
    "1015215": [1, 2],
    "994051": [1],
    "994841": [1, 2],
    "993427": [1],
    "994891": [1],
    "1007071": [1],
    "1012443": [1],
    "1013434": [1],
    "1043200": [1, 2],
    "1044628": [1],
    "1053743": [1, 2],
    "1054807": [1],
    "807216": [1, 2],
    "649597": [1],
    "634915": [1, 2],
    "805360": [1],
    "b3_189_2022": [1, 4],
}


def main():
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text(encoding="utf8"))
    root = Path(run["stage_c_root"])
    sources = root / "event_sources"
    out = root / "event_admission"
    out.mkdir(exist_ok=False)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    parent = bound_json(run["economic_refit_inputs"])["store"]
    store = Path(parent["root"])
    manifest = bound_json(
        dict(path=str(store / "manifest.json"), sha256=parent["manifest_sha256"])
    )
    schedule = load_session_schedule(
        Path(
            next(
                s["path"]
                for s in manifest["sources"]
                if Path(s.get("path", "")).name
                == "b3_session_schedule_reconstructed_v1.csv"
            )
        )
    )
    dates = np.load(store / "date_index.npy")
    decision = {s.trade_date: s.decision_at for s in schedule}
    clocks = [decision[d.astype(object)] for d in dates]
    names = np.load(store / "isin_index.npy").tolist()
    originals = {}
    for protocol, pages in PAGES.items():
        receipt = bound_json(binding(sources / (protocol + "_receipt.json")))
        pdf = binding(sources / (protocol + ".pdf"))
        assert receipt["pdf"] == pdf
        if protocol.isdigit():
            stamps = {
                re.search(r"\d{2}/\d{2}/\d{4} \d{2}:\d{2}", r["receipt"])[0]
                for r in receipt["issuer_rows"]
            }
            assert len(stamps) == 1
            local = datetime.strptime(stamps.pop(), "%d/%m/%Y %H:%M").replace(
                tzinfo=ZoneInfo("America/Sao_Paulo")
            )
            known = (local + timedelta(minutes=1)).astimezone(timezone.utc)
        else:
            # Dated circular, with no independently established intraday release.
            known = datetime(
                2022, 12, 21, tzinfo=ZoneInfo("America/Sao_Paulo")
            ).astimezone(timezone.utc)
        usable = next(str(d) for d, c in zip(dates, clocks, strict=True) if c >= known)
        originals[protocol] = dict(
            pdf=pdf,
            receipt=binding(sources / (protocol + "_receipt.json")),
            text=binding(sources / (protocol + ".txt")),
            available_at=known.isoformat(),
            available_date=usable,
            visually_qualified_pages=[
                dict(
                    page=p, image=binding(sources / "rendered" / f"{protocol}_p{p}.png")
                )
                for p in pages
            ],
        )

    def known(*protocols):
        return max(originals[p]["available_date"] for p in protocols)

    def event(
        label,
        pred,
        succ,
        day,
        ratio,
        protocols,
        *,
        delivery=None,
        cash="0",
        payment=None,
        reopening=None,
        auction=None,
    ):
        assert pred in names and succ in names
        return dict(
            id=label,
            isin=pred,
            successor_isin=succ,
            effective_date=day,
            available_date=known(*protocols),
            shares_per_prior_share=ratio,
            cash_per_prior_share=cash,
            source_protocols=list(protocols),
            delivery_date=delivery,
            payment_date=payment,
            source_reopens_date=reopening,
            fractional_auction=auction,
        )

    events = [
        event(
            "LCAM_RENT",
            "BRLCAMACNOR3",
            "BRRENTACNOR4",
            "2022-07-04",
            "0.43884446",
            ("994051", "994841", "994891"),
            delivery="2022-07-06",
            cash="0.8374919192",
            payment="2022-08-16",
            auction=dict(
                available_date=known("1013434"),
                cash_per_share=63.200889473,
                payment_date="2022-09-14",
            ),
        ),
        event(
            "JSL_holding_SIMPAR",
            "BRJSLGACNOR2",
            "BRSIMHACNOR0",
            "2020-09-18",
            "1",
            ("793185", "792562"),
            delivery="2020-09-18",
            reopening="2020-11-11",
        ),
        event(
            "JSL_logistics_ticker",
            "BRJSLGA02OR1",
            "BRJSLGACNOR2",
            "2020-11-11",
            "1",
            ("805360",),
            delivery="2020-11-11",
        ),
        event(
            "RRRP_BRAV",
            "BRRRRPACNOR5",
            "BRBRAVACNOR3",
            "2024-09-09",
            "1",
            ("1277563",),
            delivery="2024-09-09",
        ),
        event(
            "BKBR_ZAMP",
            "BRBKBRACNOR4",
            "BRZAMPACNOR5",
            "2022-10-26",
            "1",
            ("1022908",),
            delivery="2022-10-26",
        ),
        event(
            "TIMP_TIMS",
            "BRTIMPACNOR1",
            "BRTIMSACNOR5",
            "2020-10-13",
            "1",
            ("796144", "797479"),
            delivery="2020-10-13",
        ),
        event(
            "VIVT_PN_ON",
            "BRVIVTACNPR7",
            "BRVIVTACNOR0",
            "2020-11-23",
            "1",
            ("807216",),
            delivery="2020-11-23",
        ),
    ]
    for pred, ratio in (("BRMODLCDAM13", "3"), ("BRMODLACNPR9", "1")):
        events.append(
            event(
                "MODL_" + pred,
                pred,
                "BRMODLACNOR2",
                "2022-09-19",
                ratio,
                ("1011593", "1015215"),
                delivery="2022-09-19",
            )
        )
    for pred, ratio in (("BRVVARCDAM10", "3"), ("BRVVARACNPR8", "1")):
        events.append(
            event(
                "VVAR_" + pred,
                pred,
                "BRVVARACNOR1",
                "2018-11-26",
                ratio,
                ("634915", "649597"),
                delivery="2018-11-26",
            )
        )
    for pred, ratio in (
        ("BRSULACDAM12", "0.765234"),
        ("BRSULAACNOR2", "0.255078"),
        ("BRSULAACNPR9", "0.255078"),
    ):
        events.append(
            event(
                "SULA_" + pred,
                pred,
                "BRRDORACNOR8",
                "2022-12-26",
                ratio,
                ("1043200", "b3_189_2022"),
                delivery="2022-12-26",
                auction=dict(
                    available_date=known("1054807"),
                    cash_per_share=31.920637649,
                    payment_date="2023-02-10",
                ),
            )
        )
    assert all(e["available_date"] <= e["effective_date"] for e in events)
    statements = {
        "identities": "BKBR/ZAMP, RRRP/BRAV and JSLG11/new-logistics-JSLG3 are sourced same-company ordinary-share renames. TIM and old-JSL/SIMPAR are economic exchanges with different legal issuers, not unrestricted issuer-history aliases. Class/unit conversions and acquired companies do not pool histories into existing successors.",
        "delivery": "LCAM July6 and TIM October13 are explicit delivery/deposit dates. JSL holding, VIVT, MODL, VVAR and SULA use same-effective-date whole-share delivery as a separately labelled operational hypothesis tied to the sourced completed conversion/new-share trading date, not an observed custodian receipt. Freeze +2 accepted-session delivery alternatives independently on actual exposed paths. No date is claimed as a newly recovered historical fact. Prior ALSC unknown-credit locked contract is unchanged.",
        "loans": "Continuous successor quantities at the selected delivery date, retaining original principal/rate/fees/pending returns, are explicit hypotheses. No new rate, locate, source alias or B3 fractional-loan permission is inferred. Separately provision LCAM/SULA fractions per original contract on actually exposed short paths using existing qualified machinery.",
        "LCAM_cash": "R0.8374919192/share entitlement known July4 from July1 receipt994891; August16 payment only known August12 from1007071. The later payment date is a realization schedule, never a predictor, funding before payment or effect-date knowledge. Source June27/July1 wording about90days differs in start date; explicit August11 payment notice resolves actual selected schedule. Source typo AGE2022 is not corrected into an invented document.",
        "SULA_ratio": "December20 final issuer1043200 and dated B3 circular189 both state .255078/share and .765234/unit. January31 auction notice1053743 instead repeats April14's .25610/.76830 approval ratio. Preserve this conflict; use the two contemporaneous final sources for the December exchange. Treat January's ratio as a stale initial-reference inference, not independent proof of an amended final ratio. February2 auction/payment facts remain separately admissible.",
        "JSL_reuse": "Old holding JSLG equity converts to SIMH September18. Original805360 explicitly maps BRJSLGA02OR1 to reused BRJSLGACNOR2 November11 for logistics; no3.058 split is reported. Remove the intervening-price inferred3.058 action at this boundary and prevent old holding history from becoming logistics history. Source retirement ends only at the separately sourced November11 reopening; no old entitlement survives as new JSL stock.",
        "cash_exclusions": "Dissenter reimbursements, tax cost bases, optional credit facilities and subsidiary/treasury values are not ordinary portfolio cash. LCAM acquisition dividend is separate from the exchange ratio. Integer3:1 unit conversions produce no issuer fractional auction; fractional research positions remain explicit.",
        "limits": "Original receipts support minute-end knowledge, not archive revision/disclosure completeness. B3 collateral/index treatment is not client custody or model/loan history permission. SULA approximate fraction price is separate from net-total/share quotient; neither establishes exact client invoice precision. No accepted store or canonical account pointer changes in this admission.",
    }
    write_json_atomic(
        out / "plan.json",
        dict(
            exposure=run["stage_c_exposure_audit"],
            input_scope=binding(root / "event_input_scope/manifest.json"),
            previous_primary=run["corporate_replay"],
            original_replay_plan=run["stage_c_plan"],
            registration=binding(
                PROJECT / "research/preregistrations/v2_economic_data_scaling.md"
            ),
            contrasts="Reuse160 completed books. Add all newly evidenced terms together on affected folds without changing frozen forecasts/static coordinates. Delivery+2 one event at a time only if exposed; LCAM/SULA original-loan fraction convention and SULA printed/quotient precision only if exposed; existing negotiated cost/loan variants remain required on actual paths. No repeated old engineering campaign.",
            data="Separate source/target corrections and bounded identity dependencies, including reused JSL axis, before complete refit-store admission. Preserve all names/history and earlier stores. Old fits never read amended coordinates.",
            statements=statements,
        ),
    )
    write_json_atomic(
        out / "source_admission.json",
        dict(
            status="qualified_source_facts_with_explicit_account_hypotheses_pending_implementation",
            originals=originals,
            events=events,
            statements=statements,
            visual_pdf_count=len(PAGES),
            visual_page_count=sum(map(len, PAGES.values())),
            unqualified_saved_leads=[
                binding(sources / (p + ".pdf")) for p in ("653030", "992467")
            ],
            decimal_checks=dict(
                unit3=str(Decimal(1) + 2 * Decimal(1)),
                sula_unit=str(3 * Decimal(".255078")),
                sula_fraction_quotient=str(Decimal("1213618.83") / Decimal(38057)),
            ),
        ),
    )
    terms = deepcopy(bound_json(run["corporate_replay"]))
    terms["status"] = "actual_exposure_candidate_not_integrated_acceptance"
    terms["supersedes"] = run["corporate_replay"]
    terms["actual_exposure_sources"] = binding(out / "source_admission.json")
    terms["sources"].extend([v["pdf"] for v in originals.values()])
    for e in events:
        if e["id"] in ("RRRP_BRAV", "BKBR_ZAMP", "JSL_logistics_ticker"):
            terms["identity_actions"].append(
                dict(
                    predecessor_isin=e["isin"],
                    successor_isin=e["successor_isin"],
                    effective_date=e["effective_date"],
                    available_date=e["available_date"],
                    source_event=e["id"],
                )
            )
            continue
        terms["share_distributions"].append(
            dict(
                isin=e["isin"],
                effective_date=e["effective_date"],
                available_date=e["available_date"],
                cash_per_prior_share=float(e["cash_per_prior_share"]),
                payment_date=e["payment_date"],
                payment_available_date=known("1007071")
                if e["id"] == "LCAM_RENT"
                else None,
                source_reopens_date=e["source_reopens_date"],
                carry_source_value=True,
                source_event=e["id"],
                legs=[
                    dict(
                        successor_isin=e["successor_isin"],
                        shares_per_prior_share=float(e["shares_per_prior_share"]),
                        delivery_date=e["delivery_date"],
                        loan_principal_fraction=1.0,
                        fractional_auction=e["fractional_auction"],
                    )
                ],
                hypotheses=statements["delivery"] + " " + statements["loans"],
            )
        )
    terms["scalar_actions"].append(
        dict(
            isin="BRJSLGACNOR2",
            effective_date="2020-11-11",
            available_date=known("805360"),
            shares_per_prior_share=1.0,
            gross_cash_per_prior_share=0.0,
            payment_date=None,
            source_event="Remove source-disproved old-holding/logistics price-ratio inferred split; no prior holder receives3.058 logistics shares.",
        )
    )
    write_json_atomic(out / "candidate_terms.json", terms)
    run["stage_c_event_source_admission"] = binding(out / "source_admission.json")
    run["stage_c_event_candidate_terms"] = binding(out / "candidate_terms.json")
    run["stage_c_event_input_scope"] = binding(root / "event_input_scope/manifest.json")
    write_json_atomic(pointer, run)
    print(
        json.dumps(
            dict(
                events=len(events),
                visual_originals=len(PAGES),
                visual_pages=sum(map(len, PAGES.values())),
                sources=binding(out / "source_admission.json"),
            )
        )
    )


if __name__ == "__main__":
    main()
