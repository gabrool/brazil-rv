"""Source-bound account overlay for the new 2021/2023 evaluation periods."""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.decision_clock import load_session_schedule
from prepare_attention_events import save_source_overlay

PROJECT = Path(__file__).resolve().parents[1]


def main():
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    base = bound_json(run["scaling_expanded_evaluation_plan"])
    root = Path(run["scaling_expanded_evaluation_plan"]["path"]).parent
    out = root / "later_source_replays"
    out.mkdir(exist_ok=False)
    (out / "executed_preparation.py").write_bytes(Path(__file__).read_bytes())
    manifest = bound_json(
        dict(
            path=str(Path(base["store"]["root"]) / "manifest.json"),
            sha256=base["store"]["manifest_sha256"],
        )
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
    selections = {
        "837389": (2021, [1, 2]),
        "846082": (2021, [1]),
        "873872": (2021, [1, 2]),
        "864501": (2021, [1]),
        "871753": (2021, [1]),
        "874238": (2021, [1, 2, 3]),
        "1052903": (2023, [1]),
        "1056754": (2023, [1, 2]),
    }
    originals = {}
    for protocol, (year, pages) in selections.items():
        sources = root / f"event_sources_{year}"
        receipt = bound_json(binding(sources / f"{protocol}_receipt.json"))
        stamps = {
            re.search(r"\d{2}/\d{2}/\d{4} \d{2}:\d{2}", r["receipt"])[0]
            for r in receipt["issuer_rows"]
        }
        assert len(stamps) == 1
        known = (
            datetime.strptime(stamps.pop(), "%d/%m/%Y %H:%M").replace(
                tzinfo=ZoneInfo("America/Sao_Paulo")
            )
            + timedelta(minutes=1)
        ).astimezone(timezone.utc)
        originals[protocol] = dict(
            pdf=receipt["pdf"],
            receipt=binding(sources / f"{protocol}_receipt.json"),
            available_at=known.isoformat(),
            available_date=str(
                next(s.trade_date for s in schedule if s.decision_at >= known)
            ),
            visually_qualified_pages=[
                dict(
                    page=p, image=binding(sources / "rendered" / f"{protocol}_p{p}.png")
                )
                for p in pages
            ],
        )
    parent = bound_json(base["terms"])
    terms = dict(
        calendar=parent["calendar"],
        sources=[v["pdf"] for v in originals.values()],
        identity_actions=[],
        scalar_actions=[],
        share_distributions=[],
        cash_cancellations=[],
        loan_cash_settlements=[],
    )
    for before, after, effect, protocol in (
        ("BRGPCPACNOR4", "BRDEXPACNOR1", "2021-06-08", "874238"),
        ("BRGPCPACNPR1", "BRDEXPACNPR8", "2021-06-08", "874238"),
        ("BRWIZSACNOR1", "BRWIZCACNOR5", "2023-02-09", "1056754"),
    ):
        terms["identity_actions"].append(
            dict(
                predecessor_isin=before,
                successor_isin=after,
                effective_date=effect,
                available_date=originals[protocol]["available_date"],
            )
        )
    for (
        source,
        successor,
        effect,
        protocol,
        ratio,
        cash,
        pay,
        delivery,
        disposal,
        auction,
    ) in (
        (
            "BRRLOGACNOR4",
            "BRCSANACNOR6",
            "2021-03-08",
            "837389",
            1 / 3.943112,
            0,
            None,
            "2021-03-10",
            "2021-03-08",
            dict(
                available_date=originals["846082"]["available_date"],
                cash_per_share=93.72,
                payment_date="2021-03-29",
                provision_loan_fractions=False,
            ),
        ),
        (
            "BRSMLSACNOR1",
            "BRGOLLACNPR4",
            "2021-06-07",
            "873872",
            0.6601,
            5.11719919,
            "2021-06-23",
            "2021-06-09",
            None,
            dict(
                available_date=None,
                cash_per_share=None,
                payment_date=None,
                provision_loan_fractions=False,
            ),
        ),
    ):
        terms["share_distributions"].append(
            dict(
                isin=source,
                effective_date=effect,
                available_date=originals[protocol]["available_date"],
                cash_per_prior_share=cash,
                payment_date=pay,
                carry_source_value=False,
                legs=[
                    dict(
                        successor_isin=successor,
                        shares_per_prior_share=ratio,
                        delivery_date=delivery,
                        disposal_date=disposal,
                        fractional_auction=auction,
                        loan_principal_fraction=1.0,
                    )
                ],
            )
        )
    terms["hypotheses"] = [
        "RLOG ratio is one CSAN per3.943112 RLOG, not3.943112 CSAN per RLOG. Physical creditMarch10; original expressly permits shareholder tradingMarch8, represented by existing prearranged owned disposal with regular spot settlement. Loan conversion at custody/continuous quantity preserving original terms remains an explicit unsourced lender convention.",
        "RLOG printed93.72 fraction payment is primary; aggregate2572750.87/27449 is a separate precision endpoint, not invented net fees or invoice precision. KnowledgeMarch26, paymentMarch29; do not backdate toMarch19 auction.",
        "Smiles automatically receives the base .6601 GOLL+5.11719919 cash absent prior election. Do not choose the optional exchange retrospectively. June9 custody/June23 gross cash; unknown fraction-auction terms remain null, with actual signed claims retained. No domestic-CNPJ income-tax exemption claimed.",
        "GPC and WIZ meeting approvals resolve the earlier conditional notices before the effective trading changes. Same legal company and same share class, unit/no-cash account transfers only; no new loan/source alias, acquired-issuer pooling or model-history permission.",
        "All forecasts, static coordinates, eligibility, allocation risk and accepted model data remain fixed. Missing GPC/Wiz histories and RLOG/Smiles labels are separately identified data dependencies, not silently repaired here.",
        "Linx's .0126730 StoneCo BDR successor is outside the933-axis contract, and final adjusted cash was not known byJune30. Its3 held dates remain explicitly unresolved; no zero-priced leg, invented quote or future cash. The seven2021 candidate originals are not all admitted here.",
    ]
    write_json_atomic(out / "incremental_terms.json", terms)
    write_json_atomic(
        out / "source_admission.json",
        dict(
            originals=originals,
            terms=terms,
            exposure=run["scaling_expanded_qualification"],
            controls="Reuse qualified existing account transfers/custody/fractions/knowledge clocks.24 new source books, F7/F11 only, under the frozen pre-hedge runtime; compare existing baselines. No fit/feature/source census.",
            bounds="Freeze RLOG custody-first versus sourced precredit disposal; printed fraction93.72 versus aggregate quotient; Smiles next-session custody versus June9; debit+50/+100annualbp, only actual exposures. No automatic optional-election, forced-loan or broad scenario grid.",
        ),
    )
    save_source_overlay(
        base,
        terms,
        parent,
        out,
        run,
        ["F7", "F11"],
        "scaling_expanded_later_source_plan",
    )


if __name__ == "__main__":
    main()
