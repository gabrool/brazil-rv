"""Bound new held-event evidence and isolated action corrections, without replay."""

from datetime import date, datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
import shutil
import time

import numpy as np
import polars as pl
from pypdf import PdfReader

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.corporate_actions import (
    VerifiedActionTerm,
    verified_action_terms_to_table,
)
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.decision_clock import normalize_publication_time

PROJECT = Path(__file__).resolve().parents[1]


def main():
    started = time.perf_counter()
    pointer_path = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer_path.read_text())
    root = Path(run["root"]) / "held_event_sources"
    out = root / "qualification"
    out.mkdir(exist_ok=False)
    shutil.copyfile(__file__, out / "executed.py")
    shutil.copyfile(PROJECT / "ops/recover_held_event_sources.py", out / "retrieval.py")
    original = json.loads((PROJECT / "docs/v2_data_inputs.json").read_text())
    derived = json.loads((PROJECT / "docs/v2_economic_data_inputs.json").read_text())
    old = Path(original["store"]["root"])
    new = Path(derived["store"]["root"])
    manifest = bound_json(
        {
            "path": str(old / "manifest.json"),
            "sha256": original["store"]["manifest_sha256"],
        }
    )
    dates = np.load(old / "date_index.npy")
    isins = np.load(old / "isin_index.npy").tolist()
    assert dates[-1] == np.datetime64("2024-12-30") and len(isins) == 933
    indices = [
        bound_json(binding(root / name))
        for name in ("issuer_index.json", "allos_2019_index.json")
    ]
    selected = {}
    for path in sorted(root.glob("*_receipt.json")):
        receipt = bound_json(binding(path))
        pdf = Path(receipt["pdf"]["path"])
        assert binding(pdf)["sha256"] == receipt["pdf"]["sha256"]
        selected[pdf.stem] = {
            "pdf": receipt["pdf"],
            "receipt": binding(path),
            "text": binding(pdf.with_suffix(".txt")),
        }

    def source(protocol, text):
        actual = Path(selected[protocol]["text"]["path"]).read_text(encoding="utf8")
        assert text in actual, (protocol, text)
        return selected[protocol]

    source("726655", "18 de dezembro de 2019")
    source("721334", "20.12.2019")
    source("703585", "0,787808369")
    source("1264889", "0,121695988348")
    source("1274971", "50,20718")
    source("711337", "20 de setembro de 2019")
    source("718837", "0,12784527353")
    # The later bilingual delivery retains identical Portuguese contractual text.
    first = PdfReader(root / "718837.pdf").pages[0].extract_text()
    later = PdfReader(root / "718866.pdf").pages[0].extract_text()
    assert first == later

    cases = [
        dict(
            isin="BRNATUACNOR6",
            successor_isin="BRNTCOACNOR5",
            ratio="1",
            last_trade="2019-12-17",
            effective="2019-12-18",
            available="2019-12-17",
            credit="2019-12-20",
            protocols=["726655", "721334"],
            legal_consummation="2019-12-17",
            fraction="No fraction for integer original holdings under 1:1 conversion.",
        ),
        dict(
            isin="BRALSCACNOR0",
            successor_isin="BRALSOACNOR5",
            ratio="0.787808369",
            last_trade="2019-08-05",
            effective="2019-08-06",
            available="2019-08-06",
            credit=None,
            protocols=["702698", "702878", "703585", "aliansce_merger_protocol"],
            legal_consummation="2019-08-05 close",
            fraction="Protocol 2.3, PDF pp76-77: aggregate fractions, sell, distribute net fees. Auction amount/payment and explicit custody credit not recovered; attributed shares at Aug5 close are not silently called physical delivery.",
        ),
        dict(
            isin="BRSOMAACNOR3",
            successor_isin="BRAZZAACNOR9",
            ratio="0.121695988348",
            last_trade="2024-07-31",
            effective="2024-08-01",
            available="2024-08-01",
            credit="2024-08-05",
            protocols=["1264889", "1274971"],
            legal_consummation="2024-07-31 close",
            fraction="Whole shares plus non-tradable fractions; auction Aug22, result first available Aug23, payable by Aug26. Printed 50.20718 BRL is approximate, not an executable price.",
        ),
    ]
    windows = [
        (2019, date(2019, 8, 5), date(2019, 8, 8), ["ALSC3", "SSBR3", "ALSO3"]),
        (2019, date(2019, 12, 17), date(2019, 12, 20), ["NATU3", "NTCO3"]),
        (2024, date(2024, 7, 31), date(2024, 8, 5), ["SOMA3", "ARZZ3", "AZZA3"]),
        (2019, date(2019, 9, 17), date(2019, 9, 20), ["NATU3"]),
        (2019, date(2019, 11, 6), date(2019, 11, 7), ["NATU3"]),
    ]
    quotes, quote_sources = [], {}
    for year, start, stop, tickers in windows:
        rec = next(
            x
            for x in manifest["sources"]
            if str(x["path"])
            .replace("\\", "/")
            .endswith(f"/year={year}/equities_daily_{year}.parquet")
        )
        quote_sources[str(year)] = rec
        frame = (
            pl.scan_parquet(rec["path"])
            .filter(
                pl.col("trade_date").is_between(start, stop)
                & pl.col("ticker").is_in(tickers)
            )
            .collect()
        )
        quotes.append(frame)
    quote_frame = pl.concat(quotes).unique().sort("trade_date", "isin")
    quote_frame.write_parquet(out / "bounded_quotes.parquet")
    for case in cases:
        before = quote_frame.filter(
            (pl.col("trade_date") == date.fromisoformat(case["last_trade"]))
            & (pl.col("isin") == case["isin"])
        )
        after = quote_frame.filter(
            (pl.col("trade_date") == date.fromisoformat(case["effective"]))
            & (pl.col("isin") == case["successor_isin"])
        )
        assert before.height == after.height == 1
        assert after["quote_factor"][0] == before["quote_factor"][0] == 1
        i, j = isins.index(case["isin"]), isins.index(case["successor_isin"])
        t = int(np.searchsorted(dates, np.datetime64(case["effective"])))
        assert np.isnan(np.load(old / "prior_reference_close.npy", mmap_mode="r")[t, j])
        assert np.isnan(np.load(old / "raw_close.npy", mmap_mode="r")[t, i])
        case["indices"] = [i, j]
        case["source_last_close"] = before["close_brl"][0]
        case["successor_first_close"] = after["close_brl"][0]
        # Endpoint arithmetic is a shareholder entitlement, not a same-day input.
        case["gross_close_entitlement_per_original_share"] = str(
            Decimal(case["ratio"]) * Decimal(str(after["close_brl"][0]))
        )
        case["hundred_share_entitlement"] = str(Decimal(100) * Decimal(case["ratio"]))
        case["opening_reference"] = (
            "Successor public prior reference is missing. Runtime needs an explicit causal unit-identity valuation bridge; no future first close, loan alias or observed quote may be invented."
        )
        case["loan_disposition"] = (
            "Event-specific loan conversion clock, quantity truncation and rent treatment remain unadmitted. B3 index notices do not settle those terms."
        )
    # Gross source terms are admitted separately from net cash/bonus custody.
    amendments = [
        VerifiedActionTerm(
            "bonus",
            "BRNATUACNOR6",
            "CVM:019550",
            date(2019, 9, 18),
            date(2019, 9, 18),
            None,
            datetime(2019, 9, 17, 17, 40, tzinfo=timezone.utc),
            normalize_publication_time(
                datetime(2019, 9, 17, 17, 40, tzinfo=timezone.utc), precision="minute"
            ),
            2.0,
            0.0,
            "BRL",
            "CVM:711337",
            "One new ON per existing ON; Sep18 ex, Sep20 credit. No cash is paid for the printed tax basis.",
            "source_bound_gross_data_terms",
        ),
        VerifiedActionTerm(
            "jcp",
            "BRNATUACNOR6",
            "CVM:019550",
            date(2019, 11, 7),
            date(2019, 11, 7),
            date(2020, 2, 26),
            None,
            normalize_publication_time(
                datetime(2019, 11, 4, 0, 54, tzinfo=timezone.utc), precision="minute"
            ),
            1.0,
            0.12784527353,
            "BRL",
            "CVM:718837",
            "Nov3 Sunday 21:54 local receipt; Nov6 register, Nov7 ex; payment Feb26. Oct31 accounting credit is not payment or availability. 15% withholding is separate.",
            "source_bound_gross_data_terms",
        ),
    ]
    verified_action_terms_to_table(amendments).write_parquet(
        out / "natura_gross_action_amendments.parquet"
    )
    comparisons = []
    for term in amendments:
        t = int(np.searchsorted(dates, np.datetime64(term.ex_date)))
        i = isins.index(term.isin)
        values = {}
        for key in (
            "action_shares_per_prior_share",
            "action_cash_per_prior_share",
            "action_payment_session",
            "inferred_action_u1_mask",
            "inferred_action_c1_mask",
            "inferred_action_u2_mask",
        ):
            values[key] = np.load(old / (key + ".npy"), mmap_mode="r")[t, i].item()
            assert (
                values[key] == np.load(new / (key + ".npy"), mmap_mode="r")[t, i].item()
            )
        comparisons.append(
            {
                "date": str(term.ex_date),
                "old_and_accepted_derived": values,
                "source_q": term.shares_per_prior_share,
                "source_gross_cash": term.cash_per_prior_share,
                "available_at": term.available_at.isoformat(),
            }
        )
    old_units = Decimal(100)
    for r in comparisons:
        old_units *= Decimal(
            str(r["old_and_accepted_derived"]["action_shares_per_prior_share"])
        )
    arithmetic = {
        "hypothetical_100_original_shares_no_trading": {
            "old_units_after_two_events": str(old_units),
            "sourced_units": "200",
            "sourced_gross_jcp_claim": str(Decimal(200) * Decimal(".12784527353")),
            "issuer_15pct_withholding_cash": str(
                Decimal(200) * Decimal(".12784527353") * Decimal(".85")
            ),
        },
        "soma_100_original_fraction": str(
            Decimal(100) * Decimal(cases[2]["ratio"]) - 12
        ),
        "soma_auction_total_per_share": str(Decimal("916532.03") / 18255),
        "scope": "Independent Decimal entitlement arithmetic, not a booked NAV, net-CDI return, reinvestment assumption or tax-credit valuation.",
    }
    # The boundary itself has been read; no protected after-2024 consumer data.
    report = {
        "status": "source_contracts_verified_two_new_scalar_defects_pending_separate_propagation_and_accounting_admission",
        "exposure_ranking": run["loan_return_notice_qualification"],
        "old_store": original["store"],
        "accepted_derived_store": derived["store"],
        "index_bindings": [
            binding(root / x) for x in ("issuer_index.json", "allos_2019_index.json")
        ],
        "index_rows": sum(len(x["rows"]) for x in indices),
        "source_files": selected,
        "quote_sources_existing_receipts": quote_sources,
        "bounded_quote_rows": quote_frame.height,
        "bounded_quotes": binding(out / "bounded_quotes.parquet"),
        "cases": cases,
        "natura_scalar_comparisons": comparisons,
        "gross_action_amendments": binding(
            out / "natura_gross_action_amendments.parquet"
        ),
        "arithmetic": arithmetic,
        "visual_review": {
            "726655": [2],
            "721334": [2],
            "702698": [2],
            "702878": [1],
            "703585": [1],
            "aliansce_merger_protocol": [76, 77],
            "1264889": [2, 3],
            "1274971": [1],
            "711337": [1],
            "718837": [1],
        },
        "attempt_dispositions": {
            "index_first_attempt": "Index write completed; default-encoding read failed. UTF8 fix reused successful index, exact first executed code preserved.",
            "702644": "Capital-increase communication, not a custody/auction source.",
            "721222": "Withdrawal notice, not the share-credit schedule; 721334 supplies the conditional timetable ratified by 726655.",
            "718866": "Later bilingual filing has identical PT text. Its first Poppler page appeared white; unique economic terms use the visually readable earlier 718837.",
            "b3_search": "Bounded event/ticker/loan searches retrieved index/collateral circulars, not event-specific loan treatment; no implication that such circulars do not exist.",
        },
        "remaining": [
            "Same-ISIN Natura bonus has Sep18 economic quantity but Sep20 physical credit: current immediate split custody needs explicit handling.",
            "Natura Nov7 gross JCP needs separately reported withholding and any supported tax-credit/loan compensation; preserve Feb26 payment even past a replay end.",
            "Separately propagate the two source corrections through bounded wealth, labels, daily/native/auxiliary dependencies; do not rebuild unrelated accepted families or use changed coordinates for old fits.",
            "NATU/ALSC/SOMA plus source-bound SSBR->ALSO and ARZZ->AZZA need accounting claim reference and dated custody/loan/fraction admission. Unit issuer renames are not loan aliases.",
            "ALSC actual custody date and auction receipt remain unknown. SOMA printed auction price is approximate; ordinary loan fraction rules remain separate.",
            "Exposure list also contains ENAT under recall, R35513.8717 maximum; investigate next by held exposure, not PnL.",
        ],
        "first_internet_publication_and_unselected_disclosure_completeness": "not established by preserved receipts or document dates",
        "no_existing_store_or_model_mutation": True,
        "seconds": time.perf_counter() - started,
    }
    write_json_atomic(out / "manifest.json", report)
    run["held_event_source_audit"] = binding(out / "manifest.json")
    run["natura_gross_action_amendments"] = report["gross_action_amendments"]
    run["new_source_amendment_readiness"] = {
        "status": "separate_propagation_required_before_new_economic_refits",
        "evidence": binding(out / "manifest.json"),
        "accepted_store_preserved": derived["store"],
    }
    write_json_atomic(pointer_path, run)
    print(
        json.dumps(
            {
                "status": report["status"],
                "index_rows": report["index_rows"],
                "pdfs": len(selected),
                "quote_rows": quote_frame.height,
                "arithmetic": arithmetic,
                "seconds": report["seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
