"""Finish saved opening-claim books; no optimizer/account or source reruns."""

from decimal import Decimal
import json
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np
import polars as pl

from brazil_rv.execution.share_distributions import (
    ShareDelivery,
    ShareDistribution,
    recognize_distribution,
)
from brazil_rv.v2.artifacts import write_json_atomic, sha256_file
from brazil_rv.v2.data_repair import binding

PROJECT = Path(__file__).resolve().parents[1]


def main():
    started = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    root = Path(run["root"]) / "opening_claims/qualified"
    output = root / "qualification"
    output.mkdir(exist_ok=False)
    shutil.copyfile(__file__, output / "executed.py")
    plan = json.loads((root / "plan.json").read_text())
    cases = json.loads((root / "completed.json").read_text())
    terms = json.loads((root / "terms.json").read_text())
    assert len(cases) == 30
    store = Path(terms["store"]["root"])
    dates = np.load(store / "date_index.npy")
    source_refs = np.load(store / "prior_reference_close.npy", mmap_mode="r")
    checks, oracles, arrays = [], [], {}
    for case in plan["cases"]:
        name, successor = case["indices"]
        day = int(np.searchsorted(dates, np.datetime64(case["effective"])))
        refs = source_refs[day].astype(np.float64)
        assert not np.isfinite(refs[successor])
        event = ShareDistribution(
            name,
            day,
            day,
            (ShareDelivery(successor, float(case["ratio"]), None),),
            "bound source and explicit opening continuity",
            carry_source_value=True,
        )
        legs = recognize_distribution(event, refs)
        expected = Decimal(str(float(refs[name]))) / Decimal(str(float(case["ratio"])))
        assert abs(Decimal(str(legs[0].opening_mark)) - expected) < Decimal("1e-12")
        assert not np.isfinite(refs[successor])
        oracles.append(
            dict(
                isin=case["isin"],
                source_stored=float(refs[name]),
                original_printed=case["source_last_close"],
                ratio=case["ratio"],
                decimal_stored_coordinate=str(expected),
                opening_mark=legs[0].opening_mark,
                original_printed_coordinate=str(
                    Decimal(str(case["source_last_close"])) / Decimal(case["ratio"])
                ),
                precision="Runtime uses the immutable Float32 source coordinate widened before division; original printed precision shown separately.",
            )
        )
    for case in cases:
        folder = root / case["book"]
        book = json.loads((folder / "book.json").read_text())
        for filename, digest in book["files"].items():
            assert sha256_file(folder / filename) == digest
        assert (
            book["status"] == "completed"
            and not book["legacy_slot_diagnostics_applicable"]
        )
        with np.load(folder / "account.npz") as z:
            arrays[case["book"]] = {k: z[k] for k in z.files}
        a = arrays[case["book"]]
        assert a["signed_shares"].shape == (30, 933)
        independent_nav = (
            a["free_cash"]
            + a["restricted_cash"]
            + a["hedge_restricted_cash"]
            + a["unsettled_cash"]
            + a["receivables"]
            - a["payables"]
            + (a["signed_shares"] * np.nan_to_num(a["mark_price"])).sum(axis=1)
            + a["hedge_signed_shares"] * np.nan_to_num(a["hedge_mark_price"])
            - a["loan_liability"]
        )
        np.testing.assert_allclose(independent_nav, a["nav"], rtol=0, atol=5e-8)
        assert np.max(np.abs(a["reconciliation_error"])) < 1e-6
        prefix = case["book"].split("_")[0]
        source_case = next(x for x in plan["cases"] if x["isin"][2:6] == prefix)
        source_name, destination = source_case["indices"]
        if (folder / "fills.parquet").exists():
            fills = pl.read_parquet(folder / "fills.parquet")
            source_fills = fills.filter(
                (pl.col("security_index") == source_name)
                & (pl.col("fill_session") >= 6)
            )
            assert source_fills.height == 0
        if prefix == "ALSC":
            assert np.all(a["signed_shares"][6:, source_name] != 0)
            assert case["final_undelivered"] > 0
        if source_case["credit"]:
            credit = (
                int(np.searchsorted(dates, np.datetime64(source_case["credit"])))
                - int(np.searchsorted(dates, np.datetime64(source_case["effective"])))
                + 6
            )
            assert np.all(a["signed_shares"][6:credit, source_name] != 0)
        checks.append(
            dict(
                book=case["book"],
                source_no_post_effect_fill=True,
                source_quantity_retained_until_delivery=True,
            )
        )
    contrasts = []
    for sign in (1, -1):
        for capital in plan["capital"]:
            key = f"SOMA_provisioned_{sign}_{capital}"
            a = arrays[key]
            for scenario, prefix in (("continuous_loan", 8), ("auction_quotient", 22)):
                b = arrays[f"SOMA_{scenario}_{sign}_{capital}"]
                # Prefix ends immediately before custody/auction recognition.
                np.testing.assert_array_equal(a["nav"][:prefix], b["nav"][:prefix])
                np.testing.assert_array_equal(
                    a["targets"][:prefix], b["targets"][:prefix]
                )
                contrasts.append(
                    dict(
                        sign=sign,
                        capital=capital,
                        scenario=scenario,
                        final_nav_difference_brl=float(b["nav"][-1] - a["nav"][-1]),
                        final_nav_difference_bps=float(
                            (b["nav"][-1] - a["nav"][-1]) / capital * 1e4
                        ),
                        max_path_difference_bps=float(
                            np.max(np.abs(b["nav"] - a["nav"])) / capital * 1e4
                        ),
                        prefix_exact_sessions=prefix,
                    )
                )
    report = dict(
        status="saved_30_adaptive_opening_claim_books_qualified_not_final_economic_admission",
        audit_root=str(root),
        plan=binding(root / "plan.json"),
        terms=binding(root / "terms.json"),
        completed=binding(root / "completed.json"),
        oracles=oracles,
        checks=checks,
        contrasts=contrasts,
        book_cells=sum(a.size for record in arrays.values() for a in record.values()),
        max_identical_intention_nav_error=max(
            c["identical_intention_nav_error"] for c in cases
        ),
        adaptive_uncertainty={
            str(cap): max(
                c["adaptive_nav_difference_bps"]
                for c in cases
                if c["book"].endswith("_" + str(cap))
            )
            for cap in plan["capital"]
        },
        max_adaptive_target_difference=max(
            c["adaptive_target_difference"] for c in cases
        ),
        readout_label_correction="completed.json opening_claim_value is effect-close total undelivered value after current quotes, not the pre-decision opening mark. The independent three scalar oracles establish opening continuity separately; canonical producer label corrected without book reruns.",
        aggregate_case_seconds=sum(c["seconds"] for c in cases),
        total_audit_wall_seconds=None,
        limits="No whole audit wall measurement survived final report failure. Case times include exact plus fixed/adaptive account paths and readouts. No old 24-case books repeated. Engineering scores/fixed risks and bundled execution cost are not corrected model pricing/alpha. NATU bonus/JCP, full held-event loan/custody admission, ALSC actual delivery/auction and other Stage A bounds remain.",
        attempts=[
            "Initial save_book omitted required scenario metadata after account arrays were written. First new 30-session exact book repeated to recover missing event/readout metadata; initial arrays retained.",
            "Qualified run completed/saved all30books and account comparisons, then oracle JSON failed on NumPyFloat32. This qualifier reuses all saved books, reconstructs three scalar oracles only. Canonical recipe casts report scalars; helper widens input before division. Actual accounts already use Float64 marks, so their executed numeric paths are unchanged.",
        ],
        seconds=perf_counter() - started,
    )
    write_json_atomic(output / "manifest.json", report)
    run["opening_claim_audit"] = binding(output / "manifest.json")
    write_json_atomic(pointer, run)
    print(
        json.dumps({k: v for k, v in report.items() if k not in ("checks", "oracles")}),
        flush=True,
    )


if __name__ == "__main__":
    main()
