"""Finish the produced lending layer with source-specific denominator evidence.

Reuses the completed source union, six-field control and feature computation.
The initial assertion that no denominator could lose support was too strong:
the predecessor's existing capital uncertainty must survive its source rename.
"""

from dataclasses import asdict
from datetime import date
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from brazil_rv.v2 import round5_b3 as b3
from brazil_rv.v2 import round5_cvm as cvm
from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json, source_families
from brazil_rv.v2.lending_archive import (
    LENDING_ARCHIVE_SCHEMA,
    align_lending_borrow_panels,
)
from brazil_rv.v2.round5_derived import verified
from brazil_rv.v2.round5_store import align_family
from propagate_fca_financials import compare

PROJECT = Path(__file__).resolve().parents[1]
KEYS = ["date", "isin"]


def main():
    started = perf_counter()
    run = json.loads(
        (PROJECT / "docs/v2_economic_data_scaling_run.json").read_text(encoding="utf8")
    )
    pointer = json.loads(
        (PROJECT / "docs/v2_data_inputs.json").read_text(encoding="utf8")
    )
    output = Path(run["root"]) / "lending_feature_propagation"
    assert not (output / "manifest.json").exists()
    (output / "executed_qualification.py").write_bytes(Path(__file__).read_bytes())
    store = Path(pointer["store"]["root"])
    manifest = bound_json(
        {
            "path": str(store / "manifest.json"),
            "sha256": pointer["store"]["manifest_sha256"],
        }
    )
    old_inputs = bound_json(source_families(manifest)["lending"]["source_manifest"])[
        "sources"
    ]["inputs"]
    peer = bound_json(run["fca_peer_propagation"])
    float_inputs = bound_json(peer["lending_source_bindings"])
    rename = bound_json(run["rename_history_propagation"])
    bound_json(run["rename_issuer_propagation"])
    axis, isins = np.load(store / "date_index.npy"), np.load(store / "isin_index.npy")
    assert axis[-1] <= np.datetime64("2024-12-30")
    sessions = axis.astype(object).tolist()
    active = np.load(verified(rename["arrays"]["active"]), mmap_mode="r")
    ti, ni = np.where(active)
    active_keys = pl.DataFrame({"date": axis[ti], "isin": isins[ni]})
    old = pl.read_parquet(verified(peer["artifacts"]["lending"]))
    final = pl.read_parquet(output / "features.parquet")
    source_only = pl.read_parquet(output / "source_only_features.parquet")
    names = [
        c for c in old.columns if c not in KEYS and not c.endswith("_age_sessions")
    ]
    basic = [c for c in old.columns if not c.startswith("utilization_proxy")]

    # Count age changes only when the age is known (nonnegative). Null and -1
    # both mean unavailable, never extra observations in the final tensor.
    def known_ages(frame):
        return frame.with_columns(
            pl.when(pl.col(c) >= 0).then(pl.col(c)).alias(c)
            for c in frame.columns
            if c.endswith("_age_sessions")
        )

    stats = compare(known_ages(old), known_ages(final), active_keys)
    joined = old.join(final, on=KEYS, how="full", coalesce=True, suffix="_after").join(
        active_keys, on=KEYS
    )
    losses = joined.filter(
        pl.col("utilization_proxy").is_not_null()
        & pl.col("utilization_proxy_after").is_null()
    )
    assert losses.height == 126 and losses["isin"].unique().to_list() == [
        "BRALOSACNOR5"
    ]
    assert not any(stats[n]["lost"] for n in names if n != "utilization_proxy")
    selection = bound_json(run["lending_denominator_audit"])["artifacts"][
        "selected_denominators"
    ]
    original = (
        pl.read_parquet(verified(selection))
        .join(losses.select(KEYS), on=KEYS)
        .sort(KEYS)
    )
    assert original.height == 126 and original["document_id"].unique().to_list() == [
        "127027"
    ]
    assert original["snapshot_date"].unique().to_list() == [date(2023, 3, 21)]
    assert original["float_known_date"].unique().to_list() == [date(2023, 5, 19)]
    assert original["denominator"].unique().to_list() == [532365440.0]
    market_root = Path(float_inputs["base_inputs"][0]["path"]).parent
    distribution = np.load(market_root / "distribution_number.npy", mmap_mode="r")
    predecessor = isins.tolist().index("BRALSOACNOR5")
    boundary = sessions.index(date(2023, 5, 2))
    assert (
        distribution[boundary - 1, predecessor] == 102
        and distribution[boundary, predecessor] == 103
    )
    assert np.load(market_root / "observed.npy", mmap_mode="r")[
        boundary - 1 : boundary + 1, predecessor
    ].all()
    original.with_columns(
        pl.lit(date(2023, 5, 2)).alias("predecessor_distribution_boundary"),
        pl.lit(
            "same unchanged capital-unit uncertainty rule; no proven split or guessed denominator adjustment"
        ).alias("reason"),
    ).write_parquet(output / "denominator_support_losses.parquet")
    # The retained ratios are an exact arithmetic consequence of the old
    # source denominator, not a new claim that the denominator is wrong.
    np.testing.assert_array_equal(
        (original["quantity"].to_numpy() / original["denominator"].to_numpy()).astype(
            np.float32
        ),
        losses.sort(KEYS)["utilization_proxy"].to_numpy(),
    )

    balances = pl.read_parquet(output / "lending_balances.parquet")
    recovered_root = Path(run["recovered_lending"]["root"])
    bound_json(
        {
            "path": str(recovered_root / "manifest.json"),
            "sha256": run["recovered_lending"]["manifest_sha256"],
        }
    )
    rates_record = {
        "path": str(recovered_root / "lending_rates.parquet"),
        "sha256": run["recovered_lending"]["rates_sha256"],
    }
    rates = pl.read_parquet(verified(rates_record))
    # Verify no prefix can depend on post-cutoff publications. Only the prefix
    # is recomputed; the already passed full feature reconstruction is reused.
    cutoff = date(2023, 12, 28)
    stop = sessions.index(cutoff) + 1
    volume = np.load(
        Path(old_inputs["base_manifest"]["path"]).parent / "volume_brl.npy"
    )[:stop].copy()
    links = pl.read_parquet(verified(rename["history_mapping"])).to_dicts()
    for link in sorted(links, key=lambda r: r["effective_index"]):
        if max(link["effective_index"], link["known_index"]) >= stop:
            continue
        p, n, e = (
            link[k] for k in ("predecessor_index", "successor_index", "effective_index")
        )
        volume[:e, n] = volume[:e, p]
    volume[sessions.index(date(2011, 2, 16)), isins.tolist().index("BRGOLLACNPR4")] = (
        21129704
    )
    prefix = (
        b3.lending_decision_features(
            balances.filter(pl.col("available_date") <= cutoff),
            rates.filter(pl.col("available_date") <= cutoff),
            sessions[:stop],
            isins.tolist(),
            volume,
        )
        .select(basic)
        .cast({c: final.schema[c] for c in basic})
    )
    prefix = known_ages(prefix).join(active_keys, on=KEYS).sort(KEYS)
    expected = known_ages(final.select(basic)).filter(pl.col("date") <= cutoff)
    assert_frame_equal(prefix, expected, check_exact=True)
    # Independent actual family loader on source expansion and rename dates.
    days = [
        date.fromisoformat(s)
        for s in [
            "2019-10-02",
            "2020-07-16",
            "2020-10-16",
            "2023-07-11",
            "2023-10-25",
            "2023-10-27",
            "2024-05-02",
            "2024-05-03",
            "2024-11-18",
            "2024-12-30",
        ]
    ]
    actual = align_family(
        final.filter(pl.col("date").is_in(days)), days, isins.tolist(), names
    )
    expected_values = np.zeros_like(actual[0])
    expected_valid = np.zeros_like(actual[1])
    expected_ages = np.full_like(actual[2], -1)
    positions, columns = (
        {d: i for i, d in enumerate(days)},
        {s: i for i, s in enumerate(isins)},
    )
    for row in final.filter(pl.col("date").is_in(days)).iter_rows(named=True):
        t, n = positions[row["date"]], columns[row["isin"]]
        for f, field in enumerate(names):
            value, age = row[field], row[field + "_age_sessions"]
            if value is not None and np.isfinite(value):
                expected_values[t, n, f], expected_valid[t, n, f] = value, True
            if age is not None and np.isfinite(age) and age >= 0:
                expected_ages[t, n, f] = age
    for found, expected in zip(
        actual, (expected_values, expected_valid, expected_ages)
    ):
        np.testing.assert_array_equal(found, expected)
    old_balances = pl.read_parquet(verified(old_inputs["balances"]))
    old_rates = pl.read_parquet(verified(old_inputs["rates"]))
    common = dict(
        canonical_dates=sessions,
        canonical_isins=isins.tolist(),
        manifest_sha256="paired_derived_diagnostic",
        balance_sha256="paired_sources",
        rate_sha256=rates_record["sha256"],
        source_label="paired_source_attribution",
    )
    previous = align_lending_borrow_panels(
        pl.read_parquet(recovered_root / "lending_balances.parquet"), rates, **common
    )
    current = align_lending_borrow_panels(balances, rates, **common)
    for field in [
        "annual_taker_rate",
        "rate_imputed",
        "rate_placeholder",
        "shortable_strict",
        "shortable_open",
    ]:
        np.testing.assert_array_equal(getattr(previous, field), getattr(current, field))
    np.savez_compressed(
        output / "equity_borrow_panels.npz",
        **{k: v for k, v in asdict(current).items() if isinstance(v, np.ndarray)},
    )
    borrowing = {
        "eligible_gained": int(
            (active & current.shortable_balance & ~previous.shortable_balance).sum()
        ),
        "eligible_lost": int(
            (active & ~current.shortable_balance & previous.shortable_balance).sum()
        ),
        "rates_imputation_placeholder_strict_open_exact": True,
    }
    economic = output / "economic_archive"
    economic.mkdir()
    (economic / "lending_balances.parquet").hardlink_to(
        output / "lending_balances.parquet"
    )
    (economic / "lending_rates.parquet").hardlink_to(
        recovered_root / "lending_rates.parquet"
    )
    october = bound_json(float_inputs["old_lending"]["previous_manifest"])
    support = bound_json(october["previous_manifest"])
    quarantine_record = {
        "path": str(
            Path(october["previous_manifest"]["path"]).parent / "admission_audit.json"
        ),
        "sha256": support["artifacts"]["admission_audit.json"]["sha256"],
    }
    dispositions = bound_json(quarantine_record)["field_quarantines"]
    source_receipts = {
        **old_inputs,
        "recovered_rates": rates_record,
        "recovered_admission": run["recovered_lending"]["admission"],
        "prior_field_dispositions": quarantine_record,
        "prior_float_inputs": peer["lending_source_bindings"],
        "prior_selected_denominators": selection,
        "identity": run["rename_issuer_propagation"],
        "history": run["rename_history_propagation"],
    }
    archive = {
        "schema": LENDING_ARCHIVE_SCHEMA,
        "status": "complete",
        "source_label": "reconciled_feature_and_economic_sources",
        "artifacts": {
            n: {"sha256": binding(economic / n)["sha256"], "bytes": (economic / n).stat().st_size}
            for n in ["lending_balances.parquet", "lending_rates.parquet"]
        },
        "source_receipts": source_receipts,
        "field_dispositions": dispositions,
        "availability_rule": "Original report/trade D+1; no spot-rename loan alias",
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    write_json_atomic(economic / "manifest.json", archive)
    result = {
        "schema": "LENDING_FEATURE_PROPAGATION_V1",
        "source_receipts": source_receipts,
        "old_feature_balance_rows": old_balances.height,
        "recovered_balance_rows": 230454,
        "combined_balance_rows": balances.height,
        "added_balance_rows": pl.read_parquet(
            output / "added_balance_observations.parquet"
        ).height,
        "old_only_balance_rows_retained": 13501,
        "original_feature_balances_exact": True,
        "july_monetary_fields_unknown": 267,
        "gmat_unresolved_quantity_rows": 1,
        "source_dispositions": dispositions,
        "old_rate_rows": old_rates.height,
        "new_rate_rows": rates.height,
        "changed_old_rate_flow_keys": pl.read_parquet(
            output / "changed_rate_observations.parquet"
        ).height,
        "six_field_control_exact_active_rows": 184222,
        "source_only_changes": compare(
            known_ages(old.select(basic)), known_ages(source_only), active_keys
        ),
        "history_changes": compare(
            known_ages(source_only), known_ages(final.select(basic)), active_keys
        ),
        "final_changes_vs_fca": stats,
        "denominator_support_losses": 126,
        "denominator_loss_semantics": "ALOS FRE127027 snapshot2023-03-21 / known2023-05-19, 532365440 shares; May2 predecessor DISMES102to103 remains an uncertainty barrier, not proof of a split or a wrong printed denominator.",
        "borrow_changes_vs_recovered_economics": borrowing,
        "future_source_deletion_prefix_exact_through": str(cutoff),
        "actual_loader_cells": sum(x.size for x in actual),
        "actual_loader_mismatches": 0,
        "economic_archive": binding(economic / "manifest.json"),
        "artifacts": {
            p.stem: binding(p)
            for p in output.iterdir()
            if p.suffix in (".parquet", ".npz")
        },
        "source_code": {
            p.name: binding(p)
            for p in [
                output / "executed_reproducer.py",
                Path(__file__),
                Path(b3.__file__),
                Path(cvm.__file__),
            ]
        },
        "qualification_seconds": perf_counter() - started,
        "initial_attempts": [
            "First control compared inactive rows; second compared null and -1 unknown ages. The corrected six-field active control passed before source propagation.",
            "The full producer stopped after saving features because it assumed no utilization support loss. This qualification independently binds all126 losses and finishes only remaining checks.",
        ],
        "limits": [
            "Intermediate new feature coordinates, not a replacement store or old-policy input; refits required.",
            "Taker/donor rates remain modality-blended proxies; unknown source vintage/revision share persists.",
            "Public quantity is not executable locate capacity. All earlier 2% economic placeholder cells remain; no missing rate feature invented.",
            "Raw source quantities/prices and earlier accepted stores remain immutable. July ambiguous monetary amounts are not divided by100 or otherwise guessed.",
            "Remaining auxiliary/M1 and contractual wealth/labels need the separately accepted complete store.",
        ],
    }
    write_json_atomic(output / "manifest.json", result)
    print(
        json.dumps(
            {
                "stage": "complete",
                "seconds": result["qualification_seconds"],
                "gains": {n: stats[n]["gained"] for n in names},
                "losses": {n: stats[n]["lost"] for n in names},
                "borrowing": borrowing,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
