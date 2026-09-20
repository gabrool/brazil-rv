"""Resume the unchanged Float32 wealth recurrence at the last common close."""

from datetime import date
from decimal import Decimal
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.corporate_actions import (
    align_decision_known_action_terms,
    verified_action_terms_from_table,
)
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.decision_clock import (
    load_session_schedule,
    next_session_decision_cutoffs,
)

PROJECT = Path(__file__).resolve().parents[1]
FIELDS = ("open", "high", "low", "close")


def main():
    started = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    attribution = bound_json(run["natura_gross_target_attribution"])
    source_audit = bound_json(run["held_event_source_audit"])
    qualification = bound_json(run["held_event_source_qualification"])
    source = attribution["parent_store"]
    store = Path(source["root"])
    manifest = bound_json(
        {"path": str(store / "manifest.json"), "sha256": source["manifest_sha256"]}
    )
    out = Path(run["root"]) / "natura_wealth"
    out.mkdir(exist_ok=False)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    dates, isins = (
        np.load(store / "date_index.npy"),
        np.load(store / "isin_index.npy").tolist(),
    )
    axis = isins.index("BRNATUACNOR6")
    rows = np.flatnonzero(
        (dates >= np.datetime64("2019-09-17")) & (dates <= np.datetime64("2019-12-17"))
    )
    ds = dates[rows]
    quote_rec = source_audit["quote_sources_existing_receipts"]["2019"]
    quotes = (
        pl.scan_parquet(quote_rec["path"])
        .filter(
            (pl.col("isin") == isins[axis])
            & pl.col("trade_date").is_between(date(2019, 9, 17), date(2019, 12, 17))
        )
        .collect()
        .sort("trade_date")
    )
    np.testing.assert_array_equal(quotes["trade_date"].to_numpy(), ds)
    assert (quotes["quote_factor"] == 1).all()
    quotes.write_parquet(out / "source_quotes.parquet")
    raw = [quotes[k + "_brl"].to_numpy() for k in FIELDS]

    def old(key):
        return np.load(store / manifest["arrays"][key]["path"], mmap_mode="r")

    for k, values in zip(FIELDS, raw, strict=True):
        np.testing.assert_array_equal(
            values.astype(np.float32), old("raw_" + k)[rows, axis]
        )
    assert (
        old("observed")[rows, axis].all()
        and old("shareholder_wealth_valid")[rows, axis].all()
    )
    original = verified_action_terms_from_table(
        pl.read_parquet(attribution["original_terms"]["path"]).filter(
            pl.col("isin") == isins[axis]
        )
    )
    replacement = verified_action_terms_from_table(
        pl.read_parquet(attribution["replacement_terms"]["path"]).filter(
            pl.col("isin") == isins[axis]
        )
    )
    schedule = load_session_schedule(Path(qualification["schedule"]["path"]))
    selected = tuple(
        s for s in schedule if ds[0] <= np.datetime64(s.trade_date) <= ds[-1]
    )
    following = next(
        s.decision_at for s in schedule if np.datetime64(s.trade_date) > ds[-1]
    )
    cutoffs = next_session_decision_cutoffs(selected, following_decision_at=following)
    coverage = old("action_session_resolved")[rows, axis : axis + 1]

    def align(terms):
        return align_decision_known_action_terms(
            terms,
            ds,
            [isins[axis]],
            coverage_resolved=coverage,
            decision_timestamps=cutoffs,
        )

    controls, corrected = align(original), align(replacement)
    assert controls.session_resolved.all() and corrected.session_resolved.all()
    seed = np.array(
        [old("shareholder_wealth_" + k)[rows[0], axis] for k in FIELDS], np.float32
    )

    def resume(actions):
        result = np.empty((len(rows), 4), np.float32)
        result[0] = seed
        for t in range(1, len(rows)):
            scale = result[t - 1, 3] / raw[3][t - 1]
            for k in range(4):
                result[t, k] = scale * (
                    actions.shares_per_prior_share[t, 0] * raw[k][t]
                    + actions.cash_per_prior_share[t, 0]
                )
        return result

    control, values = resume(controls), resume(corrected)
    for k, field in enumerate(FIELDS):
        np.testing.assert_array_equal(
            control[:, k], old("shareholder_wealth_" + field)[rows, axis]
        )
    delta, effects, oracles = {}, {}, []
    for k, field in enumerate(FIELDS):
        changed = np.flatnonzero(values[:, k] != control[:, k])
        delta["shareholder_wealth_" + field + "__indices"] = np.column_stack(
            (rows[changed], np.full(len(changed), axis))
        )
        delta["shareholder_wealth_" + field + "__values"] = values[changed, k]
        effects[field] = len(changed)
    for term in verified_action_terms_from_table(
        pl.read_parquet(qualification["amendments"]["path"])
    ):
        t = int(np.searchsorted(ds, np.datetime64(term.ex_date)))
        factor = (
            Decimal(str(term.shares_per_prior_share)) * Decimal(str(raw[3][t]))
            + Decimal(str(term.cash_per_prior_share))
        ) / Decimal(str(raw[3][t - 1]))
        expected_close = float(Decimal(str(float(values[t - 1, 3]))) * factor)
        np.testing.assert_allclose(values[t, 3], expected_close, rtol=6e-8, atol=0)
        oracles.append(
            {
                "date": str(term.ex_date),
                "q": term.shares_per_prior_share,
                "gross_cash": term.cash_per_prior_share,
                "gross_close_factor": str(factor),
                "old_wealth_close": float(control[t, 3]),
                "new_wealth_close": float(values[t, 3]),
            }
        )
        # Reverting a later term must preserve the entire earlier recurrence.
        terms = tuple(t0 for t0 in replacement if t0.ex_date != term.ex_date) + tuple(
            t0 for t0 in original if t0.ex_date == term.ex_date
        )
        deleted = resume(align(terms))
        np.testing.assert_array_equal(values[:t], deleted[:t])
    np.savez_compressed(out / "deltas.npz", **delta)
    np.save(out / "wealth_rows.npy", values)
    np.save(out / "control_rows.npy", control)
    np.save(out / "rows.npy", rows)
    report = {
        "status": "passed_bounded_wealth_amendment_not_final_features_or_store",
        "parent_store": source,
        "source_qualification": run["held_event_source_qualification"],
        "gross_target_attribution": run["natura_gross_target_attribution"],
        "annual_quote_receipt_reused": quote_rec,
        "source_quotes": binding(out / "source_quotes.parquet"),
        "rows": binding(out / "rows.npy"),
        "deltas": binding(out / "deltas.npz"),
        "quote_rows": quotes.height,
        "control_cells": int(control.size),
        "effects": effects,
        "event_decimal_checks": oracles,
        "seed": {
            "date": str(ds[0]),
            "values": seed.tolist(),
            "contract": "Last shared pre-effect wealth close; no earlier recurrence, source census or unrelated family is rebuilt.",
        },
        "causality": "Original dated following-decision cutoffs and two future-term deletion prefixes pass. The sparse patch changes only this one already observed identity; all raw quotes and masks remain unchanged.",
        "wealth_contract": "Unchanged Float32 daily total-return recurrence used by volatility features. Cash enters q*OHLC+d; unit horizon labels separately retain unremunerated/unreinvested cash. No source quote, bonus custody or payment is synthesized.",
        "source_exit": "NATU source quotes end Dec17; no NTCO history, loan alias or successor mark is introduced. Newly sourced Dec18 succession is a separate required amendment.",
        "remaining": "Propagate corrected wealth and source terms through actual daily/native/auxiliary dependencies, final sigma/virtual neutral targets and accounting. Existing accepted store remains unchanged.",
        "seconds": perf_counter() - started,
    }
    write_json_atomic(out / "manifest.json", report)
    run["natura_wealth_amendment"] = binding(out / "manifest.json")
    write_json_atomic(pointer, run)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
