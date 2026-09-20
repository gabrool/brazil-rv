"""Resume only the two newly admitted shareholder identity chains."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.corporate_actions import (
    align_decision_known_action_terms,
    verified_action_terms_from_table,
    verified_conversion_terms_from_links,
)
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.decision_clock import (
    load_session_schedule,
    next_session_decision_cutoffs,
)

PROJECT = Path(__file__).resolve().parents[1]


def resume(raw, observed, actions, first, claim, seed, seed_valid):
    """The existing Float32 wealth recurrence, seeded at the prior close."""
    result = np.zeros((4, len(observed)), np.float32)
    valid = np.zeros(len(observed), bool)
    result[:, first - 1] = seed
    valid[first - 1] = seed_valid
    for day in range(first, len(observed)):
        prior = claim
        if not actions.session_resolved[day, prior]:
            continue
        claim = int(actions.successor_index[day, prior])
        if not observed[day, claim]:
            continue
        q, cash = (
            actions.shares_per_prior_share[day, prior],
            actions.cash_per_prior_share[day, prior],
        )
        if not valid[day - 1]:
            if q != 1 or cash != 0 or claim != prior:
                continue
            result[:, day] = raw[:, day, claim]
        else:
            scale = result[3, day - 1] / raw[3, day - 1, prior]
            candidates = scale * (q * raw[:, day, claim] + cash)
            if not (np.isfinite(candidates) & (candidates > 0)).all():
                continue
            result[:, day] = candidates
        valid[day] = True
    return result, valid


def main():
    tick = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text(encoding="utf8"))
    admission = bound_json(run["surviving_rename_admission"])
    source = Path(admission["plan"]["path"]).parent
    root = Path(admission["parent"]["root"])
    m = bound_json(
        dict(
            path=str(root / "manifest.json"),
            sha256=admission["parent"]["manifest_sha256"],
        )
    )
    out = source / "wealth"
    out.mkdir(exist_ok=False)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    links = pl.read_parquet(admission["links"]["path"])
    old_links = pl.read_parquet(root / m["tables"]["isin_succession_links"]["path"])
    dates, isins = (
        np.load(root / "date_index.npy"),
        np.load(root / "isin_index.npy").tolist(),
    )
    selected_isins = [
        "BRSSBRACNOR1",
        "BRALSOACNOR5",
        "BRALOSACNOR5",
        "BRARZZACNOR3",
        "BRAZZAACNOR9",
    ]
    axes = [isins.index(s) for s in selected_isins]
    start = int(np.searchsorted(dates, np.datetime64("2019-08-05")))
    days = dates[start:]
    write_json_atomic(
        out / "plan.json",
        {
            "parent": admission["parent"],
            "admission": run["surviving_rename_admission"],
            "source_first": str(days[0]),
            "source_last": str(days[-1]),
            "isins": selected_isins,
            "contract": "Resume original Float32 shareholder wealth from last accepted source close, with exact normalized Float64 quotes and decision-known action terms. Verify unchanged standalone successor recurrences; preserve outward prebirth coordinates. Changed recurrence tail, if any, determines dependency scope rather than an arbitrary 252-session truncation.",
            "hypotheses": "Spot identity only. No loan aliases or acquired-company history; no basket OHLC. No accepted store overwrite.",
        },
    )
    records = [
        r
        for r in m["sources"]
        if Path(r.get("path", "")).name
        in [f"equities_daily_{y}.parquet" for y in range(2019, 2025)]
    ]
    quotes = pl.concat(
        [
            pl.scan_parquet(r["path"])
            .filter(
                pl.col("isin").is_in(selected_isins)
                & (pl.col("market_type") == 10)
                & (pl.col("trade_date") >= days[0].astype(object))
            )
            .select(
                "trade_date", "isin", "open_brl", "high_brl", "low_brl", "close_brl"
            )
            .collect()
            for r in records
        ]
    ).sort("trade_date", "isin")
    assert not quotes.select(
        pl.struct("trade_date", "isin").is_duplicated().any()
    ).item()
    quotes.write_parquet(out / "quotes.parquet")
    raw = np.full((4, len(days), len(axes)), np.nan, np.float64)
    ti = np.searchsorted(days, quotes["trade_date"].to_numpy())
    ni = np.array([selected_isins.index(x) for x in quotes["isin"]])
    for k, name in enumerate(("open", "high", "low", "close")):
        raw[k, ti, ni] = quotes[name + "_brl"].to_numpy()

    def old(key):
        return np.load(root / m["arrays"][key]["path"], mmap_mode="r")

    observed = old("observed")[start:][:, axes]
    assert (np.isfinite(raw).all(axis=0)[observed]).all()
    for k, name in enumerate(("open", "high", "low", "close")):
        np.testing.assert_array_equal(
            raw[k].astype(np.float32)[observed],
            old("raw_" + name)[start:][:, axes][observed],
        )
    terms = verified_action_terms_from_table(
        pl.read_parquet(root / m["tables"]["corporate_actions_verified_terms"]["path"])
    )
    schedule = load_session_schedule(Path(admission["schedule"]["path"]))
    selected_schedule = tuple(
        s for s in schedule if days[0] <= np.datetime64(s.trade_date) <= days[-1]
    )
    following = next(
        s.decision_at for s in schedule if np.datetime64(s.trade_date) > days[-1]
    )
    cutoffs = next_session_decision_cutoffs(
        selected_schedule, following_decision_at=following
    )

    def actions(table):
        table = table.filter(pl.col("predecessor_isin").is_in(selected_isins))
        return align_decision_known_action_terms(
            (*terms, *verified_conversion_terms_from_links(table)),
            days,
            selected_isins,
            coverage_resolved=observed,
            decision_timestamps=cutoffs,
        )

    original, corrected = actions(old_links), actions(links)
    packed, basis, reports = {}, {}, []
    fields = ("open", "high", "low", "close")
    for predecessor, successor, effect in (
        ("BRSSBRACNOR1", "BRALSOACNOR5", "2019-08-06"),
        ("BRARZZACNOR3", "BRAZZAACNOR9", "2024-08-01"),
    ):
        a, b = selected_isins.index(predecessor), selected_isins.index(successor)
        first = int(np.searchsorted(days, np.datetime64(effect)))
        seed = np.array(
            [old("shareholder_wealth_" + k)[start + first - 1, axes[a]] for k in fields]
        )
        seed_valid = bool(old("shareholder_wealth_valid")[start + first - 1, axes[a]])
        # Exact standalone control uses no prior successor observation.
        control, cv = resume(
            raw, observed, original, first, b, np.zeros(4, np.float32), False
        )
        expected = np.array(
            [old("shareholder_wealth_" + k)[start:, axes[b]] for k in fields]
        )
        np.testing.assert_array_equal(control[:, first:], expected[:, first:])
        np.testing.assert_array_equal(
            cv[first:], old("shareholder_wealth_valid")[start + first :, axes[b]]
        )
        values, valid = resume(raw, observed, corrected, first, a, seed, seed_valid)
        # Scalar closed-form first transition from the prior source coordinate.
        for k in range(4):
            expected_first = np.float32(
                float(seed[3]) / raw[3, first - 1, a] * raw[k, first, b]
            )
            assert values[k, first] == expected_first
        # Continuation root is carried through the already-admitted ALSO->ALOS.
        destinations = [(axes[a], first), (axes[b], first)]
        if successor == "BRALSOACNOR5":
            c = isins.index("BRALOSACNOR5")
            destinations.append(
                (c, int(np.searchsorted(days, np.datetime64("2023-10-25"))))
            )
        for axis, begin in destinations:
            for k, name in enumerate((*fields, "valid")):
                key = "shareholder_wealth_" + name
                after = valid[begin:] if name == "valid" else values[k, begin:]
                before = old(key)[start + begin :, axis]
                changed = np.flatnonzero(
                    ~((before == after) | (np.isnan(before) & np.isnan(after)))
                )
                packed.setdefault(key + "__indices", []).append(
                    np.column_stack(
                        (changed + start + begin, np.full(len(changed), axis))
                    )
                )
                packed.setdefault(key + "__values", []).append(after[changed])
        # Internal prebirth successor history is separately routed at decision;
        # never write it into an earlier public successor store coordinate.
        basis[predecessor + "__wealth"] = values
        basis[predecessor + "__valid"] = valid
        unequal = (values[:, first:] != expected[:, first:]).any(axis=0) | (
            valid[first:] != cv[first:]
        )
        last = np.flatnonzero(unequal)
        reports.append(
            {
                "predecessor": predecessor,
                "successor": successor,
                "first": str(days[first]),
                "control_cells": int(control[:, first:].size + cv[first:].size),
                "last_successor_difference": str(days[first + last[-1]])
                if len(last)
                else None,
                "different_successor_days": int(unequal.sum()),
                "seed": seed.tolist(),
                "seed_valid": seed_valid,
            }
        )
    np.savez_compressed(
        out / "deltas.npz", **{k: np.concatenate(v) for k, v in packed.items()}
    )
    np.savez_compressed(out / "chain_basis.npz", dates=days, axes=axes, **basis)
    report = {
        "status": "wealth_recursions_qualified_dependencies_pending",
        "plan": binding(out / "plan.json"),
        "admission": run["surviving_rename_admission"],
        "parent": admission["parent"],
        "deltas": binding(out / "deltas.npz"),
        "basis": binding(out / "chain_basis.npz"),
        "quotes": binding(out / "quotes.parquet"),
        "quote_sources": records,
        "rows": quotes.height,
        "cases": reports,
        "seconds": perf_counter() - tick,
    }
    write_json_atomic(out / "manifest.json", report)
    run["surviving_rename_wealth"] = binding(out / "manifest.json")
    write_json_atomic(pointer, run)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
