"""Resume the five source-bound market wealth chains with dated JSL episodes."""

from dataclasses import replace
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
from propagate_surviving_wealth import resume

PROJECT = Path(__file__).resolve().parents[1]


def main():
    tick = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    admission = bound_json(run["stage_c_event_data_admission"])
    plan = bound_json(admission["plan"])
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
    (out / "recurrence.py").write_bytes(
        (PROJECT / "ops/propagate_surviving_wealth.py").read_bytes()
    )
    dates, names = (
        np.load(root / "date_index.npy"),
        np.load(root / "isin_index.npy").tolist(),
    )
    events = plan["events"]
    selected_names = sorted({e[k] for e in events for k in ("isin", "successor_isin")})
    axes = [names.index(s) for s in selected_names]
    start = int(np.searchsorted(dates, np.datetime64(events[0]["effective_date"]))) - 1
    days = dates[start:]
    quotes = (
        pl.read_parquet(admission["normalized_quotes"]["path"])
        .filter(pl.col("trade_date") >= days[0].astype(object))
        .select("trade_date", "isin", "open_brl", "high_brl", "low_brl", "close_brl")
        .sort("trade_date", "isin")
    )
    assert not quotes.select(
        pl.struct("trade_date", "isin").is_duplicated().any()
    ).item()
    quotes.write_parquet(out / "quotes.parquet")
    fields = ("open", "high", "low", "close")
    raw = np.full((4, len(days), len(axes)), np.nan, np.float64)
    ti, ni = (
        np.searchsorted(days, quotes["trade_date"].to_numpy()),
        np.array([selected_names.index(s) for s in quotes["isin"]]),
    )
    for k, field in enumerate(fields):
        raw[k, ti, ni] = quotes[field + "_brl"].to_numpy()

    def old(key):
        return np.load(root / m["arrays"][key]["path"], mmap_mode="r")

    observed = old("observed")[start:][:, axes]
    assert np.isfinite(raw).all(axis=0)[observed].all()
    for k, field in enumerate(fields):
        np.testing.assert_array_equal(
            raw[k].astype(np.float32)[observed],
            old("raw_" + field)[start:][:, axes][observed],
        )
    terms = verified_action_terms_from_table(
        pl.read_parquet(root / m["tables"]["corporate_actions_verified_terms"]["path"])
    )
    amended_terms = tuple(
        replace(t, shares_per_prior_share=1.0, cash_per_prior_share=0.0)
        if t.isin == "BRJSLGACNOR2" and str(t.effective_date) == "2020-11-11"
        else t
        for t in terms
    )
    changed = [(a, b) for a, b in zip(terms, amended_terms, strict=True) if a != b]
    assert len(changed) == 1 and changed[0][0].shares_per_prior_share != 1
    schedule = load_session_schedule(Path(plan["schedule"]["path"]))
    selected = tuple(
        s for s in schedule if days[0] <= np.datetime64(s.trade_date) <= days[-1]
    )
    following = next(
        s.decision_at for s in schedule if np.datetime64(s.trade_date) > days[-1]
    )
    clocks = next_session_decision_cutoffs(selected, following_decision_at=following)
    links = pl.read_parquet(admission["links"]["path"]).filter(
        pl.col("predecessor_isin").is_in(selected_names)
    )
    old_links = pl.read_parquet(
        root / m["tables"]["isin_succession_links"]["path"]
    ).filter(pl.col("predecessor_isin").is_in(selected_names))

    def align(scalars, table):
        return align_decision_known_action_terms(
            (*scalars, *verified_conversion_terms_from_links(table)),
            days,
            selected_names,
            coverage_resolved=observed,
            decision_timestamps=clocks,
        )

    original, corrected = align(terms, old_links), align(amended_terms, links)
    write_json_atomic(
        out / "plan.json",
        dict(
            parent=admission["parent"],
            admission=run["stage_c_event_data_admission"],
            names=selected_names,
            start=str(days[0]),
            stop=str(days[-1]),
            scalar_replacement=dict(
                isin="BRJSLGACNOR2",
                date="2020-11-11",
                before=changed[0][0].shares_per_prior_share,
                after=1,
            ),
            contract="Original Float32 recurrence and exact normalized Float64 quotes. Standalone successor controls once, predecessor prior-close seed, no quoted observations or issuer histories invented. Old holding JSL public continuation ends before logistics reopening; the logistics chain then owns the reused coordinate. Retain all earlier public coordinates; internal episode history is separate. Final actual tail controls downstream scope.",
        ),
    )
    packed, bases, reports = {}, {}, []
    for e in events:
        a, b = (
            selected_names.index(e["isin"]),
            selected_names.index(e["successor_isin"]),
        )
        first = int(np.searchsorted(days, np.datetime64(e["effective_date"])))
        seed = np.array(
            [old("shareholder_wealth_" + k)[start + first - 1, axes[a]] for k in fields]
        )
        seed_valid = bool(old("shareholder_wealth_valid")[start + first - 1, axes[a]])
        successor_seed = np.array(
            [old("shareholder_wealth_" + k)[start + first - 1, axes[b]] for k in fields]
        )
        successor_valid = bool(
            old("shareholder_wealth_valid")[start + first - 1, axes[b]]
        )
        control, cv = resume(
            raw, observed, original, first, b, successor_seed, successor_valid
        )
        values, valid = resume(raw, observed, corrected, first, a, seed, seed_valid)
        np.savez_compressed(
            out / (e["id"] + ".npz"),
            control=control,
            control_valid=cv,
            corrected=values,
            corrected_valid=valid,
            seed=seed,
            seed_valid=seed_valid,
        )
        expected = np.array(
            [old("shareholder_wealth_" + k)[start:, axes[b]] for k in fields]
        )
        np.testing.assert_array_equal(control[:, first:], expected[:, first:])
        np.testing.assert_array_equal(
            cv[first:], old("shareholder_wealth_valid")[start + first :, axes[b]]
        )
        end_source = (
            len(days)
            if not e["source_reopens_date"]
            else int(np.searchsorted(days, np.datetime64(e["source_reopens_date"])))
        )
        for axis, stop in ((axes[a], end_source), (axes[b], len(days))):
            for k, field in enumerate((*fields, "valid")):
                key = "shareholder_wealth_" + field
                before = old(key)[start + first : start + stop, axis]
                after = valid[first:stop] if field == "valid" else values[k, first:stop]
                ix = np.flatnonzero(
                    ~((before == after) | (np.isnan(before) & np.isnan(after)))
                )
                packed.setdefault(key + "__indices", []).append(
                    np.column_stack((ix + start + first, np.full(len(ix), axis)))
                )
                packed.setdefault(key + "__values", []).append(after[ix])
        bases[e["id"] + "__wealth"], bases[e["id"] + "__valid"] = values, valid
        differing = (values[:, first:] != expected[:, first:]).any(0) | (
            valid[first:] != cv[first:]
        )
        changed_rows = np.flatnonzero(differing)
        reports.append(
            dict(
                event=e["id"],
                control_cells=int(control[:, first:].size + cv[first:].size),
                changed_successor_days=len(changed_rows),
                last_changed=None
                if not len(changed_rows)
                else str(days[first + changed_rows[-1]]),
                source_episode_stop=None
                if end_source == len(days)
                else str(days[end_source]),
                seed=seed.tolist(),
                seed_valid=seed_valid,
            )
        )
    np.savez_compressed(
        out / "deltas.npz", **{k: np.concatenate(v) for k, v in packed.items()}
    )
    np.savez_compressed(out / "chain_basis.npz", dates=days, axes=axes, **bases)
    report = dict(
        status="wealth_chains_qualified_dependencies_pending",
        plan=binding(out / "plan.json"),
        parent=admission["parent"],
        cases=reports,
        deltas=binding(out / "deltas.npz"),
        basis=binding(out / "chain_basis.npz"),
        quotes=binding(out / "quotes.parquet"),
        seconds=perf_counter() - tick,
    )
    write_json_atomic(out / "manifest.json", report)
    run["stage_c_event_wealth"] = binding(out / "manifest.json")
    write_json_atomic(pointer, run)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
