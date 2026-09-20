"""Original allocation-risk formulas on accepted, dated security histories."""

from dataclasses import replace
import json
from pathlib import Path
import pickle
import subprocess
from time import perf_counter

import numpy as np
import polars as pl

from brazil_rv.execution.portfolio_policy import PolicyData
from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.bova11 import load_bova11_series
from brazil_rv.v2.contract import HORIZONS
from brazil_rv.v2.corporate_replay import apply_corporate_replay, load_corporate_replay
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.hedge_beta import HEDGE_BETA_SCHEMA, rolling_hedge_beta
from brazil_rv.v2.lending_archive import load_lending_borrow_panels
from brazil_rv.v2.portfolio_inputs import causal_risk
from brazil_rv.v2.portfolio_training import load_data
from brazil_rv.v2.store import open_store_for_samples
from brazil_rv.v2.validate_pipeline import _evaluation_inputs

PROJECT = Path(__file__).resolve().parents[1]


def returns(prices, seen, hedge):
    y = np.full((len(prices) - 1, prices.shape[1]), np.nan)
    good = seen[1:] & seen[:-1] & (prices[1:] >= 0) & (prices[:-1] > 0)
    np.divide(prices[1:], prices[:-1], out=y, where=good)
    y -= 1
    x = np.full(len(hedge) - 1, np.nan)
    good = (
        np.isfinite(hedge[1:])
        & np.isfinite(hedge[:-1])
        & (hedge[1:] > 0)
        & (hedge[:-1] > 0)
    )
    np.divide(hedge[1:], hedge[:-1], out=x, where=good)
    return x - 1, y


def beta_row(x, y):
    usable = np.isfinite(x[:, None]) & np.isfinite(y)
    count = usable.sum(0)
    a, b = np.where(usable, x[:, None], 0), np.where(usable, y, 0)
    sa, sb = a.sum(0), b.sum(0)
    variance = (a**2).sum(0) - sa**2 / np.maximum(count, 1)
    covariance = (a * b).sum(0) - sa * sb / np.maximum(count, 1)
    valid = (count >= 40) & (variance > 0)
    output = np.full(y.shape[1], np.nan)
    output[valid] = np.clip(0.67 * covariance[valid] / variance[valid] + 0.33, -1, 3)
    return output, valid


def variance_row(x, y, beta, active):
    residual = y - x[:, None] * beta
    finite = np.isfinite(residual)
    count = finite.sum(0)
    clean = np.where(finite, residual, 0)
    variance = np.maximum(
        (clean**2).sum(0) - clean.sum(0) ** 2 / np.maximum(count, 1), 0
    ) / np.maximum(count - 1, 1)
    supported = (count >= 20) & active
    prior = float(np.median(variance[supported])) if supported.any() else 0.02**2
    diagonal = np.maximum((count * variance + 20 * prior) / (count + 20), 1e-8)
    observed = x[np.isfinite(x)]
    factor = (
        max(float(np.var(observed, ddof=1)), 1e-8) if len(observed) >= 20 else 0.015**2
    )
    return diagonal, factor, count


def main():
    tick = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    accepted = bound_json(run["matched_data_inputs"])
    plan = bound_json(run["stage_c_plan"])
    out = Path(run["stage_c_root"]) / "refit_economics"
    out.mkdir(exist_ok=False)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    old, old_binding = load_data(Path(plan["prior_root"]), "C6")
    root = Path(accepted["store"]["root"])
    manifest = bound_json(
        dict(
            path=str(root / "manifest.json"),
            sha256=accepted["store"]["manifest_sha256"],
        )
    )
    dates = np.load(root / "date_index.npy")
    names = np.load(root / "isin_index.npy").tolist()
    indices = old.inputs.session_indices
    np.testing.assert_array_equal(
        dates[indices], np.asarray(old.inputs.dates, dtype="datetime64[D]")
    )
    assert dates[indices[-1]] <= np.datetime64("2024-12-30")
    first = int(indices[0]) - 20
    bova_spec = json.loads(
        (PROJECT / "docs/v2_round5_economic_sources.json").read_text()
    )["bova11"]
    bova = load_bova11_series(
        Path(bova_spec["root"]),
        expected_manifest_sha256=bova_spec["manifest_sha256"],
        canonical_dates=dates.astype(object).tolist(),
    )
    np.testing.assert_array_equal(
        bova.close_by_session[indices], old.inputs.bova11_close
    )
    links = pl.read_parquet(
        root / manifest["tables"]["slow_history_links"]["path"]
    ).to_dicts()
    assert all(r["known_index"] <= r["effective_index"] for r in links)
    write_json_atomic(
        out / "plan.json",
        dict(
            accepted=run["matched_data_inputs"],
            original_policy_cache=old_binding,
            bova11=bova_spec,
            account=run["economic_account"],
            terms=run["stage_c_event_candidate_terms"],
            decision_rows=[first, int(indices[-1]) + 1],
            contrast="New accepted coordinates and all admitted dated market histories, with original allocation beta/variance formulas, support, shrinkage, clipping and20-session beta fallback. Old PolicyData and calibration remain immutable; new PolicyData is solely for fresh matched fits. No model/optimizer/selector changes.",
            history="Each t consumes61wealth closes through t-1 for60adjacent returns. Successor history follows only decreasing event times admitted by t; raw-beta/fallback state can inherit a known predecessor at effect. Public prebirth arrays stay unchanged. Portfolio risk is separate from model slow beta/sigma.",
            verification="Original formula checks on selected decision windows, exact original-cache controls on the same source window, future mutation and source-action versus settlement reconciliation. No old source/feature/book campaign repeats.",
            registration=binding(
                PROJECT / "research/preregistrations/v2_economic_data_scaling.md"
            ),
        ),
    )

    def array(key):
        return np.load(root / manifest["arrays"][key]["path"], mmap_mode="r")

    wealth, seen, active = (
        array("shareholder_wealth_close"),
        array("shareholder_wealth_valid"),
        array("active"),
    )

    def history(t):
        rows = np.arange(t - 61, t)
        route = np.broadcast_to(np.arange(len(names)), (len(rows), len(names))).copy()
        known = [r for r in links if r["effective_index"] <= t]
        for name in {r["successor_index"] for r in known}:
            for j, day in enumerate(rows):
                ancestor, ceiling = name, t + 1
                while True:
                    edges = [
                        r
                        for r in known
                        if r["successor_index"] == ancestor
                        and day < r["effective_index"] < ceiling
                    ]
                    if not edges:
                        break
                    edge = max(edges, key=lambda r: r["effective_index"])
                    ancestor, ceiling = (
                        edge["predecessor_index"],
                        edge["effective_index"],
                    )
                route[j, name] = ancestor
        return (
            wealth[rows[:, None], route].astype(float),
            seen[rows[:, None], route],
            bova.close_by_session[rows],
        )

    shape = (len(dates), len(names))
    raw, raw_valid = np.full(shape, np.nan), np.zeros(shape, bool)
    resolved, diagonal, support = (
        np.full(shape, np.nan),
        np.full(shape, np.nan),
        np.zeros(shape, np.int16),
    )
    factor = np.full(len(dates), np.nan)
    previous, ages = np.ones(len(names)), np.full(len(names), 21)
    selected = {int(indices[0]), int(indices[-1])}
    selected.update(
        r["effective_index"] + offset
        for r in links
        for offset in (-1, 0, 1, 20, 60)
        if first <= r["effective_index"] + offset <= indices[-1]
    )
    formula_checks = []
    for t in range(first, int(indices[-1]) + 1):
        prices, mask, hedge = history(t)
        x, y = returns(prices, mask, hedge)
        raw[t], raw_valid[t] = beta_row(x, y)
        for link in links:
            if t == link["effective_index"]:
                a, b = link["predecessor_index"], link["successor_index"]
                previous[b], ages[b] = previous[a], ages[a]
        ages += 1
        valid = raw_valid[t]
        previous[valid], ages[valid] = raw[t, valid], 0
        resolved[t] = np.where(ages <= 20, previous, 1.0)
        diagonal[t], factor[t], support[t] = variance_row(x, y, resolved[t], active[t])
        if t in selected:
            # The unused final close deliberately carries an arbitrary value;
            # original routines must exclude that current endpoint.
            p = np.concatenate([prices, np.full((1, len(names)), 123456.0)])
            s = np.concatenate([mask, np.ones((1, len(names)), bool)])
            h = np.r_[hedge, 987654.0]
            b, v = rolling_hedge_beta(p, s, h)
            np.testing.assert_array_equal(b[-1], raw[t])
            np.testing.assert_array_equal(v[-1], raw_valid[t])
            d, f, n = causal_risk(
                p,
                s,
                h,
                np.broadcast_to(resolved[t], p.shape),
                np.broadcast_to(active[t], p.shape),
            )
            np.testing.assert_array_equal(d[-1], diagonal[t])
            np.testing.assert_array_equal(f[-1], factor[t])
            np.testing.assert_array_equal(n[-1], support[t])
            formula_checks.append(
                dict(
                    date=str(dates[t]),
                    cells=4 * len(names) + 1,
                    current_endpoint_excluded=True,
                )
            )
    np.savez_compressed(
        out / "risk.npz",
        raw_beta=raw,
        raw_valid=raw_valid,
        resolved_beta=resolved,
        diagonal=diagonal,
        factor=factor,
        support=support,
    )
    write_json_atomic(out / "formula_checks.json", formula_checks)
    # Original-cache control is bounded to its first61 available wealth closes;
    # earlier repaired corporate events do not intersect this2016 decision.
    original_store = Path(
        json.loads((Path(plan["prior_root"]) / "frozen_design.json").read_text())[
            "store"
        ]["root"]
    )
    t = int(indices[0])
    old_p = np.load(original_store / "shareholder_wealth_close.npy", mmap_mode="r")[
        t - 61 : t + 1
    ]
    old_s = np.load(original_store / "shareholder_wealth_valid.npy", mmap_mode="r")[
        t - 61 : t + 1
    ]
    b, v = rolling_hedge_beta(old_p, old_s, bova.close_by_session[t - 61 : t + 1])
    np.testing.assert_array_equal(b[-1], old.inputs.hedge_beta[0])
    np.testing.assert_array_equal(v[-1], old.inputs.hedge_beta_valid[0])
    d, f, _ = causal_risk(
        old_p,
        old_s,
        bova.close_by_session[t - 61 : t + 1],
        np.broadcast_to(old.beta[0], old_p.shape),
        np.broadcast_to(old.inputs.active[0], old_p.shape),
    )
    np.testing.assert_array_equal(d[-1], old.diagonal[0])
    np.testing.assert_array_equal(f[-1], old.factor[0])
    hedge_root = out / "hedge_beta"
    hedge_root.mkdir()
    arrays = {}
    for key, values in (
        ("hedge_beta", raw),
        ("hedge_beta_valid", raw_valid),
        ("date_index", dates),
        ("isin_index", np.asarray(names)),
    ):
        path = hedge_root / f"{key}.npy"
        np.save(path, values)
        arrays[key] = dict(
            path=path.name, bytes=path.stat().st_size, sha256=binding(path)["sha256"]
        )
    write_json_atomic(
        hedge_root / "manifest.json",
        dict(
            schema=HEDGE_BETA_SCHEMA,
            status="complete",
            store=accepted["store"],
            bova11={**bova_spec, "data_sha256": bova.data_sha256},
            arrays=arrays,
            contract=bound_json(binding(out / "plan.json")),
            implementation=binding(Path(__file__)),
        ),
    )
    cdi_source = bound_json(run["cash_calendar"])["panel"]
    with np.load(cdi_source["path"]) as z:
        cdi = z["cdi_returns"].copy()
    assert np.isfinite(cdi[indices]).all()
    lending = load_lending_borrow_panels(
        Path(run["qualified_lending"]["root"]),
        expected_manifest_sha256=run["qualified_lending"]["manifest_sha256"],
        canonical_dates=dates.astype(object).tolist(),
        canonical_isins=names,
    )
    store, access = open_store_for_samples(
        root,
        indices,
        purpose="evaluation",
        verify_hashes=True,
        history_lookbacks=61,
        history_end_offsets=0,
    )
    try:
        inputs = _evaluation_inputs(
            store,
            indices,
            np.zeros((*old.inputs.active.shape, len(HORIZONS)), np.float32),
            np.zeros((*old.inputs.active.shape, len(HORIZONS)), bool),
            cdi,
            bova.close_by_session,
            {
                **bova_spec,
                "data_sha256": bova.data_sha256,
                "hedge_beta_root": str(hedge_root),
                "hedge_beta_manifest_sha256": binding(hedge_root / "manifest.json")[
                    "sha256"
                ],
            },
            lending,
            {"matched_data": run["matched_data_inputs"]["sha256"]},
            transfer_chronology_clean=True,
            execution_policy=old.inputs.execution_policy,
        )
        refs = store.read("prior_reference_close", indices)
    finally:
        store.close()
    terms, calendar = load_corporate_replay(
        run["stage_c_event_candidate_terms"]["path"],
        run["stage_c_event_candidate_terms"]["sha256"],
    )
    cleared = []
    has = inputs.action_has_action.copy()
    successor = inputs.action_successor_index.copy()
    for event in terms["share_distributions"]:
        t = int(
            np.searchsorted(calendar, np.datetime64(event["effective_date"]))
        ) - int(indices[0])
        n = names.index(event["isin"])
        if 0 <= t < len(indices) and has[t, n]:
            assert len(event["legs"]) == 1 and event["cash_per_prior_share"] == 0
            leg = event["legs"][0]
            assert (
                inputs.action_cash_per_prior_share[t, n] == 0
                and inputs.action_shares_per_prior_share[t, n]
                == leg["shares_per_prior_share"]
                == 1
            )
            assert successor[t, n] == names.index(leg["successor_isin"])
            has[t, n] = False
            successor[t, n] = -1
            cleared.append(
                dict(
                    date=event["effective_date"],
                    isin=event["isin"],
                    reason="same gross source action represented once by explicit custody/loan distribution",
                )
            )
    inputs = replace(inputs, action_has_action=has, action_successor_index=successor)
    inputs = apply_corporate_replay(
        inputs, terms, calendar, run["stage_c_event_candidate_terms"]["sha256"]
    )
    source = bound_json(run["loan_source_panels"])["panels"]
    with np.load(source["path"]) as z:
        inputs = replace(
            inputs,
            loan_reference_prices=z["loan_reference_prices"][indices],
            hedge_annual_borrow_rate=z["hedge_annual_borrow_rate"][indices],
        )
    data = PolicyData(
        inputs,
        resolved[indices],
        diagonal[indices],
        factor[indices],
        cdi[indices - 1],
        refs,
    )
    cache = out / "policy_data.pkl"
    with cache.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    compressed = subprocess.run(
        ["compact.exe", "/C", "/F", "/EXE:LZX", str(cache)],
        capture_output=True,
        text=True,
        check=True,
    )
    (out / "storage.txt").write_text(compressed.stdout)
    report = dict(
        status="prepared_new_refit_policy_inputs_pending_saved_qualification",
        plan=binding(out / "plan.json"),
        cache={**binding(cache), "bytes": cache.stat().st_size},
        risk=binding(out / "risk.npz"),
        formula_checks=binding(out / "formula_checks.json"),
        old_control_cells=3 * len(names) + 1,
        source_actions_represented_once=cleared,
        access=access.payload(),
        store=accepted["store"],
        names=len(names),
        dates=len(indices),
        seconds=perf_counter() - tick,
    )
    write_json_atomic(out / "manifest.json", report)
    run = json.loads(pointer.read_text())
    run["stage_c_refit_economics"] = binding(out / "manifest.json")
    write_json_atomic(pointer, run)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
