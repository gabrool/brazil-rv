"""Propagate admitted same-class renames through full daily history and targets.

Writes an explicit intermediate array contract. It never replaces the sealed
store or attaches changed coordinates to a saved fit. Auxiliary and M1-derived
dependencies must be assembled separately before accepting a replacement store.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.build_store import (
    _common_state_diagnostic_panel,
    _route_decision_known_continuations,
)
from brazil_rv.v2.contract import DEVELOPMENT_END, SLOW_FEATURES
from brazil_rv.v2.corporate_actions import (
    align_decision_known_action_terms,
    align_verified_action_terms,
    build_shareholder_wealth_ohlc_into,
    infer_cotahist_action_terms,
    verified_action_terms_from_table,
    verified_conversion_terms_from_links,
)
from brazil_rv.v2.data_foundation import (
    load_isin_link_allowlist,
    panel_from_daily,
    prepare_cash_equities,
    slow_history_links,
)
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.decision_clock import (
    load_session_schedule,
    next_session_decision_cutoffs,
)
from brazil_rv.v2.feature_spec import (
    FeatureSpec,
    observation_age_sessions_into,
    transform_feature_panel_into,
)
from brazil_rv.v2.features import build_slow_features_into
from brazil_rv.v2.targets import build_economic_multi_day_targets_into
from brazil_rv.v2.store import close_memmap
from brazil_rv.v2.universe import build_daily_universe

PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finalize-existing", action="store_true")
    args = parser.parse_args()
    started = perf_counter()
    pointer = json.loads(
        (PROJECT / "docs/v2_economic_data_scaling_run.json").read_text()
    )
    parent = json.loads((PROJECT / "docs/v2_data_inputs.json").read_text())["store"]
    source = Path(parent["root"])
    manifest = bound_json(
        {"path": str(source / "manifest.json"), "sha256": parent["manifest_sha256"]}
    )
    output = Path(pointer["root"]) / "rename_history_propagation"
    if args.finalize_existing:
        finish(output, source, manifest, parent, started)
        return
    output.mkdir(exist_ok=False)
    (output / "executed_reproducer.py").write_bytes(Path(__file__).read_bytes())
    dates = np.load(source / "date_index.npy")
    isins = tuple(np.load(source / "isin_index.npy").tolist())
    assert dates[-1] <= np.datetime64(DEVELOPMENT_END)

    def old(name):
        return np.load(source / manifest["arrays"][name]["path"], mmap_mode="r")

    records = [
        r
        for r in manifest["sources"]
        if Path(r.get("path", "")).name.startswith("equities_daily_")
    ]
    # These normalized immutable sources already passed the original quote
    # census; this is the necessary new transformation, not another ASCII audit.
    daily = pl.concat(
        [pl.read_parquet(r["path"]) for r in records], how="diagonal_relaxed"
    ).filter(pl.col("trade_date") <= DEVELOPMENT_END)
    validation = prepare_cash_equities(
        daily, v1_isins=isins, require_units=True, maximum_rejection_fraction=0.005
    )
    schedule_record = next(
        r
        for r in manifest["sources"]
        if Path(r.get("path", "")).name == "b3_session_schedule_reconstructed_v1.csv"
    )
    full_schedule = load_session_schedule(Path(schedule_record["path"]))
    start = validation.accepted["trade_date"].min()
    schedule = tuple(
        s for s in full_schedule if start <= s.trade_date <= DEVELOPMENT_END
    )
    following = next(
        s.decision_at for s in full_schedule if s.trade_date > DEVELOPMENT_END
    )
    calendar = [s.trade_date for s in schedule]
    observed_dates = set(validation.accepted["trade_date"])
    panel = panel_from_daily(
        validation.accepted,
        dates=calendar,
        isins=isins,
        source_session_complete=np.array([d in observed_dates for d in calendar]),
        invalid_observations=validation.rejected,
    )
    kept = np.searchsorted(panel.dates, dates)
    np.testing.assert_array_equal(panel.dates[kept], dates)
    activity_changes = (panel.activity_valid[kept] != old("activity_valid")) | (
        panel.trade_observed[kept] != old("trade_observed")
    )
    # Preserve the sealed activity contract for rename-only attribution. The
    # separately recorded invalid-OHLC GOLL row needs its own source disposition;
    # it must not silently remove twenty unrelated eligible sessions here.
    activity_rows, activity_names = np.nonzero(activity_changes)
    activity_evidence = [
        {
            "date": str(dates[t]),
            "isin": isins[j],
            "reconstructed_valid": bool(panel.activity_valid[kept[t], j]),
            "sealed_valid": bool(old("activity_valid")[t, j]),
        }
        for t, j in zip(activity_rows, activity_names, strict=True)
    ]
    for t, j in zip(activity_rows, activity_names, strict=True):
        panel.activity_valid[kept[t], j] = old("activity_valid")[t, j]
        panel.trade_observed[kept[t], j] = old("trade_observed")[t, j]
        for value, key in (
            (panel.volume_brl, "volume_brl"),
            (panel.trades, "trade_count"),
            (panel.quantity, "quantity"),
        ):
            value[kept[t], j] = old(key)[t, j]
    write_json_atomic(
        output / "preserved_activity_contract.json",
        {
            "parent": parent,
            "rows": activity_evidence,
            "status": "separate_source_activity_disposition_pending_not_a_rename_change",
        },
    )
    config = PROJECT / "research/configs/v2/isin_links_allowlist.csv"
    links = load_isin_link_allowlist(config, daily)
    assert links.height == 3
    assert links["shares_received_per_prior_share"].to_list() == [1.0] * 3
    assert links["cash_entitlement_per_prior_share"].to_list() == [0.0] * 3
    conversions = verified_conversion_terms_from_links(links)
    links.write_parquet(output / "isin_succession_links.parquet")
    slow_history_links(
        links, dates, isins, [schedule[int(t)].decision_at for t in kept]
    ).write_parquet(output / "slow_history_links.parquet")
    ancestor, ancestor_root = manifest, source
    while "round5_extension" in ancestor["metadata"]:
        record = ancestor["metadata"]["round5_extension"]["base_store"]
        ancestor_root = Path(record["root"])
        ancestor = bound_json(
            {
                "path": str(ancestor_root / "manifest.json"),
                "sha256": record["manifest_sha256"],
            }
        )
    feature_terms = verified_action_terms_from_table(
        pl.read_parquet(
            ancestor_root
            / ancestor["tables"]["corporate_actions_verified_terms"]["path"]
        )
    )
    original_universe = build_daily_universe(
        panel.close_brl,
        panel.volume_brl,
        panel.observed,
        trade_observed=panel.trade_observed,
        activity_valid=panel.activity_valid,
        source_session_complete=panel.source_session_complete,
    )
    inferred = infer_cotahist_action_terms(
        panel.dates,
        isins,
        panel.close_brl,
        panel.quantity,
        panel.trades,
        panel.distribution_number,
        panel.observed,
        original_universe.active,
    )
    decisions = next_session_decision_cutoffs(schedule, following_decision_at=following)
    actions = align_decision_known_action_terms(
        (*feature_terms, *conversions),
        panel.dates,
        isins,
        coverage_resolved=inferred.coverage_resolved,
        decision_timestamps=decisions,
    )
    wealth = [np.empty(panel.observed.shape, np.float32) for _ in range(4)]
    wealth_valid = np.empty(panel.observed.shape, bool)
    build_shareholder_wealth_ohlc_into(
        panel.open_brl,
        panel.high_brl,
        panel.low_brl,
        panel.close_brl,
        panel.observed,
        actions,
        wealth_open=wealth[0],
        wealth_high=wealth[1],
        wealth_low=wealth[2],
        wealth_close=wealth[3],
        wealth_valid=wealth_valid,
    )
    linked = _route_decision_known_continuations(
        dates=panel.dates,
        isins=isins,
        links=links,
        decision_timestamps=[s.decision_at for s in schedule],
        raw_close=panel.close_brl,
        volume_brl=panel.volume_brl,
        trades=panel.trades,
        observed=panel.observed,
        trade_observed=panel.trade_observed,
        activity_valid=panel.activity_valid,
        ambiguous_action=~actions.session_resolved,
        shareholder_wealth_arrays=(*wealth, wealth_valid),
    )
    universe = build_daily_universe(
        linked.close_brl,
        linked.volume_brl,
        linked.observed,
        trade_observed=linked.trade_observed,
        activity_valid=linked.activity_valid,
        source_session_complete=panel.source_session_complete,
    )
    active = universe.active & linked.claim_owner
    timestep = np.zeros_like(active)
    timestep[1:] = np.maximum.accumulate(linked.observed[:-1], axis=0)
    new = {}

    def save(name, values):
        np.save(output / f"{name}.npy", values)
        new[name] = np.load(output / f"{name}.npy", mmap_mode="r")

    save("active", active[kept])
    save("prior_reference_close", universe.prior_close_brl[kept].astype(np.float32))
    save("slow_timestep_valid", timestep[kept])
    for key, value in zip(("open", "high", "low", "close"), wealth, strict=True):
        save(f"shareholder_wealth_{key}", value[kept])
    save("shareholder_wealth_valid", wealth_valid[kept])
    changed_names = {
        isins.index(r[k])
        for r in links.iter_rows(named=True)
        for k in ("predecessor_isin", "successor_isin")
    }
    untouched = np.array([j for j in range(len(isins)) if j not in changed_names])
    for name in new:
        np.testing.assert_array_equal(
            new[name][:, untouched], old(name)[:, untouched], err_msg=name
        )
    print(
        json.dumps(
            {
                "stage": "full_history_universe",
                "seconds": perf_counter() - started,
                "active_added": int((new["active"] & ~old("active")).sum()),
                "active_removed": int((old("active") & ~new["active"]).sum()),
            }
        ),
        flush=True,
    )
    # A same-class rename carries the pre-boundary source for the two intraday
    # shape statistics too. This never changes an observed raw-store quote.
    history_raw = [a.copy() for a in (panel.high_brl, panel.low_brl, panel.close_brl)]
    for row in links.iter_rows(named=True):
        a, b = (isins.index(row[k]) for k in ("predecessor_isin", "successor_isin"))
        boundary = int(
            np.searchsorted(panel.dates, np.datetime64(row["effective_date"]))
        )
        for array in history_raw:
            array[:boundary, b] = array[:boundary, a]
    shape = (*new["active"].shape, len(SLOW_FEATURES))
    slow = np.lib.format.open_memmap(
        output / "slow_values.npy", mode="w+", dtype=np.float32, shape=shape
    )
    valid = np.lib.format.open_memmap(
        output / "slow_valid.npy", mode="w+", dtype=bool, shape=shape
    )
    age = np.lib.format.open_memmap(
        output / "slow_age_sessions.npy", mode="w+", dtype=np.float32, shape=shape
    )
    specs = tuple(
        FeatureSpec(**s)
        for s in manifest["metadata"]["feature_schema"]["specifications"]
        if s["family"] == "slow"
    )
    assert tuple(s.name for s in specs) == SLOW_FEATURES
    sigma = np.full(panel.observed.shape, np.nan, np.float32)

    def consume(index, values, mask):
        transform_feature_panel_into(
            values[..., None],
            mask[..., None],
            active,
            specs[index : index + 1],
            slow[..., index : index + 1],
            valid[..., index : index + 1],
            source_rows=kept - 1,
            membership_rows=kept,
            minimum_rank_names=20,
        )
        observation_age_sessions_into(
            mask[..., None],
            active,
            age[..., index : index + 1],
            source_rows=kept - 1,
            decision_rows=kept,
        )
        if index == 8:
            sigma[1:] = np.where(mask[:-1], values[:-1], np.nan).astype(np.float32)
        print(
            json.dumps(
                {
                    "stage": "slow_feature",
                    "name": SLOW_FEATURES[index],
                    "seconds": perf_counter() - started,
                }
            ),
            flush=True,
        )

    clusters = build_slow_features_into(
        *wealth,
        linked.volume_brl,
        linked.trades,
        wealth_valid,
        active,
        panel.dates,
        raw_high=history_raw[0],
        raw_low=history_raw[1],
        raw_close=history_raw[2],
        price_observed=linked.observed,
        history_observed=linked.observed,
        activity_valid=linked.activity_valid,
        ambiguous_action=linked.ambiguous_action,
        consume=consume,
        history_links=slow_history_links(
            links, panel.dates, isins, [s.decision_at for s in schedule]
        ).to_dicts(),
    )
    for name, array in (
        ("slow_values", slow),
        ("slow_valid", valid),
        ("slow_age_sessions", age),
    ):
        array.flush()
        new[name] = array
    save("target_scale_sigma", sigma[kept])
    save("monthly_cluster_labels", clusters[kept])
    common_values, common_valid, common_table = _common_state_diagnostic_panel(
        wealth[3],
        wealth_valid,
        linked.ambiguous_action,
        active,
        sigma,
        kept,
        panel.dates,
        minimum_names=20,
    )
    save("common_state_diagnostic_values", common_values)
    save("common_state_diagnostic_valid", common_valid)
    common_table.write_parquet(output / "common_state_diagnostics.parquet")
    # Hold all previously accepted retrospective economic terms and coverage
    # fixed, adding only the three sourced conversions on the full warmup axis.
    target_terms = verified_action_terms_from_table(
        pl.read_parquet(
            source / manifest["tables"]["corporate_actions_verified_terms"]["path"]
        )
    )
    coverage = inferred.coverage_resolved.copy()
    coverage[kept] = old("action_session_resolved")
    retrospective = align_verified_action_terms(
        (*target_terms, *conversions), panel.dates, isins, coverage_resolved=coverage
    )
    for key, value in (
        ("action_shares_per_prior_share", retrospective.shares_per_prior_share),
        ("action_cash_per_prior_share", retrospective.cash_per_prior_share),
        ("action_session_resolved", retrospective.session_resolved),
        ("action_has_action", retrospective.has_action),
        ("action_successor_index", retrospective.successor_index),
    ):
        save(key, value[kept].astype(old(key).dtype))
    destinations = {}
    for argument, name in (
        ("primary", "target_primary"),
        ("primary_valid", "target_valid"),
        ("normalized_residual", "target_normalized_residual"),
        ("normalized_cross_section_valid", "target_normalized_cross_section_valid"),
        ("shareholder_midrank", "target_shareholder_midrank"),
        ("shareholder_valid", "target_shareholder_valid"),
        ("shareholder_simple_return", "target_shareholder_simple_return"),
        ("terminal_wealth", "target_terminal_wealth"),
        ("terminal_loss", "target_terminal_loss"),
        ("price_midrank", "target_price_midrank"),
        ("price_valid", "target_price_valid"),
        ("price_simple_return", "target_price_simple_return"),
    ):
        destinations[argument] = np.empty(old(name).shape, old(name).dtype)
    target_close, target_observed = panel.close_brl.copy(), panel.observed.copy()
    target_close[kept] = np.round(old("raw_close").astype(np.float64), 2)
    target_observed[kept] = old("observed")
    build_economic_multi_day_targets_into(
        target_close,
        target_observed,
        active,
        sigma,
        retrospective,
        source_rows=kept,
        **destinations,
    )
    for argument, value in destinations.items():
        name = "target_valid" if argument == "primary_valid" else "target_" + argument
        save(name, value)
    np.save(output / "date_index.npy", dates)
    np.save(output / "isin_index.npy", np.array(isins))
    basis = output / "continuation_history_basis"
    basis.mkdir(exist_ok=False)
    for name in (
        "shareholder_wealth_open",
        "shareholder_wealth_high",
        "shareholder_wealth_low",
        "shareholder_wealth_close",
        "shareholder_wealth_valid",
        "prior_reference_close",
        "target_scale_sigma",
        "slow_timestep_valid",
    ):
        path = output / f"{name}.npy"
        close_memmap(new.pop(name))
        path.rename(basis / path.name)
        values = np.load(basis / path.name)
        for row in links.iter_rows(named=True):
            j = isins.index(row["successor_isin"])
            before = dates < np.datetime64(row["effective_date"])
            values[before, j] = old(name)[before, j]
        save(name, values)
    finish(output, source, manifest, parent, started)


def finish(output, source, manifest, parent, started):
    dates = np.load(output / "date_index.npy")
    links = pl.read_parquet(output / "isin_succession_links.parquet")
    new = {p.stem: np.load(p, mmap_mode="r") for p in output.glob("*.npy")}
    differences = {}
    first_boundary = min(
        np.datetime64(r["effective_date"]) for r in links.iter_rows(named=True)
    )
    # Ten actual equity sessions, rather than calendar days, bound the longest
    # label whose endpoint can cross the first rename.
    safe_prefix = np.arange(len(dates)) < np.searchsorted(dates, first_boundary) - 10
    for name, values in new.items():
        if name not in manifest["arrays"]:
            continue
        baseline = np.load(source / manifest["arrays"][name]["path"], mmap_mode="r")
        equal = (values == baseline) | (np.isnan(values) & np.isnan(baseline))
        differences[name] = {
            "changed": int((~equal).sum()),
            "shape": list(values.shape),
            "prefix_changed": int((~equal[safe_prefix]).sum()),
        }
        # Successor history is source-routed only by a future decision. Raw
        # observations, eligibility and historical feature coordinates must not
        # become available early; those store cells remain inactive/masked.
        if differences[name]["prefix_changed"]:
            raise AssertionError(
                f"unexpected unaffected prefix differences: {name}: {differences[name]}"
            )
    result = {
        "status": "rename_history_intermediate_not_an_accepted_model_store",
        "parent": parent,
        "allowlist": binding(PROJECT / "research/configs/v2/isin_links_allowlist.csv"),
        "sources": [
            r
            for r in manifest["sources"]
            if Path(r.get("path", "")).name.startswith("equities_daily_")
            or Path(r.get("path", "")).name
            == "b3_session_schedule_reconstructed_v1.csv"
        ],
        "links": binding(output / "isin_succession_links.parquet"),
        "implementation": binding(output / "executed_reproducer.py"),
        "qualified_implementation": binding(Path(__file__)),
        "history_mapping": binding(output / "slow_history_links.parquet"),
        "preserved_activity_contract": binding(
            output / "preserved_activity_contract.json"
        ),
        "history_basis": {
            p.stem: binding(p)
            for p in (output / "continuation_history_basis").glob("*.npy")
        },
        "arrays": {name: binding(output / f"{name}.npy") for name in new},
        "differences": differences,
        "seconds": perf_counter() - started,
        "remaining": [
            "source-deletion causality and actual full-population neural window proof",
            "financial/sector/cross-market/lending and other dependent families under changed history",
            "M1-dependent features and auxiliary masks on changed eligibility",
            "source corporate baskets beyond the three rename links",
            "explicit accepted derived-store contract and refits",
        ],
        "heldout_access": False,
        "forecast_scoring": False,
    }
    write_json_atomic(output / "manifest.json", result)
    print(
        json.dumps(
            {
                "stage": "complete",
                "seconds": result["seconds"],
                "differences": differences,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
