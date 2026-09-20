"""Admit reconciled lending to new feature coordinates, preserving old sources.

The economic and feature archives have different prior coverage. Combine their
observations explicitly; do not replace the larger feature archive wholesale.
No raw PDF, accepted store, policy cache or checkpoint is changed.
"""

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
from brazil_rv.v2.round5_derived import verified

PROJECT = Path(__file__).resolve().parents[1]
KEYS = ["date", "isin"]


def main():
    begun = perf_counter()
    run = json.loads(
        (PROJECT / "docs/v2_economic_data_scaling_run.json").read_text(encoding="utf8")
    )
    pointer = json.loads(
        (PROJECT / "docs/v2_data_inputs.json").read_text(encoding="utf8")
    )
    output = Path(run["root"]) / "lending_feature_propagation"
    output.mkdir(exist_ok=False)
    (output / "executed_reproducer.py").write_bytes(Path(__file__).read_bytes())
    store = Path(pointer["store"]["root"])
    manifest = bound_json(
        {
            "path": str(store / "manifest.json"),
            "sha256": pointer["store"]["manifest_sha256"],
        }
    )
    axis = np.load(store / "date_index.npy")
    assert axis[-1] <= np.datetime64("2024-12-30")
    sessions, isins = (
        axis.astype(object).tolist(),
        np.load(store / "isin_index.npy").tolist(),
    )
    families = source_families(manifest)
    old_inputs = bound_json(families["lending"]["source_manifest"])["sources"]["inputs"]
    peer = bound_json(run["fca_peer_propagation"])
    float_inputs = bound_json(peer["lending_source_bindings"])
    rename = bound_json(run["rename_history_propagation"])
    issuer = bound_json(run["rename_issuer_propagation"])
    links = pl.read_parquet(verified(rename["history_mapping"])).to_dicts()
    active = np.load(verified(rename["arrays"]["active"]), mmap_mode="r")
    ti, ni = np.where(active)
    active_keys = pl.DataFrame({"date": axis[ti], "isin": np.asarray(isins)[ni]})
    old = pl.read_parquet(verified(peer["artifacts"]["lending"]))
    # Family writers encode unknown age as either null or -1. Both map to the
    # identical absent-age tensor; compare and store the canonical -1 here.
    old = old.with_columns(
        pl.col(c).fill_null(-1) for c in old.columns if c.endswith("_age_sessions")
    )
    basic = [c for c in old.columns if not c.startswith("utilization_proxy")]
    recovered_root = Path(run["recovered_lending"]["root"])
    recovered_manifest = bound_json(
        {
            "path": str(recovered_root / "manifest.json"),
            "sha256": run["recovered_lending"]["manifest_sha256"],
        }
    )
    receipts = dict(old_inputs)
    receipts.update(
        {
            "issuer": run["rename_issuer_propagation"],
            "rename": run["rename_history_propagation"],
            "prior_features": peer["artifacts"]["lending"],
            "recovered_admission": run["recovered_lending"]["admission"],
        }
    )

    def table(record):
        return pl.read_parquet(verified(record))

    old_balances, old_rates = table(old_inputs["balances"]), table(old_inputs["rates"])
    new = {}
    for name in ("balances", "rates"):
        filename = "lending_" + name + ".parquet"
        record = {
            "path": str(recovered_root / filename),
            "sha256": recovered_manifest["artifacts"][filename]["sha256"],
        }
        receipts["recovered_" + name] = record
        new[name] = table(record)
    # Reuse the earlier source-specific unit dispositions, not a numeric clip.
    # The printed rows remain unchanged in the recovered archive and receipts.
    october = bound_json(float_inputs["old_lending"]["previous_manifest"])
    support = bound_json(october["previous_manifest"])
    support_root = Path(october["previous_manifest"]["path"]).parent
    quarantine_record = {
        "path": str(support_root / "admission_audit.json"),
        "sha256": support["artifacts"]["admission_audit.json"]["sha256"],
    }
    disposition = bound_json(quarantine_record)["field_quarantines"]
    assert (
        disposition[0]["report_date"] == "2020-07-15"
        and disposition[0]["field"] == "balance_brl"
    )
    assert (
        disposition[1]["report_date"] == "2020-10-15"
        and disposition[1]["ticker"] == "GMAT3"
    )
    receipts["prior_field_dispositions"] = quarantine_record
    july = new["balances"].filter(pl.col("source_report_date") == date(2020, 7, 15))
    gmat = new["balances"].filter(
        (pl.col("source_report_date") == date(2020, 10, 15))
        & (pl.col("security_id") == "ISIN:BRGMATACNOR7")
    )
    assert (
        gmat.height == 1
        and gmat["lending_balance_quantity"][0] == 10
        and gmat["lending_balance_brl"][0] == 9070000
    )
    pl.concat([july, gmat]).write_parquet(output / "source_field_dispositions.parquet")
    qualified = (
        new["balances"]
        .join(
            gmat.select("source_report_date", "security_id"),
            on=["source_report_date", "security_id"],
            how="anti",
        )
        .with_columns(
            pl.when(pl.col("source_report_date") == date(2020, 7, 15))
            .then(None)
            .otherwise(pl.col("lending_balance_brl"))
            .alias("lending_balance_brl")
        )
    )
    keys = ["source_report_date", "security_id"]
    overlap = old_balances.join(qualified, on=keys, suffix="_recovered")
    for c in (
        "source_position_date",
        "available_date",
        "lending_balance_quantity",
        "lending_balance_brl",
    ):
        if overlap.filter(~pl.col(c).eq_missing(pl.col(c + "_recovered"))).height:
            raise ValueError(f"Unadjudicated lending overlap: {c}")
    additions = qualified.join(old_balances.select(keys), on=keys, how="anti")
    balances = pl.concat(
        [old_balances, additions.select(old_balances.columns)], how="vertical_relaxed"
    ).sort("available_date", "security_id")
    assert_frame_equal(
        balances.join(old_balances.select(keys), on=keys).sort(keys),
        old_balances.sort(keys),
        check_exact=True,
    )
    rates = new["rates"]
    rate_keys = ["source_trade_date", "security_id"]
    # The audited all-modality source covers every formerly accepted rate key.
    assert old_rates.join(rates.select(rate_keys), on=rate_keys, how="anti").is_empty()
    overlap_rates = old_rates.join(rates, on=rate_keys, suffix="_recovered")
    assert overlap_rates.filter(
        pl.col("available_date") != pl.col("available_date_recovered")
    ).is_empty()
    for frame in (rates, balances):
        assert not frame.select(
            pl.struct("available_date", "security_id").is_duplicated().any()
        ).item()
        assert frame["available_date"].max() <= sessions[-1]
    assert rates.filter(pl.col("registered_quantity") <= 0).is_empty()
    next_session = dict(zip(sessions[:-1], sessions[1:]))
    assert all(
        next_session[r["source_trade_date"]] == r["available_date"]
        for r in rates.iter_rows(named=True)
    )
    balances.write_parquet(output / "lending_balances.parquet")
    additions.write_parquet(output / "added_balance_observations.parquet")
    rate_changed = overlap_rates.filter(
        pl.any_horizontal(
            [
                ~pl.col(c).eq_missing(pl.col(c + "_recovered"))
                for c in (
                    "annual_taker_rate",
                    "registered_quantity",
                    "registered_contracts",
                )
            ]
        )
    )
    rate_changed.write_parquet(output / "changed_rate_observations.parquet")
    print(
        json.dumps(
            {
                "stage": "source_union",
                "balances": balances.height,
                "added": additions.height,
                "rates": rates.height,
                "rate_replacements": rate_changed.height,
            }
        ),
        flush=True,
    )

    base_market = Path(old_inputs["base_manifest"]["path"]).parent
    volume = np.load(base_market / "volume_brl.npy")

    # Preserve the original producer arithmetic and independently compare the
    # six original fields before attributing any recovered-source difference.
    def typed(frame):
        return (
            frame.select(old.columns)
            .cast(old.schema)
            .with_columns(
                pl.col(c).fill_null(-1)
                for c in old.columns
                if c.endswith("_age_sessions")
            )
            .sort(KEYS)
        )

    control = typed(
        b3.lending_decision_features(old_balances, old_rates, sessions, isins, volume)
    )
    oti, oni = np.where(np.load(store / "active.npy", mmap_mode="r"))
    original_active_keys = pl.DataFrame(
        {"date": axis[oti], "isin": np.asarray(isins)[oni]}
    )
    assert_frame_equal(
        control.select(basic)
        .join(old.select(KEYS), on=KEYS)
        .join(original_active_keys, on=KEYS)
        .sort(KEYS),
        old.select(basic).join(original_active_keys, on=KEYS).sort(KEYS),
        check_exact=True,
    )
    source_only = typed(
        b3.lending_decision_features(balances, rates, sessions, isins, volume)
    )
    source_only.select(basic).write_parquet(output / "source_only_features.parquet")
    # Only the spot ADV denominator inherits sourced same-unit history. No loan
    # rate, position or registered-flow record is copied between ISINs.
    for link in sorted(links, key=lambda r: r["effective_index"]):
        p, n, e = (
            link[k] for k in ("predecessor_index", "successor_index", "effective_index")
        )
        volume[:e, n] = volume[:e, p]
    goll = bound_json(run["goll_activity_admission"])
    # The activity correction predates every balance observation; use its exact
    # admitted physical field without re-reading the rejected quote/PDF.
    assert balances["source_position_date"].min() > date(2011, 3, 31)
    # These are the exact individually bound activity terms (the manifest binds
    # their source row). They cannot affect a 2019+ ADV20 window.
    volume[
        np.searchsorted(axis, np.datetime64("2011-02-16")), isins.index("BRGOLLACNPR4")
    ] = 21129704
    receipts["activity"] = run["goll_activity_admission"]
    assert goll
    updated = typed(
        b3.lending_decision_features(balances, rates, sessions, isins, volume)
    )
    for link in links:
        n = isins[link["successor_index"]]
        first = sessions[max(link["effective_index"], link["known_index"])]
        # Any pre-admission observation keeps its original source coordinate.
        scope = (pl.col("isin") == n) & (pl.col("date") < first)
        updated = pl.concat([updated.filter(~scope), source_only.filter(scope)]).sort(
            KEYS
        )
    identity = table(issuer["artifacts"]["identity"])
    changes = bound_json(
        float_inputs["new_cvm_files"]["capital_change_observations.json"]
    )
    for row in changes:
        row["effective"] = date.fromisoformat(row["effective"])
    market = cvm.valuation_market(
        Path(float_inputs["base_inputs"][0]["path"]).parent, history_links=links
    )
    floats = table(float_inputs["new_cvm_files"]["free_float_observations.parquet"])
    utilization, util_audit = b3.lending_utilization_features(
        balances, floats, identity, sessions, market, changes
    )
    util_fields = [c for c in old.columns if c.startswith("utilization_proxy")]
    updated = typed(
        updated.drop(util_fields).join(utilization, on=KEYS, how="full", coalesce=True)
    )
    final = updated.join(active_keys, on=KEYS, how="inner").sort(KEYS)
    final.write_parquet(output / "features.parquet")
    write_json_atomic(
        output / "production.json",
        {
            "source_union_seconds": perf_counter() - begun,
            "source_receipts": receipts,
            "denominator_audit": util_audit,
            "six_field_control_exact_active_rows": old.join(
                original_active_keys, on=KEYS
            ).height,
            "source_only_and_final": {
                p.name: binding(p)
                for p in [
                    output / "source_only_features.parquet",
                    output / "features.parquet",
                ]
            },
        },
    )
    from qualify_lending_propagation import main as qualify

    qualify()


if __name__ == "__main__":
    main()
