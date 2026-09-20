"""Bounded rename propagation for activity, options, index previews and odd lots.

Reuse admitted normalized source receipts. Never reparse the complete raw archive
or attach these changed coordinates to an old checkpoint.
"""

import argparse
import json
from datetime import date
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.auxiliary_history import (
    renamed_activity_features,
    renamed_oddlot_features,
)
from brazil_rv.v2.build_store import _prior_adv20
from brazil_rv.v2.data_repair import binding, bound_json, source_families
from brazil_rv.v2.round5_b3 import activity_decision_features
from brazil_rv.v2.round5_derived import verified
from brazil_rv.v2.round5_index import pressure_panel
from brazil_rv.v2.sidecars import derive_known_archive_features

PROJECT = Path(__file__).resolve().parents[1]
KEYS = ["date", "isin"]


def replace(frame, addition, successor, first, last):
    scope = (pl.col("isin") == successor) & pl.col("date").is_between(first, last)
    return pl.concat(
        [frame.filter(~scope), addition.select(frame.columns)], how="vertical_relaxed"
    ).sort(KEYS)


def exact_nullable(control, parent, names, first, last, isins):
    keys = pl.DataFrame({"isin": isins}).join(
        pl.DataFrame({"date": pl.date_range(first, last, eager=True)}), how="cross"
    )
    cols = KEYS + names + [n + "_age_sessions" for n in names]
    a = keys.join(control.select(cols), on=KEYS, how="left").sort(KEYS)
    b = keys.join(parent.select(cols), on=KEYS, how="left").sort(KEYS)
    assert_frame_equal(a, b, check_dtypes=False, check_exact=True)
    return a.height * len(names) * 2


def odd_family(frame, sessions):
    # Existing raw source has exact D+1 availability (already independently
    # audited). Preserve measurement/publication distance on new history rows.
    positions = {d: i for i, d in enumerate(sessions)}
    names = ["oddlot_volume_share", "oddlot_volume_share_change_5"]
    return frame.select(
        pl.col("available_date").alias("date"),
        "isin",
        *[pl.when(pl.col(n + "_mask")).then(pl.col(n)).alias(n) for n in names],
        *[
            pl.when(pl.col(n + "_mask"))
            .then(
                pl.col("available_date").replace_strict(positions)
                - pl.col("source_trade_date").replace_strict(positions)
            )
            .alias(n + "_age_sessions")
            for n in names
        ],
    ).sort(KEYS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    began = perf_counter()
    pointer_path = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer_path.read_text(encoding="utf8"))
    accepted = json.loads(
        (PROJECT / "docs/v2_data_inputs.json").read_text(encoding="utf8")
    )
    store = Path(accepted["store"]["root"])
    manifest = bound_json(
        {
            "path": str(store / "manifest.json"),
            "sha256": accepted["store"]["manifest_sha256"],
        }
    )
    families = source_families(manifest)
    rename = bound_json(run["rename_history_propagation"])
    links = pl.read_parquet(verified(rename["history_mapping"])).to_dicts()
    axis = np.load(store / "date_index.npy")
    sessions, isins = (
        axis.astype(object).tolist(),
        np.load(store / "isin_index.npy").tolist(),
    )
    assert axis[-1] <= np.datetime64("2024-12-31")
    output = args.output or Path(run["root"]) / "remaining_auxiliaries"
    output.mkdir(exist_ok=False)
    (output / "executed_reproducer.py").write_bytes(Path(__file__).read_bytes())
    parents = {
        k: pl.read_parquet(verified(families[k]["data"]))
        for k in ["microstructure", "options", "rebalance"]
    }
    final = dict(parents)
    activity_manifest = bound_json(families["microstructure"]["source_manifest"])
    root = Path(activity_manifest["builder"]["path"]).parent
    source_receipts = []

    def source(record, predicate=None):
        path = verified(record)
        source_receipts.append(record)
        lazy = pl.scan_parquet(path)
        return (lazy.filter(predicate) if predicate is not None else lazy).collect()

    # Resolve every input through the producer's bound inventory rather than
    # discovering a newer file by name.
    inventory = bound_json(activity_manifest["source_inventory"])
    inputs = {
        Path(r["path"]).parent.name + "/" + Path(r["path"]).name: r
        for r in activity_manifest["inputs"]
    }
    cash_manifest = bound_json(inputs["b3_cash_axis/manifest.json"])
    cash_record = {
        "path": str(root / "b3_cash_axis/cash.parquet"),
        "sha256": cash_manifest["artifacts"]["cash.parquet"]["sha256"],
    }
    cash = source(cash_record)
    qmanifest = bound_json(inputs["cotahist_option_volume/manifest.json"])
    quantities = []
    for r in qmanifest["files"]:
        if Path(r["output"]).stem[-4:] in {"2023", "2024"}:
            quantities.append(
                source(
                    {
                        "path": str(root / "cotahist_option_volume" / r["output"]),
                        "sha256": r["output_sha256"],
                    }
                )
            )
    quantities = pl.concat(quantities, how="vertical_relaxed")
    correction = bound_json(
        next(
            r
            for r in activity_manifest["inputs"]
            if Path(r["path"]).name == "cotahist_interval_source_audit.json"
        )
    )
    # The existing recovery is dated 2010-2011 and does not enter these windows.
    if correction.get("recovered_name_days"):
        recovered = pl.read_parquet(correction["recovered_path"])
        quantities = pl.concat(
            [
                quantities,
                recovered.filter(pl.col("source_trade_date") >= date(2023, 1, 1)),
            ],
            how="vertical_relaxed",
        )
    odd_record = next(
        r
        for r in manifest["sources"]
        if Path(r.get("path", "")).name == "odd_lot_activity.parquet"
    )
    renamed_isins = [
        isins[r[k]] for r in links for k in ["predecessor_index", "successor_index"]
    ]
    odd_source = source(
        odd_record,
        pl.col("isin").is_in(renamed_isins)
        & (
            pl.col("source_trade_date")
            >= sessions[min(r["effective_index"] for r in links) - 6]
        ),
    )
    reports, odd_patches = [], []
    for number, link in enumerate(links):
        effective = link["effective_index"]
        first, last = (
            max(effective, link["known_index"]),
            min(effective + 22, len(sessions) - 1),
        )
        days = sessions[effective - 22 : last + 1]
        predecessor, successor = (
            isins[link["predecessor_index"]],
            isins[link["successor_index"]],
        )
        spec = dict(
            predecessor=predecessor,
            successor=successor,
            effective=sessions[effective],
            known=sessions[link["known_index"]],
        )
        predicate = pl.col("isin").is_in([predecessor, successor]) & pl.col(
            "source_trade_date"
        ).is_between(days[0], days[-1])
        oi, pr = [], []
        for record in inventory:
            path = Path(record["path"])
            if path.suffix != ".parquet" or not path.name.endswith(
                ("_oi.parquet", "_cash.parquet")
            ):
                continue
            day = date.fromisoformat(path.name[:10])
            if days[0] <= day < days[-1]:
                (oi if path.name.endswith("_oi.parquet") else pr).append(
                    source(record, predicate)
                )
        frames = [
            cash.filter(predicate),
            quantities.filter(predicate),
            pl.concat(oi, how="vertical_relaxed"),
            pl.concat(pr, how="vertical_relaxed"),
        ]
        # Save the exact bounded receipts, including original assignment and
        # availability columns, so qualifications never repeat source reads.
        scope = output / successor
        scope.mkdir()
        for name, frame in zip(
            ["cash", "quantities", "snapshots", "nonregular"], frames
        ):
            frame.write_parquet(scope / f"{name}.parquet")
        control = activity_decision_features(*frames, days)
        changed = renamed_activity_features(*frames, days, **spec)
        count = 0
        for family, old, new in zip(["options", "microstructure"], control, changed):
            count += exact_nullable(
                old,
                parents[family],
                families[family]["feature_names"],
                sessions[first],
                sessions[last],
                [predecessor, successor],
            )
            final[family] = replace(
                final[family], new, successor, sessions[first], sessions[last]
            )
            new.write_parquet(scope / f"{family}.parquet")
        cutoff = sessions[min(first + 6, last)]
        mutation = renamed_activity_features(
            *(f.filter(pl.col("source_trade_date") < cutoff) for f in frames),
            days,
            **spec,
        )
        for a, b in zip(mutation, changed):
            assert_frame_equal(
                a.filter(pl.col("date") <= cutoff),
                b.filter(pl.col("date") <= cutoff),
                check_exact=True,
            )
        odd = odd_source.filter(
            pl.col("isin").is_in([predecessor, successor])
            & (pl.col("source_trade_date") >= sessions[effective - 6])
        )
        odd.write_parquet(scope / "odd_source.parquet")
        odd_days = sessions[effective - 6 :]
        before = odd_family(
            derive_known_archive_features(
                odd, odd_days, [predecessor, successor], group="oddlot"
            ),
            odd_days,
        )
        after = odd_family(renamed_oddlot_features(odd, odd_days, **spec), odd_days)
        after = after.filter(pl.col("date") <= sessions[-1])
        before.write_parquet(scope / "odd_control.parquet")
        after.write_parquet(scope / "oddlot.parquet")
        odd_patches.append(after)
        reports.append(
            {
                "link": link,
                "predecessor": predecessor,
                "successor": successor,
                "start": str(days[0]),
                "end": str(days[-1]),
                "control_cells": count,
                "causal_prefix": str(cutoff),
                "source_rows": dict(
                    zip(
                        ["cash", "quantities", "snapshots", "nonregular"],
                        [len(f) for f in frames],
                    )
                ),
            }
        )
        print(
            json.dumps({"completed": successor, "seconds": perf_counter() - began}),
            flush=True,
        )

    index_manifest = bound_json(families["rebalance"]["source_manifest"])
    snapshots = bound_json(index_manifest["snapshots"])
    portfolios = source(snapshots["output"])
    # Keep only the changed event tail plus its last effective baseline.
    portfolios = portfolios.filter(pl.col("disclosure_date") >= date(2023, 8, 1))
    base = Path(index_manifest["base_store"]["path"]).parent
    bound_json(index_manifest["base_store"])
    volume = np.load(base / "volume_brl.npy").copy()
    valid = np.load(base / "activity_valid.npy").copy()
    old_active = np.load(base / "active.npy")
    active = np.load(verified(rename["arrays"]["active"]))
    old_adv = _prior_adv20(volume, valid)
    calendar = bound_json(index_manifest["calendar"])
    future = [date.fromisoformat(d) for d in calendar["sessions"]]
    index_cash = cash.select("source_trade_date", "isin", "ticker")
    control, _ = pressure_panel(
        portfolios,
        index_cash,
        sessions,
        isins,
        old_active,
        old_adv,
        future_sessions=future,
    )
    boundary = min(r["effective_index"] for r in links)
    count = exact_nullable(
        control.filter(pl.col("date") >= sessions[boundary]),
        parents["rebalance"],
        families["rebalance"]["feature_names"],
        sessions[boundary],
        sessions[-1],
        isins,
    )
    for link in links:
        a, b, t = (
            link["predecessor_index"],
            link["successor_index"],
            link["effective_index"],
        )
        volume[:t, b], valid[:t, b] = volume[:t, a], valid[:t, a]
    adv = _prior_adv20(volume, valid)
    # Internal pre-birth continuation cannot become a public pre-birth input.
    for link in links:
        gate = max(link["effective_index"], link["known_index"])
        adv[:gate, link["successor_index"]] = old_adv[:gate, link["successor_index"]]
    eligibility_only, _ = pressure_panel(
        portfolios, index_cash, sessions, isins, active, old_adv, future_sessions=future
    )
    changed, identity_audit = pressure_panel(
        portfolios,
        index_cash,
        sessions,
        isins,
        active,
        adv,
        future_sessions=future,
        history_links=tuple(links),
    )
    for name, frame in [
        ("index_control", control),
        ("index_eligibility_only", eligibility_only),
        ("index_amended", changed),
    ]:
        frame.write_parquet(output / f"{name}.parquet")
    final["rebalance"] = pl.concat(
        [
            parents["rebalance"].filter(pl.col("date") < sessions[boundary]),
            changed.filter(pl.col("date") >= sessions[boundary]),
        ]
    ).sort(KEYS)
    cutoff = date(2024, 4, 16)
    mutation, _ = pressure_panel(
        portfolios.filter(pl.col("disclosure_date") <= cutoff),
        index_cash.filter(pl.col("source_trade_date") <= cutoff),
        sessions,
        isins,
        active,
        adv,
        future_sessions=future,
        history_links=tuple(links),
    )
    assert_frame_equal(
        changed.filter(pl.col("date") <= cutoff),
        mutation.filter(pl.col("date") <= cutoff),
        check_exact=True,
    )
    final["oddlot"] = pl.concat(odd_patches).sort(KEYS)
    for name, frame in final.items():
        frame.write_parquet(output / f"{name}.parquet")
    # GOLL's accepted activity was already retained in this separate cash path.
    goll = (
        cash.filter(
            (pl.col("isin") == "BRGOLLACNPR4")
            & (pl.col("source_trade_date") == date(2011, 2, 16))
        )
        .select("quantity", "volume_brl", "trades")
        .to_dicts()
    )
    assert goll == [{"quantity": 908400, "volume_brl": 21129704.0, "trades": 2743}]
    report = {
        "status": "produced_pending_consumer_qualification",
        "parent": accepted["store"],
        "history": run["rename_history_propagation"],
        "parent_families": {
            k: {
                x: families[k][x]
                for x in [
                    "data",
                    "source_manifest",
                    "availability_proof",
                    "feature_names",
                ]
            }
            for k in parents
        },
        "source_receipts": list({r["path"]: r for r in source_receipts}.values()),
        "scopes": reports,
        "index_control_cells": count,
        "index_identity_audit": identity_audit,
        "index_causal_prefix": str(cutoff),
        "goll_activity_already_in_source": goll,
        "artifacts": {k: binding(output / f"{k}.parquet") for k in final},
        "oddlot_scope": "successor rows only; replace only these dated axes, retain all other sealed odd-lot arrays",
        "seconds": perf_counter() - began,
        "code": [
            binding(Path(__file__)),
            binding(PROJECT / "research/src/brazil_rv/v2/auxiliary_history.py"),
            binding(PROJECT / "research/src/brazil_rv/v2/round5_index.py"),
        ],
        "model_or_heldout_reads": False,
    }
    write_json_atomic(output / "manifest.json", report)
    run["remaining_auxiliaries"] = binding(output / "manifest.json")
    write_json_atomic(pointer_path, run)
    print(
        json.dumps(
            {"completed": "remaining_auxiliaries", "seconds": report["seconds"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
