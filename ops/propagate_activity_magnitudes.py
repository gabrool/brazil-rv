"""Propagate printed activity into physical magnitude and flow interactions.

Qualify the preceding rename producer's volume dtype without rerunning its
passed sector, volatility or regression work. All earlier artifacts are retained.
"""

from datetime import date
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json, source_families
from brazil_rv.v2.features import _rolling_stat
from brazil_rv.v2.foreign_flow import decision_panel
from brazil_rv.v2.round5_derived import verified
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
    store = Path(pointer["store"]["root"])
    manifest = bound_json(
        {
            "path": str(store / "manifest.json"),
            "sha256": pointer["store"]["manifest_sha256"],
        }
    )
    families = source_families(manifest)
    rename = bound_json(run["rename_history_propagation"])
    previous = bound_json(run["rename_dependents"])
    activity = bound_json(run["goll_activity_admission"])
    output = Path(run["root"]) / "activity_magnitude_propagation"
    output.mkdir(exist_ok=False)
    (output / "executed_reproducer.py").write_bytes(Path(__file__).read_bytes())
    dates = np.load(store / "date_index.npy")
    assert dates[-1] <= np.datetime64("2024-12-31")
    sessions = dates.astype(object).tolist()
    isins = np.load(store / "isin_index.npy").tolist()
    old = pl.read_parquet(verified(previous["artifacts"]["magnitudes"]))
    links = pl.read_parquet(verified(rename["history_mapping"])).to_dicts()
    volume = np.load(store / "volume_brl.npy", mmap_mode="r")
    active = np.load(verified(rename["arrays"]["active"]), mmap_mode="r")
    valid = np.load(store / "activity_valid.npy", mmap_mode="r")
    corrected = []
    for link in links:
        a, b = link["predecessor_index"], link["successor_index"]
        first = link["effective_index"]
        start = first - 20
        # Keep a strided reduction axis as in the original all-name producer.
        # A contiguous one-column Float32 sum selects NumPy's pairwise path.
        cols = [b, (b + 1) % len(isins), (b + 2) % len(isins)]
        v = np.ascontiguousarray(volume[start:, cols])
        mask = np.ascontiguousarray(valid[start:, cols])
        v[:20, 0] = volume[start:first, a]
        mask[:20, 0] = valid[start:first, a]
        mean, known = _rolling_stat(np.where(mask, v, np.nan), 20, "mean", minimum=20)
        with np.errstate(divide="ignore", invalid="ignore"):
            value = np.log(mean).astype(np.float32)[:, 0]
        for t in range(first, len(dates)):
            local = t - start - 1
            if active[t, b] and known[local, 0] and mean[local, 0] > 0:
                corrected.append(
                    {
                        "date": sessions[t],
                        "isin": isins[b],
                        "replacement": float(value[local]),
                    }
                )
    # The historical physical family consumed stored Float32 volume. Preserve
    # that type while changing exactly the independently recovered source row.
    t, n = activity["date_index"], activity["name_index"]
    start, end = t - 20, t + 21
    cols = [n, (n + 1) % len(isins), (n + 2) % len(isins)]
    v = np.ascontiguousarray(volume[start:end, cols])
    mask = np.ascontiguousarray(valid[start:end, cols])
    before, before_valid = _rolling_stat(
        np.where(mask, v, np.nan), 20, "mean", minimum=20
    )
    baseline = (
        pl.read_parquet(verified(families["magnitudes"]["data"]))
        .filter(
            (pl.col("isin") == isins[n])
            & pl.col("date").is_between(sessions[t + 1], sessions[t + 20])
        )
        .sort("date")
    )
    np.testing.assert_array_equal(
        np.log(before[20:40, 0]).astype(np.float32),
        baseline["log_traded_value_20"].to_numpy(),
    )
    assert before_valid[20:40].all()
    v[20, 0] = activity["printed_activity"]["volume_brl"]
    mean, known = _rolling_stat(np.where(mask, v, np.nan), 20, "mean", minimum=20)
    for i in range(20):
        assert known[20 + i, 0]
        corrected.append(
            {
                "date": sessions[t + 1 + i],
                "isin": isins[n],
                "replacement": float(np.float32(np.log(mean[20 + i, 0]))),
            }
        )
    updates = pl.DataFrame(corrected).with_columns(
        pl.col("replacement").cast(pl.Float32)
    )
    joined = old.join(updates, on=KEYS, how="left")
    changed = joined.filter(
        pl.col("replacement").is_not_null()
        & ~pl.col("replacement").eq_missing(pl.col("log_traded_value_20"))
    )
    changed.write_parquet(output / "magnitude_changes.parquet")
    new = joined.with_columns(
        pl.coalesce("replacement", "log_traded_value_20").alias("log_traded_value_20")
    ).drop("replacement")
    protected = [c for c in old.columns if c != "log_traded_value_20"]
    assert_frame_equal(new.select(protected), old.select(protected), check_exact=True)
    new.write_parquet(output / "magnitudes.parquet")
    cross = pl.read_parquet(verified(previous["artifacts"]["cross_market"]))
    cross_source = bound_json(families["cross_market"]["source_manifest"])
    flow_manifest = bound_json(cross_source["sources"]["foreign_flow"])
    observations = []
    for r in flow_manifest["results"]:
        if "observation" in r:
            obs = dict(r["observation"])
            for k in ["reference_date", "publication_date"]:
                obs[k] = date.fromisoformat(obs[k])
            observations.append(obs)
    names, values, masks, ages, _ = decision_panel(observations, sessions)
    flow = pl.DataFrame(
        {
            "date": sessions,
            **{
                n + "_source": pl.Series(values[:, i]).set(
                    pl.Series(~masks[:, i]), None
                )
                for i, n in enumerate(names)
                if n in ["foreign_flow_1", "foreign_flow_5"]
            },
        }
    )
    target = (
        cross.join(changed.select(KEYS), on=KEYS)
        .join(new.select(KEYS + ["log_traded_value_20"]), on=KEYS)
        .join(flow, on="date")
    )
    for h in [1, 5]:
        target = target.with_columns(
            (pl.col(f"foreign_flow_{h}_source") * pl.col("log_traded_value_20"))
            .cast(pl.Float32)
            .alias(f"foreign_flow_{h}_times_log_volume_mean_20")
        )
    result = pl.concat(
        [
            cross.join(changed.select(KEYS), on=KEYS, how="anti"),
            target.select(cross.columns),
        ]
    ).sort(KEYS)
    protected = [
        c
        for c in cross.columns
        if c not in [f"foreign_flow_{h}_times_log_volume_mean_20" for h in [1, 5]]
    ]
    assert_frame_equal(
        result.select(protected), cross.select(protected).sort(KEYS), check_exact=True
    )
    result.write_parquet(output / "cross_market.parquet")
    ti, ni = np.where(active)
    keys = pl.DataFrame({"date": dates[ti], "isin": np.asarray(isins)[ni]})
    write_json_atomic(
        output / "manifest.json",
        {
            "status": "passed_activity_magnitude_and_precision_qualification",
            "inputs": {
                k: run[k] for k in ["rename_dependents", "goll_activity_admission"]
            },
            "physical_activity_dtype": str(volume.dtype),
            "changed_magnitude_rows": len(changed),
            "changed_goll_rows": len(changed.filter(pl.col("isin") == isins[n])),
            "all_other_values_masks_ages_exact": True,
            "effects": {
                "magnitudes": compare(old, new, keys),
                "cross_market": compare(cross, result, keys),
            },
            "artifacts": {p.stem: binding(p) for p in output.glob("*.parquet")},
            "code": binding(Path(__file__)),
            "seconds": perf_counter() - started,
            "limits": [
                "Qualified only log-volume magnitude and its flow interactions; passed sector/exposure regressions not repeated.",
                "No source price, universe, accepted store or old checkpoint changed.",
            ],
        },
    )
    print(
        json.dumps(
            {
                "stage": "complete",
                "changed_magnitude_rows": len(changed),
                "seconds": perf_counter() - started,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
