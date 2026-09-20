"""Verify new succession targets through the actual bounded CPU consumer."""

from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.contract import HORIZONS, TARGET_NEUTRALIZATION_FEATURES
from brazil_rv.v2.corporate_replay import apply_corporate_replay, load_corporate_replay
from brazil_rv.v2.data import V2DailyDataset, collate_v2_daily
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.feature_spec import FeatureSpec, feature_schema_sha256
from brazil_rv.v2.store import StoreStaging
from audit_corporate_target_inputs import PAIRS, neutral_oracle
from propagate_corporate_targets import FIELDS
from verify_corporate_replay import inputs_on_axes

PROJECT = Path(__file__).resolve().parents[1]


def main():
    tick = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text(encoding="utf8"))
    source = bound_json(run["event_source_composition"])
    parent = source["parent"]
    store = Path(parent["root"])
    m = bound_json(
        dict(path=str(store / "manifest.json"), sha256=parent["manifest_sha256"])
    )
    out = Path(run["event_source_composition"]["path"]).parent / "qualification"
    out.mkdir(exist_ok=False)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    for path in (
        PROJECT / "ops/audit_corporate_target_inputs.py",
        PROJECT / "research/src/brazil_rv/v2/store.py",
        PROJECT / "research/src/brazil_rv/v2/data.py",
    ):
        (out / path.name).write_bytes(path.read_bytes())
    dates, isins = np.load(store / "date_index.npy"), np.load(store / "isin_index.npy")
    requested = np.load(source["rows"]["path"])
    assert len(isins) == 933 and str(dates[-1]) == "2024-12-30"
    with np.load(source["deltas"]["path"]) as z:
        patches = {
            k.removesuffix("__indices"): (
                z[k].copy(),
                z[k.removesuffix("__indices") + "__values"].copy(),
            )
            for k in z.files
            if k.endswith("__indices")
        }
    arrays = {
        key: np.load(store / m["arrays"][key]["path"], mmap_mode="r")
        for key in (
            *FIELDS.values(),
            "active",
            "slow_values",
            "slow_valid",
            "slow_age_sessions",
            "slow_timestep_valid",
            "target_scale_sigma",
        )
    }
    links = pl.read_parquet(store / m["tables"]["slow_history_links"]["path"])
    specs = [
        s
        for s in m["metadata"]["feature_schema"]["specifications"]
        if s["family"] == "slow"
    ]
    risk = [m["feature_names"]["slow"].index(k) for k in TARGET_NEUTRALIZATION_FEATURES]
    checked = 0
    primary_changes = primary_gains = primary_losses = 0
    views = []
    neutral_rows, neutral_masks = [], []
    for day in requested:
        first, stop = int(day) - 59, min(int(day) + 11, len(dates))
        rows = np.arange(first, stop)
        view = out / str(dates[day])
        expected = {}
        with StoreStaging(view, dates=dates[rows], isins=isins) as staging:
            for key, array in arrays.items():
                value = array[rows].copy()
                if key in patches:
                    ix, values = patches[key]
                    keep = (ix[:, 0] >= first) & (ix[:, 0] < stop)
                    local = ix[keep].copy()
                    local[:, 0] -= first
                    value[tuple(local.T)] = values[keep]
                staging.write_array(key, value)
                expected[key] = value[59]
            staging.seal(
                feature_names={"slow": m["feature_names"]["slow"]},
                sources=[run["event_source_composition"]],
                tables={
                    "slow_history_links": links.with_columns(
                        pl.col("effective_index") - first, pl.col("known_index") - first
                    )
                },
                metadata={
                    "purpose": "Bounded succession target audit; not a complete accepted refit store",
                    "feature_schema": {
                        "minimum_rank_names": 20,
                        "specifications": specs,
                        "sha256": feature_schema_sha256(
                            [FeatureSpec(**s) for s in specs]
                        ),
                    },
                },
            )
        dataset = V2DailyDataset(
            view,
            [59],
            stage="finetune",
            lookback=60,
            include_fast=False,
            include_intraday=False,
            target_window_indices=np.arange(59, len(rows)),
        )
        sample = dataset[0]
        batch = collate_v2_daily([sample])
        assert sample["slow_features"].shape[:2] == (933, 60)
        chars = expected["slow_values"][:, risk]
        valid_chars = expected["slow_valid"][:, risk].all(axis=1) & np.isfinite(
            chars
        ).all(axis=1)
        primary, primary_mask = neutral_oracle(
            expected["target_shareholder_simple_return"],
            expected["target_valid"],
            expected["target_scale_sigma"],
            chars,
            valid_chars,
        )
        old_primary, old_mask = neutral_oracle(
            arrays["target_shareholder_simple_return"][day],
            arrays["target_valid"][day],
            expected["target_scale_sigma"],
            chars,
            valid_chars,
        )
        primary_changes += int((primary != old_primary).sum())
        primary_gains += int((primary_mask & ~old_mask).sum())
        primary_losses += int((old_mask & ~primary_mask).sum())
        neutral_rows.append(primary)
        neutral_masks.append(primary_mask)
        for value_key, mask_key, dest, mask_dest in PAIRS:
            mask = (primary_mask if dest == "targets" else expected[mask_key]).copy()
            for j, horizon in enumerate(HORIZONS):
                if 59 + horizon >= len(rows):
                    mask[:, j] = False
            value = np.where(
                mask, primary if dest == "targets" else expected[value_key], 0
            )
            for key, want in ((dest, value), (mask_dest, mask)):
                np.testing.assert_array_equal(
                    sample[key], want, err_msg=f"{dates[day]}:{key}"
                )
                np.testing.assert_array_equal(batch[key][0].numpy(), want)
                checked += want.size * 2
        dataset.store.close()
        restricted = V2DailyDataset(
            view,
            [59],
            stage="finetune",
            lookback=60,
            include_fast=False,
            include_intraday=False,
            target_window_indices=[59],
        )
        read = restricted.store.read
        read_count = [0]

        def causal_read(name, selector):
            if not name.startswith("target_"):
                assert np.atleast_1d(np.arange(len(rows))[selector]).max() <= 59
                read_count[0] += 1
            return read(name, selector)

        restricted.store.read = causal_read
        denied = restricted[0]
        for _, _, dest, mask_dest in PAIRS:
            assert not denied[mask_dest].any() and not denied[dest].any()
        restricted.store.close()
        views.append(
            dict(
                date=str(dates[day]),
                manifest=binding(view / "manifest.json"),
                bounded_feature_reads=read_count[0],
                revoked_endpoints_empty=True,
            )
        )
    np.save(out / "neutral_target_rows.npy", np.stack(neutral_rows))
    np.save(out / "neutral_valid_rows.npy", np.stack(neutral_masks))
    primary_terms, calendar = load_corporate_replay(
        source["primary_terms"]["path"], source["primary_terms"]["sha256"]
    )
    previous_terms, _ = load_corporate_replay(
        source["previous_terms"]["path"], source["previous_terms"]["sha256"]
    )
    base = inputs_on_axes(
        Path(primary_terms["store"]["root"]),
        calendar,
        np.arange(933),
        np.arange(len(calendar)),
    )
    a = apply_corporate_replay(
        base, previous_terms, calendar, source["previous_terms"]["sha256"]
    )
    b = apply_corporate_replay(
        base, primary_terms, calendar, source["primary_terms"]["sha256"]
    )
    identities = {}
    for key in ("share_distributions", "action_settlements", "loan_cash_settlements"):
        left, right = getattr(a, key), getattr(b, key)
        assert len(left) == len(right)
        for x, y in zip(left, right, strict=True):
            assert replace(x, source=y.source) == y
        identities[key] = len(right)
    result = dict(
        status="qualified_sparse_succession_data_and_identical_account_terms_not_complete_store_or_stage_a",
        composition=run["event_source_composition"],
        samples=len(views),
        names=933,
        history=60,
        target_consumer_cells=checked,
        mismatches=0,
        virtual_primary_numeric_changes=primary_changes,
        virtual_primary_gains=primary_gains,
        virtual_primary_losses=primary_losses,
        account_term_identity=identities,
        views=views,
        neutral_targets=binding(out / "neutral_target_rows.npy"),
        neutral_valid=binding(out / "neutral_valid_rows.npy"),
        feature_scope="Actual full-name/full60 slow history consumer with unchanged risk coordinates; native/scalar/auxiliary fields excluded because this target-only amendment does not modify their inputs. Complete final-store consumers remain an assembly step.",
        old_fits_or_stores_changed=False,
        model_forward_or_book_replay=False,
        seconds=perf_counter() - tick,
    )
    write_json_atomic(out / "manifest.json", result)
    run["event_composition_qualification"] = binding(out / "manifest.json")
    write_json_atomic(pointer, run)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "samples",
                    "target_consumer_cells",
                    "virtual_primary_numeric_changes",
                    "virtual_primary_gains",
                    "virtual_primary_losses",
                    "account_term_identity",
                    "seconds",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
