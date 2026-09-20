"""Independent endpoint and complete-store composition/consumer qualification."""

from decimal import Decimal
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.contract import (
    HORIZONS,
    REGISTERED_PRIMARY_TARGET,
    REGISTERED_PRIMARY_TARGET_MASK,
    TARGET_NEUTRALIZATION_FEATURES,
)
from brazil_rv.v2.corporate_actions import (
    align_verified_action_terms,
    verified_action_terms_from_table,
)
from brazil_rv.v2.data import V2DailyDataset, collate_v2_daily
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.train import _verify_additive_parent_transfer
from audit_corporate_target_inputs import neutral_oracle, PAIRS
from compose_surviving_store import changed, read_layers

PROJECT = Path(__file__).resolve().parents[1]


def main(mode):
    tick = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    out = Path(run["root"]) / "surviving_store"
    plan = json.loads((out / "plan.json").read_text())
    parent = Path(plan["parent"]["root"])
    old_manifest = bound_json(
        dict(
            path=str(parent / "manifest.json"), sha256=plan["parent"]["manifest_sha256"]
        )
    )
    dates, isins = (
        np.load(parent / "date_index.npy"),
        np.load(parent / "isin_index.npy").tolist(),
    )
    cached = {}

    def old(key):
        if key not in cached:
            cached[key] = np.load(
                parent / old_manifest["arrays"][key]["path"], mmap_mode="r"
            )
        return cached[key]

    if mode == "targets":
        (out / "executed_target_qualification.py").write_bytes(
            Path(__file__).read_bytes()
        )
        rows = np.load(out / "target_rows.npy")
        prior = bound_json(run["event_source_composition"])
        event_rows = np.load(prior["rows"]["path"])
        references, checks = {}, {}
        scope = None
        for key in (
            "target_shareholder_valid",
            "target_shareholder_simple_return",
            "target_terminal_wealth",
            "target_terminal_loss",
            "target_price_valid",
            "target_price_simple_return",
        ):
            reference = np.array(old(key)[rows], copy=True)
            reference[np.searchsorted(rows, event_rows)] = np.load(
                prior["target_arrays"][key]["path"]
            )
            value = np.load(out / (key + "_rows.npy"))
            references[key] = reference
            if key == "target_shareholder_valid":
                assert not np.any(reference & ~value)
                scope = value & ~reference
            if value.dtype == bool:
                checks[key] = dict(
                    gains=int((value & ~reference).sum()),
                    losses=int((reference & ~value).sum()),
                )
            np.testing.assert_array_equal(value[~scope], reference[~scope], err_msg=key)
        assert scope is not None
        # Selection of exact endpoint source/action terms was qualified upstream.
        # This walk independently verifies arithmetic on every newly usable outcome.
        actions = align_verified_action_terms(
            verified_action_terms_from_table(
                pl.read_parquet(out / "corporate_actions_verified_terms.parquet")
            ),
            dates,
            isins,
            coverage_resolved=old("action_session_resolved"),
        )
        close = np.round(np.asarray(old("raw_close"), np.float64), 2)
        actual = {
            k: np.load(out / (k + "_rows.npy"))
            for k in (
                "target_terminal_wealth",
                "target_shareholder_simple_return",
                "target_price_simple_return",
            )
        }
        endpoints = []
        for local, name, h in np.argwhere(scope):
            day, end = int(rows[local]), int(rows[local] + HORIZONS[h])
            assert isins[name] in (
                "BRSSBRACNOR1",
                "BRARZZACNOR3",
                "BRALSOACNOR5",
                "BRAZZAACNOR9",
            )
            claim, quantity, cash = int(name), Decimal(1), Decimal(0)
            for t in range(day + 1, end + 1):
                assert actions.session_resolved[t, claim]
                cash += quantity * Decimal(str(actions.cash_per_prior_share[t, claim]))
                quantity *= Decimal(str(actions.shares_per_prior_share[t, claim]))
                claim = int(actions.successor_index[t, claim])
            assert old("observed")[end, claim]
            equity = quantity * Decimal(str(close[end, claim]))
            entry = Decimal(str(close[day, name]))
            for key, value in (
                ("target_terminal_wealth", (equity + cash) / entry),
                ("target_shareholder_simple_return", (equity + cash) / entry - 1),
                ("target_price_simple_return", equity / entry - 1),
            ):
                np.testing.assert_array_equal(
                    actual[key][local, name, h], np.float32(float(value))
                )
            endpoints.append(
                dict(
                    isin=isins[name],
                    entry=str(dates[day]),
                    endpoint=str(dates[end]),
                    horizon=HORIZONS[h],
                    successor=isins[claim],
                    quantity=str(quantity),
                    cash=str(cash),
                    equity=str(equity),
                    entry_price=str(entry),
                )
            )
        write_json_atomic(out / "new_endpoint_arithmetic.json", endpoints)
        result = dict(
            status="passed",
            new_outcomes=len(endpoints),
            new_outcomes_by_isin={
                n: sum(e["isin"] == n for e in endpoints)
                for n in sorted({e["isin"] for e in endpoints})
            },
            prior72_outcome_proofs_reused=True,
            unchanged_raw_outcome_cells=int((~scope).sum() * 6),
            checks=checks,
            oracle=binding(out / "new_endpoint_arithmetic.json"),
            seconds=perf_counter() - tick,
        )
        write_json_atomic(out / "target_qualification.json", result)
        print(json.dumps(result), flush=True)
        return

    assert mode in ("store", "attribution")
    assembly = bound_json(run["composed_store_assembly"])
    contract = bound_json(assembly["contract"])
    root = Path(assembly["store"]["root"])
    m = bound_json(
        dict(
            path=str(root / "manifest.json"),
            sha256=assembly["store"]["manifest_sha256"],
        )
    )
    if mode == "attribution":
        (out / "executed_attribution.py").write_bytes(Path(__file__).read_bytes())
        patches = read_layers(contract["layers"])
        risk = [
            m["feature_names"]["slow"].index(f) for f in TARGET_NEUTRALIZATION_FEATURES
        ]
        affected = []
        for key in (
            "target_valid",
            "target_shareholder_simple_return",
            "target_scale_sigma",
            "slow_values",
            "slow_valid",
        ):
            for ix, _ in patches.get(key, []):
                if key.startswith("slow_"):
                    ix = ix[np.isin(ix[:, 2], risk)]
                affected.extend(ix[:, 0].tolist())
        rows = np.array(sorted(set(affected)), dtype=np.int64)
        grants = sorted(
            {
                int(t + u)
                for t in rows
                for u in range(max(HORIZONS) + 1)
                if t + u < len(dates)
            }
        )
        datasets = [
            V2DailyDataset(
                p,
                rows.tolist(),
                stage="finetune",
                lookback=60,
                target_window_indices=grants,
            )
            for p in (parent, root)
        ]
        effects = []
        for start in range(0, len(rows), 64):
            selected = rows[start : start + 64]
            values = [
                d.store.read(REGISTERED_PRIMARY_TARGET, selected) for d in datasets
            ]
            masks = [
                d.store.read(REGISTERED_PRIMARY_TARGET_MASK, selected) for d in datasets
            ]
            for i, t in enumerate(selected):
                effects.append(
                    dict(
                        date=str(dates[t]),
                        changed=int(changed(values[0][i], values[1][i]).sum()),
                        gains=int((masks[1][i] & ~masks[0][i]).sum()),
                        losses=int((masks[0][i] & ~masks[1][i]).sum()),
                    )
                )
        for d in datasets:
            d.store.close()
        np.save(out / "neutral_affected_rows.npy", rows)
        write_json_atomic(out / "neutral_by_date.json", effects)
        result = dict(
            status="complete_actual_virtual_target_attribution",
            affected_dates=len(rows),
            changes=sum(r["changed"] for r in effects),
            gains=sum(r["gains"] for r in effects),
            losses=sum(r["losses"] for r in effects),
            source=binding(out / "neutral_by_date.json"),
            scope="Actual final V2Store virtual projection versus immutable Natura parent. Reuses completed endpoint/composition/SVD proofs; no dataset samples, neural forward or new source reads.",
            seconds=perf_counter() - tick,
        )
        write_json_atomic(out / "neutral_attribution.json", result)
        print(json.dumps(result), flush=True)
        return
    qualification = out / "qualification"
    qualification.mkdir(exist_ok=False)
    (qualification / "executed.py").write_bytes(Path(__file__).read_bytes())
    patches = read_layers(contract["layers"])
    np.testing.assert_array_equal(dates, np.load(root / "date_index.npy"))
    assert (
        isins == np.load(root / "isin_index.npy").tolist()
        and m["feature_schema_sha256"] == old_manifest["feature_schema_sha256"]
    )

    def expected(key, rows):
        result = np.array(old(key)[rows], copy=True)
        for ix, values in patches.get(key, []):
            keep = np.isin(ix[:, 0], rows)
            local = ix[keep].copy()
            local[:, 0] = np.searchsorted(rows, local[:, 0])
            result[tuple(local.T)] = values[keep]
        return result

    untouched, cells = [], 0
    for key, rec in m["arrays"].items():
        assert (
            rec["shape"] == old_manifest["arrays"][key]["shape"]
            and rec["dtype"] == old_manifest["arrays"][key]["dtype"]
        )
        if key not in patches:
            assert rec["sha256"] == old_manifest["arrays"][key]["sha256"]
            untouched.append(key)
            continue
        array = np.load(root / rec["path"], mmap_mode="r")
        for start in range(0, len(dates), 64):
            rows = np.arange(start, min(len(dates), start + 64))
            np.testing.assert_array_equal(array[rows], expected(key, rows), err_msg=key)
            cells += array[rows].size
    write_json_atomic(
        qualification / "arrays.json",
        dict(
            cells=cells,
            unchanged_arrays=untouched,
            changed_arrays=list(patches),
            mismatches=0,
        ),
    )
    # Combined interactions only: earlier exhaustive per-family consumer proofs are reused.
    sample_days = [
        "2019-08-05",
        "2019-08-06",
        "2019-08-13",
        "2019-08-14",
        "2019-09-02",
        "2019-10-28",
        "2019-10-29",
        "2019-12-17",
        "2023-10-24",
        "2023-10-25",
        "2024-07-31",
        "2024-08-01",
        "2024-08-02",
        "2024-08-05",
        "2024-09-02",
        "2024-10-23",
        "2024-10-24",
        "2024-11-14",
        "2024-12-30",
    ]
    samples = [int(np.searchsorted(dates, np.datetime64(day))) for day in sample_days]
    grants = sorted(
        {t + u for t in samples for u in range(max(HORIZONS) + 1) if t + u < len(dates)}
    )
    groups = [
        k.removeprefix("sidecar_")
        for k in m["feature_names"]
        if k.startswith("sidecar_")
    ]
    dataset = V2DailyDataset(
        root,
        samples,
        stage="finetune",
        enabled_sidecars=groups,
        include_common_state=True,
        lookback=60,
        target_window_indices=grants,
    )
    risk = [m["feature_names"]["slow"].index(f) for f in TARGET_NEUTRALIZATION_FEATURES]
    mapping = pl.read_parquet(
        root / m["tables"]["native_fast_security_mapping"]["path"]
    ).sort("fast_index")
    names = mapping["store_name_index"].to_numpy()
    links = pl.read_parquet(root / m["tables"]["slow_history_links"]["path"])
    packed, virtual_changes, virtual_gains, virtual_losses = 0, 0, 0, 0
    for i, t in enumerate(samples):
        sample = dataset[i]
        batch = collate_v2_daily([sample])
        rows, current = np.arange(t - 59, t + 1), np.array([t])
        route = np.broadcast_to(np.arange(933), (60, 933)).copy()
        known = {
            r["successor_index"]: r
            for r in links.iter_rows(named=True)
            if max(r["effective_index"], r["known_index"]) <= t
        }
        for j in known:
            for h, day in enumerate(rows):
                name = j
                while name in known and day < known[name]["effective_index"]:
                    name = known[name]["predecessor_index"]
                route[h, j] = name
        idx = (np.arange(60)[:, None], route)
        valid = expected("slow_valid", rows)[idx].transpose(1, 0, 2)
        hist = expected("slow_timestep_valid", rows)[idx].T
        z = expected("slow_values", current)[0][:, risk]
        zm = expected("slow_valid", current)[0][:, risk].all(axis=1)
        primary, mask = neutral_oracle(
            expected("target_shareholder_simple_return", current)[0],
            expected("target_valid", current)[0],
            expected("target_scale_sigma", current)[0],
            z,
            zm,
        )
        original_primary, original_mask = neutral_oracle(
            old("target_shareholder_simple_return")[t],
            old("target_valid")[t],
            old("target_scale_sigma")[t],
            old("slow_values")[t][:, risk],
            old("slow_valid")[t][:, risk].all(axis=1),
        )
        virtual_changes += int(changed(primary, original_primary).sum())
        virtual_gains += int((mask & ~original_mask).sum())
        virtual_losses += int((original_mask & ~mask).sum())
        checks = {
            "active_mask": expected("active", current)[0],
            "slow_features": np.where(
                valid, expected("slow_values", rows)[idx].transpose(1, 0, 2), 0
            ),
            "slow_feature_mask": valid,
            "slow_history_mask": hist,
            "slow_feature_age_sessions": np.where(
                hist[..., None],
                expected("slow_age_sessions", rows)[idx].transpose(1, 0, 2),
                -1,
            ),
            "targets": np.where(mask, primary, 0),
            "target_mask": mask,
            "current_features": np.where(
                expected("intraday_valid", current)[0],
                expected("intraday_values", current)[0],
                0,
            ),
            "current_feature_mask": expected("intraday_valid", current)[0],
            "current_feature_age_sessions": expected("intraday_age_sessions", current)[
                0
            ],
        }
        for key, known_key, sample_key, _ in PAIRS[1:]:
            checks[sample_key] = np.where(
                expected(known_key, current)[0], expected(key, current)[0], 0
            )
        for group in groups:
            key = "sidecar_" + group
            v = expected(key + "_valid", current)[0]
            checks.update(
                {
                    key + "_values": np.where(
                        v, expected(key + "_values", current)[0], 0
                    ),
                    key + "_valid": v,
                    key + "_age_sessions": expected(key + "_age_sessions", current)[0],
                }
            )
        selected = expected("fast_present", current)[0, names] & expected(
            "fast_patch_mask", current
        )[0].any(axis=1)
        v = expected("fast_patch_valid", current)[0, selected]
        checks.update(
            fast_name_index=names[selected],
            fast_patch_values=np.where(
                v, expected("fast_patch_values", current)[0, selected], 0
            ),
            fast_patch_valid=v,
            fast_patch_mask=expected("fast_patch_mask", current)[0, selected],
        )
        for key, value in checks.items():
            np.testing.assert_array_equal(
                sample[key], value, err_msg=f"{dates[t]}:{key}"
            )
            np.testing.assert_array_equal(
                batch[key][0].numpy()[: len(value)], value, err_msg="packed:" + key
            )
            packed += value.size * 2
    dataset.store.close()
    causal = []
    for day in ("2019-08-06", "2024-08-01", "2023-10-25"):
        t = int(np.searchsorted(dates, np.datetime64(day)))
        ds = V2DailyDataset(
            root,
            [t],
            stage="finetune",
            enabled_sidecars=groups,
            lookback=60,
            target_window_indices=[t],
        )
        read = ds.store.read
        reads = []

        def bounded(name, selector):
            if not name.startswith("target_"):
                assert np.atleast_1d(np.arange(len(dates))[selector]).max() <= t
                reads.append(name)
            return read(name, selector)

        ds.store.read = bounded
        item = ds[0]
        assert not item["target_mask"].any() and not item["targets"].any()
        causal.append(
            dict(
                date=day, bounded_feature_reads=len(reads), revoked_horizons_empty=True
            )
        )
        ds.store.close()
    try:
        _verify_additive_parent_transfer(
            {"training": {"store": plan["parent"]}},
            {"training": {"store": assembly["store"]}},
            (parent, root),
        )
    except ValueError as e:
        rejection = str(e)
        assert rejection == "parent transfer changed a protected array, table or axis"
    else:
        raise AssertionError("Old weights accepted changed coordinates")
    result = dict(
        status="passed_complete_store_and_combined_consumers",
        assembly=run["composed_store_assembly"],
        array_checks=binding(qualification / "arrays.json"),
        samples=len(samples),
        sample_dates=sample_days,
        names=933,
        history=60,
        packed_cells=packed,
        mismatches=0,
        causal=causal,
        old_parent_rejected=rejection,
        selected_sample_neutral_changes=virtual_changes,
        selected_sample_neutral_gains=virtual_gains,
        selected_sample_neutral_losses=virtual_losses,
        scope="Composition verification; earlier exhaustive per-family proofs reused. No neural forward/scoring/fit.",
        seconds=perf_counter() - tick,
    )
    write_json_atomic(qualification / "manifest.json", result)
    run["composed_store_input_audit"] = binding(qualification / "manifest.json")
    write_json_atomic(pointer, run)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
