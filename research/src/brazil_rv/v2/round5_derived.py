"""Materialize causal magnitude, sector and market families from sealed sources."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from .artifacts import sha256_file, write_json_atomic
from .bova11 import load_bova11_series
from .contract import DEVELOPMENT_END, FINETUNE_START, PRETRAIN_END
from .features import exact_log_return
from .research_rounds import _git_identity
from .round5_exposures import (
    EXPOSURES,
    FEATURE_NAMES as MARKET_NAMES,
    SECTOR_FEATURE_NAMES,
    SERIES,
    exposure_panel,
    market_shocks,
    sector_relative_panel,
    shock_axes,
)
from .round5_magnitude import FEATURE_NAMES as MAGNITUDE_NAMES, magnitude_panel
from .round5_market import OBSERVATION_SCHEMA, vix_close
from .store import open_store_for_samples, peak_rss_bytes


def verified(record: dict) -> Path:
    path = Path(record["path"])
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"Round-5 derived source identity differs: {path}")
    return path


def bind(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def identity_axes(identity, dates, isins):
    """Use only the bridge's already decision-dated rows, without another carry."""
    shape = len(dates), len(isins)
    sectors, issuers = np.full(shape, "", object), np.full(shape, "", object)
    days, names = (
        {d: i for i, d in enumerate(dates)},
        {n: i for i, n in enumerate(isins)},
    )
    for row in identity.iter_rows(named=True):
        day, name = days.get(row["date"]), names.get(row["isin"])
        if day is None or name is None:
            continue
        if row["identity_known_date"] > row["date"]:
            raise ValueError("sector identity precedes its publication")
        if row["sector"] and (
            row["sector_known_date"] is None or row["sector_known_date"] > row["date"]
        ):
            raise ValueError("sector code precedes its receipt-known translation")
        issuer = row["cnpj"][:8] + ":" + row["cvm_code"]
        if issuers[day, name] and issuers[day, name] != issuer:
            raise ValueError("ambiguous contemporaneous sector issuer")
        sectors[day, name] = row["sector"] or ""
        issuers[day, name] = issuer
    return sectors, issuers


def beta_source_age(wealth, wealth_valid, bova, beta_valid):
    """Latest adjacent return pair actually used by the t-1 economic beta fit."""
    pairs = np.zeros_like(wealth_valid)
    pairs[1:] = (
        wealth_valid[1:]
        & wealth_valid[:-1]
        & np.isfinite(wealth[1:])
        & np.isfinite(wealth[:-1])
        & (wealth[1:] >= 0)
        & (wealth[:-1] > 0)
        & (
            np.isfinite(bova[1:])
            & np.isfinite(bova[:-1])
            & (bova[1:] > 0)
            & (bova[:-1] > 0)
        )[:, None]
    )
    rows = np.arange(len(wealth))[:, None]
    latest = np.maximum.accumulate(np.where(pairs, rows, -1), axis=0)
    age = np.full(wealth.shape, -1, np.float32)
    age[1:] = np.where(beta_valid[1:], rows[1:] - latest[:-1], -1)
    return age


def cdi_axis(early: Path, accepted: Path, dates: list[date]):
    early_rows = {
        datetime.strptime(row["data"], "%d/%m/%Y").date(): float(row["valor"]) / 100
        for row in json.loads(early.read_text(encoding="utf-8"))
    }
    accepted_rows = dict(
        pl.read_parquet(accepted).select("trade_date", "daily_cdi_rate").iter_rows()
    )
    overlap = early_rows.keys() & accepted_rows.keys()
    difference = max(
        (abs(early_rows[d] - accepted_rows[d]) for d in overlap), default=0
    )
    if difference > 1e-12:
        raise ValueError(
            "new CDI history conflicts with the accepted overlapping source"
        )
    combined = {**early_rows, **accepted_rows}
    result = np.array([combined.get(day, np.nan) for day in dates])
    return result, {
        "overlap_rows": len(overlap),
        "maximum_absolute_difference": difference,
        "missing_development_sessions": int((~np.isfinite(result)).sum()),
    }


def write_family(
    output,
    family,
    names,
    values,
    valid,
    ages,
    dates,
    isins,
    active,
    sources,
    code,
    proof,
    extra=None,
):
    """Archive physical values; the one store writer applies registered transforms."""
    root = output / family
    root.mkdir(parents=True, exist_ok=False)
    known = valid & np.isfinite(values) & active[..., None]
    d, n = np.where(known.any(axis=-1))
    columns = {
        "date": pl.Series(np.asarray(dates, dtype="datetime64[D]")[d]),
        "isin": np.asarray(isins)[n],
    }
    for column, name in enumerate(names):
        mask = known[d, n, column]
        columns[name] = pl.Series(np.asarray(values[d, n, column], np.float32)).set(
            pl.Series(~mask), None
        )
        columns[f"{name}_age_sessions"] = np.where(mask, ages[d, n, column], -1)
    data = root / "features.parquet"
    pl.DataFrame(columns).write_parquet(data)
    years = np.array([d.year for d in dates])
    coverage = []
    for year in np.unique(years):
        selected = years == year
        coverage.append(
            {
                "year": int(year),
                "active_name_days": int(active[selected].sum()),
                "valid_by_feature": dict(
                    zip(names, known[selected].sum((0, 1)).tolist(), strict=True)
                ),
            }
        )
    proof_path = root / "availability_proof.json"
    write_json_atomic(
        proof_path,
        {
            "causality_passed": True,
            "first_available_decision_passed": True,
            "data_sha256": sha256_file(data),
            "joined_fixture": proof,
            "source_code": code,
        },
    )
    manifest = {
        "family": family,
        "feature_names": list(names),
        "source_code": code,
        "sources": sources,
        "data": bind(data),
        "availability_proof": bind(proof_path),
        "coverage": coverage,
        "first_usable_date": str(dates[int(d.min())]) if len(d) else None,
        "last_usable_date": str(dates[int(d.max())]) if len(d) else None,
        "consumer_end": DEVELOPMENT_END.isoformat(),
        "fit_dependent_clipping": False,
        "physical_values_before_store_transform": True,
        **(extra or {}),
    }
    write_json_atomic(root / "manifest.json", manifest)
    return {
        "family": family,
        "feature_names": list(names),
        "status": "admitted",
        "source_manifest": bind(root / "manifest.json"),
        "data": bind(data),
        "availability_proof": bind(proof_path),
    }


def joined_clock_fixture():
    """Exercise actual shock assembly and calendar admission, not timestamp labels."""
    dates = [date(2024, 7, 1) + timedelta(days=i) for i in range(12)]
    empty = pl.DataFrame(
        schema={
            "series": pl.String,
            "reference_date": pl.Date,
            "previous_date": pl.Date,
            "available_at": pl.Datetime("us", "UTC"),
            "log_return": pl.Float64,
        }
    )
    checks = []
    for hour, expected in ((16, 6), (21, 7)):
        rows = [
            {
                "series": "ptax_brl_per_usd",
                "reference_date": day,
                "available_at": datetime.combine(day, datetime.min.time(), UTC).replace(
                    hour=hour
                ),
                "value": 5 + i * 0.01,
                "source_file": "fixture",
            }
            for i, day in enumerate(dates)
        ]
        before = shock_axes(
            market_shocks(pl.DataFrame(rows, schema=OBSERVATION_SCHEMA), empty), dates
        )["fx"][0]
        rows[6]["value"] += 0.1
        after = shock_axes(
            market_shocks(pl.DataFrame(rows, schema=OBSERVATION_SCHEMA), empty), dates
        )["fx"][0]
        np.testing.assert_array_equal(before[:expected], after[:expected])
        changed = np.flatnonzero(~np.isclose(before[:, 0], after[:, 0], equal_nan=True))
        assert int(changed[0]) == expected
        checks.append(
            {"fixing_utc_hour": hour, "first_changed_decision_index": expected}
        )
    return {
        "market_first_decision_mutations": checks,
        "additional_joined_tests": [
            "test_v2_round5_market.py",
            "test_v2_round5_exposures.py",
            "test_v2_round5_magnitude.py",
            "test_v2_round5_derived.py",
        ],
    }


def build(plan_path: Path, output: Path):
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    code = _git_identity()
    paths = {name: verified(record) for name, record in plan["inputs"].items()}
    test_report = ET.parse(paths["joined_test_report"]).getroot()
    if test_report.findall(".//failure") or test_report.findall(".//error"):
        raise ValueError("joined family acceptance tests failed")
    if not test_report.findall(".//testcase"):
        raise ValueError("joined family acceptance report has no executed tests")
    base = paths["base_manifest"].parent
    axis = np.load(base / "date_index.npy", allow_pickle=False)
    if axis.max() > np.datetime64(DEVELOPMENT_END):
        raise PermissionError("derived family source contains held-out consumer dates")
    rows = np.arange(len(axis))
    samples = rows[
        (axis <= np.datetime64(PRETRAIN_END)) | (axis >= np.datetime64(FINETUNE_START))
    ]
    store, access = open_store_for_samples(
        base, samples, purpose="training", history_lookbacks=60, history_end_offsets=0
    )
    dates, isins = axis.astype(object).tolist(), store.isins
    output.mkdir(parents=True, exist_ok=False)
    sources = {
        "plan": bind(plan_path),
        "inputs": plan["inputs"],
        "access": access.payload(),
    }
    fixture = joined_clock_fixture()
    admissions = []
    try:
        active = store.read("active", rows)
        wealth = store.read("shareholder_wealth_close", rows)
        wealth_valid = store.read("shareholder_wealth_valid", rows)
        parent_valid = store.read("slow_valid", rows)
        slow_names = tuple(store.manifest["feature_names"]["slow"])
        bova_root = paths["bova_manifest"].parent
        bova = np.asarray(
            load_bova11_series(
                bova_root,
                expected_manifest_sha256=plan["inputs"]["bova_manifest"]["sha256"],
                canonical_dates=dates,
            ).close_by_session
        )
        beta_manifest = json.loads(paths["beta_manifest"].read_text(encoding="utf-8"))
        beta_arrays = {}
        for name, record in beta_manifest["arrays"].items():
            path = verified(
                {
                    "path": str(paths["beta_manifest"].parent / record["path"]),
                    "sha256": record["sha256"],
                }
            )
            beta_arrays[name] = np.load(path, allow_pickle=False)
        np.testing.assert_array_equal(beta_arrays["date_index"], axis)
        np.testing.assert_array_equal(beta_arrays["isin_index"], isins)
        beta_age = beta_source_age(
            wealth, wealth_valid, bova, beta_arrays["hedge_beta_valid"]
        )
        values, valid, ages = magnitude_panel(
            wealth_open=store.read("shareholder_wealth_open", rows),
            wealth_high=store.read("shareholder_wealth_high", rows),
            wealth_low=store.read("shareholder_wealth_low", rows),
            wealth_close=wealth,
            wealth_valid=wealth_valid,
            volume_brl=store.read("volume_brl", rows),
            activity_valid=store.read("activity_valid", rows),
            daily_feature_valid=parent_valid,
            slow_age_sessions=store.read("slow_age_sessions", rows),
            slow_feature_names=slow_names,
            economic_beta=beta_arrays["hedge_beta"],
            economic_beta_valid=beta_arrays["hedge_beta_valid"],
            economic_beta_age_sessions=beta_age,
        )
        admissions.append(
            write_family(
                output,
                "magnitudes",
                MAGNITUDE_NAMES,
                values,
                valid,
                ages,
                dates,
                isins,
                active,
                sources,
                code,
                fixture,
                {
                    "availability_rule": "daily parent support and physical wealth/activity end t-1; economic beta already decision-dated t",
                    "beta_age": "sessions since latest adjacent observed wealth/BOVA pair used by beta",
                },
            )
        )
        del values, valid, ages, beta_arrays, beta_age

        sectors, issuers = identity_axes(
            pl.read_parquet(paths["identity"]), dates, isins
        )
        sector_inputs = np.zeros((*active.shape, 3))
        sector_mask = np.zeros_like(sector_inputs, bool)
        for column, horizon in enumerate((5, 21, 252)):
            values, valid = exact_log_return(
                wealth, horizon, shareholder_wealth_valid=wealth_valid
            )
            parent_field = f"log_return_{horizon}"
            if column == 2:
                short, short_valid = exact_log_return(
                    wealth, 21, shareholder_wealth_valid=wealth_valid
                )
                values -= short
                valid &= short_valid
                parent_field = "momentum_12_1"
            sector_inputs[1:, :, column] = values[:-1]
            sector_mask[1:, :, column] = (
                valid[:-1] & parent_valid[1:, :, slow_names.index(parent_field)]
            )
        values, valid = sector_relative_panel(
            sector_inputs, sector_mask, sectors, issuers, active
        )
        ages = np.where(valid, 1, -1).astype(np.float32)
        admissions.append(
            write_family(
                output,
                "sector",
                SECTOR_FEATURE_NAMES,
                values,
                valid,
                ages,
                dates,
                isins,
                active,
                sources,
                code,
                fixture,
                {
                    "availability_rule": "own-receipt FCA classification and receipt-known numeric-code translation; exact wealth returns through t-1; no future code evidence",
                    "sector_vintage": "sector_known_date gates classification; missing or ambiguous current translation stays missing",
                    "age_rule": "age1 for the completed return endpoint; classification and translation receipt dates remain separate metadata, never reset the market measurement age",
                    "known_sector_active_name_days": int(
                        (active & (sectors != "")).sum()
                    ),
                },
            )
        )
        del values, valid, ages, sector_inputs, sector_mask

        levels = pl.read_parquet(paths["market_levels"]).with_columns(
            pl.when(pl.col("series") == "vix_close")
            .then(
                pl.col("reference_date").map_elements(
                    vix_close, return_dtype=pl.Datetime("us", "UTC")
                )
            )
            .otherwise(pl.col("available_at"))
            .alias("available_at")
        )
        levels = pl.concat(
            [levels, pl.read_parquet(paths["di_levels"])], how="diagonal_relaxed"
        )
        returns = pl.concat(
            [
                pl.read_parquet(paths["futures_returns"]),
                pl.read_parquet(paths["us_returns"]),
            ],
            how="diagonal_relaxed",
        )
        for frame in (levels, returns):
            if frame["reference_date"].max() > DEVELOPMENT_END:
                raise PermissionError("market feature source contains held-out rows")
        # Explicit B3 calendar prevents a missing DI vertex becoming a fake one-day move.
        calendars = {name: dates for name in SERIES if name.startswith("br_di_")}
        shocks = market_shocks(levels, returns, level_calendars=calendars)
        shocks.write_parquet(output / "market_shocks.parquet")
        axes = shock_axes(shocks, dates)
        cdi, cdi_audit = cdi_axis(paths["cdi_early"], paths["cdi_accepted"], dates)
        pl.DataFrame({"date": dates, "daily_cdi_rate": cdi}).write_parquet(
            output / "cdi.parquet"
        )
        equity_return, return_mask = exact_log_return(
            wealth, 1, shareholder_wealth_valid=wealth_valid
        )
        # Preserve the parent's causal identity/action support for historical pairs.
        return_mask[:-1] &= parent_valid[1:, :, slow_names.index("log_return_1")]
        return_mask[-1] = False
        equity_return -= np.log1p(cdi)[:, None]
        shape = (*active.shape, len(MARKET_NAMES))
        values, valid, ages = (
            np.zeros(shape, np.float32),
            np.zeros(shape, bool),
            np.full(shape, -1, np.float32),
        )
        for factor, (current, source_ages, _, _) in axes.items():
            for column, horizon in enumerate((1, 5)):
                field = MARKET_NAMES.index(f"shock_{factor}_{horizon}")
                known = np.isfinite(current[:, column])
                values[:, :, field] = np.where(known, current[:, column], 0)[:, None]
                valid[:, :, field] = known[:, None]
                ages[:, :, field] = source_ages[:, column, None]
        distributions = []
        for name, factor in EXPOSURES.items():
            current, shock_age, historical, public_index = axes[factor]
            beta, mask, age = exposure_panel(
                equity_return, return_mask, historical, public_index, sectors, issuers
            )
            field = MARKET_NAMES.index(f"exposure_{name}")
            values[..., field], valid[..., field], ages[..., field] = beta, mask, age
            sample = beta[mask & active]
            distributions.append(
                {
                    "exposure": name,
                    "valid_active_name_days": int(sample.size),
                    "quantiles_01_50_99": np.quantile(
                        sample, [0.01, 0.5, 0.99]
                    ).tolist()
                    if sample.size
                    else None,
                }
            )
            for column, horizon in enumerate((1, 5)):
                field = MARKET_NAMES.index(f"exposure_{name}_times_shock_{horizon}")
                known = mask & np.isfinite(current[:, column, None])
                values[..., field] = np.where(known, beta * current[:, column, None], 0)
                valid[..., field] = known
                ages[..., field] = np.where(
                    known, np.maximum(age, shock_age[:, column, None]), -1
                )
        admissions.append(
            write_family(
                output,
                "cross_market",
                MARKET_NAMES,
                values,
                valid,
                ages,
                dates,
                isins,
                active,
                sources,
                code,
                fixture,
                {
                    "availability_rule": "first15:45 at/after actual market fixing; no archive upload lag; prior120 regression excludes current equity outcome",
                    "shocks": bind(output / "market_shocks.parquet"),
                    "cdi": bind(output / "cdi.parquet"),
                    "cdi_overlap_audit": cdi_audit,
                    "exposure_distributions": distributions,
                    "unavailable_fields": plan["unavailable_market_fields"],
                    "source_gaps": json.loads(
                        shocks.group_by("factor")
                        .agg(
                            pl.len(),
                            pl.col("shock_1").null_count().alias("null_1"),
                            pl.col("shock_5").null_count().alias("null_5"),
                        )
                        .write_json()
                    ),
                },
            )
        )
        if peak_rss_bytes() > 8 * 1024**3:
            raise MemoryError("Round-5 derived build exceeded8GiB")
        result = {
            "sources": sources,
            "code": code,
            "families": admissions,
            "peak_rss_bytes": peak_rss_bytes(),
        }
        write_json_atomic(output / "manifest.json", result)
        return result
    finally:
        store.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.plan, args.output)
    print(
        json.dumps(
            {
                "families": [x["family"] for x in result["families"]],
                "peak_rss_bytes": result["peak_rss_bytes"],
            }
        )
    )


if __name__ == "__main__":
    main()
