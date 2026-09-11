"""Bounded Round-6 extensions of the accepted physical family archives."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from .artifacts import sha256_file, write_json_atomic
from .contract import DEVELOPMENT_END, FINETUNE_START, PRETRAIN_END
from .bova11 import load_bova11_series
from .cross_market_returns import admit_brent, relative_return_panel
from .features import exact_log_return
from .foreign_flow import decision_panel
from .research_rounds import _git_identity
from .round5_b3 import lending_decision_features
from .round5_derived import bind, identity_axes, verified, write_family
from .round5_exposures import FEATURE_NAMES, exposure_panel, market_shocks, shock_axes
from .round5_store import align_family
from .store import open_store_for_samples, peak_rss_bytes

RATE_FIELDS = ("loan_rate", "loan_rate_change_5", "new_loan_volume_surprise")


def replace_lending_rates(old, rates, balances, sessions, isins, volume):
    replacement = lending_decision_features(balances, rates, sessions, isins, volume)
    columns = [c for f in RATE_FIELDS for c in (f, f + "_age_sessions")]
    return (
        old.drop(columns)
        .join(
            replacement.select("date", "isin", *columns),
            on=["date", "isin"],
            how="full",
            coalesce=True,
        )
        .select(old.columns)
        .sort("date", "isin")
    )


def extend_lending(plan_path: Path, output: Path) -> dict:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    paths = {name: verified(value) for name, value in plan["inputs"].items()}
    code = _git_identity()
    source = paths["base_manifest"].parent
    axis = np.load(source / "date_index.npy", allow_pickle=False)
    if axis.max() > np.datetime64(DEVELOPMENT_END):
        raise PermissionError("Round-6 base enters held-out history")
    rows = np.arange(len(axis))
    samples = rows[
        (axis <= np.datetime64(PRETRAIN_END)) | (axis >= np.datetime64(FINETUNE_START))
    ]
    store, access = open_store_for_samples(
        source, samples, purpose="training", history_lookbacks=60, history_end_offsets=0
    )
    try:
        sessions, isins = axis.astype(object).tolist(), store.isins
        active = store.read("active", rows)
        volume = store.read("volume_brl", rows)
        rates = pl.read_parquet(paths["rates"])
        balances = pl.read_parquet(paths["balances"])
        old = pl.read_parquet(paths["old_lending"])
        for frame, field in (
            (rates, "available_date"),
            (balances, "available_date"),
            (old, "date"),
        ):
            if frame[field].max() > DEVELOPMENT_END:
                raise PermissionError("lending source enters held-out history")
        result = replace_lending_rates(old, rates, balances, sessions, isins, volume)
        protected = [
            c
            for c in old.columns
            if not any(c == f or c == f + "_age_sessions" for f in RATE_FIELDS)
        ]
        matched = result.join(
            old.select("date", "isin"), on=["date", "isin"], how="inner"
        )
        if (
            not old.select(protected)
            .sort("date", "isin")
            .equals(matched.select(protected).sort("date", "isin"))
        ):
            raise ValueError(
                "lending extension changed a protected balance/utilization field"
            )
        original_rates = pl.read_parquet(paths["original_rates"])
        keys = ["available_date", "security_id"]
        retained = (
            rates.join(original_rates.select(keys), on=keys, how="inner")
            .select(original_rates.columns)
            .sort(keys)
        )
        if not retained.equals(original_rates.sort(keys)):
            raise ValueError(
                "latest-vintage extension changed an original rate observation"
            )
        first_added = rates.join(original_rates.select(keys), on=keys, how="anti")[
            "available_date"
        ].min()
        if (
            not old.filter(pl.col("date") < first_added)
            .sort("date", "isin")
            .equals(result.filter(pl.col("date") < first_added).sort("date", "isin"))
        ):
            raise ValueError("new rates altered feature history before first addition")
        cutoff = date(2024, 6, 28)
        changed = rates.with_columns(
            [
                pl.when(pl.col("available_date") > cutoff)
                .then(pl.col(c) * 7)
                .otherwise(pl.col(c))
                .alias(c)
                for c in ("annual_taker_rate", "registered_quantity")
            ]
        )
        mutated = replace_lending_rates(old, changed, balances, sessions, isins, volume)
        if not result.filter(pl.col("date") <= cutoff).equals(
            mutated.filter(pl.col("date") <= cutoff)
        ):
            raise ValueError("future lending rates/quantities changed earlier features")
        names = tuple(
            c
            for c in old.columns
            if c not in ("date", "isin") and not c.endswith("_age_sessions")
        )
        values, valid, ages = align_family(result, sessions, isins, names)
        output.mkdir(parents=True, exist_ok=False)
        admission = write_family(
            output,
            "lending",
            names,
            values,
            valid,
            ages,
            sessions,
            isins,
            active,
            {
                "plan": bind(plan_path),
                "inputs": plan["inputs"],
                "access": access.payload(),
            },
            code,
            {
                "future_mutation_cutoff": str(cutoff),
                "future_rate_quantity_multiplier": 7,
                "past_exact": True,
                "all_original_observations_exact": retained.height,
                "unaffected_fields_exact": protected[2:],
                "before_first_addition_exact": str(first_added),
            },
            {
                "source_vintage": "latest_vintage_unknown_revision_share",
                "changed_fields": list(RATE_FIELDS),
                "rate_source_manifest": bind(paths["rate_manifest"]),
            },
        )
        write_json_atomic(
            output / "manifest.json",
            {
                "code": code,
                "admission": admission,
                "source_plan_sha256": sha256_file(plan_path),
            },
        )
        return admission
    finally:
        store.close()


def extend_cross_market(plan_path: Path, output: Path) -> dict:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    paths = {name: verified(value) for name, value in plan["inputs"].items()}
    code = _git_identity()
    report = ET.parse(paths["causality_tests"]).getroot()
    if (
        not report.findall(".//testcase")
        or report.findall(".//failure")
        or report.findall(".//error")
    ):
        raise ValueError("cross-market causality tests did not pass")
    source = paths["base_manifest"].parent
    axis = np.load(source / "date_index.npy", allow_pickle=False)
    if axis.max() > np.datetime64(DEVELOPMENT_END):
        raise PermissionError("cross-market base enters held-out history")
    rows = np.arange(len(axis))
    samples = rows[
        (axis <= np.datetime64(PRETRAIN_END)) | (axis >= np.datetime64(FINETUNE_START))
    ]
    store, access = open_store_for_samples(
        source, samples, purpose="training", history_lookbacks=60, history_end_offsets=0
    )
    try:
        sessions, isins = axis.astype(object).tolist(), store.isins
        active = store.read("active", rows)
        wealth = store.read("shareholder_wealth_close", rows)
        wealth_valid = store.read("shareholder_wealth_valid", rows)
        returns, return_valid = exact_log_return(
            wealth, 1, shareholder_wealth_valid=wealth_valid
        )
        parent_valid = store.read("slow_valid", rows)
        return_field = list(store.manifest["feature_names"]["slow"]).index(
            "log_return_1"
        )
        return_valid[:-1] &= parent_valid[1:, :, return_field]
        return_valid[-1] = False
        del wealth, wealth_valid, parent_valid
        levels = pl.read_parquet(paths["market_levels"])
        us = pl.read_parquet(paths["us_returns"])
        cash = pl.read_parquet(
            paths["cash_identity"], columns=["source_trade_date", "ticker", "isin"]
        )
        for frame, field in (
            (levels, "reference_date"),
            (us, "reference_date"),
            (cash, "source_trade_date"),
        ):
            if frame[field].max() > DEVELOPMENT_END:
                raise PermissionError("cross-market source enters held-out history")
        bova = load_bova11_series(
            paths["bova_manifest"].parent,
            expected_manifest_sha256=plan["inputs"]["bova_manifest"]["sha256"],
            canonical_dates=sessions,
        ).close_by_session
        bova_return = np.full(len(sessions), np.nan)
        bova_return[1:] = np.log(bova[1:] / bova[:-1])
        identity = json.loads(paths["adr_identity"].read_text(encoding="utf-8"))
        gap, gap_mask, ewz, ewz_mask, flag = relative_return_panel(
            us,
            levels,
            cash,
            identity["pairs"],
            sessions,
            isins,
            returns,
            return_valid,
            bova_return,
        )
        additions = {
            "adr_return_gap_1": (gap, gap_mask, np.ones_like(gap)),
            "ewz_minus_bova11_1": (
                ewz[:, None],
                ewz_mask[:, None],
                np.ones((len(sessions), 1)),
            ),
            "adr_listed_flag": (flag, (rows > 0)[:, None], np.ones_like(flag)),
        }
        # Actual source-mutation check includes both US/FX values and dated
        # assignments, not only the arithmetic fixture.
        cutoff = date(2024, 6, 28)
        changed_us = us.with_columns(
            pl.when(pl.col("reference_date") > cutoff)
            .then(pl.col("log_return") * 7)
            .otherwise(pl.col("log_return"))
            .alias("log_return")
        )
        changed_fx = levels.with_columns(
            pl.when(pl.col("reference_date") > cutoff)
            .then(pl.col("value") * 3)
            .otherwise(pl.col("value"))
            .alias("value")
        )
        changed_gap = relative_return_panel(
            changed_us,
            changed_fx,
            cash.filter(pl.col("source_trade_date") <= cutoff),
            identity["pairs"],
            sessions,
            isins,
            returns,
            return_valid,
            bova_return,
        )
        prefix = axis <= np.datetime64(cutoff)
        for a, b in zip((gap, gap_mask, ewz, ewz_mask, flag), changed_gap, strict=True):
            np.testing.assert_array_equal(a[prefix], b[prefix])
        del changed_gap, changed_us, changed_fx
        oil = admit_brent(levels.filter(pl.col("series") == "brent_spot"), sessions)
        shocks = market_shocks(oil, us.head(0))
        current, shock_age, historical, public_index = shock_axes(shocks, sessions)[
            "oil"
        ]
        sectors, issuers = identity_axes(
            pl.read_parquet(paths["cvm_identity"]), sessions, isins
        )
        cdi = pl.read_parquet(paths["cdi"])
        if cdi["date"].to_list() != sessions:
            raise ValueError("CDI axis differs from the cross-market store")
        excess = returns - np.log1p(cdi["daily_cdi_rate"].to_numpy())[:, None]
        beta, beta_mask, beta_age = exposure_panel(
            excess, return_valid, historical, public_index, sectors, issuers
        )
        additions["exposure_oil"] = beta, beta_mask, beta_age
        for j, h in enumerate((1, 5)):
            known = np.isfinite(current[:, j, None])
            additions[f"shock_oil_{h}"] = (
                current[:, j, None],
                known,
                shock_age[:, j, None],
            )
            joint = known & beta_mask
            additions[f"exposure_oil_times_shock_{h}"] = (
                beta * current[:, j, None],
                joint,
                np.maximum(beta_age, shock_age[:, j, None]),
            )
        del sectors, issuers, excess, returns, return_valid
        flow_manifest = json.loads(paths["foreign_flow"].read_text(encoding="utf-8"))
        observations = []
        for record in flow_manifest["results"]:
            if "observation" in record:
                observation = dict(record["observation"])
                for field in ("reference_date", "publication_date"):
                    observation[field] = date.fromisoformat(observation[field])
                observations.append(observation)
        flow_names, flow, flow_mask, flow_age, flow_ledger = decision_panel(
            observations, sessions
        )
        magnitude, magnitude_mask, magnitude_age = align_family(
            pl.read_parquet(
                paths["magnitudes"],
                columns=[
                    "date",
                    "isin",
                    "log_traded_value_20",
                    "log_traded_value_20_age_sessions",
                ],
            ),
            sessions,
            isins,
            ("log_traded_value_20",),
        )
        magnitude, magnitude_mask, magnitude_age = (
            magnitude[..., 0],
            magnitude_mask[..., 0],
            magnitude_age[..., 0],
        )
        for j, name in enumerate(flow_names):
            value, mask, age = (
                flow[:, j, None],
                flow_mask[:, j, None],
                flow_age[:, j, None],
            )
            additions[name] = value, mask, age
            if name in ("foreign_flow_1", "foreign_flow_5"):
                additions[name + "_times_log_volume_mean_20"] = (
                    value * magnitude,
                    mask & magnitude_mask,
                    np.maximum(age, magnitude_age),
                )
                additions[name + "_times_adr_listed_flag"] = (
                    value * flag,
                    mask,
                    np.maximum(age, 1),
                )
        d, n = np.where(active)
        columns = {"date": pl.Series(axis[d]), "isin": np.asarray(isins)[n]}
        for name, (value, mask, age) in additions.items():
            value, mask, age = (
                np.broadcast_to(x, active.shape)[d, n] for x in (value, mask, age)
            )
            known = mask & np.isfinite(value)
            columns[name] = pl.Series(np.asarray(value, np.float32)).set(
                pl.Series(~known), None
            )
            columns[name + "_age_sessions"] = pl.Series(
                np.asarray(age, np.float32)
            ).set(pl.Series(~known), None)
        extended = pl.DataFrame(columns)
        old = pl.read_parquet(paths["old_cross_market"])
        protected = [
            c
            for c in old.columns
            if c in ("date", "isin")
            or (
                c.removesuffix("_age_sessions") in FEATURE_NAMES
                and c.removesuffix("_age_sessions") not in additions
            )
        ]
        result = (
            old.select(protected)
            .join(extended, on=["date", "isin"], how="full", coalesce=True)
            .sort("date", "isin")
        )
        if (
            not result.join(
                old.select("date", "isin"), on=["date", "isin"], how="inner"
            )
            .select(protected)
            .sort("date", "isin")
            .equals(old.select(protected).sort("date", "isin"))
        ):
            raise ValueError(
                "cross-market extension changed an unrelated existing field"
            )
        if peak_rss_bytes() > 8 * 1024**3:
            raise MemoryError("cross-market extension exceeded the 8-GiB bound")
        output.mkdir(parents=True, exist_ok=False)
        data = output / "features.parquet"
        result.select(
            "date",
            "isin",
            *[c for f in FEATURE_NAMES for c in (f, f + "_age_sessions")],
        ).write_parquet(data)
        flow_ledger.write_parquet(output / "published_flow_differences.parquet")
        shocks.write_parquet(output / "oil_shocks.parquet")
        proof = output / "availability_proof.json"
        write_json_atomic(
            proof,
            {
                "causality_passed": True,
                "first_available_decision_passed": True,
                "data_sha256": sha256_file(data),
                "source_code": code,
                "fixture_report": bind(paths["causality_tests"]),
                "actual_us_fx_identity_future_mutation_cutoff": str(cutoff),
                "prefix_exact": True,
                "untouched_fields_exact": protected[2:],
                "timing_conventions": "Brent next B3 decision; BDI date-only next decision; US exact previous-session closes and known PTAX.",
            },
        )
        coverage = (
            result.with_columns(pl.col("date").dt.year().alias("year"))
            .group_by("year")
            .agg(
                pl.len().alias("active_name_days"),
                *[pl.col(f).count().alias(f) for f in FEATURE_NAMES],
            )
            .sort("year")
            .to_dicts()
        )
        manifest = output / "manifest.json"
        write_json_atomic(
            manifest,
            {
                "source_code": code,
                "family": "cross_market",
                "feature_names": list(FEATURE_NAMES),
                "data": bind(data),
                "availability_proof": bind(proof),
                "sources": plan["inputs"],
                "source_plan": bind(plan_path),
                "coverage": coverage,
                "source_vintage": "market-specific; new BDI and adjusted US/Brent archives latest_vintage_unknown_revision_share",
                "consumer_end": str(DEVELOPMENT_END),
                "access": access.payload(),
                "peak_rss_bytes": peak_rss_bytes(),
            },
        )
        return {
            "family": "cross_market",
            "feature_names": list(FEATURE_NAMES),
            "status": "admitted",
            "source_manifest": bind(manifest),
            "data": bind(data),
            "availability_proof": bind(proof),
        }
    finally:
        store.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--family", choices=("lending", "cross_market"), default="lending"
    )
    args = parser.parse_args()
    build = extend_lending if args.family == "lending" else extend_cross_market
    print(json.dumps(build(args.plan, args.output), indent=2))


if __name__ == "__main__":
    main()
