"""Bounded Round-6 extensions of the accepted physical family archives."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from .artifacts import sha256_file, write_json_atomic
from .contract import DEVELOPMENT_END, FINETUNE_START, PRETRAIN_END
from .research_rounds import _git_identity
from .round5_b3 import lending_decision_features
from .round5_derived import bind, verified, write_family
from .round5_store import align_family
from .store import open_store_for_samples

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(extend_lending(args.plan, args.output), indent=2))


if __name__ == "__main__":
    main()
