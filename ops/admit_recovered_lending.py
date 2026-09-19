"""Build a separately bound economic lending archive from completed source audits.

Does not parse PDFs again, touch model features, or launch a financial replay.
"""

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.lending_archive import (
    LENDING_ARCHIVE_SCHEMA,
    load_lending_borrow_panels,
)
from brazil_rv.v2.lending_recovery import recover_rates, recover_balances

PROJECT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def bind(path):
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def audit_tables(root):
    """Bind parsed observations to their completed receipts, without another census."""
    inventory = read(root / "source_inventory.json")
    frames, receipts = [], []
    for source in inventory["records"]:
        receipt_path = root / f"{source['report_date']}.json"
        receipt = read(receipt_path)
        if (
            receipt["source"] != source
            or receipt["parser_sha256"] != inventory["parser_sha256"]
        ):
            raise ValueError("audit receipt has a different source/parser contract")
        if receipt["status"] != "parsed" or receipt["rows"] == 0:
            continue
        path = receipt_path.with_suffix(".parquet")
        if sha256_file(path) != receipt["table_sha256"]:
            raise ValueError("audited parsed observations changed")
        frame = pl.read_parquet(path)
        if frame.height != receipt["rows"]:
            raise ValueError("parsed row count differs from receipt")
        frames.append(frame)
        receipts.append(bind(receipt_path))
    return pl.concat(frames, how="vertical_relaxed"), receipts


def main():
    run_pointer = read(PROJECT / "docs/v2_economic_data_scaling_run.json")
    root = Path(run_pointer["root"])
    output = root / "recovered_lending"
    if output.exists():
        raise FileExistsError(output)
    accepted = read(PROJECT / "docs/v2_round6_inputs.json")["evaluation_sources"][
        "design"
    ]["lending_archive"]
    old_root = Path(accepted["root"])
    if sha256_file(old_root / "manifest.json") != accepted["manifest_sha256"]:
        raise ValueError("accepted lending manifest differs")
    old_manifest = read(old_root / "manifest.json")
    tables = {}
    for name in ("lending_balances.parquet", "lending_rates.parquet"):
        path = old_root / name
        if sha256_file(path) != old_manifest["artifacts"][name]["sha256"]:
            raise ValueError("accepted lending data differs")
        tables[name] = pl.read_parquet(path)
    store_pointer = read(PROJECT / "docs/v2_data_inputs.json")["store"]
    store = Path(store_pointer["root"])
    if sha256_file(store / "manifest.json") != store_pointer["manifest_sha256"]:
        raise ValueError("accepted store identity differs")
    dates = np.load(store / "date_index.npy").astype("datetime64[D]")
    if dates[-1] > np.datetime64("2024-12-30"):
        raise PermissionError("development-only lending admission")
    isins = np.load(store / "isin_index.npy").tolist()
    active = np.load(store / "active.npy", mmap_mode="r")

    rate_audit = root / "lending_source_audit"
    balance_audit = root / "lending_balance_audit"
    raw_rates, rate_receipts = audit_tables(rate_audit)
    if read(rate_audit / "result.json")["failed_extractions"]:
        raise ValueError("registered-loan reconciliation is incomplete")
    # Reconstruct aggregate rates from the already verified printed-row tables.
    recovered = (
        raw_rates.group_by("report_date", "isin")
        .agg(
            pl.col("contracts").sum().alias("registered_contracts"),
            pl.col("quantity").sum().alias("registered_quantity"),
            pl.col("value_brl").sum().alias("registered_value_brl"),
            (
                (pl.col("quantity") * pl.col("taker_avg")).sum()
                / pl.col("quantity").sum()
                / 100
            ).alias("annual_taker_rate"),
            (
                (pl.col("quantity") * pl.col("donor_avg")).sum()
                / pl.col("quantity").sum()
                / 100
            ).alias("annual_donor_rate"),
        )
        .filter(pl.col("registered_quantity") > 0)
        .rename({"report_date": "source_trade_date"})
        .with_columns((pl.lit("ISIN:") + pl.col("isin")).alias("security_id"))
    )
    rates = recover_rates(tables["lending_rates.parquet"], recovered, dates)

    raw_balances, balance_receipts = audit_tables(balance_audit)
    mapped_path = balance_audit / "recovered_balances.parquet"
    mapped = pl.read_parquet(mapped_path)
    # The only enrichment allowed between printed balances and this table is
    # historical identity/publication timing, never changed quantities or BRL.
    keys = ["report_date", "position_date", "ticker"]
    joined = raw_balances.join(
        mapped, on=keys, suffix="_mapped", how="full", coalesce=True
    )
    if (
        joined.height != raw_balances.height
        or joined.filter(
            pl.col("quantity").is_null()
            | pl.col("quantity_mapped").is_null()
            | (pl.col("quantity") != pl.col("quantity_mapped"))
            | (pl.col("balance_brl") != pl.col("balance_brl_mapped"))
        ).height
    ):
        raise ValueError("identity-enriched balances differ from verified printed rows")
    balance_result = read(balance_audit / "result.json")
    if balance_result["identity_conflicts"]:
        raise ValueError("unresolved historical identity conflict")
    balances = recover_balances(tables["lending_balances.parquet"], mapped, dates)
    output.mkdir()
    artifacts = {}
    for name, frame in [
        ("lending_rates.parquet", rates),
        ("lending_balances.parquet", balances),
    ]:
        path = output / name
        frame.write_parquet(path)
        artifacts[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    manifest = dict(
        schema=LENDING_ARCHIVE_SCHEMA,
        source_label="bdi_recovered_development",
        status="complete",
        official_validation_accessed=False,
        test_accessed=False,
        artifacts=artifacts,
        accepted_archive=accepted,
        model_store=store_pointer,
        audit_sources=[
            bind(rate_audit / "source_inventory.json"),
            bind(rate_audit / "result.json"),
            bind(balance_audit / "source_inventory.json"),
            bind(balance_audit / "result.json"),
            bind(mapped_path),
        ],
        parsed_receipts=rate_receipts + balance_receipts,
        availability_rule="first supplied development session strictly after source report/trade date",
        rate_rule="unchanged: own rate up to 60 sessions, otherwise causal cross-sectional 75th percentile; 2% explicit pre-first-rate placeholder",
        rate_semantics="positive-flow quantity-weighted annual taker proxy; donor retained separately; modality blended, not an executable quote or identified commission",
        balance_semantics="published outstanding quantity, not executable locate capacity",
        scope="admitted reconciled observations for new economics; no feature/cache replacement or completed full-account acceptance",
        unresolved_balance_attempts=len(balance_result["failed_extractions"]),
        unmapped_balance_rows=mapped.filter(pl.col("isin").is_null()).height,
    )
    write_json_atomic(output / "manifest.json", manifest)
    # Exercise the real consumer on the unchanged full population/calendar.
    common = dict(canonical_dates=dates.astype(object).tolist(), canonical_isins=isins)
    old = load_lending_borrow_panels(
        old_root, expected_manifest_sha256=accepted["manifest_sha256"], **common
    )
    new = load_lending_borrow_panels(
        output, expected_manifest_sha256=sha256_file(output / "manifest.json"), **common
    )
    delta = new.annual_taker_rate - old.annual_taker_rate
    report = dict(
        rates=rates.height,
        accepted_rates=tables["lending_rates.parquet"].height,
        balances=balances.height,
        accepted_balances=tables["lending_balances.parquet"].height,
        dates=len(dates),
        securities=len(isins),
        eligible_stock_days=int(active.sum()),
        active_rate_changed=int((active & (np.abs(delta) > 1e-8)).sum()),
        active_rate_lower=int((active & (delta < -1e-8)).sum()),
        active_rate_higher=int((active & (delta > 1e-8)).sum()),
        active_imputed_before=int((active & old.rate_imputed).sum()),
        active_imputed_after=int((active & new.rate_imputed).sum()),
        active_placeholder_before=int((active & old.rate_placeholder).sum()),
        active_placeholder_after=int((active & new.rate_placeholder).sum()),
        active_balance_shortable_gained=int(
            (active & ~old.shortable_balance & new.shortable_balance).sum()
        ),
        active_balance_shortable_lost=int(
            (active & old.shortable_balance & ~new.shortable_balance).sum()
        ),
        source_unavailable_before=len(old.source_unavailable_dates),
        source_unavailable_after=len(new.source_unavailable_dates),
        current_feature_coordinates_changed=False,
        historical_outcomes_read=False,
    )
    np.savez_compressed(
        output / "equity_borrow_panels.npz",
        **{k: v for k, v in asdict(new).items() if isinstance(v, np.ndarray)},
    )
    report["panels"] = bind(output / "equity_borrow_panels.npz")
    report["manifest"] = bind(output / "manifest.json")
    write_json_atomic(output / "admission.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
