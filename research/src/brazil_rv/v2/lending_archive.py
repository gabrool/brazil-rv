from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl
from numpy.typing import NDArray

from brazil_rv.preprocessing.bdi_lending import build_sidecar
from brazil_rv.preprocessing.bdi_lending_strong import (
    build_lending_strong_features,
)


LENDING_ARCHIVE_SCHEMA = "BRAZIL_RV_V2_LENDING_ARCHIVE_V2"
BORROW_SOURCE_LABEL = "lending_archive_v2_2009_202412"
REGISTRATION_FEE_ANNUAL = 0.0025
PRE_FIRST_CAUSAL_RATE_PLACEHOLDER = 0.02
RATE_LOOKBACK_SESSIONS = 60
STRICT_LOOKBACK_SESSIONS = 20
SOURCE_END = date(2024, 12, 30)


@dataclass(frozen=True)
class LendingBorrowPanels:
    annual_taker_rate: NDArray[np.float64]
    rate_imputed: NDArray[np.bool_]
    rate_placeholder: NDArray[np.bool_]
    shortable_strict: NDArray[np.bool_]
    shortable_balance: NDArray[np.bool_]
    shortable_open: NDArray[np.bool_]
    manifest_sha256: str
    balance_sha256: str
    rate_sha256: str
    source_label: str
    source_unavailable_dates: tuple[date, ...]
    source_placeholder_dates: tuple[date, ...]

    @property
    def availability(self) -> dict[str, NDArray[np.bool_]]:
        return {
            "borrow_strict": self.shortable_strict,
            "borrow_balance": self.shortable_balance,
            "borrow_open": self.shortable_open,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _verified_output(root: Path, filename: str) -> tuple[dict[str, object], Path]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output = root / filename
    if manifest.get("status") != "complete" or manifest.get("output_sha256") != _sha256(
        output
    ):
        raise ValueError(f"source lending artifact is stale: {root}")
    return manifest, output


def _verify_raw_snapshot(root: Path) -> str:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("contract_version") != "B3_BDI_CHAPTER_05_PDF_SNAPSHOT_V1"
        or manifest.get("status") != "complete"
    ):
        raise ValueError("new-month BDI snapshot contract is invalid")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("new-month BDI snapshot lacks its file inventory")
    downloaded = 0
    for record in files:
        if not isinstance(record, dict) or record.get("status") != "downloaded":
            continue
        path = root / str(record["filename"])
        if path.stat().st_size != int(record["bytes"]) or _sha256(path) != record[
            "sha256"
        ]:
            raise ValueError(f"raw BDI PDF identity mismatch: {path}")
        downloaded += 1
    if downloaded != 127 or downloaded != int(manifest.get("downloaded_pdf_count", -1)):
        raise ValueError("rev4b requires exactly 127 sealed new-month BDI PDFs")
    return _sha256(manifest_path)


def _raw_rate_rows(frame: pl.DataFrame) -> pl.DataFrame:
    value = "lending_taker_fee_level_log_tanh"
    mask = f"{value}_mask"
    encoded = frame.filter(pl.col(mask)).get_column(value).to_numpy().astype(np.float64)
    if (
        not np.isfinite(encoded).all()
        or np.any(encoded < 0.0)
        or np.any(encoded >= 1.0)
    ):
        raise ValueError("lending-rate transform is outside its invertible domain")
    annual_decimal = 0.01 * np.expm1(2.0 * np.arctanh(encoded))
    roundtrip = np.tanh(np.log1p(100.0 * annual_decimal) / 2.0)
    if not np.allclose(roundtrip, encoded, rtol=0.0, atol=2e-15):
        raise ValueError("lending-rate transform does not invert exactly")
    return (
        frame.filter(pl.col(mask))
        .select(
            "source_trade_date",
            "available_date",
            "security_id",
            "registered_contracts",
            "registered_quantity",
        )
        .with_columns(pl.Series("annual_taker_rate", annual_decimal))
        .sort("available_date", "source_trade_date", "security_id")
    )


def _identical(left: pl.DataFrame, right: pl.DataFrame) -> bool:
    return left.schema == right.schema and left.equals(right, null_equal=True)


def build_lending_archive_v2(
    *,
    old_balance_root: Path,
    old_rate_root: Path,
    new_pdf_root: Path,
    calendar_dir: Path,
    assignments_path: Path,
    output_root: Path,
) -> dict[str, object]:
    """Extend the sealed v1 lending observations without rebuilding the store."""

    roots = [
        Path(old_balance_root).resolve(strict=True),
        Path(old_rate_root).resolve(strict=True),
        Path(new_pdf_root).resolve(strict=True),
    ]
    calendar = Path(calendar_dir).resolve(strict=True)
    assignments = Path(assignments_path).resolve(strict=True)
    output = Path(output_root).resolve()
    if output.exists():
        raise FileExistsError(output)
    old_balance_manifest, old_balance_path = _verified_output(
        roots[0], "bdi_lending_open_balance.parquet"
    )
    old_rate_manifest, old_rate_path = _verified_output(
        roots[1], "bdi_lending_strong.parquet"
    )
    raw_manifest_sha = _verify_raw_snapshot(roots[2])

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        new_balance_root = temporary / "new_month_parse" / "open_balance"
        new_rate_root = temporary / "new_month_parse" / "rates"
        build_sidecar(
            roots[2],
            calendar,
            assignments,
            new_balance_root,
            start=date(2024, 7, 1),
            end=SOURCE_END,
        )
        build_lending_strong_features(
            pdf_dirs=[roots[2]],
            calendar_dir=calendar,
            assignments_path=assignments,
            output_dir=new_rate_root,
            start=date(2024, 7, 1),
            end=SOURCE_END,
        )

        old_balance = pl.read_parquet(old_balance_path)
        new_balance = pl.read_parquet(
            new_balance_root / "bdi_lending_open_balance.parquet"
        )
        balance_columns = [
            "source_position_date",
            "source_report_date",
            "available_date",
            "security_id",
            "source_identity_method",
            "lending_balance_quantity",
            "lending_balance_brl",
        ]
        old_balance_raw = old_balance.select(balance_columns).sort(
            "available_date", "security_id"
        )
        new_balance_raw_all = new_balance.select(balance_columns).sort(
            "available_date", "security_id"
        )
        new_balance_raw = new_balance_raw_all.filter(
            pl.col("available_date") <= pl.lit(SOURCE_END)
        )
        excluded_balance_rows = new_balance_raw_all.height - new_balance_raw.height
        balances = pl.concat([old_balance_raw, new_balance_raw], how="vertical")
        balances = balances.sort("available_date", "security_id")
        if balances.select(
            pl.struct("available_date", "security_id").is_duplicated().any()
        ).item():
            raise ValueError("combined lending balances have duplicate as-of keys")
        overlap_balances = balances.filter(
            pl.col("available_date")
            <= old_balance_raw.get_column("available_date").max()
        )
        if not _identical(overlap_balances, old_balance_raw):
            raise ValueError("lending balance overlap identity failed")

        old_rates = _raw_rate_rows(pl.read_parquet(old_rate_path))
        new_rates_all = _raw_rate_rows(
            pl.read_parquet(new_rate_root / "bdi_lending_strong.parquet")
        )
        new_rates = new_rates_all.filter(
            pl.col("available_date") <= pl.lit(SOURCE_END)
        )
        excluded_rate_rows = new_rates_all.height - new_rates.height
        rates = pl.concat([old_rates, new_rates], how="vertical").sort(
            "available_date", "source_trade_date", "security_id"
        )
        if rates.select(
            pl.struct("available_date", "security_id").is_duplicated().any()
        ).item():
            raise ValueError("combined lending rates have duplicate as-of keys")
        overlap_rates = rates.filter(
            pl.col("available_date") <= old_rates.get_column("available_date").max()
        )
        if not _identical(overlap_rates, old_rates):
            raise ValueError("lending rate overlap identity failed")

        balance_path = temporary / "lending_balances.parquet"
        rate_path = temporary / "lending_rates.parquet"
        balances.write_parquet(balance_path, compression="zstd", statistics=True)
        rates.write_parquet(rate_path, compression="zstd", statistics=True)
        prior_names = set(old_balance_raw.get_column("security_id").to_list()) | set(
            old_rates.get_column("security_id").to_list()
        )
        new_names = set(new_balance_raw_all.get_column("security_id").to_list()) | set(
            new_rates_all.get_column("security_id").to_list()
        )
        never_seen = sorted(new_names - prior_names)
        overlap = {
            "status": "passed",
            "balance_row_count": old_balance_raw.height,
            "rate_row_count": old_rates.height,
            "balance_source_last_available_date": str(
                old_balance_raw.get_column("available_date").max()
            ),
            "rate_source_last_available_date": str(
                old_rates.get_column("available_date").max()
            ),
            "new_month_names_never_seen_before_count": len(never_seen),
            "new_month_names_never_seen_before": never_seen,
        }
        overlap_path = temporary / "overlap_identity.json"
        overlap_path.write_text(
            json.dumps(overlap, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        artifacts = {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in (balance_path, rate_path, overlap_path)
        }
        manifest: dict[str, object] = {
            "schema": LENDING_ARCHIVE_SCHEMA,
            "source_label": BORROW_SOURCE_LABEL,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
            "consumer_calendar_start": "2009-01-02",
            "consumer_calendar_end": SOURCE_END.isoformat(),
            "first_balance_observation": str(
                balances.get_column("source_position_date").min()
            ),
            "last_balance_observation": str(
                balances.get_column("source_position_date").max()
            ),
            "first_rate_observation": str(
                rates.get_column("source_trade_date").min()
            ),
            "last_rate_observation": str(
                rates.get_column("source_trade_date").max()
            ),
            "availability_rule": (
                "source report/trade publication session D first available on the "
                "next B3 session"
            ),
            "rate_rule": (
                "last observed taker rate within 60 sessions; otherwise the same-"
                "decision-row cross-sectional 75th percentile of such observed rates"
            ),
            "registration_fee_annual_decimal": REGISTRATION_FEE_ANNUAL,
            "availability_cells": {
                "borrow_strict": "lending trade observed in prior 20 sessions",
                "borrow_balance": (
                    "latest published positive open balance or lending trade in prior "
                    "60 sessions"
                ),
                "borrow_open": "all names whenever a causal cross-sectional rate exists",
            },
            "source_unavailable_rule": (
                "before the first causal cross-sectional rate, borrow_open remains "
                "unpriceable rather than using a future rate"
            ),
            "post_consumer_end_rows_excluded": {
                "balance": excluded_balance_rows,
                "rate": excluded_rate_rows,
                "reason": (
                    "D+1 availability falls after the registered 2024-12-30 "
                    "consumer boundary"
                ),
            },
            "old_balance": {
                "root": str(roots[0]),
                "manifest_sha256": _sha256(roots[0] / "manifest.json"),
                "data_sha256": str(old_balance_manifest["output_sha256"]),
            },
            "old_rate": {
                "root": str(roots[1]),
                "manifest_sha256": _sha256(roots[1] / "manifest.json"),
                "data_sha256": str(old_rate_manifest["output_sha256"]),
            },
            "new_pdf_snapshot": {
                "root": str(roots[2]),
                "manifest_sha256": raw_manifest_sha,
                "pdf_count": 127,
            },
            "calendar_directory": str(calendar),
            "assignments": {
                "path": str(assignments),
                "sha256": _sha256(assignments),
            },
            "balance_rows": balances.height,
            "rate_rows": rates.height,
            "overlap_identity": overlap,
            "artifacts": artifacts,
            "official_validation_accessed": False,
            "test_accessed": False,
        }
        if any(
            value.year >= 2025
            for frame, columns in (
                (
                    balances,
                    (
                        "source_position_date",
                        "source_report_date",
                        "available_date",
                    ),
                ),
                (rates, ("source_trade_date", "available_date")),
            )
            for column in columns
            for value in frame.get_column(column).to_list()
        ):
            raise PermissionError("lending archive builder refuses 2025/2026 rows")
        shutil.rmtree(temporary / "new_month_parse")
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    except BaseException:
        if temporary.exists() and temporary.parent == output.parent:
            shutil.rmtree(temporary)
        raise
    return manifest


def load_lending_borrow_panels(
    root: Path,
    *,
    expected_manifest_sha256: str,
    canonical_dates: Sequence[date],
    canonical_isins: Sequence[str],
) -> LendingBorrowPanels:
    """Hash-verify and causally align the three registered borrow cells."""

    archive = Path(root).resolve(strict=True)
    manifest_path = archive / "manifest.json"
    manifest_sha = _sha256(manifest_path)
    if manifest_sha != expected_manifest_sha256:
        raise ValueError("lending archive manifest SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != LENDING_ARCHIVE_SCHEMA
        or manifest.get("source_label") != BORROW_SOURCE_LABEL
        or manifest.get("status") != "complete"
        or manifest.get("official_validation_accessed") is not False
        or manifest.get("test_accessed") is not False
    ):
        raise ValueError("lending archive contract is invalid")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("lending archive lacks its artifact inventory")
    for filename in ("lending_balances.parquet", "lending_rates.parquet"):
        record = artifacts.get(filename)
        path = archive / filename
        if (
            not isinstance(record, dict)
            or path.stat().st_size != int(record.get("bytes", -1))
            or _sha256(path) != record.get("sha256")
        ):
            raise ValueError(f"lending archive artifact mismatch: {filename}")

    dates = tuple(canonical_dates)
    if not dates or any(left >= right for left, right in zip(dates, dates[1:])):
        raise ValueError("canonical lending calendar must be strictly increasing")
    date_positions = {value: index for index, value in enumerate(dates)}
    name_positions = {
        str(value).removeprefix("ISIN:"): index
        for index, value in enumerate(canonical_isins)
    }
    if len(name_positions) != len(canonical_isins):
        raise ValueError("canonical lending security axis contains duplicates")
    balances = pl.read_parquet(archive / "lending_balances.parquet")
    rates = pl.read_parquet(archive / "lending_rates.parquet")
    for column in ("source_position_date", "source_report_date", "available_date"):
        if any(value.year >= 2025 for value in balances.get_column(column).to_list()):
            raise PermissionError("lending archive loader refuses 2025/2026 rows")
    for column in ("source_trade_date", "available_date"):
        if any(value.year >= 2025 for value in rates.get_column(column).to_list()):
            raise PermissionError("lending archive loader refuses 2025/2026 rows")

    balance_events: dict[int, list[tuple[int, int, int]]] = {}
    for row in balances.iter_rows(named=True):
        source = date_positions.get(row["source_report_date"])
        available = date_positions.get(row["available_date"])
        name = name_positions.get(str(row["security_id"]).removeprefix("ISIN:"))
        if source is None or available is None or name is None:
            continue
        if available != source + 1:
            raise ValueError(
                "lending balance is not aligned to exact publication-D+1 availability"
            )
        balance_events.setdefault(available, []).append(
            (name, source, int(row["lending_balance_quantity"]))
        )

    rate_events: dict[int, list[tuple[int, int, float]]] = {}
    for row in rates.iter_rows(named=True):
        source = date_positions.get(row["source_trade_date"])
        available = date_positions.get(row["available_date"])
        name = name_positions.get(str(row["security_id"]).removeprefix("ISIN:"))
        if source is None or available is None or name is None:
            continue
        value = float(row["annual_taker_rate"])
        if available != source + 1:
            raise ValueError("lending rate is not aligned to exact D+1 availability")
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("lending archive contains an invalid taker rate")
        rate_events.setdefault(available, []).append((name, source, value))

    shape = (len(dates), len(canonical_isins))
    annual_rate = np.full(shape, np.nan, dtype=np.float64)
    imputed = np.zeros(shape, dtype=np.bool_)
    placeholder = np.zeros(shape, dtype=np.bool_)
    strict = np.zeros(shape, dtype=np.bool_)
    balance_cell = np.zeros(shape, dtype=np.bool_)
    open_cell = np.zeros(shape, dtype=np.bool_)
    last_balance = np.zeros(shape[1], dtype=np.int64)
    last_rate = np.full(shape[1], np.nan, dtype=np.float64)
    last_rate_session = np.full(shape[1], -1, dtype=np.int64)
    unavailable_dates: list[date] = []
    placeholder_dates: list[date] = []
    first_causal_rate_session = min(rate_events, default=None)
    if first_causal_rate_session is None:
        raise ValueError("lending archive has no rate event on the canonical axes")
    for day in range(shape[0]):
        for name, _source, quantity in balance_events.get(day, ()):
            last_balance[name] = quantity
        for name, source, value in rate_events.get(day, ()):
            last_rate[name] = value
            last_rate_session[name] = source
        rate_age = day - last_rate_session
        rate_recent = np.isfinite(last_rate) & (rate_age >= 0) & (
            rate_age <= RATE_LOOKBACK_SESSIONS
        )
        observed = last_rate[rate_recent]
        cross_sectional_rate = (
            float(np.quantile(observed, 0.75)) if observed.size else np.nan
        )
        if not np.isfinite(cross_sectional_rate):
            if day < first_causal_rate_session:
                annual_rate[day] = PRE_FIRST_CAUSAL_RATE_PLACEHOLDER
                placeholder[day] = True
                strict[day] = True
                balance_cell[day] = True
                open_cell[day] = True
                placeholder_dates.append(dates[day])
                continue
            unavailable_dates.append(dates[day])
            continue
        annual_rate[day] = np.where(rate_recent, last_rate, cross_sectional_rate)
        imputed[day] = ~rate_recent
        priced = np.isfinite(annual_rate[day])
        positive_balance = last_balance > 0
        # ``borrow_strict`` is the observed-trade comparator.  A published
        # balance without a lending trade is intentionally insufficient here;
        # that broader causal evidence belongs to ``borrow_balance``.
        strict[day] = priced & rate_recent & (
            rate_age <= STRICT_LOOKBACK_SESSIONS
        )
        balance_cell[day] = priced & (positive_balance | rate_recent)
        open_cell[day] = priced
    return LendingBorrowPanels(
        annual_taker_rate=annual_rate,
        rate_imputed=imputed,
        rate_placeholder=placeholder,
        shortable_strict=strict,
        shortable_balance=balance_cell,
        shortable_open=open_cell,
        manifest_sha256=manifest_sha,
        balance_sha256=str(artifacts["lending_balances.parquet"]["sha256"]),
        rate_sha256=str(artifacts["lending_rates.parquet"]["sha256"]),
        source_label=BORROW_SOURCE_LABEL,
        source_unavailable_dates=tuple(unavailable_dates),
        source_placeholder_dates=tuple(placeholder_dates),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the rev4b lending archive")
    parser.add_argument("--old-balance-root", type=Path, required=True)
    parser.add_argument("--old-rate-root", type=Path, required=True)
    parser.add_argument("--new-pdf-root", type=Path, required=True)
    parser.add_argument("--calendar-dir", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_lending_archive_v2(
        old_balance_root=args.old_balance_root,
        old_rate_root=args.old_rate_root,
        new_pdf_root=args.new_pdf_root,
        calendar_dir=args.calendar_dir,
        assignments_path=args.assignments,
        output_root=args.out,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
