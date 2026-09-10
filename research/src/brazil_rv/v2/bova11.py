from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl
from numpy.typing import NDArray

from .artifacts import sha256_file
from .contract import DEVELOPMENT_END, OFFICIAL_START

BOVA11_ISIN = "BRBOVACTF003"
BOVA11_TICKER = "BOVA11"
BOVA11_SECURITY_SPEC = "CI"
# The same BOVA11 cash-market instrument used BDI 02 from 2019-08-19 to
# 2019-12-30 and BDI 14 otherwise. BDI is a source classification, not identity.
BOVA11_BDI_CODES = ("02", "14")
BOVA11_MARKET_TYPE = 10
BOVA11_SCHEMA = "BRAZIL_RV_V2_BOVA11_HEDGE_SERIES_V1"
_RECORD_LENGTH = 245


@dataclass(frozen=True)
class Bova11Series:
    close_by_session: NDArray[np.float64]
    manifest_path: Path
    manifest_sha256: str
    data_path: Path
    data_sha256: str


def _text(raw: bytes) -> str:
    return raw.decode("latin-1", errors="strict").strip()


def _integer(raw: bytes) -> int:
    value = raw.decode("ascii", errors="strict").strip()
    return int(value) if value else 0


def _day(raw: bytes) -> date:
    return datetime.strptime(raw.decode("ascii"), "%Y%m%d").date()


def _member(archive: zipfile.ZipFile, source: Path) -> str:
    names = [name for name in archive.namelist() if name.upper().endswith(".TXT")]
    if not names:
        raise ValueError(f"COTAHIST archive has no TXT member: {source}")
    preferred = [name for name in names if "COTAHIST" in name.upper()]
    return sorted(preferred or names)[0]


def _archive_rows(source: Path) -> tuple[list[tuple[date, float]], dict[str, object]]:
    rows: list[tuple[date, float]] = []
    header_count = 0
    trailer_count = 0
    malformed_count = 0
    bdi_row_counts = dict.fromkeys(BOVA11_BDI_CODES, 0)
    with zipfile.ZipFile(source) as archive:
        member = _member(archive, source)
        with archive.open(member) as handle:
            for raw in handle:
                line = raw.rstrip(b"\r\n")
                if len(line) != _RECORD_LENGTH:
                    malformed_count += 1
                    continue
                if line[:2] == b"00":
                    header_count += 1
                    continue
                if line[:2] == b"99":
                    trailer_count += 1
                    continue
                if line[:2] != b"01":
                    continue
                if (
                    _text(line[10:12]) not in BOVA11_BDI_CODES
                    or _text(line[12:24]) != BOVA11_TICKER
                    or _integer(line[24:27]) != BOVA11_MARKET_TYPE
                    or _text(line[39:49]).split()[0] != BOVA11_SECURITY_SPEC
                    or _text(line[230:242]) != BOVA11_ISIN
                ):
                    continue
                quote_factor = _integer(line[210:217]) or 1
                close = _integer(line[108:121]) / (100.0 * quote_factor)
                if close <= 0.0 or not np.isfinite(close):
                    raise ValueError(f"BOVA11 has an invalid close in {source}")
                rows.append((_day(line[2:10]), close))
                bdi_row_counts[_text(line[10:12])] += 1
    if header_count != 1 or trailer_count != 1 or malformed_count:
        raise ValueError(
            f"malformed COTAHIST archive {source}: headers={header_count}, "
            f"trailers={trailer_count}, bad_lengths={malformed_count}"
        )
    return rows, {
        "path": str(source.resolve()),
        "bytes": source.stat().st_size,
        "sha256": sha256_file(source),
        "txt_member": member,
        "bova11_row_count": len(rows),
        "bdi_row_counts": bdi_row_counts,
    }


def build_bova11_series(
    archives: Sequence[Path], output_root: Path
) -> dict[str, object]:
    """Build one immutable, development-only BOVA11 close series."""

    sources = tuple(Path(path).resolve(strict=True) for path in archives)
    if not sources or len(set(sources)) != len(sources):
        raise ValueError("BOVA11 construction requires unique COTAHIST archives")
    if output_root.exists():
        raise FileExistsError(output_root)
    rows: list[tuple[date, float]] = []
    source_records: list[dict[str, object]] = []
    for source in sources:
        archive_rows, record = _archive_rows(source)
        rows.extend(archive_rows)
        source_records.append(record)
    if not rows:
        raise ValueError("COTAHIST archives contain no exact BOVA11 observations")
    rows.sort()
    by_day: dict[date, float] = {}
    for day, close in rows:
        if day >= OFFICIAL_START:
            raise PermissionError("BOVA11 artifact refuses 2025/2026 observations")
        if day > DEVELOPMENT_END:
            raise PermissionError("BOVA11 artifact exceeds the development boundary")
        prior = by_day.setdefault(day, close)
        if prior != close:
            raise ValueError(f"conflicting BOVA11 close observations on {day}")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
    try:
        data_path = temporary / "bova11_close.parquet"
        pl.DataFrame(
            {
                "trade_date": list(by_day),
                "close_brl": list(by_day.values()),
            },
            schema={"trade_date": pl.Date, "close_brl": pl.Float64},
        ).write_parquet(data_path)
        manifest: dict[str, object] = {
            "schema": BOVA11_SCHEMA,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
            "security": {
                "ticker": BOVA11_TICKER,
                "isin": BOVA11_ISIN,
                "security_spec": BOVA11_SECURITY_SPEC,
                "market_type": BOVA11_MARKET_TYPE,
                "bdi_codes": list(BOVA11_BDI_CODES),
            },
            "scope": "development sessions only; 2025/2026 rows are forbidden",
            "first_date": min(by_day).isoformat(),
            "last_date": max(by_day).isoformat(),
            "observation_count": len(by_day),
            "data_file": data_path.name,
            "data_bytes": data_path.stat().st_size,
            "data_sha256": sha256_file(data_path),
            "sources": source_records,
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_root)
    except BaseException:
        if temporary.exists() and temporary.parent == output_root.parent:
            shutil.rmtree(temporary)
        raise
    return manifest


def load_bova11_series(
    artifact_root: Path,
    *,
    expected_manifest_sha256: str,
    canonical_dates: Sequence[date],
) -> Bova11Series:
    """Hash-verify and align BOVA11 without opening a protected session."""

    root = artifact_root.resolve(strict=True)
    manifest_path = root / "manifest.json"
    actual_manifest_sha256 = sha256_file(manifest_path)
    if actual_manifest_sha256 != expected_manifest_sha256.casefold():
        raise ValueError("BOVA11 manifest SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != BOVA11_SCHEMA or manifest.get("status") != "complete":
        raise ValueError("BOVA11 manifest contract is invalid")
    if any(
        manifest.get("security", {}).get(key) != value
        for key, value in {
            "ticker": BOVA11_TICKER,
            "isin": BOVA11_ISIN,
            "security_spec": BOVA11_SECURITY_SPEC,
            "market_type": BOVA11_MARKET_TYPE,
        }.items()
    ):
        raise ValueError("BOVA11 manifest security identity is invalid")
    data_path = root / str(manifest.get("data_file"))
    if data_path.stat().st_size != manifest.get("data_bytes") or sha256_file(
        data_path
    ) != manifest.get("data_sha256"):
        raise ValueError("BOVA11 close series hash or byte count mismatch")
    frame = pl.read_parquet(data_path)
    if frame.columns != ["trade_date", "close_brl"]:
        raise ValueError("BOVA11 close series has unexpected columns")
    if frame.get_column("trade_date").n_unique() != frame.height:
        raise ValueError("BOVA11 close series contains duplicate dates")
    source_dates = tuple(frame.get_column("trade_date").to_list())
    if source_dates != tuple(sorted(source_dates)):
        raise ValueError("BOVA11 close dates are not strictly ordered")
    if any(day >= OFFICIAL_START for day in source_dates):
        raise PermissionError("BOVA11 loader refuses 2025/2026 observations")
    close = np.asarray(frame.get_column("close_brl"), dtype=np.float64)
    if np.any(~np.isfinite(close) | (close <= 0.0)):
        raise ValueError("BOVA11 close series contains invalid values")
    canonical = tuple(canonical_dates)
    if not canonical or any(
        left >= right for left, right in zip(canonical, canonical[1:], strict=False)
    ):
        raise ValueError("canonical calendar must be nonempty and strictly ordered")
    date_index = {day: index for index, day in enumerate(canonical)}
    first_canonical, last_canonical = canonical[0], canonical[-1]
    foreign = [
        day
        for day in source_dates
        if first_canonical <= day <= last_canonical and day not in date_index
    ]
    if foreign:
        raise ValueError(
            f"BOVA11 rows are outside the canonical calendar: {foreign[:5]}"
        )
    aligned = np.full(len(canonical), np.nan, dtype=np.float64)
    for day, value in zip(source_dates, close, strict=True):
        index = date_index.get(day)
        if index is not None:
            aligned[index] = value
    return Bova11Series(
        close_by_session=aligned,
        manifest_path=manifest_path,
        manifest_sha256=actual_manifest_sha256,
        data_path=data_path,
        data_sha256=str(manifest["data_sha256"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the immutable development-only BOVA11 hedge series"
    )
    parser.add_argument("--archive", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    manifest = build_bova11_series(arguments.archive, arguments.out)
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
