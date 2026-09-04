from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import polars as pl
from numpy.typing import NDArray

from .universe import session_calendar

VALID_ISIN = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")
CASH_EQUITY_SPECS = frozenset(
    {"ON", "OR", "PN", "PNA", "PNB", "PNC", "PND", "PNE", "PNF", "UNT"}
)


@dataclass(frozen=True)
class DailyPanel:
    dates: NDArray[np.datetime64]
    isins: tuple[str, ...]
    open_brl: NDArray[np.float64]
    high_brl: NDArray[np.float64]
    low_brl: NDArray[np.float64]
    close_brl: NDArray[np.float64]
    volume_brl: NDArray[np.float64]
    trades: NDArray[np.float64]
    quantity: NDArray[np.float64]
    distribution_number: NDArray[np.float64]
    observed: NDArray[np.bool_]

    def __post_init__(self) -> None:
        shape = (self.dates.size, len(self.isins))
        arrays = (
            self.open_brl,
            self.high_brl,
            self.low_brl,
            self.close_brl,
            self.volume_brl,
            self.trades,
            self.quantity,
            self.distribution_number,
            self.observed,
        )
        if self.dates.ndim != 1 or any(value.shape != shape for value in arrays):
            raise ValueError("DailyPanel arrays are not aligned")
        if len(set(self.isins)) != len(self.isins) or any(
            not VALID_ISIN.fullmatch(value) for value in self.isins
        ):
            raise ValueError("DailyPanel must have unique valid ISIN identities")
        if self.dates.size and np.any(np.diff(self.dates.astype("datetime64[D]").astype(np.int64)) <= 0):
            raise ValueError("DailyPanel dates must be strictly increasing")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_records(paths: Iterable[Path]) -> list[dict[str, object]]:
    unique = {Path(value).resolve() for value in paths}
    return [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(unique, key=str)
    ]


def _identity_column(frame: pl.DataFrame) -> str:
    if "isin" in frame.columns:
        return "isin"
    if "security_id" in frame.columns:
        return "security_id"
    raise ValueError("COTAHIST data has no ISIN identity column")


def filter_cash_equities(
    daily: pl.DataFrame, *, v1_isins: Sequence[str] = ()
) -> pl.DataFrame:
    """Apply the v2 cash-equity contract and normalize identity to ``isin``."""

    identity = _identity_column(daily)
    spec = "security_spec_base" if "security_spec_base" in daily.columns else "security_spec"
    required = {"trade_date", identity, spec, "market_type"}
    if not required.issubset(daily.columns):
        raise ValueError(f"COTAHIST columns missing: {sorted(required - set(daily.columns))}")
    bdi = "bdi_code" if "bdi_code" in daily.columns else "cod_bdi"
    if bdi not in daily.columns:
        raise ValueError("COTAHIST data has no BDI code")
    allowed = pl.col(spec).is_in(CASH_EQUITY_SPECS)
    if v1_isins:
        allowed |= pl.col(identity).is_in(tuple(v1_isins))
    result = daily.filter(
        (pl.col("market_type").cast(pl.Int64) == 10)
        & (pl.col(bdi).cast(pl.String).str.zfill(2) == "02")
        & allowed
    )
    if identity != "isin":
        result = result.rename({identity: "isin"})
    invalid = result.filter(
        pl.col("isin").is_null()
        | ~pl.col("isin").cast(pl.String).str.contains(VALID_ISIN.pattern)
    )
    if invalid.height:
        raise ValueError("v2 cash-equity rows contain non-ISIN/fallback identities")
    if result.select(pl.struct("trade_date", "isin").n_unique()).item() != result.height:
        raise ValueError("COTAHIST must contain one row per date and ISIN")
    return result.sort("trade_date", "isin")


def load_cotahist(paths: Sequence[Path], *, v1_isins: Sequence[str] = ()) -> pl.DataFrame:
    if not paths:
        raise ValueError("at least one parsed COTAHIST file is required")
    return filter_cash_equities(
        pl.concat((pl.read_parquet(path) for path in paths), how="diagonal_relaxed"),
        v1_isins=v1_isins,
    )


def build_security_master(
    daily: pl.DataFrame, *, succession_links: pl.DataFrame | None = None
) -> pl.DataFrame:
    """Create ticker segments and their audited continuation identity."""

    identity = _identity_column(daily)
    ticker = "ticker" if "ticker" in daily.columns else "latest_ticker"
    if ticker not in daily.columns or "trade_date" not in daily.columns:
        raise ValueError("daily data must contain trade_date, ISIN, and ticker")
    rows: list[dict[str, object]] = []
    frame = daily.select(identity, ticker, "trade_date").sort(identity, "trade_date")
    for key, group in frame.group_by(identity, maintain_order=True):
        isin = str(key[0] if isinstance(key, tuple) else key)
        if not VALID_ISIN.fullmatch(isin):
            raise ValueError(f"Invalid permanent identity: {isin}")
        current_ticker: str | None = None
        first = last = None
        for value_ticker, value_date in group.select(ticker, "trade_date").iter_rows():
            value_ticker = str(value_ticker).strip().upper()
            if not value_ticker:
                raise ValueError(f"Blank ticker for {isin}")
            if current_ticker is not None and value_ticker != current_ticker:
                rows.append(
                    {"isin": isin, "ticker": current_ticker, "first_date": first, "last_date": last}
                )
                first = value_date
            elif current_ticker is None:
                first = value_date
            current_ticker, last = value_ticker, value_date
        if current_ticker is not None:
            rows.append(
                {"isin": isin, "ticker": current_ticker, "first_date": first, "last_date": last}
            )
    master = pl.DataFrame(rows).sort("isin", "first_date")
    links = (
        detect_isin_successions(daily)
        if succession_links is None
        else succession_links
    )
    roots = continuation_identity_axis(
        tuple(master.get_column("isin").unique(maintain_order=True).to_list()),
        links,
    )
    continuation = pl.DataFrame(
        {
            "isin": master.get_column("isin").unique(maintain_order=True),
            "continuation_isin": roots,
        }
    )
    return master.join(continuation, on="isin", how="left", validate="m:1").sort(
        "isin", "first_date"
    )


def detect_isin_successions(daily: pl.DataFrame) -> pl.DataFrame:
    """Detect exact same-ticker ISIN changes on adjacent COTAHIST sessions.

    A successor must make its first appearance on the session immediately after
    the predecessor's final appearance.  This excludes ticker reuse after a
    gap, concurrent identities, and a previously listed ISIN returning under a
    ticker.
    """

    identity = _identity_column(daily)
    ticker = "ticker" if "ticker" in daily.columns else "latest_ticker"
    if ticker not in daily.columns or "trade_date" not in daily.columns:
        raise ValueError("daily data must contain trade_date, ISIN, and ticker")
    frame = (
        daily.select(
            pl.col("trade_date"),
            pl.col(identity).cast(pl.String).alias("isin"),
            pl.col(ticker).cast(pl.String).str.strip_chars().str.to_uppercase().alias("ticker"),
        )
        .filter(pl.col("isin").is_not_null() & (pl.col("ticker") != ""))
        .unique()
        .sort("trade_date", "ticker", "isin")
    )
    schema = {
        "ticker": pl.String,
        "predecessor_isin": pl.String,
        "successor_isin": pl.String,
        "predecessor_last_date": pl.Date,
        "successor_first_date": pl.Date,
        "continuation_isin": pl.String,
    }
    if frame.is_empty():
        return pl.DataFrame(schema=schema)
    ambiguous = (
        frame.group_by("trade_date", "ticker")
        .agg(pl.col("isin").n_unique().alias("identity_count"))
        .filter(pl.col("identity_count") != 1)
    )
    if ambiguous.height:
        raise ValueError("one COTAHIST ticker maps to multiple ISINs on one session")

    calendar = frame.get_column("trade_date").unique().sort().to_list()
    date_index = {value: index for index, value in enumerate(calendar)}
    bounds = frame.group_by("isin").agg(
        pl.col("trade_date").min().alias("first_date"),
        pl.col("trade_date").max().alias("last_date"),
    )
    first = dict(bounds.select("isin", "first_date").iter_rows())
    last = dict(bounds.select("isin", "last_date").iter_rows())
    candidates: list[dict[str, object]] = []
    for key, group in frame.group_by("ticker", maintain_order=True):
        ticker_value = str(key[0] if isinstance(key, tuple) else key)
        observations = group.select("trade_date", "isin").sort("trade_date").iter_rows()
        previous: tuple[object, str] | None = None
        for value_date, value_isin in observations:
            value_isin = str(value_isin)
            if previous is not None:
                prior_date, prior_isin = previous
                if (
                    prior_isin != value_isin
                    and date_index[value_date] == date_index[prior_date] + 1
                    and last[prior_isin] == prior_date
                    and first[value_isin] == value_date
                ):
                    candidates.append(
                        {
                            "ticker": ticker_value,
                            "predecessor_isin": prior_isin,
                            "successor_isin": value_isin,
                            "predecessor_last_date": prior_date,
                            "successor_first_date": value_date,
                        }
                    )
            previous = (value_date, value_isin)
    if not candidates:
        return pl.DataFrame(schema=schema)
    links = pl.DataFrame(candidates).unique().sort("successor_first_date", "ticker")
    if (
        links.get_column("predecessor_isin").n_unique() != links.height
        or links.get_column("successor_isin").n_unique() != links.height
    ):
        raise ValueError("ISIN succession links must be one-to-one")
    roots: dict[str, str] = {}
    for row in links.iter_rows(named=True):
        predecessor = str(row["predecessor_isin"])
        successor = str(row["successor_isin"])
        root = roots.get(predecessor, predecessor)
        if successor == root:
            raise ValueError("ISIN succession links contain a cycle")
        roots[successor] = root
    return links.with_columns(
        pl.col("successor_isin")
        .replace_strict(roots, default=pl.col("successor_isin"))
        .alias("continuation_isin")
    ).select(*schema)


def continuation_identity_axis(
    isins: Sequence[str], links: pl.DataFrame
) -> tuple[str, ...]:
    """Map each permanent ISIN to the root of its audited continuation chain."""

    required = {"predecessor_isin", "successor_isin"}
    if not required.issubset(links.columns):
        if links.is_empty():
            return tuple(isins)
        raise ValueError("ISIN succession table has the wrong schema")
    roots = {str(isin): str(isin) for isin in isins}
    rows = (
        links.sort("successor_first_date").iter_rows(named=True)
        if "successor_first_date" in links.columns
        else links.iter_rows(named=True)
    )
    for row in rows:
        predecessor = str(row["predecessor_isin"])
        successor = str(row["successor_isin"])
        if predecessor not in roots or successor not in roots:
            raise ValueError("ISIN succession link is outside the panel axis")
        roots[successor] = roots[predecessor]
    return tuple(roots[str(isin)] for isin in isins)


def inherit_linked_history(
    values: NDArray[np.generic],
    dates: Sequence[object],
    isins: Sequence[str],
    links: pl.DataFrame,
    *,
    copy: bool = True,
) -> NDArray[np.generic]:
    """Copy predecessor rows into the successor's strictly prior history.

    The successor's first and later observations are never overwritten.  Links
    are applied chronologically so a multi-ISIN chain carries its complete
    causal history forward.
    """

    source = np.asarray(values)
    if source.ndim < 2 or source.shape[:2] != (len(dates), len(isins)):
        raise ValueError("linked-history array is misaligned")
    if links.is_empty():
        return source
    output = source.copy() if copy else source
    if not output.flags.writeable:
        raise ValueError("in-place linked-history destination is read-only")
    date_lookup = {
        np.datetime64(value, "D"): index for index, value in enumerate(dates)
    }
    isin_lookup = {str(value): index for index, value in enumerate(isins)}
    for row in links.sort("successor_first_date").iter_rows(named=True):
        successor_date = np.datetime64(row["successor_first_date"], "D")
        boundary = date_lookup.get(successor_date)
        predecessor = isin_lookup.get(str(row["predecessor_isin"]))
        successor = isin_lookup.get(str(row["successor_isin"]))
        if boundary is None and len(dates) and successor_date < np.datetime64(dates[0], "D"):
            continue
        if boundary is None or predecessor is None or successor is None:
            raise ValueError("ISIN succession link is outside the history axes")
        output[:boundary, successor] = output[:boundary, predecessor]
    return output


def verify_v1_mapping(assignments: pl.DataFrame, available_isins: Sequence[str]) -> pl.DataFrame:
    """Verify and return the exact one-to-one v1 security-id to ISIN mapping."""

    if not {"security_id", "isin"}.issubset(assignments.columns):
        raise ValueError("v1 assignments require security_id and isin")
    mapping = assignments.select("security_id", "isin").unique()
    if mapping.height != assignments.get_column("security_id").n_unique():
        raise ValueError("one v1 security_id maps to multiple ISINs")
    if mapping.get_column("isin").n_unique() != mapping.height:
        raise ValueError("multiple v1 security_ids map to one ISIN")
    invalid = [value for value in mapping.get_column("isin").to_list() if not VALID_ISIN.fullmatch(str(value))]
    if invalid:
        raise ValueError(f"v1 mapping contains invalid ISINs: {invalid}")
    missing = sorted(set(mapping.get_column("isin").to_list()) - set(available_isins))
    if missing:
        raise ValueError(f"v1 ISINs missing from daily foundation: {missing}")
    return mapping.sort("security_id")


def panel_from_daily(
    daily: pl.DataFrame,
    *,
    dates: Sequence[object] | None = None,
    isins: Sequence[str] | None = None,
) -> DailyPanel:
    identity = _identity_column(daily)
    calendar = tuple(dates) if dates is not None else session_calendar(daily)
    security_axis = tuple(sorted(isins if isins is not None else daily.get_column(identity).unique().to_list()))
    if len(set(security_axis)) != len(security_axis):
        raise ValueError("duplicate ISIN on requested axis")
    date_lookup = {value: index for index, value in enumerate(calendar)}
    isin_lookup = {value: index for index, value in enumerate(security_axis)}
    shape = (len(calendar), len(security_axis))
    values = {
        name: np.full(shape, np.nan, dtype=np.float64)
        for name in (
            "open_brl",
            "high_brl",
            "low_brl",
            "close_brl",
            "volume_brl",
            "trades",
            "quantity",
            "distribution_number",
        )
    }
    observed = np.zeros(shape, dtype=np.bool_)
    needed = {"trade_date", identity, *(set(values) - {"distribution_number"})}
    if not needed.issubset(daily.columns):
        raise ValueError(f"daily panel columns missing: {sorted(needed - set(daily.columns))}")
    selected = daily
    if "distribution_number" not in selected.columns:
        selected = selected.with_columns(
            pl.lit(None, dtype=pl.Int64).alias("distribution_number")
        )
    for row in selected.select("trade_date", identity, *values).iter_rows(named=True):
        date_index = date_lookup.get(row["trade_date"])
        isin_index = isin_lookup.get(row[identity])
        if date_index is None or isin_index is None:
            continue
        if observed[date_index, isin_index]:
            raise ValueError("duplicate date/ISIN daily row")
        for name in values:
            value = row[name]
            values[name][date_index, isin_index] = np.nan if value is None else float(value)
        observed[date_index, isin_index] = True
    return DailyPanel(
        dates=np.asarray(calendar, dtype="datetime64[D]"),
        isins=security_axis,
        observed=observed,
        **values,
    )
