from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl
from numpy.typing import NDArray


ACTION_TYPES = frozenset(
    {"split", "reverse_split", "bonus", "dividend", "jcp", "subscription_rights"}
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class AdjustmentResult:
    price_factor: NDArray[np.float64]
    total_return_factor: NDArray[np.float64]
    adjusted_open: NDArray[np.float64]
    adjusted_high: NDArray[np.float64]
    adjusted_low: NDArray[np.float64]
    adjusted_close: NDArray[np.float64]
    total_return_close: NDArray[np.float64]
    unresolved: NDArray[np.bool_]


def normalize_yfinance_actions(
    actions: pl.DataFrame,
    *,
    isin: str,
    ticker: str,
    fetched_at: datetime,
) -> pl.DataFrame:
    """Convert yfinance's dividends/splits frame into the canonical action rows."""

    date_column = "Date" if "Date" in actions.columns else "date"
    dividend_column = "Dividends" if "Dividends" in actions.columns else "dividends"
    split_column = "Stock Splits" if "Stock Splits" in actions.columns else "stock_splits"
    required = {date_column, dividend_column, split_column}
    if not required.issubset(actions.columns):
        raise ValueError("yfinance action frame lacks Date/Dividends/Stock Splits")
    rows: list[dict[str, object]] = []
    for raw_date, raw_dividend, raw_split in actions.select(
        date_column, dividend_column, split_column
    ).iter_rows():
        ex_date = raw_date.date() if isinstance(raw_date, datetime) else raw_date
        dividend = _finite_or_zero(raw_dividend)
        split = _finite_or_zero(raw_split)
        if split:
            rows.append(
                {
                    "isin": isin,
                    "ex_date": ex_date,
                    "action_type": "split" if split >= 1 else "reverse_split",
                    "split_factor": split,
                    "cash_distribution_brl": 0.0,
                    "known_date": ex_date,
                    "source_ticker": ticker,
                    "source": "yfinance.actions",
                    "fetched_at": fetched_at,
                    "unresolved": False,
                }
            )
        if dividend:
            rows.append(
                {
                    "isin": isin,
                    "ex_date": ex_date,
                    "action_type": "dividend",
                    "split_factor": 1.0,
                    "cash_distribution_brl": dividend,
                    "known_date": ex_date,
                    "source_ticker": ticker,
                    "source": "yfinance.actions",
                    "fetched_at": fetched_at,
                    "unresolved": False,
                }
            )
    schema = {
        "isin": pl.String,
        "ex_date": pl.Date,
        "action_type": pl.String,
        "split_factor": pl.Float64,
        "cash_distribution_brl": pl.Float64,
        "known_date": pl.Date,
        "source_ticker": pl.String,
        "source": pl.String,
        "fetched_at": pl.Datetime("us"),
        "unresolved": pl.Boolean,
    }
    return pl.DataFrame(rows, schema=schema)


def _empty_yfinance_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={"date": pl.Date, "dividends": pl.Float64, "stock_splits": pl.Float64}
    )


def _finite_or_zero(value: object) -> float:
    try:
        number = float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0
    return number if np.isfinite(number) else 0.0


def _extract_yfinance_actions(raw: Any, symbol: str) -> pl.DataFrame | None:
    """Extract one symbol from either orientation of yfinance MultiIndex output."""

    if raw is None or raw.empty:
        return None
    frame = raw
    columns = raw.columns
    if getattr(columns, "nlevels", 1) > 1:
        level_zero = set(columns.get_level_values(0))
        level_one = set(columns.get_level_values(1))
        if symbol in level_zero:
            frame = raw[symbol]
        elif symbol in level_one:
            frame = raw.xs(symbol, axis=1, level=1)
        else:
            return None
    if frame.empty:
        return None
    price_columns = [
        column
        for column in ("Open", "High", "Low", "Close", "Adj Close")
        if column in frame.columns
    ]
    price_values = (
        np.asarray(frame[price_columns].to_numpy(), dtype=np.float64)
        if price_columns
        else np.empty(0, dtype=np.float64)
    )
    if price_columns and not np.isfinite(price_values).any():
        # yfinance keeps failed symbols in a batch MultiIndex, filling their
        # complete price frame with NaN.  That is a provider failure, not proof
        # that a live symbol had genuinely zero actions.
        return None
    rows: list[dict[str, object]] = []
    for timestamp, record in frame.iterrows():
        dividend = _finite_or_zero(record.get("Dividends", 0.0))
        split = _finite_or_zero(record.get("Stock Splits", 0.0))
        if dividend or split:
            rows.append(
                {
                    "date": timestamp.date(),
                    "dividends": dividend,
                    "stock_splits": split,
                }
            )
    return pl.DataFrame(
        rows,
        schema={"date": pl.Date, "dividends": pl.Float64, "stock_splits": pl.Float64},
    )


def _download_yfinance_batch(
    tickers: Sequence[str], *, start: date, end: date
) -> dict[str, pl.DataFrame]:
    """Download one bounded symbol batch; callers isolate and record failures."""

    import yfinance as yf

    symbols = [f"{ticker}.SA" for ticker in tickers]
    raw = yf.download(
        symbols,
        start=start.isoformat(),
        end=end.isoformat(),
        actions=True,
        auto_adjust=False,
        group_by="ticker",
        threads=True,
        progress=False,
    )
    extracted = {
        ticker: _extract_yfinance_actions(raw, symbol)
        for ticker, symbol in zip(tickers, symbols, strict=True)
    }
    return {ticker: frame for ticker, frame in extracted.items() if frame is not None}


def acquire_yfinance_actions(
    security_master: pl.DataFrame,
    cache_dir: Path,
    *,
    refresh: bool = False,
    fetched_at: datetime | None = None,
    batch_size: int = 64,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Fetch and immutably cache yfinance actions for every ticker segment.

    The segment-specific cache key prevents a historical ticker from being bound
    outside its observed ISIN interval. Existing cache files are reused unless
    ``refresh`` is explicit. The returned audit records exact cache hashes.
    """

    required = {"isin", "ticker", "first_date", "last_date"}
    if not required.issubset(security_master.columns):
        raise ValueError(
            f"security master columns missing: {sorted(required - set(security_master.columns))}"
        )
    if (
        security_master.select(
            pl.struct("isin", "ticker", "first_date", "last_date").n_unique()
        ).item()
        != security_master.height
    ):
        raise ValueError("security master contains duplicate ticker segments")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    cache = Path(cache_dir).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    timestamp = fetched_at or datetime.now(timezone.utc)
    timestamp_key = timestamp.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    frames: list[pl.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    rows = list(
        security_master.sort("isin", "first_date", "ticker").iter_rows(named=True)
    )
    pending: list[dict[str, object]] = []
    cached_paths: dict[str, Path] = {}
    identities: dict[str, str] = {}
    for row in rows:
        identity = "|".join(
            (
                str(row["isin"]),
                str(row["ticker"]),
                row["first_date"].isoformat(),
                row["last_date"].isoformat(),
            )
        )
        key = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        identities[identity] = key
        existing = sorted(cache.glob(f"{row['ticker']}_{key}_*.parquet"))
        legacy = cache / f"{row['ticker']}_{key}.parquet"
        if legacy.exists():
            existing.append(legacy)
        if existing and not refresh:
            cached_paths[identity] = sorted(existing)[-1]
        else:
            pending.append(row)

    downloaded: dict[str, pl.DataFrame] = {}
    failures: dict[str, str] = {}
    unique_tickers = sorted({str(row["ticker"]) for row in pending})
    bounds = {
        ticker: (
            min(row["first_date"] for row in pending if row["ticker"] == ticker),
            max(row["last_date"] for row in pending if row["ticker"] == ticker),
        )
        for ticker in unique_tickers
    }
    for offset in range(0, len(unique_tickers), batch_size):
        batch = unique_tickers[offset : offset + batch_size]
        start = min(bounds[ticker][0] for ticker in batch)
        end = max(bounds[ticker][1] for ticker in batch) + timedelta(days=1)
        try:
            downloaded.update(_download_yfinance_batch(batch, start=start, end=end))
        except Exception as error:  # provider failures are recorded per symbol
            for ticker in batch:
                failures[ticker] = f"{type(error).__name__}: {error}"[:1000]

    for row in rows:
        identity = "|".join(
            (
                str(row["isin"]),
                str(row["ticker"]),
                row["first_date"].isoformat(),
                row["last_date"].isoformat(),
            )
        )
        ticker = str(row["ticker"])
        path = cached_paths.get(identity)
        status = "cache_hit"
        error = None
        if path is None:
            raw_frame = downloaded.get(ticker)
            if raw_frame is None:
                error = failures.get(ticker, "symbol missing from provider response")
                audit_rows.append(
                    {
                        "isin": row["isin"],
                        "ticker": ticker,
                        "first_date": row["first_date"],
                        "last_date": row["last_date"],
                        "cache_path": None,
                        "cache_sha256": None,
                        "action_rows": 0,
                        "status": "failed",
                        "error": error,
                    }
                )
                continue
            raw_frame = raw_frame.filter(
                pl.col("date").is_between(
                    row["first_date"], row["last_date"], closed="both"
                )
            )
            frame = normalize_yfinance_actions(
                raw_frame,
                isin=str(row["isin"]),
                ticker=ticker,
                fetched_at=timestamp,
            )
            key = identities[identity]
            path = cache / f"{ticker}_{key}_{timestamp_key}.parquet"
            if path.exists():
                raise FileExistsError(f"immutable action cache already exists: {path}")
            temporary = path.with_suffix(".parquet.tmp")
            frame.write_parquet(temporary)
            temporary.replace(path)
            status = "zero_actions" if frame.is_empty() else "downloaded"
        frame = pl.read_parquet(path)
        frames.append(frame)
        audit_rows.append(
            {
                "isin": row["isin"],
                "ticker": ticker,
                "first_date": row["first_date"],
                "last_date": row["last_date"],
                "cache_path": str(path),
                "cache_sha256": _sha256(path),
                "action_rows": frame.height,
                "status": status,
                "error": error,
            }
        )
    if frames:
        combined = pl.concat(frames, how="vertical_relaxed")
    else:
        combined = normalize_yfinance_actions(
            _empty_yfinance_frame(),
            isin="BR0000000000",
            ticker="EMPTY",
            fetched_at=timestamp,
        )
    return validate_action_table(combined), pl.DataFrame(audit_rows)


def validate_action_table(actions: pl.DataFrame) -> pl.DataFrame:
    required = {
        "isin",
        "ex_date",
        "action_type",
        "split_factor",
        "cash_distribution_brl",
        "unresolved",
    }
    if not required.issubset(actions.columns):
        raise ValueError(f"action columns missing: {sorted(required - set(actions.columns))}")
    invalid_types = set(actions.get_column("action_type").unique()) - ACTION_TYPES
    if invalid_types:
        raise ValueError(f"unknown action types: {sorted(invalid_types)}")
    if actions.filter(
        (pl.col("split_factor") <= 0)
        | ~pl.col("split_factor").is_finite()
        | (pl.col("cash_distribution_brl") < 0)
        | ~pl.col("cash_distribution_brl").is_finite()
    ).height:
        raise ValueError("action factors/distributions must be finite and non-negative")
    if "known_date" in actions.columns and actions.filter(
        pl.col("known_date").is_not_null() & (pl.col("known_date") > pl.col("ex_date"))
    ).height:
        raise ValueError("an action cannot be treated as known after its ex-date")
    return actions.sort("isin", "ex_date", "action_type")


def align_action_arrays(
    actions: pl.DataFrame,
    dates: Sequence[date | np.datetime64],
    isins: Sequence[str],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Aggregate action rows onto a date/ISIN axis."""

    checked = validate_action_table(actions)
    normalized_dates = tuple(
        value.astype(object) if isinstance(value, np.datetime64) else value for value in dates
    )
    date_lookup = {value: index for index, value in enumerate(normalized_dates)}
    isin_lookup = {value: index for index, value in enumerate(isins)}
    shape = (len(normalized_dates), len(isins))
    split = np.ones(shape, dtype=np.float64)
    cash = np.zeros(shape, dtype=np.float64)
    unresolved = np.zeros(shape, dtype=np.bool_)
    for row in checked.iter_rows(named=True):
        date_index = date_lookup.get(row["ex_date"])
        isin_index = isin_lookup.get(row["isin"])
        if date_index is None or isin_index is None:
            continue
        split[date_index, isin_index] *= float(row["split_factor"])
        cash[date_index, isin_index] += float(row["cash_distribution_brl"])
        unresolved[date_index, isin_index] |= bool(row["unresolved"])
    return split, cash, unresolved


def action_presence_array(
    actions: pl.DataFrame,
    dates: Sequence[date | np.datetime64],
    isins: Sequence[str],
) -> NDArray[np.bool_]:
    """Mark every recorded action, including neutral-factor rights/bonuses."""

    checked = validate_action_table(actions)
    normalized_dates = tuple(_as_date(value) for value in dates)
    date_lookup = {value: index for index, value in enumerate(normalized_dates)}
    isin_lookup = {value: index for index, value in enumerate(isins)}
    present = np.zeros((len(normalized_dates), len(isins)), dtype=np.bool_)
    for ex_date, isin in checked.select("ex_date", "isin").iter_rows():
        date_index = date_lookup.get(ex_date)
        isin_index = isin_lookup.get(isin)
        if date_index is not None and isin_index is not None:
            present[date_index, isin_index] = True
    return present


def provider_failure_mask(
    acquisition_audit: pl.DataFrame,
    dates: Sequence[date | np.datetime64],
    isins: Sequence[str],
    observed: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    """Conservatively mark observed dates in provider-failed ticker segments."""

    required = {"isin", "first_date", "last_date", "status"}
    if not required.issubset(acquisition_audit.columns):
        raise ValueError(
            "action acquisition audit columns missing: "
            f"{sorted(required - set(acquisition_audit.columns))}"
        )
    normalized_dates = np.asarray(dates, dtype="datetime64[D]")
    seen = np.asarray(observed, dtype=np.bool_)
    if seen.shape != (len(normalized_dates), len(isins)):
        raise ValueError("provider failure mask axes are misaligned")
    isin_lookup = {value: index for index, value in enumerate(isins)}
    failed = np.zeros(seen.shape, dtype=np.bool_)
    for row in acquisition_audit.filter(pl.col("status") == "failed").iter_rows(
        named=True
    ):
        isin_index = isin_lookup.get(str(row["isin"]))
        if isin_index is None:
            continue
        first = np.datetime64(_as_date(row["first_date"]), "D")
        last = np.datetime64(_as_date(row["last_date"]), "D")
        within = (normalized_dates >= first) & (normalized_dates <= last)
        failed[within, isin_index] = seen[within, isin_index]
    return failed


def detect_distribution_changes(
    distribution_number: NDArray[np.floating],
    observed: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    """Mark a COTAHIST distribution-number change at the next observed row."""

    distribution = np.asarray(distribution_number, dtype=np.float64)
    seen = np.asarray(observed, dtype=np.bool_)
    if distribution.shape != seen.shape or distribution.ndim != 2:
        raise ValueError("distribution-number inputs must align [date, name]")
    changed = np.zeros(seen.shape, dtype=np.bool_)
    for name in range(distribution.shape[1]):
        prior: float | None = None
        for day in np.flatnonzero(seen[:, name]):
            current = distribution[day, name]
            if not np.isfinite(current):
                continue
            if prior is not None and current != prior:
                changed[day, name] = True
            prior = float(current)
    return changed


def distribution_review_table(
    dates: Sequence[date | np.datetime64],
    isins: Sequence[str],
    changed: NDArray[np.bool_],
    recorded_action: NDArray[np.bool_],
) -> pl.DataFrame:
    change = np.asarray(changed, dtype=np.bool_)
    recorded = np.asarray(recorded_action, dtype=np.bool_)
    if change.shape != recorded.shape or change.shape != (len(dates), len(isins)):
        raise ValueError("distribution review axes are misaligned")
    rows = [
        {
            "date": _as_date(dates[day]),
            "isin": isins[name],
            "recorded_action": bool(recorded[day, name]),
            "unresolved": not bool(recorded[day, name]),
        }
        for day, name in np.argwhere(change)
    ]
    return pl.DataFrame(
        rows,
        schema={
            "date": pl.Date,
            "isin": pl.String,
            "recorded_action": pl.Boolean,
            "unresolved": pl.Boolean,
        },
    )


def causal_adjustment_factors(
    raw_close: NDArray[np.floating],
    split_factor: NDArray[np.floating],
    cash_distribution_brl: NDArray[np.floating],
    unresolved: NDArray[np.bool_] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute forward-only price and total-return factors.

    Historical rows are never rewritten when a later action occurs. A split of
    ``s`` multiplies the factor from its ex-date onward. A cash distribution
    multiplies the total-return factor by ``1 + cash / raw_close`` on its ex-date.
    """

    close = np.asarray(raw_close, dtype=np.float64)
    split = np.asarray(split_factor, dtype=np.float64)
    cash = np.asarray(cash_distribution_brl, dtype=np.float64)
    bad = np.zeros(close.shape, dtype=np.bool_) if unresolved is None else np.asarray(unresolved, dtype=np.bool_)
    if close.ndim != 2 or split.shape != close.shape or cash.shape != close.shape or bad.shape != close.shape:
        raise ValueError("daily action arrays must be aligned [date, name]")
    if np.any(~np.isfinite(split) | (split <= 0)) or np.any(~np.isfinite(cash) | (cash < 0)):
        raise ValueError("invalid action factor arrays")
    price = np.ones(close.shape, dtype=np.float64)
    distribution = np.ones(close.shape, dtype=np.float64)
    for date_index in range(close.shape[0]):
        if date_index:
            price[date_index] = price[date_index - 1]
            distribution[date_index] = distribution[date_index - 1]
        # ``unresolved`` is normally a target-eligibility flag, not permission
        # to erase a recorded action. The sole exception is a cash event for
        # which this exact ISIN has no positive ex-date close: the specified
        # reinvestment factor is then unknowable. The caller must flag that
        # cell unresolved; substituting another session's close would change
        # the return definition.
        price[date_index] *= split[date_index]
        has_cash = cash[date_index] > 0
        invalid_cash = has_cash & (~np.isfinite(close[date_index]) | (close[date_index] <= 0))
        unflagged_invalid = invalid_cash & ~bad[date_index]
        if unflagged_invalid.any():
            slots = np.flatnonzero(unflagged_invalid).tolist()
            raise ValueError(f"cash distribution has no positive ex-date close at slots {slots}")
        has_cash &= ~invalid_cash
        distribution[date_index, has_cash] *= 1.0 + cash[date_index, has_cash] / close[date_index, has_cash]
    return price, price * distribution


def cash_reinvestment_unavailable_mask(
    raw_close: NDArray[np.floating],
    cash_distribution_brl: NDArray[np.floating],
) -> NDArray[np.bool_]:
    """Flag cash events lacking the exact positive ex-date close."""

    close = np.asarray(raw_close, dtype=np.float64)
    cash = np.asarray(cash_distribution_brl, dtype=np.float64)
    if close.ndim != 2 or cash.shape != close.shape:
        raise ValueError("cash-reinvestment inputs must align [date, name]")
    return (cash > 0) & (~np.isfinite(close) | (close <= 0))


def cash_reinvestment_review_table(
    dates: Sequence[date | np.datetime64],
    isins: Sequence[str],
    cash_distribution_brl: NDArray[np.floating],
    raw_close: NDArray[np.floating],
    observed: NDArray[np.bool_] | None = None,
) -> pl.DataFrame:
    """Retain recorded cash events whose exact reinvestment close is absent."""

    cash = np.asarray(cash_distribution_brl, dtype=np.float64)
    close = np.asarray(raw_close, dtype=np.float64)
    seen = (
        np.isfinite(close)
        if observed is None
        else np.asarray(observed, dtype=np.bool_)
    )
    unavailable = cash_reinvestment_unavailable_mask(close, cash)
    if (
        unavailable.shape != (len(dates), len(isins))
        or seen.shape != unavailable.shape
    ):
        raise ValueError("cash-reinvestment axes are misaligned")
    rows = [
        {
            "trade_date": _as_date(dates[day]),
            "isin": str(isins[name]),
            "cash_distribution_brl": float(cash[day, name]),
            "raw_close_brl": (
                float(close[day, name]) if np.isfinite(close[day, name]) else None
            ),
            "observed": bool(seen[day, name]),
            "unresolved": True,
            "status": "unresolved_missing_ex_date_close",
        }
        for day, name in np.argwhere(unavailable)
    ]
    return pl.DataFrame(
        rows,
        schema={
            "trade_date": pl.Date,
            "isin": pl.String,
            "cash_distribution_brl": pl.Float64,
            "raw_close_brl": pl.Float64,
            "observed": pl.Boolean,
            "unresolved": pl.Boolean,
            "status": pl.String,
        },
    )


def adjust_daily_ohlc(
    raw_open: NDArray[np.floating],
    raw_high: NDArray[np.floating],
    raw_low: NDArray[np.floating],
    raw_close: NDArray[np.floating],
    split_factor: NDArray[np.floating],
    cash_distribution_brl: NDArray[np.floating],
    unresolved: NDArray[np.bool_] | None = None,
) -> AdjustmentResult:
    close = np.asarray(raw_close, dtype=np.float64)
    arrays = tuple(np.asarray(value, dtype=np.float64) for value in (raw_open, raw_high, raw_low))
    if any(value.shape != close.shape for value in arrays):
        raise ValueError("OHLC arrays are misaligned")
    bad = np.zeros(close.shape, dtype=np.bool_) if unresolved is None else np.asarray(unresolved, dtype=np.bool_)
    price_factor, total_return_factor = causal_adjustment_factors(
        close, split_factor, cash_distribution_brl, bad
    )
    return AdjustmentResult(
        price_factor=price_factor,
        total_return_factor=total_return_factor,
        adjusted_open=arrays[0] * price_factor,
        adjusted_high=arrays[1] * price_factor,
        adjusted_low=arrays[2] * price_factor,
        adjusted_close=close * price_factor,
        total_return_close=close * total_return_factor,
        unresolved=bad.copy(),
    )


def detect_split_candidates(
    raw_close: NDArray[np.floating],
    quantity: NDArray[np.floating],
    observed: NDArray[np.bool_],
    *,
    log_price_jump: float = 0.35,
    maximum_log_offset_error: float = 0.20,
) -> NDArray[np.bool_]:
    """Flag price jumps with approximately offsetting traded-quantity jumps.

    This is deliberately an audit signal, not an inferred adjustment factor.
    """

    close = np.asarray(raw_close, dtype=np.float64)
    qty = np.asarray(quantity, dtype=np.float64)
    seen = np.asarray(observed, dtype=np.bool_)
    if close.shape != qty.shape or close.shape != seen.shape or close.ndim != 2:
        raise ValueError("split-detection inputs must be aligned [date, name]")
    output = np.zeros(close.shape, dtype=np.bool_)
    price_change = np.log(close[1:] / close[:-1])
    quantity_change = np.log(qty[1:] / qty[:-1])
    comparable = (
        seen[1:]
        & seen[:-1]
        & np.isfinite(price_change)
        & np.isfinite(quantity_change)
        & (close[1:] > 0)
        & (close[:-1] > 0)
        & (qty[1:] > 0)
        & (qty[:-1] > 0)
    )
    output[1:] = (
        comparable
        & (np.abs(price_change) > log_price_jump)
        & (np.abs(price_change + quantity_change) <= maximum_log_offset_error)
    )
    return output


def split_review_table(
    dates: Sequence[date],
    isins: Sequence[str],
    detected: NDArray[np.bool_],
    recorded_split_factor: NDArray[np.floating],
) -> pl.DataFrame:
    detected = np.asarray(detected, dtype=np.bool_)
    recorded = np.asarray(recorded_split_factor, dtype=np.float64)
    if detected.shape != recorded.shape or detected.shape != (len(dates), len(isins)):
        raise ValueError("split review axes are misaligned")
    rows: list[dict[str, object]] = []
    for date_index, isin_index in np.argwhere(detected | (recorded != 1.0)):
        has_record = bool(recorded[date_index, isin_index] != 1.0)
        has_detection = bool(detected[date_index, isin_index])
        rows.append(
            {
                "date": dates[date_index],
                "isin": isins[isin_index],
                "detected_jump": has_detection,
                "recorded_split_factor": float(recorded[date_index, isin_index]),
                "agreement": has_detection == has_record,
                "unresolved": has_detection != has_record,
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "date": pl.Date,
            "isin": pl.String,
            "detected_jump": pl.Boolean,
            "recorded_split_factor": pl.Float64,
            "agreement": pl.Boolean,
            "unresolved": pl.Boolean,
        },
    )


def action_coverage_table(
    actions: pl.DataFrame,
    dates: Sequence[date | np.datetime64],
    isins: Sequence[str],
    acquisition_audit: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Report action counts and segment-level provider status per ISIN-year.

    A successful query with no rows is reported as a true provider zero; it is
    never conflated with a failed ticker-segment request.  The source taxonomy
    is disclosed because Yahoo's free actions surface does not separately
    guarantee bonus, JCP, or subscription-right classifications.
    """

    checked = validate_action_table(actions)
    years = sorted({_as_date(value).year for value in dates})
    counts: dict[tuple[str, int], tuple[int, int]] = {}
    for row in checked.iter_rows(named=True):
        key = (str(row["isin"]), row["ex_date"].year)
        total, unresolved = counts.get(key, (0, 0))
        counts[key] = (total + 1, unresolved + int(bool(row["unresolved"])))
    audit_rows = (
        list(acquisition_audit.iter_rows(named=True))
        if acquisition_audit is not None
        else []
    )
    if acquisition_audit is not None and not {
        "isin",
        "first_date",
        "last_date",
        "status",
    }.issubset(acquisition_audit.columns):
        raise ValueError("corporate-action acquisition audit has wrong schema")
    rows: list[dict[str, object]] = []
    for isin in isins:
        for year in years:
            segments = [
                row
                for row in audit_rows
                if str(row["isin"]) == isin
                and _as_date(row["first_date"]).year <= year
                and _as_date(row["last_date"]).year >= year
            ]
            failed = sum(str(row["status"]) == "failed" for row in segments)
            succeeded = len(segments) - failed
            zeros = sum(
                str(row["status"]) != "failed"
                and int(row.get("action_rows") or 0) == 0
                for row in segments
            )
            action_count, unresolved_count = counts.get((isin, year), (0, 0))
            if failed and succeeded:
                status = "partial_provider_failure"
            elif failed:
                status = "provider_failure"
            elif succeeded and action_count:
                status = "covered_actions"
            elif succeeded:
                status = "covered_zero_actions"
            else:
                status = "no_acquisition_segment"
            rows.append(
                {
                "isin": isin,
                "year": year,
                "action_count": action_count,
                "unresolved_count": unresolved_count,
                "ticker_segment_count": len(segments),
                "successful_segment_count": succeeded,
                "zero_action_segment_count": zeros,
                "failed_segment_count": failed,
                "acquisition_status": status,
                "provider_taxonomy": "yfinance dividends and stock splits",
                "source_limitations": (
                    "bonus, JCP, and subscription-right taxonomy is not "
                    "guaranteed by the provider and unresolved source gaps "
                    "remain target-masked"
                ),
            }
            )
    return pl.DataFrame(
        rows,
        schema={
            "isin": pl.String,
            "year": pl.Int32,
            "action_count": pl.Int32,
            "unresolved_count": pl.Int32,
            "ticker_segment_count": pl.Int32,
            "successful_segment_count": pl.Int32,
            "zero_action_segment_count": pl.Int32,
            "failed_segment_count": pl.Int32,
            "acquisition_status": pl.String,
            "provider_taxonomy": pl.String,
            "source_limitations": pl.String,
        },
    )


def audit_m1_adjustment_status(
    dates: Sequence[date | np.datetime64],
    isins: Sequence[str],
    m1_session_close: NDArray[np.floating],
    raw_daily_close: NDArray[np.floating],
    adjusted_daily_close: NDArray[np.floating],
    split_factor: NDArray[np.floating],
    cash_distribution_brl: NDArray[np.floating],
    *,
    relative_tolerance: float = 2e-3,
) -> pl.DataFrame:
    """Classify M1 units using both level and pre/post event-day ratios."""

    minute = np.asarray(m1_session_close, dtype=np.float64)
    raw = np.asarray(raw_daily_close, dtype=np.float64)
    adjusted = np.asarray(adjusted_daily_close, dtype=np.float64)
    split = np.asarray(split_factor, dtype=np.float64)
    cash = np.asarray(cash_distribution_brl, dtype=np.float64)
    shape = (len(dates), len(isins))
    if any(value.shape != shape for value in (minute, raw, adjusted, split, cash)):
        raise ValueError("M1 adjustment-audit axes are misaligned")
    rows: list[dict[str, object]] = []
    for day, name in np.argwhere((split != 1.0) | (cash > 0)):
        prior_day = next(
            (
                candidate
                for candidate in range(day - 1, -1, -1)
                if all(
                    np.isfinite(value[candidate, name])
                    and value[candidate, name] > 0
                    for value in (minute, raw, adjusted)
                )
            ),
            None,
        )
        comparable = (
            np.isfinite(minute[day, name])
            and np.isfinite(raw[day, name])
            and np.isfinite(adjusted[day, name])
            and minute[day, name] > 0
            and raw[day, name] > 0
            and adjusted[day, name] > 0
        )
        raw_error = (
            abs(minute[day, name] / raw[day, name] - 1.0) if comparable else np.nan
        )
        adjusted_error = (
            abs(minute[day, name] / adjusted[day, name] - 1.0)
            if comparable
            else np.nan
        )
        ratio_comparable = comparable and prior_day is not None
        m1_event_ratio = (
            minute[day, name] / minute[prior_day, name]
            if ratio_comparable
            else np.nan
        )
        raw_event_ratio = (
            raw[day, name] / raw[prior_day, name]
            if ratio_comparable
            else np.nan
        )
        adjusted_event_ratio = (
            adjusted[day, name] / adjusted[prior_day, name]
            if ratio_comparable
            else np.nan
        )
        raw_ratio_error = (
            abs(m1_event_ratio / raw_event_ratio - 1.0)
            if ratio_comparable and raw_event_ratio > 0
            else np.nan
        )
        adjusted_ratio_error = (
            abs(m1_event_ratio / adjusted_event_ratio - 1.0)
            if ratio_comparable and adjusted_event_ratio > 0
            else np.nan
        )
        if not comparable:
            status = "missing"
        elif ratio_comparable and raw_ratio_error <= relative_tolerance and raw_ratio_error <= adjusted_ratio_error:
            status = "raw_unadjusted"
        elif ratio_comparable and adjusted_ratio_error <= relative_tolerance:
            status = "price_adjusted"
        elif raw_error <= relative_tolerance and raw_error <= adjusted_error:
            status = "raw_unadjusted"
        elif adjusted_error <= relative_tolerance:
            status = "price_adjusted"
        else:
            status = "mismatch"
        rows.append(
            {
                "trade_date": _as_date(dates[day]),
                "isin": isins[name],
                "split_factor": float(split[day, name]),
                "cash_distribution_brl": float(cash[day, name]),
                "m1_close": float(minute[day, name]),
                "raw_daily_close": float(raw[day, name]),
                "adjusted_daily_close": float(adjusted[day, name]),
                "raw_relative_error": float(raw_error),
                "adjusted_relative_error": float(adjusted_error),
                "prior_trade_date": (
                    _as_date(dates[prior_day]) if prior_day is not None else None
                ),
                "m1_pre_post_ratio": float(m1_event_ratio),
                "raw_daily_pre_post_ratio": float(raw_event_ratio),
                "adjusted_daily_pre_post_ratio": float(adjusted_event_ratio),
                "raw_ratio_relative_error": float(raw_ratio_error),
                "adjusted_ratio_relative_error": float(adjusted_ratio_error),
                "status": status,
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "trade_date": pl.Date,
            "isin": pl.String,
            "split_factor": pl.Float64,
            "cash_distribution_brl": pl.Float64,
            "m1_close": pl.Float64,
            "raw_daily_close": pl.Float64,
            "adjusted_daily_close": pl.Float64,
            "raw_relative_error": pl.Float64,
            "adjusted_relative_error": pl.Float64,
            "prior_trade_date": pl.Date,
            "m1_pre_post_ratio": pl.Float64,
            "raw_daily_pre_post_ratio": pl.Float64,
            "adjusted_daily_pre_post_ratio": pl.Float64,
            "raw_ratio_relative_error": pl.Float64,
            "adjusted_ratio_relative_error": pl.Float64,
            "status": pl.String,
        },
    )


def _as_date(value: date | np.datetime64) -> date:
    if isinstance(value, np.datetime64):
        return value.astype("datetime64[D]").astype(object)
    return value


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Acquire cached ISIN-bound yfinance corporate actions")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--security-master", type=Path)
    source.add_argument("--cotahist-root", type=Path)
    parser.add_argument("--v1-assignments", type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> None:
    args = parse_args(arguments)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    input_paths: list[Path]
    if args.security_master:
        security_master = pl.read_parquet(args.security_master)
        input_paths = [args.security_master]
    else:
        from brazil_rv.preprocessing.contract import EXPECTED_EQUITIES

        from .contract import COTAHIST_YEARS
        from .data_foundation import (
            build_security_master,
            filter_cash_equities,
            load_cotahist,
        )

        if args.v1_assignments is None:
            raise ValueError(
                "--cotahist-root acquisition requires --v1-assignments for "
                "the frozen cash-equity exceptions"
            )
        if args.v1_assignments.is_dir():
            from brazil_rv.preprocessing.io import load_assignments

            assignments = load_assignments(args.v1_assignments)
            assignment_path = (
                args.v1_assignments / "xp_accepted_source_assignments_v1.parquet"
            )
        else:
            assignments = pl.read_parquet(args.v1_assignments)
            assignment_path = args.v1_assignments
        if (
            assignments.height != EXPECTED_EQUITIES
            or assignments.get_column("isin").n_unique() != EXPECTED_EQUITIES
        ):
            raise ValueError(
                f"v1 assignments must bind exactly {EXPECTED_EQUITIES} ISINs"
            )

        input_paths = sorted(
            args.cotahist_root.glob("year=*/equities_daily_*.parquet")
        )
        v1_isins = tuple(assignments.get_column("isin").to_list())
        daily = load_cotahist(input_paths, v1_isins=v1_isins).filter(
            pl.col("trade_date").dt.year().is_in(COTAHIST_YEARS)
        )
        daily = filter_cash_equities(daily, v1_isins=v1_isins)
        security_master = build_security_master(daily)
        input_paths.append(assignment_path)
    actions, audit = acquire_yfinance_actions(
        security_master,
        args.cache_dir,
        refresh=args.refresh,
    )
    master_path = output / "security_master.parquet"
    actions_path = output / "corporate_actions.parquet"
    audit_path = output / "yfinance_acquisition_audit.parquet"
    security_master.write_parquet(master_path)
    actions.write_parquet(actions_path)
    audit.write_parquet(audit_path)
    manifest = {
        "schema": "V2_CORPORATE_ACTIONS_V1",
        "provider_taxonomy": "yfinance dividends and stock splits",
        "source_limitations": (
            "bonus, JCP, and subscription-right taxonomy is not guaranteed "
            "by the provider; failed ticker segments remain unresolved"
        ),
        "acquisition_status_counts": {
            str(row["status"]): int(row["len"])
            for row in audit.group_by("status").len().iter_rows(named=True)
        },
        "security_master": {
            "path": master_path.name,
            "rows": security_master.height,
            "sha256": _sha256(master_path),
        },
        "actions": {"path": actions_path.name, "rows": actions.height, "sha256": _sha256(actions_path)},
        "acquisition_audit": {"path": audit_path.name, "rows": audit.height, "sha256": _sha256(audit_path)},
        "inputs": [
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in input_paths
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
