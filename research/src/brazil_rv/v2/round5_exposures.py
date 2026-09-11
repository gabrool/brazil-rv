"""Decision-available market shocks and prior-only sector-shrunk exposures."""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from .round5_market import first_available_decision

# Rate changes are percentage points; level archives remain annual decimals.
SERIES = {
    "ptax_brl_per_usd": "fx",
    **{f"br_di_{n}": f"rates_br_{n}" for n in (30, 90, 180, 360, 720, 1080)},
    **{f"us_treasury_{n}": f"rates_us_{n}" for n in ("3m", "2y", "5y", "10y")},
    "brent_spot": "oil",
    "vix_close": "vix",
    "dce_iron": "iron",
    "shfe_rb": "rebar",
    "shfe_hc": "hrc",
    "shfe_sp": "pulp",
    "us_EWZ": "ewz",
}
EXPOSURES = {
    "fx": "fx",
    "rates_br": "rates_br_360",
    "rates_us": "rates_us_10y",
    "oil": "oil",
    "iron": "iron",
    "vix": "vix",
}
FEATURE_NAMES = (
    *(f"shock_{key}_{h}" for key in SERIES.values() for h in (1, 5)),
    *(f"exposure_{key}" for key in EXPOSURES),
    *(f"exposure_{key}_times_shock_{h}" for key in EXPOSURES for h in (1, 5)),
    "adr_return_gap_1",
    "ewz_minus_bova11_1",
    "adr_listed_flag",
    *(
        f"foreign_flow_{h}{suffix}"
        for h in (1, 5)
        for suffix in ("", "_times_log_volume_mean_20", "_times_adr_listed_flag")
    ),
    "foreign_flow_month_reset",
    "foreign_flow_methodology_change",
)
SECTOR_FEATURE_NAMES = (
    "name_minus_sector_return_5",
    "name_minus_sector_return_21",
    "sector_momentum_12_1",
)


def market_shocks(
    levels: pl.DataFrame, returns: pl.DataFrame, *, level_calendars: dict | None = None
) -> pl.DataFrame:
    """Changes over exact source observations, retaining both clocks and gaps.

    Levels use log changes (rates: percentage-point differences). Futures and
    US returns are already exact-contract/wealth returns: five-session changes
    require five linked source returns, not a spliced continuous price index.
    Archive producers establish their calendars and exclude ambiguous gaps.
    Untimed levels never receive an invented fixing time.
    """
    output = []
    for (series,), frame in levels.partition_by("series", as_dict=True).items():
        if series not in SERIES:
            continue
        rows = frame.sort("reference_date").to_dicts()
        calendar = (level_calendars or {}).get(
            series, [row["reference_date"] for row in rows]
        )
        positions = {day: index for index, day in enumerate(calendar)}
        by_date = {row["reference_date"]: row for row in rows}
        for index, row in enumerate(rows):
            if row["available_at"] is None:
                continue
            record = {"reference_date": row["reference_date"]}
            record["factor"] = SERIES[series]
            for horizon in (1, 5):
                value = None
                available = row["available_at"]
                position = positions[row["reference_date"]]
                before = (
                    by_date.get(calendar[position - horizon])
                    if position >= horizon
                    else None
                )
                if before is not None:
                    if before["available_at"] is not None:
                        available = max(available, before["available_at"])
                        if series.startswith(("br_di_", "us_treasury_")):
                            value = 100 * (row["value"] - before["value"])
                        elif row["value"] > 0 and before["value"] > 0:
                            value = np.log(row["value"] / before["value"])
                record[f"shock_{horizon}"] = value
                record[f"available_at_{horizon}"] = available
            output.append(record)
    for (series,), frame in returns.partition_by("series", as_dict=True).items():
        if series not in SERIES:
            continue
        rows = frame.sort("reference_date").to_dicts()
        for index, row in enumerate(rows):
            if row["available_at"] is None:
                continue
            chunk = rows[max(0, index - 4) : index + 1]
            complete = (
                len(chunk) == 5
                and all(
                    a["reference_date"] == b["previous_date"]
                    for a, b in zip(chunk, chunk[1:])
                )
                and all(r["available_at"] is not None for r in chunk)
            )
            output.append(
                {
                    "factor": SERIES[series],
                    "reference_date": row["reference_date"],
                    "available_at_1": row["available_at"],
                    "available_at_5": max(r["available_at"] for r in chunk)
                    if complete
                    else row["available_at"],
                    "shock_1": row["log_return"],
                    "shock_5": sum(r["log_return"] for r in chunk)
                    if complete
                    else None,
                }
            )
    return pl.DataFrame(
        output,
        schema={
            "factor": pl.String,
            "reference_date": pl.Date,
            "available_at_1": pl.Datetime("us", "UTC"),
            "available_at_5": pl.Datetime("us", "UTC"),
            "shock_1": pl.Float64,
            "shock_5": pl.Float64,
        },
    ).sort("factor", "reference_date")


def shock_axes(shocks: pl.DataFrame, sessions: list[date]):
    """Latest observed source-session shock as known state, with true source age.

    A holiday carries the last known shock and increments its age; it is never
    inserted as a new zero-return print or added again to a five-session sum.
    Historical regression coordinates retain reference date and availability.
    """
    result = {}
    positions = {day: index for index, day in enumerate(sessions)}
    for factor in SERIES.values():
        selected = shocks.filter(pl.col("factor") == factor).to_dicts()
        historical = np.full(len(sessions), np.nan)
        public_index = np.full(len(sessions), len(sessions), np.int32)
        current = np.full((len(sessions), 2), np.nan)
        ages = np.full((len(sessions), 2), -1, np.float32)
        for row in selected:
            available = first_available_decision(row["available_at_1"], sessions)
            if row["reference_date"] in positions:
                position = positions[row["reference_date"]]
                if np.isfinite(historical[position]):
                    raise ValueError(
                        "multiple market vintages require a versioned shock ledger"
                    )
                historical[position] = (
                    row["shock_1"] if row["shock_1"] is not None else np.nan
                )
                public_index[position] = available
        for column, horizon in enumerate((1, 5)):
            events = [
                (
                    first_available_decision(row[f"available_at_{horizon}"], sessions),
                    row[f"available_at_{horizon}"],
                    int(np.searchsorted(sessions, row["reference_date"])),
                    row,
                )
                for row in selected
            ]
            events.sort(key=lambda e: (e[0], e[1], e[3]["reference_date"]))
            cursor, latest = 0, None
            for index in range(len(sessions)):
                while cursor < len(events) and events[cursor][0] <= index:
                    event = events[cursor]
                    if (
                        latest is None
                        or event[3]["reference_date"] >= latest[3]["reference_date"]
                    ):
                        latest = event
                    cursor += 1
                if latest is not None:
                    value = latest[3][f"shock_{horizon}"]
                    if value is not None and np.isfinite(value):
                        current[index, column] = value
                        ages[index, column] = max(0, index - latest[2])
        result[factor] = (current, ages, historical, public_index)
    return result


def sector_shrink(beta, variance, valid, sectors, issuers, ages):
    """Leave-one-issuer-out empirical-Bayes prior, with an unshrunk fallback.

    Peer share classes first become one issuer mean. With >=3 other issuers,
    tau²=max(sample variance of issuer slopes - mean estimation variance,0),
    and posterior=(tau²*own + own_variance*peer_mean)/(tau²+own_variance).
    A missing/small sector does not discard an otherwise estimable exposure.
    """
    result = beta.copy()
    result_ages = ages.copy()
    for sector in set(sectors) - {None, ""}:
        members = np.flatnonzero(valid & (sectors == sector) & (issuers != ""))
        peer = {}
        for issuer in set(issuers[members]):
            selected = members[issuers[members] == issuer]
            # Classes are correlated claims on one issuer: do not divide the
            # estimation variance by the number of share classes.
            peer[issuer] = (
                float(beta[selected].mean()),
                float(variance[selected].mean()),
            )
        for name in members:
            others = [
                value for issuer, value in peer.items() if issuer != issuers[name]
            ]
            if len(others) < 3:
                continue
            means, errors = np.asarray(others).T
            tau = max(float(means.var(ddof=1) - errors.mean()), 0.0)
            denominator = tau + variance[name]
            if denominator > 0:
                result[name] = (
                    tau * beta[name] + variance[name] * means.mean()
                ) / denominator
                if variance[name] > 0:
                    result_ages[name] = max(ages[name], float(ages[members].max()))
    return result, result_ages


def exposure_panel(
    excess_returns,
    return_valid,
    historical,
    public_index,
    sectors,
    issuers,
    *,
    window=120,
    minimum=60,
):
    """OLS with intercept on completed returns in [t-120,t); no outcome on t.

    Only reference-date matched observations already public at decision t enter
    the fit. Heterogeneous calendars reduce observed pairs, not add fake zeros.
    Sector and issuer labels are those known at the current decision.
    """
    shape = excess_returns.shape
    values = np.zeros(shape, np.float32)
    valid = np.zeros(shape, bool)
    ages = np.full(shape, -1, np.float32)
    for index in range(minimum, shape[0]):
        start = max(0, index - window)
        y = np.asarray(excess_returns[start:index], np.float64)
        x = np.asarray(historical[start:index], np.float64)[:, None]
        mask = (
            return_valid[start:index]
            & np.isfinite(y)
            & np.isfinite(x)
            & (public_index[start:index, None] <= index)
        )
        count = mask.sum(axis=0)
        xx, yy = np.where(mask, x, 0), np.where(mask, y, 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            mx, my = xx.sum(axis=0) / count, yy.sum(axis=0) / count
            sxx = (xx * xx).sum(axis=0) - count * mx * mx
            sxy = (xx * yy).sum(axis=0) - count * mx * my
            syy = (yy * yy).sum(axis=0) - count * my * my
            beta = sxy / sxx
            variance = np.maximum(syy - beta * sxy, 0) / (count - 2) / sxx
        known = (
            (count >= minimum)
            & (sxx > 1e-14)
            & np.isfinite(beta)
            & np.isfinite(variance)
        )
        last = np.max(np.where(mask, np.arange(start, index)[:, None], -1), axis=0)
        source_ages = np.where(known, index - last, -1)
        shrunk, source_ages = sector_shrink(
            beta, variance, known, sectors[index], issuers[index], source_ages
        )
        values[index] = np.where(known, shrunk, 0)
        valid[index] = known
        ages[index] = source_ages
    return values, valid, ages


def sector_relative_panel(returns, return_valid, sectors, issuers, active):
    """Known-sector issuer-equal peers, with no self/other-share-class inclusion.

    Input columns are decision-aligned log returns 5,21 and momentum252-21.
    At least two other issuers are needed for each relative measurement.
    """
    values, valid = (
        np.zeros_like(returns, dtype=np.float32),
        np.zeros_like(return_valid),
    )
    for day in range(len(returns)):
        for sector in set(sectors[day]) - {None, ""}:
            for column in range(3):
                names = np.flatnonzero(
                    active[day]
                    & return_valid[day, :, column]
                    & (sectors[day] == sector)
                    & (issuers[day] != "")
                )
                grouped = {
                    issuer: float(
                        returns[
                            day, names[issuers[day, names] == issuer], column
                        ].mean()
                    )
                    for issuer in set(issuers[day, names])
                }
                for name in names:
                    peers = [
                        value
                        for issuer, value in grouped.items()
                        if issuer != issuers[day, name]
                    ]
                    if len(peers) >= 2:
                        values[day, name, column] = (
                            returns[day, name, column] - np.mean(peers)
                            if column < 2
                            else np.mean(peers)
                        )
                        valid[day, name, column] = True
    return values, valid
