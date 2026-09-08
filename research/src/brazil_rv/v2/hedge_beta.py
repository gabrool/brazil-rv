from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .artifacts import sha256_file, write_json_atomic
from .bova11 import load_bova11_series
from .contract import DEVELOPMENT_END, FINETUNE_START, PRETRAIN_END
from .store import open_store_for_samples

HEDGE_BETA_SCHEMA = "BRAZIL_RV_V2_HEDGE_BETA_V1"
HEDGE_BETA_LOOKBACK = 60
HEDGE_BETA_MINIMUM_PAIRS = 40
HEDGE_BETA_MAX_AGE = 20


@dataclass(frozen=True)
class HedgeBetaPanel:
    values: NDArray[np.float64]
    valid: NDArray[np.bool_]
    dates: NDArray[np.datetime64]
    isins: tuple[str, ...]
    manifest_sha256: str
    bova11_manifest_sha256: str


def resolve_hedge_beta(
    values: NDArray[np.floating],
    valid: NDArray[np.bool_],
    *,
    history: tuple[NDArray[np.floating], NDArray[np.bool_]] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Use today's valid beta, then the last valid beta <=20 sessions old, then 1.

    History contains the contiguous sessions immediately preceding the ledger
    window. Supplying its last 20 rows preserves this rule at evaluation boundaries.
    Invalid payloads are ignored, even if they contain finite stored zeros.
    """
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=np.bool_)
    if values.ndim != 2 or valid.shape != values.shape:
        raise ValueError("hedge beta and validity must align [date, name]")
    if not np.isfinite(values[valid]).all():
        raise ValueError("valid hedge beta must be finite")
    previous = np.ones(values.shape[1], dtype=np.float64)
    age = np.full(values.shape[1], HEDGE_BETA_MAX_AGE + 1, dtype=np.int64)
    if history is not None:
        past_values, past_valid = (np.asarray(item) for item in history)
        if (
            past_values.ndim != 2
            or past_values.shape[1] != values.shape[1]
            or past_valid.shape != past_values.shape
        ):
            raise ValueError("hedge-beta history must align securities")
        if not np.isfinite(past_values[past_valid]).all():
            raise ValueError("valid historical hedge beta must be finite")
        for row, mask in zip(
            past_values[-HEDGE_BETA_MAX_AGE:],
            past_valid[-HEDGE_BETA_MAX_AGE:],
            strict=True,
        ):
            age += 1
            previous[mask], age[mask] = row[mask], 0
    resolved = np.empty_like(values)
    for day, (row, mask) in enumerate(zip(values, valid, strict=True)):
        age += 1
        previous[mask], age[mask] = row[mask], 0
        resolved[day] = np.where(age <= HEDGE_BETA_MAX_AGE, previous, 1.0)
    return resolved, ~valid


def rolling_hedge_beta(
    wealth_close: NDArray[np.floating],
    wealth_valid: NDArray[np.bool_],
    hedge_close: NDArray[np.floating],
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """OLS with an intercept, using only return endpoints t-60 through t-1.

    Each equity return requires valid adjacent shareholder-wealth observations.
    Gaps are never bridged. The 60-session window is a calendar window, not the
    latest 60 observed returns. Forty valid paired returns are required.
    """
    wealth = np.asarray(wealth_close, dtype=np.float64)
    seen = np.asarray(wealth_valid, dtype=np.bool_)
    hedge = np.asarray(hedge_close, dtype=np.float64)
    if wealth.ndim != 2 or seen.shape != wealth.shape or hedge.shape != (len(wealth),):
        raise ValueError("hedge-beta prices and masks must align by date/security")
    y = np.full_like(wealth, np.nan)
    pairs = (
        seen[1:]
        & seen[:-1]
        & np.isfinite(wealth[1:])
        & np.isfinite(wealth[:-1])
        & (wealth[1:] >= 0.0)
        & (wealth[:-1] > 0.0)
    )
    np.divide(wealth[1:], wealth[:-1], out=y[1:], where=pairs)
    y[1:] -= 1.0
    x = np.full(len(wealth), np.nan, dtype=np.float64)
    hedge_pairs = (
        np.isfinite(hedge[1:])
        & np.isfinite(hedge[:-1])
        & (hedge[1:] > 0.0)
        & (hedge[:-1] > 0.0)
    )
    np.divide(hedge[1:], hedge[:-1], out=x[1:], where=hedge_pairs)
    x[1:] -= 1.0
    values = np.full_like(wealth, np.nan)
    valid = np.zeros(wealth.shape, dtype=np.bool_)
    for day in range(HEDGE_BETA_MINIMUM_PAIRS + 1, len(wealth)):
        start = max(1, day - HEDGE_BETA_LOOKBACK)
        local_x = x[start:day, None]
        local_y = y[start:day]
        usable = np.isfinite(local_x) & np.isfinite(local_y)
        count = usable.sum(axis=0)
        clean_x = np.where(usable, local_x, 0.0)
        clean_y = np.where(usable, local_y, 0.0)
        denominator_count = np.maximum(count, 1)
        sum_x, sum_y = clean_x.sum(axis=0), clean_y.sum(axis=0)
        variance = (clean_x**2).sum(axis=0) - sum_x**2 / denominator_count
        covariance = (clean_x * clean_y).sum(axis=0) - sum_x * sum_y / denominator_count
        supported = (count >= HEDGE_BETA_MINIMUM_PAIRS) & (variance > 0.0)
        values[day, supported] = np.clip(
            0.67 * covariance[supported] / variance[supported] + 0.33, -1.0, 3.0
        )
        valid[day] = supported
    return values, valid


def build_hedge_beta_sidecar(
    *,
    store_root: Path,
    expected_store_manifest_sha256: str,
    bova11_root: Path,
    expected_bova11_manifest_sha256: str,
    output_root: Path,
    end_date: date = DEVELOPMENT_END,
) -> dict[str, object]:
    """Materialize development-only economic betas without rewriting the store."""
    if end_date > DEVELOPMENT_END:
        raise PermissionError("hedge-beta sidecar refuses 2025/2026 dates")
    if output_root.exists():
        raise FileExistsError(output_root)
    if any(
        output_root.resolve().is_relative_to(root.resolve())
        for root in (store_root, bova11_root)
    ):
        raise ValueError("hedge-beta output must be outside immutable source roots")
    manifest_path = store_root / "manifest.json"
    if sha256_file(manifest_path) != expected_store_manifest_sha256:
        raise ValueError("hedge-beta source store manifest SHA-256 mismatch")
    axis = np.load(store_root / "date_index.npy", allow_pickle=False)
    rows = np.flatnonzero(axis <= np.datetime64(end_date))
    if not rows.size:
        raise ValueError("hedge-beta sidecar has no development dates")
    # The ten-session pretrain embargo is causal history, never a sample.
    sample_rows = rows[
        (axis[rows] <= np.datetime64(PRETRAIN_END))
        | (axis[rows] >= np.datetime64(FINETUNE_START))
    ]
    store, access = open_store_for_samples(
        store_root,
        sample_rows,
        purpose="evaluation",
        verify_hashes=True,
        history_lookbacks=60,
        history_end_offsets=0,
    )
    dates = store.dates[rows]
    bova = load_bova11_series(
        bova11_root,
        expected_manifest_sha256=expected_bova11_manifest_sha256,
        canonical_dates=dates.astype(object).tolist(),
    )
    try:
        beta, valid = rolling_hedge_beta(
            store.read("shareholder_wealth_close", rows),
            store.read("shareholder_wealth_valid", rows),
            bova.close_by_session,
        )
        active = np.asarray(store.read("active", rows), dtype=np.bool_)
    finally:
        store.close()
    annual = []
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    for year in np.unique(years):
        used = (years == year)[:, None] & valid & active
        sample = beta[used]
        annual.append(
            {
                "year": int(year),
                "active_valid_name_days": int(sample.size),
                "active_name_days": int(active[years == year].sum()),
                "quantiles_01_05_50_95_99": (
                    np.quantile(sample, [0.01, 0.05, 0.50, 0.95, 0.99]).tolist()
                    if sample.size
                    else None
                ),
                "fraction_between_0_5_and_1_5": (
                    float(((sample >= 0.5) & (sample <= 1.5)).mean())
                    if sample.size
                    else None
                ),
            }
        )
    output_root.mkdir(parents=True, exist_ok=False)
    arrays = {}
    for name, values in (
        ("hedge_beta", beta),
        ("hedge_beta_valid", valid),
        ("date_index", dates),
        ("isin_index", np.asarray(store.isins)),
    ):
        path = output_root / f"{name}.npy"
        np.save(path, values, allow_pickle=False)
        arrays[name] = {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    manifest = {
        "schema": HEDGE_BETA_SCHEMA,
        "status": "complete",
        "implementation_sha256": sha256_file(Path(__file__)),
        "store": {
            "root": str(store_root.resolve()),
            "manifest_sha256": expected_store_manifest_sha256,
        },
        "bova11": {
            "root": str(bova11_root.resolve()),
            "manifest_sha256": bova.manifest_sha256,
            "data_sha256": bova.data_sha256,
        },
        "contract": {
            "equity_return": "adjacent shareholder-wealth simple return",
            "benchmark_return": "adjacent BOVA11 simple close return",
            "last_return_endpoint": "t-1",
            "calendar_window_sessions": HEDGE_BETA_LOOKBACK,
            "minimum_valid_pairs": HEDGE_BETA_MINIMUM_PAIRS,
            "estimator": "OLS slope with intercept",
            "blume": [0.67, 0.33],
            "clip": [-1.0, 3.0],
        },
        "first_date": str(dates[0]),
        "last_date": str(dates[-1]),
        "arrays": arrays,
        "distribution_by_year": annual,
        "access": access.payload(),
    }
    write_json_atomic(output_root / "manifest.json", manifest)
    return manifest


def load_hedge_beta_sidecar(
    root: Path,
    *,
    expected_manifest_sha256: str,
    expected_store_manifest_sha256: str,
    expected_bova11_manifest_sha256: str,
    canonical_dates: Sequence[date],
    isins: Sequence[str],
) -> HedgeBetaPanel:
    """Verify economic beta identity and align only admitted development dates."""
    dates = np.asarray(canonical_dates, dtype="datetime64[D]")
    if dates.size and np.any(dates > np.datetime64(DEVELOPMENT_END)):
        raise PermissionError("hedge-beta sidecar refuses 2025/2026 dates")
    path = root / "manifest.json"
    if sha256_file(path) != expected_manifest_sha256:
        raise ValueError("hedge-beta manifest SHA-256 mismatch")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != HEDGE_BETA_SCHEMA
        or manifest.get("status") != "complete"
    ):
        raise ValueError("invalid hedge-beta sidecar manifest")
    if (
        manifest["store"]["manifest_sha256"] != expected_store_manifest_sha256
        or manifest["bova11"]["manifest_sha256"] != expected_bova11_manifest_sha256
    ):
        raise ValueError("hedge-beta store/BOVA11 source identity differs")
    arrays = {}
    for name, record in manifest["arrays"].items():
        source = root / record["path"]
        if (
            source.stat().st_size != record["bytes"]
            or sha256_file(source) != record["sha256"]
        ):
            raise ValueError(f"hedge-beta array identity differs: {name}")
        arrays[name] = np.load(source, allow_pickle=False)
    source_dates = arrays["date_index"]
    if np.any(source_dates > np.datetime64(DEVELOPMENT_END)):
        raise PermissionError("hedge-beta sidecar contains 2025/2026 dates")
    if tuple(arrays["isin_index"].tolist()) != tuple(isins):
        raise ValueError("hedge-beta security identity differs")
    positions = np.searchsorted(source_dates, dates)
    if np.any(positions >= len(source_dates)) or not np.array_equal(
        source_dates[positions], dates
    ):
        raise ValueError("hedge-beta sidecar does not cover requested dates")
    return HedgeBetaPanel(
        values=arrays["hedge_beta"][positions],
        valid=arrays["hedge_beta_valid"][positions],
        dates=dates,
        isins=tuple(isins),
        manifest_sha256=expected_manifest_sha256,
        bova11_manifest_sha256=expected_bova11_manifest_sha256,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build causal economic betas against BOVA11"
    )
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--store-manifest-sha256", required=True)
    parser.add_argument("--bova11", type=Path, required=True)
    parser.add_argument("--bova11-manifest-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    build_hedge_beta_sidecar(
        store_root=arguments.store,
        expected_store_manifest_sha256=arguments.store_manifest_sha256,
        bova11_root=arguments.bova11,
        expected_bova11_manifest_sha256=arguments.bova11_manifest_sha256,
        output_root=arguments.out,
    )


if __name__ == "__main__":
    main()
