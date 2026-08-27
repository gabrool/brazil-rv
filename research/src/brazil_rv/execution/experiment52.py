from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import torch

from ..modeling.analyze import align_observations
from ..modeling.data import feature_store_identity
from ..modeling.provenance import repository_commit
from ..modeling.three_fold_sidecar_screen import crossfit_patience_observations
from ..preprocessing.contract import (
    EQUITY_SESSION_MINUTES,
    PRICE_VOL_REFERENCE,
)
from ..preprocessing.economics_targets import economics_input_identity
from .config import ExecutionConfig
from .inputs import (
    causal_liquidity,
    causal_rank_scores,
    causal_roll_spreads,
    expand_refreshes,
    iter_discovery_equity_grids,
    lagged_quarter_spreads,
    load_daily_cdi_rates,
    load_discovery_prediction_archive,
    write_discovery_prediction_manifest,
)
from .policy import BandPolicy
from .report import DailyExecutionResult, write_execution_report
from .simulator import MarketReplay, simulate

SCHEMA = "EXPERIMENT52_C0_BASELINE_V1"
FOLDS = ("fold_c", "fold_a", "fold_b")
SEEDS = (11, 29, 47)
BANDS = (0.0, 0.5, 1.0, 2.0)
BLENDS = {
    "equal": (1 / 3, 1 / 3, 1 / 3),
    "h30_only": (1.0, 0.0, 0.0),
    "front_loaded": (0.5, 0.3, 0.2),
}
CDI_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_npz(path: Path, values: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as output:
        np.savez(output, **values)
    os.replace(temporary, path)


def _band_name(value: float) -> str:
    return str(value).replace(".", "p")


def execution_cells() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "cell_id": f"band_{_band_name(band)}__blend_{blend_name}",
            "band": band,
            "blend_name": blend_name,
            "horizon_blend": list(blend),
            "config": ExecutionConfig(band=band, horizon_blend=blend).to_dict(),
            "config_sha256": ExecutionConfig(band=band, horizon_blend=blend).sha256,
        }
        for band in BANDS
        for blend_name, blend in BLENDS.items()
    )


def stored_daily_volatility(vol_regime: np.ndarray) -> np.ndarray:
    """Invert the store's causal log-volatility field to daily return units."""
    values = np.asarray(vol_regime, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Stored volatility regime contains non-finite values")
    return PRICE_VOL_REFERENCE * np.sqrt(EQUITY_SESSION_MINUTES) * np.exp(values)


def daily_readout(
    daily: Sequence[DailyExecutionResult], nav_brl: float
) -> dict[str, float | int]:
    if not daily:
        raise ValueError("An execution readout requires daily results")
    net = np.asarray([row.net_pnl_brl for row in daily], dtype=np.float64)
    gross = float(sum(row.gross_pnl_brl for row in daily))
    costs = float(sum(row.spread_cost_brl + row.fees_brl for row in daily))
    standard_deviation = float(np.std(net, ddof=1)) if len(net) > 1 else 0.0
    if standard_deviation <= 0.0:
        raise ValueError("Annualized Sharpe requires nonzero daily PnL variation")
    return {
        "date_count": len(daily),
        "mean_daily_net_pnl_brl": float(np.mean(net)),
        "std_daily_net_pnl_brl": standard_deviation,
        "annualized_net_sharpe": float(
            np.sqrt(252.0) * np.mean(net) / standard_deviation
        ),
        "net_to_gross_ratio": float(sum(net) / gross) if gross != 0.0 else 0.0,
        "daily_cost_drag_bps_of_nav": costs / (len(daily) * nav_brl) * 10_000.0,
    }


def rotation_designation(
    rows: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    by_key = {(str(row["cell_id"]), str(row["fold"])): row for row in rows}
    cells = sorted({cell for cell, _ in by_key})
    if set(fold for _, fold in by_key) != set(FOLDS) or len(by_key) != len(cells) * 3:
        raise ValueError(
            "Rotation input must contain every cell on exactly three folds"
        )

    table: list[dict[str, object]] = []
    winners: list[dict[str, object]] = []
    for heldout in FOLDS:
        training_folds = tuple(fold for fold in FOLDS if fold != heldout)
        ranked = sorted(
            (
                {
                    "heldout_fold": heldout,
                    "cell_id": cell,
                    "other_fold_mean_annualized_net_sharpe": float(
                        np.mean(
                            [
                                float(by_key[(cell, fold)]["annualized_net_sharpe"])
                                for fold in training_folds
                            ]
                        )
                    ),
                    "heldout_annualized_net_sharpe": float(
                        by_key[(cell, heldout)]["annualized_net_sharpe"]
                    ),
                    "heldout_net_pnl_brl": float(
                        by_key[(cell, heldout)]["net_pnl_brl"]
                    ),
                }
                for cell in cells
            ),
            key=lambda row: (
                -float(row["other_fold_mean_annualized_net_sharpe"]),
                str(row["cell_id"]),
            ),
        )
        for rank, row in enumerate(ranked, start=1):
            table.append({**row, "rotation_rank": rank})
        winners.append(ranked[0])

    counts = Counter(str(row["cell_id"]) for row in winners)
    maximum = max(counts.values())
    finalists = [cell for cell, count in counts.items() if count == maximum]
    heldout_means = {
        cell: float(
            np.mean(
                [float(by_key[(cell, fold)]["annualized_net_sharpe"]) for fold in FOLDS]
            )
        )
        for cell in finalists
    }
    best_mean = max(heldout_means.values())
    selected = [
        cell
        for cell in finalists
        if math.isclose(heldout_means[cell], best_mean, rel_tol=0.0, abs_tol=1e-15)
    ]
    if len(selected) != 1:
        raise ValueError("C0 rule remains exactly tied after its frozen tie-break")
    return table, {
        "schema": "EXPERIMENT52_C0_DESIGNATION_V1",
        "rule": (
            "cell winning most two-fold rotations; tie uses higher mean held-out "
            "annualized net Sharpe"
        ),
        "c0_cell_id": selected[0],
        "rotation_win_count": maximum,
        "rotation_winners": winners,
        "official_validation_accessed": False,
        "test_accessed": False,
    }


def _fold_run(root: Path, fold: str, seed: int) -> Path:
    return root / "stage_c" / "candidates" / "prune_r2" / fold / f"seed_{seed}"


def _fold_analysis(root: Path, fold: str) -> Path:
    return (
        root
        / "stage_c"
        / "analysis"
        / "prune_r2"
        / "patience3_raw"
        / fold
        / "standalone"
        / "analysis.json"
    )


def _selection_source(
    experiment41_root: Path, fold: str
) -> tuple[dict[str, object], dict[str, object]]:
    analysis_path = _fold_analysis(experiment41_root, fold)
    analysis = _read_json(analysis_path)
    replays = analysis.get("comparison_metadata", {}).get("candidate_patience_replays")
    if not isinstance(replays, dict) or set(replays) != {
        f"seed_{seed}" for seed in SEEDS
    }:
        raise ValueError(f"Experiment-41 replays differ for {fold}")
    sources: list[dict[str, object]] = []
    selection_window: dict[str, object] | None = None
    store_identity: dict[str, object] | None = None
    for seed in SEEDS:
        run = _fold_run(experiment41_root, fold, seed)
        manifest_path = run / "run_manifest.json"
        reference_path = run / "validation_reference.npz"
        manifest = _read_json(manifest_path)
        if (
            manifest.get("status") != "completed"
            or manifest.get("seed") != seed
            or manifest.get("split", {}).get("selection") != fold
            or manifest.get("split", {}).get("test_accessed") is not False
            or manifest.get("official_validation_accessed") is True
            or manifest.get("test_accessed") is True
        ):
            raise ValueError(f"Experiment-41 source manifest differs: {fold}/{seed}")
        current_window = manifest["split"]["selection_window"]
        current_store = manifest["feature_store_identity"]
        if selection_window is None:
            selection_window = current_window
            store_identity = current_store
        elif selection_window != current_window or store_identity != current_store:
            raise ValueError(f"Experiment-41 members do not align: {fold}")
        selected_epochs = sorted(
            {int(row["selected_epoch"]) for row in replays[f"seed_{seed}"]}
        )
        prediction_artifacts = [
            _artifact(run / "validation_predictions" / f"epoch_{epoch:02d}.npz")
            for epoch in selected_epochs
        ]
        sources.append(
            {
                "seed": seed,
                "run_manifest": _artifact(manifest_path),
                "validation_reference": _artifact(reference_path),
                "selected_epochs": selected_epochs,
                "selected_predictions": prediction_artifacts,
                "frozen_replays": replays[f"seed_{seed}"],
            }
        )
    if selection_window is None or store_identity is None:
        raise ValueError(f"No Experiment-41 sources found for {fold}")
    return (
        {
            "fold": fold,
            "analysis": _artifact(analysis_path),
            "members": sources,
        },
        {
            "status": "completed",
            "schema": "EXPERIMENT52_DISCOVERY_ENSEMBLE_SOURCE_V1",
            "feature_store_identity": store_identity,
            "split": {
                "training": fold,
                "selection": fold,
                "selection_window": selection_window,
                "test_accessed": False,
            },
            "constituent_sources": sources,
            "analysis": _artifact(analysis_path),
            "official_validation_accessed": False,
            "test_accessed": False,
        },
    )


def _materialize_fold_predictions(
    *, store: Path, experiment41_root: Path, fold: str, output_dir: Path
) -> dict[str, object]:
    source_record, aggregate_manifest = _selection_source(experiment41_root, fold)
    members = {}
    replays = aggregate_manifest["constituent_sources"]
    for source in replays:
        seed = int(source["seed"])
        observations, _ = crossfit_patience_observations(
            _fold_run(experiment41_root, fold, seed),
            source["frozen_replays"],
        )
        members[f"seed_{seed}"] = observations
    aligned = align_observations(members)
    reference = next(iter(aligned.values()))
    dates = np.unique(reference.date_idx)
    decisions = np.unique(reference.decision_idx)
    date_position = np.searchsorted(dates, reference.date_idx)
    decision_position = np.searchsorted(decisions, reference.decision_idx)
    if dates.size * decisions.size != reference.sample_id.size:
        raise ValueError(f"Experiment-41 source is not a complete grid: {fold}")
    membership = np.load(
        store / "equity_membership.npy", mmap_mode="r", allow_pickle=False
    )
    ready = np.load(store / "equity_data_ready.npy", mmap_mode="r", allow_pickle=False)
    activity = np.asarray(membership[dates] & ready[dates], dtype=bool)
    valid = np.broadcast_to(
        activity[:, None, :, None],
        (dates.size, decisions.size, reference.predictions.shape[1], 3),
    )
    ranked_members = []
    for observations in aligned.values():
        grid = np.empty(valid.shape, dtype=observations.predictions.dtype)
        grid[date_position, decision_position] = observations.predictions[..., :3]
        ranked_members.append(causal_rank_scores(grid, valid))
    ensemble = np.mean(ranked_members, axis=0, dtype=np.float64)
    flat = ensemble[date_position, decision_position]

    fold_dir = output_dir / fold
    fold_dir.mkdir(parents=True)
    prediction_path = fold_dir / "ensemble_predictions.npz"
    reference_path = fold_dir / "prediction_reference.npz"
    source_manifest_path = fold_dir / "source_manifest.json"
    wrapper_path = fold_dir / "execution_manifest.json"
    _atomic_npz(prediction_path, {"scores": flat})
    _atomic_npz(
        reference_path,
        {
            "sample_id": reference.sample_id,
            "date_idx": reference.date_idx,
            "decision_idx": reference.decision_idx,
        },
    )
    _atomic_json(source_manifest_path, aggregate_manifest)
    refresh_minutes = tuple(15 + 5 * index for index in range(decisions.size))
    wrapper = write_discovery_prediction_manifest(
        wrapper_path,
        store=store,
        prediction_path=prediction_path,
        reference_path=reference_path,
        source_manifest_path=source_manifest_path,
        split=fold,
        refresh_minutes=refresh_minutes,
        prediction_key="scores",
    )
    load_discovery_prediction_archive(
        prediction_path, reference_path, wrapper_path, store
    )
    return {
        **source_record,
        "ensemble_prediction": _artifact(prediction_path),
        "prediction_reference": _artifact(reference_path),
        "source_manifest": _artifact(source_manifest_path),
        "execution_manifest": _artifact(wrapper_path),
        "execution_wrapper": wrapper,
    }


def _fetch_cdi(output_dir: Path, start: date, end: date) -> dict[str, object]:
    output_dir.mkdir(parents=True)
    parameters = urllib.parse.urlencode(
        {
            "formato": "json",
            "dataInicial": start.strftime("%d/%m/%Y"),
            "dataFinal": end.strftime("%d/%m/%Y"),
        }
    )
    url = f"{CDI_URL}?{parameters}"
    request = urllib.request.Request(url, headers={"User-Agent": "Brazil-RV/1"})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        raw = response.read()
    rows = json.loads(raw)
    if not isinstance(rows, list) or not rows:
        raise ValueError("BCB SGS series 12 returned no observations")
    parsed = []
    for row in rows:
        parsed.append(
            {
                "trade_date": datetime.strptime(row["data"], "%d/%m/%Y").date(),
                "daily_cdi_rate": float(str(row["valor"]).replace(",", ".")) / 100.0,
            }
        )
    table = pl.DataFrame(parsed).sort("trade_date")
    if table["trade_date"].n_unique() != table.height:
        raise ValueError("BCB SGS series 12 contains duplicate dates")
    raw_path = output_dir / "bcb_sgs12_raw.json"
    parquet_path = output_dir / "daily_cdi.parquet"
    manifest_path = output_dir / "cdi_manifest.json"
    raw_path.write_bytes(raw)
    table.write_parquet(parquet_path)
    manifest = {
        "schema": "EXPERIMENT52_CDI_SGS12_V1",
        "source": "Banco Central do Brasil SGS series 12",
        "source_url": url,
        "retrieved_at_utc": _now(),
        "source_unit": "percent per business day",
        "stored_unit": "fractional return per business day",
        "conversion": "daily_cdi_rate = source valor / 100",
        "first_date": table.item(0, "trade_date").isoformat(),
        "last_date": table.item(-1, "trade_date").isoformat(),
        "row_count": table.height,
        "raw": _artifact(raw_path),
        "parquet": _artifact(parquet_path),
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    _atomic_json(manifest_path, manifest)
    return {**manifest, "manifest": _artifact(manifest_path)}


def freeze_program(
    *,
    store: Path,
    experiment41_root: Path,
    economics_dir: Path,
    output_dir: Path,
) -> Path:
    store, experiment41_root, economics_dir, output_dir = (
        path.resolve() for path in (store, experiment41_root, economics_dir, output_dir)
    )
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    try:
        store_identity = feature_store_identity(store)
        economics = economics_input_identity(economics_dir, store_identity)
        predictions_dir = output_dir / "wrapped_predictions"
        sources = {
            fold: _materialize_fold_predictions(
                store=store,
                experiment41_root=experiment41_root,
                fold=fold,
                output_dir=predictions_dir,
            )
            for fold in FOLDS
        }
        fold_dates = []
        for fold in FOLDS:
            with np.load(
                Path(sources[fold]["prediction_reference"]["path"]),
                allow_pickle=False,
            ) as values:
                fold_dates.extend(np.unique(values["date_idx"]).tolist())
        date_index = pl.read_parquet(store / "date_index.parquet").sort("date_idx")
        selected_dates = date_index.filter(
            pl.col("date_idx").is_in(sorted(set(fold_dates)))
        )["trade_date"]
        cdi = _fetch_cdi(output_dir / "cdi", min(selected_dates), max(selected_dates))
        load_daily_cdi_rates(
            Path(cdi["parquet"]["path"]),
            tuple(selected_dates),
            str(cdi["parquet"]["sha256"]),
        )
        design = {
            "schema": SCHEMA,
            "status": "frozen",
            "created_at": _now(),
            "repository_commit": repository_commit(),
            "purpose": "first end-to-end discovery-only execution measurement",
            "cpu_only": True,
            "store": {
                "path": str(store),
                "identity": store_identity,
                "manifest": _artifact(store / "manifest.json"),
            },
            "experiment41_root": str(experiment41_root),
            "fold_sources": sources,
            "economics_inputs": economics,
            "roll_schedule": _artifact(economics_dir / "roll_schedule.parquet"),
            "cdi": cdi,
            "cells": list(execution_cells()),
            "frictionless": "same cell with fee_bps=0 and full spreads exactly zero",
            "daily_std_ddof": 1,
            "rotation_rule": (
                "rank by mean annualized net Sharpe on other two folds; most "
                "rotation wins; tie by higher mean held-out Sharpe"
            ),
            "official_validation_accessed": False,
            "test_accessed": False,
        }
        design_path = output_dir / "frozen_design.json"
        _atomic_json(design_path, design)
        _atomic_json(
            output_dir / "program_manifest.json",
            {
                "schema": SCHEMA,
                "status": "frozen",
                "repository_commit": design["repository_commit"],
                "frozen_design": _artifact(design_path),
                "official_validation_accessed": False,
                "test_accessed": False,
            },
        )
    except BaseException:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise
    return output_dir / "frozen_design.json"


def _save_array(directory: Path, name: str, values: np.ndarray) -> dict[str, object]:
    path = directory / name
    with path.open("wb") as output:
        np.save(output, values, allow_pickle=False)
    return {"path": name, "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _build_market_cache(
    *, store: Path, economics_dir: Path, design: Mapping[str, object], output_dir: Path
) -> dict[str, object]:
    final = output_dir / "market_inputs"
    if final.exists():
        manifest = _read_json(final / "manifest.json")
        for name, record in manifest["artifacts"].items():
            if _sha256(final / name) != record["sha256"]:
                raise ValueError(f"Market cache hash mismatch: {name}")
        return manifest
    temporary = output_dir / ".market_inputs.tmp"
    temporary.mkdir()
    archives = {}
    for fold in FOLDS:
        source = design["fold_sources"][fold]
        archives[fold] = load_discovery_prediction_archive(
            Path(source["ensemble_prediction"]["path"]),
            Path(source["prediction_reference"]["path"]),
            Path(source["execution_manifest"]["path"]),
            store,
        )
    requested_idx = np.unique(
        np.concatenate([archive.date_idx for archive in archives.values()])
    )
    date_index = pl.read_parquet(store / "date_index.parquet").sort("date_idx")
    selected_table = date_index.filter(pl.col("date_idx").is_in(requested_idx))
    if selected_table.height != requested_idx.size or not np.array_equal(
        selected_table["date_idx"].to_numpy(), requested_idx
    ):
        raise ValueError("Discovery fold dates differ from the canonical store")
    trade_dates = tuple(selected_table["trade_date"])
    equity_table = pl.read_parquet(store / "equity_index.parquet").sort("equity_slot")
    security_ids = tuple(equity_table["security_id"])
    shape = (requested_idx.size, EQUITY_SESSION_MINUTES, len(security_ids))
    open_price = np.zeros(shape, dtype=np.float64)
    observed = np.zeros(shape, dtype=bool)
    minute_notional = np.full(shape, np.nan, dtype=np.float64)
    active = np.zeros((requested_idx.size, len(security_ids)), dtype=bool)
    adv = np.full(active.shape, np.nan, dtype=np.float64)
    fallback = np.full(active.shape, np.nan, dtype=np.float64)
    seen = np.zeros(len(security_ids), dtype=bool)
    for grid in iter_discovery_equity_grids(store):
        positions = {value: index for index, value in enumerate(grid.trade_dates)}
        source_positions = np.asarray([positions[value] for value in trade_dates])
        slot = grid.equity_slot
        liquidity_adv, profile = causal_liquidity(
            grid.close[..., None],
            grid.real_volume[..., None],
            grid.observed[..., None],
            lookback=20,
        )
        roll = causal_roll_spreads(
            grid.close[..., None], grid.observed[..., None], lookback=60
        )
        open_price[:, :, slot] = grid.open_price[source_positions]
        observed[:, :, slot] = grid.observed[source_positions]
        minute_notional[:, :, slot] = profile[source_positions, :, 0]
        active[:, slot] = grid.active[source_positions]
        adv[:, slot] = liquidity_adv[source_positions, 0]
        fallback[:, slot] = roll[source_positions, 0]
        seen[slot] = True
    if not seen.all():
        raise ValueError("Raw market bridge did not emit every permanent security")
    scheduled = lagged_quarter_spreads(
        economics_dir / "roll_schedule.parquet",
        trade_dates,
        security_ids,
        str(design["roll_schedule"]["sha256"]),
    )
    full_spread = np.where(np.isfinite(scheduled), scheduled, fallback)
    slow = np.load(store / "equity_slow.npy", mmap_mode="r", allow_pickle=False)
    sigma = stored_daily_volatility(np.asarray(slow[requested_idx, :, 0]))
    artifacts = {
        name: _save_array(temporary, name, values)
        for name, values in {
            "date_idx.npy": requested_idx,
            "open_price.npy": open_price,
            "open_observed.npy": observed,
            "active.npy": active,
            "adv20_brl.npy": adv,
            "minute_notional20_brl.npy": minute_notional,
            "full_spread.npy": full_spread,
            "sigma_daily.npy": sigma,
        }.items()
    }
    selected_table.write_parquet(temporary / "dates.parquet")
    artifacts["dates.parquet"] = _artifact(temporary / "dates.parquet")
    manifest = {
        "schema": "EXPERIMENT52_MARKET_INPUTS_V1",
        "created_at": _now(),
        "store_manifest": _artifact(store / "manifest.json"),
        "roll_schedule": design["roll_schedule"],
        "liquidity": "causal_liquidity lookback=20, current session excluded",
        "roll_fallback": "causal_roll_spreads lookback=60, prior sessions only",
        "scheduled_spread_count": int(np.isfinite(scheduled).sum()),
        "fallback_spread_count": int(
            (~np.isfinite(scheduled) & np.isfinite(fallback)).sum()
        ),
        "unresolved_spread_count": int((~np.isfinite(full_spread)).sum()),
        "sigma": (
            "PRICE_VOL_REFERENCE*sqrt(405)*exp(equity_slow vol_regime); "
            "stored causal clipped field, dimensionless daily return units"
        ),
        "artifacts": artifacts,
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    _atomic_json(temporary / "manifest.json", manifest)
    os.replace(temporary, final)
    return _read_json(final / "manifest.json")


def _load_cache_array(root: Path, name: str) -> np.ndarray:
    return np.load(root / "market_inputs" / name, mmap_mode="r", allow_pickle=False)


def _daily_results(dates: Sequence[date], result: object) -> list[DailyExecutionResult]:
    names = (
        "net_pnl_brl",
        "gross_pnl_brl",
        "spread_cost_brl",
        "fees_brl",
        "cdi_earned_brl",
        "turnover_brl",
        "max_intraday_gross_brl",
        "forced_fill_count",
    )
    values = {name: getattr(result, name).detach().cpu().numpy() for name in names}
    return [
        DailyExecutionResult(
            trade_date=value,
            net_pnl_brl=float(values["net_pnl_brl"][index]),
            gross_pnl_brl=float(values["gross_pnl_brl"][index]),
            spread_cost_brl=float(values["spread_cost_brl"][index]),
            fees_brl=float(values["fees_brl"][index]),
            cdi_earned_brl=float(values["cdi_earned_brl"][index]),
            turnover_brl=float(values["turnover_brl"][index]),
            max_intraday_gross_brl=float(values["max_intraday_gross_brl"][index]),
            forced_fill_count=int(values["forced_fill_count"][index]),
        )
        for index, value in enumerate(dates)
    ]


def _load_existing_report(
    path: Path,
    *,
    config: ExecutionConfig,
    input_sha256: Mapping[str, str],
) -> tuple[list[DailyExecutionResult], dict[str, str]]:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    digest, name = sidecar.read_text(encoding="utf-8").strip().split("  ")
    payload = _read_json(path)
    expected_inputs = dict(sorted(input_sha256.items()))
    if (
        name != path.name
        or digest != _sha256(path)
        or payload.get("schema") != "B3_EXECUTION_BACKTEST_REPORT_V1"
        or payload.get("config_sha256") != config.sha256
        or payload.get("input_sha256") != expected_inputs
    ):
        raise ValueError(f"Existing execution report differs: {path}")
    daily = [
        DailyExecutionResult(
            trade_date=date.fromisoformat(str(row["trade_date"])),
            net_pnl_brl=float(row["net_pnl_brl"]),
            gross_pnl_brl=float(row["gross_pnl_brl"]),
            spread_cost_brl=float(row["spread_cost_brl"]),
            fees_brl=float(row["fees_brl"]),
            cdi_earned_brl=float(row["cdi_earned_brl"]),
            turnover_brl=float(row["turnover_brl"]),
            max_intraday_gross_brl=float(row["max_intraday_gross_brl"]),
            forced_fill_count=int(row["forced_fill_count"]),
        )
        for row in payload.get("daily", [])
    ]
    if not daily:
        raise ValueError(f"Existing execution report has no daily rows: {path}")
    return daily, {
        "path": str(path.resolve()),
        "sha256": digest,
        "sha256_path": str(sidecar.resolve()),
    }


def _run_one(
    *,
    market: MarketReplay,
    ranks: torch.Tensor,
    rank_valid: torch.Tensor,
    refresh_mask: torch.Tensor,
    sigma: torch.Tensor,
    config: ExecutionConfig,
    dates: Sequence[date],
) -> list[DailyExecutionResult]:
    if any(tensor.device.type != "cpu" for tensor in (ranks, sigma, market.open_price)):
        raise ValueError("Experiment 52 is CPU only")
    result = simulate(
        market,
        ranks,
        rank_valid,
        refresh_mask,
        sigma,
        BandPolicy(config),
        config,
    )
    return _daily_results(dates, result)


def run_program(*, store: Path, economics_dir: Path, output_dir: Path) -> Path:
    store, economics_dir, output_dir = (
        path.resolve() for path in (store, economics_dir, output_dir)
    )
    manifest_path = output_dir / "program_manifest.json"
    design_path = output_dir / "frozen_design.json"
    manifest = _read_json(manifest_path)
    design = _read_json(design_path)
    operational_commit = os.environ.get(
        "EXPERIMENT52_OPERATIONAL_COMMIT", repository_commit()
    )
    if len(operational_commit) != 40 or any(
        value not in "0123456789abcdef" for value in operational_commit
    ):
        raise ValueError("Experiment 52 operational commit must be a full Git hash")
    if (
        manifest.get("status") != "frozen"
        or manifest.get("frozen_design", {}).get("sha256") != _sha256(design_path)
        or design.get("repository_commit") != repository_commit()
        or design.get("cells") != list(execution_cells())
    ):
        raise ValueError("Experiment 52 frozen state differs")
    _build_market_cache(
        store=store, economics_dir=economics_dir, design=design, output_dir=output_dir
    )
    market_manifest_path = output_dir / "market_inputs" / "manifest.json"
    cdi_path = Path(design["cdi"]["parquet"]["path"])
    cache_dates = pl.read_parquet(output_dir / "market_inputs" / "dates.parquet")
    cache_date_idx = _load_cache_array(output_dir, "date_idx.npy")
    cache_position = {
        int(value): index for index, value in enumerate(cache_date_idx.tolist())
    }
    reports = output_dir / "reports"
    reports.mkdir(exist_ok=True)
    rows: list[dict[str, object]] = []
    frictionless_assumption = output_dir / "frictionless_assumption.json"
    _atomic_json(
        frictionless_assumption,
        {
            "fee_bps": 0.0,
            "full_spread": 0.0,
            "all_other_configuration_and_market_inputs": "unchanged",
        },
    )
    for fold in FOLDS:
        source = design["fold_sources"][fold]
        archive = load_discovery_prediction_archive(
            Path(source["ensemble_prediction"]["path"]),
            Path(source["prediction_reference"]["path"]),
            Path(source["execution_manifest"]["path"]),
            store,
        )
        positions = np.asarray(
            [cache_position[int(value)] for value in archive.date_idx]
        )
        fold_dates = tuple(
            cache_dates.filter(pl.col("date_idx").is_in(archive.date_idx))["trade_date"]
        )
        if len(fold_dates) != archive.date_idx.size:
            raise ValueError(f"Market dates differ from prediction dates: {fold}")
        ranks, valid, _, refresh = expand_refreshes(
            archive.ranks,
            archive.valid,
            archive.refresh_minutes,
            EQUITY_SESSION_MINUTES,
        )
        dtype = torch.float64
        spread_values = np.asarray(
            _load_cache_array(output_dir, "full_spread.npy")[positions]
        )
        market_kwargs = {
            "open_price": torch.as_tensor(
                np.asarray(_load_cache_array(output_dir, "open_price.npy")[positions]),
                dtype=dtype,
            ),
            "open_observed": torch.as_tensor(
                np.asarray(
                    _load_cache_array(output_dir, "open_observed.npy")[positions]
                )
            ),
            "active": torch.as_tensor(
                np.asarray(_load_cache_array(output_dir, "active.npy")[positions])
            ),
            "adv20_brl": torch.as_tensor(
                np.asarray(_load_cache_array(output_dir, "adv20_brl.npy")[positions]),
                dtype=dtype,
            ),
            "minute_notional20_brl": torch.as_tensor(
                np.asarray(
                    _load_cache_array(output_dir, "minute_notional20_brl.npy")[
                        positions
                    ]
                ),
                dtype=dtype,
            ),
            "daily_cdi_rate": torch.as_tensor(
                load_daily_cdi_rates(
                    cdi_path,
                    fold_dates,
                    str(design["cdi"]["parquet"]["sha256"]),
                ),
                dtype=dtype,
            ),
        }
        rank_tensor = torch.as_tensor(ranks, dtype=dtype)
        valid_tensor = torch.as_tensor(valid)
        refresh_tensor = torch.as_tensor(refresh)
        sigma_tensor = torch.as_tensor(
            np.asarray(_load_cache_array(output_dir, "sigma_daily.npy")[positions]),
            dtype=dtype,
        )
        inputs = {
            "frozen_design": _sha256(design_path),
            "prediction": str(source["ensemble_prediction"]["sha256"]),
            "prediction_reference": str(source["prediction_reference"]["sha256"]),
            "prediction_wrapper": str(source["execution_manifest"]["sha256"]),
            "prediction_source": str(source["source_manifest"]["sha256"]),
            "market_inputs": _sha256(market_manifest_path),
            "cdi": str(design["cdi"]["parquet"]["sha256"]),
            "roll_schedule": str(design["roll_schedule"]["sha256"]),
        }
        for cell in execution_cells():
            config = ExecutionConfig(
                band=float(cell["band"]),
                horizon_blend=tuple(float(value) for value in cell["horizon_blend"]),
            )
            measured_path = reports / fold / f"{cell['cell_id']}__measured.json"
            if measured_path.exists():
                measured, measured_report = _load_existing_report(
                    measured_path, config=config, input_sha256=inputs
                )
            else:
                measured = _run_one(
                    market=MarketReplay(
                        full_spread=torch.as_tensor(spread_values, dtype=dtype),
                        **market_kwargs,
                    ),
                    ranks=rank_tensor,
                    rank_valid=valid_tensor,
                    refresh_mask=refresh_tensor,
                    sigma=sigma_tensor,
                    config=config,
                    dates=fold_dates,
                )
                measured_report = write_execution_report(
                    measured_path, config=config, input_sha256=inputs, daily=measured
                )
            frictionless_config = replace(config, fee_bps=0.0)
            frictionless_path = reports / fold / f"{cell['cell_id']}__frictionless.json"
            frictionless_inputs = {
                **inputs,
                "frictionless_assumption": _sha256(frictionless_assumption),
            }
            if frictionless_path.exists():
                frictionless, frictionless_report = _load_existing_report(
                    frictionless_path,
                    config=frictionless_config,
                    input_sha256=frictionless_inputs,
                )
            else:
                frictionless = _run_one(
                    market=MarketReplay(
                        full_spread=torch.zeros_like(
                            torch.as_tensor(spread_values, dtype=dtype)
                        ),
                        **market_kwargs,
                    ),
                    ranks=rank_tensor,
                    rank_valid=valid_tensor,
                    refresh_mask=refresh_tensor,
                    sigma=sigma_tensor,
                    config=frictionless_config,
                    dates=fold_dates,
                )
                frictionless_report = write_execution_report(
                    frictionless_path,
                    config=frictionless_config,
                    input_sha256=frictionless_inputs,
                    daily=frictionless,
                )
            metrics = daily_readout(measured, config.nav_brl)
            frictionless_metrics = daily_readout(frictionless, config.nav_brl)
            rows.append(
                {
                    "fold": fold,
                    "cell_id": cell["cell_id"],
                    "band": cell["band"],
                    "blend_name": cell["blend_name"],
                    "config_sha256": config.sha256,
                    "net_pnl_brl": float(sum(row.net_pnl_brl for row in measured)),
                    "gross_pnl_brl": float(sum(row.gross_pnl_brl for row in measured)),
                    "spread_cost_brl": float(
                        sum(row.spread_cost_brl for row in measured)
                    ),
                    "fees_brl": float(sum(row.fees_brl for row in measured)),
                    "cdi_earned_brl": float(
                        sum(row.cdi_earned_brl for row in measured)
                    ),
                    "turnover_brl": float(sum(row.turnover_brl for row in measured)),
                    "forced_fill_count": int(
                        sum(row.forced_fill_count for row in measured)
                    ),
                    **metrics,
                    "frictionless_net_pnl_brl": float(
                        sum(row.net_pnl_brl for row in frictionless)
                    ),
                    "frictionless_annualized_net_sharpe": frictionless_metrics[
                        "annualized_net_sharpe"
                    ],
                    "frictionless_minus_measured_net_pnl_brl": float(
                        sum(row.net_pnl_brl for row in frictionless)
                        - sum(row.net_pnl_brl for row in measured)
                    ),
                    "measured_report_sha256": measured_report["sha256"],
                    "frictionless_report_sha256": frictionless_report["sha256"],
                }
            )
    summary_path = output_dir / "cell_fold_summary.parquet"
    pl.DataFrame(rows).write_parquet(summary_path)
    rotation, designation = rotation_designation(rows)
    c0_cell = next(
        cell
        for cell in execution_cells()
        if cell["cell_id"] == designation["c0_cell_id"]
    )
    designation["c0_configuration"] = c0_cell
    rotation_path = output_dir / "rotation_table.parquet"
    pl.DataFrame(rotation).write_parquet(rotation_path)
    designation_path = output_dir / "c0_designation.json"
    _atomic_json(designation_path, designation)
    completed = {
        **manifest,
        "status": "completed",
        "completed_at": _now(),
        "operational_commit": operational_commit,
        "market_inputs": _artifact(market_manifest_path),
        "cell_fold_summary": _artifact(summary_path),
        "rotation_table": _artifact(rotation_path),
        "c0_designation": _artifact(designation_path),
        "report_count": len(list(reports.rglob("*.json"))),
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    _atomic_json(manifest_path, completed)
    audit_program(output_dir)
    return designation_path


def audit_program(output_dir: Path) -> Path:
    output_dir = output_dir.resolve()
    manifest = _read_json(output_dir / "program_manifest.json")
    designation = _read_json(output_dir / "c0_designation.json")
    summary = pl.read_parquet(output_dir / "cell_fold_summary.parquet")
    rotation = pl.read_parquet(output_dir / "rotation_table.parquet")
    reports = sorted((output_dir / "reports").rglob("*.json"))
    if (
        manifest.get("status") != "completed"
        or manifest.get("official_validation_accessed") is not False
        or manifest.get("test_accessed") is not False
        or designation.get("official_validation_accessed") is not False
        or designation.get("test_accessed") is not False
        or summary.height != 36
        or rotation.height != 36
        or len(reports) != 72
        or designation.get("c0_cell_id") not in set(summary["cell_id"])
    ):
        raise ValueError("Experiment 52 completion contract failed")
    for report in reports:
        sidecar = report.with_suffix(report.suffix + ".sha256")
        digest, name = sidecar.read_text(encoding="utf-8").strip().split("  ")
        payload = _read_json(report)
        if (
            name != report.name
            or digest != _sha256(report)
            or payload.get("schema") != "B3_EXECUTION_BACKTEST_REPORT_V1"
            or len(payload.get("daily", [])) == 0
        ):
            raise ValueError(f"Execution report audit failed: {report}")
    files = [
        path
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name != "final_audit.json"
    ]
    audit = {
        "schema": "EXPERIMENT52_FINAL_AUDIT_V1",
        "created_at": _now(),
        "status": "passed",
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "artifacts": [
            {
                "path": path.relative_to(output_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in files
        ],
        "summary_rows": summary.height,
        "rotation_rows": rotation.height,
        "report_count": len(reports),
        "c0_cell_id": designation["c0_cell_id"],
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    path = output_dir / "final_audit.json"
    _atomic_json(path, audit)
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Experiment 52 C0 baseline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--store", type=Path, required=True)
    freeze.add_argument("--experiment41-root", type=Path, required=True)
    freeze.add_argument("--economics-dir", type=Path, required=True)
    freeze.add_argument("--output-dir", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--store", type=Path, required=True)
    run.add_argument("--economics-dir", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "freeze":
        path = freeze_program(
            store=args.store,
            experiment41_root=args.experiment41_root,
            economics_dir=args.economics_dir,
            output_dir=args.output_dir,
        )
    elif args.command == "run":
        path = run_program(
            store=args.store,
            economics_dir=args.economics_dir,
            output_dir=args.output_dir,
        )
    else:
        path = audit_program(args.output_dir)
    print(path)


if __name__ == "__main__":
    main()
