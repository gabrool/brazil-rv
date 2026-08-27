from __future__ import annotations

import argparse
import math
import os
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import torch

from ..modeling.data import feature_store_identity
from ..modeling.provenance import repository_commit
from ..preprocessing.contract import EQUITY_SESSION_MINUTES
from .config import ExecutionConfig
from .experiment52 import (
    FOLDS,
    _artifact,
    _atomic_json,
    _daily_results,
    _load_cache_array,
    _load_existing_report,
    _read_json,
    _sha256,
    daily_readout,
)
from .inputs import (
    expand_refreshes,
    load_daily_cdi_rates,
    load_discovery_prediction_archive,
)
from .policy import ConcentratedPolicy
from .report import write_execution_report
from .simulator import MarketReplay, simulate, tradeable_universe

SCHEMA = "EXPERIMENT53_FEASIBLE_REGION_V1"
KS = (10, 20, 40)
BANDS = (0.5, 1.5)
COST_SCALES = (0.0, 1.0)
GROSS_TARGETS = (1.0, 2.0)
UNIVERSES = ("full", "top_half_adv")
VARIANTS = ("measured", "frictionless", "half_spread")
NAME_CAP_FRACTION_OF_GROSS = 0.05
C1_MINIMUM_DEPLOYED_GROSS_FRACTION = 0.50


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _number_name(value: float) -> str:
    return str(value).replace(".", "p")


def execution_cells() -> tuple[dict[str, object], ...]:
    cells = []
    for k in KS:
        for band in BANDS:
            for cost_scale in COST_SCALES:
                for gross_target in GROSS_TARGETS:
                    for universe in UNIVERSES:
                        config = ExecutionConfig(
                            gross_target=gross_target,
                            name_cap_fraction_of_gross=(NAME_CAP_FRACTION_OF_GROSS),
                            horizon_blend=(1 / 3, 1 / 3, 1 / 3),
                            band=band,
                            cost_band_scale=cost_scale,
                            concentration_k=k,
                            top_half_adv=universe == "top_half_adv",
                        )
                        cells.append(
                            {
                                "cell_id": (
                                    f"k{k}__band{_number_name(band)}"
                                    f"__c{_number_name(cost_scale)}"
                                    f"__gross{_number_name(gross_target)}"
                                    f"__universe_{universe}"
                                ),
                                "k": k,
                                "band_base": band,
                                "cost_band_scale": cost_scale,
                                "gross_target": gross_target,
                                "universe": universe,
                                "measured_config": config.to_dict(),
                                "measured_config_sha256": config.sha256,
                            }
                        )
    if len(cells) != 48 or len({str(row["cell_id"]) for row in cells}) != 48:
        raise ArithmeticError("Experiment 53 grid must contain exactly 48 cells")
    return tuple(cells)


def _config(cell: Mapping[str, object], variant: str) -> ExecutionConfig:
    config = ExecutionConfig(
        gross_target=float(cell["gross_target"]),
        name_cap_fraction_of_gross=NAME_CAP_FRACTION_OF_GROSS,
        horizon_blend=(1 / 3, 1 / 3, 1 / 3),
        band=float(cell["band_base"]),
        cost_band_scale=float(cell["cost_band_scale"]),
        concentration_k=int(cell["k"]),
        top_half_adv=str(cell["universe"]) == "top_half_adv",
    )
    if variant == "measured":
        return config
    if variant == "frictionless":
        return replace(config, fee_bps=0.0, spread_schedule_multiplier=0.0)
    if variant == "half_spread":
        return replace(config, spread_schedule_multiplier=0.5)
    raise ValueError(f"Unknown Experiment 53 variant: {variant}")


def _verify_experiment52(root: Path) -> dict[str, object]:
    manifest = _read_json(root / "program_manifest.json")
    audit_path = root / "final_audit.json"
    audit = _read_json(audit_path)
    designation = _read_json(root / "c0_designation.json")
    if (
        manifest.get("status") != "completed"
        or audit.get("status") != "passed"
        or manifest.get("official_validation_accessed") is not False
        or manifest.get("test_accessed") is not False
        or audit.get("official_validation_accessed") is not False
        or audit.get("test_accessed") is not False
        or designation.get("c0_cell_id") != "band_2p0__blend_equal"
    ):
        raise ValueError("Experiment 52 source contract differs")
    for record in audit.get("artifacts", []):
        path = root / str(record["path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(record["bytes"])
            or _sha256(path) != record["sha256"]
        ):
            raise ValueError(f"Experiment 52 source audit differs: {path}")
    return {
        "root": str(root),
        "program_manifest": _artifact(root / "program_manifest.json"),
        "frozen_design": _artifact(root / "frozen_design.json"),
        "market_inputs": _artifact(root / "market_inputs" / "manifest.json"),
        "c0_designation": _artifact(root / "c0_designation.json"),
        "final_audit": _artifact(audit_path),
        "final_audit_artifact_count": len(audit["artifacts"]),
    }


def freeze_program(
    *, experiment52_root: Path, preregistration: Path, output_dir: Path
) -> Path:
    experiment52_root, preregistration, output_dir = (
        path.resolve() for path in (experiment52_root, preregistration, output_dir)
    )
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    try:
        source = _verify_experiment52(experiment52_root)
        source_design = _read_json(experiment52_root / "frozen_design.json")
        store = Path(str(source_design["store"]["path"]))
        store_identity = feature_store_identity(store)
        if store_identity != source_design["store"]["identity"]:
            raise ValueError("Experiment 52 canonical store identity differs")
        design = {
            "schema": SCHEMA,
            "status": "frozen",
            "created_at": _now(),
            "repository_commit": repository_commit(),
            "preregistration": _artifact(preregistration),
            "source_experiment52": source,
            "store": {"path": str(store), "identity": store_identity},
            "fold_sources": source_design["fold_sources"],
            "cdi": source_design["cdi"],
            "roll_schedule": source_design["roll_schedule"],
            "cells": list(execution_cells()),
            "variants": {
                "measured": {"fee_bps": 2.0, "spread_schedule_multiplier": 1.0},
                "frictionless": {
                    "fee_bps": 0.0,
                    "spread_schedule_multiplier": 0.0,
                },
                "half_spread": {
                    "fee_bps": 2.0,
                    "spread_schedule_multiplier": 0.5,
                },
            },
            "amendment_a53_1": {
                "timing": "pre-score",
                "name_cap_fraction_of_gross": (NAME_CAP_FRACTION_OF_GROSS),
                "capacity_completion": (
                    "extend each side in deterministic rank order until the "
                    "inclusive name-cap-and-ADV-cap capacity reaches its side target"
                ),
                "selection_extension_telemetry": "one count per refresh",
                "c1_minimum_mean_deployed_gross_fraction_of_target_per_fold": (
                    C1_MINIMUM_DEPLOYED_GROSS_FRACTION
                ),
            },
            "roll_sanity_contract": {
                "status": "ratio_unavailable_under_frozen_inputs",
                "reason": (
                    "Experiment 52 does not contain a causal historical quoted-tick "
                    "archive; MT5 historical spread is prohibited as a market spread "
                    "and the point-in-time catalogue snapshot is not historical"
                ),
                "effect": "informational table only; no schedule change",
            },
            "rotation_rule": (
                "measured variants only; exclude every cell failing the 50% "
                "mean-deployed-gross guard on any fold; otherwise the Experiment 52 "
                "two-fold rotation majority and mean-heldout-Sharpe tie-break"
            ),
            "interpretation_contract": {
                "net_positive": (
                    "existence proof and lower bound for a future learned policy"
                ),
                "all_negative": (
                    "verdict on this hand-policy family only; never a conclusion "
                    "that the alpha is untradeable"
                ),
                "retention": "retain both C0 and C1 as learned-policy baselines",
            },
            "cpu_only": True,
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
                "created_at": _now(),
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


def liquidity_terciles(adv: torch.Tensor, tradeable: torch.Tensor) -> torch.Tensor:
    if adv.shape != tradeable.shape or adv.ndim != 2:
        raise ValueError("ADV and tradeable mask must align as [day,name]")
    output = torch.zeros_like(tradeable, dtype=torch.int8)
    for day in range(adv.shape[0]):
        names = torch.nonzero(tradeable[day], as_tuple=False).flatten()
        if not names.numel():
            continue
        order = names[torch.argsort(adv[day, names], stable=True)]
        for tercile, group in enumerate(torch.tensor_split(order, 3), start=1):
            output[day, group] = tercile
    return output


def _safe_share(value: float, total: float) -> float | None:
    return value / total if total != 0.0 else None


def _diagnostics(
    *,
    result: object,
    market: MarketReplay,
    config: ExecutionConfig,
    fold: str,
    cell_id: str,
    variant: str,
    dates: Sequence[date],
    refresh_mask: torch.Tensor,
) -> tuple[list[dict[str, object]], list[dict[str, object]], pl.DataFrame]:
    tradeable = tradeable_universe(market, config)
    tiers = liquidity_terciles(market.adv20_brl, tradeable)
    arrays = {
        name: getattr(result, name).detach().cpu()
        for name in (
            "turnover_by_name_brl",
            "gross_pnl_by_name_brl",
            "spread_cost_by_name_brl",
            "fees_by_name_brl",
            "round_trip_count_by_name",
            "round_trip_gross_pnl_by_name_brl",
            "round_trip_cost_by_name_brl",
        )
    }
    totals = {
        "turnover_brl": float(arrays["turnover_by_name_brl"].sum()),
        "spread_cost_brl": float(arrays["spread_cost_by_name_brl"].sum()),
        "gross_pnl_brl": float(arrays["gross_pnl_by_name_brl"].sum()),
        "net_pnl_before_cdi_brl": float(
            (
                arrays["gross_pnl_by_name_brl"]
                - arrays["spread_cost_by_name_brl"]
                - arrays["fees_by_name_brl"]
            ).sum()
        ),
    }
    liquidity_rows = []
    trade_rows = []
    for tercile in (1, 2, 3):
        mask = tiers == tercile
        values = {
            "turnover_brl": float(arrays["turnover_by_name_brl"][mask].sum()),
            "spread_cost_brl": float(arrays["spread_cost_by_name_brl"][mask].sum()),
            "gross_pnl_brl": float(arrays["gross_pnl_by_name_brl"][mask].sum()),
            "net_pnl_before_cdi_brl": float(
                (
                    arrays["gross_pnl_by_name_brl"]
                    - arrays["spread_cost_by_name_brl"]
                    - arrays["fees_by_name_brl"]
                )[mask].sum()
            ),
        }
        liquidity_rows.append(
            {
                "fold": fold,
                "cell_id": cell_id,
                "variant": variant,
                "liquidity_tercile": tercile,
                **values,
                **{
                    f"{name}_share": _safe_share(value, totals[name])
                    for name, value in values.items()
                },
            }
        )
        count = int(arrays["round_trip_count_by_name"][mask].sum())
        gross = float(arrays["round_trip_gross_pnl_by_name_brl"][mask].sum())
        cost = float(arrays["round_trip_cost_by_name_brl"][mask].sum())
        trade_rows.append(
            {
                "fold": fold,
                "cell_id": cell_id,
                "variant": variant,
                "liquidity_tercile": tercile,
                "round_trip_count": count,
                "gross_alpha_brl": gross,
                "cost_brl": cost,
                "mean_gross_alpha_per_round_trip_brl": (
                    gross / count if count else None
                ),
                "mean_cost_per_round_trip_brl": cost / count if count else None,
            }
        )

    extension = result.selection_extended_count.detach().cpu().numpy()
    refresh = refresh_mask.detach().cpu().numpy().astype(bool)
    day_idx, minute_idx = np.nonzero(refresh)
    extension_rows = pl.DataFrame(
        {
            "fold": [fold] * len(day_idx),
            "cell_id": [cell_id] * len(day_idx),
            "variant": [variant] * len(day_idx),
            "trade_date": [dates[index] for index in day_idx],
            "refresh_minute": minute_idx.astype(np.int16),
            "selection_extended_count": extension[day_idx, minute_idx].astype(np.int16),
        }
    )
    return liquidity_rows, trade_rows, extension_rows


def _bundle_paths(root: Path, fold: str, cell_id: str, variant: str) -> dict[str, Path]:
    stem = root / fold / f"{cell_id}__{variant}"
    return {
        "report": stem.with_suffix(".json"),
        "liquidity": stem.with_name(stem.name + "__liquidity.parquet"),
        "trades": stem.with_name(stem.name + "__trades.parquet"),
        "extensions": stem.with_name(stem.name + "__extensions.parquet"),
        "readout": stem.with_name(stem.name + "__readout.json"),
    }


def _load_bundle(paths: Mapping[str, Path]) -> dict[str, object] | None:
    if not paths["readout"].is_file():
        return None
    payload = _read_json(paths["readout"])
    for name in ("report", "liquidity", "trades", "extensions"):
        path = paths[name]
        record = payload.get("artifacts", {}).get(name)
        if (
            not path.is_file()
            or not isinstance(record, dict)
            or record.get("sha256") != _sha256(path)
        ):
            return None
    return payload


def _run_bundle(
    *,
    paths: Mapping[str, Path],
    market: MarketReplay,
    ranks: torch.Tensor,
    rank_valid: torch.Tensor,
    refresh_mask: torch.Tensor,
    sigma: torch.Tensor,
    config: ExecutionConfig,
    inputs: Mapping[str, str],
    dates: Sequence[date],
    fold: str,
    cell_id: str,
    variant: str,
) -> dict[str, object]:
    existing = _load_bundle(paths)
    if existing is not None:
        _load_existing_report(paths["report"], config=config, input_sha256=inputs)
        return existing
    result = simulate(
        market,
        ranks,
        rank_valid,
        refresh_mask,
        sigma,
        ConcentratedPolicy(config),
        config,
    )
    daily = _daily_results(dates, result)
    report = write_execution_report(
        paths["report"], config=config, input_sha256=inputs, daily=daily
    )
    liquidity, trades, extensions = _diagnostics(
        result=result,
        market=market,
        config=config,
        fold=fold,
        cell_id=cell_id,
        variant=variant,
        dates=dates,
        refresh_mask=refresh_mask,
    )
    pl.DataFrame(liquidity).write_parquet(paths["liquidity"])
    pl.DataFrame(trades).write_parquet(paths["trades"])
    extensions.write_parquet(paths["extensions"])
    metrics = daily_readout(daily, config.nav_brl)
    all_cash_cdi = float(config.nav_brl * market.daily_cdi_rate.detach().cpu().sum())
    mean_deployed = float(result.mean_deployed_gross_brl.mean())
    deployed_fraction = mean_deployed / (config.nav_brl * config.gross_target)
    summary = {
        "fold": fold,
        "cell_id": cell_id,
        "variant": variant,
        "config_sha256": config.sha256,
        "net_pnl_brl": float(result.net_pnl_brl.sum()),
        "gross_pnl_brl": float(result.gross_pnl_brl.sum()),
        "spread_cost_brl": float(result.spread_cost_brl.sum()),
        "fees_brl": float(result.fees_brl.sum()),
        "cdi_earned_brl": float(result.cdi_earned_brl.sum()),
        "turnover_brl": float(result.turnover_brl.sum()),
        "all_cash_cdi_brl": all_cash_cdi,
        "net_excess_over_all_cash_cdi_brl": float(result.net_pnl_brl.sum())
        - all_cash_cdi,
        "mean_deployed_gross_brl": mean_deployed,
        "mean_deployed_gross_fraction_of_target": deployed_fraction,
        "c1_fold_eligible": (
            variant == "measured"
            and deployed_fraction >= C1_MINIMUM_DEPLOYED_GROSS_FRACTION
        ),
        "selection_extended_refresh_count": int(
            (result.selection_extended_count > 0).sum()
        ),
        "selection_extended_name_count": int(result.selection_extended_count.sum()),
        **metrics,
    }
    payload = {
        "schema": "EXPERIMENT53_RUN_READOUT_V1",
        "summary": summary,
        "artifacts": {
            "report": report,
            "liquidity": _artifact(paths["liquidity"]),
            "trades": _artifact(paths["trades"]),
            "extensions": _artifact(paths["extensions"]),
        },
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    _atomic_json(paths["readout"], payload)
    return payload


def rotation_designation(
    rows: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    measured = [row for row in rows if row["variant"] == "measured"]
    by_key = {(str(row["cell_id"]), str(row["fold"])): row for row in measured}
    cells = sorted({cell for cell, _ in by_key})
    if len(cells) != 48 or len(by_key) != 144:
        raise ValueError("C1 input must contain all 144 measured cell-fold results")
    eligible = {
        cell
        for cell in cells
        if all(bool(by_key[(cell, fold)]["c1_fold_eligible"]) for fold in FOLDS)
    }
    if not eligible:
        raise ValueError("No Experiment 53 cell passes the C1 deployment guard")
    table = []
    winners = []
    for heldout in FOLDS:
        training = tuple(fold for fold in FOLDS if fold != heldout)
        ranked = sorted(
            eligible,
            key=lambda cell: (
                -float(
                    np.mean(
                        [
                            float(by_key[(cell, fold)]["annualized_net_sharpe"])
                            for fold in training
                        ]
                    )
                ),
                cell,
            ),
        )
        winners.append(ranked[0])
        rank_by_cell = {cell: rank for rank, cell in enumerate(ranked, start=1)}
        for cell in cells:
            table.append(
                {
                    "heldout_fold": heldout,
                    "cell_id": cell,
                    "c1_eligible": cell in eligible,
                    "rotation_rank": rank_by_cell.get(cell),
                    "other_fold_mean_annualized_net_sharpe": (
                        float(
                            np.mean(
                                [
                                    float(by_key[(cell, fold)]["annualized_net_sharpe"])
                                    for fold in training
                                ]
                            )
                        )
                        if cell in eligible
                        else None
                    ),
                    "heldout_annualized_net_sharpe": float(
                        by_key[(cell, heldout)]["annualized_net_sharpe"]
                    ),
                    "heldout_net_pnl_brl": float(
                        by_key[(cell, heldout)]["net_pnl_brl"]
                    ),
                    "heldout_net_excess_over_all_cash_cdi_brl": float(
                        by_key[(cell, heldout)]["net_excess_over_all_cash_cdi_brl"]
                    ),
                }
            )
    counts = Counter(winners)
    maximum = max(counts.values())
    finalists = [cell for cell, count in counts.items() if count == maximum]
    means = {
        cell: float(
            np.mean(
                [float(by_key[(cell, fold)]["annualized_net_sharpe"]) for fold in FOLDS]
            )
        )
        for cell in finalists
    }
    best = max(means.values())
    selected = [
        cell
        for cell in finalists
        if math.isclose(means[cell], best, rel_tol=0.0, abs_tol=1e-15)
    ]
    if len(selected) != 1:
        raise ValueError("C1 rule remains exactly tied after its frozen tie-break")
    return table, {
        "schema": "EXPERIMENT53_C1_DESIGNATION_V1",
        "rule": (
            "50%-deployed-gross eligible cells only; cell winning most two-fold "
            "rotations; tie uses higher mean held-out annualized net Sharpe"
        ),
        "c1_cell_id": selected[0],
        "rotation_win_count": maximum,
        "rotation_winners": [
            {"heldout_fold": fold, "cell_id": cell}
            for fold, cell in zip(FOLDS, winners, strict=True)
        ],
        "eligible_cell_count": len(eligible),
        "ineligible_cell_ids": sorted(set(cells) - eligible),
        "official_validation_accessed": False,
        "test_accessed": False,
    }


def _roll_sanity(source_root: Path, output_dir: Path) -> tuple[Path, Path]:
    dates = pl.read_parquet(source_root / "market_inputs" / "dates.parquet")
    spread = np.asarray(_load_cache_array(source_root, "full_spread.npy"))
    source_design = _read_json(source_root / "frozen_design.json")
    store = Path(str(source_design["store"]["path"]))
    securities = pl.read_parquet(store / "equity_index.parquet").sort("equity_slot")
    rows = []
    for slot, security_id in enumerate(securities["security_id"]):
        values = spread[:, slot]
        finite = values[np.isfinite(values)]
        rows.append(
            {
                "security_id": security_id,
                "observed_date_count": int(finite.size),
                "median_lagged_roll_spread_bps": (
                    float(np.median(finite) * 10_000) if finite.size else None
                ),
                "quoted_tick_model_bps": None,
                "roll_to_tick_model_ratio": None,
                "suspect_ratio_gt_3": None,
                "ratio_status": "unavailable_under_frozen_input_contract",
            }
        )
    table_path = output_dir / "roll_schedule_sanity.parquet"
    pl.DataFrame(rows).write_parquet(table_path)
    analysis_path = output_dir / "roll_schedule_sanity.json"
    _atomic_json(
        analysis_path,
        {
            "schema": "EXPERIMENT53_ROLL_SANITY_V1",
            "status": "tick_ratio_unavailable",
            "date_count": dates.height,
            "security_count": len(rows),
            "lagged_roll_spread_bps_distribution": {
                name: float(value)
                for name, value in zip(
                    ("minimum", "p25", "median", "p75", "maximum"),
                    np.quantile(
                        spread[np.isfinite(spread)] * 10_000, [0, 0.25, 0.5, 0.75, 1]
                    ),
                    strict=True,
                )
            },
            "reason": (
                "No causal historical quoted-tick archive exists in the frozen "
                "Experiment 52 input contract. MT5 historical spread is not a "
                "market spread, and the catalogue is a current snapshot."
            ),
            "schedule_changed": False,
            "table": _artifact(table_path),
            "official_validation_accessed": False,
            "test_accessed": False,
        },
    )
    return table_path, analysis_path


def run_program(*, output_dir: Path) -> Path:
    output_dir = output_dir.resolve()
    manifest_path = output_dir / "program_manifest.json"
    design_path = output_dir / "frozen_design.json"
    manifest = _read_json(manifest_path)
    design = _read_json(design_path)
    if (
        manifest.get("status") != "frozen"
        or manifest.get("frozen_design", {}).get("sha256") != _sha256(design_path)
        or design.get("repository_commit") != repository_commit()
        or design.get("cells") != list(execution_cells())
    ):
        raise ValueError("Experiment 53 frozen state differs")
    source_root = Path(str(design["source_experiment52"]["root"]))
    _verify_experiment52(source_root)
    source_design = _read_json(source_root / "frozen_design.json")
    store = Path(str(design["store"]["path"]))
    cache_dates = pl.read_parquet(source_root / "market_inputs" / "dates.parquet")
    cache_idx = _load_cache_array(source_root, "date_idx.npy")
    cache_position = {int(value): index for index, value in enumerate(cache_idx)}
    reports_root = output_dir / "reports"
    reports_root.mkdir(exist_ok=True)
    variant_contract_path = output_dir / "variant_contract.json"
    _atomic_json(variant_contract_path, design["variants"])
    rows = []
    liquidity_paths = []
    trade_paths = []
    extension_paths = []
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
        ranks, valid, _, refresh = expand_refreshes(
            archive.ranks,
            archive.valid,
            archive.refresh_minutes,
            EQUITY_SESSION_MINUTES,
        )
        dtype = torch.float64
        market = MarketReplay(
            open_price=torch.as_tensor(
                np.asarray(_load_cache_array(source_root, "open_price.npy")[positions]),
                dtype=dtype,
            ),
            open_observed=torch.as_tensor(
                np.asarray(
                    _load_cache_array(source_root, "open_observed.npy")[positions]
                )
            ),
            active=torch.as_tensor(
                np.asarray(_load_cache_array(source_root, "active.npy")[positions])
            ),
            full_spread=torch.as_tensor(
                np.asarray(
                    _load_cache_array(source_root, "full_spread.npy")[positions]
                ),
                dtype=dtype,
            ),
            adv20_brl=torch.as_tensor(
                np.asarray(_load_cache_array(source_root, "adv20_brl.npy")[positions]),
                dtype=dtype,
            ),
            minute_notional20_brl=torch.as_tensor(
                np.asarray(
                    _load_cache_array(source_root, "minute_notional20_brl.npy")[
                        positions
                    ]
                ),
                dtype=dtype,
            ),
            daily_cdi_rate=torch.as_tensor(
                load_daily_cdi_rates(
                    Path(source_design["cdi"]["parquet"]["path"]),
                    fold_dates,
                    str(source_design["cdi"]["parquet"]["sha256"]),
                ),
                dtype=dtype,
            ),
        )
        rank_tensor = torch.as_tensor(ranks, dtype=dtype)
        valid_tensor = torch.as_tensor(valid)
        refresh_tensor = torch.as_tensor(refresh)
        sigma_tensor = torch.as_tensor(
            np.asarray(_load_cache_array(source_root, "sigma_daily.npy")[positions]),
            dtype=dtype,
        )
        inputs = {
            "frozen_design": _sha256(design_path),
            "prediction": str(source["ensemble_prediction"]["sha256"]),
            "prediction_reference": str(source["prediction_reference"]["sha256"]),
            "prediction_wrapper": str(source["execution_manifest"]["sha256"]),
            "prediction_source": str(source["source_manifest"]["sha256"]),
            "market_inputs": str(
                design["source_experiment52"]["market_inputs"]["sha256"]
            ),
            "cdi": str(source_design["cdi"]["parquet"]["sha256"]),
            "roll_schedule": str(source_design["roll_schedule"]["sha256"]),
            "variant_contract": _sha256(variant_contract_path),
        }
        for cell in execution_cells():
            cell_id = str(cell["cell_id"])
            for variant in VARIANTS:
                config = _config(cell, variant)
                paths = _bundle_paths(reports_root, fold, cell_id, variant)
                paths["report"].parent.mkdir(parents=True, exist_ok=True)
                bundle = _run_bundle(
                    paths=paths,
                    market=market,
                    ranks=rank_tensor,
                    rank_valid=valid_tensor,
                    refresh_mask=refresh_tensor,
                    sigma=sigma_tensor,
                    config=config,
                    inputs=inputs,
                    dates=fold_dates,
                    fold=fold,
                    cell_id=cell_id,
                    variant=variant,
                )
                rows.append(bundle["summary"])
                liquidity_paths.append(paths["liquidity"])
                trade_paths.append(paths["trades"])
                extension_paths.append(paths["extensions"])
    summary_path = output_dir / "cell_fold_variant_summary.parquet"
    pl.DataFrame(rows).write_parquet(summary_path)
    liquidity_path = output_dir / "liquidity_terciles.parquet"
    pl.concat([pl.read_parquet(path) for path in liquidity_paths]).write_parquet(
        liquidity_path
    )
    trades_path = output_dir / "per_trade_economics.parquet"
    pl.concat([pl.read_parquet(path) for path in trade_paths]).write_parquet(
        trades_path
    )
    extensions_path = output_dir / "selection_extensions.parquet"
    pl.concat([pl.read_parquet(path) for path in extension_paths]).write_parquet(
        extensions_path
    )
    rotation, designation = rotation_designation(rows)
    c1 = next(
        cell
        for cell in execution_cells()
        if cell["cell_id"] == designation["c1_cell_id"]
    )
    designation["c1_configuration"] = c1
    designation["c0_cell_id"] = "band_2p0__blend_equal"
    rotation_path = output_dir / "rotation_table.parquet"
    pl.DataFrame(rotation).write_parquet(rotation_path)
    designation_path = output_dir / "c1_designation.json"
    _atomic_json(designation_path, designation)
    roll_table, roll_analysis = _roll_sanity(source_root, output_dir)
    measured = pl.DataFrame([row for row in rows if row["variant"] == "measured"])
    measured_by_cell = measured.group_by("cell_id").agg(
        pl.col("net_pnl_brl").sum(),
        pl.col("net_excess_over_all_cash_cdi_brl").sum(),
    )
    result_path = output_dir / "experiment53_result.json"
    positive_net = measured_by_cell.filter(pl.col("net_pnl_brl") > 0)
    positive_excess = measured_by_cell.filter(
        pl.col("net_excess_over_all_cash_cdi_brl") > 0
    )
    _atomic_json(
        result_path,
        {
            "schema": "EXPERIMENT53_RESULT_V1",
            "interpretation": (
                "measured net-positive cells are an existence proof and lower bound"
                if positive_net.height
                else (
                    "all measured cells were negative; this is a verdict on this "
                    "hand-policy family only and is not evidence that the alpha is "
                    "untradeable"
                )
            ),
            "measured_net_positive_cell_ids": sorted(positive_net["cell_id"]),
            "measured_net_excess_positive_cell_ids": sorted(positive_excess["cell_id"]),
            "c0_cell_id": "band_2p0__blend_equal",
            "c1_cell_id": designation["c1_cell_id"],
            "official_validation_accessed": False,
            "test_accessed": False,
        },
    )
    completed = {
        **manifest,
        "status": "completed",
        "completed_at": _now(),
        "operational_commit": os.environ.get(
            "EXPERIMENT53_OPERATIONAL_COMMIT", repository_commit()
        ),
        "cell_fold_variant_summary": _artifact(summary_path),
        "liquidity_terciles": _artifact(liquidity_path),
        "per_trade_economics": _artifact(trades_path),
        "selection_extensions": _artifact(extensions_path),
        "rotation_table": _artifact(rotation_path),
        "c1_designation": _artifact(designation_path),
        "roll_schedule_sanity": _artifact(roll_table),
        "roll_schedule_sanity_analysis": _artifact(roll_analysis),
        "result": _artifact(result_path),
        "report_count": len(list(reports_root.glob("*/*.json"))) // 2,
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    _atomic_json(manifest_path, completed)
    audit_program(output_dir)
    return designation_path


def audit_program(output_dir: Path) -> Path:
    output_dir = output_dir.resolve()
    manifest = _read_json(output_dir / "program_manifest.json")
    designation = _read_json(output_dir / "c1_designation.json")
    result = _read_json(output_dir / "experiment53_result.json")
    summary = pl.read_parquet(output_dir / "cell_fold_variant_summary.parquet")
    liquidity = pl.read_parquet(output_dir / "liquidity_terciles.parquet")
    trades = pl.read_parquet(output_dir / "per_trade_economics.parquet")
    rotation = pl.read_parquet(output_dir / "rotation_table.parquet")
    extensions = pl.read_parquet(output_dir / "selection_extensions.parquet")
    reports = sorted((output_dir / "reports").glob("*/*.json"))
    standard_reports = [
        path for path in reports if not path.name.endswith("__readout.json")
    ]
    readouts = [path for path in reports if path.name.endswith("__readout.json")]
    if (
        manifest.get("status") != "completed"
        or manifest.get("official_validation_accessed") is not False
        or manifest.get("test_accessed") is not False
        or designation.get("official_validation_accessed") is not False
        or designation.get("test_accessed") is not False
        or result.get("official_validation_accessed") is not False
        or result.get("test_accessed") is not False
        or summary.height != 432
        or summary.filter(pl.col("variant") == "measured").height != 144
        or liquidity.height != 1296
        or trades.height != 1296
        or rotation.height != 144
        or extensions.is_empty()
        or extensions["selection_extended_count"].min() < 0
        or len(standard_reports) != 432
        or len(readouts) != 432
        or designation.get("c1_cell_id") not in set(summary["cell_id"])
    ):
        raise ValueError("Experiment 53 completion contract failed")
    for report in standard_reports:
        sidecar = report.with_suffix(report.suffix + ".sha256")
        digest, name = sidecar.read_text(encoding="utf-8").strip().split("  ")
        payload = _read_json(report)
        if (
            name != report.name
            or digest != _sha256(report)
            or payload.get("schema") != "B3_EXECUTION_BACKTEST_REPORT_V1"
            or not payload.get("daily")
        ):
            raise ValueError(f"Execution report audit failed: {report}")
    files = [
        path
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name != "final_audit.json"
    ]
    audit = {
        "schema": "EXPERIMENT53_FINAL_AUDIT_V1",
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
        "standard_report_count": len(standard_reports),
        "selection_extension_rows": extensions.height,
        "c1_cell_id": designation["c1_cell_id"],
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    path = output_dir / "final_audit.json"
    _atomic_json(path, audit)
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Experiment 53 feasible map")
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--experiment52-root", type=Path, required=True)
    freeze.add_argument("--preregistration", type=Path, required=True)
    freeze.add_argument("--output-dir", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--output-dir", type=Path, required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "freeze":
        path = freeze_program(
            experiment52_root=args.experiment52_root,
            preregistration=args.preregistration,
            output_dir=args.output_dir,
        )
    elif args.command == "run":
        path = run_program(output_dir=args.output_dir)
    else:
        path = audit_program(args.output_dir)
    print(path)


if __name__ == "__main__":
    main()
