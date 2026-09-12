"""Round-7 fixed experiment roster and pre-score protocol materialization."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import numpy as np
import torch

from .artifacts import write_json_atomic
from .characteristic_model import CharacteristicConfig, CharacteristicModel
from .config import ModelConfig
from .model import DailyMultiHorizonModel
from .round5_derived import bind
from .round7_data import NATIVE_FIELDS, PROJECT, registered_sources

SCREEN_FOLDS = ("F2", "F6", "F10", "F14")
SEEDS = (11, 29, 47)
CELLS = (
    {"cell": "A0", "graph": "s0", "recipe": "old", "inputs": "slow", "rho": 0.125},
    {"cell": "A1", "graph": "s0", "recipe": "R", "inputs": "slow", "rho": 0.125},
    {"cell": "A2", "graph": "s0", "recipe": "R", "inputs": "slow", "rho": 0.05},
    {"cell": "A3", "graph": "s0", "recipe": "R", "inputs": "all", "rho": 0.05},
    {"cell": "B1", "graph": "c1", "recipe": "R", "inputs": "slow", "rho": 0.125},
    {"cell": "B2", "graph": "c1", "recipe": "R", "inputs": "slow", "rho": 0.10},
    {"cell": "B3", "graph": "c1", "recipe": "R", "inputs": "slow", "rho": 0.05},
    {"cell": "B4", "graph": "c1", "recipe": "R", "inputs": "all", "rho": 0.05},
    {"cell": "B5", "graph": "c1", "recipe": "R", "inputs": "all", "rho": 0.125},
    {"cell": "B6", "graph": "c1_no_gru", "recipe": "R", "inputs": "all", "rho": 0.05},
    {
        "cell": "B7",
        "graph": "c1_five_heads",
        "recipe": "R",
        "inputs": "all",
        "rho": 0.05,
    },
    {
        "cell": "B8",
        "graph": "c1",
        "recipe": "R",
        "inputs": "all",
        "rho": 0.05,
        "loss": "pearson_on_ranks",
    },
    {
        "cell": "B9",
        "graph": "c1_attention",
        "recipe": "R",
        "inputs": "all",
        "rho": 0.05,
    },
    {"cell": "B10", "graph": "c1_tabm", "recipe": "R", "inputs": "all", "rho": 0.05},
    {"cell": "B11", "graph": "c1", "recipe": "R", "inputs": "all", "rho": None},
)


def cross_market_partition(names):
    common = tuple(
        n
        for n in names
        if n.startswith("shock_")
        or n
        in {
            "foreign_flow_1",
            "foreign_flow_5",
            "foreign_flow_month_reset",
            "foreign_flow_methodology_change",
            "ewz_minus_bova11_1",
        }
    )
    return common, tuple(n for n in names if n not in common)


def configuration(cell, names):
    families = {
        k.removeprefix("sidecar_"): tuple(v)
        for k, v in names.items()
        if k.startswith("sidecar_")
    }
    families["fundamentals_native"] = NATIVE_FIELDS
    if cell["inputs"] == "slow":
        families = {}
    if cell["graph"] == "s0":
        return ModelConfig(
            slow_feature_count=len(names["slow"]),
            current_feature_count=0,
            disable_fast_stream=True,
            use_bf16=True,
            sidecar_feature_counts=tuple(
                (n, len(v)) for n, v in sorted(families.items())
            ),
        )
    common = ()
    if families:
        common, families["cross_market"] = cross_market_partition(
            families["cross_market"]
        )
    return CharacteristicConfig(
        family_counts=tuple((n, len(v)) for n, v in sorted(families.items())),
        common_field_count=len(common) + (3 if common else 0),
        temporal=cell["graph"] != "c1_no_gru",
        context="attention" if cell["graph"] == "c1_attention" else "pool",
        members=8 if cell["graph"] == "c1_tabm" else 1,
        horizons=(1, 2, 3, 5, 10) if cell["graph"] == "c1_five_heads" else (3, 5, 10),
    )


def pretrain_key(cell):
    return cell["graph"] + "_" + cell["inputs"]


def calibrate_budget(selection_ic):
    """Fixed equal-fit, trailing-five mean rule; no evaluation score is consumed."""
    curve = np.asarray(selection_ic, dtype=np.float64)
    if curve.shape != (12, 60) or not np.isfinite(curve).all():
        raise ValueError(
            "calibration requires twelve complete 60-epoch selection curves"
        )
    mean = curve.mean(axis=0)
    trailing = np.convolve(mean, np.ones(5) / 5, mode="valid")
    maximum = trailing[15:].max()  # Epochs 20--60, inclusive.
    for budget in range(20, 61, 5):
        if trailing[budget - 5] >= maximum - 0.001:
            return budget
    return 60


def register():
    _, manifest, _, accepted = registered_sources()
    names = manifest["feature_names"]
    common, per_name = cross_market_partition(names["sidecar_cross_market"])
    cells = []
    for cell in CELLS:
        config = configuration(cell, names)
        # Graph construction only: no forward, training or data score.
        torch.manual_seed(11)
        model = (
            DailyMultiHorizonModel(config)
            if cell["graph"] == "s0"
            else CharacteristicModel(config)
        )
        cells.append(
            {
                **cell,
                "loss": cell.get("loss", "soft_spearman"),
                "pretrain_key": pretrain_key(cell),
                "config": asdict(config),
                "parameters": sum(p.numel() for p in model.parameters()),
            }
        )
    protocol = {
        "schema": "BRAZIL_RV_ROUND7_PROTOCOL_V1",
        "source_registration": bind(PROJECT / "research/preregistrations/v2_round7.md"),
        "implementation_resolutions": bind(
            PROJECT / "research/preregistrations/v2_round7_implementation.md"
        ),
        "base_store": accepted["store"],
        "repaired_store": None,
        "screen_folds": SCREEN_FOLDS,
        "seeds": SEEDS,
        "confirmation_seeds": [61, 79, 97],
        "cells": cells,
        "pretraining_keys": sorted({pretrain_key(c) for c in CELLS}),
        "pretraining_main_fits": 3 * len({pretrain_key(c) for c in CELLS}),
        "common_cross_market_fields": common,
        "per_name_cross_market_fields": per_name,
        "common_diagnostic_fields": names["common_state_diagnostic"],
        "recipe": {
            "budget_epochs": None,
            "calibration_epochs": 60,
            "calibration_rule": "earliest multiple of five >=20 whose equal-fit trailing-five-epoch mean selection IC is within .001 of its maximum over epochs 20--60; otherwise 60",
            "warmup_fraction": 0.05,
            "final_lr_fraction": 0.05,
            "learning_rate": 0.0003,
            "weight_decay": 0.01,
            "norm_bias_weight_decay": 0.0,
            "tail_average": "last ceil(B/4) epoch-end states, uniform",
            "dropout": 0.1,
            "unique_dates_per_batch": 16,
            "precision": "BF16 forward/backward autocast; FP32 parameters, optimizer, losses and moments",
            "lookback": 60,
        },
        "advancement": "A1 plus up to four non-anchor candidates, ranked by paired IC against A0 and within .002 of the leader; ties by cell order",
        "reuse": "A0 screen subset from anchor; advanced cells reuse exact screen fits; calibration reused as B4 only for B=60 and exact full contract identity",
        "forward_capture": False,
        "heldout_access": False,
        "experiment_scores_started": False,
    }
    path = PROJECT / "research/preregistrations/v2_round7.json"
    write_json_atomic(path, protocol)
    return protocol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["register"])
    parser.parse_args()
    protocol = register()
    print(
        json.dumps(
            {
                "pretraining_main_fits": protocol["pretraining_main_fits"],
                "common_field_count": len(protocol["common_cross_market_fields"]) + 3,
                "parameters": {c["cell"]: c["parameters"] for c in protocol["cells"]},
            }
        )
    )


if __name__ == "__main__":
    main()
