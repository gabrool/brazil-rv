"""Read-only path resolution for sealed Round-7 reports."""

from __future__ import annotations

import json

from .round7 import SCREEN_FOLDS, SEEDS


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def trajectory(root, cell, fold, seed):
    if (
        cell == "B4"
        and (root / "budget.json").exists()
        and read(root / "budget.json")["B"] == 60
        and fold in SCREEN_FOLDS
        and seed in SEEDS
    ):
        return root / "calibration" / f"{fold}_seed_{seed}"
    return root / "trajectories" / cell / f"{fold}_seed_{seed}"
