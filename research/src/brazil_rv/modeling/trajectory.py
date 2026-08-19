from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .contract import (
    DIAGNOSTIC_EARLY_STOP_PATIENCE,
    DIAGNOSTIC_MIN_IC_IMPROVEMENT,
    MAX_EPOCHS,
)
from .engine import state_dict_to_cpu

EMA_KEYS = {
    0.98: "ema_098",
    0.99: "ema_099",
    0.995: "ema_0995",
}
ELIGIBLE_RULES = (
    "final_raw",
    "final_ema_098",
    "final_ema_099",
    "final_ema_0995",
    "last3_weight_average",
    "last5_weight_average",
    "tail3_prediction_average",
    "tail5_prediction_average",
)
DIAGNOSTIC_RULES = (
    "diagnostic_patience3_raw",
    "diagnostic_best_epoch_raw",
)


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        if decay not in EMA_KEYS:
            raise ValueError(f"Unsupported EMA decay: {decay}")
        self.decay = decay
        self.key = EMA_KEYS[decay]
        self.shadow = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, value in model.state_dict().items():
            shadow = self.shadow[name]
            if torch.is_floating_point(shadow):
                shadow.mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                shadow.copy_(value)

    def cpu_state_dict(self) -> dict[str, torch.Tensor]:
        return state_dict_to_cpu(self.shadow)


@contextmanager
def temporarily_load_state(
    model: nn.Module, state_dict: Mapping[str, torch.Tensor]
):
    original = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    model.load_state_dict(state_dict, strict=True)
    try:
        yield
    finally:
        model.load_state_dict(original, strict=True)


def average_state_dicts(
    states: Sequence[Mapping[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    if not states:
        raise ValueError("At least one state dictionary is required")
    names = tuple(states[0])
    if any(tuple(state) != names for state in states[1:]):
        raise ValueError("State dictionaries have different keys")
    averaged: dict[str, torch.Tensor] = {}
    for name in names:
        values = [state[name] for state in states]
        if torch.is_floating_point(values[0]):
            total = values[0].detach().to(dtype=torch.float64).clone()
            for value in values[1:]:
                total.add_(value.detach().to(dtype=torch.float64))
            averaged[name] = (total / len(values)).to(dtype=values[0].dtype)
        else:
            averaged[name] = values[-1].detach().clone()
    return averaged


def checkpoint_path(run_dir: Path, epoch: int) -> Path:
    if not 1 <= epoch <= MAX_EPOCHS:
        raise ValueError(f"Epoch is outside the trajectory: {epoch}")
    return run_dir / "checkpoints" / f"epoch_{epoch:02d}.pt"


def prediction_path(run_dir: Path, epoch: int) -> Path:
    if not 1 <= epoch <= MAX_EPOCHS:
        raise ValueError(f"Epoch is outside the trajectory: {epoch}")
    return run_dir / "validation_predictions" / f"epoch_{epoch:02d}.npz"


def load_checkpoint(run_dir: Path, epoch: int) -> dict[str, object]:
    checkpoint = torch.load(
        checkpoint_path(run_dir, epoch), map_location="cpu", weights_only=False
    )
    if checkpoint.get("epoch") != epoch:
        raise ValueError("Checkpoint epoch metadata differs from its filename")
    return checkpoint


def _tail_epochs(length: int) -> range:
    if length not in (3, 5):
        raise ValueError(f"Unsupported tail length: {length}")
    return range(MAX_EPOCHS - length + 1, MAX_EPOCHS + 1)


def _diagnostic_epoch(run_dir: Path, rule: str) -> int:
    diagnostics = json.loads(
        (run_dir / "trajectory_diagnostics.json").read_text(encoding="utf-8")
    )
    key = {
        "diagnostic_patience3_raw": "patience3",
        "diagnostic_best_epoch_raw": "retrospective_best_epoch",
    }[rule]
    value = diagnostics[key]
    return int(value["selected_epoch"] if isinstance(value, dict) else value)


def model_state_dicts_for_rule(
    run_dir: Path, rule: str
) -> tuple[dict[str, torch.Tensor], ...]:
    if rule not in ELIGIBLE_RULES:
        raise ValueError(f"Rule is not eligible for frozen evaluation: {rule}")
    final = load_checkpoint(run_dir, MAX_EPOCHS)
    if rule == "final_raw":
        return (final["model_state_dict"],)
    if rule.startswith("final_ema_"):
        key = rule.removeprefix("final_")
        return (final["ema_state_dicts"][key],)
    if rule.endswith("_weight_average"):
        length = int(rule.removeprefix("last").split("_", 1)[0])
        states = [
            load_checkpoint(run_dir, epoch)["model_state_dict"]
            for epoch in _tail_epochs(length)
        ]
        return (average_state_dicts(states),)
    length = int(rule.removeprefix("tail").split("_", 1)[0])
    return tuple(
        load_checkpoint(run_dir, epoch)["model_state_dict"]
        for epoch in _tail_epochs(length)
    )


def predictions_for_rule(run_dir: Path, rule: str) -> np.ndarray:
    if rule in DIAGNOSTIC_RULES:
        epoch = _diagnostic_epoch(run_dir, rule)
        with np.load(prediction_path(run_dir, epoch), allow_pickle=False) as values:
            return values["raw"].copy()
    if rule not in ELIGIBLE_RULES:
        raise ValueError(f"Unknown trajectory rule: {rule}")
    if rule.startswith("final_"):
        key = rule.removeprefix("final_")
        with np.load(
            prediction_path(run_dir, MAX_EPOCHS), allow_pickle=False
        ) as values:
            return values[key].copy()
    with np.load(
        run_dir / "validation_predictions" / "tail_candidates.npz",
        allow_pickle=False,
    ) as values:
        return values[rule].copy()


def simulate_patience3(scores: Sequence[float]) -> dict[str, float | int]:
    if len(scores) != MAX_EPOCHS:
        raise ValueError(f"Expected {MAX_EPOCHS} raw validation scores")
    best_score = -float("inf")
    best_epoch = 0
    stale = 0
    stopped_epoch = MAX_EPOCHS
    for epoch, score in enumerate(scores, start=1):
        if score > best_score + DIAGNOSTIC_MIN_IC_IMPROVEMENT:
            best_score = float(score)
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= DIAGNOSTIC_EARLY_STOP_PATIENCE:
                stopped_epoch = epoch
                break
    return {
        "selected_epoch": best_epoch,
        "selected_score": best_score,
        "stopped_epoch": stopped_epoch,
        "selection_eligible": False,
    }


def retrospective_best_epoch(scores: Sequence[float]) -> int:
    if len(scores) != MAX_EPOCHS:
        raise ValueError(f"Expected {MAX_EPOCHS} raw validation scores")
    return int(np.nanargmax(np.asarray(scores, dtype=np.float64))) + 1


def load_frozen_selection(path: Path) -> dict[str, object]:
    selection = json.loads(path.read_text(encoding="utf-8"))
    if selection.get("schema") != "TRAJECTORY_SELECTION":
        raise ValueError("Selection file has an unknown schema")
    rule = selection.get("selected_rule")
    if rule not in ELIGIBLE_RULES:
        raise ValueError("Selection file does not contain an eligible rule")
    return selection
