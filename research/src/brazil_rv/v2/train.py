from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler

from brazil_rv.modeling.contract import (
    ADAMW_BETAS,
    ADAMW_EPS,
    ADAMW_LR,
    ADAMW_WEIGHT_DECAY,
    EARLY_STOP_PATIENCE,
    GRADIENT_CLIP,
    MAX_EPOCHS,
    MIN_IC_IMPROVEMENT,
    SAM_RHO,
    WARMUP_FRACTION,
)
from brazil_rv.modeling.optim import learning_rate_factor
from brazil_rv.modeling.trajectory import ModelEMA

from .artifacts import sha256_file, write_json_atomic
from .config import ModelConfig
from .contract import (
    CHECKPOINT_INPUT_SCHEMA,
    DECISION_FEATURE_CONTRACT,
    DECISION_FEATURE_ALIGNMENT,
    DECISION_SAMPLE_SCHEMA,
    DEVELOPMENT_END,
    DEVELOPMENT_FOLDS,
    DEFAULT_HORIZON_LOSS_WEIGHTS,
    HORIZONS,
    PRIMARY_HORIZONS,
    TRADED_PRIMARY_HORIZONS,
    FINAL_EMA_SCHEMA,
    FEATURE_AGE_CONTRACT,
    FINETUNE_START,
    MODEL_INPUT_SCHEMA,
    PRETRAIN_END,
    RAW_PATIENCE_SCHEMA,
    SOFT_RANK_TEMPERATURE,
    STORE_START,
    TRAINING_STAGE_SCHEMA,
    V1_READ_SEEDS,
)
from .data import V2DailyDataset, collate_v2_daily, stage_fast_name_count
from .losses import multi_horizon_loss, multi_horizon_loss_normalizers
from .model import DailyMultiHorizonModel
from .normalization import average_ranks
from .round5_magnitude import FitClip
from .splits import development_folds


class DatePairBatchSampler(Sampler[list[int]]):
    """Yield dataset row positions in adjacent-session, full-date pairs.

    ``date_indices`` are the ordered global session identities represented by
    the dataset rows. ``session_indices`` may override them when those
    identities are not integer calendar positions. Sampling is uniform over
    adjacent-pair terminal dates (or uses the configured terminal-date decay),
    so an interior date can occur once as each side of two distinct pairs.
    """

    def __init__(
        self,
        date_indices: Sequence[int],
        *,
        pairs_per_batch: int = 8,
        seed: int = 29,
        session_indices: Sequence[int] | None = None,
        time_decay_half_life: float | None = None,
        drop_last: bool = False,
    ) -> None:
        indices = np.asarray(date_indices, dtype=np.int64)
        if indices.ndim != 1 or indices.size < 2:
            raise ValueError("date_indices must contain at least two dates")
        if np.unique(indices).size != indices.size:
            raise ValueError("date_indices must identify unique full cross-sections")
        if pairs_per_batch <= 0:
            raise ValueError("pairs_per_batch must be positive")
        sessions = (
            indices
            if session_indices is None
            else np.asarray(session_indices, dtype=np.int64)
        )
        if sessions.shape != indices.shape or np.any(np.diff(sessions) <= 0):
            raise ValueError("session_indices must be aligned and strictly increasing")
        if time_decay_half_life is not None and time_decay_half_life <= 0:
            raise ValueError("time_decay_half_life must be positive")
        self.date_indices = indices
        self.session_indices = sessions
        self.pairs_per_batch = pairs_per_batch
        self.seed = seed
        self.time_decay_half_life = time_decay_half_life
        self.drop_last = drop_last
        self.epoch = 0
        self.pair_starts = np.flatnonzero(np.diff(sessions) == 1)
        if not self.pair_starts.size:
            raise ValueError("no adjacent in-window session pairs are available")

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch

    def __len__(self) -> int:
        pair_count = self.pair_starts.size
        if self.drop_last:
            return pair_count // self.pairs_per_batch
        return math.ceil(pair_count / self.pairs_per_batch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        pair_starts = self.pair_starts.copy()
        if self.time_decay_half_life is None:
            rng.shuffle(pair_starts)
        else:
            ages = self.session_indices[-1] - self.session_indices[pair_starts + 1]
            weights = np.power(0.5, ages / self.time_decay_half_life)
            weights /= weights.sum()
            pair_starts = rng.choice(
                pair_starts, size=pair_starts.size, replace=True, p=weights
            )
        for start in range(0, pair_starts.size, self.pairs_per_batch):
            chosen = pair_starts[start : start + self.pairs_per_batch]
            if chosen.size < self.pairs_per_batch and self.drop_last:
                continue
            yield [int(position + offset) for position in chosen for offset in (0, 1)]


def pretrain_internal_split(
    date_indices: Sequence[int],
    *,
    selection_fraction: float = 0.10,
    embargo_sessions: int = 70,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split P dates into chronological fit, embargo, and final holdout axes."""

    indices = np.asarray(date_indices, dtype=np.int64)
    if indices.ndim != 1 or indices.size < 2 or np.any(np.diff(indices) != 1):
        raise ValueError("pretrain dates must be a contiguous session axis")
    if not 0.0 < selection_fraction < 1.0 or embargo_sessions <= 0:
        raise ValueError("pretrain holdout controls are invalid")
    selection_count = max(1, math.floor(indices.size * selection_fraction))
    selection_start = indices.size - selection_count
    fit_end = selection_start - embargo_sessions
    if fit_end < 2:
        raise ValueError("pretrain span is too short for fit, embargo, and holdout")
    return (
        indices[:fit_end].copy(),
        indices[fit_end:selection_start].copy(),
        indices[selection_start:].copy(),
    )


def reshape_date_pair_batch(values: torch.Tensor) -> torch.Tensor:
    if values.shape[0] % 2:
        raise ValueError("a date-pair batch must contain an even number of date rows")
    return values.reshape(values.shape[0] // 2, 2, *values.shape[1:])


def set_deterministic_seed(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _parameter_owners(model: nn.Module) -> dict[int, tuple[nn.Module, str]]:
    owners: dict[int, tuple[nn.Module, str]] = {}
    for module in model.modules():
        for attribute, parameter in module.named_parameters(recurse=False):
            owners[id(parameter)] = (module, attribute)
    return owners


def build_optimizer(
    model: nn.Module,
    *,
    pretrained_parameter_names: frozenset[str] | None = None,
    learning_rate: float = ADAMW_LR,
    pretrained_lr_multiplier: float = 0.3,
    weight_decay: float = ADAMW_WEIGHT_DECAY,
) -> torch.optim.AdamW:
    if pretrained_parameter_names is None:
        pretrained_parameter_names = frozenset(
            getattr(model, "pretrained_parameter_names", frozenset())
        )
    owners = _parameter_owners(model)
    named = dict(model.named_parameters())
    unknown = pretrained_parameter_names - named.keys()
    if unknown:
        raise ValueError(f"unknown pretrained parameters: {sorted(unknown)}")
    routed: dict[tuple[bool, bool], list[nn.Parameter]] = {
        (pretrained, decay): []
        for pretrained in (False, True)
        for decay in (False, True)
    }
    for name, parameter in named.items():
        if not parameter.requires_grad:
            continue
        module, attribute = owners[id(parameter)]
        decay = not (
            attribute == "bias"
            or isinstance(module, (nn.RMSNorm, nn.Embedding))
            or (module is model and attribute == "absent_state")
        )
        routed[(name in pretrained_parameter_names, decay)].append(parameter)
    groups = []
    for (pretrained, decay), parameters in routed.items():
        if not parameters:
            continue
        groups.append(
            {
                "params": parameters,
                "lr": learning_rate * (pretrained_lr_multiplier if pretrained else 1.0),
                "weight_decay": weight_decay if decay else 0.0,
                "pretrained": pretrained,
            }
        )
    return torch.optim.AdamW(groups, betas=ADAMW_BETAS, eps=ADAMW_EPS)


@dataclass(frozen=True)
class SAMStepResult:
    first_loss: float
    second_loss: float
    first_gradient_norm: float
    update_gradient_norm: float
    branch_gradient_norms: dict[str, float] | None = None


def _rng_state() -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return torch.get_rng_state(), cuda


def _restore_rng(state: tuple[torch.Tensor, list[torch.Tensor] | None]) -> None:
    torch.set_rng_state(state[0])
    if state[1] is not None:
        torch.cuda.set_rng_state_all(state[1])


def sam_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    closure: Callable[[], torch.Tensor],
    *,
    rho: float = SAM_RHO,
    gradient_clip: float = GRADIENT_CLIP,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ema: ModelEMA | None = None,
) -> SAMStepResult:
    return sam_accumulated_step(
        model,
        optimizer,
        (closure,),
        rho=rho,
        gradient_clip=gradient_clip,
        scheduler=scheduler,
        ema=ema,
    )


def sam_accumulated_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    closures: Sequence[Callable[[], torch.Tensor]],
    *,
    rho: float = SAM_RHO,
    gradient_clip: float = GRADIENT_CLIP,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ema: ModelEMA | None = None,
    record_branch_gradients: bool = False,
) -> SAMStepResult:
    """Apply one SAM update from loss contributions over one effective batch."""

    if rho <= 0:
        raise ValueError("rho must be positive")
    if not closures:
        raise ValueError("SAM requires at least one loss contribution")
    parameters = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    originals = tuple(parameter.detach().clone() for parameter in parameters)
    start_rng = _rng_state()
    optimizer.zero_grad(set_to_none=True)
    try:
        first_value = 0.0
        for closure in closures:
            first_loss = closure()
            if first_loss.numel() != 1 or not bool(torch.isfinite(first_loss.detach())):
                raise FloatingPointError("SAM first-pass training loss is non-finite")
            first_loss.backward()
            first_value += float(first_loss.detach())
        first_norm = torch.nn.utils.clip_grad_norm_(
            parameters, float("inf"), error_if_nonfinite=True
        )
        scale = rho / (first_norm + 1e-12)
        with torch.no_grad():
            for parameter in parameters:
                if parameter.grad is not None:
                    parameter.add_(parameter.grad * scale.to(parameter.dtype))
        optimizer.zero_grad(set_to_none=True)
        _restore_rng(start_rng)
        second_value = 0.0
        for closure in closures:
            second_loss = closure()
            if second_loss.numel() != 1 or not bool(
                torch.isfinite(second_loss.detach())
            ):
                raise FloatingPointError("SAM second-pass training loss is non-finite")
            second_loss.backward()
            second_value += float(second_loss.detach())
        with torch.no_grad():
            for parameter, original in zip(parameters, originals, strict=True):
                parameter.copy_(original)
        branch_norms = None
        if record_branch_gradients:
            from .branch_diagnostics import gradient_norms

            branch_norms = gradient_norms(model)
        update_norm = torch.nn.utils.clip_grad_norm_(
            parameters, gradient_clip, error_if_nonfinite=True
        )
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if ema is not None:
            ema.update(model)
        optimizer.zero_grad(set_to_none=True)
        return SAMStepResult(
            first_loss=first_value,
            second_loss=second_value,
            first_gradient_norm=float(first_norm.detach()),
            update_gradient_norm=float(update_norm.detach()),
            branch_gradient_norms=branch_norms,
        )
    except BaseException:
        with torch.no_grad():
            for parameter, original in zip(parameters, originals, strict=True):
                parameter.copy_(original)
        optimizer.zero_grad(set_to_none=True)
        raise


@dataclass
class PatienceTracker:
    patience: int = EARLY_STOP_PATIENCE
    maximum_epochs: int = MAX_EPOCHS
    minimum_improvement: float = MIN_IC_IMPROVEMENT
    best_score: float = -float("inf")
    selected_epoch: int = 0
    stopped_epoch: int | None = None
    stale_epochs: int = 0
    best_state_dict: dict[str, torch.Tensor] | None = field(default=None, repr=False)

    def update(self, epoch: int, score: float, model: nn.Module) -> bool:
        if not 1 <= epoch <= self.maximum_epochs:
            raise ValueError("epoch is outside the frozen training trajectory")
        if not math.isfinite(score):
            raise ValueError("selection score must be finite")
        if self.stopped_epoch is not None:
            raise RuntimeError("patience tracker was updated after stopping")
        if score > self.best_score + self.minimum_improvement:
            self.best_score = float(score)
            self.selected_epoch = epoch
            self.stale_epochs = 0
            self.best_state_dict = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        else:
            self.stale_epochs += 1
        should_stop = self.stale_epochs >= self.patience or epoch == self.maximum_epochs
        if should_stop:
            self.stopped_epoch = epoch
        return should_stop

    def restore(self, model: nn.Module) -> None:
        if self.best_state_dict is None:
            raise RuntimeError("no Patience state has been selected")
        model.load_state_dict(self.best_state_dict, strict=True)

    def metadata(self) -> dict[str, float | int | None]:
        return {
            "patience": self.patience,
            "maximum_epochs": self.maximum_epochs,
            "selected_epoch": self.selected_epoch,
            "selected_score": self.best_score,
            "stopped_epoch": self.stopped_epoch,
        }


def rank_average_ensemble(
    members: Sequence[np.ndarray], score_mask: np.ndarray
) -> np.ndarray:
    """Tie-aware per-date/head rank-average of ensemble members."""

    arrays = tuple(np.asarray(member) for member in members)
    if not arrays or arrays[0].ndim != 3:
        raise ValueError("ensemble members must have shape [date, name, head]")
    if any(member.shape != arrays[0].shape for member in arrays[1:]):
        raise ValueError("ensemble members must have identical shapes")
    date_count, name_count, head_count = arrays[0].shape
    mask = np.asarray(score_mask, dtype=bool)
    if mask.shape == (date_count, name_count):
        mask = np.repeat(mask[..., None], head_count, axis=-1)
    if mask.shape != arrays[0].shape:
        raise ValueError("score_mask is misaligned with ensemble members")
    result = np.zeros(arrays[0].shape, dtype=np.float32)
    for date in range(date_count):
        for head in range(head_count):
            valid = mask[date, :, head]
            if not valid.any():
                continue
            ranks = np.stack(
                [
                    average_ranks(member[date, valid, head].astype(np.float64))
                    for member in arrays
                ]
            )
            result[date, valid, head] = ranks.mean(axis=0)
    return result


def load_pretrain_handoff(
    model: DailyMultiHorizonModel,
    checkpoint: Path,
    *,
    expected_sha256: str | None = None,
    expected_seed: int | None = None,
    fine_tune_input_contract: Mapping[str, object] | None = None,
) -> frozenset[str]:
    """Load stage-P state while retaining the stage-F fast initialization."""

    if expected_sha256 is None:
        raise ValueError("stage-P handoff requires its expected SHA-256")
    actual_sha256 = sha256_file(checkpoint)
    if actual_sha256 != expected_sha256:
        raise ValueError("stage-P checkpoint SHA-256 differs from the frozen manifest")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != RAW_PATIENCE_SCHEMA
        or payload.get("stage") != "P"
        or (expected_seed is not None and payload.get("seed") != expected_seed)
        or not isinstance(payload.get("fold"), str)
        or not payload.get("fold")
    ):
        raise ValueError(
            "stage-P handoff schema, stage, optional seed, or fold is invalid"
        )
    if type(payload.get("transfer_chronology_clean")) is not bool:
        raise ValueError("stage-P handoff lacks explicit transfer chronology")
    contract = _verified_checkpoint_input_contract(payload)
    fine_config = model_config_contract(model.config)
    pretrain_config = model_config_contract(stage_p_model_config(model.config))
    if contract.get("model_config") != pretrain_config:
        raise ValueError("stage-P model contract differs from stage F")
    if fine_tune_input_contract is not None:
        if fine_tune_input_contract.get("model_config") != fine_config:
            raise ValueError("stage-P and stage-F model contracts differ")
        pretrain_inputs = contract.get("training")
        fine_inputs = fine_tune_input_contract.get("training")
        if not isinstance(pretrain_inputs, Mapping) or not isinstance(
            fine_inputs, Mapping
        ):
            raise ValueError("stage-P handoff input provenance is missing")
        identities = [
            _input_static_identity(pretrain_inputs),
            _input_static_identity(fine_inputs),
        ]
        # Each stage estimates clipping from its own fit dates. Its frozen
        # bounds remain part of scoring identity, not P/F structural identity.
        for identity in identities:
            features = identity.get("features")
            if isinstance(features, Mapping):
                identity["features"] = {
                    name: value
                    for name, value in features.items()
                    if name != "magnitude_clip"
                }
        if identities[0] != identities[1]:
            raise ValueError("stage-P and stage-F store/feature identities differ")
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("stage-P checkpoint has no model_state_dict")
    source = {
        str(name).removeprefix("_orig_mod."): value
        for name, value in state.items()
        if isinstance(value, torch.Tensor)
    }
    current = model.state_dict()
    transferred: dict[str, torch.Tensor] = {}
    initialized: set[str] = set()
    parameter_names = dict(model.named_parameters())
    for name, expected in current.items():
        if name.startswith("fast_encoder."):
            transferred[name] = expected
            continue
        if name not in source or source[name].shape != expected.shape:
            raise ValueError(f"stage-P checkpoint is incompatible at {name}")
        transferred[name] = source[name]
        if name in parameter_names:
            initialized.add(name)
    model.load_state_dict(transferred, strict=True)
    names = frozenset(initialized)
    model.pretrained_parameter_names |= names
    model.pretrain_checkpoint_sha256 = actual_sha256
    return names


def load_stage_checkpoint(
    model: DailyMultiHorizonModel,
    checkpoint: Path,
    *,
    expected_sha256: str | None = None,
    expected_model_config: ModelConfig | None = None,
) -> str:
    """Strictly load a complete raw-Patience or final-EMA stage archive."""

    actual_sha256 = sha256_file(checkpoint)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError("stage checkpoint SHA-256 differs from the manifest")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("schema") not in {
        RAW_PATIENCE_SCHEMA,
        FINAL_EMA_SCHEMA,
    }:
        raise ValueError("file is not a v2 stage checkpoint")
    contract = _verified_checkpoint_input_contract(payload)
    contract_config = (
        model.config if expected_model_config is None else expected_model_config
    )
    if contract.get("model_config") != model_config_contract(contract_config):
        raise ValueError("stage checkpoint model config differs from the model")
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("stage checkpoint has no model_state_dict")
    model.load_state_dict(state, strict=True)
    return actual_sha256


def _model_forward(model: nn.Module, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return model(
        batch["slow_features"],
        batch["slow_feature_mask"],
        batch["slow_history_mask"],
        batch["active_mask"],
        current_features=batch.get("current_features"),
        current_feature_mask=batch.get("current_feature_mask"),
        slow_feature_age_sessions=batch["slow_feature_age_sessions"],
        current_feature_age_sessions=batch.get("current_feature_age_sessions"),
        common_state_features=batch.get("common_state_features"),
        fast_patch_mask=batch.get("fast_patch_mask"),
        fast_present=batch.get("fast_present"),
        fast_state_position=batch.get("fast_state_position"),
        v1_equity_slow=batch.get("v1_equity_slow"),
        fast_patch_values=batch.get("fast_patch_values"),
        fast_patch_valid=batch.get("fast_patch_valid"),
        fast_name_index=batch.get("fast_name_index"),
        sidecars={
            key.removeprefix("sidecar_").removesuffix("_values"): (
                value,
                batch[key.removesuffix("_values") + "_valid"],
                batch[key.removesuffix("_values") + "_age_sessions"],
            )
            for key, value in batch.items()
            if key.startswith("sidecar_") and key.endswith("_values")
        },
    )


def _to_device(
    batch: Mapping[str, object],
    device: torch.device,
    *,
    omit_fast_stream: bool = False,
) -> dict[str, torch.Tensor]:
    names = {
        "slow_features",
        "slow_feature_mask",
        "slow_history_mask",
        "slow_feature_age_sessions",
        "active_mask",
        "current_features",
        "current_feature_mask",
        "current_feature_age_sessions",
        "common_state_features",
        "fast_patch_values",
        "fast_patch_valid",
        "fast_patch_mask",
        "fast_name_index",
        "fast_present",
        "fast_state_position",
        "v1_equity_slow",
        "targets",
        "target_mask",
        "to_close_target",
        "to_close_mask",
    }
    if omit_fast_stream:
        names -= {
            "fast_patch_values",
            "fast_patch_valid",
            "fast_patch_mask",
            "fast_name_index",
            "fast_state_position",
            "v1_equity_slow",
        }
    transferred = {
        name: value.to(device, non_blocking=device.type == "cuda")
        for name, value in batch.items()
        if (name in names or name.startswith("sidecar_"))
        and isinstance(value, torch.Tensor)
    }
    present = transferred.get("fast_present")
    if omit_fast_stream and present is not None:
        transferred["fast_present"] = torch.zeros_like(present)
    if not omit_fast_stream and present is not None and not torch.any(present.bool()):
        for name in (
            "fast_patch_values",
            "fast_patch_valid",
            "fast_patch_mask",
            "fast_name_index",
            "fast_state_position",
            "v1_equity_slow",
        ):
            transferred.pop(name, None)
    targets = transferred.get("targets")
    target_mask = transferred.get("target_mask")
    if targets is None or target_mask is None:
        raise ValueError("training batches require targets and target_mask")
    if targets.shape != target_mask.shape or targets.ndim != 3:
        raise ValueError("batched targets and target_mask are misaligned")
    if targets.shape[-1] != 5:
        raise ValueError("v2 primary targets must contain exactly five daily horizons")
    current = transferred.get("current_features")
    current_mask = transferred.get("current_feature_mask")
    current_age = transferred.get("current_feature_age_sessions")
    if any(value is not None for value in (current, current_mask, current_age)) and (
        current is None
        or current_mask is None
        or current_age is None
        or current.shape != current_mask.shape
        or current.shape != current_age.shape
    ):
        raise ValueError(
            "training batches require aligned current features, masks, and ages"
        )
    to_close = transferred.get("to_close_target")
    to_close_mask = transferred.get("to_close_mask")
    if (to_close is None) != (to_close_mask is None):
        raise ValueError("to-close target and mask must be provided together")
    if to_close is not None and to_close_mask is not None:
        if to_close.ndim == 2:
            transferred["to_close_target"] = to_close = to_close[..., None]
        if to_close_mask.ndim == 2:
            transferred["to_close_mask"] = to_close_mask = to_close_mask[..., None]
        if (
            to_close.shape != targets.shape[:-1] + (1,)
            or to_close_mask.shape != to_close.shape
        ):
            raise ValueError("to-close target and mask are misaligned")
    return transferred


def _date_pair_microbatches(
    batch: Mapping[str, object], pairs_per_microbatch: int
) -> tuple[dict[str, object], ...]:
    """Slice a collated effective batch without splitting an adjacent-date pair."""

    slow = batch.get("slow_features")
    if not isinstance(slow, torch.Tensor) or slow.ndim < 1 or slow.shape[0] % 2:
        raise ValueError("microbatching requires complete adjacent date pairs")
    pair_count = slow.shape[0] // 2
    if not 1 <= pairs_per_microbatch <= pair_count:
        raise ValueError("pairs_per_microbatch is outside the effective batch")
    step = 2 * pairs_per_microbatch
    output: list[dict[str, object]] = []
    for start in range(0, slow.shape[0], step):
        stop = min(start + step, slow.shape[0])
        piece: dict[str, object] = {}
        for name, value in batch.items():
            if (
                isinstance(value, torch.Tensor)
                and value.ndim > 0
                and value.shape[0] == slow.shape[0]
            ):
                piece[name] = value[start:stop]
            elif isinstance(value, list) and len(value) == slow.shape[0]:
                piece[name] = value[start:stop]
            else:
                piece[name] = value
        output.append(piece)
    return tuple(output)


def _common_primary_selection_score(
    predictions: np.ndarray,
    targets: np.ndarray,
    target_mask: np.ndarray,
    active_mask: np.ndarray,
    *,
    horizons: tuple[int, ...] = PRIMARY_HORIZONS,
) -> float:
    """Mean daily IC for the declared, already sliced heads on common support."""

    expected = predictions.shape
    if (
        predictions.ndim != 3
        or expected[-1] != len(horizons)
        or targets.shape != expected
        or target_mask.shape != expected
        or active_mask.shape != expected[:2]
    ):
        raise ValueError("selection arrays are misaligned")
    daily_primary: list[float] = []
    for date in range(predictions.shape[0]):
        common = (
            active_mask[date].astype(bool, copy=False)
            & target_mask[date].astype(bool, copy=False).all(axis=1)
            & np.isfinite(predictions[date]).all(axis=1)
            & np.isfinite(targets[date]).all(axis=1)
        )
        if common.sum() < 20:
            continue
        correlations: list[float] = []
        for head in range(len(horizons)):
            left = average_ranks(predictions[date, common, head].astype(np.float64))
            right = average_ranks(targets[date, common, head].astype(np.float64))
            left -= left.mean()
            right -= right.mean()
            denominator = np.sqrt(np.sum(left**2) * np.sum(right**2))
            if denominator <= 0:
                correlations = []
                break
            correlations.append(float(np.sum(left * right) / denominator))
        if len(correlations) == len(horizons):
            daily_primary.append(float(np.mean(correlations)))
    if not daily_primary:
        raise ValueError(
            "selection window lacks a common declared-head population of at least 20 names"
        )
    return float(np.mean(daily_primary))


def _selection_score(
    model: nn.Module,
    loader: Iterable[Mapping[str, object]],
    device: torch.device,
    *,
    stage: str,
    use_bf16: bool,
    disable_fast_stream: bool = False,
    selection_horizons: tuple[int, ...] = TRADED_PRIMARY_HORIZONS,
    slow_only: bool = False,
) -> float:
    model.eval()
    prediction_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    active_rows: list[np.ndarray] = []
    date_rows: list[np.ndarray | None] = []
    horizons = PRIMARY_HORIZONS if stage == "P" else selection_horizons
    head_indices = [HORIZONS.index(h) for h in horizons]
    with torch.no_grad():
        for cpu_batch in loader:
            _validate_stage_batch(
                stage, cpu_batch, require_date_pairs=False, slow_only=slow_only
            )
            batch = _to_device(
                cpu_batch, device, omit_fast_stream=stage == "P" or disable_fast_stream
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_bf16 and device.type == "cuda",
            ):
                predictions = _model_forward(model, batch)
            prediction_rows.append(predictions[..., head_indices].float().cpu().numpy())
            target_rows.append(
                batch["targets"][..., head_indices].float().cpu().numpy()
            )
            mask_rows.append(
                batch["target_mask"][..., head_indices].bool().cpu().numpy()
            )
            active_rows.append(batch["active_mask"].bool().cpu().numpy())
            date_index = cpu_batch.get("date_index")
            if date_index is None:
                date_rows.append(None)
            elif isinstance(date_index, torch.Tensor) and date_index.ndim == 1:
                date_rows.append(date_index.detach().cpu().numpy().astype(np.int64))
            else:
                raise ValueError(
                    "selection date_index must be a one-dimensional tensor"
                )
    if not prediction_rows:
        raise ValueError("selection loader produced no dates")
    predictions = np.concatenate(prediction_rows)
    targets = np.concatenate(target_rows)
    mask = np.concatenate(mask_rows)
    active = np.concatenate(active_rows)
    if any(row is not None for row in date_rows):
        if any(row is None for row in date_rows):
            raise ValueError("selection batches must consistently provide date_index")
        date_indices = np.concatenate([row for row in date_rows if row is not None])
        if date_indices.shape != (predictions.shape[0],):
            raise ValueError("selection date_index is misaligned with model rows")
        if np.unique(date_indices).size != date_indices.size:
            raise ValueError("selection loader must emit each date exactly once")
        order = np.argsort(date_indices, kind="stable")
        date_indices = date_indices[order]
        if np.any(np.diff(date_indices) != 1):
            raise ValueError("selection dates must form one contiguous session axis")
        predictions = predictions[order]
        targets = targets[order]
        mask = mask[order]
        active = active[order]
    return _common_primary_selection_score(
        predictions, targets, mask, active, horizons=horizons
    )


def _configure_inductor_compiler() -> None:
    if platform.machine() != "aarch64":
        return
    compiler = shutil.which("g++-12")
    if compiler is None:
        raise RuntimeError("Arm64 Inductor requires g++-12 for the GH200 Armv9 target")
    torch._inductor.config.cpp.cxx = (compiler,)


def compile_forward(
    model: nn.Module,
    *,
    backend: str = "inductor",
    mode: str | None = "max-autotune",
) -> nn.Module:
    # PyTorch requires the RNN opt-in before Dynamo will capture nn.GRU.
    torch._dynamo.config.allow_rnn = True
    if backend == "inductor":
        _configure_inductor_compiler()
        # The sparse fast path has a data-dependent present-name count.  CUDA
        # graph capture can reuse a stale dynamic buffer when the whole GRU +
        # sparse TCN graph is composed, even though every compiled subgraph is
        # finite.  Let Inductor compile that graph but skip CUDA graph capture
        # whenever it detects the dynamic shape.
        torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
    options: dict[str, object] = {
        "backend": backend,
        "fullgraph": True,
        # Training microbatches and chronological selection batches have
        # different leading widths.  Keep that date axis symbolic while the
        # dataset's fixed stage-wide name padding prevents name-axis
        # recompilation.
        "dynamic": True,
    }
    if mode is not None:
        options["mode"] = mode
    return torch.compile(model, **options)


def _unique_compiled_graphs() -> int:
    """Return Dynamo's process-local count of successfully compiled graphs."""

    return int(torch._dynamo.utils.counters["stats"]["unique_graphs"])


def _atomic_torch_save(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _set_loader_epoch(loader: Iterable[Mapping[str, object]], epoch: int) -> None:
    seen: set[int] = set()
    for candidate in (
        loader,
        getattr(loader, "batch_sampler", None),
        getattr(loader, "sampler", None),
    ):
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        setter = getattr(candidate, "set_epoch", None)
        if callable(setter):
            setter(epoch)


def _require_production_pair_sampler(
    loader: Iterable[Mapping[str, object]],
    *,
    time_decay_half_life: float | None = None,
) -> None:
    sampler = getattr(loader, "batch_sampler", None)
    if (
        not isinstance(sampler, DatePairBatchSampler)
        or sampler.pairs_per_batch != 8
        or sampler.drop_last is not True
        or sampler.time_decay_half_life != time_decay_half_life
    ):
        raise ValueError(
            "production training requires DatePairBatchSampler with "
            "exactly 8 pairs and drop_last=True"
        )


def _loader_access_payload(
    loader: Iterable[Mapping[str, object]],
) -> dict[str, object] | None:
    candidate: object | None = loader
    seen: set[int] = set()
    while candidate is not None and id(candidate) not in seen:
        seen.add(id(candidate))
        ledger = getattr(candidate, "access_ledger", None)
        if ledger is not None:
            payload = getattr(ledger, "payload", None)
            if not callable(payload):
                raise TypeError("loader access_ledger must expose payload()")
            result = payload()
            if not isinstance(result, dict):
                raise TypeError("loader access ledger payload must be a dictionary")
            return result
        candidate = getattr(candidate, "dataset", None)
    return None


def _loader_input_payload(
    loader: Iterable[Mapping[str, object]],
) -> dict[str, object] | None:
    candidate: object | None = loader
    seen: set[int] = set()
    while candidate is not None and id(candidate) not in seen:
        seen.add(id(candidate))
        store = getattr(candidate, "store", None)
        date_indices = getattr(candidate, "date_indices", None)
        root = getattr(store, "root", None)
        if root is not None and date_indices is not None:
            store_root = Path(root).resolve()
            indices = np.asarray(date_indices, dtype="<i8")
            if indices.ndim != 1 or not indices.size:
                raise ValueError("loader date_indices must be a nonempty vector")
            manifest = getattr(store, "manifest", None)
            dates = getattr(store, "dates", None)
            if not isinstance(manifest, Mapping) or dates is None:
                raise ValueError("loader store must expose its manifest and date axis")
            if np.any(indices < 0) or np.any(indices >= len(dates)):
                raise ValueError("loader date indices are outside the store axis")
            if np.any(np.diff(indices) <= 0):
                raise ValueError("loader dates must be strictly ordered and unique")
            feature_names = manifest.get("feature_names")
            if not isinstance(feature_names, Mapping):
                raise ValueError("store manifest lacks ordered feature names")
            feature_schema_sha256 = manifest.get("feature_schema_sha256")
            if (
                not isinstance(feature_schema_sha256, str)
                or len(feature_schema_sha256) != 64
            ):
                raise ValueError("store manifest lacks its feature-schema SHA-256")
            enabled_sidecars = tuple(
                str(value) for value in getattr(candidate, "enabled_sidecars", ())
            )
            slow_names = list(feature_names.get("slow", ()))
            sidecar_names: dict[str, list[str]] = {}
            for group in enabled_sidecars:
                values = feature_names.get(f"sidecar_{group}")
                if not isinstance(values, list) or not all(
                    isinstance(value, str) and value for value in values
                ):
                    raise ValueError(f"store lacks ordered sidecar names for {group}")
                sidecar_names[group] = list(values)
            if not slow_names or not all(
                isinstance(value, str) and value for value in slow_names
            ):
                raise ValueError("store slow feature names are malformed")
            intraday_names = list(feature_names.get("intraday", ()))
            if not getattr(candidate, "include_intraday", True):
                intraday_names = []
            if not all(isinstance(value, str) and value for value in intraday_names):
                raise ValueError("store intraday feature names are malformed")
            native_fast_names = list(feature_names.get("native_fast", ()))
            if not getattr(candidate, "include_fast", True):
                native_fast_names = []
            common_names = (
                list(feature_names.get("common_state_diagnostic", ()))
                if getattr(candidate, "include_common_state", False)
                else []
            )
            if not all(isinstance(value, str) and value for value in native_fast_names):
                raise ValueError("store native-fast feature names are malformed")
            selected_dates = np.asarray(dates[indices], dtype="datetime64[D]")
            date_strings = [str(value) for value in selected_dates]
            target_indices = np.asarray(
                getattr(candidate, "target_window_indices", indices), dtype="<i8"
            )
            if (
                target_indices.ndim != 1
                or not target_indices.size
                or np.any(np.diff(target_indices) <= 0)
            ):
                raise ValueError("loader target window must be chronological")
            target_dates = np.asarray(dates[target_indices], dtype="datetime64[D]")
            stage = str(getattr(candidate, "stage", ""))
            segments = _model_input_segments(
                np.asarray(dates, dtype="datetime64[D]"), indices, stage
            )
            entry_alignment = (
                segments[0]["entry_alignment"]
                if len({segment["entry_alignment"] for segment in segments}) == 1
                else "per_segment"
            )
            metadata = manifest.get("metadata", {})
            if not isinstance(metadata, Mapping):
                raise ValueError("store manifest metadata is malformed")
            age_contract = metadata.get("feature_age_contract")
            if age_contract != FEATURE_AGE_CONTRACT:
                raise ValueError("store feature-age contract is not canonical")
            decision_contract = metadata.get("slow_entry_alignment")
            if decision_contract != DECISION_FEATURE_CONTRACT:
                raise ValueError("store decision-feature alignment is not canonical")
            action_terms_source = metadata.get("action_terms_source")
            schedule_source = metadata.get("schedule_source")
            if not isinstance(action_terms_source, str) or not action_terms_source:
                raise ValueError("store lacks its corporate-action source tier")
            if not isinstance(schedule_source, str) or not schedule_source:
                raise ValueError("store lacks its session-schedule source tier")
            fast_identity = {
                "native_fast": metadata.get("native_fast"),
                "native_fast_security_mapping": (manifest.get("tables", {}) or {}).get(
                    "native_fast_security_mapping"
                ),
            }
            external_resolutions = [
                resolution.payload()
                for resolution in getattr(
                    candidate, "external_artifact_resolutions", ()
                )
            ]
            magnitude_provenance = {}
            if "magnitudes" in enabled_sidecars:
                clip = getattr(candidate, "magnitude_clip", None)
                if not isinstance(clip, FitClip):
                    raise ValueError(
                        "magnitude inputs require frozen fit clipping bounds"
                    )
                if clip.lower.size != len(sidecar_names["magnitudes"]):
                    raise ValueError(
                        "magnitude clipping width differs from store fields"
                    )
                magnitude_provenance["magnitude_clip"] = clip.payload()
            return {
                "schema": MODEL_INPUT_SCHEMA,
                "store": {
                    "schema": manifest.get("schema"),
                    "manifest_sha256": sha256_file(store_root / "manifest.json"),
                    "feature_schema_sha256": feature_schema_sha256,
                    "axes": manifest.get("axes"),
                    "fast_identity": fast_identity,
                    "external_artifact_resolutions": external_resolutions,
                    "action_terms_source": action_terms_source,
                    "schedule_source": schedule_source,
                },
                "features": {
                    "decision_sample_schema": DECISION_SAMPLE_SCHEMA,
                    "decision_feature_contract": dict(DECISION_FEATURE_CONTRACT),
                    "feature_age_contract": dict(FEATURE_AGE_CONTRACT),
                    "ordered_slow_names": slow_names,
                    "sidecar_encoding": "masked_zero_initialized_residual_projection",
                    "enabled_sidecar_groups": list(enabled_sidecars),
                    "ordered_sidecar_names": sidecar_names,
                    **magnitude_provenance,
                    "ordered_intraday_names": intraday_names,
                    "ordered_native_fast_names": native_fast_names,
                    "ordered_common_state_names": common_names,
                    "common_state_transform": "stored_causal_values_invalid_zeroed"
                    if common_names
                    else None,
                },
                "target": {
                    "value_array": str(getattr(candidate, "primary_target_name", "")),
                    "validity_array": str(
                        getattr(candidate, "primary_target_mask_name", "")
                    ),
                },
                "lookback_sessions": int(getattr(candidate, "lookback", 0)),
                "entry_alignment": entry_alignment,
                "segments": segments,
                "canonical_splits": _canonical_split_payload(dates),
                "dates": {
                    "indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
                    "identity_sha256": hashlib.sha256(
                        json.dumps(date_strings, separators=(",", ":")).encode()
                    ).hexdigest(),
                    "count": int(indices.size),
                    "first_index": int(indices[0]),
                    "last_index": int(indices[-1]),
                    "first_date": date_strings[0],
                    "last_date": date_strings[-1],
                },
                "target_window": {
                    "indices_sha256": hashlib.sha256(
                        target_indices.tobytes()
                    ).hexdigest(),
                    "identity_sha256": hashlib.sha256(
                        json.dumps(
                            [str(value) for value in target_dates],
                            separators=(",", ":"),
                        ).encode()
                    ).hexdigest(),
                    "count": int(target_indices.size),
                    "first_index": int(target_indices[0]),
                    "last_index": int(target_indices[-1]),
                    "first_date": str(target_dates[0]),
                    "last_date": str(target_dates[-1]),
                },
            }
        candidate = getattr(candidate, "dataset", None)
    return None


def _model_input_segments(
    axis: np.ndarray, indices: np.ndarray, stage: str
) -> list[dict[str, object]]:
    breakpoints = np.flatnonzero(np.diff(indices) != 1) + 1
    index_segments = np.split(indices, breakpoints)
    if stage != "joint" and len(index_segments) != 1:
        raise ValueError("non-joint loader dates must form one contiguous axis")
    if stage == "joint" and len(index_segments) > 2:
        raise ValueError("joint loader dates may contain only P and F segments")
    result: list[dict[str, object]] = []
    for values in index_segments:
        selected = np.asarray(axis[values], dtype="datetime64[D]")
        pretrain = selected <= np.datetime64(PRETRAIN_END)
        finetune = selected >= np.datetime64(FINETUNE_START)
        if pretrain.all():
            name, alignment = "P", DECISION_FEATURE_ALIGNMENT
        elif finetune.all():
            name, alignment = "F", DECISION_FEATURE_ALIGNMENT
        else:
            raise ValueError("loader segment crosses an unauthorized P/F boundary")
        date_strings = [str(value) for value in selected]
        little_endian = np.asarray(values, dtype="<i8")
        result.append(
            {
                "name": name,
                "entry_alignment": alignment,
                "indices_sha256": hashlib.sha256(little_endian.tobytes()).hexdigest(),
                "identity_sha256": hashlib.sha256(
                    json.dumps(date_strings, separators=(",", ":")).encode()
                ).hexdigest(),
                "count": int(values.size),
                "first_index": int(values[0]),
                "last_index": int(values[-1]),
                "first_date": date_strings[0],
                "last_date": date_strings[-1],
            }
        )
    names = [str(segment["name"]) for segment in result]
    if stage == "joint" and names not in (["P"], ["F"], ["P", "F"]):
        raise ValueError("joint loader segments must be ordered P then F")
    return result


def _date_span_payload(dates: np.ndarray, indices: np.ndarray) -> dict[str, object]:
    values = np.asarray(indices, dtype=np.int64)
    if values.ndim != 1 or not values.size:
        raise ValueError("canonical split dates must be nonempty")
    selected = np.asarray(dates[values], dtype="datetime64[D]")
    return {
        "first_index": int(values[0]),
        "last_index": int(values[-1]),
        "first_date": str(selected[0]),
        "last_date": str(selected[-1]),
        "count": int(values.size),
    }


def _canonical_split_payload(dates: object) -> dict[str, object]:
    """Derive the registered P/F boundaries from the store's date axis."""

    axis = np.asarray(dates, dtype="datetime64[D]")
    result: dict[str, object] = {}
    pretrain = np.flatnonzero(
        (axis >= np.datetime64(STORE_START)) & (axis <= np.datetime64(PRETRAIN_END))
    ).astype(np.int64)
    if (
        pretrain.size
        and axis[pretrain[0]] == np.datetime64(STORE_START)
        and axis[pretrain[-1]] == np.datetime64(PRETRAIN_END)
    ):
        fit, _, selection = pretrain_internal_split(pretrain)
        result["P"] = {
            "fit": _date_span_payload(axis, fit),
            "selection": _date_span_payload(axis, selection),
            "embargo_sessions": 70,
            "selection_fraction": 0.10,
        }
    python_dates = axis.astype(object).tolist()
    try:
        folds = development_folds(python_dates)
    except ValueError:
        folds = ()
    positions = {value: index for index, value in enumerate(python_dates)}
    for fold in folds:
        fit_indices = np.asarray(
            [positions[value] for value in fold.fit_dates], dtype=np.int64
        )
        selection_indices = np.asarray(
            [positions[value] for value in fold.selection_dates], dtype=np.int64
        )
        purge_before_indices = np.asarray(
            [positions[value] for value in fold.purge_before_dates], dtype=np.int64
        )
        purge_after_indices = np.asarray(
            [positions[value] for value in fold.purge_after_dates], dtype=np.int64
        )
        evaluation_indices = np.asarray(
            [positions[value] for value in fold.evaluation_dates], dtype=np.int64
        )
        result[fold.name] = {
            "fit": _date_span_payload(axis, fit_indices),
            "purge_before": _date_span_payload(axis, purge_before_indices),
            "selection": _date_span_payload(axis, selection_indices),
            "purge_after": _date_span_payload(axis, purge_after_indices),
            "evaluation": _date_span_payload(axis, evaluation_indices),
            "label_intervals_sha256": fold.payload()["label_intervals_sha256"],
            "no_label_overlap_assertion": True,
        }
    return result


def _canonical_payload_sha256(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _repository_commit_if_available() -> str | None:
    try:
        value = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=Path(__file__).resolve().parents[4],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return value if len(value) == 40 else None


def model_config_contract(config: ModelConfig) -> dict[str, object]:
    """Return architecture/training controls without a machine-local init path."""

    payload = asdict(config)
    payload.pop("fast_pretrained_checkpoint")
    payload["horizon_loss_weights"] = list(config.horizon_loss_weights)
    payload["selection_horizons"] = list(config.selection_horizons)
    payload["sidecar_feature_counts"] = [
        list(item) for item in config.sidecar_feature_counts
    ]
    return payload


def stage_p_model_config(config: ModelConfig) -> ModelConfig:
    """Fine-only H/P/selection arms reuse the exact uniform parent P graph."""
    return replace(
        config,
        horizon_loss_weights=DEFAULT_HORIZON_LOSS_WEIGHTS,
        lambda_persistence=0.0,
        selection_horizons=TRADED_PRIMARY_HORIZONS,
        disable_fast_stream=config.current_feature_count == 0,
    )


def _magnitude_dataset(loader: object) -> V2DailyDataset | None:
    candidate: object | None = loader
    seen: set[int] = set()
    while candidate is not None and id(candidate) not in seen:
        seen.add(id(candidate))
        if isinstance(candidate, V2DailyDataset):
            return candidate if "magnitudes" in candidate.enabled_sidecars else None
        candidate = getattr(candidate, "dataset", None)
    return None


def _fit_magnitude_clip(train_loader: object, selection_loader: object) -> None:
    training = _magnitude_dataset(train_loader)
    selection = _magnitude_dataset(selection_loader)
    if training is None:
        return
    indices = training.date_indices
    # Read only authorized fit rows, excluding history, embargo and selection.
    fit = FitClip.fit(
        training.store.read("sidecar_magnitudes_values", indices),
        training.store.read("sidecar_magnitudes_valid", indices),
        training.store.read("active", indices),
        np.arange(indices.size),
    )
    training.magnitude_clip = FitClip(fit.lower, fit.upper, tuple(indices.tolist()))
    if selection is not None:
        selection.magnitude_clip = training.magnitude_clip


def build_checkpoint_input_contract(
    model_config: ModelConfig,
    train_loader: Iterable[Mapping[str, object]],
    selection_loader: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    _fit_magnitude_clip(train_loader, selection_loader)
    training = _loader_input_payload(train_loader)
    selection = _loader_input_payload(selection_loader)
    if training is None or selection is None:
        raise ValueError("checkpoint inputs require model-aware dataset loaders")
    result: dict[str, object] = {
        "schema": CHECKPOINT_INPUT_SCHEMA,
        "implementation_commit": _repository_commit_if_available(),
        "model_config": model_config_contract(model_config),
        "training": training,
        "selection": selection,
    }
    result["sha256"] = _canonical_payload_sha256(result)
    return result


def _verified_checkpoint_input_contract(
    payload: Mapping[str, object],
) -> dict[str, object]:
    raw = payload.get("input_contract")
    if not isinstance(raw, Mapping):
        raise ValueError("checkpoint lacks its canonical input contract")
    contract = dict(raw)
    recorded = contract.pop("sha256", None)
    if not isinstance(recorded, str) or recorded != _canonical_payload_sha256(contract):
        raise ValueError("checkpoint input contract hash mismatch")
    contract["sha256"] = recorded
    if contract.get("schema") != CHECKPOINT_INPUT_SCHEMA:
        raise ValueError("checkpoint input contract schema is not recognized")
    return contract


def _input_static_identity(payload: Mapping[str, object]) -> dict[str, object]:
    store = payload.get("store")
    if isinstance(store, Mapping):
        store_identity = dict(store)
        resolutions = store_identity.get("external_artifact_resolutions")
        if isinstance(resolutions, list):
            store_identity["external_artifact_resolutions"] = [
                {
                    key: resolution.get(key)
                    for key in ("recorded_path", "bytes", "sha256")
                }
                for resolution in resolutions
                if isinstance(resolution, Mapping)
            ]
    else:
        store_identity = store
    return {
        "store": store_identity,
        "features": payload.get("features"),
        "lookback_sessions": payload.get("lookback_sessions"),
    }


def _validate_tracked_stage_inputs(
    stage: str,
    fold: str,
    model_config: ModelConfig,
    training: Mapping[str, object],
    selection: Mapping[str, object],
) -> None:
    if _input_static_identity(training) != _input_static_identity(selection):
        raise ValueError("training and selection model inputs are not identical")
    features = training.get("features")
    if not isinstance(features, Mapping):
        raise ValueError("model input feature provenance is missing")
    ordered = features.get("ordered_slow_names")
    if not isinstance(ordered, list) or len(ordered) != model_config.slow_feature_count:
        raise ValueError("model slow width differs from ordered store feature names")
    sidecar_names = features.get("ordered_sidecar_names", {})
    if dict(model_config.sidecar_feature_counts) != {
        name: len(values) for name, values in sidecar_names.items()
    }:
        raise ValueError("model sidecar widths differ from ordered store feature names")
    current = features.get("ordered_intraday_names")
    if (
        not isinstance(current, list)
        or len(current) != model_config.current_feature_count
    ):
        raise ValueError("model current width differs from ordered store feature names")
    if (
        len(features.get("ordered_common_state_names", []))
        != model_config.common_state_feature_count
    ):
        raise ValueError("model common-state width differs from ordered store fields")
    if training.get("lookback_sessions") != model_config.slow_lookback:
        raise ValueError("model lookback differs from the input store contract")
    train_dates = training.get("dates")
    selection_dates = selection.get("dates")
    if not isinstance(train_dates, Mapping) or not isinstance(selection_dates, Mapping):
        raise ValueError("checkpoint input date provenance is missing")
    train_last = int(train_dates["last_index"])
    selection_first = int(selection_dates["first_index"])
    if train_last >= selection_first:
        raise ValueError("fit and selection dates must be ordered and disjoint")
    gap = selection_first - train_last - 1
    train_start_date = np.datetime64(str(train_dates["first_date"]))
    selection_start_date = np.datetime64(str(selection_dates["first_date"]))
    selection_end_date = np.datetime64(str(selection_dates["last_date"]))
    training_splits = training.get("canonical_splits")
    selection_splits = selection.get("canonical_splits")
    if not isinstance(training_splits, Mapping) or training_splits != selection_splits:
        raise ValueError("training and selection canonical split provenance differs")
    if stage == "P":
        if (
            training.get("entry_alignment") != DECISION_FEATURE_ALIGNMENT
            or selection.get("entry_alignment") != DECISION_FEATURE_ALIGNMENT
            or train_start_date < np.datetime64(STORE_START)
            or selection_end_date > np.datetime64(PRETRAIN_END)
        ):
            raise ValueError("stage P inputs are outside the canonical pretrain split")
        if gap != 70:
            raise ValueError("stage P requires its exact 70-session embargo")
        split_name = "P"
    elif stage in {"F", "J"}:
        if stage == "F":
            valid_alignment = (
                training.get("entry_alignment") == DECISION_FEATURE_ALIGNMENT
                and selection.get("entry_alignment") == DECISION_FEATURE_ALIGNMENT
            )
        else:
            valid_alignment = (
                training.get("entry_alignment") == DECISION_FEATURE_ALIGNMENT
                and selection.get("entry_alignment") == DECISION_FEATURE_ALIGNMENT
            )
        if (
            not valid_alignment
            or selection_start_date < np.datetime64(FINETUNE_START)
            or selection_end_date > np.datetime64(DEVELOPMENT_END)
        ):
            raise ValueError(
                f"stage {stage} inputs are outside the canonical P/F alignment"
            )
        if stage == "F" and train_start_date < np.datetime64(FINETUNE_START):
            raise ValueError("stage F inputs are outside the canonical fine-tune split")
        if gap != 10:
            raise ValueError(f"stage {stage} requires its first 10-session purge")
        matches = [
            name
            for name in DEVELOPMENT_FOLDS
            if fold == name or fold.startswith(f"{name}_")
        ]
        if len(matches) != 1:
            raise ValueError(
                f"stage {stage} fold must identify a canonical development fold"
            )
        split_name = matches[0]
    else:
        return
    expected = training_splits.get(split_name)
    if not isinstance(expected, Mapping):
        raise ValueError(f"store cannot prove the canonical {split_name} split")
    expected_fit = expected.get("fit")
    expected_selection = expected.get("selection")
    if not isinstance(expected_fit, Mapping) or not isinstance(
        expected_selection, Mapping
    ):
        raise ValueError("canonical split boundary provenance is malformed")
    if stage == "J":
        training_segments = training.get("segments")
        selection_segments = selection.get("segments")
        if (
            not isinstance(training_segments, list)
            or not isinstance(selection_segments, list)
            or any(not isinstance(item, Mapping) for item in training_segments)
            or any(not isinstance(item, Mapping) for item in selection_segments)
            or [item.get("name") for item in training_segments] != ["P", "F"]
            or [item.get("name") for item in selection_segments] != ["F"]
        ):
            raise ValueError("stage J requires ordered P/F training and F selection")
        pretrain_segment, fine_segment = training_segments
        selection_segment = selection_segments[0]
        assert isinstance(pretrain_segment, Mapping)
        assert isinstance(fine_segment, Mapping)
        assert isinstance(selection_segment, Mapping)
        pretrain_split = training_splits.get("P")
        if not isinstance(pretrain_split, Mapping):
            raise ValueError("stage J store cannot prove the canonical P window")
        pretrain_fit = pretrain_split.get("fit")
        pretrain_selection = pretrain_split.get("selection")
        if not isinstance(pretrain_fit, Mapping) or not isinstance(
            pretrain_selection, Mapping
        ):
            raise ValueError("stage J canonical P provenance is malformed")
        if (
            pretrain_segment.get("entry_alignment") != DECISION_FEATURE_ALIGNMENT
            or fine_segment.get("entry_alignment") != DECISION_FEATURE_ALIGNMENT
            or selection_segment.get("entry_alignment") != DECISION_FEATURE_ALIGNMENT
            or pretrain_segment.get("first_date") != str(STORE_START)
            or pretrain_segment.get("last_date") != str(PRETRAIN_END)
            or pretrain_segment.get("first_index") != pretrain_fit.get("first_index")
            or pretrain_segment.get("last_index")
            != pretrain_selection.get("last_index")
            or pretrain_segment.get("count")
            != int(pretrain_segment["last_index"])
            - int(pretrain_segment["first_index"])
            + 1
            or int(pretrain_segment["last_index"]) + 1
            >= int(fine_segment["first_index"])
        ):
            raise ValueError("stage J P/F segment boundary or alignment is invalid")
        for actual in (pretrain_segment, fine_segment, selection_segment):
            if any(
                not isinstance(actual.get(field), str) or len(str(actual[field])) != 64
                for field in ("indices_sha256", "identity_sha256")
            ):
                raise ValueError("stage J segment identity hashes are malformed")
        for actual, registered in (
            (fine_segment, expected_fit),
            (selection_segment, expected_selection),
        ):
            if any(
                actual.get(field) != registered.get(field)
                for field in (
                    "first_index",
                    "last_index",
                    "first_date",
                    "last_date",
                    "count",
                )
            ):
                raise ValueError(
                    f"stage J inputs differ from canonical {split_name} boundaries"
                )
        return
    train_first = int(train_dates["first_index"])
    selection_last = int(selection_dates["last_index"])
    if (
        train_first < int(expected_fit["first_index"])
        or train_last != int(expected_fit["last_index"])
        or selection_first != int(expected_selection["first_index"])
        or selection_last > int(expected_selection["last_index"])
    ):
        raise ValueError(
            f"fit/selection loaders differ from the canonical {split_name} boundaries"
        )


def _validate_stage_batch(
    stage: str,
    batch: Mapping[str, object],
    *,
    require_date_pairs: bool = True,
    expected_pairs: int | None = None,
    slow_only: bool = False,
) -> None:
    required = {
        "slow_features",
        "slow_feature_mask",
        "slow_history_mask",
        "slow_feature_age_sessions",
        "active_mask",
        "targets",
        "target_mask",
    }
    if not slow_only:
        required |= {
            "current_features",
            "current_feature_mask",
            "current_feature_age_sessions",
        }
    missing = required - batch.keys()
    if missing:
        raise ValueError(f"training batch is missing tensors: {sorted(missing)}")
    slow_features = batch["slow_features"]
    if not isinstance(slow_features, torch.Tensor):
        raise TypeError("collated training arrays must be tensors")
    slow_feature_mask = batch["slow_feature_mask"]
    slow_history_mask = batch["slow_history_mask"]
    slow_feature_age = batch["slow_feature_age_sessions"]
    if (
        not isinstance(slow_feature_mask, torch.Tensor)
        or slow_feature_mask.shape != slow_features.shape
        or not isinstance(slow_history_mask, torch.Tensor)
        or slow_history_mask.shape != slow_features.shape[:-1]
        or not isinstance(slow_feature_age, torch.Tensor)
        or slow_feature_age.shape != slow_features.shape
    ):
        raise ValueError("slow values, validity, and age tensors are misaligned")
    current_features = batch.get("current_features")
    current_feature_mask = batch.get("current_feature_mask")
    current_feature_age = batch.get("current_feature_age_sessions")
    if not slow_only and (
        not isinstance(current_features, torch.Tensor)
        or current_features.shape[:2] != slow_features.shape[:2]
        or not isinstance(current_feature_mask, torch.Tensor)
        or current_feature_mask.shape != current_features.shape
        or not isinstance(current_feature_age, torch.Tensor)
        or current_feature_age.shape != current_features.shape
    ):
        raise ValueError("current values, validity, and age tensors are misaligned")
    if require_date_pairs and slow_features.shape[0] % 2:
        raise ValueError("training batches must contain complete adjacent date pairs")
    if expected_pairs is not None and slow_features.shape[0] != 2 * expected_pairs:
        raise ValueError(
            f"training batches must contain exactly {expected_pairs} date pairs"
        )
    date_index = batch.get("date_index")
    if date_index is not None:
        if not isinstance(date_index, torch.Tensor) or date_index.shape != (
            slow_features.shape[0],
        ):
            raise ValueError("training date_index is misaligned with model rows")
        if require_date_pairs:
            pairs = date_index.reshape(-1, 2)
            if torch.any(pairs[:, 1] - pairs[:, 0] != 1):
                raise ValueError("training date pairs must be adjacent sessions")
    compact_names = (
        "fast_patch_values",
        "fast_patch_valid",
        "fast_patch_mask",
        "fast_name_index",
        "fast_state_position",
    )
    compact_present = {name for name in compact_names if name in batch}
    if compact_present and len(compact_present) != len(compact_names):
        raise ValueError("compact fast tensors must be supplied together")
    if compact_present:
        values = batch["fast_patch_values"]
        valid = batch["fast_patch_valid"]
        patch_mask = batch["fast_patch_mask"]
        name_index = batch["fast_name_index"]
        state_position = batch["fast_state_position"]
        if (
            not isinstance(values, torch.Tensor)
            or values.ndim != 4
            or not isinstance(valid, torch.Tensor)
            or valid.shape != values.shape
            or not isinstance(patch_mask, torch.Tensor)
            or patch_mask.shape != values.shape[:-1]
            or not isinstance(name_index, torch.Tensor)
            or name_index.shape != values.shape[:2]
            or not isinstance(state_position, torch.Tensor)
            or state_position.shape != values.shape[:2]
        ):
            raise ValueError("compact fast tensors are misaligned")
    if stage == "P":
        present = batch.get("fast_present")
        if present is None and compact_present:
            raise ValueError("stage P must explicitly mark every fast stream absent")
        if present is not None and (
            not isinstance(present, torch.Tensor) or torch.any(present.bool())
        ):
            raise ValueError("stage P cannot access a present fast stream")
        if compact_present and batch["fast_patch_values"].shape[1] != 0:
            raise ValueError("stage P compact fast allocation must have K=0")
    elif not slow_only:
        present = batch.get("fast_present")
        if not isinstance(present, torch.Tensor):
            raise ValueError("F/J batches require the fast-presence flag")
        if torch.any(present.bool()) and (
            "fast_patch_values" not in batch
            or "fast_patch_valid" not in batch
            or "fast_patch_mask" not in batch
            or "fast_name_index" not in batch
            or "fast_state_position" not in batch
        ):
            raise ValueError("present fast samples require compact values and metadata")
    to_close_mask = batch.get("to_close_mask")
    present = batch.get("fast_present")
    if isinstance(to_close_mask, torch.Tensor) and isinstance(present, torch.Tensor):
        if torch.any(to_close_mask.bool() & ~present.bool()):
            raise ValueError("to-close labels require a present fast stream")


@dataclass(frozen=True)
class StageTrainingResult:
    stage: str
    seed: int
    fold: str
    epochs_completed: int
    raw_patience_checkpoint: Path
    final_ema_checkpoint: Path
    history_path: Path
    manifest_path: Path
    selected_epoch: int
    stopped_epoch: int


def train_stage(
    *,
    stage: str,
    seed: int,
    fold: str,
    train_loader: Iterable[Mapping[str, object]],
    selection_loader: Iterable[Mapping[str, object]],
    output_dir: Path,
    model_config: ModelConfig,
    pretrain_checkpoint: Path | None = None,
    expected_pretrain_sha256: str | None = None,
    maximum_epochs: int = MAX_EPOCHS,
    patience: int = EARLY_STOP_PATIENCE,
    learning_rate: float = ADAMW_LR,
    sam_rho: float = SAM_RHO,
    device: torch.device | None = None,
    selection_parity: int | None = None,
    microbatch_pairs: int = 8,
    record_branch_diagnostics: bool = False,
) -> StageTrainingResult:
    """Run one frozen P/F/J trajectory and archive raw-Patience plus final EMA."""

    if stage not in {"P", "F", "J"}:
        raise ValueError("stage must be P, F, or J")
    if selection_parity is not None:
        raise ValueError("v2 rev3 research stages do not accept selection parity")
    if seed not in V1_READ_SEEDS:
        raise ValueError("seed differs from the accepted v1 read roster")
    if not fold:
        raise ValueError("fold must be nonempty")
    if not 1 <= maximum_epochs <= MAX_EPOCHS:
        raise ValueError("maximum_epochs must be between one and twenty")
    if not 1 <= microbatch_pairs <= 8:
        raise ValueError("microbatch_pairs must be between one and eight")
    if stage == "P" and pretrain_checkpoint is not None:
        raise ValueError("stage P cannot initialize itself from a pretrain checkpoint")
    if (
        model_config.fast_encoder_mode == "legacy_v1_contaminated"
        and not model_config.fast_pretrained
    ):
        raise ValueError(
            "training in contaminated legacy mode requires its declared v1 checkpoint"
        )
    expected_decay = 756.0 if stage == "J" else None
    if model_config.time_decay_half_life_sessions != expected_decay:
        raise ValueError(
            f"stage {stage} requires time_decay_half_life_sessions={expected_decay}"
        )
    access_ledgers = {
        "training": _loader_access_payload(train_loader),
        "selection": _loader_access_payload(selection_loader),
    }
    if any(payload is None for payload in access_ledgers.values()):
        raise ValueError(
            "production train and selection loaders must expose authorized access ledgers"
        )
    for purpose, payload in access_ledgers.items():
        if payload is not None and payload.get("purpose") != purpose:
            raise ValueError(f"{purpose} loader has the wrong access purpose")
    official_validation_accessed = any(
        bool(payload and payload.get("official_validation_accessed"))
        for payload in access_ledgers.values()
    )
    test_accessed = any(
        bool(payload and payload.get("test_accessed"))
        for payload in access_ledgers.values()
    )
    if test_accessed:
        raise ValueError("v2 training cannot access the sealed test window")
    if official_validation_accessed:
        raise ValueError("v2 training/selection cannot access official validation")
    checkpoint_input_contract = build_checkpoint_input_contract(
        model_config, train_loader, selection_loader
    )
    input_stores = {
        name: checkpoint_input_contract[name] for name in ("training", "selection")
    }
    training_inputs = checkpoint_input_contract["training"]
    selection_inputs = checkpoint_input_contract["selection"]
    assert isinstance(training_inputs, Mapping)
    assert isinstance(selection_inputs, Mapping)
    training_store = training_inputs.get("store")
    selection_store = selection_inputs.get("store")
    if not isinstance(training_store, Mapping) or not isinstance(
        selection_store, Mapping
    ):
        raise ValueError("training inputs lack canonical store identity")
    feature_schema_sha256 = training_store.get("feature_schema_sha256")
    if (
        not isinstance(feature_schema_sha256, str)
        or selection_store.get("feature_schema_sha256") != feature_schema_sha256
    ):
        raise ValueError("training and selection feature schemas differ")
    action_terms_source = training_store.get("action_terms_source")
    schedule_source = training_store.get("schedule_source")
    if (
        not isinstance(action_terms_source, str)
        or not isinstance(schedule_source, str)
        or selection_store.get("action_terms_source") != action_terms_source
        or selection_store.get("schedule_source") != schedule_source
    ):
        raise ValueError("training and selection source-tier labels differ")
    _validate_tracked_stage_inputs(
        stage, fold, model_config, training_inputs, selection_inputs
    )
    if (pretrain_checkpoint is None) != (expected_pretrain_sha256 is None):
        raise ValueError(
            "stage-P handoff checkpoint and expected SHA-256 must be set together"
        )
    _require_production_pair_sampler(
        train_loader,
        time_decay_half_life=model_config.time_decay_half_life_sessions,
    )
    set_deterministic_seed(seed)
    model = DailyMultiHorizonModel(model_config)
    pretrain_provenance: dict[str, object] | None = None
    transfer_chronology_clean = model_config.fast_encoder_mode == "native"
    if pretrain_checkpoint is not None:
        load_pretrain_handoff(
            model,
            pretrain_checkpoint,
            expected_sha256=expected_pretrain_sha256,
            expected_seed=None,
            fine_tune_input_contract=checkpoint_input_contract,
        )
        pretrain_payload = torch.load(
            pretrain_checkpoint, map_location="cpu", weights_only=True
        )
        assert isinstance(pretrain_payload, Mapping)
        pretrain_contract = _verified_checkpoint_input_contract(pretrain_payload)
        pretrain_transfer_clean = pretrain_payload.get("transfer_chronology_clean")
        if type(pretrain_transfer_clean) is not bool:
            raise ValueError("stage-P handoff lacks explicit transfer chronology")
        transfer_chronology_clean &= pretrain_transfer_clean
        pretrain_provenance = {
            "schema": pretrain_payload["schema"],
            "stage": pretrain_payload["stage"],
            "seed": pretrain_payload["seed"],
            "fold": pretrain_payload["fold"],
            "checkpoint_sha256": model.pretrain_checkpoint_sha256,
            "input_contract_sha256": pretrain_contract["sha256"],
            "transfer_chronology_clean": pretrain_transfer_clean,
        }
    owned_names = {
        "raw_patience.pt",
        "final_ema.pt",
        "history.json",
        "history.json.sha256",
        "run_manifest.json",
        "run_manifest.json.sha256",
    }
    if output_dir.exists():
        if not output_dir.is_dir() or any(
            (output_dir / name).exists() for name in owned_names
        ):
            raise FileExistsError(output_dir)
    else:
        output_dir.mkdir(parents=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = build_optimizer(
        model,
        learning_rate=learning_rate,
    )
    try:
        steps_per_epoch = len(train_loader)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError(
            "train_loader must expose its deterministic epoch length"
        ) from error
    if steps_per_epoch <= 0:
        raise ValueError("training loader produced no date pairs")
    total_steps = steps_per_epoch * maximum_epochs
    warmup_steps = max(1, math.floor(WARMUP_FRACTION * total_steps))

    def schedule(step: int) -> float:
        return learning_rate_factor(
            min(step + 1, total_steps), total_steps, warmup_steps
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    ema = ModelEMA(model, 0.995)
    tracker = PatienceTracker(patience=patience, maximum_epochs=maximum_epochs)
    compiled_graphs_before = _unique_compiled_graphs()
    forward_model = compile_forward(model) if model_config.compile_forward else model
    training_compiled_graph_count = 0
    selection_compiled_graph_count = 0
    history: list[dict[str, object]] = []
    for epoch in range(1, maximum_epochs + 1):
        _set_loader_epoch(train_loader, epoch - 1)
        model.train()
        losses: list[float] = []
        branch_norms: dict[str, list[float]] = {}
        for cpu_batch in train_loader:
            _validate_stage_batch(
                stage,
                cpu_batch,
                expected_pairs=8,
                slow_only=model_config.current_feature_count == 0,
            )
            full_target_mask = reshape_date_pair_batch(cpu_batch["target_mask"]).bool()
            if model_config.to_close_weight:
                full_to_close_mask = cpu_batch.get("to_close_mask")
                if not isinstance(full_to_close_mask, torch.Tensor):
                    raise ValueError(
                        "weighted to-close auxiliary requires its target and mask"
                    )
                if full_to_close_mask.ndim == 2:
                    full_to_close_mask = full_to_close_mask[..., None]
                full_target_mask = torch.cat(
                    (
                        full_target_mask,
                        reshape_date_pair_batch(full_to_close_mask).bool(),
                    ),
                    dim=-1,
                )
            full_active = reshape_date_pair_batch(cpu_batch["active_mask"]).bool()
            normalization_counts = {
                name: value.to(device, non_blocking=device.type == "cuda")
                for name, value in multi_horizon_loss_normalizers(
                    full_target_mask,
                    score_mask=(
                        full_active if model_config.lambda_persistence else None
                    ),
                ).items()
            }

            def make_closure(
                cpu_microbatch: Mapping[str, object],
            ) -> Callable[[], torch.Tensor]:
                def closure() -> torch.Tensor:
                    batch = _to_device(
                        cpu_microbatch,
                        device,
                        omit_fast_stream=stage == "P"
                        or model_config.disable_fast_stream,
                    )
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=model_config.use_bf16 and device.type == "cuda",
                    ):
                        flat_scores = _model_forward(forward_model, batch)
                        scores = reshape_date_pair_batch(flat_scores[..., :5])
                        targets = reshape_date_pair_batch(batch["targets"])
                        target_mask = reshape_date_pair_batch(batch["target_mask"])
                        if model_config.to_close_weight:
                            to_close = batch.get("to_close_target")
                            to_close_mask = batch.get("to_close_mask")
                            if to_close is None or to_close_mask is None:
                                raise ValueError(
                                    "weighted to-close auxiliary requires its target and mask"
                                )
                            scores = torch.cat(
                                (
                                    scores,
                                    reshape_date_pair_batch(flat_scores[..., 5:]),
                                ),
                                dim=-1,
                            )
                            targets = torch.cat(
                                (targets, reshape_date_pair_batch(to_close)), dim=-1
                            )
                            target_mask = torch.cat(
                                (
                                    target_mask,
                                    reshape_date_pair_batch(to_close_mask).bool(),
                                ),
                                dim=-1,
                            )
                        active = reshape_date_pair_batch(batch["active_mask"])
                        return multi_horizon_loss(
                            scores,
                            targets,
                            target_mask,
                            score_mask=active,
                            persistence_weight=model_config.lambda_persistence,
                            horizon_loss_weights=model_config.horizon_loss_weights,
                            temperature=model_config.soft_rank_temperature,
                            to_close_weight=model_config.to_close_weight,
                            normalization_counts=normalization_counts,
                        )

                return closure

            closures = tuple(
                make_closure(microbatch)
                for microbatch in _date_pair_microbatches(cpu_batch, microbatch_pairs)
            )
            compiled_before_update = _unique_compiled_graphs()
            update = sam_accumulated_step(
                model,
                optimizer,
                closures,
                rho=sam_rho,
                scheduler=scheduler,
                ema=ema,
                record_branch_gradients=record_branch_diagnostics,
            )
            training_compiled_graph_count += (
                _unique_compiled_graphs() - compiled_before_update
            )
            losses.append(update.first_loss)
            for name, norm in (update.branch_gradient_norms or {}).items():
                branch_norms.setdefault(name, []).append(norm)
        if not losses:
            raise ValueError("training loader produced no date pairs")
        compiled_before_selection = _unique_compiled_graphs()
        selection_score = _selection_score(
            forward_model,
            selection_loader,
            device,
            stage=stage,
            use_bf16=model_config.use_bf16,
            disable_fast_stream=model_config.disable_fast_stream,
            selection_horizons=model_config.selection_horizons,
            slow_only=model_config.current_feature_count == 0,
        )
        selection_compiled_graph_count += (
            _unique_compiled_graphs() - compiled_before_selection
        )
        history.append(
            {
                "epoch": epoch,
                "training_loss": float(np.mean(losses)),
                "selection_score": selection_score,
            }
        )
        if record_branch_diagnostics:
            history[-1]["branch_gradient_norms"] = {
                name: {
                    "mean": float(np.mean(values)),
                    "maximum": max(values),
                    "updates": len(values),
                }
                for name, values in branch_norms.items()
            }
            history[-1]["gradient_timing"] = "second_SAM_pass_before_global_clipping"
        if tracker.update(epoch, selection_score, model):
            break
    if tracker.best_state_dict is None or tracker.stopped_epoch is None:
        raise RuntimeError("training ended without a selected Patience state")
    compiled_graph_count = _unique_compiled_graphs() - compiled_graphs_before
    if compiled_graph_count != (
        training_compiled_graph_count + selection_compiled_graph_count
    ):
        raise RuntimeError("compiled graph attribution is incomplete")
    if model_config.compile_forward and compiled_graph_count < 1:
        raise RuntimeError("compiled training produced no Dynamo graph")
    if not model_config.compile_forward and compiled_graph_count != 0:
        raise RuntimeError("eager training unexpectedly compiled a Dynamo graph")
    raw_path = output_dir / "raw_patience.pt"
    ema_path = output_dir / "final_ema.pt"
    history_path = output_dir / "history.json"
    manifest_path = output_dir / "run_manifest.json"
    _atomic_torch_save(
        raw_path,
        {
            "schema": RAW_PATIENCE_SCHEMA,
            "stage": stage,
            "seed": seed,
            "fold": fold,
            "model_state_dict": tracker.best_state_dict,
            "patience": tracker.metadata(),
            "input_contract": checkpoint_input_contract,
            "fast_initialization_provenance": model.fast_initialization_provenance,
            "transfer_chronology_clean": transfer_chronology_clean,
            "feature_schema_sha256": feature_schema_sha256,
            "action_terms_source": action_terms_source,
            "schedule_source": schedule_source,
        },
    )
    _atomic_torch_save(
        ema_path,
        {
            "schema": FINAL_EMA_SCHEMA,
            "stage": stage,
            "seed": seed,
            "fold": fold,
            "model_state_dict": ema.cpu_state_dict(),
            "epoch": tracker.stopped_epoch,
            "input_contract": checkpoint_input_contract,
            "fast_initialization_provenance": model.fast_initialization_provenance,
            "transfer_chronology_clean": transfer_chronology_clean,
            "feature_schema_sha256": feature_schema_sha256,
            "action_terms_source": action_terms_source,
            "schedule_source": schedule_source,
        },
    )
    history_sha256 = write_json_atomic(history_path, history)
    model_config_payload = asdict(model_config)
    checkpoint = model_config_payload.get("fast_pretrained_checkpoint")
    model_config_payload["fast_pretrained_checkpoint"] = (
        None if checkpoint is None else str(Path(checkpoint).resolve())
    )
    write_json_atomic(
        manifest_path,
        {
            "schema": TRAINING_STAGE_SCHEMA,
            "status": "completed",
            "stage": stage,
            "seed": seed,
            "fold": fold,
            "epochs_completed": tracker.stopped_epoch,
            "selected_epoch": tracker.selected_epoch,
            "model_config": model_config_payload,
            "compiled_graph_count": compiled_graph_count,
            "compiled_graphs": {
                "training": training_compiled_graph_count,
                "selection": selection_compiled_graph_count,
                "total": compiled_graph_count,
            },
            "fast_checkpoint_sha256": model.fast_checkpoint_sha256,
            "fast_initialization_provenance": model.fast_initialization_provenance,
            "transfer_chronology_clean": transfer_chronology_clean,
            "feature_schema_sha256": feature_schema_sha256,
            "action_terms_source": action_terms_source,
            "schedule_source": schedule_source,
            "pretrain_checkpoint": (
                None
                if pretrain_checkpoint is None
                else str(pretrain_checkpoint.resolve())
            ),
            "pretrain_checkpoint_sha256": model.pretrain_checkpoint_sha256,
            "pretrain_checkpoint_provenance": pretrain_provenance,
            "checkpoint_input_contract": checkpoint_input_contract,
            "access_ledgers": access_ledgers,
            "input_stores": input_stores,
            "optimizer": {
                "name": "sam_adamw",
                "learning_rate": learning_rate,
                "pretrained_lr_multiplier": 0.3,
                "rho": sam_rho,
                "weight_decay": ADAMW_WEIGHT_DECAY,
                "pretrained_parameter_count": len(model.pretrained_parameter_names),
                "effective_batch_date_pairs": 8,
                "microbatch_date_pairs": microbatch_pairs,
            },
            "patience": tracker.metadata(),
            "artifacts": {
                raw_path.name: sha256_file(raw_path),
                ema_path.name: sha256_file(ema_path),
                history_path.name: history_sha256,
            },
            "official_validation_accessed": official_validation_accessed,
            "test_accessed": test_accessed,
        },
    )
    return StageTrainingResult(
        stage=stage,
        seed=seed,
        fold=fold,
        epochs_completed=tracker.stopped_epoch,
        raw_patience_checkpoint=raw_path,
        final_ema_checkpoint=ema_path,
        history_path=history_path,
        manifest_path=manifest_path,
        selected_epoch=tracker.selected_epoch,
        stopped_epoch=tracker.stopped_epoch,
    )


def _cli_stage_indices(
    store_root: Path, stage: str, fold: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dates = np.load(store_root / "date_index.npy", allow_pickle=False).astype(
        "datetime64[D]", copy=False
    )
    if stage == "P":
        pretrain = np.flatnonzero(
            (dates >= np.datetime64(STORE_START))
            & (dates <= np.datetime64(PRETRAIN_END))
        ).astype(np.int64)
        fit, _, selection = pretrain_internal_split(pretrain)
        return fit, selection, selection, fit
    python_dates = dates.astype(object).tolist()
    by_name = {item.name: item for item in development_folds(python_dates)}
    try:
        selected = by_name[fold]
    except KeyError as error:
        raise ValueError(f"unknown development fold: {fold}") from error
    positions = {value: index for index, value in enumerate(python_dates)}
    fit = np.asarray([positions[value] for value in selected.fit_dates], dtype=np.int64)
    selection = np.asarray(
        [positions[value] for value in selected.selection_dates], dtype=np.int64
    )
    fit_target_window = np.asarray(
        [
            positions[value]
            for value in (*selected.fit_dates, *selected.purge_before_dates)
        ],
        dtype=np.int64,
    )
    evaluation = np.asarray(
        [positions[value] for value in selected.evaluation_dates], dtype=np.int64
    )
    if stage == "J":
        pretrain = np.flatnonzero(
            (dates >= np.datetime64(STORE_START))
            & (dates <= np.datetime64(PRETRAIN_END))
        ).astype(np.int64)
        fit = np.concatenate((pretrain, fit))
        fit_target_window = np.concatenate((pretrain, fit_target_window))
    return fit, selection, evaluation, fit_target_window


def _cli_feature_counts(
    store_root: Path, sidecars: Sequence[str]
) -> tuple[int, tuple[tuple[str, int], ...]]:
    manifest = json.loads((store_root / "manifest.json").read_text(encoding="utf-8"))
    names = manifest.get("feature_names")
    if not isinstance(names, Mapping) or not isinstance(names.get("slow"), list):
        raise ValueError("store manifest lacks ordered slow feature names")
    counts = []
    for group in sorted(sidecars):
        values = names.get(f"sidecar_{group}")
        if not isinstance(values, list):
            raise ValueError(f"store lacks the requested sidecar group: {group}")
        counts.append((group, len(values)))
    return len(names["slow"]), tuple(counts)


def _cli_current_feature_count(store_root: Path) -> int:
    manifest = json.loads((store_root / "manifest.json").read_text(encoding="utf-8"))
    names = manifest.get("feature_names")
    if not isinstance(names, Mapping) or not isinstance(names.get("intraday"), list):
        raise ValueError("store manifest lacks ordered current feature names")
    count = len(names["intraday"])
    if count <= 0:
        raise ValueError("store current feature axis must be nonempty")
    return count


def _train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one canonical v2 trajectory")
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--score-output-dir", type=Path)
    parser.add_argument("--stage", choices=("P", "F", "J"), required=True)
    parser.add_argument("--fold", default="F1")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--maximum-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=EARLY_STOP_PATIENCE)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--pairs-per-batch", type=int, default=8)
    parser.add_argument("--microbatch-pairs", type=int, default=8)
    parser.add_argument("--selection-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sidecar", action="append", default=[])
    parser.add_argument("--disable-fast-stream", action="store_true")
    parser.add_argument("--slow-only", action="store_true")
    parser.add_argument("--common-state", action="store_true")
    parser.add_argument(
        "--horizon-loss-weights",
        type=float,
        nargs=5,
        default=DEFAULT_HORIZON_LOSS_WEIGHTS,
    )
    parser.add_argument(
        "--selection-horizons", type=int, nargs="+", default=TRADED_PRIMARY_HORIZONS
    )
    parser.add_argument("--record-branch-diagnostics", action="store_true")
    parser.add_argument("--lambda-persistence", type=float, default=0.0)
    parser.add_argument(
        "--to-close-weight", type=float, choices=(0.0, 0.2), default=0.0
    )
    parser.add_argument(
        "--soft-rank-temperature", type=float, default=SOFT_RANK_TEMPERATURE
    )
    parser.add_argument(
        "--use-bf16", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--compile-forward", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--fast-pretrained-checkpoint", type=Path)
    parser.add_argument("--fast-pretrained-sha256")
    parser.add_argument("--allow-contaminated-v1-initialization", action="store_true")
    parser.add_argument("--pretrain-checkpoint", type=Path)
    parser.add_argument("--pretrain-sha256")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from .data import V2DailyDataset

    arguments = _train_parser().parse_args(argv)
    store_root = arguments.store.resolve()
    stage = str(arguments.stage)
    fold = "pretrain_internal" if stage == "P" else str(arguments.fold)
    (
        fit_indices,
        selection_indices,
        evaluation_indices,
        fit_target_window,
    ) = _cli_stage_indices(store_root, stage, fold)
    dataset_stage = {"P": "pretrain", "F": "finetune", "J": "joint"}[stage]
    sidecars = tuple(dict.fromkeys(str(value) for value in arguments.sidecar))
    feature_options = dict(
        include_intraday=not arguments.slow_only,
        include_fast=not (arguments.disable_fast_stream or arguments.slow_only),
        include_common_state=arguments.common_state,
    )
    train_dataset = V2DailyDataset(
        store_root,
        fit_indices,
        stage=dataset_stage,
        lookback=arguments.lookback,
        enabled_sidecars=sidecars,
        **feature_options,
        purpose="training",
        target_window_indices=fit_target_window,
    )
    selection_dataset = V2DailyDataset(
        store_root,
        selection_indices,
        stage=dataset_stage,
        lookback=arguments.lookback,
        enabled_sidecars=sidecars,
        **feature_options,
        purpose="selection",
        target_window_indices=selection_indices,
    )
    decay = 756.0 if stage == "J" else None
    sampler = DatePairBatchSampler(
        train_dataset.date_indices,
        pairs_per_batch=arguments.pairs_per_batch,
        seed=arguments.seed,
        time_decay_half_life=decay,
        drop_last=True,
    )
    fixed_fast_name_count = (
        0 if stage == "P" else stage_fast_name_count(train_dataset, selection_dataset)
    )
    stage_collate = partial(
        collate_v2_daily, fixed_fast_name_count=fixed_fast_name_count
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=arguments.num_workers,
        collate_fn=stage_collate,
    )
    selection_loader = DataLoader(
        selection_dataset,
        batch_size=arguments.selection_batch_size,
        shuffle=False,
        num_workers=arguments.num_workers,
        collate_fn=stage_collate,
    )
    fast_checkpoint = arguments.fast_pretrained_checkpoint
    slow_count, sidecar_counts = _cli_feature_counts(store_root, sidecars)
    model_config = ModelConfig(
        slow_feature_count=slow_count,
        sidecar_feature_counts=sidecar_counts,
        current_feature_count=0
        if arguments.slow_only
        else _cli_current_feature_count(store_root),
        common_state_feature_count=3 if arguments.common_state else 0,
        slow_lookback=arguments.lookback,
        disable_fast_stream=arguments.disable_fast_stream or arguments.slow_only,
        fast_encoder_mode=(
            "legacy_v1_contaminated" if fast_checkpoint is not None else "native"
        ),
        fast_pretrained=fast_checkpoint is not None,
        fast_pretrained_checkpoint=fast_checkpoint,
        fast_pretrained_sha256=arguments.fast_pretrained_sha256,
        allow_contaminated_v1_initialization=(
            arguments.allow_contaminated_v1_initialization
        ),
        lambda_persistence=arguments.lambda_persistence,
        horizon_loss_weights=tuple(arguments.horizon_loss_weights),
        selection_horizons=tuple(arguments.selection_horizons),
        to_close_weight=arguments.to_close_weight,
        soft_rank_temperature=arguments.soft_rank_temperature,
        use_bf16=arguments.use_bf16,
        compile_forward=arguments.compile_forward,
        time_decay_half_life_sessions=decay,
    )
    if stage == "P":
        model_config = stage_p_model_config(model_config)
    device = None if arguments.device == "auto" else torch.device(arguments.device)
    result = train_stage(
        stage=stage,
        seed=arguments.seed,
        fold=fold,
        train_loader=train_loader,
        selection_loader=selection_loader,
        output_dir=arguments.output_dir,
        model_config=model_config,
        pretrain_checkpoint=arguments.pretrain_checkpoint,
        expected_pretrain_sha256=arguments.pretrain_sha256,
        maximum_epochs=arguments.maximum_epochs,
        patience=arguments.patience,
        device=device,
        microbatch_pairs=arguments.microbatch_pairs,
        record_branch_diagnostics=arguments.record_branch_diagnostics,
    )
    if arguments.score_output_dir is not None:
        from .score import score_checkpoint_artifact

        score_dataset = V2DailyDataset(
            store_root,
            evaluation_indices,
            stage="pretrain" if stage == "P" else "evaluation",
            lookback=arguments.lookback,
            enabled_sidecars=sidecars,
            **feature_options,
            purpose="evaluation",
        )
        score_loader = DataLoader(
            score_dataset,
            batch_size=arguments.selection_batch_size,
            shuffle=False,
            num_workers=arguments.num_workers,
            collate_fn=partial(
                collate_v2_daily,
                fixed_fast_name_count=(
                    0
                    if stage == "P"
                    else max(
                        fixed_fast_name_count,
                        stage_fast_name_count(score_dataset),
                    )
                ),
            ),
        )
        score_checkpoint_artifact(
            checkpoint=result.raw_patience_checkpoint,
            model_config=model_config,
            loader=score_loader,
            output_dir=arguments.score_output_dir,
            expected_checkpoint_sha256=sha256_file(result.raw_patience_checkpoint),
            device=device,
            record_branch_diagnostics=arguments.record_branch_diagnostics,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
