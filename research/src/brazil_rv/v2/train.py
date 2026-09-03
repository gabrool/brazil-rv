from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Sampler

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
    ALLOWED_SEEDS,
    DEVELOPMENT_END,
    FINETUNE_START,
    PRETRAIN_END,
    STORE_START,
)
from .losses import multi_horizon_loss
from .model import DailyMultiHorizonModel
from .normalization import average_ranks
from .splits import BLOCK_PARITY_SESSIONS, development_folds


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
    if rho <= 0:
        raise ValueError("rho must be positive")
    parameters = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    originals = tuple(parameter.detach().clone() for parameter in parameters)
    start_rng = _rng_state()
    optimizer.zero_grad(set_to_none=True)
    try:
        first_loss = closure()
        first_loss.backward()
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
        second_loss = closure()
        second_loss.backward()
        with torch.no_grad():
            for parameter, original in zip(parameters, originals, strict=True):
                parameter.copy_(original)
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
            first_loss=float(first_loss.detach()),
            second_loss=float(second_loss.detach()),
            first_gradient_norm=float(first_norm.detach()),
            update_gradient_norm=float(update_norm.detach()),
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


def block_parity_mask(
    date_count: int,
    parity: int,
    *,
    block_size: int = BLOCK_PARITY_SESSIONS,
) -> np.ndarray:
    if date_count <= 0 or block_size <= 0:
        raise ValueError("date_count and block_size must be positive")
    if parity not in (0, 1):
        raise ValueError("parity must be zero or one")
    return (np.arange(date_count) // block_size) % 2 == parity


def stitch_block_parity_predictions(
    selected_on_even: np.ndarray,
    selected_on_odd: np.ndarray,
) -> np.ndarray:
    """Stitch opposite-parity evaluations without shortening the date axis."""

    even_model = np.asarray(selected_on_even)
    odd_model = np.asarray(selected_on_odd)
    if even_model.shape != odd_model.shape or even_model.ndim < 1:
        raise ValueError("block-parity prediction arrays must have identical shapes")
    even_dates = block_parity_mask(even_model.shape[0], 0)
    result = np.empty_like(even_model)
    # The model selected on odd blocks evaluates even blocks, and vice versa.
    result[even_dates] = odd_model[even_dates]
    result[~even_dates] = even_model[~even_dates]
    return result


def rank_average_ensemble(
    members: Sequence[np.ndarray], score_mask: np.ndarray
) -> np.ndarray:
    """Tie-aware per-date/head rank-average of seed or parity members."""

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
    """Load stage-P learned state without replacing the separately bound v1 TCN."""

    if expected_sha256 is None or expected_seed is None:
        raise ValueError("stage-P handoff requires its expected SHA-256 and seed")
    actual_sha256 = sha256_file(checkpoint)
    if actual_sha256 != expected_sha256:
        raise ValueError("stage-P checkpoint SHA-256 differs from the frozen manifest")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != "V2_RAW_PATIENCE"
        or payload.get("stage") != "P"
        or payload.get("seed") != expected_seed
        or not isinstance(payload.get("fold"), str)
        or not payload.get("fold")
    ):
        raise ValueError("stage-P handoff schema, stage, seed, or fold is invalid")
    contract = _verified_checkpoint_input_contract(payload)
    if contract.get("model_config") != model_config_contract(model.config):
        raise ValueError("stage-P model contract differs from stage F")
    if fine_tune_input_contract is not None:
        if fine_tune_input_contract.get("model_config") != contract.get(
            "model_config"
        ):
            raise ValueError("stage-P and stage-F model contracts differ")
        pretrain_inputs = contract.get("training")
        fine_inputs = fine_tune_input_contract.get("training")
        if not isinstance(pretrain_inputs, Mapping) or not isinstance(
            fine_inputs, Mapping
        ):
            raise ValueError("stage-P handoff input provenance is missing")
        if _input_static_identity(pretrain_inputs) != _input_static_identity(
            fine_inputs
        ):
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
        "V2_RAW_PATIENCE",
        "V2_FINAL_EMA_0995",
    }:
        raise ValueError("file is not a v2 stage checkpoint")
    contract = _verified_checkpoint_input_contract(payload)
    contract_config = model.config if expected_model_config is None else expected_model_config
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
        batch["slow_history_mask"],
        batch["active_mask"],
        batch.get("fast_patches"),
        batch.get("fast_patch_mask"),
        batch.get("fast_present"),
        batch.get("days_since_last_slow_row"),
        batch.get("fast_state_position"),
        batch.get("v1_equity_slow"),
    )


def _to_device(
    batch: Mapping[str, object],
    device: torch.device,
    *,
    omit_fast_stream: bool = False,
) -> dict[str, torch.Tensor]:
    names = {
        "slow_features",
        "slow_history_mask",
        "active_mask",
        "fast_patches",
        "fast_patch_mask",
        "fast_present",
        "days_since_last_slow_row",
        "fast_state_position",
        "v1_equity_slow",
        "targets",
        "target_mask",
        "to_close_target",
        "to_close_mask",
    }
    if omit_fast_stream:
        names -= {
            "fast_patches",
            "fast_patch_mask",
            "fast_state_position",
            "v1_equity_slow",
        }
    transferred = {
        name: value.to(device, non_blocking=device.type == "cuda")
        for name, value in batch.items()
        if name in names and isinstance(value, torch.Tensor)
    }
    present = transferred.get("fast_present")
    if (
        not omit_fast_stream
        and present is not None
        and not torch.any(present.bool())
    ):
        for name in (
            "fast_patches",
            "fast_patch_mask",
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
    if targets.shape[-1] == 5:
        to_close = transferred.pop("to_close_target", None)
        to_close_mask = transferred.pop("to_close_mask", None)
        if to_close is None:
            to_close = torch.zeros_like(targets[..., :1])
            to_close_mask = torch.zeros_like(target_mask[..., :1])
        else:
            if to_close.ndim == 2:
                to_close = to_close[..., None]
            if to_close_mask is None:
                raise ValueError("to-close target and mask must be provided together")
            if to_close_mask.ndim == 2:
                to_close_mask = to_close_mask[..., None]
            if (
                to_close.shape != targets.shape[:-1] + (1,)
                or to_close_mask.shape != to_close.shape
            ):
                raise ValueError("to-close target and mask are misaligned")
        transferred["targets"] = torch.cat((targets, to_close), dim=-1)
        transferred["target_mask"] = torch.cat(
            (target_mask.bool(), to_close_mask.bool()), dim=-1
        )
    elif targets.shape[-1] != 6:
        raise ValueError("v2 targets must contain five horizons plus optional to-close")
    return transferred


def _selection_score(
    model: nn.Module,
    loader: Iterable[Mapping[str, object]],
    device: torch.device,
    *,
    stage: str,
    parity: int | None,
    use_bf16: bool,
) -> float:
    model.eval()
    prediction_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    date_rows: list[np.ndarray | None] = []
    with torch.no_grad():
        for cpu_batch in loader:
            _validate_stage_batch(stage, cpu_batch, require_date_pairs=False)
            batch = _to_device(cpu_batch, device, omit_fast_stream=stage == "P")
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_bf16 and device.type == "cuda",
            ):
                predictions = _model_forward(model, batch)
            prediction_rows.append(predictions[..., :4].float().cpu().numpy())
            target_rows.append(batch["targets"][..., :4].float().cpu().numpy())
            mask_rows.append(batch["target_mask"][..., :4].bool().cpu().numpy())
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
    selected_dates = (
        np.ones(predictions.shape[0], dtype=bool)
        if parity is None
        else block_parity_mask(predictions.shape[0], parity)
    )
    correlations: list[list[float]] = [[] for _ in range(4)]
    for head in range(4):
        for date in np.flatnonzero(selected_dates):
            valid = mask[date, :, head]
            if valid.sum() < 2:
                continue
            left = average_ranks(predictions[date, valid, head].astype(np.float64))
            right = average_ranks(targets[date, valid, head].astype(np.float64))
            left -= left.mean()
            right -= right.mean()
            denominator = np.sqrt(np.sum(left**2) * np.sum(right**2))
            if denominator > 0:
                correlations[head].append(float(np.sum(left * right) / denominator))
    if any(not values for values in correlations):
        raise ValueError("selection parity lacks a primary-horizon IC group")
    horizon_means = [np.mean(values) for values in correlations]
    return float(np.mean(horizon_means))


def compile_forward(
    model: nn.Module,
    *,
    backend: str = "inductor",
    mode: str | None = "max-autotune",
) -> nn.Module:
    # PyTorch requires the RNN opt-in before Dynamo will capture nn.GRU.
    torch._dynamo.config.allow_rnn = True
    options: dict[str, object] = {
        "backend": backend,
        "fullgraph": True,
        "dynamic": False,
    }
    if mode is not None:
        options["mode"] = mode
    return torch.compile(model, **options)


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
) -> None:
    sampler = getattr(loader, "batch_sampler", None)
    if (
        not isinstance(sampler, DatePairBatchSampler)
        or sampler.pairs_per_batch != 8
        or sampler.drop_last is not True
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
                slow_names.extend(values)
            if not slow_names or not all(
                isinstance(value, str) and value for value in slow_names
            ):
                raise ValueError("store slow feature names are malformed")
            intraday_names = list(feature_names.get("intraday", ()))
            if not all(
                isinstance(value, str) and value for value in intraday_names
            ):
                raise ValueError("store intraday feature names are malformed")
            selected_dates = np.asarray(dates[indices], dtype="datetime64[D]")
            date_strings = [str(value) for value in selected_dates]
            stage = str(getattr(candidate, "stage", ""))
            segments = _model_input_segments(
                np.asarray(dates, dtype="datetime64[D]"), indices, stage
            )
            entry_alignment = (
                segments[0]["entry_alignment"]
                if len(segments) == 1
                else "per_segment"
            )
            metadata = manifest.get("metadata", {})
            if not isinstance(metadata, Mapping):
                raise ValueError("store manifest metadata is malformed")
            fast_identity = {
                key: metadata.get(key)
                for key in (
                    "v1_fast_store",
                    "v1_fast_files",
                    "v1_store_v2_zero_dynamic_channels",
                    "v1_store_v2_zero_slow_fields",
                    "v1_isin_subset_verified",
                    "v1_calendar_verified",
                )
            }
            return {
                "schema": "BRAZIL_RV_V2_MODEL_INPUT_V1",
                "store": {
                    "schema": manifest.get("schema"),
                    "manifest_sha256": sha256_file(store_root / "manifest.json"),
                    "axes": manifest.get("axes"),
                    "fast_identity": fast_identity,
                },
                "features": {
                    "ordered_slow_and_sidecar_names": slow_names,
                    "enabled_sidecar_groups": list(enabled_sidecars),
                    "ordered_sidecar_names": sidecar_names,
                    "ordered_intraday_names": intraday_names,
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
            name, alignment = "P", "through_t"
        elif finetune.all():
            name, alignment = "F", "through_t_minus_1"
        else:
            raise ValueError("loader segment crosses an unauthorized P/F boundary")
        date_strings = [str(value) for value in selected]
        little_endian = np.asarray(values, dtype="<i8")
        result.append(
            {
                "name": name,
                "entry_alignment": alignment,
                "indices_sha256": hashlib.sha256(
                    little_endian.tobytes()
                ).hexdigest(),
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


def _date_span_payload(
    dates: np.ndarray, indices: np.ndarray
) -> dict[str, object]:
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
        (axis >= np.datetime64(STORE_START))
        & (axis <= np.datetime64(PRETRAIN_END))
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
        result[fold.name] = {
            "fit": _date_span_payload(axis, fit_indices),
            "selection": _date_span_payload(axis, selection_indices),
            "embargo_sessions": 75,
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
    return payload


def build_checkpoint_input_contract(
    model_config: ModelConfig,
    train_loader: Iterable[Mapping[str, object]],
    selection_loader: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    training = _loader_input_payload(train_loader)
    selection = _loader_input_payload(selection_loader)
    if training is None or selection is None:
        raise ValueError("checkpoint inputs require model-aware dataset loaders")
    result: dict[str, object] = {
        "schema": "BRAZIL_RV_V2_CHECKPOINT_INPUT_V1",
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
    if contract.get("schema") != "BRAZIL_RV_V2_CHECKPOINT_INPUT_V1":
        raise ValueError("checkpoint input contract schema is not recognized")
    return contract


def _input_static_identity(payload: Mapping[str, object]) -> dict[str, object]:
    return {
        key: payload.get(key)
        for key in ("store", "features", "lookback_sessions")
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
    ordered = features.get("ordered_slow_and_sidecar_names")
    if not isinstance(ordered, list) or len(ordered) != model_config.slow_feature_count:
        raise ValueError("model slow width differs from ordered store feature names")
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
    if (
        not isinstance(training_splits, Mapping)
        or training_splits != selection_splits
    ):
        raise ValueError("training and selection canonical split provenance differs")
    if stage == "P":
        if (
            training.get("entry_alignment") != "through_t"
            or selection.get("entry_alignment") != "through_t"
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
                training.get("entry_alignment") == "through_t_minus_1"
                and selection.get("entry_alignment") == "through_t_minus_1"
            )
        else:
            valid_alignment = (
                training.get("entry_alignment") == "per_segment"
                and selection.get("entry_alignment") == "through_t_minus_1"
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
        if gap != 75:
            raise ValueError(f"stage {stage} requires its exact 75-session embargo")
        matches = [
            name
            for name in ("F1", "F2", "F3")
            if fold == name or fold.startswith(f"{name}_")
        ]
        if len(matches) != 1:
            raise ValueError(
                f"stage {stage} fold must identify canonical F1, F2, or F3"
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
            pretrain_segment.get("entry_alignment") != "through_t"
            or fine_segment.get("entry_alignment") != "through_t_minus_1"
            or selection_segment.get("entry_alignment") != "through_t_minus_1"
            or pretrain_segment.get("first_date") != str(STORE_START)
            or pretrain_segment.get("last_date") != str(PRETRAIN_END)
            or pretrain_segment.get("first_index")
            != pretrain_fit.get("first_index")
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
                not isinstance(actual.get(field), str)
                or len(str(actual[field])) != 64
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
) -> None:
    required = {
        "slow_features",
        "slow_history_mask",
        "active_mask",
        "targets",
        "target_mask",
    }
    missing = required - batch.keys()
    if missing:
        raise ValueError(f"training batch is missing tensors: {sorted(missing)}")
    slow_features = batch["slow_features"]
    if not isinstance(slow_features, torch.Tensor):
        raise TypeError("collated training arrays must be tensors")
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
    if stage == "P":
        present = batch.get("fast_present")
        if present is None and "fast_patches" in batch:
            raise ValueError("stage P must explicitly mark every fast stream absent")
        if present is not None and (
            not isinstance(present, torch.Tensor) or torch.any(present.bool())
        ):
            raise ValueError("stage P cannot access a present fast stream")
        days = batch.get("days_since_last_slow_row")
        if days is not None and (
            not isinstance(days, torch.Tensor) or torch.any(days != 0)
        ):
            raise ValueError("stage P requires days_since_last_slow_row = 0")
    else:
        present = batch.get("fast_present")
        days = batch.get("days_since_last_slow_row")
        if not isinstance(present, torch.Tensor) or not isinstance(days, torch.Tensor):
            raise ValueError("F/J batches require both stage-alignment flags")
        if torch.any((days != 0) & (days != 1)):
            raise ValueError("days_since_last_slow_row must be zero or one")
        if stage == "F" and torch.any(days != 1):
            raise ValueError("stage F requires days_since_last_slow_row = 1")
        if torch.any((days == 0) & present.bool()):
            raise ValueError("joint pretrain rows cannot expose the fast stream")
        if torch.any(present.bool()) and (
            "fast_patches" not in batch
            or "fast_patch_mask" not in batch
            or "v1_equity_slow" not in batch
        ):
            raise ValueError(
                "present fast samples require patches, their mask, and v1 equity slow"
            )
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
    selection_parity: int | None
    evaluation_parity: int | None


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
    selection_parity: int | None = None,
    maximum_epochs: int = MAX_EPOCHS,
    patience: int = EARLY_STOP_PATIENCE,
    learning_rate: float = ADAMW_LR,
    sam_rho: float = SAM_RHO,
    device: torch.device | None = None,
    allow_untracked_test_loaders: bool = False,
) -> StageTrainingResult:
    """Run one frozen P/F/J trajectory and archive raw-Patience plus final EMA."""

    if stage not in {"P", "F", "J"}:
        raise ValueError("stage must be P, F, or J")
    if seed not in ALLOWED_SEEDS:
        raise ValueError("seed differs from the frozen v2 screen roster")
    if not fold:
        raise ValueError("fold must be nonempty")
    if not 1 <= maximum_epochs <= MAX_EPOCHS:
        raise ValueError("maximum_epochs must be between one and twenty")
    if stage in {"F", "J"} and selection_parity is None:
        raise ValueError("development F/J stages require an explicit block parity")
    if stage == "P" and selection_parity is not None:
        raise ValueError("stage P uses its embargoed internal holdout without parity")
    if stage == "P" and pretrain_checkpoint is not None:
        raise ValueError("stage P cannot initialize itself from a pretrain checkpoint")
    access_ledgers = {
        "training": _loader_access_payload(train_loader),
        "selection": _loader_access_payload(selection_loader),
    }
    input_stores = {
        "training": _loader_input_payload(train_loader),
        "selection": _loader_input_payload(selection_loader),
    }
    if not allow_untracked_test_loaders and any(
        payload is None for payload in access_ledgers.values()
    ):
        raise ValueError(
            "production train and selection loaders must expose authorized access ledgers"
        )
    if not allow_untracked_test_loaders and any(
        payload is None for payload in input_stores.values()
    ):
        raise ValueError(
            "production train and selection loaders must expose canonical model inputs"
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
    if all(payload is not None for payload in input_stores.values()):
        checkpoint_input_contract = build_checkpoint_input_contract(
            model_config, train_loader, selection_loader
        )
        training_inputs = checkpoint_input_contract["training"]
        selection_inputs = checkpoint_input_contract["selection"]
        assert isinstance(training_inputs, Mapping)
        assert isinstance(selection_inputs, Mapping)
        _validate_tracked_stage_inputs(
            stage, fold, model_config, training_inputs, selection_inputs
        )
    else:
        checkpoint_input_contract = {
            "schema": "BRAZIL_RV_V2_CHECKPOINT_INPUT_V1",
            "implementation_commit": _repository_commit_if_available(),
            "model_config": model_config_contract(model_config),
            "training": None,
            "selection": None,
            "untracked_test_loaders": True,
        }
        checkpoint_input_contract["sha256"] = _canonical_payload_sha256(
            checkpoint_input_contract
        )
    if (pretrain_checkpoint is None) != (expected_pretrain_sha256 is None):
        raise ValueError(
            "stage-P handoff checkpoint and expected SHA-256 must be set together"
        )
    if not allow_untracked_test_loaders:
        _require_production_pair_sampler(train_loader)
    set_deterministic_seed(seed)
    model = DailyMultiHorizonModel(model_config)
    pretrain_provenance: dict[str, object] | None = None
    if pretrain_checkpoint is not None:
        load_pretrain_handoff(
            model,
            pretrain_checkpoint,
            expected_sha256=expected_pretrain_sha256,
            expected_seed=seed,
            fine_tune_input_contract=checkpoint_input_contract,
        )
        pretrain_payload = torch.load(
            pretrain_checkpoint, map_location="cpu", weights_only=True
        )
        assert isinstance(pretrain_payload, Mapping)
        pretrain_contract = _verified_checkpoint_input_contract(pretrain_payload)
        pretrain_provenance = {
            "schema": pretrain_payload["schema"],
            "stage": pretrain_payload["stage"],
            "seed": pretrain_payload["seed"],
            "fold": pretrain_payload["fold"],
            "checkpoint_sha256": model.pretrain_checkpoint_sha256,
            "input_contract_sha256": pretrain_contract["sha256"],
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
    forward_model = compile_forward(model) if model_config.compile_forward else model
    history: list[dict[str, float | int]] = []
    for epoch in range(1, maximum_epochs + 1):
        _set_loader_epoch(train_loader, epoch - 1)
        model.train()
        losses: list[float] = []
        for cpu_batch in train_loader:
            _validate_stage_batch(
                stage,
                cpu_batch,
                expected_pairs=None if allow_untracked_test_loaders else 8,
            )
            batch = _to_device(cpu_batch, device, omit_fast_stream=stage == "P")

            def closure() -> torch.Tensor:
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=model_config.use_bf16 and device.type == "cuda",
                ):
                    flat_scores = _model_forward(forward_model, batch)
                    scores = reshape_date_pair_batch(flat_scores)
                    targets = reshape_date_pair_batch(batch["targets"])
                    target_mask = reshape_date_pair_batch(batch["target_mask"])
                    active = reshape_date_pair_batch(batch["active_mask"])
                    return multi_horizon_loss(
                        scores,
                        targets,
                        target_mask,
                        score_mask=active,
                        persistence_weight=model_config.lambda_persistence,
                        temperature=model_config.soft_rank_temperature,
                    )

            update = sam_step(
                model,
                optimizer,
                closure,
                rho=sam_rho,
                scheduler=scheduler,
                ema=ema,
            )
            losses.append(update.first_loss)
        if not losses:
            raise ValueError("training loader produced no date pairs")
        selection_score = _selection_score(
            forward_model,
            selection_loader,
            device,
            stage=stage,
            parity=selection_parity,
            use_bf16=model_config.use_bf16,
        )
        history.append(
            {
                "epoch": epoch,
                "training_loss": float(np.mean(losses)),
                "selection_score": selection_score,
            }
        )
        if tracker.update(epoch, selection_score, model):
            break
    if tracker.best_state_dict is None or tracker.stopped_epoch is None:
        raise RuntimeError("training ended without a selected Patience state")
    raw_path = output_dir / "raw_patience.pt"
    ema_path = output_dir / "final_ema.pt"
    history_path = output_dir / "history.json"
    manifest_path = output_dir / "run_manifest.json"
    _atomic_torch_save(
        raw_path,
        {
            "schema": "V2_RAW_PATIENCE",
            "stage": stage,
            "seed": seed,
            "fold": fold,
            "model_state_dict": tracker.best_state_dict,
            "patience": tracker.metadata(),
            "input_contract": checkpoint_input_contract,
        },
    )
    _atomic_torch_save(
        ema_path,
        {
            "schema": "V2_FINAL_EMA_0995",
            "stage": stage,
            "seed": seed,
            "fold": fold,
            "model_state_dict": ema.cpu_state_dict(),
            "epoch": tracker.stopped_epoch,
            "input_contract": checkpoint_input_contract,
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
            "schema": "BRAZIL_RV_V2_TRAINING_STAGE_V1",
            "status": "completed",
            "stage": stage,
            "seed": seed,
            "fold": fold,
            "epochs_completed": tracker.stopped_epoch,
            "selected_epoch": tracker.selected_epoch,
            "selection_parity": selection_parity,
            "evaluation_parity": (
                None if selection_parity is None else 1 - selection_parity
            ),
            "model_config": model_config_payload,
            "fast_checkpoint_sha256": model.fast_checkpoint_sha256,
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
            "allow_untracked_test_loaders": allow_untracked_test_loaders,
            "optimizer": {
                "name": "sam_adamw",
                "learning_rate": learning_rate,
                "pretrained_lr_multiplier": 0.3,
                "rho": sam_rho,
                "weight_decay": ADAMW_WEIGHT_DECAY,
                "pretrained_parameter_count": len(model.pretrained_parameter_names),
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
        selection_parity=selection_parity,
        evaluation_parity=(None if selection_parity is None else 1 - selection_parity),
    )
