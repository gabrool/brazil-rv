"""Selection-aware characteristic training with independent member rank losses."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from brazil_rv.modeling.engine import _soft_spearman_group_losses
from brazil_rv.modeling.trajectory import ModelEMA, temporarily_load_state

from .artifacts import sha256_file, write_json_atomic
from .characteristic_model import CharacteristicModel
from .contract import HORIZONS, TARGET_NEUTRALIZATION_TIE_POLICY
from .data import V2DailyDataset, stage_name_count
from .model import DailyMultiHorizonModel
from .normalization import average_ranks
from .research_rounds import _git_identity
from .round7 import configuration, pretrain_key
from .round7_preprocessing import Round7Preprocessing
from .selection_rules import smoothed_selection
from .train import (
    DateBatchSampler,
    _atomic_torch_save,
    _cli_stage_indices,
    _canonical_payload_sha256,
    _model_forward,
    _rng_state,
    _restore_rng,
    _unique_compiled_graphs,
    compile_forward,
    set_deterministic_seed,
)

CHECKPOINT_SCHEMA = "BRAZIL_RV_SELECTED_CHECKPOINT_V1"


def autocast_dtype(device):
    # BF16 emulation is reported as supported on Turing, but has no Tensor Cores.
    return (
        torch.float16
        if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] < 8
        else torch.bfloat16
    )


@dataclass(frozen=True)
class TrainingRecipe:
    learning_rate: float = 1e-4
    rho: float | None = 0.125
    adaptive: bool = False
    eta: float = 0.01
    transferred_multiplier: float = 0.3
    schedule_epochs: int = 60
    patience: int = 5
    minimum_improvement: float = 0.0001
    # Trailing-window mean of the selection IC decides improvement/patience and
    # selects the window's centre epoch; 1 is the exact historical raw rule.
    selection_smoothing: int = 1


def recipe_contract(recipe):
    """Recipe payload for the frozen contract; the default smoothing keeps old hashes exact."""
    payload = asdict(recipe)
    if payload.get("selection_smoothing", 1) == 1:
        payload.pop("selection_smoothing", None)
    return payload


def _cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu_copy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_copy(v) for v in value)
    return value


def member_loss(scores, targets, mask, *, kind="soft_spearman"):
    """Each date/member/head ranks only securities; then equally average heads."""
    b, n, members, heads = scores.shape
    predictions = scores.permute(0, 2, 1, 3).reshape(b * members, n, heads).float()
    targets = (
        targets[:, None].expand(-1, members, -1, -1).reshape_as(predictions).float()
    )
    mask = mask[:, None].expand(-1, members, -1, -1).reshape_as(predictions).bool()
    if kind == "soft_spearman":
        losses, groups = _soft_spearman_group_losses(predictions, targets, mask)
    elif kind == "pearson_on_ranks":
        count = mask.sum(dim=1)
        groups = count >= 2
        x = (
            predictions
            - (predictions * mask).sum(dim=1, keepdim=True)
            / count[:, None].clamp_min(1)
        ) * mask
        y = (
            targets
            - (targets * mask).sum(dim=1, keepdim=True) / count[:, None].clamp_min(1)
        ) * mask
        denominator = (
            (x.square().sum(dim=1) * y.square().sum(dim=1)).clamp_min(1e-12).sqrt()
        )
        losses = (1.0 - (x * y).sum(dim=1) / denominator) * groups
    else:
        raise ValueError("unknown registered loss")
    return (losses.sum(dim=0) / groups.sum(dim=0).clamp_min(1)).mean()


def recipe_optimizer(
    model, *, cuda, learning_rate=3e-4, transferred=(), transferred_multiplier=1.0
):
    """Module-owned bias/norm exclusions and explicit transferred LR groups."""
    from .train import _parameter_owners

    owners = _parameter_owners(model)
    transferred = frozenset(transferred)
    named = dict(model.named_parameters())
    if transferred - named.keys():
        raise ValueError("transferred parameter names differ from the model")
    routed = {}
    for name, parameter in named.items():
        if not parameter.requires_grad:
            continue
        module, attribute = owners[id(parameter)]
        excluded = attribute.startswith("bias") or isinstance(
            module, (nn.LayerNorm, nn.RMSNorm)
        )
        multiplier = transferred_multiplier if name in transferred else 1.0
        routed.setdefault((excluded, multiplier), []).append(parameter)
    return torch.optim.AdamW(
        [
            {
                "params": parameters,
                "weight_decay": 0.0 if excluded else 0.01,
                "lr": learning_rate * multiplier,
                "lr_multiplier": multiplier,
                "adaptive": not excluded,
            }
            for (excluded, multiplier), parameters in routed.items()
        ],
        lr=learning_rate,
        fused=cuda,
    )


def learning_rate_fraction(update, total_updates):
    """Five-percent warmup, cosine to five-percent peak at the final update."""
    warmup = max(1, math.ceil(0.05 * total_updates))
    if update < warmup:
        return (update + 1) / warmup
    progress = (update - warmup + 1) / max(1, total_updates - warmup)
    return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))


def sam_perturbations(optimizer, rho, *, adaptive=False, eta=0.01):
    """L2 SAM/ASAM: epsilon=rho*T²g/||Tg||; norm/bias metric is identity."""
    if not adaptive:
        used = [
            p
            for group in optimizer.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        gradients = [p.grad for p in used]
        norm = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(gradients)))
        return used, torch._foreach_mul(gradients, rho / norm.clamp_min(1e-12)), norm
    used, metric, gradients = [], [], []
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            used.append(parameter)
            metric.append(
                parameter.detach().abs().add(eta)
                if adaptive and group["adaptive"]
                else torch.ones_like(parameter)
            )
            gradients.append(parameter.grad)
    weighted = torch._foreach_mul(gradients, metric)
    norm = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(weighted)))
    perturbations = torch._foreach_mul(weighted, metric)
    torch._foreach_mul_(perturbations, rho / norm.clamp_min(1e-12))
    return used, perturbations, norm


def optimizer_step(
    model,
    optimizer,
    closure,
    rho,
    *,
    adaptive=False,
    eta=0.01,
    diagnostics=None,
    scaler=None,
):
    """Exact-restore SAM/ASAM with reused RNG; None means one-pass AdamW."""
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    start_rng = _rng_state() if rho is not None else None
    retries = 0

    def backward():
        nonlocal retries
        rng = _rng_state() if scaler is not None else None
        while True:
            value = closure()
            if scaler is None:
                value.backward()
            else:
                scaler.scale(value).backward()
                scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(
                parameters, float("inf"), error_if_nonfinite=scaler is None
            )
            if scaler is None or bool(torch.isfinite(norm)):
                return value, norm
            # Retry the same observations/dropout at a lower scale. Never skip
            # an update (which would change the registered sampling/schedule).
            scaler.update()
            retries += 1
            if retries > 16 or not bool(torch.isfinite(value)):
                raise FloatingPointError("FP16 loss/gradients remain nonfinite")
            optimizer.zero_grad(set_to_none=True)
            _restore_rng(rng)

    loss, first_norm = backward()
    if diagnostics is not None:
        diagnostics["clean_gradient_norm"] = float(first_norm)
    if rho is None:
        descent_norm = torch.nn.utils.clip_grad_norm_(
            parameters, 1.0, error_if_nonfinite=True
        )
        if diagnostics is not None:
            diagnostics["descent_gradient_norm"] = float(descent_norm)
        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
            scaler.update()
        return float(loss.detach()), 0.0
    used, perturbations, metric_norm = sam_perturbations(
        optimizer, rho, adaptive=adaptive, eta=eta
    )
    originals = [p.detach().clone() for p in used]
    if scaler is not None:
        # unscale_ has consumed the clean pass. update resets its per-optimizer
        # state before the perturbed pass; it does not touch model/optimizer.
        scaler.update()
    try:
        with torch.no_grad():
            torch._foreach_add_(used, perturbations)
        optimizer.zero_grad(set_to_none=True)
        _restore_rng(start_rng)
        second, _ = backward()
    finally:
        with torch.no_grad():
            torch._foreach_copy_(used, originals)
    descent_norm = torch.nn.utils.clip_grad_norm_(
        parameters, 1.0, error_if_nonfinite=True
    )
    if diagnostics is not None:
        diagnostics.update(
            loss_scale_retries=retries,
            descent_gradient_norm=float(descent_norm),
            metric_gradient_norm=float(metric_norm),
            perturbation_norm=float(
                torch.linalg.vector_norm(
                    torch.stack(torch._foreach_norm(perturbations))
                )
            ),
        )
    if scaler is None:
        optimizer.step()
    else:
        scaler.step(optimizer)
        scaler.update()
    return float(loss.detach()), float((second - loss).detach())


def model_batch(cpu_batch, device):
    keys = {
        "slow_features",
        "slow_feature_mask",
        "slow_history_mask",
        "slow_feature_age_sessions",
        "active_mask",
        "common_state_features",
        "common_state_feature_mask",
        "common_state_age_sessions",
        "targets",
        "target_mask",
        "economic_target",
        "economic_mask",
    }
    return {
        key: value.to(device, non_blocking=device.type == "cuda")
        for key, value in cpu_batch.items()
        if isinstance(value, torch.Tensor)
        and (key in keys or key.startswith("sidecar_"))
    }


def forward(model, batch, *, characteristic, return_hidden=False):
    if not characteristic:
        result = _model_forward(model, batch, return_hidden=return_hidden)
        scores, hidden = result if return_hidden else (result, None)
        scores = scores[..., : len(HORIZONS)].unsqueeze(2)
        return (scores, hidden.unsqueeze(2)) if return_hidden else scores
    return model(
        batch["slow_features"],
        batch["slow_feature_mask"],
        batch["slow_history_mask"],
        batch["active_mask"],
        slow_feature_age_sessions=batch["slow_feature_age_sessions"],
        sidecars={
            key.removeprefix("sidecar_").removesuffix("_values"): (
                value,
                batch[key.removesuffix("_values") + "_valid"],
                batch[key.removesuffix("_values") + "_age_sessions"],
            )
            for key, value in batch.items()
            if key.startswith("sidecar_") and key.endswith("_values")
        },
        common_state=(
            batch["common_state_features"],
            batch["common_state_feature_mask"],
            batch["common_state_age_sessions"],
        )
        if "common_state_features" in batch
        else None,
        **({"return_hidden": True} if return_hidden else {}),
    )


def daily_primary_ic(predictions, targets, mask, active):
    """Same equal-head, common-population daily IC as the standing selector."""
    daily = np.full(len(predictions), np.nan)
    for day in range(len(predictions)):
        common = (
            active[day]
            & mask[day].all(axis=1)
            & np.isfinite(predictions[day]).all(axis=1)
            & np.isfinite(targets[day]).all(axis=1)
        )
        if common.sum() < 20:
            continue
        correlations = []
        for h in range(predictions.shape[-1]):
            x, y = (
                average_ranks(v[day, common, h].astype(np.float64))
                for v in (predictions, targets)
            )
            x, y = x - x.mean(), y - y.mean()
            scale = np.linalg.norm(x) * np.linalg.norm(y)
            correlations.append(float(x @ y / scale) if scale > 0 else np.nan)
        daily[day] = np.mean(correlations)
    return daily


class TrainingObjective(nn.Module):
    """Compile model and FP32 ranking loss together, fusing pairwise reductions."""

    def __init__(
        self,
        model,
        *,
        characteristic,
        head_indices,
        loss_kind,
        cuda,
        economic_weight=0.0,
    ):
        super().__init__()
        self.model = model
        self.characteristic = characteristic
        self.head_indices = head_indices
        self.loss_kind = loss_kind
        self.cuda = cuda
        self.amp_dtype = autocast_dtype(next(model.parameters()).device)
        self.economic_weight = economic_weight
        self.economic_enabled = bool(economic_weight)

    def forward(self, batch):
        with torch.autocast(
            device_type="cuda" if self.cuda else "cpu",
            dtype=self.amp_dtype,
            enabled=self.cuda,
        ):
            result = forward(
                self.model,
                batch,
                characteristic=self.characteristic,
                return_hidden=self.economic_enabled,
            )
            if self.economic_enabled:
                scores, hidden = result
                economic = self.model.economic_head(hidden).squeeze(-1).float()
            else:
                scores = result
        loss = member_loss(
            scores,
            batch["targets"][..., self.head_indices],
            batch["target_mask"][..., self.head_indices]
            & batch["active_mask"][..., None],
            kind=self.loss_kind,
        )
        if self.economic_enabled:
            from .economic_objective import economic_loss

            loss = loss + self.economic_weight * economic_loss(
                economic, batch["economic_target"], batch["economic_mask"]
            )
        return loss


def selection_readout(model, loader, device, *, characteristic, model_horizons):
    model.eval()
    model_heads = [model_horizons.index(h) for h in (3, 5, 10)]
    target_heads = [HORIZONS.index(h) for h in (3, 5, 10)]
    dates, daily = [], []
    with torch.no_grad():
        for cpu_batch in loader:
            batch = model_batch(cpu_batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype(device),
                enabled=device.type == "cuda",
            ):
                scores = (
                    forward(model, batch, characteristic=characteristic)
                    .float()
                    .mean(dim=2)[..., model_heads]
                )
            daily.extend(
                daily_primary_ic(
                    scores.cpu().numpy(),
                    batch["targets"][..., target_heads].cpu().numpy(),
                    batch["target_mask"][..., target_heads].cpu().numpy(),
                    batch["active_mask"].cpu().numpy(),
                ).tolist()
            )
            dates.extend(cpu_batch["date_index"].tolist())
    if not np.isfinite(daily).any():
        raise ValueError("selection has no defined common-head IC")
    return {
        "date_indices": dates,
        "daily_ic": [float(x) if np.isfinite(x) else None for x in daily],
        "mean_ic": float(np.nanmean(daily)),
    }


def sequential_batches(count, size=16):
    """Cover all dates in order without a singleton tail that recompiles."""
    return [
        part.tolist()
        for part in np.array_split(np.arange(count), math.ceil(count / size))
    ]


class DateTensorCache:
    """Factor overlapping histories by (session, permanent security identity).

    Only the canonical collator supplies values. Repeated histories share storage,
    not observations; gathering restores the original full date/name/60 tensor.
    Snapshots/labels stay on their decision-date axis. No dtype compression.
    """

    def __init__(self, source, date_indices, security_count, device):
        start = time.perf_counter()
        count = len(date_indices)
        self.tensors, self.histories = {}, {}
        self.dates = torch.as_tensor(np.asarray(date_indices).copy())
        self.security_count = security_count
        offset = 0
        materialized_bytes = 0
        for cpu in source:
            batch = model_batch(cpu, torch.device("cpu"))
            size = len(cpu["date_index"])
            if not self.tensors:
                self.lookback = batch["slow_features"].shape[2]
                self.first = int(min(date_indices)) - self.lookback + 1
                slots = (int(max(date_indices)) - self.first + 1) * security_count + 1
                seen = np.zeros(slots, dtype=bool)
                seen[0] = True  # padded security: zero in every canonical tensor
                self.names = torch.empty(
                    (count, cpu["name_index"].shape[1]), dtype=torch.long, device=device
                )
                self.endpoints = (self.dates.to(device) - self.first) * security_count
                self.lags = torch.arange(self.lookback - 1, -1, -1, device=device)
                self.histories = {
                    key: torch.zeros(
                        (slots, *value.shape[3:]), dtype=value.dtype, device=device
                    )
                    for key, value in batch.items()
                    if key.startswith("slow_")
                }
                self.tensors = {
                    key: torch.empty(
                        (count, *value.shape[1:]), dtype=value.dtype, device=device
                    )
                    for key, value in batch.items()
                    if key not in self.histories
                }
            names = cpu["name_index"].numpy()
            steps = cpu["date_index"].numpy()[:, None] - np.arange(
                self.lookback - 1, -1, -1
            )
            keys = (
                (steps[:, None, :] - self.first) * security_count + names[..., None] + 1
            )
            keys = np.where(names[..., None] >= 0, keys, 0).ravel()
            unique, positions = np.unique(keys, return_index=True)
            fresh = ~seen[unique]
            unique, positions = unique[fresh], positions[fresh]
            seen[unique] = True
            destination = torch.as_tensor(unique, device=device)
            self.names[offset : offset + size].copy_(cpu["name_index"].to(device))
            for key, value in batch.items():
                materialized_bytes += value.numel() * value.element_size()
                if key in self.histories:
                    values = value.reshape(-1, *value.shape[3:])[positions].to(device)
                    self.histories[key].index_copy_(0, destination, values)
                else:
                    self.tensors[key][offset : offset + size].copy_(value.to(device))
            offset += size
        if offset != count:
            raise ValueError(
                "date cache did not consume the complete permitted population"
            )
        self.bytes = sum(
            t.numel() * t.element_size()
            for t in (
                *self.tensors.values(),
                *self.histories.values(),
                self.names,
                self.endpoints,
                self.lags,
            )
        )
        self.materialized_bytes = materialized_bytes
        self.seconds = time.perf_counter() - start

    def gather(self, positions):
        index = torch.as_tensor(positions, dtype=torch.long, device=self.names.device)
        names = self.names.index_select(0, index)
        keys = (
            self.endpoints.index_select(0, index)[:, None, None]
            - self.lags * self.security_count
            + names[..., None]
            + 1
        )
        keys = torch.where(names[..., None] >= 0, keys, 0)
        result = {
            name: value.index_select(0, index) for name, value in self.tensors.items()
        }
        result.update({name: value[keys] for name, value in self.histories.items()})
        result["date_index"] = self.dates[positions]
        return result

    def batches(self, sampler):
        return CachedDateBatches(self, sampler)


class CachedDateBatches:
    def __init__(self, cache, sampler):
        self.cache, self.sampler = cache, sampler

    def __len__(self):
        return len(self.sampler)

    def __iter__(self):
        return (self.cache.gather(positions) for positions in self.sampler)


def unexposed_families(dataset, preparation):
    """No P value or known-age exposure: such encoders have no learned mapping."""
    result = []
    for family, scaler in preparation.families.items():
        if any(scaler.support):
            continue
        exposed = False
        for start in range(0, len(dataset.date_indices), 128):
            rows = dataset.date_indices[start : start + 128]
            ages = dataset.store.read(f"sidecar_{family}_age_sessions", rows)
            if family in preparation.source_columns:
                ages = ages[..., preparation.source_columns[family]]
            if np.any((ages >= 0) & dataset.store.read("active", rows)[..., None]):
                exposed = True
                break
        if not exposed:
            result.append(family)
    return result


def transferred_parameter_names(model, unexposed):
    """Wholly cold family encoders use full F LR; preserve learned shared tensors."""
    prefixes = tuple(
        prefix
        for name in unexposed
        for prefix in (f"families.{name}.", f"sidecar_projections.{name}")
    )
    return tuple(
        name for name, _ in model.named_parameters() if not name.startswith(prefixes)
    )


def train(
    store_root,
    output,
    *,
    cell,
    stage,
    fold,
    seed,
    epochs=60,
    recipe=TrainingRecipe(),
    parent=None,
    parent_sha256=None,
    device=None,
    compiled=True,
    export_scores=False,
    diagnostics=True,
    economic_targets=None,
    economic_weight=0.0,
    ema_half_life_epochs=None,
):
    code = _git_identity()
    cell_name = cell["cell"]
    rho, loss_kind = recipe.rho, "soft_spearman"
    smoothing = int(recipe.selection_smoothing)
    if smoothing < 1:
        raise ValueError("selection_smoothing must be at least one epoch")
    manifest = json.loads((store_root / "manifest.json").read_text(encoding="utf-8"))
    if "data_repair" not in manifest["metadata"]:
        raise ValueError("training requires the accepted post-data store")
    config = configuration(cell, manifest["feature_names"])
    characteristic = cell["graph"] != "s0"
    horizons = config.horizons if characteristic else HORIZONS
    families = tuple(
        n
        for n, _ in (
            config.family_counts if characteristic else config.sidecar_feature_counts
        )
    )
    fit, select, _, fit_window = _cli_stage_indices(store_root, stage, fold)
    split_common = characteristic and "cross_market" in families
    options = dict(
        stage="pretrain" if stage == "P" else "finetune",
        lookback=60,
        enabled_sidecars=families,
        include_intraday=False,
        include_fast=False,
        include_common_state=split_common,
        compact_names=True,
    )
    training = V2DailyDataset(
        store_root, fit, target_window_indices=fit_window, purpose="training", **options
    )
    selection = V2DailyDataset(store_root, select, purpose="selection", **options)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = autocast_dtype(device)
    scaler = (
        torch.amp.GradScaler("cuda", init_scale=256.0)
        if device.type == "cuda" and amp_dtype == torch.float16
        else None
    )
    try:
        parent_payload = None
        parent_preprocessing = None
        if stage == "F":
            if parent is None or parent_sha256 != sha256_file(parent):
                raise ValueError("F requires a hash-bound compatible Stage-P parent")
            parent_payload = torch.load(parent, map_location="cpu", weights_only=True)
            if parent_payload.get("schema") != CHECKPOINT_SCHEMA:
                raise ValueError("candidate F requires a selected compatible P parent")
            parent_preprocessing = Round7Preprocessing.from_payload(
                parent_payload["contract"]["preprocessing"]
            )
        preparation = Round7Preprocessing.fit(
            training,
            split_common=split_common,
            parent=parent_preprocessing,
            excluded_fields=cell.get("excluded_fields"),
        )
        width = stage_name_count(training, selection)
        collator = partial(preparation.collate, fixed_name_count=width)
        economic_contract = None
        if economic_weight:
            from .economic_objective import EconomicCollator

            collator = EconomicCollator(
                collator, economic_targets, fit, fit_window, training.store.isins
            )
            economic_contract = {**collator.contract, "weight": economic_weight}
        sampler = DateBatchSampler(fit, seed=seed)
        train_loader = DataLoader(
            training,
            batch_sampler=sampler,
            collate_fn=collator,
            pin_memory=device.type == "cuda",
        )
        selection_loader = DataLoader(
            selection,
            batch_sampler=sequential_batches(len(selection)),
            collate_fn=collator,
            pin_memory=device.type == "cuda",
        )
        clean_fit_loader = DataLoader(
            training,
            batch_sampler=sequential_batches(len(training)),
            collate_fn=collator,
            pin_memory=device.type == "cuda",
        )
        probe_indices = np.unique(
            np.linspace(0, len(training) - 1, min(64, len(training)), dtype=int)
        )
        probe_loader = DataLoader(
            training,
            batch_sampler=[
                probe_indices[b].tolist()
                for b in sequential_batches(len(probe_indices))
            ],
            collate_fn=collator,
            pin_memory=device.type == "cuda",
        )
        cache_resources = None
        if device.type == "cuda":
            training_cache = DateTensorCache(
                clean_fit_loader, fit, len(training.store.isins), device
            )
            selection_cache = DateTensorCache(
                selection_loader, select, len(selection.store.isins), device
            )
            train_loader = training_cache.batches(sampler)
            selection_loader = selection_cache.batches(
                sequential_batches(len(selection))
            )
            clean_fit_loader = training_cache.batches(sequential_batches(len(training)))
            probe_loader = training_cache.batches(
                [
                    probe_indices[b].tolist()
                    for b in sequential_batches(len(probe_indices))
                ]
            )
            cache_resources = {
                "bytes": training_cache.bytes + selection_cache.bytes,
                "materialized_bytes": training_cache.materialized_bytes
                + selection_cache.materialized_bytes,
                "layout": "session_security_history; exact canonical tensors",
                "preparation_seconds": training_cache.seconds + selection_cache.seconds,
            }
        diagnostic_batch = model_batch(
            collator([training[int(i)] for i in probe_indices[[0, -1]]]), device
        )
        contract = {
            "code": code,
            "store_manifest_sha256": sha256_file(store_root / "manifest.json"),
            "config": asdict(config),
            "pretrain_key": pretrain_key(cell),
            "stage": stage,
            "fold": fold,
            "seed": seed,
            "epochs": epochs,
            "recipe": recipe_contract(recipe),
            "cell": cell,
            "selection_policy": (
                "raw_checkpoint; earlier ties; fixed-schedule patience from epoch 1"
                if smoothing == 1
                else (
                    "raw_checkpoint; earlier ties; fixed-schedule patience from "
                    f"epoch 1; trailing-{smoothing}-epoch mean selection score "
                    "selects the window's centre epoch"
                )
            ),
            "probe_date_indices": fit[probe_indices].tolist(),
            "module_diagnostics": diagnostics,
            "rho": rho,
            "loss": loss_kind,
            "preprocessing": preparation.payload(),
            "unexposed_families": unexposed_families(training, preparation),
            "target_group_tie_policy": TARGET_NEUTRALIZATION_TIE_POLICY,
            "fit_target_window": fit_window.tolist(),
            "access": {
                "training": training.access_ledger.payload(),
                "selection": selection.access_ledger.payload(),
            },
            "padded_name_count": width,
            "compile": compiled,
            "runtime": {
                "torch": str(torch.__version__),
                "autocast_dtype": str(amp_dtype) if device.type == "cuda" else None,
                "gradient_scaling": scaler is not None,
                "cache_layout": "session_security_history",
            },
            "date_tensor_cache": device.type == "cuda",
            "parent_sha256": parent_sha256,
            "economic_auxiliary": economic_contract,
        }
        set_deterministic_seed(seed)
        model = (
            CharacteristicModel(config)
            if characteristic
            else DailyMultiHorizonModel(config)
        ).to(device)
        if stage == "F":
            if parent is None or parent_sha256 != sha256_file(parent):
                raise ValueError("F requires a hash-bound compatible Stage-P parent")
            payload = parent_payload
            if payload["stage"] != "P" or payload["seed"] != seed:
                raise ValueError("parent stage/seed differs")
            parent_contract = payload["contract"]
            if (
                parent_contract["pretrain_key"] != pretrain_key(cell)
                or parent_contract["config"] != asdict(config)
                or parent_contract["store_manifest_sha256"]
                != contract["store_manifest_sha256"]
            ):
                raise ValueError("parent graph/input/store contract differs")
            model.load_state_dict(payload["model_state_dict"], strict=True)
        transferred = (
            transferred_parameter_names(
                model, parent_payload["contract"]["unexposed_families"]
            )
            if stage == "F"
            else ()
        )
        contract["transferred_parameters"] = list(transferred)
        if ema_half_life_epochs is not None:
            contract["ema"] = {
                "half_life_epochs": ema_half_life_epochs,
                "decay": 2 ** (-1 / (ema_half_life_epochs * len(train_loader))),
                "selection": "same prior selection; raw patience bounds shared trajectory",
            }
        if economic_weight:
            from .economic_objective import attach_economic_head

            attach_economic_head(model)
        optimizer = recipe_optimizer(
            model,
            cuda=device.type == "cuda",
            learning_rate=recipe.learning_rate,
            transferred=transferred,
            transferred_multiplier=recipe.transferred_multiplier,
        )
        best_ic, best_epoch, stale = -float("inf"), 0, 0
        ema = ModelEMA(model, contract["ema"]["decay"]) if "ema" in contract else None
        ema_best_ic, ema_best_epoch = -float("inf"), 0
        module_diagnostics = []
        start_epoch, history = 1, []
        previous_compilation_sessions = []
        # Raw per-epoch selection means and, for smoothed selection, the states
        # of the last `smoothing` epochs so the window's centre stays selectable.
        selection_curve, recent_states = [], []
        resume_path = output / "resume.pt"
        if (output / "run_manifest.json").exists():
            finished = json.loads(
                (output / "run_manifest.json").read_text(encoding="utf-8")
            )
            if _canonical_payload_sha256(
                finished["contract"]
            ) != _canonical_payload_sha256(contract):
                raise ValueError("completed fit differs from requested contract")
            for name, digest in finished["artifacts"].items():
                if sha256_file(output / name) != digest:
                    raise ValueError(f"completed fit artifact changed: {name}")
            if (
                finished.get("score_manifest_sha256")
                and sha256_file(output / "scores/score_manifest.json")
                != finished["score_manifest_sha256"]
            ):
                raise ValueError("completed score manifest changed")
            if export_scores and not finished.get("score_manifest_sha256"):
                from .round7_score import score

                score(
                    store_root,
                    output / "selected.pt",
                    output / "scores",
                    expected_sha256=finished["artifacts"]["selected.pt"],
                    compiled=compiled,
                    device=device,
                    fixed_name_count=width,
                )
                finished["score_manifest_sha256"] = sha256_file(
                    output / "scores/score_manifest.json"
                )
                finished["scoring_complete"] = True
                write_json_atomic(output / "run_manifest.json", finished)
            return finished
        if resume_path.exists():
            payload = torch.load(resume_path, map_location=device, weights_only=True)
            if _canonical_payload_sha256(
                payload["contract"]
            ) != _canonical_payload_sha256(contract):
                raise ValueError("resume differs from the frozen training contract")
            model.load_state_dict(payload["model_state_dict"])
            if ema is not None:
                ema.shadow = payload["ema_state_dict"]
                ema_best_ic, ema_best_epoch = (
                    payload["ema_best_ic"],
                    payload["ema_best_epoch"],
                )
            optimizer.load_state_dict(payload["optimizer_state_dict"])
            if scaler is not None:
                scaler.load_state_dict(payload["grad_scaler"])
            history, start_epoch = payload["history"], payload["epoch"] + 1
            best_ic, best_epoch, stale = (
                payload["best_ic"],
                payload["best_epoch"],
                payload["stale"],
            )
            selection_curve = [r["selection"]["mean_ic"] for r in history]
            recent_states = payload.get("recent_states", [])
            module_diagnostics = payload["module_diagnostics"]
            checkpoint = {
                k: payload[k]
                for k in (
                    "schema",
                    "stage",
                    "seed",
                    "fold",
                    "epoch",
                    "contract",
                    "model_state_dict",
                )
            }
            previous_compilation_sessions = payload.get("compilation_sessions", [])
            _restore_rng(
                (
                    payload["cpu_rng"].cpu(),
                    [v.cpu() for v in payload["cuda_rng"]]
                    if payload["cuda_rng"] is not None
                    else None,
                )
            )
        else:
            # run_many creates only its two log files before starting the child.
            if output.exists() and any(
                p.name not in {"launcher.stdout.log", "launcher.stderr.log"}
                for p in output.iterdir()
            ):
                raise ValueError("incomplete fit has no resumable epoch checkpoint")
            output.mkdir(parents=True, exist_ok=True)
            (output / "epochs").mkdir()
        objective = TrainingObjective(
            model,
            characteristic=characteristic,
            head_indices=[HORIZONS.index(h) for h in horizons],
            loss_kind=loss_kind,
            cuda=device.type == "cuda",
            economic_weight=economic_weight,
        )
        training_forward = compile_forward(objective) if compiled else objective
        forward_model = compile_forward(model) if compiled else model
        total_updates = recipe.schedule_epochs * len(train_loader)
        train_graphs = selection_graphs = 0
        run_start = time.perf_counter()

        def diagnostic(label):
            from .training_diagnostics import probe

            if diagnostics:
                module_diagnostics.append(
                    {
                        "state": label,
                        **probe(
                            model,
                            optimizer,
                            diagnostic_batch,
                            characteristic=characteristic,
                            horizons=horizons,
                            rho=rho,
                            adaptive=recipe.adaptive,
                            eta=recipe.eta,
                            amp_dtype=amp_dtype,
                        ),
                    }
                )

        if start_epoch == 1:
            diagnostic("initial")
        for epoch in range(start_epoch, epochs + 1):
            if stale >= recipe.patience:
                break
            epoch_start = time.perf_counter()
            model.train()
            sampler.set_epoch(epoch)
            losses, gaps, clipped, retries = [], [], [], []
            for batch_number, cpu_batch in enumerate(train_loader):
                lr = recipe.learning_rate * learning_rate_fraction(
                    (epoch - 1) * len(train_loader) + batch_number, total_updates
                )
                for group in optimizer.param_groups:
                    group["lr"] = lr * group["lr_multiplier"]
                batch = model_batch(cpu_batch, device)

                before = _unique_compiled_graphs()
                step_diagnostics = {}
                loss, gap = optimizer_step(
                    model,
                    optimizer,
                    lambda: training_forward(batch),
                    rho,
                    adaptive=recipe.adaptive,
                    eta=recipe.eta,
                    diagnostics=step_diagnostics,
                    scaler=scaler,
                )
                if ema is not None:
                    ema.update(model)
                train_graphs += _unique_compiled_graphs() - before
                losses.append(loss)
                gaps.append(gap)
                clipped.append(step_diagnostics["descent_gradient_norm"] > 1.0)
                retries.append(step_diagnostics.get("loss_scale_retries", 0))
            before = _unique_compiled_graphs()
            readout = selection_readout(
                forward_model,
                selection_loader,
                device,
                characteristic=characteristic,
                model_horizons=horizons,
            )
            ema_readout = None
            if ema is not None:
                rng = _rng_state()
                try:
                    with temporarily_load_state(model, ema.shadow):
                        ema_readout = selection_readout(
                            forward_model,
                            selection_loader,
                            device,
                            characteristic=characteristic,
                            model_horizons=horizons,
                        )
                finally:
                    _restore_rng(rng)
            rng = _rng_state()
            try:
                fit_readout = selection_readout(
                    forward_model,
                    probe_loader,
                    device,
                    characteristic=characteristic,
                    model_horizons=horizons,
                )
            finally:
                _restore_rng(rng)
            selection_graphs += _unique_compiled_graphs() - before
            selection_curve.append(readout["mean_ic"])
            smoothed_scores, centres = smoothed_selection(selection_curve, smoothing)
            selection_score = float(smoothed_scores[-1])
            centre_epoch = int(centres[-1]) + 1
            improved = selection_score > best_ic + recipe.minimum_improvement
            if improved:
                best_ic, best_epoch, stale = selection_score, centre_epoch, 0
            else:
                stale += 1
            record = {
                "epoch": epoch,
                "training_loss": float(np.mean(losses)),
                "sam_gap": float(np.mean(gaps)),
                "selection": readout,
                "clean_fit_probe": fit_readout,
                "gradient_clip_fraction": float(np.mean(clipped)),
                "selected": improved,
                "updates": len(losses),
                "loss_scale_retries": sum(retries),
                "loss_scale": scaler.get_scale() if scaler is not None else None,
                "seconds": time.perf_counter() - epoch_start,
                "final_learning_rate": lr,
            }
            if ema_readout is not None:
                record["ema_selection"] = ema_readout
            if smoothing > 1:
                record["selection_score"] = selection_score
                record["selected_epoch"] = best_epoch if improved else None
            history.append(record)
            if epoch in (1, 3):
                diagnostic(f"epoch_{epoch}")
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            checkpoint = {
                "schema": CHECKPOINT_SCHEMA,
                "stage": stage,
                "seed": seed,
                "fold": fold,
                "epoch": epoch,
                "contract": contract,
                "model_state_dict": state,
            }
            _atomic_torch_save(output / "epochs" / f"epoch_{epoch:03d}.pt", checkpoint)
            if smoothing > 1:
                recent_states.append(
                    {
                        "epoch": epoch,
                        "model_state_dict": state,
                        "optimizer_state_dict": _cpu_copy(optimizer.state_dict()),
                    }
                )
                del recent_states[:-smoothing]
            if improved and smoothing == 1:
                _atomic_torch_save(
                    output / "selected.pt",
                    {
                        **checkpoint,
                        "selection_ic": best_ic,
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                )
            elif improved:
                chosen = next(s for s in recent_states if s["epoch"] == best_epoch)
                _atomic_torch_save(
                    output / "selected.pt",
                    {
                        **checkpoint,
                        "epoch": chosen["epoch"],
                        "model_state_dict": chosen["model_state_dict"],
                        "selection_ic": best_ic,
                        "optimizer_state_dict": chosen["optimizer_state_dict"],
                    },
                )
            if (
                ema_readout is not None
                and ema_readout["mean_ic"] > ema_best_ic + recipe.minimum_improvement
            ):
                ema_best_ic, ema_best_epoch = ema_readout["mean_ic"], epoch
                _atomic_torch_save(
                    output / "selected_ema.pt",
                    {
                        **checkpoint,
                        "model_state_dict": ema.cpu_state_dict(),
                        "selection_ic": ema_best_ic,
                        "weight_rule": "update_ema",
                    },
                )
            rng, cuda_rng = _rng_state()
            _atomic_torch_save(
                resume_path,
                {
                    **checkpoint,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_ic": best_ic,
                    "grad_scaler": scaler.state_dict() if scaler is not None else None,
                    "best_epoch": best_epoch,
                    "stale": stale,
                    **(
                        {
                            "ema_state_dict": ema.cpu_state_dict(),
                            "ema_best_ic": ema_best_ic,
                            "ema_best_epoch": ema_best_epoch,
                        }
                        if ema is not None
                        else {}
                    ),
                    "module_diagnostics": module_diagnostics,
                    "history": history,
                    **({"recent_states": recent_states} if smoothing > 1 else {}),
                    "cpu_rng": rng,
                    "cuda_rng": cuda_rng,
                    "compilation_sessions": [
                        *previous_compilation_sessions,
                        {
                            "start_epoch": start_epoch,
                            "end_epoch": epoch,
                            "training": train_graphs,
                            "selection": selection_graphs,
                        },
                    ],
                },
            )
            write_json_atomic(output / "history.json", history)
            print(
                json.dumps(
                    {
                        "cell": cell_name,
                        "stage": stage,
                        "fold": fold,
                        "seed": seed,
                        "epoch": epoch,
                        "budget": epochs,
                        "selection_ic": readout["mean_ic"],
                        "seconds": record["seconds"],
                    }
                ),
                flush=True,
            )
        terminal_readout = selection_readout(
            forward_model,
            clean_fit_loader,
            device,
            characteristic=characteristic,
            model_horizons=horizons,
        )
        selected = torch.load(
            output / "selected.pt", map_location=device, weights_only=True
        )
        model.load_state_dict(selected["model_state_dict"])
        selected_fit = selection_readout(
            forward_model,
            clean_fit_loader,
            device,
            characteristic=characteristic,
            model_horizons=horizons,
        )
        optimizer.load_state_dict(selected["optimizer_state_dict"])
        diagnostic("selected")
        write_json_atomic(
            output / "diagnostics.json",
            {
                "module_probes": module_diagnostics,
                "selected_clean_fit": selected_fit,
                "terminal_clean_fit": terminal_readout,
            },
        )
        report = {
            "schema": "BRAZIL_RV_SELECTED_FIT_V1",
            "status": "completed",
            "seed": seed,
            "fold": fold,
            "stage": stage,
            "contract": contract,
            "epochs_completed": len(history),
            "selected_epoch": best_epoch,
            **(
                {"ema_selected_epoch": ema_best_epoch, "ema_selection_ic": ema_best_ic}
                if ema is not None
                else {}
            ),
            "selection_ic": best_ic,
            **({"selection_smoothing": smoothing} if smoothing > 1 else {}),
            "stop_reason": "patience" if stale >= recipe.patience else "ceiling",
            "date_tensor_cache_resources": cache_resources,
            "compiled_graphs": {
                "training": max(
                    [
                        train_graphs,
                        *(s["training"] for s in previous_compilation_sessions),
                    ]
                ),
                "selection": max(
                    [
                        selection_graphs,
                        *(s["selection"] for s in previous_compilation_sessions),
                    ]
                ),
            },
            "compiled_graph_count_rule": "maximum per process; complete session records retained",
            "compilation_sessions": [
                *previous_compilation_sessions,
                {
                    "start_epoch": start_epoch,
                    "end_epoch": len(history),
                    "training": train_graphs,
                    "selection": selection_graphs,
                },
            ],
            "seconds_this_process": time.perf_counter() - run_start,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated()
            if device.type == "cuda"
            else 0,
            "artifacts": {
                str(p.relative_to(output)): sha256_file(p)
                for p in [
                    output / "selected.pt",
                    *([output / "selected_ema.pt"] if ema is not None else []),
                    output / "history.json",
                    output / "diagnostics.json",
                    *sorted((output / "epochs").glob("*.pt")),
                ]
            },
        }
        if export_scores:
            from .round7_score import score

            score(
                store_root,
                output / "selected.pt",
                output / "scores",
                expected_sha256=report["artifacts"]["selected.pt"],
                compiled=compiled,
                device=device,
                reusable_models=(model, forward_model),
                fixed_name_count=width,
            )
            report["score_manifest_sha256"] = sha256_file(
                output / "scores/score_manifest.json"
            )
            report["scoring_complete"] = True
        write_json_atomic(output / "run_manifest.json", report)
        # A sealed fit resumes from its hash-checked artifacts. The large transient
        # optimizer/RNG checkpoint is useful only while training is incomplete.
        resume_path.unlink(missing_ok=True)
        return report
    finally:
        training.store.close()
        selection.store.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cell-spec", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--stage", choices=("P", "F"), required=True)
    parser.add_argument("--fold", default="pretrain_internal")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--parent-sha256")
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--export-scores", action="store_true")
    args = parser.parse_args()
    train(
        args.store,
        args.output,
        cell=json.loads(args.cell_spec.read_text(encoding="utf-8")),
        recipe=TrainingRecipe(**json.loads(args.recipe.read_text(encoding="utf-8"))),
        stage=args.stage,
        fold=args.fold,
        seed=args.seed,
        epochs=args.epochs,
        parent=args.parent,
        parent_sha256=args.parent_sha256,
        compiled=not args.eager,
        export_scores=args.export_scores,
    )


if __name__ == "__main__":
    main()
