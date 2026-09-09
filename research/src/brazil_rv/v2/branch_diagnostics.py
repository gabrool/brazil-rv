"""Read-only branch diagnostics for the registered Round-3 experiment."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch
from torch import nn

from .model import DailyMultiHorizonModel


def gradient_norms(model: nn.Module) -> dict[str, float]:
    """L2 norms of the existing gradients, without changing them or drawing RNG."""
    totals: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        branch = name.split(".")[0]
        squared = (
            parameter.grad.detach().float().square().sum()
            if parameter.grad is not None
            else parameter.new_zeros((), dtype=torch.float32)
        )
        totals[branch] = totals.get(branch, torch.zeros_like(squared)) + squared
    if not totals:
        return {}
    values = torch.stack(list(totals.values())).sqrt().cpu().tolist()
    return dict(zip(totals, values, strict=True))


def gate_activations(
    model: DailyMultiHorizonModel,
    loader: Iterable[Mapping[str, object]],
    device: torch.device,
) -> dict[str, object]:
    """Summarize sigmoid gates on active evaluation rows of the selected checkpoint.

    An eager, inference-only pass runs after canonical scores are computed. Hooks
    never enter the compiled training/scoring graph; labels are ignored. Preserve
    RNG even if iteration through a DataLoader draws its worker seed.
    """
    from .score import _forward, _model_batch

    totals: dict[str, dict[int, torch.Tensor]] = {"fast": {}, "pool": {}}
    active: torch.Tensor
    present: torch.Tensor
    archive_counts = torch.zeros(2, dtype=torch.int64, device=device)
    effective_counts = torch.zeros_like(archive_counts)
    date_count = 0

    def capture(branch: str):
        def hook(_module, _args, output):
            values = output.detach().float().sigmoid()
            for state in (0, 1):
                selected = values[active & (present == bool(state))]
                if not selected.numel():
                    continue
                row = torch.stack(
                    (
                        selected.new_tensor(selected.numel(), dtype=torch.float64),
                        selected.double().sum(),
                        selected.double().square().sum(),
                        selected.min().double(),
                        selected.max().double(),
                    )
                )
                previous = totals[branch].get(state)
                if previous is not None:
                    row[:3] += previous[:3]
                    row[3] = torch.minimum(row[3], previous[3])
                    row[4] = torch.maximum(row[4], previous[4])
                totals[branch][state] = row

        return hook

    handles = [
        model.fast_gate.register_forward_hook(capture("fast")),
        model.pool_gate.register_forward_hook(capture("pool")),
    ]
    was_training = model.training
    model.eval()
    try:
        with torch.random.fork_rng(), torch.inference_mode():
            for cpu_batch in loader:
                batch = _model_batch(
                    cpu_batch, device, omit_fast_stream=model.config.disable_fast_stream
                )
                active = batch["active_mask"].bool()
                present = batch["fast_present"].bool()
                archive_present = cpu_batch["fast_present"].to(device).bool()
                for state in (0, 1):
                    archive_counts[state] += (
                        active & (archive_present == bool(state))
                    ).sum()
                    effective_counts[state] += (active & (present == bool(state))).sum()
                date_count += active.shape[0]
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=model.config.use_bf16 and device.type == "cuda",
                ):
                    _forward(model, batch)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    report: dict[str, object] = {
        "population": "active_evaluation_name_days",
        "checkpoint": "raw_patience",
        "pass": "eager_inference_after_canonical_scoring",
        "disable_fast_stream": model.config.disable_fast_stream,
        "date_count": date_count,
        "archive_fast_present_counts": archive_counts.cpu().tolist(),
        "effective_fast_present_counts": effective_counts.cpu().tolist(),
        "gates": {},
    }
    for branch, states in totals.items():
        rows = {}
        for state in (0, 1):
            row = states.get(state)
            if row is None:
                rows[str(state)] = {
                    "channel_observations": 0,
                    "mean": None,
                    "std": None,
                }
                continue
            count, total, squares, minimum, maximum = row.cpu().tolist()
            mean = total / count
            rows[str(state)] = {
                "channel_observations": int(count),
                "mean": mean,
                "std": max(0.0, squares / count - mean * mean) ** 0.5,
                "minimum": minimum,
                "maximum": maximum,
            }
        report["gates"][branch] = rows
    return report
