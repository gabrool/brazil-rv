"""Independent-date full-model relational teachers with observable donor roles."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from types import MethodType

import numpy as np
import torch

from .artifacts import write_json_atomic
from .characteristic_model import CharacteristicConfig, CharacteristicModel
from .round7_training import (
    TrainingObjective,
    daily_primary_ic,
    forward,
    learning_rate_fraction,
    optimizer_step,
    recipe_optimizer,
)
from .post_data_program import RECIPES
from .train import compile_forward, set_deterministic_seed


def teacher_batch(seed, task, *, count=16, recipients=32, donors=16, persistent=False):
    """Fresh dates: recipients' observable queries address independent donors.

    Channels 0:2 carry recipient queries; 2:4 donor keys; 4:6 messages;
    6 the observed shared regime and 7 the donor-role flag. Recipient messages
    are zero, so donor-only labels have no own-stock predictor. No hidden ID or
    relation matrix is supplied. Other channels are independent nuisance inputs.
    """
    generator = torch.Generator().manual_seed(seed)
    names, steps, fields = recipients + donors, 60, 32
    x = torch.randn(count, names, steps, fields, generator=generator) * 0.3
    queries = torch.randn(count, recipients, 2, generator=generator)
    keys = torch.randn(count, donors, 2, generator=generator)
    messages = torch.randn(count, donors, steps, 2, generator=generator)
    if persistent:
        # Isolate stock routing from finding a brief historical message. This
        # diagnostic does not replace the dynamic/lagged acceptance teachers.
        messages = messages[..., -1:, :].expand_as(messages)
    regime = torch.randint(2, (count,), generator=generator)
    x[..., :8] = 0
    x[:, :recipients, :, :2] = queries[:, :, None, :]
    x[:, recipients:, :, 2:4] = keys[:, :, None, :]
    x[:, recipients:, :, 4:6] = messages
    x[..., 6] = (regime * 2 - 1)[:, None, None]
    x[:, recipients:, :, 7] = 1
    active = torch.ones(count, names, dtype=torch.bool)
    for day in range(count):
        # Variable universes within a fixed pad, with at least 24 target names.
        active[day, recipients - int(day % 8) : recipients] = False
        active[day, names - int(day % 4) :] = False
    padding = torch.randint(12, (count, names), generator=generator)
    history = torch.arange(steps)[None, None, :] >= padding[..., None]
    observed = torch.rand(count, names, steps, generator=generator) > 0.05
    observed[..., -1] = True
    # Required teacher endpoints are observed, including in sparse histories.
    observed[..., -21] = True
    observed[..., -41] = True
    valid = (history & observed)[..., None].expand_as(x) & active[..., None, None]
    weights = torch.softmax(
        (queries @ keys.transpose(-1, -2) / 2**0.5).masked_fill(
            ~active[:, None, recipients:], -torch.inf
        ),
        -1,
    )
    if task == "own":
        # A historical lookup, with no current-state shortcut. The original
        # static-query fixture tested current information, not own history.
        signal = x[:, :recipients, -21, 8] + 0.5 * x[:, :recipients, -41, 9]
    elif task == "peer":
        signal = (weights @ messages[..., -1, :1]).squeeze(-1)
    elif task == "lagged":
        signal = (
            weights @ (messages[..., -21, :1] + 0.5 * messages[..., -41, 1:2])
        ).squeeze(-1)
    elif task == "context":
        message = torch.where(
            regime[:, None] == 0, messages[..., -1, 0], messages[..., -21, 1]
        )
        signal = (weights @ message[..., None]).squeeze(-1)
    else:
        raise ValueError("unknown relational teacher")
    targets = torch.zeros(count, names, 5)
    target_mask = torch.zeros_like(targets, dtype=torch.bool)
    for day in range(count):
        included = torch.nonzero(active[day, :recipients]).flatten()
        ranks = signal[day, included].argsort().argsort().float()
        targets[day, included] = (ranks / (len(included) - 1))[:, None]
        target_mask[day, included] = True
    return {
        "slow_features": torch.where(valid, x, 0.0),
        "slow_feature_mask": valid,
        "slow_history_mask": history & active[..., None],
        "slow_feature_age_sessions": torch.where(valid, 0.0, -1.0),
        "active_mask": active,
        "targets": targets,
        "target_mask": target_mask,
    }


def uniform_peer(self, values, valid):
    width = values.shape[-1]
    v = self.qkv(self.norm(values))[..., 2 * width :]
    pooled = torch.where(valid[..., None], v.float(), 0.0).sum(
        1, keepdim=True
    ) / valid.sum(1)[:, None, None].clamp_min(1)
    result = self.output_norm(values + self.output(pooled.to(v.dtype)))
    return torch.where(valid[..., None], result, 0.0)


def fp32_head(self, values):
    # Bounded precision diagnostic, not a financial model change.
    with torch.autocast(device_type=values.device.type, enabled=False):
        return torch.nn.functional.linear(values.float(), self.weight, self.bias)


def no_current_core(self, values):
    # All current fields remain in the final temporal token. Isolate competition
    # from the separate current-state MLP; this is not a financial candidate.
    return values.new_zeros((*values.shape[:-1], self[-1].normalized_shape[0]))


def run(
    output,
    *,
    task,
    recipe_name,
    control,
    seed=11,
    steps=512,
    cuda=False,
    compiled=False,
    persistent=False,
    precision="bf16",
    learning_rate=None,
):
    """Every update sees new dates; validation seeds never occur in training."""
    torch.set_num_threads(6)
    set_deterministic_seed(seed)
    device = torch.device("cuda" if cuda else "cpu")
    config = CharacteristicConfig(temporal_encoder="attention", peer_timing="early")
    model = CharacteristicModel(config).to(device)
    if precision == "fp32_head":
        model.head.forward = MethodType(fp32_head, model.head)
    if control == "no_current_core":
        model.core.forward = MethodType(no_current_core, model.core)
    if control == "uniform":
        model.temporal_peer.peer.forward = MethodType(
            uniform_peer, model.temporal_peer.peer
        )
    recipe = RECIPES[recipe_name]
    if learning_rate is not None:
        recipe = replace(recipe, learning_rate=learning_rate)
    optimizer = recipe_optimizer(model, cuda=cuda, learning_rate=recipe.learning_rate)
    rho = None if control == "adamw" else recipe.rho
    objective = TrainingObjective(
        model,
        characteristic=True,
        head_indices=[2, 3, 4],
        loss_kind="soft_spearman",
        cuda=cuda and precision != "fp32",
    )
    objective = compile_forward(objective) if compiled else objective
    prediction_model = compile_forward(model) if compiled else model

    def batch(batch_seed, size=16):
        value = teacher_batch(batch_seed, task, count=size, persistent=persistent)
        if control == "own":
            # Remove donor observations from this control only. The labels still
            # use independently generated donors; there is no own-stock shortcut.
            value["slow_features"][:, 32:] = 0
            value["slow_feature_mask"][:, 32:] = False
            value["slow_history_mask"][:, 32:] = False
            value["slow_feature_age_sessions"][:, 32:] = -1
        return {name: tensor.to(device) for name, tensor in value.items()}

    validation = [batch(100_000_000 + seed * 100 + i) for i in range(8)]
    validation_support = []

    def readout():
        model.eval()
        values = []
        with torch.no_grad():
            for sample in validation:
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=cuda and precision != "fp32",
                ):
                    scores = (
                        forward(prediction_model, sample, characteristic=True)
                        .float()
                        .mean(2)
                    )
                values.extend(
                    daily_primary_ic(
                        scores.cpu().numpy(),
                        sample["targets"][..., 2:].cpu().numpy(),
                        sample["target_mask"][..., 2:].cpu().numpy(),
                        sample["active_mask"].cpu().numpy(),
                    )
                )
        model.train()
        defined = np.isfinite(values)
        validation_support.append(int(defined.sum()))
        # A flat prediction can make IC undefined. Retain that failure as null
        # and its support, instead of losing the run to JSON's NaN rejection.
        return float(np.mean(values)) if defined.all() else None

    from .training_diagnostics import probe

    diagnostic_batch = batch(seed * 1_000_000, size=2)

    def diagnostic():
        return probe(
            model,
            optimizer,
            diagnostic_batch,
            characteristic=True,
            horizons=config.horizons,
            rho=rho,
            adaptive=recipe.adaptive,
            eta=recipe.eta,
            use_bf16=precision != "fp32",
        )

    diagnostics = {"initial": diagnostic()}

    history = [{"update": 0, "validation_ic": readout()}]
    start = time.perf_counter()
    for update in range(steps):
        sample = batch(seed * 1_000_000 + update)
        for group in optimizer.param_groups:
            group["lr"] = recipe.learning_rate * learning_rate_fraction(update, steps)
        loss, gap = optimizer_step(
            model, optimizer, lambda: objective(sample), rho, adaptive=recipe.adaptive
        )
        if (update + 1) % 64 == 0 or update + 1 == steps:
            record = {
                "update": update + 1,
                "validation_ic": readout(),
                "training_loss": loss,
                "sam_gap": gap,
                "seconds": time.perf_counter() - start,
            }
            history.append(record)
            print(json.dumps({"task": task, "control": control, **record}), flush=True)
            write_json_atomic(
                output,
                {
                    "status": "running",
                    "task": task,
                    "control": control,
                    "seed": seed,
                    "history": history,
                },
            )
    diagnostics["terminal"] = diagnostic()
    result = {
        "status": "completed",
        "task": task,
        "control": control,
        "seed": seed,
        "config": asdict(config),
        "recipe": asdict(recipe),
        "rho": rho,
        "updates": steps,
        "unique_training_dates": steps * 16,
        "independent_validation_dates": 128,
        "history": history,
        "defined_validation_dates": validation_support,
        "diagnostics": diagnostics,
        "seconds": time.perf_counter() - start,
        "compiled": compiled,
        "validation_recipients_only": True,
        "persistent_message_diagnostic": persistent,
        "precision": precision,
        "own_task_definition": "observed recipient channels 8 at t-21 and 9 at t-41; current snapshot independent",
        "financial_data_read": False,
        "torch": torch.__version__,
        "source_hashes_lf": {
            p.name: hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for p in Path(__file__).parent.glob("*.py")
            if p.name
            in {
                "relational_engineering.py",
                "round7_training.py",
                "characteristic_model.py",
                "temporal_pathway.py",
                "post_data_program.py",
            }
        },
    }
    write_json_atomic(output, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--task", choices=("own", "peer", "lagged", "context"), required=True
    )
    parser.add_argument("--recipe", choices=tuple(RECIPES), default="sam125")
    parser.add_argument(
        "--control",
        choices=("full", "own", "uniform", "adamw", "no_current_core"),
        default="full",
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--persistent", action="store_true")
    parser.add_argument(
        "--precision", choices=("bf16", "fp32_head", "fp32"), default="bf16"
    )
    args = parser.parse_args()
    run(
        args.output,
        task=args.task,
        recipe_name=args.recipe,
        control=args.control,
        seed=args.seed,
        steps=args.steps,
        cuda=args.cuda,
        compiled=args.compile,
        persistent=args.persistent,
        precision=args.precision,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()
