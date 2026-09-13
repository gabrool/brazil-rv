"""Unseen-example synthetic peer-timing check; never consumes market labels."""

import argparse
import time
from pathlib import Path

import torch

from .artifacts import write_json_atomic
from .characteristic_model import CharacteristicConfig, CharacteristicModel
from .model import encode_slow_history
from .round7 import PATHWAY_CELLS


def sample(seed, dates=4, names=16):
    generator = torch.Generator().manual_seed(seed)
    values = torch.zeros(dates, names, 60, 32)
    groups = (torch.arange(names) >= names // 2).long()
    shocks = torch.randn(dates, names, generator=generator)
    values[..., 0] = 2 * groups[None, :, None].float() - 1
    values[..., -3, 1] = shocks
    # Leave-one-out group return at a known historical lag. Own returns alone
    # have zero population covariance with the target; group identity is an
    # observed input. This isolates peer aggregation without an additional
    # randomly selected leader and multiplicative exposure-learning task.
    totals = shocks.reshape(dates, 2, names // 2).sum(-1)
    targets = (totals[:, groups] - shocks) / (names // 2 - 1)
    data = dict(
        slow_features=values,
        slow_feature_mask=torch.ones_like(values, dtype=torch.bool),
        slow_history_mask=torch.ones(dates, names, 60, dtype=torch.bool),
        slow_feature_age_sessions=torch.zeros_like(values),
        active_mask=torch.ones(dates, names, dtype=torch.bool),
    )
    return data, targets


def run(output, steps=800):
    torch.set_num_threads(4)
    results = []
    validation, target = sample(98765, dates=48)
    for cell in PATHWAY_CELLS:
        torch.manual_seed(11)
        model = CharacteristicModel(
            CharacteristicConfig(
                temporal_encoder=cell["temporal_encoder"],
                peer_timing=cell["peer_timing"],
            )
        )
        head = torch.nn.Linear(64, 1)
        parameters = [
            p
            for name, p in model.named_parameters()
            if name.startswith(("slow_", "temporal_peer"))
        ] + list(head.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=0.001, weight_decay=0.01)

        def predict(data):
            sequence = encode_slow_history(
                data["slow_features"],
                data["slow_feature_mask"],
                data["slow_history_mask"],
                data["slow_feature_age_sessions"],
                config=model.temporal_config,
                input_projection=model.slow_input_projection,
                input_norm=model.slow_input_norm,
                encoder=model.slow_encoder,
                return_sequence=True,
                temporal_attention=cell["temporal_encoder"] == "attention",
            )
            state = model.temporal_peer(
                sequence, data["slow_history_mask"], data["active_mask"]
            )
            return head(state).squeeze(-1)

        started = time.perf_counter()
        for step in range(steps):
            data, labels = sample(1000 + step, dates=8)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = (predict(data) - labels).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            scores = predict(validation)
            ranks = scores.argsort(-1).argsort(-1).float()
            x = ranks - ranks.mean(-1, keepdim=True)
            target_ranks = target.argsort(-1).argsort(-1).float()
            y = target_ranks - target_ranks.mean(-1, keepdim=True)
            ic = (
                (x * y).sum(-1) / (x.square().sum(-1) * y.square().sum(-1)).sqrt()
            ).mean()
        result = {
            "cell": cell["cell"],
            "steps": steps,
            "heldout_synthetic_ic": float(ic),
            "exceeds_reference_threshold": float(ic) > 0.5,
            "seconds": time.perf_counter() - started,
        }
        results.append(result)
        write_json_atomic(
            output,
            {
                "synthetic_only": True,
                "advancement_weight": 0,
                "fresh_training_dates_per_update": 8,
                "unseen_validation_dates": 48,
                "steps_fixed_in_advance": steps,
                "scope": "actual full-window temporal and peer components plus a linear diagnostic head; not the B4 trunk",
                "synthetic_loss": "continuous teacher MSE; AdamW .001 decay .01 clip1, dropout .1, fixed800 updates",
                "qualification": "full-model SAM peer-task failures are retained separately; this establishes representation learnability only",
                "results": results,
            },
        )
        print(result, flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
