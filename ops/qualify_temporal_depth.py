"""Existing one-layer graph identity after preparing the registered depth choices."""

from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys
from time import perf_counter

import torch

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.characteristic_model import CharacteristicConfig, CharacteristicModel
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location("brazil_rv.v2." + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    tick = perf_counter()
    torch.set_num_threads(1)
    run = json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())
    root = Path(run["root"]) / "capacity_depth_engineering"
    plan = bound_json(binding(root / "plan.json"))
    previous = plan["previous_runtime"]
    old_temporal = load(
        "_depth_control_temporal", previous["temporal_pathway.py"]["path"]
    )
    old = load(
        "_depth_control_characteristic", previous["characteristic_model.py"]["path"]
    )
    for name in ("HistoricalAttention", "PeerAttention", "TemporalPeerPathway"):
        setattr(old, name, getattr(old_temporal, name))
    torch.manual_seed(37)
    history = torch.ones(2, 7, 60, dtype=torch.bool)
    history[:, 0] = False
    history[:, 1, :23] = False
    valid = history[..., None].expand(-1, -1, -1, 32).clone()
    valid[:, 2, 30:33] = False
    sample = dict(
        slow_features=torch.randn(2, 7, 60, 32),
        slow_feature_mask=valid,
        slow_history_mask=history,
        slow_feature_age_sessions=torch.zeros(2, 7, 60, 32),
        active_mask=torch.ones(2, 7, dtype=torch.bool),
    )
    checks = []
    for encoder in ("attention", "gru"):
        config = CharacteristicConfig(temporal_encoder=encoder, peer_timing="early")
        torch.manual_seed(11)
        control = old.CharacteristicModel(old.CharacteristicConfig(**asdict(config)))
        torch.manual_seed(11)
        candidate = CharacteristicModel(config)
        left, right = control.state_dict(), candidate.state_dict()
        assert left.keys() == right.keys()
        for name in left:
            torch.testing.assert_close(left[name], right[name], atol=0, rtol=0)
        for training in (False, True):
            control.train(training)
            candidate.train(training)
            torch.manual_seed(91)
            expected = control(**sample)
            torch.manual_seed(91)
            actual = candidate(**sample)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            expected.square().mean().backward()
            actual.square().mean().backward()
            for (name, p), (other, q) in zip(
                control.named_parameters(), candidate.named_parameters(), strict=True
            ):
                assert name == other
                torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)
            control.zero_grad(set_to_none=True)
            candidate.zero_grad(set_to_none=True)
        checks.append(
            dict(
                encoder=encoder,
                state_cells=sum(x.numel() for x in left.values()),
                output_cells=actual.numel(),
                gradient_cells=sum(p.numel() for p in candidate.parameters()),
                modes=["evaluation", "training_with_matched_dropout_rng"],
                exact=True,
            )
        )
    report = dict(
        passed=True,
        plan=binding(root / "plan.json"),
        checks=checks,
        current_runtime={
            name: binding(PROJECT / "research/src/brazil_rv/v2" / name)
            for name in previous
        },
        seconds=perf_counter() - tick,
        scope="Two existing one-layer graphs on two synthetic dates/seven names/full60; exact initialization/state keys, output and all gradients in evaluation and training modes. This is implementation identity, not a new source/store/model outcome or authorization to start a depth fit. New two-layer masking/causality/fullgraph behavior is tested separately.",
    )
    write_json_atomic(root / "identity.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
