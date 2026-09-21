"""Bounded old/new eager and compiled forward/backward check for C6 heads."""

import importlib.util
import json
from pathlib import Path
from time import perf_counter

import torch

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.config import ModelConfig
from brazil_rv.v2.data_repair import binding
from brazil_rv.v2.model import DailyMultiHorizonModel
from brazil_rv.v2.train import compile_forward

PROJECT = Path(__file__).resolve().parents[1]


def main():
    tick = perf_counter()
    torch.set_num_threads(1)
    run = json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())
    out = Path(run["stage_c_refit_root"]) / "c6_compile"
    spec = importlib.util.spec_from_file_location(
        "brazil_rv.v2.executed_model", out / "failed_model.py"
    )
    prior = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prior)
    config = ModelConfig(
        slow_feature_count=32,
        current_feature_count=0,
        disable_fast_stream=True,
        use_bf16=True,
        dropout=0,
    )
    torch.manual_seed(11)
    old = prior.DailyMultiHorizonModel(config).cuda()
    new = DailyMultiHorizonModel(config).cuda()
    new.load_state_dict(old.state_dict())
    assert old.state_dict().keys() == new.state_dict().keys()
    x = torch.randn(1, 933, 60, 32, device="cuda")
    valid = torch.rand_like(x) > 0.1
    active = torch.rand(1, 933, device="cuda") > 0.1
    args = (x, valid, valid.any(-1), active)
    kwargs = dict(slow_feature_age_sessions=torch.where(valid, 0.0, -1.0))

    def evaluate(model):
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            scores = model(*args, **kwargs)
            loss = scores.float().square().sum()
        loss.backward()
        return scores.detach(), {
            k: p.grad.detach().clone()
            for k, p in model.named_parameters()
            if p.grad is not None
        }

    expected, old_grad = evaluate(old)
    actual, new_grad = evaluate(new)
    assert torch.equal(expected, actual)
    assert old_grad.keys() == new_grad.keys()
    assert all(torch.equal(old_grad[k], new_grad[k]) for k in old_grad)
    compiled = compile_forward(new)
    compiled_scores, _ = evaluate(compiled)
    torch.testing.assert_close(compiled_scores, actual)
    # _orig_mod names belong to the wrapper; compare the same original parameters.
    for k, p in new.named_parameters():
        if k in new_grad:
            torch.testing.assert_close(p.grad, new_grad[k], atol=1e-5, rtol=0.02)
    torch.cuda.synchronize()
    report = dict(
        passed=True,
        graph_parameters=sum(p.numel() for p in new.parameters()),
        names=933,
        history=60,
        precision="original BF16 autocast",
        original_eager_values_and_gradients_exact=True,
        compiled_forward_and_backward=True,
        maximum_compiled_score_error=float(
            (compiled_scores.float() - actual.float()).abs().max()
        ),
        seconds=perf_counter() - tick,
        old=binding(out / "failed_model.py"),
        current=binding(PROJECT / "research/src/brazil_rv/v2/model.py"),
        recipe=binding(Path(__file__)),
        scope="Synthetic full-population compile proof, no historical observations or fit/selector/recipe change; dropout disabled for deterministic arithmetic comparison.",
    )
    write_json_atomic(out / "qualification.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
