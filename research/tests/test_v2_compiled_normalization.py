"""Compiler temporaries cannot become inputs to residual normalization."""

import copy

import pytest
import torch
from torch import nn

from brazil_rv.v2.train import compile_forward, set_deterministic_seed


class ResidualNormalization(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(96, 96)
        self.norm = nn.LayerNorm(96)

    def forward(self, residual, values, valid):
        return torch.where(
            valid[..., None], self.norm(residual + self.linear(values)), 0.0
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA compiler regression")
def test_compiled_residual_normalization_ignores_allocator_contents(monkeypatch):
    from torch._inductor.codecache import PyCodeCache

    set_deterministic_seed(13)
    reference = ResidualNormalization().cuda().eval()
    model = copy.deepcopy(reference)
    residual = torch.randn(120, 256, 96, device="cuda")
    values = torch.randn_like(residual)
    valid = torch.rand(120, 256, device="cuda") > 0.15
    weights = torch.randn_like(residual)
    compiled = compile_forward(model)

    def evaluate(instance, backward=False):
        instance.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            prediction = instance(residual, values, valid)
            loss = (prediction * weights).mean()
        if backward:
            loss.backward()
        return prediction.detach().clone()

    expected = evaluate(reference, backward=True)
    actual = evaluate(compiled, backward=True)
    torch.testing.assert_close(actual, expected, atol=3e-3, rtol=2e-2)
    for source, target in zip(reference.parameters(), model.parameters(), strict=True):
        torch.testing.assert_close(target.grad, source.grad, atol=1e-6, rtol=2e-2)

    # Capture inference separately from AOTAutograd before changing allocations.
    with torch.no_grad():
        evaluate(compiled)
    modules = [m for m in PyCodeCache.modules if hasattr(m, "empty_strided_cuda")]
    assert modules
    for fill in (float("nan"), 0.0, 1.0):
        with monkeypatch.context() as patch:
            for module in modules:
                original = module.empty_strided_cuda

                def initialized(*args, _original=original, _fill=fill, **kwargs):
                    tensor = _original(*args, **kwargs)
                    if tensor.is_floating_point():
                        tensor.fill_(_fill)
                    return tensor

                patch.setattr(module, "empty_strided_cuda", initialized)
            with torch.no_grad():
                actual = evaluate(compiled)
            torch.testing.assert_close(actual, expected, atol=3e-3, rtol=2e-2)
            assert torch.isfinite(actual).all()
            assert not actual[~valid].any()
