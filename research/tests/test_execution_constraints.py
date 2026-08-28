from __future__ import annotations

import pytest
import torch

from brazil_rv.execution.constraints import project_weights, project_weights_bounded


def test_projection_is_exact_neutral_capped_and_sign_preserving() -> None:
    raw = torch.tensor(
        [[3.0, 1.0, -2.0, -2.0], [1.5, 0.4, -0.6, -1.7]],
        dtype=torch.float64,
    )
    caps = torch.tensor(
        [[0.35, 0.35, 0.35, 0.35], [0.50, 0.35, 0.40, 0.45]],
        dtype=torch.float64,
    )
    result = project_weights(raw, torch.ones_like(raw, dtype=torch.bool), caps, 1.2)

    torch.testing.assert_close(
        result.sum(dim=-1), torch.zeros(2, dtype=torch.float64), atol=1e-10, rtol=0
    )
    torch.testing.assert_close(
        result.abs().sum(dim=-1),
        torch.full((2,), 1.2, dtype=torch.float64),
        atol=1e-10,
        rtol=0,
    )
    assert (result.abs() <= caps + 1e-12).all()
    assert ((result == 0) | (torch.sign(result) == torch.sign(raw))).all()


def test_projection_is_exact_identity_for_feasible_input() -> None:
    raw = torch.tensor([0.3, 0.3, -0.2, -0.4], dtype=torch.float64)
    result = project_weights(raw, torch.ones(4, dtype=torch.bool), 0.5, 1.2)
    assert torch.equal(result, raw)


@pytest.mark.parametrize(
    ("raw", "caps"),
    (
        ([1.0, 0.5, 0.0], [1.0, 1.0, 1.0]),
        ([1.0, 0.5, -1.0], [0.2, 0.2, 1.0]),
    ),
    ids=("missing-short-side", "insufficient-long-cap"),
)
def test_projection_rejects_infeasible_signed_capacity(
    raw: list[float], caps: list[float]
) -> None:
    with pytest.raises(ValueError, match="infeasible"):
        project_weights(
            torch.tensor(raw, dtype=torch.float64),
            torch.ones(len(raw), dtype=torch.bool),
            torch.tensor(caps, dtype=torch.float64),
            1.0,
        )


def test_projection_passes_gradcheck_away_from_kinks() -> None:
    raw = torch.tensor([1.4, 1.0, -1.1, -1.3], dtype=torch.float64, requires_grad=True)
    mask = torch.ones(4, dtype=torch.bool)
    caps = torch.ones(4, dtype=torch.float64)

    assert torch.autograd.gradcheck(
        lambda value: project_weights(value, mask, caps, 1.4),
        (raw,),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-3,
    )


def test_projection_gradient_is_finite_when_every_name_is_cap_saturated() -> None:
    raw = torch.tensor([2.0, 1.0, -1.0, -2.0], dtype=torch.float64, requires_grad=True)
    result = project_weights(raw, torch.ones(4, dtype=torch.bool), 0.25, 1.0)
    loss = result @ torch.arange(1.0, 5.0, dtype=torch.float64)
    loss.backward()

    torch.testing.assert_close(
        result, torch.tensor([0.25, 0.25, -0.25, -0.25], dtype=torch.float64)
    )
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()


def test_masked_names_have_zero_output_and_zero_gradient() -> None:
    raw = torch.tensor(
        [1.4, 999.0, 0.8, -1.3, -0.7, -999.0],
        dtype=torch.float64,
        requires_grad=True,
    )
    mask = torch.tensor([True, False, True, True, True, False])
    result = project_weights(raw, mask, 1.0, 1.6)
    loss = result @ torch.arange(1.0, 7.0, dtype=torch.float64)
    (gradient,) = torch.autograd.grad(loss, raw)

    assert torch.equal(result[~mask], torch.zeros(2, dtype=torch.float64))
    assert torch.equal(gradient[~mask], torch.zeros(2, dtype=torch.float64))
    assert gradient[mask].abs().sum() > 0


def test_bounded_projection_is_neutral_capped_and_never_scales_up() -> None:
    raw = torch.tensor(
        [[0.8, 0.2, -0.4, -0.1], [0.1, 0.2, -0.15, -0.15]],
        dtype=torch.float64,
    )
    caps = torch.tensor(
        [[0.5, 0.5, 0.3, 0.3], [0.5, 0.5, 0.5, 0.5]],
        dtype=torch.float64,
    )
    result = project_weights_bounded(
        raw, torch.ones_like(raw, dtype=torch.bool), caps, 0.8
    )

    torch.testing.assert_close(
        result.sum(dim=-1), torch.zeros(2, dtype=torch.float64), atol=1e-10, rtol=0
    )
    assert (result.abs() <= caps + 1e-12).all()
    assert (result.abs().sum(dim=-1) <= 0.8 + 1e-12).all()
    assert result[0].abs().sum() <= raw[0].clamp(-caps[0], caps[0]).abs().sum()
    torch.testing.assert_close(result[1], raw[1])


def test_bounded_projection_zero_and_uniform_shrink_contract() -> None:
    mask = torch.ones(4, dtype=torch.bool)
    raw = torch.tensor([0.8, 0.4, -0.6, -0.3], dtype=torch.float64)
    full = project_weights_bounded(raw, mask, 1.0, 2.0)
    shrunk = project_weights_bounded(raw * 0.25, mask, 1.0, 2.0)
    zero = project_weights_bounded(torch.zeros_like(raw), mask, 1.0, 2.0)

    assert shrunk.abs().sum() <= full.abs().sum()
    assert torch.equal(zero, torch.zeros_like(zero))


def test_bounded_projection_passes_gradcheck_away_from_kinks() -> None:
    raw = torch.tensor([0.4, 0.2, -0.3, -0.1], dtype=torch.float64, requires_grad=True)
    mask = torch.ones(4, dtype=torch.bool)
    assert torch.autograd.gradcheck(
        lambda value: project_weights_bounded(value, mask, 1.0, 2.0),
        (raw,),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-3,
    )


def test_bounded_projection_masks_output_and_gradient_at_zero_kink() -> None:
    raw = torch.zeros(5, dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([True, True, False, True, True])
    result = project_weights_bounded(raw, mask, 0.5, 1.0)
    gradient = torch.autograd.grad(
        result @ torch.arange(1.0, 6.0, dtype=torch.float64), raw
    )[0]

    assert torch.equal(result, torch.zeros_like(result))
    assert gradient[mask].abs().sum() > 0
    assert torch.equal(gradient[~mask], torch.zeros(1, dtype=torch.float64))


def test_bounded_projection_zero_kink_has_finite_float32_gradient() -> None:
    raw = torch.zeros(5, dtype=torch.float32, requires_grad=True)
    mask = torch.tensor([True, True, False, True, True])
    result = project_weights_bounded(raw, mask, 0.5, 1.0)
    gradient = torch.autograd.grad(
        result @ torch.arange(1.0, 6.0, dtype=torch.float32), raw
    )[0]

    assert torch.isfinite(gradient).all()
    assert gradient[mask].abs().sum() > 0
    assert torch.equal(gradient[~mask], torch.zeros(1, dtype=torch.float32))
