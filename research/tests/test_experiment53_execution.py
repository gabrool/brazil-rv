from __future__ import annotations

import torch

from brazil_rv.execution.config import ExecutionConfig
from brazil_rv.execution.experiment53 import (
    FOLDS,
    VARIANTS,
    execution_cells,
    liquidity_terciles,
    rotation_designation,
)
from brazil_rv.execution.policy import ConcentratedPolicy
from brazil_rv.execution.simulator import MarketReplay, tradeable_universe


def _policy_state(names: int) -> tuple[torch.Tensor, ...]:
    zeros = torch.zeros((1, names), dtype=torch.float64)
    return (
        zeros,
        torch.full_like(zeros, 0.02),
        zeros.clone(),
        torch.zeros(1, dtype=torch.bool),
    )


def _ranks(values: list[float]) -> torch.Tensor:
    first = torch.tensor(values, dtype=torch.float64)
    return torch.stack((first, first, first), dim=-1)[None]


def test_capacity_completion_extends_both_sides_deterministically() -> None:
    config = ExecutionConfig(
        gross_target=1.0,
        concentration_k=2,
        name_cap_fraction_of_gross=0.05,
        horizon_blend=(1.0, 0.0, 0.0),
        band=0.0,
    )
    policy = ConcentratedPolicy(config)
    current, sigma, previous, initialized = _policy_state(8)

    target, initialized = policy.step(
        _ranks(list(range(8))),
        torch.tensor([True]),
        current,
        sigma,
        previous,
        initialized,
        torch.ones((1, 8), dtype=torch.bool),
        torch.full((1, 8), 0.15, dtype=torch.float64),
        torch.zeros((1, 8), dtype=torch.float64),
    )

    assert initialized.item()
    assert policy.last_selection_extended_count is not None
    assert policy.last_selection_extended_count.item() == 4
    assert torch.count_nonzero(target > 0).item() == 4
    assert torch.count_nonzero(target < 0).item() == 4


def test_selection_exit_hysteresis_retains_only_inside_k_exit() -> None:
    config = ExecutionConfig(
        gross_target=1.0,
        concentration_k=2,
        horizon_blend=(1.0, 0.0, 0.0),
        band=0.0,
    )
    policy = ConcentratedPolicy(config)
    current, sigma, previous, initialized = _policy_state(8)
    common = (
        torch.tensor([True]),
        current,
        sigma,
    )
    masks = (
        torch.ones((1, 8), dtype=torch.bool),
        torch.ones((1, 8), dtype=torch.float64),
        torch.zeros((1, 8), dtype=torch.float64),
    )
    first, initialized = policy.step(
        _ranks([0, 1, 2, 3, 4, 5, 6, 7]),
        *common,
        previous,
        initialized,
        *masks,
    )
    assert first[0, 6] > 0

    inside, initialized = policy.step(
        _ranks([0, 1, 2, 3, 4, 6, 5, 7]),
        *common,
        first,
        initialized,
        *masks,
    )
    assert inside[0, 6] > 0

    outside, _ = policy.step(
        _ranks([0, 1, 2, 3, 6, 5, 4, 7]),
        *common,
        inside,
        initialized,
        *masks,
    )
    assert outside[0, 6] == 0


def test_cost_scaled_band_is_applied_per_name() -> None:
    config = ExecutionConfig(
        gross_target=1.0,
        concentration_k=1,
        horizon_blend=(1.0, 0.0, 0.0),
        band=0.5,
        cost_band_scale=1.0,
    )
    policy = ConcentratedPolicy(config)
    current = torch.tensor([[-0.485, 0.0, 0.0, 0.485]], dtype=torch.float64)
    sigma = torch.full((1, 4), 0.02, dtype=torch.float64)
    previous = torch.tensor([[-0.1, 0.0, 0.0, 0.1]], dtype=torch.float64)

    target, _ = policy.step(
        _ranks([0, 1, 2, 3]),
        torch.tensor([True]),
        current,
        sigma,
        previous,
        torch.tensor([True]),
        torch.ones((1, 4), dtype=torch.bool),
        torch.ones((1, 4), dtype=torch.float64),
        torch.tensor([[0.0, 0.0, 0.0, 0.01]], dtype=torch.float64),
    )

    assert target[0, 0] == -0.5
    assert target[0, 3] == 0.1


def test_top_half_adv_is_a_deterministic_mask_and_scenario_is_hashed() -> None:
    dtype = torch.float64
    market = MarketReplay(
        open_price=torch.full((1, 2, 4), 10.0, dtype=dtype),
        open_observed=torch.ones((1, 2, 4), dtype=torch.bool),
        active=torch.ones((1, 4), dtype=torch.bool),
        full_spread=torch.zeros((1, 4), dtype=dtype),
        adv20_brl=torch.tensor([[10.0, 20.0, 30.0, 40.0]], dtype=dtype),
        minute_notional20_brl=torch.ones((1, 2, 4), dtype=dtype),
        daily_cdi_rate=torch.zeros(1, dtype=dtype),
    )
    config = ExecutionConfig(min_adv_brl=1.0, top_half_adv=True)

    assert torch.equal(
        tradeable_universe(market, config),
        torch.tensor([[False, False, True, True]]),
    )
    half = ExecutionConfig(spread_schedule_multiplier=0.5)
    assert half.sha256 != ExecutionConfig().sha256
    assert half.to_dict()["spread_schedule_multiplier"] == 0.5


def test_experiment53_grid_is_exact_and_amended_cap_is_local() -> None:
    cells = execution_cells()
    assert len(cells) == 48
    assert set(VARIANTS) == {"measured", "frictionless", "half_spread"}
    assert {int(row["k"]) for row in cells} == {10, 20, 40}
    assert {float(row["gross_target"]) for row in cells} == {1.0, 2.0}
    assert {
        float(row["measured_config"]["name_cap_fraction_of_gross"]) for row in cells
    } == {0.05}
    assert ExecutionConfig().name_cap_fraction_of_gross == 0.025


def test_liquidity_terciles_use_prior_adv_order_only() -> None:
    adv = torch.tensor([[5.0, 1.0, 4.0, 2.0, 3.0]], dtype=torch.float64)
    mask = torch.tensor([[True, True, True, True, False]])
    tiers = liquidity_terciles(adv, mask)
    assert torch.equal(tiers, torch.tensor([[3, 1, 2, 1, 0]], dtype=torch.int8))


def test_c1_rotation_excludes_any_cell_failing_one_fold_guard() -> None:
    rows = []
    for cell, sharpes, eligible in (
        ("eligible", (1.0, 1.0, 1.0), (True, True, True)),
        ("guarded", (10.0, 10.0, 10.0), (True, False, True)),
    ):
        for fold, sharpe, allowed in zip(FOLDS, sharpes, eligible, strict=True):
            rows.append(
                {
                    "cell_id": cell,
                    "fold": fold,
                    "variant": "measured",
                    "annualized_net_sharpe": sharpe,
                    "net_pnl_brl": sharpe,
                    "net_excess_over_all_cash_cdi_brl": sharpe,
                    "c1_fold_eligible": allowed,
                }
            )
    template = list(rows)
    for index in range(46):
        cell = f"other_{index:02d}"
        for fold in FOLDS:
            rows.append(
                {
                    **template[0],
                    "cell_id": cell,
                    "fold": fold,
                    "annualized_net_sharpe": -10.0 - index,
                }
            )

    table, designation = rotation_designation(rows)

    assert len(table) == 144
    assert designation["c1_cell_id"] == "eligible"
    assert "guarded" in designation["ineligible_cell_ids"]
