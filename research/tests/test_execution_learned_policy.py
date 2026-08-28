from __future__ import annotations

import copy
import math
from dataclasses import replace

import pytest
import torch

from brazil_rv.execution.config import ExecutionConfig
from brazil_rv.execution.features import PolicyState
from brazil_rv.execution.policy import BandPolicy, NeuralPolicy
from brazil_rv.execution.simulator import MarketReplay, simulate
from brazil_rv.execution.trainer import (
    PolicyBatch,
    PolicyTrainer,
    PolicyTrainerConfig,
    policy_objective,
)


def _config() -> ExecutionConfig:
    return ExecutionConfig(
        nav_brl=1_000_000.0,
        gross_target=1.0,
        participation_rate=1.0,
        name_cap_fraction_of_gross=0.5,
        adv_cap_fraction=1.0,
        fee_bps=0.5,
        max_spread_bps=100.0,
        min_adv_brl=1.0,
        taper_minutes=1,
        horizon_blend=(1 / 3, 1 / 3, 1 / 3),
    )


def _batch(*, edge: bool, daily_cdi: float = 0.0) -> PolicyBatch:
    dtype = torch.float64
    days, minutes, names = 8, 4, 4
    base_rank = torch.tensor([-1.0, -0.35, 0.35, 1.0], dtype=dtype)
    price = torch.full((days, minutes, names), 10.0, dtype=dtype)
    if edge:
        price[:, 2:] = 10 * (1 + 0.02 * base_rank)
    market = MarketReplay(
        open_price=price,
        open_observed=torch.ones_like(price, dtype=torch.bool),
        active=torch.ones((days, names), dtype=torch.bool),
        full_spread=torch.full((days, minutes, names), 0.0002, dtype=dtype),
        adv20_brl=torch.full((days, names), 10_000_000.0, dtype=dtype),
        minute_notional20_brl=torch.full_like(price, 10_000_000.0),
        daily_cdi_rate=torch.full((days,), daily_cdi, dtype=dtype),
    )
    ranks = base_rank[None, None, :, None].expand(days, minutes, names, 3).clone()
    if not edge:
        generator = torch.Generator().manual_seed(991)
        half_ranks = (
            torch.rand((days // 2, 1, names, 3), generator=generator, dtype=dtype) * 2
            - 1
        )
        ranks = (
            torch.cat((half_ranks, half_ranks), dim=0)
            .expand(days, minutes, names, 3)
            .clone()
        )
        noise_return = (
            torch.rand((days // 2, names), generator=generator, dtype=dtype) - 0.5
        ) * 0.01
        price[: days // 2, 2:] = 10 * (1 + noise_return[:, None])
        price[days // 2 :, 2:] = 10 * (1 - noise_return[:, None])
    valid = torch.ones_like(ranks, dtype=torch.bool)
    refresh = torch.zeros((days, minutes), dtype=torch.bool)
    refresh[:, 0] = True
    return PolicyBatch(
        market,
        ranks,
        valid,
        refresh,
        torch.full((days, names), 0.02, dtype=dtype),
    )


def _dummy_state(mask: torch.Tensor, *, masked_value: float = 0.0) -> PolicyState:
    days, names = mask.shape
    per_name = torch.zeros((days, names, 16), dtype=torch.float64)
    per_name[..., 0] = torch.linspace(-1, 1, names, dtype=torch.float64)
    per_name[:, ~mask[0], 0] = masked_value
    return PolicyState(
        per_name=per_name,
        portfolio=torch.zeros((days, 3), dtype=torch.float64),
        time=torch.zeros((days, 2), dtype=torch.float64),
        tradeable_mask=mask,
        cap_weights=torch.full((days, names), 0.5, dtype=torch.float64),
    )


def test_band_policy_is_bit_identical_with_optional_policy_state() -> None:
    config = _config()
    policy = BandPolicy(config).double()
    ranks = torch.linspace(-1, 1, 4, dtype=torch.float64)[None, :, None].expand(1, 4, 3)
    shape = ranks.shape[:2]
    args = (
        ranks,
        torch.tensor([True]),
        torch.zeros(shape, dtype=torch.float64),
        torch.full(shape, 0.02, dtype=torch.float64),
        torch.zeros(shape, dtype=torch.float64),
        torch.tensor([False]),
        torch.ones(shape, dtype=torch.bool),
        torch.full(shape, 0.5, dtype=torch.float64),
        torch.zeros(shape, dtype=torch.float64),
    )
    without_state = policy.step(*args)
    with_state = policy.step(*args, _dummy_state(torch.ones(shape, dtype=torch.bool)))

    assert torch.equal(without_state[0], with_state[0])
    assert torch.equal(without_state[1], with_state[1])


def test_neural_policy_zero_init_seed_and_mask_contract() -> None:
    config = _config()
    left = NeuralPolicy(config, seed=17).double()
    right = NeuralPolicy(config, seed=17).double()
    other = NeuralPolicy(config, seed=18).double()
    assert all(
        torch.equal(a, b)
        for a, b in zip(
            left.state_dict().values(), right.state_dict().values(), strict=True
        )
    )
    assert any(
        not torch.equal(a, b)
        for a, b in zip(
            left.trunk.state_dict().values(),
            other.trunk.state_dict().values(),
            strict=True,
        )
    )

    mask = torch.tensor([[True, True, False, True]])
    ranks = torch.zeros((1, 4, 3), dtype=torch.float64)
    state = _dummy_state(mask)
    target, initialized = left.step(
        ranks,
        torch.tensor([True]),
        torch.zeros((1, 4), dtype=torch.float64),
        torch.ones((1, 4), dtype=torch.float64),
        torch.zeros((1, 4), dtype=torch.float64),
        torch.tensor([False]),
        mask,
        state.cap_weights,
        torch.zeros((1, 4), dtype=torch.float64),
        state,
    )
    assert torch.equal(target, torch.zeros_like(target))
    assert initialized.item()

    loss = target @ torch.arange(1.0, 5.0, dtype=torch.float64)
    loss.sum().backward()
    reference_gradient = left.output.weight.grad.clone()
    changed = NeuralPolicy(config, seed=17).double()
    changed_state = _dummy_state(mask, masked_value=1e6)
    changed_target, _ = changed.step(
        ranks,
        torch.tensor([True]),
        torch.zeros((1, 4), dtype=torch.float64),
        torch.ones((1, 4), dtype=torch.float64),
        torch.zeros((1, 4), dtype=torch.float64),
        torch.tensor([False]),
        mask,
        changed_state.cap_weights,
        torch.zeros((1, 4), dtype=torch.float64),
        changed_state,
    )
    (changed_target @ torch.arange(1.0, 5.0, dtype=torch.float64)).sum().backward()
    torch.testing.assert_close(
        changed.output.weight.grad, reference_gradient, rtol=0, atol=0
    )


def test_neural_policy_cannot_act_before_first_valid_refresh() -> None:
    config = _config()
    policy = NeuralPolicy(config, seed=13).double()
    with torch.no_grad():
        policy.output.weight.copy_(
            torch.linspace(-0.2, 0.2, policy.output.weight.numel()).reshape_as(
                policy.output.weight
            )
        )
    mask = torch.ones((1, 4), dtype=torch.bool)
    state = _dummy_state(mask)
    target, initialized = policy.step(
        torch.zeros((1, 4, 3), dtype=torch.float64),
        torch.tensor([False]),
        torch.zeros((1, 4), dtype=torch.float64),
        torch.ones((1, 4), dtype=torch.float64),
        torch.zeros((1, 4), dtype=torch.float64),
        torch.tensor([False]),
        mask,
        state.cap_weights,
        torch.zeros((1, 4), dtype=torch.float64),
        state,
    )
    assert not initialized.item()
    assert torch.equal(target, torch.zeros_like(target))

    per_name_refresh = torch.tensor([[True, False, False, False]])
    only_other_name_tradeable = torch.tensor([[False, True, False, False]])
    other_state = _dummy_state(only_other_name_tradeable)
    target, initialized = policy.step(
        torch.zeros((1, 4, 3), dtype=torch.float64),
        per_name_refresh,
        torch.zeros((1, 4), dtype=torch.float64),
        torch.ones((1, 4), dtype=torch.float64),
        torch.zeros((1, 4), dtype=torch.float64),
        torch.tensor([False]),
        only_other_name_tradeable,
        other_state.cap_weights,
        torch.zeros((1, 4), dtype=torch.float64),
        other_state,
    )
    assert not initialized.item()
    assert torch.equal(target, torch.zeros_like(target))


def test_neural_policy_supports_semantically_bound_fourth_horizon() -> None:
    with pytest.raises(ValueError, match="differs from the execution blend"):
        NeuralPolicy(_config(), horizon_count=4)

    config = replace(_config(), horizon_blend=(0.25, 0.25, 0.25, 0.25))
    base = _batch(edge=False)
    ranks = torch.cat((base.ranks, base.ranks[..., :1]), dim=-1)
    policy = NeuralPolicy(
        config,
        horizon_names=("30m", "60m", "120m", "to_close"),
        seed=19,
    ).double()
    result = simulate(
        base.market,
        ranks,
        torch.ones_like(ranks, dtype=torch.bool),
        base.refresh_mask,
        base.sigma,
        policy,
        config,
    )

    assert policy.trunk[0].normalized_shape == (23,)
    assert policy.contract_metadata["horizon_names"][-1] == "to_close"
    assert torch.equal(result.turnover_brl, torch.zeros_like(result.turnover_brl))


def test_neural_policy_rejects_same_width_reordered_horizon_state() -> None:
    policy = NeuralPolicy(_config(), seed=19).double()
    mask = torch.ones((1, 4), dtype=torch.bool)
    original = _dummy_state(mask)
    reordered = PolicyState(
        per_name=original.per_name,
        portfolio=original.portfolio,
        time=original.time,
        tradeable_mask=original.tradeable_mask,
        cap_weights=original.cap_weights,
        horizon_names=("60m", "30m", "120m"),
    )
    with pytest.raises(ValueError, match="ordered horizons"):
        policy.step(
            torch.zeros((1, 4, 3), dtype=torch.float64),
            torch.tensor([True]),
            torch.zeros((1, 4), dtype=torch.float64),
            torch.ones((1, 4), dtype=torch.float64),
            torch.zeros((1, 4), dtype=torch.float64),
            torch.tensor([False]),
            mask,
            original.cap_weights,
            torch.zeros((1, 4), dtype=torch.float64),
            reordered,
        )


def test_untrained_neural_policy_is_all_cash_and_earns_exact_cdi() -> None:
    batch = _batch(edge=False, daily_cdi=0.001)
    config = _config()
    policy = NeuralPolicy(config, seed=23).double()
    result = simulate(
        batch.market,
        batch.ranks,
        batch.rank_valid,
        batch.refresh_mask,
        batch.sigma,
        policy,
        config,
    )

    assert torch.equal(result.turnover_brl, torch.zeros_like(result.turnover_brl))
    assert torch.equal(
        result.mean_deployed_gross_brl, torch.zeros_like(result.mean_deployed_gross_brl)
    )
    torch.testing.assert_close(
        result.net_pnl_brl,
        torch.full((8,), config.nav_brl * 0.001, dtype=torch.float64),
        rtol=0,
        atol=1e-8,
    )


def test_trainer_learns_planted_edge_and_noise_stays_all_cash() -> None:
    config = _config()
    edge_batch = _batch(edge=True)
    policy = NeuralPolicy(config, seed=31).double()
    trainer = PolicyTrainer(
        policy,
        config,
        PolicyTrainerConfig(learning_rate=5e-3, seed=31),
    )
    initial_result = simulate(
        edge_batch.market,
        edge_batch.ranks,
        edge_batch.rank_valid,
        edge_batch.refresh_mask,
        edge_batch.sigma,
        policy,
        config,
    )
    initial_objective = policy_objective(
        initial_result.net_pnl_brl, edge_batch.market.daily_cdi_rate, config.nav_brl, 0
    )[0]
    trainer.train_step(edge_batch)
    final_result = simulate(
        edge_batch.market,
        edge_batch.ranks,
        edge_batch.rank_valid,
        edge_batch.refresh_mask,
        edge_batch.sigma,
        policy,
        config,
    )
    final_objective = policy_objective(
        final_result.net_pnl_brl, edge_batch.market.daily_cdi_rate, config.nav_brl, 0
    )[0]
    assert final_objective > initial_objective
    assert final_result.mean_deployed_gross_brl.mean() > 0

    noise_batch = _batch(edge=False, daily_cdi=0.0001)
    cash_policy = NeuralPolicy(config, seed=37).double()
    cash_trainer = PolicyTrainer(
        cash_policy,
        config,
        PolicyTrainerConfig(learning_rate=5e-3, seed=37),
    )
    for _ in range(4):
        cash_trainer.train_step(noise_batch)
    cash_result = simulate(
        noise_batch.market,
        noise_batch.ranks,
        noise_batch.rank_valid,
        noise_batch.refresh_mask,
        noise_batch.sigma,
        cash_policy,
        config,
    )
    assert torch.equal(
        cash_result.turnover_brl, torch.zeros_like(cash_result.turnover_brl)
    )


def test_policy_objective_gradient_includes_fill_costs() -> None:
    base = _batch(edge=True)
    flat_price = torch.full_like(base.market.open_price, 10.0)
    cost_batch = replace(base, market=replace(base.market, open_price=flat_price))
    zero_cost_batch = replace(
        cost_batch,
        market=replace(
            cost_batch.market,
            full_spread=torch.zeros_like(cost_batch.market.full_spread),
        ),
    )

    def objective_gradient(
        config: ExecutionConfig, batch: PolicyBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        policy = NeuralPolicy(config, seed=53).double()
        with torch.no_grad():
            policy.trunk[1].weight.zero_()
            policy.trunk[1].bias.zero_()
            policy.trunk[1].weight[0, 0] = 1
            policy.trunk[1].weight[1, 0] = -1
            policy.trunk[3].weight.zero_()
            policy.trunk[3].bias.zero_()
            policy.trunk[3].weight[0, 0] = 1
            policy.trunk[3].weight[0, 1] = -1
            policy.output.weight.zero_()
            policy.output.weight[0, 0] = 0.01
        result = simulate(
            batch.market,
            batch.ranks,
            batch.rank_valid,
            batch.refresh_mask,
            batch.sigma,
            policy,
            config,
        )
        objective = policy_objective(
            result.net_pnl_brl,
            batch.market.daily_cdi_rate,
            config.nav_brl,
            0,
        )[0]
        (-objective).backward()
        return result.turnover_brl, policy.output.weight.grad.detach().clone()

    cost_turnover, cost_gradient = objective_gradient(_config(), cost_batch)
    zero_turnover, zero_gradient = objective_gradient(
        replace(_config(), fee_bps=0.0), zero_cost_batch
    )

    assert cost_turnover.sum() > 0
    assert zero_turnover.sum() > 0
    assert cost_gradient.norm() > 0
    torch.testing.assert_close(zero_gradient, torch.zeros_like(zero_gradient))


def test_training_and_selection_hooks_are_deterministic(tmp_path) -> None:
    config = _config()
    batch = _batch(edge=True)
    trainers: list[PolicyTrainer] = []
    for _ in range(2):
        policy = NeuralPolicy(config, seed=41).double()
        trainer = PolicyTrainer(
            policy,
            config,
            PolicyTrainerConfig(learning_rate=3e-3, seed=41, patience=2),
        )
        for _ in range(3):
            trainer.train_step(batch)
        trainers.append(trainer)
    assert all(
        torch.equal(a, b)
        for a, b in zip(
            trainers[0].policy.state_dict().values(),
            trainers[1].policy.state_dict().values(),
            strict=True,
        )
    )
    trainer = trainers[0]
    assert not trainer.update_monitor(1.0)
    selected = copy.deepcopy(trainer.policy.state_dict())
    with torch.no_grad():
        next(trainer.policy.parameters()).add_(1)
    trainer.restore_best()
    assert all(
        torch.equal(value, selected[name])
        for name, value in trainer.policy.state_dict().items()
    )
    checkpoint_path = tmp_path / "policy.pt"
    trainer.save_checkpoint(checkpoint_path)
    restored = PolicyTrainer(
        NeuralPolicy(config, seed=41).double(),
        config,
        PolicyTrainerConfig(learning_rate=3e-3, seed=41, patience=2),
    )
    restored.load_checkpoint(checkpoint_path)
    assert all(
        torch.equal(a, b)
        for a, b in zip(
            trainer.policy.state_dict().values(),
            restored.policy.state_dict().values(),
            strict=True,
        )
    )
    semantic_mismatch = PolicyTrainer(
        NeuralPolicy(
            config,
            horizon_names=("60m", "30m", "120m"),
            seed=41,
        ).double(),
        config,
        PolicyTrainerConfig(learning_rate=3e-3, seed=41, patience=2),
    )
    with pytest.raises(ValueError, match="checkpoint contract"):
        semantic_mismatch.load_checkpoint(checkpoint_path)


def test_policy_objective_is_exact_bps_mean_minus_population_standard_deviation() -> (
    None
):
    net = torch.tensor([1_100.0, 900.0, 1_300.0], dtype=torch.float64)
    cdi = torch.tensor([0.0001, 0.0001, 0.0001], dtype=torch.float64)

    objective, excess, standard_deviation = policy_objective(
        net, cdi, 1_000_000.0, 0.10
    )

    expected_excess = (net - 100.0) / 100.0
    expected_net_bps = net / 100.0
    torch.testing.assert_close(excess, expected_excess)
    torch.testing.assert_close(standard_deviation, expected_net_bps.std(correction=0))
    torch.testing.assert_close(
        objective,
        expected_excess.mean() - 0.10 * expected_net_bps.std(correction=0),
    )


def test_sam_and_scan_checkpointing_are_available_but_opt_in() -> None:
    config = _config()
    trainer = PolicyTrainer(
        NeuralPolicy(config, seed=47).double(),
        config,
        PolicyTrainerConfig(seed=47, use_sam=True, gradient_checkpointing=True),
    )
    metrics = trainer.train_step(_batch(edge=True))
    assert math.isfinite(metrics.gradient_norm)
