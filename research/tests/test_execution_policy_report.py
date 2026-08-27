from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import date

import torch

from brazil_rv.execution.config import ExecutionConfig
from brazil_rv.execution.policy import BandPolicy
from brazil_rv.execution.report import (
    DailyExecutionResult,
    execution_report_payload,
    write_execution_report,
)


def _ranks(values: tuple[float, ...]) -> torch.Tensor:
    first = torch.tensor(values, dtype=torch.float64)
    return torch.stack((first, -first, torch.zeros_like(first)), dim=-1)[None]


def _state(name_count: int) -> tuple[torch.Tensor, ...]:
    zeros = torch.zeros((1, name_count), dtype=torch.float64)
    return (
        zeros,
        torch.ones_like(zeros),
        zeros.clone(),
        torch.zeros(1, dtype=torch.bool),
    )


def test_band_policy_obeys_refresh_and_first_build_semantics() -> None:
    config = ExecutionConfig(horizon_blend=(1.0, 0.0, 0.0), band=float("inf"))
    policy = BandPolicy(config)
    ranks = _ranks((-1.0, -0.5, 0.5, 1.0))
    current, sigma, previous, initialized = _state(4)

    built, initialized = policy.step(
        ranks,
        torch.tensor([True]),
        current,
        sigma,
        previous,
        initialized,
    )
    assert initialized.item()
    assert torch.allclose(built.sum(dim=-1), torch.zeros(1, dtype=torch.float64))
    assert torch.allclose(
        built.abs().sum(dim=-1), torch.tensor([2.0], dtype=torch.float64)
    )

    held, initialized = policy.step(
        -ranks,
        torch.tensor([False]),
        current,
        sigma,
        built,
        initialized,
    )
    assert torch.equal(held, built)
    held_after_refresh, _ = policy.step(
        -ranks,
        torch.tensor([True]),
        current,
        sigma,
        built,
        initialized,
    )
    assert torch.equal(held_after_refresh, built)


def test_zero_band_updates_at_arbitrary_refreshes_and_blend_isolates() -> None:
    config = ExecutionConfig(horizon_blend=(1.0, 0.0, 0.0), band=0.0)
    policy = BandPolicy(config)
    ranks = _ranks((-1.0, -0.5, 0.5, 1.0))
    current, sigma, previous, initialized = _state(4)
    target, initialized = policy.step(
        ranks,
        torch.tensor([True]),
        current,
        sigma,
        previous,
        initialized,
    )

    mutated_other_horizons = ranks.clone()
    mutated_other_horizons[..., 1:] = 100_000.0
    held, _ = policy.step(
        mutated_other_horizons,
        torch.tensor([False]),
        current,
        sigma,
        target,
        initialized,
    )
    assert torch.equal(held, target)

    refreshed, _ = policy.step(
        -mutated_other_horizons,
        torch.tensor([True]),
        current,
        sigma,
        target,
        initialized,
    )
    expected, _ = policy.step(
        -ranks,
        torch.tensor([True]),
        current,
        sigma,
        target,
        initialized,
    )
    assert torch.equal(refreshed, expected)
    assert torch.equal(refreshed, -target)


def test_config_and_report_hashes_are_deterministic(tmp_path) -> None:
    config = ExecutionConfig()
    assert config.sha256 == ExecutionConfig().sha256
    assert config.sha256 != replace(config, fee_bps=2.1).sha256
    assert len(ExecutionConfig(band=float("inf")).sha256) == 64

    first = DailyExecutionResult(
        trade_date=date(2024, 1, 3),
        net_pnl_brl=86.0,
        gross_pnl_brl=100.0,
        spread_cost_brl=10.0,
        fees_brl=5.0,
        cdi_earned_brl=1.0,
        turnover_brl=50_000.0,
        max_intraday_gross_brl=20_000_000.0,
        forced_fill_count=1,
    )
    second = DailyExecutionResult(
        trade_date=date(2024, 1, 2),
        net_pnl_brl=-38.0,
        gross_pnl_brl=-30.0,
        spread_cost_brl=5.0,
        fees_brl=4.0,
        cdi_earned_brl=1.0,
        turnover_brl=40_000.0,
        max_intraday_gross_brl=19_000_000.0,
        forced_fill_count=0,
    )
    a_hash = "a" * 64
    b_hash = "b" * 64
    left = execution_report_payload(
        config=config,
        input_sha256={"store": a_hash, "predictions": b_hash},
        daily=(first, second),
    )
    right = execution_report_payload(
        config=config,
        input_sha256={"predictions": b_hash, "store": a_hash},
        daily=(second, first),
    )
    assert left == right
    assert left["aggregate"]["net_pnl_brl"] == 48.0
    assert left["daily"][0]["trade_date"] == "2024-01-02"

    first_meta = write_execution_report(
        tmp_path / "first.json",
        config=config,
        input_sha256={"store": a_hash, "predictions": b_hash},
        daily=(first, second),
    )
    second_meta = write_execution_report(
        tmp_path / "second.json",
        config=config,
        input_sha256={"predictions": b_hash, "store": a_hash},
        daily=(second, first),
    )
    assert first_meta["sha256"] == second_meta["sha256"]
    assert (
        first_meta["sha256"]
        == hashlib.sha256((tmp_path / "first.json").read_bytes()).hexdigest()
    )
    assert (tmp_path / "first.json.sha256").read_text(encoding="utf-8") == (
        f"{first_meta['sha256']}  first.json\n"
    )
    assert json.loads((tmp_path / "first.json").read_text(encoding="utf-8")) == left
