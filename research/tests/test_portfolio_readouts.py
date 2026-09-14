from dataclasses import replace

import numpy as np
import pytest
import torch

from brazil_rv.execution.portfolio_policy import PreferenceModel, exact_replay
from brazil_rv.v2.portfolio_inputs import Calibration
from brazil_rv.v2.portfolio_readouts import interval, save_book, verify_book
from brazil_rv.v2.research_rounds import _folded_bootstrap
from test_portfolio_policy import policy_fixture


def test_fast_block_intervals_match_the_registered_draws():
    rng = np.random.default_rng(9)
    arrays = [rng.normal(1, 40, n) for n in (121, 126, 123, 127)]
    original = _folded_bootstrap(
        arrays, replications=10_000, seed=20260914, block_length=20
    )
    fast = interval(arrays)
    for key in ("estimate", "lower_95", "upper_95"):
        assert fast[key] == pytest.approx(original[key], abs=1e-12)


def test_model_roll_does_not_change_prior_decisions_and_book_reconciles(tmp_path):
    torch.set_num_threads(1)
    data = policy_fixture()
    positive = Calibration(np.zeros(3), np.ones(3), np.array([0.0006, 0, 0]), 0.0)
    negative = Calibration(np.zeros(3), np.ones(3), np.array([-0.0006, 0, 0]), 0.0)
    a = PreferenceModel(data, positive, np.arange(10))
    b = PreferenceModel(data, negative, np.arange(10))
    stable, old_targets, _ = exact_replay(data, a, 0, 12)
    result, targets, previous = exact_replay(
        data, {day: a if day < 6 else b for day in range(12)}, 0, 12
    )
    np.testing.assert_array_equal(targets[:6], old_targets[:6])
    np.testing.assert_array_equal(result.nav[:6], stable.nav[:6])
    assert not np.allclose(targets[6:], old_targets[6:])
    provenance = {"scenario": "base", "policy": "optimizer"}
    record = save_book(
        tmp_path / "book", data, result, targets, previous, 0, 2, provenance
    )
    assert record["burn_in_sessions"] == 2
    assert record["summary"]["sessions"] == 10
    assert verify_book(tmp_path / "book", provenance) == record
    daily = record["daily"]
    reconstructed = (
        np.array(daily["equity_gross_bps"])
        + daily["hedge_gross_bps"]
        + np.array(daily["interest_bps"])
        - daily["trading_cost_bps"]
        - np.array(daily["borrow_bps"])
        - daily["cdi_bps"]
    )
    np.testing.assert_allclose(reconstructed, daily["net_excess_bps"], atol=1e-9)
    np.testing.assert_array_equal(daily["turnover"], result.turnover_fraction_nav[2:])


def test_cash_readout_earns_same_cdi_benchmark(tmp_path):
    data = policy_fixture()
    data.inputs = replace(
        data.inputs, cdi_returns=np.full(len(data.inputs.dates), 0.0004)
    )
    zeros = np.zeros((12, len(data.inputs.security_ids) + 1))
    result, targets, previous = exact_replay(data, None, 0, 12, targets=zeros)
    record = save_book(
        tmp_path / "cash",
        data,
        result,
        targets,
        previous,
        0,
        2,
        {"scenario": "base", "policy": "cash"},
    )
    summary = record["summary"]
    assert summary["mean"]["net_excess_bps"] == pytest.approx(0, abs=1e-10)
    assert summary["absolute_compounded_return"] > 0
    assert summary["relative_to_cdi_compounded_return"] == pytest.approx(0, abs=1e-12)
    assert summary["all_cash_sessions"] == 10


def test_all_cost_scenarios_and_zero_residual_reuse_are_source_bound(
    tmp_path, monkeypatch
):
    from brazil_rv.v2 import portfolio_readouts as readouts
    from brazil_rv.v2.execution_policy import ExecutionPolicy

    data = policy_fixture()
    data.inputs = replace(
        data.inputs,
        execution_policy=ExecutionPolicy(horizons=(3, 5, 10), buffer_per_quintile=9),
    )
    calibration = Calibration(np.zeros(3), np.ones(3), np.array([0.0006, 0, 0]), 0.0)
    model = PreferenceModel(data, calibration, np.arange(10))
    models = {p: model for p in readouts.POLICIES}
    sources = {f"learned_{seed}": {"selected_epoch": 0} for seed in (11, 29, 47)}
    bounds = {
        "fit": np.arange(8),
        "selection": np.arange(8, 10),
        "evaluation": np.arange(11, 25),
    }
    monkeypatch.setattr(readouts, "_git_identity", lambda: {"commit": "fixture"})
    monkeypatch.setattr(
        readouts, "models_for_fold", lambda *args: (models, sources, bounds)
    )
    (tmp_path / "frozen_design.json").write_text("{}")
    readouts.evaluate_books(tmp_path, "S0", fold="F1", loaded=(data, "fixture"))
    books = list((tmp_path / "books/S0/F1").glob("*/*/book.json"))
    assert len(books) == 21
    for path in books:
        record = readouts.read(path)
        assert readouts.verify_book(path.parent, record["provenance"]) == record
        if path.parent.name.startswith("learned_"):
            assert "equivalent_book" in record
            assert not record["files"]
    assert (tmp_path / "bridge/S0/F1/forecast.json").exists()
