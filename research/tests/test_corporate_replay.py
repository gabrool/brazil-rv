from copy import copy, deepcopy
from dataclasses import replace

import numpy as np
import pytest

from brazil_rv.execution.portfolio_account import tensor
from brazil_rv.execution.portfolio_policy import exact_replay, policy_ledger_config
from brazil_rv.v2.corporate_replay import apply_corporate_replay, load_corporate_replay
from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from test_portfolio_policy import policy_fixture


def fixture():
    data = policy_fixture()
    calendar = np.asarray(data.inputs.dates, dtype="datetime64[D]")
    resolved = data.inputs.action_session_resolved.copy()
    resolved[2:, 0] = False
    close = data.inputs.raw_close.copy()
    close[2:, 0] = np.nan
    data.inputs = replace(
        data.inputs,
        session_indices=np.arange(len(calendar)),
        action_session_resolved=resolved,
        raw_close=close,
    )
    terms = {
        "cash_cancellations": [
            {
                "isin": data.inputs.security_ids[0],
                "available_date": str(calendar[6]),
                "recognition_date": str(calendar[6]),
                "coverage_start": str(calendar[2]),
                "coverage_available_date": str(calendar[2]),
                "payment_date": str(calendar[8]),
                "cash_per_share": 102,
            }
        ],
        "loan_cash_settlements": [
            {
                "isin": data.inputs.security_ids[0],
                "settlement_date": str(calendar[3]),
                "available_date": str(calendar[3]),
                "cash_per_share": 101,
            }
        ],
    }
    return data, terms, calendar


def test_sparse_admission_preserves_model_objects_and_unrelated_accounting_cells():
    data, terms, calendar = fixture()
    original = data.inputs
    revised = apply_corporate_replay(original, terms, calendar, "a" * 64)
    assert not original.action_session_resolved[2, 0]
    assert revised.action_session_resolved[:, 0].all()
    for key in [
        "active",
        "raw_close",
        "scores",
        "score_mask",
        "prior_feature_values",
        "shareholder_simple_returns",
        "shareholder_target_mask",
        "scaled_midrank_targets",
        "neutral_midrank_targets",
        "target_scale_sigma",
    ]:
        assert getattr(original, key) is getattr(revised, key)
    for key in [
        "action_shares_per_prior_share",
        "action_cash_per_prior_share",
        "action_session_resolved",
        "action_has_action",
        "action_payment_session",
    ]:
        np.testing.assert_array_equal(
            getattr(original, key)[:, 1:], getattr(revised, key)[:, 1:]
        )
    assert revised.action_cash_per_prior_share[:6, 0].sum() == 0
    assert revised.action_cash_per_prior_share[6, 0] == 102
    assert revised.action_payment_session[6, 0] == 8
    assert revised.source_artifact_hashes["corporate_replay"] == "a" * 64
    with pytest.raises(ValueError, match="already applied"):
        apply_corporate_replay(revised, terms, calendar, "a" * 64)


@pytest.mark.parametrize("side", [-1, 1])
def test_admitted_terms_reach_both_accounts_without_rebuilding_policy_features(side):
    data, terms, calendar = fixture()
    revised = copy(data)
    revised.inputs = apply_corporate_replay(data.inputs, terms, calendar, "a" * 64)
    assert revised.static is data.static
    config = policy_ledger_config(
        initial_capital_brl=10_000_000,
        annual_borrow_rate=0,
        borrow_source="uniform",
        borrow_fee_multiplier=0,
        cost_bps_per_side=0,
        hedge_cost_bps_per_side=0,
        planned_absolute_net_cap=1.5,
        planned_absolute_beta_cap=1.5,
        planned_name_weight_cap=0.5,
    )
    targets = np.zeros((10, len(data.inputs.security_ids) + 1))
    targets[:2, 0] = side * 0.4
    account = revised.initial_account(0, config)
    records = [
        revised.step(account, tensor(t), day, terminal=day == 9)
        for day, t in enumerate(targets)
    ]
    exact, _, _ = exact_replay(revised, None, 0, 10, config=config, targets=targets)
    np.testing.assert_allclose([x["nav"].item() for x in records], exact.nav, atol=1e-8)
    if side > 0:
        assert exact.receivables[6:8] == pytest.approx([4_080_000] * 2)
        assert exact.receivables[8] == 0
        assert exact.nav[-1] == pytest.approx(10_080_000)
    else:
        assert exact.loan_cash_payments[0].session == 3
        assert exact.nav[-1] == pytest.approx(9_960_000)
        assert not exact.receivables.any()


def test_source_price_and_future_recognition_do_not_change_earlier_action_cells():
    data, terms, calendar = fixture()
    future = deepcopy(terms)
    future["cash_cancellations"][0]["cash_per_share"] = 1000
    first = apply_corporate_replay(data.inputs, terms, calendar, "a" * 64)
    second = apply_corporate_replay(data.inputs, future, calendar, "b" * 64)
    np.testing.assert_array_equal(
        first.action_cash_per_prior_share[:6], second.action_cash_per_prior_share[:6]
    )
    future["cash_cancellations"][0]["recognition_date"] = str(calendar[5])
    with pytest.raises(ValueError, match="availability"):
        apply_corporate_replay(data.inputs, future, calendar, "b" * 64)
    wrong = calendar + np.timedelta64(1, "D")
    with pytest.raises(ValueError, match="dates differ"):
        apply_corporate_replay(data.inputs, terms, wrong, "a" * 64)


def test_loader_binds_original_sources_and_calendar(tmp_path):
    _, terms, calendar = fixture()
    dates = tmp_path / "dates.npy"
    np.save(dates, calendar)
    source = tmp_path / "source.txt"
    source.write_text("original dated source")
    terms["calendar"] = {"path": str(dates), "sha256": sha256_file(dates)}
    terms["sources"] = [{"path": str(source), "sha256": sha256_file(source)}]
    path = tmp_path / "terms.json"
    digest = write_json_atomic(path, terms)
    _, actual = load_corporate_replay(path, digest)
    np.testing.assert_array_equal(actual, calendar)
    source.write_text("changed source")
    with pytest.raises(ValueError, match="source changed"):
        load_corporate_replay(path, digest)


def test_sliced_axis_keeps_payment_dates_and_prior_loan_prohibition():
    data, terms, calendar = fixture()
    original = data.inputs
    start = 5
    from dataclasses import fields

    sliced = replace(
        original,
        **{
            field.name: getattr(original, field.name)[start:]
            for field in fields(original)
            if isinstance(getattr(original, field.name), np.ndarray)
            and getattr(original, field.name).shape[:1] == (len(calendar),)
        },
        dates=original.dates[start:],
    )
    revised = apply_corporate_replay(sliced, terms, calendar, "a" * 64)
    assert revised.loan_cash_settlements[0].effective_session == -2
    assert revised.action_cash_per_prior_share[1, 0] == 102
    assert revised.action_payment_session[1, 0] == 3
    assert not revised.initial_unresolved_action[0]
    assert revised.scores is sliced.scores
