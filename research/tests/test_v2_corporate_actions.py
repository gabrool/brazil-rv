from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import polars as pl

from brazil_rv.v2 import corporate_actions as actions_module
from brazil_rv.v2.corporate_actions import (
    AlignedActionTerms,
    VerifiedActionTerm,
    _extract_yfinance_actions,
    acquire_yfinance_actions,
    action_coverage_resolved_mask,
    align_action_payment_sessions,
    align_action_arrays,
    align_decision_known_action_terms,
    align_verified_action_terms,
    apply_contractual_action,
    action_calendar_alignment_table,
    action_coverage_table,
    audit_m1_adjustment_status,
    build_shareholder_wealth_ohlc,
    detect_cotahist_actions,
    detect_distribution_changes,
    infer_cotahist_action_terms,
    normalize_cached_action_schema,
    normalize_yfinance_actions,
    provider_actions_to_verified_terms,
    unadjust_yfinance_cash_distributions,
    verified_conversion_terms_from_links,
    verified_action_terms_from_table,
    verified_action_terms_to_table,
)
from brazil_rv.v2.decision_clock import SessionDefinition, next_session_decision_cutoffs


def _term(
    action_type: str,
    *,
    ex_date: date,
    q: float = 1.0,
    d: float = 0.0,
    sequence: int = 0,
    payment_date: date | None = None,
    resolved: bool = True,
    resulting_isin: str | None = None,
    effective_date: date | None = None,
    available_at: datetime | None = None,
    currency: str = "BRL",
) -> VerifiedActionTerm:
    return VerifiedActionTerm(
        action_type=action_type,
        isin="BRTESTACNOR1",
        issuer_id="TEST",
        effective_date=ex_date if effective_date is None else effective_date,
        ex_date=ex_date,
        payment_date=payment_date,
        announced_at=datetime(2023, 12, 1, 12, tzinfo=timezone.utc),
        available_at=(
            datetime(2023, 12, 1, 13, tzinfo=timezone.utc)
            if available_at is None
            else available_at
        ),
        shares_per_prior_share=q,
        cash_per_prior_share=d,
        currency=currency,
        source="issuer filing",
        evidence="immutable filing sha256:abc",
        coverage_status="verified",
        resulting_isin=resulting_isin,
        sequence=sequence,
        resolved=resolved,
    )


def test_verified_terms_require_evidence_and_leave_complex_actions_unresolved() -> None:
    with np.testing.assert_raises(ValueError):
        _term("subscription_rights", ex_date=date(2024, 1, 3), resolved=True)
    unresolved = _term("subscription_rights", ex_date=date(2024, 1, 3), resolved=False)
    assert not unresolved.resolved
    with np.testing.assert_raises(ValueError):
        VerifiedActionTerm(
            action_type="dividend",
            isin="BRTESTACNOR1",
            issuer_id=None,
            effective_date=date(2024, 1, 3),
            ex_date=date(2024, 1, 3),
            payment_date=None,
            announced_at=None,
            available_at=datetime(2023, 12, 1),
            shares_per_prior_share=1.0,
            cash_per_prior_share=1.0,
            currency="BRL",
            source="issuer filing",
            evidence="filing",
            coverage_status="verified",
        )
    with np.testing.assert_raises_regex(ValueError, "BRL currency"):
        _term("dividend", ex_date=date(2024, 1, 3), d=1.0, currency="USD")


def test_verified_action_table_round_trip_preserves_full_contract() -> None:
    term = _term(
        "dividend",
        ex_date=date(2024, 1, 3),
        d=1.25,
        payment_date=date(2024, 2, 1),
    )
    table = verified_action_terms_to_table([term])
    assert {
        "effective_date",
        "ex_date",
        "payment_date",
        "announced_at",
        "available_at",
        "shares_per_prior_share",
        "cash_per_prior_share",
        "currency",
        "source",
        "evidence",
        "coverage_status",
    }.issubset(table.columns)
    assert verified_action_terms_from_table(table) == (term,)


def test_action_sequence_uses_common_units_and_signed_cash_obligation() -> None:
    dates = [date(2024, 1, 2), date(2024, 1, 3)]
    aligned = align_verified_action_terms(
        [
            _term("split", ex_date=dates[1], q=2.0, sequence=0),
            _term("dividend", ex_date=dates[1], d=1.0, sequence=1),
        ],
        dates,
        ["BRTESTACNOR1"],
        coverage_resolved=np.ones((2, 1), dtype=np.bool_),
    )
    # The later dividend is per post-split share: two cash units per opening share.
    assert aligned.shares_per_prior_share[1, 0] == 2.0
    assert aligned.cash_per_prior_share[1, 0] == 2.0
    long_shares, long_cash = apply_contractual_action(
        3.0,
        5.0,
        shares_per_prior_share=2.0,
        cash_per_prior_share=1.0,
    )
    short_shares, short_cash = apply_contractual_action(
        -3.0,
        5.0,
        shares_per_prior_share=2.0,
        cash_per_prior_share=1.0,
    )
    assert (long_shares, long_cash) == (6.0, 8.0)
    assert (short_shares, short_cash) == (-6.0, 2.0)


def test_action_payment_sessions_preserve_unknown_and_separate_due_dates() -> None:
    dates = [
        date(2024, 1, 2),
        date(2024, 1, 3),
        date(2024, 1, 4),
        date(2024, 1, 5),
    ]
    aligned = align_action_payment_sessions(
        [
            _term(
                "dividend",
                ex_date=dates[1],
                d=1.0,
                payment_date=dates[3],
            ),
            _term("jcp", ex_date=dates[2], d=0.5, sequence=1),
        ],
        dates,
        ["BRTESTACNOR1"],
    )
    assert aligned[:, 0].tolist() == [-1, 3, -1, -1]

    with np.testing.assert_raises_regex(ValueError, "different payment dates"):
        align_action_payment_sessions(
            [
                _term(
                    "dividend",
                    ex_date=dates[1],
                    d=1.0,
                    payment_date=dates[2],
                ),
                _term(
                    "jcp",
                    ex_date=dates[1],
                    d=0.5,
                    sequence=1,
                    payment_date=dates[3],
                ),
            ],
            dates,
            ["BRTESTACNOR1"],
        )


def test_shareholder_wealth_ohlc_uses_contract_terms_and_no_payment_gain() -> None:
    raw_open = np.asarray([[100.0], [54.0], [53.0]])
    raw_high = np.asarray([[101.0], [56.0], [55.0]])
    raw_low = np.asarray([[99.0], [53.0], [52.0]])
    raw_close = np.asarray([[100.0], [55.0], [54.0]])
    actions = AlignedActionTerms(
        shares_per_prior_share=np.asarray([[1.0], [2.0], [1.0]]),
        cash_per_prior_share=np.asarray([[0.0], [1.0], [0.0]]),
        session_resolved=np.ones((3, 1), dtype=np.bool_),
        has_action=np.asarray([[False], [True], [False]]),
    )
    wealth = build_shareholder_wealth_ohlc(
        raw_open,
        raw_high,
        raw_low,
        raw_close,
        np.ones((3, 1), dtype=np.bool_),
        actions,
    )
    assert wealth.valid.all()
    # q=2,d=1 recognizes the entitlement once on the event session.
    assert wealth.close[1, 0] == 111.0
    # The following (possible payment) session has no second distribution gain.
    np.testing.assert_allclose(wealth.close[2, 0], 111.0 * 54.0 / 55.0)


def test_verified_simple_conversion_follows_the_successor_isin() -> None:
    dates = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    aligned = align_verified_action_terms(
        [
            _term(
                "simple_conversion",
                ex_date=dates[1],
                resulting_isin="BRTESTACNPR0",
            )
        ],
        dates,
        ["BRTESTACNOR1", "BRTESTACNPR0"],
        coverage_resolved=np.ones((3, 2), dtype=np.bool_),
    )
    assert aligned.successor_index is not None
    assert aligned.successor_index[1, 0] == 1
    raw_close = np.asarray([[100.0, np.nan], [np.nan, 105.0], [np.nan, 110.0]])
    wealth = build_shareholder_wealth_ohlc(
        raw_close.copy(),
        raw_close.copy(),
        raw_close.copy(),
        raw_close,
        np.isfinite(raw_close),
        aligned,
    )
    np.testing.assert_array_equal(wealth.close[:, 0], [100.0, 105.0, 110.0])


def test_conversion_uses_effective_date_and_carries_same_session_cash() -> None:
    dates = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    aligned = align_verified_action_terms(
        [
            _term(
                "simple_conversion",
                effective_date=dates[1],
                ex_date=dates[1],
                d=2.0,
                resulting_isin="BRTESTACNPR0",
            )
        ],
        dates,
        ["BRTESTACNOR1", "BRTESTACNPR0"],
        coverage_resolved=np.ones((3, 2), dtype=np.bool_),
    )
    assert aligned.has_action[1, 0]
    assert aligned.successor_index is not None
    assert aligned.successor_index[1, 0] == 1
    assert aligned.cash_per_prior_share[1, 0] == 2.0

    with np.testing.assert_raises_regex(ValueError, "common event session"):
        _term(
            "simple_conversion",
            effective_date=dates[1],
            ex_date=dates[2],
            d=2.0,
            resulting_isin="BRTESTACNPR0",
        )


def test_verified_isin_link_becomes_canonical_conversion_term() -> None:
    effective = date(2024, 1, 3)
    known = datetime(2024, 1, 2, 18, tzinfo=timezone.utc)
    terms = verified_conversion_terms_from_links(
        pl.DataFrame(
            {
                "predecessor_isin": ["BRTESTACNOR1"],
                "successor_isin": ["BRTESTACNPR0"],
                "effective_date": [effective],
                "first_known_at": [known],
                "shares_received_per_prior_share": [1.5],
                "cash_entitlement_per_prior_share": [2.0],
                "currency": ["BRL"],
                "source": ["issuer filing"],
                "evidence_sha256": ["a" * 64],
            }
        )
    )
    assert len(terms) == 1
    term = terms[0]
    assert term.effective_date == effective
    assert term.ex_date == effective
    assert term.available_at == known
    assert term.shares_per_prior_share == 1.5
    assert term.cash_per_prior_share == 2.0
    assert term.resulting_isin == "BRTESTACNPR0"


def test_empty_isin_link_table_needs_no_conversion_schema() -> None:
    links = pl.DataFrame(
        schema={"predecessor_isin": pl.String, "successor_isin": pl.String}
    )
    assert verified_conversion_terms_from_links(links) == ()


def test_pure_conversion_uses_effective_date_not_ex_date() -> None:
    dates = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    aligned = align_verified_action_terms(
        [
            _term(
                "simple_conversion",
                effective_date=dates[1],
                ex_date=dates[2],
                resulting_isin="BRTESTACNPR0",
            )
        ],
        dates,
        ["BRTESTACNOR1", "BRTESTACNPR0"],
        coverage_resolved=np.ones((3, 2), dtype=np.bool_),
    )
    assert aligned.successor_index is not None
    assert aligned.successor_index[1, 0] == 1
    assert aligned.successor_index[2, 0] == 0


def test_decision_known_alignment_marks_later_acquired_action_unresolved() -> None:
    dates = [date(2024, 1, 2), date(2024, 1, 3)]
    decisions = [
        datetime(2024, 1, 2, 18, 45, tzinfo=timezone.utc),
        datetime(2024, 1, 3, 18, 45, tzinfo=timezone.utc),
    ]
    late = _term(
        "split",
        ex_date=dates[1],
        q=2.0,
        available_at=datetime(2024, 1, 4, 12, tzinfo=timezone.utc),
    )
    coverage = np.ones((2, 1), dtype=np.bool_)
    retrospective = align_verified_action_terms(
        [late], dates, ["BRTESTACNOR1"], coverage_resolved=coverage
    )
    historical = align_decision_known_action_terms(
        [late],
        dates,
        ["BRTESTACNOR1"],
        coverage_resolved=coverage,
        decision_timestamps=decisions,
    )
    empty_historical = align_decision_known_action_terms(
        [],
        dates,
        ["BRTESTACNOR1"],
        coverage_resolved=coverage,
        decision_timestamps=decisions,
    )
    assert retrospective.shares_per_prior_share[1, 0] == 2.0
    assert retrospective.has_action[1, 0]
    np.testing.assert_array_equal(
        historical.shares_per_prior_share,
        empty_historical.shares_per_prior_share,
    )
    np.testing.assert_array_equal(historical.has_action, empty_historical.has_action)
    assert not historical.has_action[1, 0]
    assert not historical.session_resolved[1, 0]
    assert empty_historical.session_resolved[1, 0]


def test_end_of_session_inferred_term_is_consumed_only_at_next_decision() -> None:
    dates = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    schedule = tuple(
        SessionDefinition(
            trade_date=value,
            continuous_open=datetime.min.time().replace(hour=10),
            decision_time=datetime.min.time().replace(hour=15, minute=45),
            continuous_close=datetime.min.time().replace(hour=16, minute=55),
            auction_close=datetime.min.time().replace(hour=17),
            source="reconstructed_v1:fixture",
        )
        for value in dates
    )
    end_of_event = _term(
        "split",
        ex_date=dates[1],
        q=2.0,
        available_at=datetime.fromisoformat("2024-01-03T23:59:59-03:00"),
    )
    coverage = np.ones((3, 1), dtype=np.bool_)
    same_session = align_decision_known_action_terms(
        [end_of_event],
        dates,
        ["BRTESTACNOR1"],
        coverage_resolved=coverage,
        decision_timestamps=[row.decision_at for row in schedule],
    )
    following_session = align_decision_known_action_terms(
        [end_of_event],
        dates,
        ["BRTESTACNOR1"],
        coverage_resolved=coverage,
        decision_timestamps=next_session_decision_cutoffs(schedule),
    )
    assert not same_session.has_action[1, 0]
    assert not same_session.session_resolved[1, 0]
    assert following_session.has_action[1, 0]
    assert following_session.session_resolved[1, 0]
    assert following_session.shares_per_prior_share[1, 0] == 2.0


def test_inferred_actions_cover_readable_cells_and_apply_u1_c1_rules() -> None:
    dates = np.arange("2024-01-02", "2024-01-05", dtype="datetime64[D]")
    names = tuple(f"BRTEST{i:07d}" for i in range(21))
    close = np.full((3, 21), 100.0)
    close[1, 1:] = 101.0
    close[1, 0] = 95.0
    close[2, 0] = 47.5
    quantity = np.full_like(close, 100.0)
    trades = np.full_like(close, 10.0)
    dismes = np.ones_like(close)
    dismes[1:, 0] = 2.0
    dismes[2, 0] = 3.0
    observed = np.ones_like(close, dtype=np.bool_)
    dismes[0, -1] = np.nan
    result = infer_cotahist_action_terms(
        dates,
        names,
        close,
        quantity,
        trades,
        dismes,
        observed,
        np.ones_like(observed),
    )
    assert result.c1_event[1, 0]
    c1 = next(term for term in result.terms if term.ex_date == date(2024, 1, 3))
    np.testing.assert_allclose(c1.cash_per_prior_share, 6.0)
    assert c1.coverage_status == "inferred"
    assert result.u1_event[2, 0]
    u1 = next(term for term in result.terms if term.ex_date == date(2024, 1, 4))
    np.testing.assert_allclose(u1.shares_per_prior_share, 97.5 / 47.5)
    assert result.coverage_resolved[0, :-1].all()
    assert not result.coverage_resolved[0, -1]


def test_u2_uses_trade_size_and_delayed_dismes_does_not_double_adjust() -> None:
    dates = np.arange("2024-01-02", "2024-01-06", dtype="datetime64[D]")
    close = np.asarray([[100.0], [100.0], [50.0], [51.0]])
    quantity = np.asarray([[100.0], [100.0], [200.0], [190.0]])
    trades = np.asarray([[10.0], [10.0], [10.0], [10.0]])
    dismes = np.asarray([[1.0], [1.0], [1.0], [2.0]])
    observed = np.ones_like(close, dtype=np.bool_)
    result = infer_cotahist_action_terms(
        dates,
        ("BRTESTACNOR1",),
        close,
        quantity,
        trades,
        dismes,
        observed,
        observed,
    )
    assert result.u2_event[:, 0].tolist() == [False, False, True, False]
    assert result.u2_per_trade_event[2, 0]
    assert result.u2_per_trade_candidate[2, 0]
    assert result.u2_total_quantity_candidate[2, 0]
    assert result.c1_event[3, 0]
    delayed = next(term for term in result.terms if term.ex_date == date(2024, 1, 5))
    assert delayed.action_type == "cash_distribution"
    assert delayed.cash_per_prior_share == 0.0


def test_inferred_action_prefix_is_future_mutation_invariant() -> None:
    dates = np.arange("2024-01-02", "2024-01-08", dtype="datetime64[D]")
    close = np.asarray([[100.0], [100.0], [50.0], [51.0], [52.0], [53.0]])
    quantity = np.asarray([[100.0], [100.0], [200.0], [190.0], [180.0], [170.0]])
    trades = np.full_like(close, 10.0)
    dismes = np.ones_like(close)
    observed = np.ones_like(close, dtype=np.bool_)
    baseline = infer_cotahist_action_terms(
        dates,
        ("BRTESTACNOR1",),
        close,
        quantity,
        trades,
        dismes,
        observed,
        observed,
    )
    mutated_close = close.copy()
    mutated_close[4:] = 1_000.0
    mutated = infer_cotahist_action_terms(
        dates,
        ("BRTESTACNOR1",),
        mutated_close,
        quantity,
        trades,
        dismes,
        observed,
        observed,
    )
    np.testing.assert_array_equal(baseline.u2_event[:4], mutated.u2_event[:4])
    np.testing.assert_array_equal(
        baseline.large_move_no_action[:4], mutated.large_move_no_action[:4]
    )
    assert [
        term.evidence for term in baseline.terms if term.ex_date <= date(2024, 1, 5)
    ] == [term.evidence for term in mutated.terms if term.ex_date <= date(2024, 1, 5)]


def test_explicit_verified_term_resolves_event_outside_blanket_coverage() -> None:
    dates = [date(2024, 1, 2), date(2024, 1, 3)]
    aligned = align_verified_action_terms(
        [_term("split", ex_date=dates[1], q=2.0)],
        dates,
        ["BRTESTACNOR1"],
        coverage_resolved=np.zeros((2, 1), dtype=np.bool_),
    )

    assert not aligned.session_resolved[0, 0]
    assert aligned.session_resolved[1, 0]
    assert aligned.has_action[1, 0]
    assert aligned.shares_per_prior_share[1, 0] == 2.0


def test_cotahist_split_detection_and_provider_alignment_are_independent() -> None:
    dates = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    close = np.asarray([[100.0], [50.0], [51.0]])
    quantity = np.asarray([[1_000.0], [2_000.0], [2_100.0]])
    observed = np.ones_like(close, dtype=np.bool_)
    dismes = np.asarray([[1.0], [2.0], [2.0]])
    detected = detect_cotahist_actions(close, quantity, dismes, observed)
    assert detected.split_event[:, 0].tolist() == [False, True, False]

    actions = normalize_yfinance_actions(
        pl.DataFrame(
            {
                "date": [dates[1]],
                "dividends": [0.5],
                "stock_splits": [2.0],
            }
        ),
        isin="BRTESTACNOR1",
        ticker="TEST3",
        fetched_at=datetime(2024, 2, 1, tzinfo=timezone.utc),
    )
    split, cash, unresolved = align_action_arrays(actions, dates, ["BRTESTACNOR1"])
    np.testing.assert_array_equal(split[:, 0], [1.0, 2.0, 1.0])
    np.testing.assert_array_equal(cash[:, 0], [0.0, 0.5, 0.0])
    assert not unresolved.any()


def test_cotahist_classifies_cash_and_ambiguous_dismes_changes() -> None:
    close = np.asarray([[100.0, 100.0], [100.0, 100.0], [95.0, 99.0], [95.0, 99.0]])
    quantity = np.asarray(
        [[100.0, 100.0], [100.0, 100.0], [50.0, 100.0], [50.0, 100.0]]
    )
    dismes = np.asarray([[1.0, 1.0], [1.0, 1.0], [2.0, 2.0], [2.0, 2.0]])
    result = detect_cotahist_actions(
        close, quantity, dismes, np.ones_like(close, dtype=bool)
    )
    assert result.ambiguous_event[2, 0]
    assert result.cash_event[2, 1]
    assert not result.split_event[2].any()

    # Price jumps without a DISMES change are audit anomalies only.
    no_dismes = np.ones((4, 1))
    large_drop = np.asarray([[100.0], [100.0], [80.0], [80.0]])
    large_result = detect_cotahist_actions(
        large_drop,
        np.full_like(large_drop, 100.0),
        no_dismes,
        np.ones_like(large_drop, dtype=bool),
    )
    assert not large_result.event_candidate.any()
    assert not large_result.cash_event.any()
    assert large_result.price_jump_anomaly_mask[2, 0]

    moderate_drop = np.asarray([[100.0], [100.0], [94.0], [94.0]])
    moderate_result = detect_cotahist_actions(
        moderate_drop,
        np.asarray([[100.0], [100.0], [50.0], [50.0]]),
        no_dismes,
        np.ones_like(moderate_drop, dtype=bool),
    )
    assert not moderate_result.event_candidate.any()
    assert not moderate_result.ambiguous_event.any()
    assert moderate_result.price_jump_anomaly_mask[2, 0]


def test_jump_only_examples_are_audit_anomalies_not_action_terms() -> None:
    examples = (
        [100, 100, 100, 110, 110, 110, 110, 110],
        [100, 100, 100, 80, 80, 80, 80, 80],
        [100, 100, 100, 105, 105, 105, 105, 105],
        [100, 100, 100, 105, 100, 100, 100, 100],
        [100, 100, 100, 90, 92, 92, 92, 92],
    )
    for raw in examples:
        close = np.asarray(raw, dtype=np.float64)[:, None]
        result = detect_cotahist_actions(
            close,
            np.full_like(close, 100.0),
            np.ones_like(close),
            np.ones_like(close, dtype=np.bool_),
        )
        assert not result.event_candidate.any()
        assert not result.split_event.any()
        assert not result.cash_event.any()
        assert not result.ambiguous_event.any()
        assert result.price_jump_anomaly_mask.any()


def test_cotahist_classifier_is_causal_and_exercises_all_classes() -> None:
    close = np.asarray([[100.0], [100.0], [50.0], [51.0], [52.0]])
    quantity = np.asarray([[100.0], [100.0], [200.0], [210.0], [220.0]])
    dismes = np.asarray([[1.0], [1.0], [2.0], [2.0], [2.0]])
    observed = np.ones_like(close, dtype=np.bool_)
    original = detect_cotahist_actions(close, quantity, dismes, observed)
    mutated_close = close.copy()
    mutated_quantity = quantity.copy()
    mutated_close[3:] *= 7.0
    mutated_quantity[3:] *= 0.2
    changed = detect_cotahist_actions(mutated_close, mutated_quantity, dismes, observed)
    assert original.split_event[2, 0]
    np.testing.assert_array_equal(original.split_event[:3], changed.split_event[:3])
    np.testing.assert_array_equal(original.price_ratio[:3], changed.price_ratio[:3])

    def classify(price: float, qty: float) -> tuple[bool, bool, bool]:
        values = np.asarray([[100.0], [100.0], [price]])
        quantities = np.asarray([[100.0], [100.0], [qty]])
        distributions = np.asarray([[1.0], [1.0], [2.0]])
        result = detect_cotahist_actions(
            values, quantities, distributions, np.ones_like(values, dtype=np.bool_)
        )
        return (
            bool(result.split_event[2, 0]),
            bool(result.cash_event[2, 0]),
            bool(result.ambiguous_event[2, 0]),
        )

    assert classify(99.0, 100.0) == (False, True, False)
    assert classify(94.0, 100.0) == (False, False, True)
    assert classify(90.0, 220.0) == (False, True, False)


def test_strict_fallback_and_large_ratio_tolerance_boundary() -> None:
    close = np.asarray([[100.0], [100.0], [1_000.0]])
    quantity = np.asarray([[100.0], [100.0], [10.0]])
    dismes = np.ones_like(close)
    observed = np.ones_like(close, dtype=np.bool_)
    disabled = detect_cotahist_actions(close, quantity, dismes, observed)
    enabled = detect_cotahist_actions(
        close,
        quantity,
        dismes,
        observed,
        undocumented_split_fallback=True,
    )
    assert disabled.price_jump_anomaly_mask[2, 0]
    assert not disabled.split_event.any()
    assert enabled.split_event[2, 0]
    assert not enabled.price_jump_anomaly_mask[2, 0]

    for log_price, tolerance in ((0.299999, 0.15), (0.300001, 0.35)):
        price = 100.0 * np.exp(log_price)
        quantity_ratio = np.exp(-log_price + tolerance * 0.5)
        result = detect_cotahist_actions(
            np.asarray([[100.0], [100.0], [price]]),
            np.asarray([[100.0], [100.0], [100.0 * quantity_ratio]]),
            np.asarray([[1.0], [1.0], [2.0]]),
            np.ones((3, 1), dtype=np.bool_),
        )
        assert result.split_event[2, 0]


def test_legacy_canonical_cache_schema_is_upgraded_in_memory() -> None:
    fetched_at = datetime(2024, 2, 1, tzinfo=timezone.utc)
    legacy = pl.DataFrame(
        {
            "isin": ["BRTESTACNOR1"],
            "ex_date": [date(2024, 1, 2)],
            "action_type": ["dividend"],
            "split_factor": [1.0],
            "cash_distribution_brl": [0.5],
        }
    )
    upgraded = normalize_cached_action_schema(
        legacy,
        isin="BRTESTACNOR1",
        ticker="TEST3",
        fetched_at=fetched_at,
    )
    assert upgraded[0, "provider_cash_distribution_brl"] == 0.5
    assert upgraded[0, "cash_unit_adjustment_factor"] == 1.0
    assert upgraded[0, "known_date"] == date(2024, 1, 2)
    assert upgraded[0, "source_ticker"] == "TEST3"


def test_legacy_provider_cache_schema_is_normalized_in_memory() -> None:
    fetched_at = datetime(2024, 2, 1, tzinfo=timezone.utc)
    cached = pl.DataFrame(
        {
            "Date": [date(2024, 1, 15)],
            "Dividends": [0.5],
            "Stock Splits": [0.0],
        }
    )
    upgraded = normalize_cached_action_schema(
        cached,
        isin="BRTESTACNOR1",
        ticker="TEST3",
        fetched_at=fetched_at,
    )
    assert upgraded[0, "action_type"] == "dividend"
    assert upgraded[0, "cash_distribution_brl"] == 0.5
    assert upgraded[0, "source_ticker"] == "TEST3"


def test_normalize_yfinance_emits_split_and_dividend_rows() -> None:
    frame = pl.DataFrame(
        {
            "date": [date(2024, 1, 2)],
            "dividends": [0.5],
            "stock_splits": [2.0],
        }
    )
    result = normalize_yfinance_actions(
        frame,
        isin="BRTESTACNOR1",
        ticker="TEST3",
        fetched_at=datetime(2024, 2, 1, tzinfo=timezone.utc),
    )
    assert set(result.get_column("action_type")) == {"split", "dividend"}


def test_yfinance_nan_cells_are_not_actions() -> None:
    frame = pl.DataFrame(
        {
            "date": [date(2024, 1, 2), date(2024, 1, 3)],
            "dividends": [float("nan"), 0.5],
            "stock_splits": [float("nan"), float("nan")],
        }
    )
    result = normalize_yfinance_actions(
        frame,
        isin="BRTESTACNOR1",
        ticker="TEST3",
        fetched_at=datetime(2024, 2, 1, tzinfo=timezone.utc),
    )
    assert result.height == 1
    assert result[0, "action_type"] == "dividend"
    assert result[0, "cash_distribution_brl"] == 0.5


def test_yfinance_all_nan_symbol_frame_is_provider_failure() -> None:
    columns = pd.MultiIndex.from_product(
        [["FAIL3.SA"], ["Close", "Dividends", "Stock Splits"]]
    )
    missing = pd.DataFrame(
        [[np.nan, np.nan, np.nan]],
        index=pd.DatetimeIndex(["2024-01-02"]),
        columns=columns,
    )
    assert _extract_yfinance_actions(missing, "FAIL3.SA") is None

    present = missing.copy()
    present.loc[:, ("FAIL3.SA", "Close")] = 10.0
    extracted = _extract_yfinance_actions(present, "FAIL3.SA")
    assert extracted is not None
    assert extracted.height == 1
    assert extracted[0, "dividends"] == 0.0


def test_batched_acquisition_records_symbol_failure_without_aborting(
    tmp_path, monkeypatch
) -> None:
    master = pl.DataFrame(
        {
            "isin": ["BRTESTACNOR1", "BRTESTACNPR0"],
            "ticker": ["GOOD3", "BAD4"],
            "first_date": [date(2024, 1, 1), date(2024, 1, 1)],
            "last_date": [date(2024, 1, 31), date(2024, 1, 31)],
        }
    )

    def fake_download(tickers, *, start, end):
        del start, end
        if tickers == ["BAD4"]:
            raise RuntimeError("provider failure")
        return {
            "GOOD3": pl.DataFrame(
                {
                    "date": [date(2024, 1, 10)],
                    "dividends": [0.25],
                    "stock_splits": [0.0],
                }
            )
        }

    monkeypatch.setattr(actions_module, "_download_yfinance_batch", fake_download)
    result, audit = acquire_yfinance_actions(
        master,
        tmp_path,
        fetched_at=datetime(2024, 2, 1, tzinfo=timezone.utc),
        batch_size=1,
    )
    assert result.height == 1
    assert result[0, "isin"] == "BRTESTACNOR1"
    assert set(audit.get_column("status")) == {"downloaded", "failed"}
    assert audit.filter(pl.col("status") == "failed")[0, "cache_path"] is None


def test_historical_segment_retries_under_current_isin_ticker(
    tmp_path, monkeypatch
) -> None:
    master = pl.DataFrame(
        {
            "isin": ["BRTESTACNOR1", "BRTESTACNOR1"],
            "ticker": ["OLD3", "NEW3"],
            "first_date": [date(2020, 1, 1), date(2024, 1, 1)],
            "last_date": [date(2020, 1, 31), date(2024, 1, 31)],
        }
    )

    def fake_download(tickers, *, start, end):
        del end
        ticker = tickers[0]
        if ticker == "OLD3":
            return {
                ticker: pl.DataFrame(
                    {
                        "date": [date(2024, 1, 10)],
                        "dividends": [0.0],
                        "stock_splits": [0.0],
                        "price_observed": [True],
                    }
                )
            }
        day = date(2020, 1, 10) if start.year == 2020 else date(2024, 1, 10)
        return {
            ticker: pl.DataFrame(
                {
                    "date": [day],
                    "dividends": [0.25 if day.year == 2020 else 0.0],
                    "stock_splits": [0.0],
                    "price_observed": [True],
                }
            )
        }

    monkeypatch.setattr(actions_module, "_download_yfinance_batch", fake_download)
    result, audit = acquire_yfinance_actions(
        master,
        tmp_path,
        fetched_at=datetime(2024, 2, 1, tzinfo=timezone.utc),
        batch_size=1,
    )
    old = audit.filter(pl.col("ticker") == "OLD3").row(0, named=True)
    assert old["query_ticker"] == "NEW3"
    assert old["status"] == "downloaded_current_ticker"
    assert result.filter(pl.col("ex_date") == date(2020, 1, 10)).height == 1


def test_provider_failure_state_does_not_enter_cotahist_classification() -> None:
    observed = np.ones((5, 1), dtype=bool)
    dismes = np.asarray([[1.0], [1.0], [2.0], [2.0], [2.0]])
    result = detect_cotahist_actions(
        np.full((5, 1), 100.0),
        np.full((5, 1), 1_000.0),
        dismes,
        observed,
    )
    assert result.cash_event[:, 0].tolist() == [False, False, True, False, False]
    changed = detect_distribution_changes(dismes, observed)
    np.testing.assert_array_equal(changed, result.event_candidate)


def test_cash_units_use_only_strictly_later_split_factors() -> None:
    fetched_at = datetime(2024, 2, 1, tzinfo=timezone.utc)
    same_day = normalize_yfinance_actions(
        pl.DataFrame(
            {
                "date": [date(2020, 1, 2), date(2021, 1, 4)],
                "dividends": [1.0, 0.0],
                "stock_splits": [2.0, 5.0],
            }
        ),
        isin="BRTESTACNOR1",
        ticker="TEST3",
        fetched_at=fetched_at,
    )
    adjusted = unadjust_yfinance_cash_distributions(same_day)
    dividend = adjusted.filter(pl.col("action_type") == "dividend").row(0, named=True)
    assert dividend["provider_cash_distribution_brl"] == 1.0
    assert dividend["cash_unit_adjustment_factor"] == 5.0
    assert dividend["cash_distribution_brl"] == 5.0


def test_action_alignment_audit_counts_off_calendar_rows() -> None:
    actions = normalize_yfinance_actions(
        pl.DataFrame(
            {
                "date": [date(2024, 1, 6)],
                "dividends": [0.5],
                "stock_splits": [0.0],
            }
        ),
        isin="BRTESTACNOR1",
        ticker="TEST3",
        fetched_at=datetime(2024, 2, 1, tzinfo=timezone.utc),
    )
    audit = action_calendar_alignment_table(
        actions, [date(2024, 1, 5), date(2024, 1, 8)], ["BRTESTACNOR1"]
    )
    assert audit.to_dicts() == [
        {
            "isin": "BRTESTACNOR1",
            "ex_date": date(2024, 1, 6),
            "action_type": "dividend",
            "reason": "off_calendar_ex_date",
        }
    ]


def test_action_coverage_distinguishes_true_zero_from_provider_failure() -> None:
    actions = normalize_yfinance_actions(
        pl.DataFrame(
            {
                "date": [date(2024, 6, 3)],
                "dividends": [0.25],
                "stock_splits": [0.0],
            }
        ),
        isin="BRTESTACNOR1",
        ticker="TEST3",
        fetched_at=datetime(2024, 7, 1, tzinfo=timezone.utc),
    )
    audit = pl.DataFrame(
        {
            "isin": ["BRTESTACNOR1", "BRTESTACNPR0"],
            "first_date": [date(2024, 1, 1), date(2024, 1, 1)],
            "last_date": [date(2024, 12, 31), date(2024, 12, 31)],
            "status": ["downloaded", "zero_actions"],
            "action_rows": [1, 0],
        }
    )
    table = action_coverage_table(
        actions,
        [date(2024, 1, 2)],
        ["BRTESTACNOR1", "BRTESTACNPR0"],
        audit,
    )
    assert table[0, "acquisition_status"] == "covered_actions"
    assert table[1, "acquisition_status"] == "covered_zero_actions"
    failed = audit.with_columns(
        pl.when(pl.col("isin") == "BRTESTACNPR0")
        .then(pl.lit("failed"))
        .otherwise(pl.col("status"))
        .alias("status")
    )
    table = action_coverage_table(
        actions,
        [date(2024, 1, 2)],
        ["BRTESTACNOR1", "BRTESTACNPR0"],
        failed,
    )
    assert table[1, "acquisition_status"] == "provider_failure"


def test_m1_adjustment_audit_uses_pre_post_event_ratios() -> None:
    dates = [date(2024, 1, 2), date(2024, 1, 3)]
    raw = np.asarray([[100.0], [50.0]])
    adjusted = np.asarray([[100.0], [100.0]])
    split = np.asarray([[1.0], [2.0]])
    cash = np.zeros_like(raw)
    raw_m1 = raw.copy()
    report = audit_m1_adjustment_status(
        dates,
        ["BRTESTACNOR1"],
        raw_m1,
        raw,
        adjusted,
        split,
        cash,
    )
    assert report[0, "prior_trade_date"] == dates[0]
    assert report[0, "m1_pre_post_ratio"] == 0.5
    assert report[0, "status"] == "raw_unadjusted"

    adjusted_m1 = adjusted.copy()
    report = audit_m1_adjustment_status(
        dates,
        ["BRTESTACNOR1"],
        adjusted_m1,
        raw,
        adjusted,
        split,
        cash,
    )
    assert report[0, "status"] == "price_adjusted"


def test_provider_actions_become_explicit_terms_without_invented_complex_terms() -> (
    None
):
    fetched = datetime(2024, 7, 1, tzinfo=timezone.utc)
    scalar = normalize_yfinance_actions(
        pl.DataFrame(
            {
                "date": [date(2024, 6, 3)],
                "dividends": [1.25],
                "stock_splits": [2.0],
            }
        ),
        isin="BRTESTACNOR1",
        ticker="TEST3",
        fetched_at=fetched,
    )
    complex_row = scalar.head(1).with_columns(
        pl.lit("subscription_rights").alias("action_type"),
        pl.lit(1.0).alias("split_factor"),
        pl.lit(0.0).alias("cash_distribution_brl"),
        pl.lit(True).alias("unresolved"),
    )
    terms = provider_actions_to_verified_terms(
        pl.concat([scalar, complex_row], how="vertical_relaxed")
    )
    split = next(term for term in terms if term.action_type == "split")
    dividend = next(term for term in terms if term.action_type == "dividend")
    rights = next(term for term in terms if term.action_type == "subscription_rights")
    assert split.shares_per_prior_share == 2.0
    assert dividend.cash_per_prior_share == 1.25
    assert split.available_at == fetched
    assert split.announced_at is None and split.payment_date is None
    assert not rights.resolved
    assert (rights.shares_per_prior_share, rights.cash_per_prior_share) == (1.0, 0.0)


def test_action_coverage_requires_explicit_completeness_and_rejects_overlap() -> None:
    dates = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    audit = pl.DataFrame(
        {
            "isin": ["BRTESTACNOR1", "BRTESTACNOR1", "BRTESTACNPR0"],
            "first_date": [dates[0], dates[1], dates[0]],
            "last_date": [dates[-1], dates[1], dates[-1]],
            "status": ["downloaded", "failed", "zero_actions"],
            "economic_terms_complete": [True, True, True],
        }
    )
    resolved = action_coverage_resolved_mask(
        audit, dates, ["BRTESTACNOR1", "BRTESTACNPR0"]
    )
    assert resolved[:, 0].tolist() == [True, False, True]
    assert resolved[:, 1].tolist() == [True, True, True]

    provider_only = audit.drop("economic_terms_complete")
    assert not action_coverage_resolved_mask(
        provider_only, dates, ["BRTESTACNOR1", "BRTESTACNPR0"]
    ).any()
