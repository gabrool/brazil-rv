from dataclasses import replace
from datetime import date, time, timedelta

import numpy as np
import polars as pl

from brazil_rv.v2.build_store import build_daily_store
from brazil_rv.v2.contract import HORIZONS
from brazil_rv.v2.decision_clock import SessionDefinition
from brazil_rv.v2.evaluate import evaluate_scores
from brazil_rv.v2.store import open_store_for_samples, sha256_file
from brazil_rv.v2.validate_pipeline import _evaluation_inputs
from hedge_beta_fixtures import write_hedge_beta_fixture
from test_v2_validate_pipeline import _borrow_panels


def test_inferred_close_mutation_cannot_change_event_day_orders(tmp_path):
    """Real inferred-action builder -> store adapter -> evaluator -> ledger."""
    dates = [date(2023, 1, 2) + timedelta(days=i) for i in range(75)]
    names = tuple(f"BRTEST{i:06d}" for i in range(40))
    event = 70
    rows = []
    for day, day_date in enumerate(dates):
        for name, isin in enumerate(names):
            price = 100 * np.exp(0.001 * day + 0.01 * np.sin(day * 0.31 + name))
            rows.append(
                dict(
                    trade_date=day_date,
                    isin=isin,
                    ticker=f"TEST{name}3",
                    security_spec_base="ON",
                    bdi_code="02",
                    market_type=10,
                    open_brl=price * 0.999,
                    high_brl=price * 1.01,
                    low_brl=price * 0.90,
                    close_brl=price,
                    volume_brl=30_000_000.0,
                    trades=100.0,
                    quantity=100_000.0,
                    distribution_number=2 if name == 0 and day >= event else 1,
                    currency="BRL",
                    quote_factor=1.0,
                )
            )
    daily = pl.DataFrame(rows)
    schedule = tuple(
        SessionDefinition(
            trade_date=d,
            continuous_open=time(10),
            decision_time=time(15, 45),
            continuous_close=time(16, 45),
            auction_close=time(17),
            source="fixture",
        )
        for d in dates
    )
    provider = pl.DataFrame(
        schema={
            "isin": pl.String,
            "ex_date": pl.Date,
            "action_type": pl.String,
            "split_factor": pl.Float64,
            "cash_distribution_brl": pl.Float64,
            "unresolved": pl.Boolean,
        }
    )
    evaluations = []
    terms = []
    for variant, adjustment in enumerate((0.98, 0.94)):
        changed = daily.with_columns(
            pl.when(
                (pl.col("trade_date") == dates[event]) & (pl.col("isin") == names[0])
            )
            .then(pl.col("close_brl") * adjustment)
            .otherwise(pl.col("close_brl"))
            .alias("close_brl")
        )
        root = build_daily_store(
            changed,
            provider,
            tmp_path / f"store_{variant}",
            session_schedule=schedule,
            minimum_rank_names=20,
            store_start=None,
            action_terms_source="inferred_cotahist_dismes_v1",
            action_acquisition_audit=pl.DataFrame(
                {
                    "isin": list(names),
                    "first_date": [dates[0]] * 40,
                    "last_date": [dates[-1]] * 40,
                    "status": ["downloaded"] * 40,
                    "action_rows": [0] * 40,
                    "economic_terms_complete": [True] * 40,
                }
            ),
        )
        indices = np.arange(68, 75, dtype=np.int64)
        store, _ = open_store_for_samples(
            root,
            indices,
            purpose="evaluation",
            history_lookbacks=20,
            history_end_offsets=-1,
        )
        binding = write_hedge_beta_fixture(
            tmp_path / f"beta_{variant}",
            dates=np.asarray(dates, dtype="datetime64[D]"),
            isins=names,
            store_sha256=sha256_file(root / "manifest.json"),
            bova11_sha256="a" * 64,
        )
        scores = (
            np.broadcast_to(np.arange(40)[None, :, None], (7, 40, len(HORIZONS)))
            .astype(np.float32)
            .copy()
        )
        # A held-name exit on the event day ensures this is a nonempty order test.
        scores[2:, 0] = 50
        try:
            inputs = _evaluation_inputs(
                store,
                indices,
                scores,
                np.ones_like(scores, dtype=np.bool_),
                np.zeros(75),
                np.full(75, 100.0),
                {"manifest_sha256": "a" * 64, "data_sha256": "b" * 64, **binding},
                _borrow_panels(75, 40),
                {},
                transfer_chronology_clean=True,
            )
            terms.append(inputs.action_cash_per_prior_share[2, 0])
            evaluations.append(
                evaluate_scores(
                    replace(inputs, source_artifact_hashes={"fixture": "c" * 64}),
                    window_name="causality_fixture",
                )
            )
        finally:
            store.close()
    assert terms[0] != terms[1]
    intentions = [
        [
            o
            for o in e.report["economics"]["headline_audit"]["intended_orders"]
            if o["decision_session"] == 2
        ]
        for e in evaluations
    ]
    assert intentions[0]
    assert intentions[0] == intentions[1]
    assert (
        evaluations[0].report["economics"]["headline_audit"]["fills"]
        != evaluations[1].report["economics"]["headline_audit"]["fills"]
    )
