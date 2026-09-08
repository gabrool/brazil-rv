from __future__ import annotations

import copy
import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import brazil_rv.v2.research_rounds as research_rounds
from brazil_rv.v2.artifacts import sha256_file
from brazil_rv.v2.contract import HORIZONS, SLOW_FEATURES
from brazil_rv.v2.evaluate import EvaluationInputs, _input_hashes, evaluate_scores
from brazil_rv.v2.research_rounds import (
    RESEARCH_SCORE_SCHEMA,
    RUNG_GROUPS,
    _ResearchEvaluation,
    _economics_not_worse,
    _evaluation_from_artifacts,
    _fold_indices,
    _folded_bootstrap,
    _feature_names,
    _gbdt_features,
    _paired_readouts,
    _point_is_negative,
    _pretrain_indices,
    _resolved_sidecar_groups,
    _score_artifact,
    _small_interval_spanning_zero,
    _store_build_implementation_commit,
    _verify_development_acceptance,
    _window_target_mask,
    seal_root,
)
from brazil_rv.v2.splits import development_folds


def test_rev3_registration_replaces_the_voided_research_entrypoints(
    tmp_path: Path,
) -> None:
    assert research_rounds.PREREGISTRATION.name == "v2_round1_round2_rev3.md"
    assert research_rounds.PREREGISTRATION.is_file()
    with pytest.raises(FileNotFoundError):
        research_rounds.run_round1(output_root=tmp_path / "absent", num_threads=1)


def test_rev3_machine_protocol_matches_folds_evaluator_ledger_and_sources() -> None:
    protocol = research_rounds.load_registration_protocol()
    assert protocol == research_rounds.registration_protocol_from_code()

    raw_dates = np.arange(
        np.datetime64("2010-01-04"),
        np.datetime64("2025-01-01"),
        dtype="datetime64[D]",
    )
    dates = tuple(
        raw_dates[np.is_busday(raw_dates)].astype("datetime64[D]").astype(object)
    )
    folds = development_folds(dates)
    assert protocol["purge_sessions"] == {
        "fit_to_selection": 10,
        "selection_to_evaluation": 10,
    }
    assert protocol["selection_sessions"] == 55
    assert protocol["cross_fit"] == "none"
    assert protocol["models_per_fold_seed"] == 1
    for fold in folds:
        registered = protocol["evaluation_window_per_fold"][fold.name]
        assert registered == {
            "start": fold.evaluation_dates[0].isoformat(),
            "end": fold.evaluation_dates[-1].isoformat(),
        }
    assert (
        protocol["primary_population_rule"]
        == research_rounds.primary_population_protocol()
    )
    assert protocol["headline_cell"] == research_rounds.headline_ledger_protocol()
    assert protocol["source_tier_labels"] == {
        "action_terms_source": "inferred_cotahist_dismes_v1",
        "schedule_source": "reconstructed_v1",
    }


def test_registration_protocol_requires_ineligible_hold_sessions(
    tmp_path: Path,
) -> None:
    registered = research_rounds.PREREGISTRATION.read_text(encoding="utf-8")
    stale = tmp_path / "stale_registration.md"
    stale.write_text(
        registered.replace('      "ineligible_hold_sessions": 5,\n', "", 1),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="machine-readable registration protocol"):
        research_rounds.verify_registration_protocol(stale)
    assert (
        research_rounds.verify_registration_protocol()["headline_cell"]["ledger"][
            "ineligible_hold_sessions"
        ]
        == 5
    )


def test_acceptance_binds_store_hash_and_store_build_separately_from_freeze(
    tmp_path: Path,
) -> None:
    store_build_commit = "a" * 40
    store_manifest = {"metadata": {"implementation_git_commit": store_build_commit}}
    acceptance_implementation = {
        "commit": "b" * 40,
        "tracked_worktree_clean": True,
    }
    source_tiers = {
        "action_terms_source": "inferred_cotahist_dismes_v1",
        "schedule_source": "reconstructed_v1",
    }
    report = {
        "schema": "BRAZIL_RV_V2_PIPELINE_VALIDATION_V9",
        "status": "completed",
        "engineering_acceptance_status": "development_grade_inferred_actions",
        "research_claim": False,
        "official_validation_accessed": False,
        "test_accessed": False,
        "transfer_chronology_clean": True,
        "code": acceptance_implementation,
        "store_build_implementation_commit": store_build_commit,
        "sources": {"store": {"manifest_sha256": "c" * 64}},
        "results": {"development_acceptance": {"labels": source_tiers, "reasons": []}},
    }
    path = tmp_path / "acceptance.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    assert _store_build_implementation_commit(store_manifest) == store_build_commit
    verified = _verify_development_acceptance(
        path,
        expected_sha256=sha256_file(path),
        store_manifest_sha256="c" * 64,
        expected_implementation=acceptance_implementation,
        store_build_implementation_commit=store_build_commit,
        source_tiers=source_tiers,
    )
    assert verified["code"] == acceptance_implementation

    with pytest.raises(ValueError, match="store-build provenance"):
        _verify_development_acceptance(
            path,
            expected_sha256=sha256_file(path),
            store_manifest_sha256="c" * 64,
            expected_implementation=acceptance_implementation,
            store_build_implementation_commit="d" * 40,
            source_tiers=source_tiers,
        )


def _evaluation_pair() -> tuple[_ResearchEvaluation, _ResearchEvaluation]:
    day_count = 25
    name_count = 80
    raw_dates = np.busday_offset(
        np.datetime64("2024-01-02"), np.arange(day_count), roll="forward"
    )
    dates = tuple(raw_dates.astype("datetime64[D]").astype(object).tolist())
    name_rank = np.arange(name_count, dtype=np.float64)[None, :, None]
    horizon_shift = np.arange(len(HORIZONS), dtype=np.float64)[None, None, :]
    targets = np.broadcast_to(
        name_rank + 0.01 * horizon_shift,
        (day_count, name_count, len(HORIZONS)),
    ).copy()
    target_mask = np.ones_like(targets, dtype=np.bool_)
    for index, horizon in enumerate(HORIZONS):
        target_mask[-horizon:, :, index] = False
    active = np.ones((day_count, name_count), dtype=np.bool_)
    raw_close = (
        100.0
        + np.arange(day_count, dtype=np.float64)[:, None] * 0.01
        + np.arange(name_count, dtype=np.float64)[None, :] * 0.1
    )
    identity_successor = np.broadcast_to(
        np.arange(name_count, dtype=np.int64)[None, :],
        (day_count, name_count),
    ).copy()
    common = {
        "dates": dates,
        "session_indices": np.arange(1, 1 + day_count, dtype=np.int64),
        "calendar_identity_sha256": "a" * 64,
        "scaled_midrank_targets": targets,
        "scaled_target_mask": target_mask,
        "neutral_midrank_targets": targets.copy(),
        "neutral_target_mask": target_mask.copy(),
        "shareholder_midrank_targets": targets.copy(),
        "shareholder_simple_returns": targets * 0.0001,
        "shareholder_target_mask": target_mask.copy(),
        "price_midrank_targets": targets.copy(),
        "price_target_mask": target_mask.copy(),
        "active": active,
        "raw_close": raw_close,
        "action_shares_per_prior_share": np.ones_like(raw_close),
        "action_cash_per_prior_share": np.zeros_like(raw_close),
        "action_session_resolved": np.ones_like(active),
        "action_has_action": np.zeros_like(active),
        "action_successor_index": identity_successor,
        "action_payment_session": np.full_like(identity_successor, -1),
        "security_ids": tuple(f"BRTEST{index:04d}" for index in range(name_count)),
        "target_scale_sigma": np.full_like(raw_close, 0.02),
        "history_age_sessions": np.zeros_like(raw_close, dtype=np.float64),
        "initial_reference_price": np.full(name_count, np.nan, dtype=np.float64),
        "eventual_survives_to_final_year": np.broadcast_to(
            np.arange(name_count)[None, :] < name_count // 2,
            (day_count, name_count),
        ).copy(),
        "prior_feature_values": {
            name: np.zeros_like(raw_close, dtype=np.float32)
            for name in (
                "yang_zhang_vol_20",
                "beta_60",
                "log_volume_mean_20",
                "momentum_12_1",
                "log_return_5",
            )
        },
        "cdi_returns": np.zeros(day_count, dtype=np.float64),
        "source_artifact_hashes": {"fixture": "b" * 64},
        "source_archive_present": {"lending": np.ones_like(active)},
        "source_feature_valid": {"lending": np.ones_like(active)},
        "annual_borrow_rate_by_name": np.full_like(raw_close, 0.02),
        "shortable": np.ones_like(active),
    }
    for key in (
        "scaled_midrank_targets",
        "neutral_midrank_targets",
        "shareholder_midrank_targets",
        "shareholder_simple_returns",
        "price_midrank_targets",
    ):
        common[key] = np.where(target_mask, common[key], 0.0)
    common["neutral_target_mask"] = target_mask.copy()
    base_scores = targets.copy()
    candidate_mask = np.ones_like(target_mask)
    baseline_mask = np.ones_like(target_mask)
    candidate_mask[:, :5] = False
    baseline_mask[:, -5:] = False
    candidate_inputs = EvaluationInputs(
        scores=base_scores,
        score_mask=candidate_mask,
        transfer_chronology_clean=True,
        **common,
    )
    baseline_inputs = EvaluationInputs(
        scores=-base_scores,
        score_mask=baseline_mask,
        # Cleanliness is not a paired-population identity key for labelled
        # development diagnostics. Official evaluation rejects this input.
        transfer_chronology_clean=False,
        **common,
    )
    candidate = _ResearchEvaluation(
        result=evaluate_scores(candidate_inputs, window_name="fixture"),
        inputs=candidate_inputs,
    )
    baseline = _ResearchEvaluation(
        result=evaluate_scores(baseline_inputs, window_name="fixture"),
        inputs=baseline_inputs,
    )
    return candidate, baseline


def test_folded_bootstrap_never_crosses_fold_boundaries_or_drops_dates() -> None:
    left = np.arange(25, dtype=np.float64)
    right = np.arange(25, 50, dtype=np.float64)
    right[::2] = np.nan
    result = _folded_bootstrap((left, right), replications=100)
    assert result["fold_boundary_preserved"] is True
    assert result["possible_observations"] == 50
    assert result["finite_observations"] == 37
    assert result["estimate"] == pytest.approx(
        np.nanmean(np.concatenate((left, right)))
    )
    assert result["undefined_reason"] is None


def test_folded_bootstrap_records_undefined_readout_as_null() -> None:
    result = _folded_bootstrap(
        (np.full(25, np.nan), np.full(25, np.nan)),
        replications=100,
    )
    assert result["estimate"] is None
    assert result["lower_95"] is None
    assert result["upper_95"] is None
    assert result["possible_observations"] == 50
    assert result["finite_observations"] == 0
    assert result["undefined_reason"] == "no_defined_daily_values"


def test_folded_bootstrap_serializes_sparse_unsupported_interval_as_null() -> None:
    values = np.full(40, np.nan, dtype=np.float64)
    values[-1] = 2.0

    result = _folded_bootstrap((values,), replications=1, seed=0)

    assert result["estimate"] == 2.0
    assert result["lower_95"] is None
    assert result["upper_95"] is None
    assert result["finite_bootstrap_replications"] == 0
    assert result["undefined_reason"] == "no_finite_bootstrap_draws"
    json.dumps(result, allow_nan=False)


def test_round2_joint_plan_expects_both_training_graphs() -> None:
    common = {
        "name": "job",
        "seed": 11,
        "fold": "F1",
        "run_dir": Path("run"),
        "command": ["python"],
        "source_tiers": {
            "action_terms_source": "inferred_cotahist_dismes_v1",
            "schedule_source": "reconstructed_v1",
        },
    }

    joint = research_rounds._plan_job(stage="J", **common)
    fine = research_rounds._plan_job(stage="F", **common)

    assert joint["expected_manifest"]["compiled_graph_count"] == 3
    assert joint["expected_manifest"]["compiled_graphs"] == {
        "training": 2,
        "selection": 1,
        "total": 3,
    }
    assert fine["expected_manifest"]["compiled_graph_count"] == 2
    assert fine["expected_manifest"]["compiled_graphs"]["training"] == 1


def test_undefined_economics_cannot_establish_not_worse() -> None:
    undefined = {"estimate": None, "lower_95": None, "upper_95": None}
    positive = {"estimate": 1.0, "lower_95": 0.5, "upper_95": 1.5}
    assert _point_is_negative(undefined) is False
    assert _economics_not_worse(undefined, positive) is False
    assert _economics_not_worse(positive, undefined) is False
    assert _small_interval_spanning_zero(undefined) is False


def test_paired_readouts_use_the_exact_common_four_head_population() -> None:
    candidate, baseline = _evaluation_pair()
    candidate_folds = {fold: candidate for fold in ("F1", "F2", "F3")}
    baseline_folds = {fold: baseline for fold in ("F1", "F2", "F3")}

    paired = _paired_readouts(candidate_folds, baseline_folds)

    assert set(paired["pooled"]) == {
        "primary_neutral_target_ic",
        "shareholder_rank_ic",
        "price_return_rank_ic",
        "persistence_1",
        "persistence_5",
        "shareholder_return_spread_bps_per_holding_session",
        "headline_net_excess_bps",
    }
    primary = paired["pooled"]["primary_neutral_target_ic"]
    assert primary["estimate"] == pytest.approx(2.0)
    assert primary["possible_observations"] == 75
    assert primary["finite_observations"] == 60
    assert paired["pooled"]["shareholder_rank_ic"]["estimate"] == pytest.approx(2.0)
    assert paired["pooled"]["price_return_rank_ic"]["estimate"] == pytest.approx(2.0)
    assert paired["pooled"]["persistence_1"]["estimate"] == pytest.approx(0.0)
    first = paired["population_audit"]["F1"]["primary_neutral_target_ic"][0]
    assert first["common_candidate_baseline_name_count"] == 70
    assert first["delta"] == pytest.approx(2.0)
    assert first["undefined_reason"] is None


def test_pretrain_samples_include_first_canonical_decision_snapshot() -> None:
    dates = np.asarray(
        ["2010-01-04", "2010-01-05", "2010-01-06"], dtype="datetime64[D]"
    )
    assert _pretrain_indices(dates).tolist() == [0, 1, 2]


def test_round_fit_targets_may_end_in_registered_purge_before() -> None:
    dates = np.arange(
        np.datetime64("2010-01-04"),
        np.datetime64("2025-01-01"),
        dtype="datetime64[D]",
    )
    dates = dates[np.is_busday(dates)]
    fit, _, _, fit_target_window, _ = _fold_indices(dates)
    values = np.ones((len(fit["F1"]), 1, len(HORIZONS)), dtype=np.bool_)

    fit_only = _window_target_mask(values, fit["F1"])
    registered = _window_target_mask(
        values,
        fit["F1"],
        target_window_indices=fit_target_window["F1"],
    )

    assert not fit_only[-1].any()
    assert registered[-1, 0, HORIZONS.index(10)]


def test_paired_economics_is_entirely_undefined_if_either_fold_is_unresolved() -> None:
    candidate, baseline = _evaluation_pair()
    report = copy.deepcopy(candidate.result.report)
    report["economics"]["headline"]["economics_unresolved"] = True
    unresolved = _ResearchEvaluation(
        result=replace(candidate.result, report=report),
        inputs=candidate.inputs,
    )

    paired = _paired_readouts(
        {fold: unresolved for fold in ("F1", "F2", "F3")},
        {fold: baseline for fold in ("F1", "F2", "F3")},
    )

    economics = paired["pooled"]["headline_net_excess_bps"]
    assert economics["estimate"] is None
    assert economics["finite_observations"] == 0
    assert economics["possible_observations"] == 75


def test_evaluation_reconstruction_uses_hash_bound_scores_and_canonical_store(
    tmp_path: Path,
) -> None:
    candidate, _ = _evaluation_pair()
    inputs = candidate.inputs
    indices = np.asarray(inputs.session_indices, dtype=np.int64)
    day_count, name_count = np.asarray(inputs.active).shape

    def prefixed(values: np.ndarray) -> np.ndarray:
        return np.concatenate((np.zeros_like(values[:1]), values), axis=0)

    arrays = {
        "target_valid": prefixed(np.asarray(inputs.scaled_target_mask)),
        "target_primary_neutral_valid": prefixed(
            np.asarray(inputs.neutral_target_mask)
        ),
        "target_shareholder_valid": prefixed(
            np.asarray(inputs.shareholder_target_mask)
        ),
        "target_price_valid": prefixed(np.asarray(inputs.price_target_mask)),
        "target_primary": prefixed(np.asarray(inputs.scaled_midrank_targets)),
        "target_primary_neutral": prefixed(np.asarray(inputs.neutral_midrank_targets)),
        "target_shareholder_midrank": prefixed(
            np.asarray(inputs.shareholder_midrank_targets)
        ),
        "target_shareholder_simple_return": prefixed(
            np.asarray(inputs.shareholder_simple_returns)
        ),
        "target_price_midrank": prefixed(np.asarray(inputs.price_midrank_targets)),
        "active": prefixed(np.asarray(inputs.active)),
        "raw_close": prefixed(np.asarray(inputs.raw_close)),
        "prior_reference_close": np.full(
            (day_count + 1, name_count), np.nan, dtype=np.float64
        ),
        "audit_eventual_survives_to_final_year": prefixed(
            np.asarray(inputs.eventual_survives_to_final_year)
        ),
        "action_shares_per_prior_share": prefixed(
            np.asarray(inputs.action_shares_per_prior_share)
        ),
        "action_cash_per_prior_share": prefixed(
            np.asarray(inputs.action_cash_per_prior_share)
        ),
        "action_session_resolved": prefixed(np.asarray(inputs.action_session_resolved)),
        "action_has_action": prefixed(np.asarray(inputs.action_has_action)),
        "action_successor_index": prefixed(np.asarray(inputs.action_successor_index)),
        "action_payment_session": prefixed(np.asarray(inputs.action_payment_session)),
        "target_scale_sigma": prefixed(np.asarray(inputs.target_scale_sigma)),
        "slow_values": np.zeros(
            (day_count + 1, name_count, len(SLOW_FEATURES)), dtype=np.float32
        ),
        "slow_valid": np.ones(
            (day_count + 1, name_count, len(SLOW_FEATURES)), dtype=np.bool_
        ),
        "sidecar_lending_values": np.full(
            (day_count + 1, name_count, 1), np.arcsinh(2.0), dtype=np.float32
        ),
        "sidecar_lending_valid": np.ones(
            (day_count + 1, name_count, 1), dtype=np.bool_
        ),
        "sidecar_lending_age_sessions": np.zeros(
            (day_count + 1, name_count, 1), dtype=np.float32
        ),
    }

    fixture_dates = [date(2024, 1, 1), *inputs.dates]
    lending_rate_path = tmp_path / "bdi_lending_strong.parquet"
    lending_rate_encoded = np.tanh(np.log1p(2.0) / 2.0)
    pl.DataFrame(
        {
            "source_trade_date": pl.Series(
                np.repeat(fixture_dates[:-1], name_count).tolist(), dtype=pl.Date
            ),
            "available_date": pl.Series(
                np.repeat(fixture_dates[1:], name_count).tolist(), dtype=pl.Date
            ),
            "security_id": np.tile(
                [f"ISIN:{isin}" for isin in inputs.security_ids], day_count
            ),
            "lending_taker_fee_level_log_tanh": np.full(
                day_count * name_count, lending_rate_encoded, dtype=np.float64
            ),
            "lending_taker_fee_level_log_tanh_mask": np.ones(
                day_count * name_count, dtype=np.bool_
            ),
        }
    ).write_parquet(lending_rate_path)

    class FixtureStore:
        manifest = {
            "axes": {"date_identity_sha256": inputs.calendar_identity_sha256},
            "feature_names": {
                "slow": list(SLOW_FEATURES),
                "sidecar_lending": ["loan_rate"],
            },
            "metadata": {
                "action_terms_source": inputs.action_terms_source,
                "schedule_source": inputs.schedule_source,
                "corporate_action_contract": {
                    "stored_action_arrays": ("retrospective outcome/accounting terms")
                },
            },
            "sources": [
                {
                    "path": str(lending_rate_path),
                    "bytes": lending_rate_path.stat().st_size,
                    "sha256": sha256_file(lending_rate_path),
                }
            ],
        }
        dates = np.asarray(fixture_dates, dtype="datetime64[D]")
        isins = inputs.security_ids

        def read(self, name: str, selector: np.ndarray) -> np.ndarray:
            return np.asarray(arrays[name][selector]).copy()

        def read_target(
            self, name: str, selector: np.ndarray, *, valid_mask: np.ndarray
        ) -> np.ndarray:
            values = self.read(name, selector)
            return np.where(valid_mask, values, 0.0)

    scores_path = tmp_path / "scores.npy"
    mask_path = tmp_path / "score_mask.npy"
    np.save(scores_path, inputs.scores, allow_pickle=False)
    np.save(mask_path, inputs.score_mask, allow_pickle=False)
    (tmp_path / "score_manifest.json").write_text(
        json.dumps(
            {
                "schema": RESEARCH_SCORE_SCHEMA,
                "status": "completed",
                "official_validation_accessed": False,
                "test_accessed": False,
                "transfer_chronology_clean": True,
                "action_terms_source": "inferred_cotahist_dismes_v1",
                "schedule_source": "reconstructed_v1",
                "metadata": {"evaluation_date_indices": indices.tolist()},
                "artifacts": {
                    "scores.npy": {
                        "bytes": scores_path.stat().st_size,
                        "sha256": sha256_file(scores_path),
                    },
                    "score_mask.npy": {
                        "bytes": mask_path.stat().st_size,
                        "sha256": sha256_file(mask_path),
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    path = tmp_path / "evaluation.json"
    path.write_text(json.dumps(candidate.result.report), encoding="utf-8")
    cdi = np.zeros(day_count + 1, dtype=np.float64)

    reconstructed = _evaluation_from_artifacts(
        path,
        store=FixtureStore(),
        indices=indices,
        cdi=cdi,
    )

    assert _input_hashes(reconstructed.inputs) == _input_hashes(inputs)
    assert np.array_equal(
        reconstructed.result.primary_score_mask,
        candidate.result.primary_score_mask,
    )
    assert np.allclose(
        reconstructed.result.daily_primary_ic,
        candidate.result.daily_primary_ic,
        equal_nan=True,
    )

    manifest_path = tmp_path / "score_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metadata"] = {"fold": "F1"}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="axis differs"):
        _evaluation_from_artifacts(
            path,
            store=FixtureStore(),
            indices=indices,
            cdi=cdi,
        )
    recovered = _evaluation_from_artifacts(
        path,
        store=FixtureStore(),
        indices=indices,
        cdi=cdi,
        allow_legacy_missing_indices=True,
        expected_fold="F1",
    )
    assert _input_hashes(recovered.inputs) == _input_hashes(inputs)
    with pytest.raises(ValueError, match="fold differs"):
        _evaluation_from_artifacts(
            path,
            store=FixtureStore(),
            indices=indices,
            cdi=cdi,
            allow_legacy_missing_indices=True,
            expected_fold="F2",
        )


@pytest.mark.parametrize(
    ("schema", "clean", "error", "message"),
    (
        (
            "BRAZIL_RV_V2_RESEARCH_SCORE_V1",
            True,
            ValueError,
            "incomplete score artifact",
        ),
        (
            RESEARCH_SCORE_SCHEMA,
            False,
            PermissionError,
            "contaminated transfer chronology",
        ),
    ),
)
def test_score_artifact_rejects_stale_or_contaminated_manifest_before_arrays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    schema: str,
    clean: bool,
    error: type[Exception],
    message: str,
) -> None:
    payload = {
        "schema": schema,
        "status": "completed",
        "official_validation_accessed": False,
        "test_accessed": False,
        "transfer_chronology_clean": clean,
        "action_terms_source": "inferred_cotahist_dismes_v1",
        "schedule_source": "reconstructed_v1",
        "artifacts": {},
    }
    (tmp_path / "score_manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        np,
        "load",
        lambda *args, **kwargs: pytest.fail("score array was opened"),
    )

    with pytest.raises(error, match=message):
        _score_artifact(tmp_path, require_clean_transfer=True)


def test_evaluation_rejects_contaminated_report_before_score_array_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate, _ = _evaluation_pair()
    report = copy.deepcopy(candidate.result.report)
    report["transfer_chronology_clean"] = False
    path = tmp_path / "evaluation.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(
        np,
        "load",
        lambda *args, **kwargs: pytest.fail("score or store payload was opened"),
    )

    with pytest.raises(PermissionError, match="contaminated or unknown"):
        _evaluation_from_artifacts(
            path,
            store=object(),
            indices=np.asarray([1], dtype=np.int64),
            cdi=np.zeros(2, dtype=np.float64),
        )


def test_network_score_artifact_binds_date_security_and_feature_axes(
    tmp_path: Path,
) -> None:
    arrays = {
        "scores.npy": np.zeros((2, 3, len(HORIZONS)), dtype=np.float32),
        "score_mask.npy": np.ones((2, 3, len(HORIZONS)), dtype=np.bool_),
        "date_index.npy": np.asarray(
            ["2024-01-02", "2024-01-03"], dtype="datetime64[D]"
        ),
        "isin_index.npy": np.asarray(["BR1", "BR2", "BR3"]),
    }
    records = {}
    for name, values in arrays.items():
        path = tmp_path / name
        np.save(path, values, allow_pickle=False)
        records[name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    (tmp_path / "score_manifest.json").write_text(
        json.dumps(
            {
                "schema": "BRAZIL_RV_V2_SCORE_ARTIFACT_V2",
                "status": "completed",
                "official_validation_accessed": False,
                "test_accessed": False,
                "transfer_chronology_clean": True,
                "action_terms_source": "inferred_cotahist_dismes_v1",
                "schedule_source": "reconstructed_v1",
                "feature_schema_sha256": "c" * 64,
                "artifacts": records,
            }
        ),
        encoding="utf-8",
    )

    scores, mask = _score_artifact(
        tmp_path,
        require_clean_transfer=True,
        expected_dates=arrays["date_index.npy"],
        expected_isins=("BR1", "BR2", "BR3"),
        expected_feature_schema_sha256="c" * 64,
    )
    assert scores.shape == mask.shape == (2, 3, len(HORIZONS))
    with pytest.raises(ValueError, match="date axis"):
        _score_artifact(
            tmp_path,
            require_clean_transfer=True,
            expected_dates=np.asarray(
                ["2024-01-03", "2024-01-04"], dtype="datetime64[D]"
            ),
            expected_isins=("BR1", "BR2", "BR3"),
            expected_feature_schema_sha256="c" * 64,
        )


def test_registered_gbdt_ladder_is_exact_and_cumulative() -> None:
    assert tuple(RUNG_GROUPS) == (
        "a_slow",
        "b_intraday",
        "c_lending",
        "d_all_sidecars",
    )
    assert RUNG_GROUPS["c_lending"] == ("lending",)
    assert RUNG_GROUPS["d_all_sidecars"] == (
        "lending",
        "oddlot",
        "options",
        "rebalance",
        "events",
        "fundamentals",
    )

    store = type(
        "FixtureStore",
        (),
        {
            "manifest": {
                "feature_names": {
                    "slow": ["slow_a", "slow_b"],
                    "intraday": ["current_a", "current_b"],
                    **{
                        f"sidecar_{group}": [f"{group}_a"]
                        for group in RUNG_GROUPS["d_all_sidecars"]
                    },
                }
            }
        },
    )()
    assert _feature_names(store, "a_slow") == (
        "slow_a",
        "slow_b",
        "slow_a__age_sessions",
        "slow_b__age_sessions",
    )
    assert _feature_names(store, "c_lending") == (
        "slow_a",
        "slow_b",
        "slow_a__age_sessions",
        "slow_b__age_sessions",
        "current_a",
        "current_b",
        "current_a__age_sessions",
        "current_b__age_sessions",
        "fast_present",
        "lending_a",
        "lending_a__age_sessions",
    )


def test_all_sidecars_resolves_materialized_and_explicitly_missing_capabilities() -> (
    None
):
    store = type(
        "Store",
        (),
        {
            "manifest": {
                "feature_names": {
                    "slow": ["slow_a"],
                    "intraday": ["intraday_a"],
                    "sidecar_lending": ["lending_a"],
                    "sidecar_oddlot": ["oddlot_a"],
                },
                "metadata": {
                    "sidecar_capabilities": {
                        "options": {"enabled": [], "source_missing": ["option_a"]},
                        "rebalance": {
                            "enabled": [],
                            "source_missing": ["rebalance_a"],
                        },
                        "events": {"enabled": [], "source_missing": ["event_a"]},
                        "fundamentals": {
                            "enabled": [],
                            "source_missing": ["fundamental_a"],
                        },
                    }
                },
            }
        },
    )()

    assert _resolved_sidecar_groups(store, RUNG_GROUPS["d_all_sidecars"]) == (
        "lending",
        "oddlot",
    )
    assert _feature_names(store, "d_all_sidecars")[-4:] == (
        "lending_a",
        "lending_a__age_sessions",
        "oddlot_a",
        "oddlot_a__age_sessions",
    )

    missing_without_provenance = copy.deepcopy(store.manifest)
    del missing_without_provenance["metadata"]["sidecar_capabilities"]["options"]
    invalid = type("InvalidStore", (), {"manifest": missing_without_provenance})()
    with pytest.raises(ValueError, match="source-missing sidecar capability: options"):
        _resolved_sidecar_groups(invalid, RUNG_GROUPS["d_all_sidecars"])


def test_round_gbdt_adapter_uses_shared_views_and_preserves_frozen_order() -> None:
    arrays = {
        "active": np.asarray([[True, False], [True, True]], dtype=np.bool_),
        "slow_values": np.asarray(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            dtype=np.float32,
        ),
        "slow_valid": np.ones((2, 2, 2), dtype=np.bool_),
        "slow_age_sessions": np.asarray(
            [[[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]],
            dtype=np.float32,
        ),
        "intraday_values": np.asarray(
            [[[10.0, 11.0], [12.0, 13.0]], [[14.0, 15.0], [16.0, 17.0]]],
            dtype=np.float32,
        ),
        "intraday_valid": np.ones((2, 2, 2), dtype=np.bool_),
        "intraday_age_sessions": np.zeros((2, 2, 2), dtype=np.float32),
        "fast_present": np.asarray([[True, True], [False, True]], dtype=np.bool_),
        "sidecar_lending_values": np.asarray(
            [[[20.0], [99.0]], [[22.0], [23.0]]], dtype=np.float32
        ),
        "sidecar_lending_valid": np.asarray(
            [[[True], [False]], [[True], [True]]], dtype=np.bool_
        ),
        "sidecar_lending_age_sessions": np.asarray(
            [[[1.0], [-1.0]], [[3.0], [4.0]]], dtype=np.float32
        ),
    }

    class Store:
        dates = np.asarray(["2024-01-02", "2024-01-03"], dtype="datetime64[D]")
        isins = ("BR1", "BR2")
        manifest = {
            "feature_names": {
                "slow": ["slow_a", "slow_b"],
                "intraday": ["current_a", "current_b"],
                "sidecar_lending": ["lending_a"],
            }
        }

        @staticmethod
        def read(name, indices):
            return arrays[name][indices]

    actual = _gbdt_features(
        Store(),
        np.asarray([0, 1], dtype=np.int64),
        "c_lending",
        pretrain_mask=np.asarray([True, False]),
    )

    assert actual.shape == (2, 2, 11)
    np.testing.assert_array_equal(actual[..., :2], arrays["slow_values"])
    assert np.isnan(actual[0, :, 4:8]).all()
    assert actual[0, :, 8].tolist() == [0.0, 0.0]
    np.testing.assert_array_equal(actual[1, :, 4:6], arrays["intraday_values"][1])
    assert np.isnan(actual[0, 1, 9:11]).all()
    assert actual[1, 1, 9] == 23.0
    assert _feature_names(Store(), "c_lending") == (
        "slow_a",
        "slow_b",
        "slow_a__age_sessions",
        "slow_b__age_sessions",
        "current_a",
        "current_b",
        "current_a__age_sessions",
        "current_b__age_sessions",
        "fast_present",
        "lending_a",
        "lending_a__age_sessions",
    )


def test_superseded_root_seals_literal_contaminated_chronology(
    tmp_path: Path,
) -> None:
    (tmp_path / "superseded.json").write_text(
        json.dumps(
            {
                "status": "superseded_by_fix_pass_3",
                "research_claim": False,
                "official_validation_accessed": False,
                "test_accessed": False,
                "transfer_chronology_clean": False,
            }
        ),
        encoding="utf-8",
    )
    seal_root(root=tmp_path, research_claim=False)
    access = json.loads((tmp_path / "access_audit.json").read_text(encoding="utf-8"))
    inventory = json.loads(
        (tmp_path / "artifact_inventory.json").read_text(encoding="utf-8")
    )
    assert access["research_claim"] is False
    assert access["transfer_chronology_clean"] is False
    assert inventory["research_claim"] is False
    assert inventory["transfer_chronology_clean"] is False
