from __future__ import annotations

import copy
import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from brazil_rv.v2 import evaluate as ev
from brazil_rv.v2 import round5_replay as replay
from brazil_rv.v2.artifacts import canonical_json_bytes, sha256_file, write_json_atomic
from brazil_rv.v2.execution_policy import ExecutionPolicy
from test_v2_evaluate import _fixture


@pytest.mark.parametrize(
    ("settle", "selected", "expected"),
    [
        (
            False,
            False,
            "71866c0aba5eff29870412a81c975b83474df2d73efebd90152e02abf8063d2a",
        ),
        (
            True,
            False,
            "d4162a6b0cc304a55bacc83b8ddaf22ce4b381b16d56c06d65c42b322c2a84d3",
        ),
        (
            True,
            True,
            "92c5a9a1018577c3da195bf41a7f9782d28cd09a4537b8fd6e6b6b7b71d075a4",
        ),
    ],
)
def test_economics_extraction_preserves_entire_pre_refactor_report(
    settle, selected, expected
):
    # Hashes captured before extracting the accounting helper. These cover the
    # entire report, including all accounting and untouched score-only statistics.
    inputs = _fixture()
    if selected:
        inputs = replace(
            inputs,
            execution_policy=ExecutionPolicy(
                horizons=(3, 5, 10), buffer_per_quintile=9
            ),
        )
    report = ev.evaluate_scores(
        inputs, window_name="F1", settle_terminal_residuals=settle
    ).report
    assert hashlib.sha256(canonical_json_bytes(report)).hexdigest() == expected


def _new_provenance(inputs):
    return replace(
        inputs,
        bova11_manifest_sha256="a" * 64,
        bova11_data_sha256="b" * 64,
        hedge_beta_manifest_sha256="c" * 64,
        source_artifact_hashes={
            **inputs.source_artifact_hashes,
            "bova11_manifest": "a" * 64,
            "bova11_data": "b" * 64,
            "hedge_beta_manifest": "c" * 64,
        },
    )


def test_unchanged_numerical_inputs_reuse_exact_accounting(monkeypatch):
    inputs = _fixture()
    original = ev.evaluate_scores(
        inputs, window_name="F1", settle_terminal_residuals=True
    ).report
    changed = _new_provenance(inputs)

    def forbidden(*args, **kwargs):
        raise AssertionError("unchanged numerical inputs must not rerun accounting")

    monkeypatch.setattr(ev, "_evaluate_economics", forbidden)
    result, record = replay.replay_report(original, changed, treatment="bova_only")
    expected = copy.deepcopy(original["economics"])
    expected["contract"]["hedge_beta_manifest_sha256"] = "c" * 64
    assert result["economics"] == expected
    assert record["accounting_reused_by_exact_input_identity"]
    assert replay.protected_projection(
        result, "bova_only"
    ) == replay.protected_projection(original, "bova_only")


def test_repaired_bova_accounting_matches_full_evaluation_without_score_statistics(
    monkeypatch,
):
    repaired = _fixture()
    old_close = np.asarray(repaired.bova11_close).copy()
    old_close[4:9] = np.nan
    original_inputs = replace(repaired, bova11_close=old_close)
    original = ev.evaluate_scores(
        original_inputs, window_name="F4", settle_terminal_residuals=True
    ).report
    changed = _new_provenance(repaired)
    expected = ev.evaluate_scores(
        changed, window_name="F4", settle_terminal_residuals=True
    ).report

    def forbidden(*args, **kwargs):
        raise AssertionError("ledger replay must not recompute score-only statistics")

    monkeypatch.setattr(ev, "_daily_metrics", forbidden)
    monkeypatch.setattr(ev, "_diagnostics", forbidden)
    monkeypatch.setattr(ev, "_bootstrap_payload", forbidden)
    result, record = replay.replay_report(original, changed, treatment="bova_only")
    assert result == expected
    assert record["accounting_recomputed"]


@pytest.mark.parametrize("field", ["scores", "neutral_midrank_targets", "active"])
def test_replay_rejects_any_protected_input_change(field):
    inputs = _fixture()
    original = ev.evaluate_scores(inputs, window_name="F4").report
    values = np.asarray(getattr(inputs, field)).copy()
    values.flat[0] = (
        not values.flat[0] if values.dtype == np.bool_ else values.flat[0] + 0.1
    )
    with pytest.raises(ValueError, match="protected input"):
        replay.replay_report(
            original, replace(inputs, **{field: values}), treatment="bova_only"
        )


def test_replay_does_not_hide_unregistered_source_changes():
    inputs = _fixture()
    original = ev.evaluate_scores(inputs, window_name="F4").report
    changed = replace(
        inputs,
        source_artifact_hashes={
            **inputs.source_artifact_hashes,
            "unregistered_source": "d" * 64,
        },
    )
    with pytest.raises(ValueError, match="protected non-ledger"):
        replay.replay_report(original, changed, treatment="bova_only")


def test_lending_repair_cannot_silently_change_the_shortable_universe():
    inputs = _fixture()
    original = ev.evaluate_scores(inputs, window_name="F4").report
    masks = {
        key: value.copy() for key, value in inputs.shortable_by_borrow_source.items()
    }
    masks["borrow_balance"][0, 0] = ~masks["borrow_balance"][0, 0]
    changed = replace(inputs, shortable_by_borrow_source=masks)
    with pytest.raises(ValueError, match="protected shortability"):
        replay.replay_report(original, changed, treatment="lending_only")


def test_bova_numerical_change_cannot_escape_affected_folds():
    inputs = _fixture()
    original = ev.evaluate_scores(inputs, window_name="F6").report
    with pytest.raises(ValueError, match="outside F4/F5"):
        replay.replay_report(
            original,
            replace(inputs, bova11_close=np.asarray(inputs.bova11_close) * 1.01),
            treatment="bova_only",
        )


def test_observed_rates_change_only_cost_inputs_and_preserve_shortability():
    inputs = _fixture()
    shape = (int(inputs.session_indices[-1]) + 1, len(inputs.security_ids))
    rates = SimpleNamespace(
        annual_taker_rate=np.full(shape, 0.04),
        rate_imputed=np.zeros(shape, dtype=np.bool_),
        rate_placeholder=np.zeros(shape, dtype=np.bool_),
        manifest_sha256="a" * 64,
        rate_sha256="b" * 64,
    )
    changed = replay.ReplayInputs.repaired(
        SimpleNamespace(rates=rates), inputs, "lending_only"
    )
    assert changed.shortable_by_borrow_source is inputs.shortable_by_borrow_source
    assert changed.source_archive_present is inputs.source_archive_present
    assert changed.source_feature_valid is inputs.source_feature_valid
    original = ev.evaluate_scores(
        inputs, window_name="F4", settle_terminal_residuals=True
    ).report
    expected = ev.evaluate_scores(
        changed, window_name="F4", settle_terminal_residuals=True
    ).report
    result, record = replay.replay_report(original, changed, treatment="lending_only")
    assert result == expected
    assert record["accounting_recomputed"]
    with pytest.raises(ValueError, match="no admitted replacement"):
        replay.ReplayInputs.repaired(
            SimpleNamespace(rates=None), inputs, "lending_only"
        )


def test_inventory_enumeration_includes_465_books_and_preserves_rejection(tmp_path):
    registration = {"round4_roots": {}}
    for group in replay.BOOK_PATTERNS:
        root = tmp_path / group
        root.mkdir()
        if group == "screening":
            prefixes = [
                f"aggregates/screening/{arm}/F{fold}"
                for arm in ("S0", "fast_off", "H", "L", "C", "P")
                for fold in range(1, 15)
            ]
            prefixes += [
                f"aggregates/screening/selection_1235/F{fold}" for fold in (12, 13, 14)
            ]
        elif group == "seed_audit":
            prefixes = [
                f"omit_{seed}/{arm}/F{fold}"
                for seed in (11, 29, 47)
                for arm in ("S0", "fast_off", "H", "L", "C", "P")
                for fold in range(1, 15)
            ]
        else:
            prefixes = [
                f"cpu_replay/gbdt/candidate_{candidate}/F{fold}"
                for candidate in range(9)
                for fold in range(1, 15)
            ]
        files = [
            {"path": f"{prefix}/{filename}", "bytes": 0, "sha256": "0" * 64}
            for prefix in prefixes
            for filename in (
                "evaluation.json",
                "scores.npy",
                "score_mask.npy",
                "score_manifest.json",
            )
        ]
        if group == "seed_audit":
            files.append(
                {"path": "omit_29/P/F14/rejected.json", "bytes": 0, "sha256": "0" * 64}
            )
        if group == "cpu_replay":
            write_json_atomic(root / "frozen_design.json", {"source": "sealed"})
            files.append(
                {
                    "path": "frozen_design.json",
                    "bytes": (root / "frozen_design.json").stat().st_size,
                    "sha256": sha256_file(root / "frozen_design.json"),
                }
            )
        write_json_atomic(root / "artifact_inventory.json", {"files": files})
        registration["round4_roots"][group] = {
            "root": str(root),
            "inventory_sha256": sha256_file(root / "artifact_inventory.json"),
        }
    books, design = replay.enumerate_books(registration)
    assert len(books) == 465
    assert design == {"source": "sealed"}
    assert [
        book["key"] for book in books if book["historical_status"] == "rejected"
    ] == ["omit_29/P/F14"]


def test_completed_replay_rejects_mutated_output(tmp_path: Path):
    write_json_atomic(tmp_path / "evaluation.json", {"economics": "verified"})
    book = {"files": {"evaluation.json": {"sha256": "a" * 64}}}
    record = {
        "frozen_design_sha256": "b" * 64,
        "source_evaluation_sha256": "a" * 64,
        "treatment": "bova_only",
        "historical_status": "rejected",
        "output_hashes": {"evaluation.json": sha256_file(tmp_path / "evaluation.json")},
    }
    write_json_atomic(tmp_path / "comparison.json", record)
    assert (
        replay._completed(tmp_path, "b" * 64, book, "bova_only")["historical_status"]
        == "rejected"
    )
    write_json_atomic(tmp_path / "evaluation.json", {"economics": "changed"})
    with pytest.raises(ValueError, match="completed replay file differs"):
        replay._completed(tmp_path, "b" * 64, book, "bova_only")
