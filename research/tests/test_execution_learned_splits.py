from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta

import pytest

from brazil_rv.execution.splits import (
    load_purged_training_folds,
    policy_evaluation_slices,
    purged_training_folds,
)


def _dates() -> tuple[date, ...]:
    start = date(2021, 8, 16)
    return tuple(start + timedelta(days=index) for index in range(716))


def _sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_purged_folds_cover_every_date_once_and_embargo_fit_neighbors() -> None:
    dates = _dates()
    manifest = purged_training_folds(dates)

    assert len(manifest.folds) == 5
    assert [len(fold.heldout_dates) for fold in manifest.folds] == [
        144,
        143,
        143,
        143,
        143,
    ]
    assert sorted(
        value for fold in manifest.folds for value in fold.heldout_dates
    ) == list(dates)
    for fold in manifest.folds:
        assert set(fold.fit_dates).isdisjoint(fold.heldout_dates)
        assert set(fold.fit_dates).isdisjoint(fold.embargo_dates)
        start = dates.index(fold.heldout_dates[0])
        stop = dates.index(fold.heldout_dates[-1]) + 1
        assert fold.heldout_dates == dates[start:stop]
        expected_embargo = dates[max(0, start - 5) : start] + dates[stop : stop + 5]
        assert fold.embargo_dates == expected_embargo
        assert (
            fold.fit_dates
            == dates[: max(0, start - 5)] + dates[min(len(dates), stop + 5) :]
        )


def test_purged_manifest_hash_and_file_are_deterministic(tmp_path) -> None:
    left = purged_training_folds(_dates())
    right = purged_training_folds(_dates())
    path = tmp_path / "purged_folds.json"
    left.write(path)

    assert left.sha256 == right.sha256
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["sha256"] == left.sha256
    assert payload["training_date_identity_sha256"] == _sha256(
        [value.isoformat() for value in _dates()]
    )
    for fold in payload["folds"]:
        assert fold["fit_date_identity_sha256"] == _sha256(fold["fit_dates"])
        assert fold["heldout_date_identity_sha256"] == _sha256(fold["heldout_dates"])
        assert fold["embargo_date_identity_sha256"] == _sha256(fold["embargo_dates"])
        nested = dict(fold)
        nested.pop("sha256")
        assert fold["sha256"] == _sha256(nested)
    outer = dict(payload)
    outer.pop("sha256")
    assert payload["sha256"] == _sha256(outer)
    assert load_purged_training_folds(path, _dates()) == left
    with pytest.raises(FileExistsError):
        left.write(path)

    payload["folds"][0]["fit_dates"][0] = payload["folds"][0]["heldout_dates"][0]
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="differs"):
        load_purged_training_folds(tampered, _dates())

    shifted = purged_training_folds(
        tuple(value + timedelta(days=1) for value in _dates())
    )
    shifted_path = tmp_path / "shifted.json"
    shifted.write(shifted_path)
    with pytest.raises(ValueError, match="canonical TRAIN axis"):
        load_purged_training_folds(shifted_path, _dates())


def test_policy_slices_reproduce_c_a_b_windows() -> None:
    fit_c = tuple(date(2022, 1, 1) + timedelta(days=index) for index in range(407))
    select_c = tuple(date(2023, 4, 3) + timedelta(days=index) for index in range(105))
    remaining = tuple(date(2023, 9, 1) + timedelta(days=index) for index in range(204))
    training_dates = fit_c + select_c + remaining
    slices = policy_evaluation_slices(training_dates, training_dates)

    assert tuple(item.name for item in slices) == ("fold_c", "fold_a", "fold_b")
    assert tuple(len(item.dates) for item in slices) == (105, 102, 102)
    assert slices[1].dates == training_dates[512:614]
    assert slices[2].dates == training_dates[614:716]
    assert set(slices[0].dates).isdisjoint(slices[1].dates)
    assert set(slices[1].dates).isdisjoint(slices[2].dates)


def test_split_contract_rejects_unregistered_design_and_malformed_policy_axis() -> None:
    with pytest.raises(ValueError, match="exactly 716"):
        purged_training_folds(_dates()[:-1])
    with pytest.raises(ValueError, match="unique"):
        purged_training_folds(_dates()[:-1] + (_dates()[-2],))
    with pytest.raises(ValueError, match="chronological"):
        purged_training_folds(tuple(reversed(_dates())))
    with pytest.raises(ValueError, match="exactly 5"):
        purged_training_folds(_dates(), fold_count=4)
    with pytest.raises(ValueError, match="exactly 5"):
        purged_training_folds(_dates(), embargo_sessions=4)

    before = tuple(date(2021, 11, 19) + timedelta(days=index) for index in range(500))
    fold_c = tuple(date(2023, 4, 3) + timedelta(days=index) for index in range(105))
    after = tuple(date(2023, 7, 17) + timedelta(days=index) for index in range(111))
    with pytest.raises(ValueError, match="canonical C/A/B"):
        policy_evaluation_slices(before + fold_c + after, before + fold_c + after)

    canonical = before + fold_c + after
    shifted = tuple(value + timedelta(days=1) for value in canonical)
    with pytest.raises(ValueError, match="canonical TRAIN axis"):
        policy_evaluation_slices(shifted, canonical)
