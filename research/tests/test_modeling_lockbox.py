from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from brazil_rv.modeling.evaluate import load_current_run
from brazil_rv.modeling.run_discovery_campaign import (
    DISCOVERY_FOLDS,
    EXTERNAL_DATA_READOUT_CONTRACT,
    run_campaign,
)


def _manifest(path: Path, *, window: str, frozen: bool) -> None:
    (path / "run_manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "split": {"training": window},
                "frozen_selection": (
                    {"selected_rule": "final_ema_099"} if frozen else None
                ),
            }
        ),
        encoding="utf-8",
    )


def test_discovery_campaign_has_no_split_or_test_control() -> None:
    assert DISCOVERY_FOLDS == ("fold_a", "fold_b")
    assert tuple(inspect.signature(run_campaign).parameters) == (
        "store",
        "output_dir",
        "sidecar_dir",
    )
    assert EXTERNAL_DATA_READOUT_CONTRACT == {
        "primary": "bidirectional_odd_even_crossfit_patience3_raw",
        "secondary": "final_ema_0995",
        "trajectory_rule_reselection": False,
    }


def test_external_evaluation_rejects_discovery_runs(tmp_path: Path) -> None:
    _manifest(tmp_path, window="fold_a", frozen=True)
    with pytest.raises(ValueError, match="official-window"):
        load_current_run(tmp_path)


def test_external_evaluation_requires_an_internal_frozen_rule(
    tmp_path: Path,
) -> None:
    _manifest(tmp_path, window="official", frozen=False)
    with pytest.raises(ValueError, match="frozen selection"):
        load_current_run(tmp_path)
