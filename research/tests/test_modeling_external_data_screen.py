from __future__ import annotations

import json
from pathlib import Path

import pytest

from brazil_rv.modeling.contract import ALLOWED_SEEDS
from brazil_rv.modeling.external_data_screen import compare_external_data_campaign


def test_external_screen_rejects_fresh_trajectory_rule_selection(
    tmp_path: Path,
) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    (campaign / "campaign_manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "external_sidecar": {"feature_count": 1},
                "folds": ["fold_a", "fold_b"],
                "seeds": list(ALLOWED_SEEDS),
                "official_validation_accessed": False,
                "test_accessed": False,
                "trajectory_selection": "freshly_selected.json",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="completed external-data"):
        compare_external_data_campaign(campaign, tmp_path / "analysis")
