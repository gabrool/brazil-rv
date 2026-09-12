import json

from brazil_rv.v2.round7 import CELLS, pretrain_key
from brazil_rv.v2.round7_program import phase_tasks, trajectory


def test_phase_inventory_reuses_parents_screen_and_confirmation(tmp_path):
    cells = {c["cell"]: c for c in CELLS}
    p = phase_tasks(tmp_path, "anchor_p") + phase_tasks(tmp_path, "pretrain_r")
    assert len(p) == 24
    assert len({(pretrain_key(cells[c]), s) for c, _, _, s in p}) == 24
    assert len(phase_tasks(tmp_path, "anchor_f")) == 42
    assert len(phase_tasks(tmp_path, "calibration")) == 12
    (tmp_path / "budget.json").write_text(json.dumps({"B": 35}))
    assert len(phase_tasks(tmp_path, "screen")) == 168
    (tmp_path / "budget.json").write_text(json.dumps({"B": 60}))
    assert len(phase_tasks(tmp_path, "screen")) == 156
    assert trajectory(tmp_path, "B4", "F2", 11) == tmp_path / "calibration/F2_seed_11"
    assert (
        trajectory(tmp_path, "B4", "F2", 61) == tmp_path / "trajectories/B4/F2_seed_61"
    )
    (tmp_path / "advancement.json").write_text(
        json.dumps({"cells": ["A1", "B1", "B3", "B4", "B9"]})
    )
    assert len(phase_tasks(tmp_path, "confirmation")) == 150
    for leader, parent_count in (("B9", 6), ("A1", 3)):
        (tmp_path / "confirmation_leader.json").write_text(json.dumps({"cell": leader}))
        assert len(phase_tasks(tmp_path, "six_seed_p")) == parent_count
        assert len(phase_tasks(tmp_path, "six_seed_f")) == 84


def test_three_traded_heads_pair_with_five_head_anchor_on_the_same_population():
    import numpy as np
    import pytest
    from dataclasses import replace
    from test_v2_evaluate import _fixture, TRIAGE_PROTOCOL
    from brazil_rv.v2.evaluate import evaluate_scores, paired_comparison

    inputs = _fixture()
    mask = inputs.score_mask.copy()
    mask[..., :2] = False
    c1 = evaluate_scores(replace(inputs, score_mask=mask), window_name="F2")
    s0 = evaluate_scores(inputs, window_name="F2")
    result = paired_comparison(c1, s0, protocol=TRIAGE_PROTOCOL)
    defined = [
        row["delta"]
        for row in result["daily_primary_ic_delta_table"]
        if row["delta"] is not None
    ]
    assert defined and np.max(np.abs(defined)) == pytest.approx(0.0)
