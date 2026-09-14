from pathlib import Path

import numpy as np
import torch

from brazil_rv.execution.opportunity_policy import OpportunityPolicy
from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2 import controller_program as program
from brazil_rv.v2.controller_synthetic import synthetic_data
from brazil_rv.v2.decision_program import benchmark_excess_returns, calibrations
from brazil_rv.v2.portfolio_program import read


def test_continuous_fallback_loads_actual_checkpoint_and_liquidates(
    tmp_path, monkeypatch
):
    data, _, _ = synthetic_data(days=128)
    bounds = {
        "F1": {
            "fit": np.arange(40),
            "selection": np.arange(40, 50),
            "evaluation": np.arange(50, 80),
        },
        "F2": {
            "fit": np.arange(70),
            "selection": np.arange(70, 80),
            "evaluation": np.arange(80, 128),
        },
    }
    write_json_atomic(
        tmp_path / "phase2/screen_summary.json", {"survivors": [["C6", "reliability"]]}
    )
    for fold, window in bounds.items():
        path = program.policy_path(tmp_path, "C6", fold, "reliability", 11)
        path.mkdir(parents=True)
        mapping = calibrations(data, window["fit"], benchmark_excess_returns(data))[
            "benchmark"
        ]
        model = OpportunityPolicy(data, mapping, window["fit"], kind="reliability")
        torch.save({"model": model.state_dict()}, path / "selected.pt")
        write_json_atomic(
            path / "run_manifest.json",
            {
                "status": "completed",
                "selected_sha256": sha256_file(path / "selected.pt"),
                "fallback": "learned" if fold == "F1" else "cash",
            },
        )
    monkeypatch.setattr(program, "DEVELOPMENT_FOLDS", ("F1", "F2"))
    monkeypatch.setattr(program, "_git_identity", lambda: {"commit": "fixture"})
    monkeypatch.setattr(
        "brazil_rv.v2.portfolio_training.load_data", lambda *_: (data, "fixture")
    )
    monkeypatch.setattr(
        "brazil_rv.v2.portfolio_training.windows",
        lambda root, source, fold: bounds[fold],
    )
    monkeypatch.setattr(
        "brazil_rv.v2.controller_context.load_context", lambda *_: data.context
    )
    program.continuous(tmp_path, "C6")
    output = Path(tmp_path) / "phase2/continuous/C6/fallback"
    record = read(output / "book.json")
    assert len(record["dates"]) == 78
    assert record["summary"]["all_cash_sessions"] >= 47
    assert max(abs(v) for v in record["daily"]["net_excess_bps"][-30:]) < 1e-8
    assert read(output.parent / "comparison.json")["heldout_accessed"] is False
