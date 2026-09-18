import numpy as np
import torch

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.foundation_readouts import compare, paired_interval
from brazil_rv.v2.portfolio_readouts import interval


def test_original_average_uses_only_epochs_ending_at_raw_selection(tmp_path):
    from brazil_rv.v2.artifacts import sha256_file
    from brazil_rv.v2.foundation_averaging import prepare

    source = tmp_path / "source"
    source.mkdir()
    contract = {"code": {"commit": "a" * 40}}
    states = []
    for epoch in (1, 2, 3, 4):
        path = source / f"epoch_{epoch}.pt"
        torch.save(
            {
                "stage": "F",
                "fold": "F2",
                "seed": 11,
                "epoch": epoch,
                "contract": contract,
                "model_state_dict": {"weight": torch.tensor(float(epoch))},
                "selection_ic": 0.03,
            },
            path,
        )
        if epoch <= 3:
            states.append(
                {
                    "epoch": epoch,
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "available": True,
                }
            )
    write_json_atomic(tmp_path / "frozen_design.json", {"store": {"root": "unused"}})
    write_json_atomic(
        tmp_path / "inventory.json",
        {
            "trajectories": {
                "C6/F2/11": {
                    "available": True,
                    "selected_epoch": 3,
                    "selected": {
                        "path": str(source / "epoch_3.pt"),
                        "sha256": sha256_file(source / "epoch_3.pt"),
                    },
                    "trailing_states": states,
                }
            }
        },
    )
    prepare(tmp_path)
    result = torch.load(
        tmp_path / "checkpoint_average/C6/F2_seed_11/selected.pt", weights_only=True
    )
    assert result["model_state_dict"]["weight"] == 2.0
    assert result["epoch"] == 3
    prepare(tmp_path)  # Idempotent reuse is hash-bound.


def test_paired_bounds_match_registered_block_sampling_and_preserve_constant_delta():
    rng = np.random.default_rng(19)
    arrays = [rng.normal(size=105), rng.normal(size=113)]
    actual, existing = paired_interval(arrays), interval(arrays)
    assert actual["lower_95"] == existing["lower_95"]
    assert actual["upper_95"] == existing["upper_95"]
    assert (
        actual["lower_95"]
        < actual["lower_90"]
        < actual["upper_90"]
        < actual["upper_95"]
    )


def test_noninferiority_is_not_a_significance_failure_and_ema_needs_improvement(
    tmp_path,
):
    folds = ["F2", "F6", "F10", "F14"]
    for fold in folds:
        for member in ("11", "29", "47", "ensemble"):
            for cell, delta in (("control", 0.0), ("similar", -0.1), ("worse", -0.3)):
                directory = tmp_path / "books" / cell / "raw" / fold / member
                write_json_atomic(
                    directory / "book.json",
                    {
                        "dates": list(range(100)),
                        "daily": {"net_excess_bps": [1.0 + delta] * 100},
                    },
                )
                write_json_atomic(
                    directory / "forecast_readout.json",
                    {
                        "dates": list(range(100)),
                        "neutral_ic": [0.025 + delta * 0.001] * 100,
                    },
                )
    similar = compare(tmp_path, "similar", "control", folds)
    assert similar["screen_noninferiority"] and similar["admitted"]
    assert not similar["screen_improvement"]
    assert not compare(
        tmp_path, "similar", "control", folds, allow_noninferiority=False
    )["admitted"]
    assert not compare(tmp_path, "worse", "control", folds)["admitted"]
    archived = {
        (fold, member): tmp_path / "books/control/raw" / fold / member
        for fold in folds
        for member in ("11", "29", "47", "ensemble")
    }
    reused = compare(
        tmp_path, "similar", "archived_control", folds, control_paths=archived
    )
    assert reused["metrics"] == similar["metrics"]
