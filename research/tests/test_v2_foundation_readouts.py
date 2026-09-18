import numpy as np

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.foundation_readouts import compare, paired_interval
from brazil_rv.v2.portfolio_readouts import interval


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
