from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import torch

from brazil_rv.modeling.trajectory import ModelEMA
from brazil_rv.v2.foundation_program import input_cells
from brazil_rv.v2.round7 import configuration
from brazil_rv.v2.round7_preprocessing import Round7Preprocessing


def test_input_cells_change_only_registered_families_and_absent_field():
    names = {
        "slow": [f"s{i}" for i in range(32)],
        "sidecar_options": ["observed", "uncovered_call_share"],
        "sidecar_cross_market": ["shock_oil", "exposure_oil"],
    }
    for family in (
        "fundamentals",
        "magnitudes",
        "microstructure",
        "oddlot",
        "sector",
        "rebalance",
    ):
        names["sidecar_" + family] = ["value"]
    cells = input_cells(names)
    full = configuration(cells["TE_full"], names)
    assert dict(full.family_counts)["options"] == 1
    for name, excluded in (
        ("TE_no_micro", {"microstructure"}),
        ("TE_no_weak", {"oddlot", "sector", "rebalance"}),
    ):
        config = configuration(cells[name], names)
        assert (
            set(dict(full.family_counts)) - set(dict(config.family_counts)) == excluded
        )
        a, b = asdict(full), asdict(config)
        a.pop("family_counts")
        b.pop("family_counts")
        assert a == b
    without_market = {**cells["TE_full"], "families": ["fundamentals"]}
    assert configuration(without_market, names).common_field_count == 0


def test_subset_scaler_is_fit_only_and_age_only_observations_survive():
    arrays = {
        "sidecar_options_values": np.array(
            [[[1.0, 999.0, 0.0]], [[3.0, 999.0, 0.0]], [[1e9, -1e9, 0.0]]]
        ),
        "sidecar_options_valid": np.array([[[True, False, False]]] * 3),
        "active": np.ones((3, 1), bool),
    }
    store = SimpleNamespace(
        manifest={
            "feature_names": {"sidecar_options": ["observed", "empty", "age_only"]},
            "metadata": {
                "feature_schema": {
                    "specifications": [
                        {
                            "family": "sidecar_options",
                            "name": n,
                            "transform": "continuous",
                        }
                        for n in ("observed", "empty", "age_only")
                    ]
                }
            },
        },
        read=lambda key, rows: arrays[key][rows],
    )
    dataset = SimpleNamespace(
        store=store, date_indices=np.array([0, 1]), enabled_sidecars=("options",)
    )
    prep = Round7Preprocessing.fit(
        dataset, split_common=False, excluded_fields={"options": ["empty"]}
    )
    arrays["sidecar_options_values"][2] *= -1000
    other = Round7Preprocessing.fit(
        dataset, split_common=False, excluded_fields={"options": ["empty"]}
    )
    assert prep.payload() == other.payload()
    assert prep.families["options"].support == (2, 0)
    assert prep.feature_names["options"] == ("observed", "age_only")
    raw = {
        "sidecar_options_values": arrays["sidecar_options_values"][0],
        "sidecar_options_valid": arrays["sidecar_options_valid"][0],
        "sidecar_options_age_sessions": np.array([[0.0, -1.0, 5.0]]),
    }
    transformed = prep.transform_sample(raw)
    assert transformed["sidecar_options_age_sessions"].tolist() == [[0.0, 5.0]]


def test_epoch_half_life_ema_is_independent_of_update_count():
    for updates in (7, 19):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        ema = ModelEMA(model, 2 ** (-1 / updates))
        with torch.no_grad():
            model.weight.fill_(1.0)
        for _ in range(updates):
            ema.update(model)
        torch.testing.assert_close(
            ema.shadow["weight"], torch.full_like(model.weight, 0.5)
        )
