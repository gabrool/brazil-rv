"""Read-only input support and parameter census; no labels, fits or selection."""

from collections import defaultdict
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.characteristic_model import CharacteristicModel
from brazil_rv.v2.model import DailyMultiHorizonModel
from brazil_rv.v2.objective_program import CELLS
from brazil_rv.v2.round7 import configuration


def main():
    project = Path(__file__).resolve().parents[1]
    pointer = json.loads(
        (project / "docs/v2_data_inputs.json").read_text(encoding="utf-8")
    )
    root = Path(pointer["store"]["root"])
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert sha256_file(root / "manifest.json") == pointer["store"]["manifest_sha256"]
    assert manifest["axes"]["date_end"] <= "2024-12-30"
    names = manifest["feature_names"]
    active = np.load(root / "active.npy", mmap_mode="r")
    dates = np.load(root / "date_index.npy", mmap_mode="r")
    fields = {}
    source_arrays = {}

    def array(key):
        entry = manifest["arrays"][key]
        source_arrays[key] = entry
        return np.load(root / entry["path"], mmap_mode="r")

    for family in ("slow", *sorted(k for k in names if k.startswith("sidecar_"))):
        values, valid = array(family + "_values"), array(family + "_valid")
        age_key = "slow_age_sessions" if family == "slow" else family + "_age_sessions"
        age = array(age_key)
        for column, name in enumerate(names[family]):
            mask = valid[..., column] & active
            observed = values[..., column][mask]
            finite = observed[np.isfinite(observed)]
            known_age = (age[..., column] >= 0) & active
            per_date = mask.sum(1)
            nonempty = np.flatnonzero(per_date)
            fields[f"{family}:{name}"] = {
                "valid_active_stock_days": int(mask.sum()),
                "nonfinite_valid_values": int(len(observed) - len(finite)),
                "known_age_active_stock_days": int(known_age.sum()),
                "minimum": float(finite.min()) if len(finite) else None,
                "maximum": float(finite.max()) if len(finite) else None,
                "constant_observed_value": bool(
                    len(finite) and finite.min() == finite.max()
                ),
                "first_value_date": str(dates[nonempty[0]]) if len(nonempty) else None,
                "last_value_date": str(dates[nonempty[-1]]) if len(nonempty) else None,
                "value_and_age_absent": not bool(mask.any() or known_age.any()),
                "valid_by_year": {
                    str(year): int(
                        per_date[
                            dates.astype("datetime64[Y]").astype(int) + 1970 == year
                        ].sum()
                    )
                    for year in range(2010, 2025)
                },
            }

    models = {}
    attention = configuration(CELLS["TE_all"], names)
    configurations = {
        "C6": configuration(CELLS["C6"], names),
        "TE_all": attention,
        "TE_readout128": replace(attention, width=128, inner_width=128),
        "TE_readout128_two_blocks": replace(
            attention, width=128, inner_width=128, blocks=2
        ),
        "TE_temporal96": replace(attention, hidden_width=96),
    }
    for label, config in configurations.items():
        model = (
            DailyMultiHorizonModel(config)
            if label == "C6"
            else CharacteristicModel(config)
        )
        groups = defaultdict(int)
        for name, value in model.named_parameters():
            groups[name.split(".")[0]] += value.numel()
        models[label] = {
            "config": asdict(config),
            "parameters": sum(groups.values()),
            "parameters_by_module": dict(groups),
        }
    write_json_atomic(
        project / "docs/v2_model_design_audit.json",
        {
            "scope": "Descriptive engineering census only. No labels, training, performance selection or roster changes. Hypothetical model sizes have no measured accuracy or runtime.",
            "script_sha256": sha256_file(Path(__file__)),
            "store": pointer["store"],
            "source_arrays_from_accepted_manifest": source_arrays,
            "active_stock_days": int(active.sum()),
            "fields": fields,
            "models": models,
        },
    )
    print(
        json.dumps(
            {
                "fields": len(fields),
                "absent_value_and_age": [
                    k for k, v in fields.items() if v["value_and_age_absent"]
                ],
                "constant_observed_value": [
                    k for k, v in fields.items() if v["constant_observed_value"]
                ],
                "nonfinite_valid_values": sum(
                    v["nonfinite_valid_values"] for v in fields.values()
                ),
                "models": {k: v["parameters"] for k, v in models.items()},
            }
        )
    )


if __name__ == "__main__":
    main()
