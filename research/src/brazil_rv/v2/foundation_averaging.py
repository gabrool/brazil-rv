"""Fixed pre-selected epoch averaging of original neutral F trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from brazil_rv.modeling.trajectory import average_state_dicts
from .artifacts import sha256_file, write_json_atomic
from .foundation_program import read
from .train import _atomic_torch_save


def prepare(root):
    inventory = read(root / "inventory.json")
    prepared = {}
    for key, item in inventory["trajectories"].items():
        if not item["available"]:
            prepared[key] = {"status": "unavailable"}
            continue
        arm, fold, seed = key.split("/")
        output = root / "checkpoint_average" / arm / f"{fold}_seed_{seed}"
        path = output / "selected.pt"
        selected = item["selected"]
        if sha256_file(Path(selected["path"])) != selected["sha256"]:
            raise ValueError("source selected state changed")
        source = torch.load(selected["path"], map_location="cpu", weights_only=True)
        if (
            source["stage"] != "F"
            or source["fold"] != fold
            or source["seed"] != int(seed)
            or source["epoch"] != item["selected_epoch"]
        ):
            raise ValueError("averaging source trajectory differs")
        states = []
        for entry in item["trailing_states"]:
            if sha256_file(Path(entry["path"])) != entry["sha256"]:
                raise ValueError("source epoch state changed")
            state = torch.load(entry["path"], map_location="cpu", weights_only=True)
            if (
                state["contract"] != source["contract"]
                or state["epoch"] != entry["epoch"]
            ):
                raise ValueError(
                    "averaging crosses parameter or preprocessing coordinates"
                )
            states.append(state["model_state_dict"])
        binding = {
            "selected_source": selected,
            "epochs": item["trailing_states"],
            "rule": "trailing up to three epochs ending at raw selected epoch",
            "selection_ic_semantics": "raw selector value determining the cutoff; not averaged-model IC",
        }
        if path.exists():
            receipt = read(output / "average.json")
            if receipt["binding"] != binding or sha256_file(path) != receipt["sha256"]:
                raise ValueError("saved checkpoint average changed")
        else:
            payload = {
                key: value
                for key, value in source.items()
                if key != "optimizer_state_dict"
            }
            payload.update(
                model_state_dict=average_state_dicts(states), weight_rule=binding
            )
            _atomic_torch_save(path, payload)
            write_json_atomic(
                output / "average.json",
                {"binding": binding, "sha256": sha256_file(path)},
            )
        prepared[key] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "epochs": [v["epoch"] for v in item["trailing_states"]],
            "training_commit": source["contract"]["code"]["commit"],
        }
    write_json_atomic(
        root / "averaging_plan.json",
        {
            "checkpoints": prepared,
            "store": read(root / "frozen_design.json")["store"],
            "inventory_sha256": sha256_file(root / "inventory.json"),
            "next": "Score with original compatible inference source; compare original neutral raw checkpoints. No retraining.",
        },
    )
    print(
        {"averages_prepared": sum("path" in v for v in prepared.values())}, flush=True
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    prepare(args.root)


if __name__ == "__main__":
    main()
