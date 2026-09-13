"""Separately frozen pathway factorial using the existing Round-7 fit engine."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .artifacts import sha256_file, write_json_atomic
from .characteristic_model import CharacteristicModel
from .contract import DEVELOPMENT_FOLDS
from .data_roots import resolve_external_root
from .research_rounds import _git_identity
from .round7 import (
    CELLS,
    PATHWAY_CELLS,
    SCREEN_FOLDS,
    SEEDS,
    configuration,
)
from .round7_data import PROJECT
from .round7_program import read
from .round7_seed_audit import run as omission_audit

ORDER = tuple(c["cell"] for c in PATHWAY_CELLS)
PAIRS = (("GL", "GE"), ("TL", "TE"))
IC = "primary_neutral_target_ic"
NET = "headline_net_excess_bps"


def register():
    inputs = read(PROJECT / "docs/v2_round7_inputs.json")
    store = resolve_external_root(inputs["store"]["root"])[0]
    names = read(store / "manifest.json")["feature_names"]
    cells = []
    for cell in PATHWAY_CELLS:
        config = configuration(cell, names)
        model = CharacteristicModel(config)
        cells.append(
            {
                **cell,
                "config": asdict(config),
                "parameters": sum(p.numel() for p in model.parameters()),
            }
        )
    result = {
        "schema": "BRAZIL_RV_PATHWAY_PROTOCOL_V1",
        "cells": cells,
        "store": inputs["store"],
        "screen_folds": SCREEN_FOLDS,
        "seeds": SEEDS,
        "registration_sha256": sha256_file(
            PROJECT / "research/preregistrations/v2_round7_pathway.md"
        ),
        "original_protocol_sha256": sha256_file(
            PROJECT / "research/preregistrations/v2_round7.json"
        ),
        "primary_contrasts": ["GE_minus_GL", "TE_minus_TL"],
        "secondary_contrasts": ["TL_minus_GL", "TE_minus_GE", "interaction"],
        "stage_p_epochs": 60,
        "fine_tune_budget": "inherit_original_B4_calibration",
        "rho": 0.05,
        "forward_capture": False,
        "heldout_access": False,
    }
    write_json_atomic(
        PROJECT / "research/preregistrations/v2_round7_pathway.json", result
    )
    return {c["cell"]: c["parameters"] for c in cells}


def freeze(root, original):
    code = _git_identity()
    original = original.resolve()
    base = read(original / "frozen_design.json")
    protocol = read(PROJECT / "research/preregistrations/v2_round7_pathway.json")
    if protocol["registration_sha256"] != sha256_file(
        PROJECT / "research/preregistrations/v2_round7_pathway.md"
    ):
        raise ValueError("machine registration must match the current written protocol")
    if base["store"]["manifest_sha256"] != protocol["store"]["manifest_sha256"]:
        raise ValueError("original and extension stores must match")
    root.mkdir(parents=True, exist_ok=False)
    result = {
        **base,
        "implementation": code,
        "pathway_extension": True,
        "pathway_protocol": protocol,
        "original_run": {
            "root": str(original),
            "frozen_design_sha256": sha256_file(original / "frozen_design.json"),
        },
        "comparison_roots": {c["cell"]: str(original) for c in CELLS},
        "pathway_registration_sha256": sha256_file(
            PROJECT / "research/preregistrations/v2_round7_pathway.md"
        ),
    }
    write_json_atomic(root / "frozen_design.json", result)
    return result["implementation"]


def original_at(design):
    original = resolve_external_root(design["original_run"]["root"])[0]
    if (
        sha256_file(original / "frozen_design.json")
        != design["original_run"]["frozen_design_sha256"]
    ):
        raise ValueError("bound original experiment changed")
    return original


def advance(result):
    if set(result["folds"]) != set(SCREEN_FOLDS) or tuple(result["seeds"]) != SEEDS:
        raise ValueError(
            "pathway screen requires four registered folds and three seeds"
        )

    def delta(cell):
        return result["paired"][f"{cell}_minus_B4"]

    def fold_delta(cell, fold):
        row = delta(cell)
        if "opposite_of" in row:
            return -result["paired"][row["opposite_of"]]["folds"][fold][IC]["estimate"]
        return row["folds"][fold][IC]["estimate"]

    points = {c: delta(c)["pooled"][IC]["estimate"] for c in ORDER}
    qualified = [
        pair
        for pair in PAIRS
        if any(
            points[c] >= 0.002 and sum(fold_delta(c, f) > 0 for f in SCREEN_FOLDS) >= 3
            for c in pair
        )
    ]
    if qualified:
        leader = max(points.values())
        qualified = [
            pair
            for pair in PAIRS
            if pair in qualified
            or any(points[c] > 0 and points[c] >= leader - 0.002 for c in pair)
        ]
    return {
        "cells": [c for pair in qualified for c in pair],
        "screen_deltas": points,
        "interpretation": "screening decision, not a test of absence or a held-out result",
    }


def confirmation_choice(result):
    if (
        set(result["folds"]) != set(DEVELOPMENT_FOLDS)
        or tuple(result["seeds"]) != SEEDS
    ):
        raise ValueError(
            "confirmation requires all fourteen folds and three matched seeds"
        )
    candidates = [c for c in ORDER if c in result["cells"]]
    return max(
        candidates, key=lambda c: result["readouts"][c]["pooled"][IC]["estimate"]
    )


def select(result, omissions, reference):
    candidate = confirmation_choice(result)
    if (
        omissions["candidate"] != candidate
        or omissions["reference"] != reference
        or tuple(omissions["seeds"]) != SEEDS
    ):
        raise ValueError(
            "omissions must use the highest-IC candidate and original designation"
        )
    pair = result["paired"][f"{candidate}_minus_{reference}"]["pooled"]
    net = result["readouts"][candidate]["pooled"][NET]["estimate"]
    eligible = (
        pair[IC]["estimate"] is not None
        and pair[IC]["estimate"] > 0
        and pair[IC]["lower_95"] is not None
        and pair[IC]["lower_95"] > 0
        and omissions["all_omissions_positive"]
        and net is not None
        and net >= 0
        and pair[NET]["estimate"] is not None
        and pair[NET]["estimate"] >= 0
    )
    matched = {"GE": "GL", "TE": "TL"}.get(candidate)
    return {
        "cell": candidate,
        "reference": reference,
        "eligible": bool(eligible),
        "extension_cells": [candidate] + ([matched] if matched else []),
        "paired": pair,
        "all_omissions_positive": omissions["all_omissions_positive"],
        "interpretation": "candidate frozen before fresh seeds; nominal intervals follow development selection",
    }


def finalize(decision, six=None, omissions=None):
    candidate, reference = decision["cell"], decision["reference"]
    accepted = False
    if decision["eligible"]:
        if (
            tuple(six["seeds"]) != (11, 29, 47, 61, 79, 97)
            or set(six["folds"]) != set(DEVELOPMENT_FOLDS)
            or omissions["candidate"] != candidate
            or omissions["reference"] != reference
            or tuple(omissions["seeds"]) != tuple(six["seeds"])
        ):
            raise ValueError(
                "finalization requires the frozen candidate and six matched seeds"
            )
        pair = six["paired"][f"{candidate}_minus_{reference}"]["pooled"]
        net = six["readouts"][candidate]["pooled"][NET]["estimate"]
        accepted = (
            pair[IC]["estimate"] is not None
            and pair[IC]["estimate"] > 0
            and pair[NET]["estimate"] is not None
            and pair[NET]["estimate"] >= 0
            and net is not None
            and net >= 0
            and omissions["all_omissions_positive"]
        )
    return {
        "status": "complete",
        "original_designation": reference,
        "designation": candidate if accepted else reference,
        "extension_candidate": candidate,
        "extension_accepted": bool(accepted),
        "reason": "registered paired gain and seed agreement"
        if accepted
        else "no qualifying stable extension gain",
        "interpretation": "development research; no untouched holdout or deployment claim",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=("register", "freeze", "advance", "select", "finalize"),
    )
    parser.add_argument("--root", type=Path)
    parser.add_argument("--original", type=Path)
    args = parser.parse_args()
    if args.action == "register":
        result = register()
    elif args.action == "freeze":
        result = freeze(args.root, args.original)
    elif args.action == "advance":
        result = advance(read(args.root / "pathway_screen_result.json"))
        path = args.root / "advancement.json"
        if path.exists():
            raise FileExistsError(path)
        write_json_atomic(path, result)
    else:
        root = args.root
        original = original_at(read(root / "frozen_design.json"))
        reference = read(original / "decision.json")["designation"]
        path = root / (
            "confirmation_leader.json" if args.action == "select" else "decision.json"
        )
        if path.exists():
            raise FileExistsError(path)
        sources = {"original_decision": sha256_file(original / "decision.json")}
        if args.action == "select":
            source = root / "pathway_confirmation_result.json"
            panel = read(source)
            omissions = omission_audit(
                root,
                candidate=confirmation_choice(panel),
                reference=reference,
                seeds=SEEDS,
                group="pathway_confirmation",
            )
            result = select(panel, omissions, reference)
            sources["confirmation"] = sha256_file(source)
            sources["omissions"] = sha256_file(root / "pathway_confirmation_loso.json")
        else:
            if not read(root / "advancement.json")["cells"]:
                result = {
                    "status": "complete",
                    "original_designation": reference,
                    "designation": reference,
                    "extension_candidate": None,
                    "extension_accepted": False,
                    "reason": "no encoder pair met the fixed screen advancement rule",
                    "interpretation": "no demonstrated benefit under this screen, not proof of absence",
                    "source_hashes": {
                        **sources,
                        "advancement": sha256_file(root / "advancement.json"),
                        "screen": sha256_file(root / "pathway_screen_result.json"),
                    },
                }
                write_json_atomic(path, result)
                print(json.dumps(result))
                return
            decision = read(root / "confirmation_leader.json")
            six, omissions = None, None
            if decision["eligible"]:
                six = read(root / "pathway_six_seed_result.json")
                omissions = omission_audit(
                    root,
                    candidate=decision["cell"],
                    reference=reference,
                    group="pathway_six_seed",
                )
                sources["six_seed"] = sha256_file(root / "pathway_six_seed_result.json")
                sources["omissions"] = sha256_file(root / "pathway_six_seed_loso.json")
            result = finalize(decision, six, omissions)
            sources["confirmation_leader"] = sha256_file(
                root / "confirmation_leader.json"
            )
        result["source_hashes"] = sources
        write_json_atomic(path, result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
