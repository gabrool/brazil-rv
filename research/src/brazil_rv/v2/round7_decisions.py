"""Registered screen advancement and IC-first confirmation selection."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import research_rounds as rr
from .artifacts import sha256_file, write_json_atomic
from .contract import DEVELOPMENT_FOLDS
from .round7 import CELLS, SCREEN_FOLDS, SEEDS
from .round7_program import read

IC = "primary_neutral_target_ic"
NET = "headline_net_excess_bps"
ORDER = tuple(c["cell"] for c in CELLS)


def advance(result, maximum_candidates=5):
    if set(result["folds"]) != set(SCREEN_FOLDS) or tuple(result["seeds"]) != SEEDS:
        raise ValueError(
            "advancement requires the registered four-fold, three-seed screen"
        )
    points = {
        c: result["paired"][f"{c}_minus_A0"]["pooled"][IC]["estimate"]
        for c in result["cells"]
        if c != "A0"
    }
    ordered = sorted(points, key=lambda c: (-points[c], ORDER.index(c)))
    leader = ordered[0]
    selected = ["A1"]
    selected += [
        c for c in ordered if c != "A1" and points[c] >= points[leader] - 0.002
    ][: maximum_candidates - 1]
    return {
        "cells": selected,
        "screen_leader": leader,
        "paired_ic_points": points,
        "label": "screened_on_F2_F6_F10_F14",
        "maximum_candidates": maximum_candidates,
        "rule": "A1 mandatory; other candidates within .002 of the best paired IC, in descending order; registration order breaks ties",
    }


def select(result):
    if (
        set(result["folds"]) != set(DEVELOPMENT_FOLDS)
        or tuple(result["seeds"]) != SEEDS
    ):
        raise ValueError(
            "seed extension requires the full matched fourteen-fold confirmation"
        )
    rows = {c: r["pooled"] for c, r in result["readouts"].items()}
    eligible = [
        c
        for c in ORDER
        if c in rows
        and rows[c][IC]["estimate"] is not None
        and rows[c][NET]["estimate"] is not None
        and rows[c][NET]["estimate"] >= 0
    ]
    # AdamW is diagnostic unless it leads outright on the matched full panel.
    if "B11" in eligible and any(
        rows[c][IC]["estimate"] >= rows["B11"][IC]["estimate"]
        for c in eligible
        if c != "B11"
    ):
        eligible.remove("B11")
    leader = max(eligible, key=lambda c: rows[c][IC]["estimate"], default=None)
    overrides = []
    for cell in eligible:
        if cell == leader:
            continue
        pair = result["paired"][f"{cell}_minus_{leader}"]["pooled"]
        if rr._interval_includes_zero(pair[IC]) and rr._interval_is_positive(pair[NET]):
            overrides.append(cell)
    override = max(overrides, key=lambda c: rows[c][NET]["estimate"], default=None)
    return {
        "cell": override or leader,
        "ic_leader": leader,
        "economics_override": override,
        "eligible": eligible,
        "ineligible": [c for c in rows if c not in eligible],
        "seed_extension": "61/79/97 for this frozen candidate and A0; do not rerank it against unextended three-seed candidates",
        "source_panel": "matched fourteen-fold three-seed confirmation",
        "read_2025_authorized": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("advance", "select"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--maximum-candidates", type=int, choices=(3, 5), default=5)
    args = parser.parse_args()
    source = args.root / (
        "screen_result.json" if args.action == "advance" else "confirmation_result.json"
    )
    output = args.root / (
        "advancement.json" if args.action == "advance" else "confirmation_leader.json"
    )
    if output.exists():
        raise FileExistsError(output)
    result = (
        advance(read(source), args.maximum_candidates)
        if args.action == "advance"
        else select(read(source))
    )
    result["source_result_sha256"] = sha256_file(source)
    write_json_atomic(output, result)
    print(result)


if __name__ == "__main__":
    main()
