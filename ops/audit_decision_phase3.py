"""Verify the completed registered Phase 3 artifacts without refitting models."""

import argparse
import json
from pathlib import Path

import numpy as np

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def audit(root, store, output):
    design = read(root / "phase3/frozen_design.json")
    dates = np.load(store / "date_index.npy")
    isins = np.load(store / "isin_index.npy")
    active = np.load(store / "active.npy", mmap_mode="r")
    expected_store = read(Path("docs/v2_data_inputs.json"))["store"]["manifest_sha256"]
    assert sha256_file(store / "manifest.json") == expected_store
    records, verified = [], 0
    for kind, count in (("screen", 48), ("confirmation", 120)):
        plan = read(root / f"phase3/{kind}_plan.json")
        assert len(plan["jobs"]) == count
        for job in plan["jobs"]:
            folder = Path(job["run_dir"])
            manifest = read(folder / "run_manifest.json")
            contract = manifest["contract"]
            assert contract["code"] == design["implementation"]
            assert contract["store_manifest_sha256"] == expected_store
            config = contract["config"]
            assert config.get("lookback", config.get("slow_lookback")) == 60
            assert contract["epochs"] == 60
            assert contract["compile"] and contract["date_tensor_cache"]
            for name, digest in manifest["artifacts"].items():
                assert sha256_file(folder / name) == digest, folder / name
                verified += 1
            scores = folder / "scores"
            assert (
                sha256_file(scores / "score_manifest.json")
                == manifest["score_manifest_sha256"]
            )
            sm = read(scores / "score_manifest.json")
            assert not sm["official_validation_accessed"] and not sm["test_accessed"]
            assert sm["transfer_chronology_clean"]
            assert sm["checkpoint"]["sha256"] == manifest["artifacts"]["selected.pt"]
            for name, record in sm["artifacts"].items():
                assert sha256_file(scores / name) == record["sha256"], scores / name
                verified += 1
            rows = np.asarray(sm["dataset"]["date_indices"])
            assert np.array_equal(np.load(scores / "date_index.npy"), dates[rows])
            assert np.array_equal(np.load(scores / "isin_index.npy"), isins)
            assert dates[rows].max() <= np.datetime64("2024-12-30")
            mask = np.load(scores / "score_mask.npy")
            values = np.load(scores / "scores.npy")
            for index, horizon in enumerate(sm["axes"]["horizons"]):
                expected = (
                    active[rows]
                    if horizon in sm["trained_horizons"]
                    else np.zeros_like(active[rows])
                )
                assert np.array_equal(mask[:, :, index], expected)
            assert np.isfinite(values[mask]).all()
            aux = contract["economic_auxiliary"]
            if aux:
                assert aux["weight"] == 0.25 and aux["scale"] > 0
                assert max(aux["fit_target_window"]) < min(rows)
                assert set(aux["fit_date_indices"]).issubset(aux["fit_target_window"])
                cardinal = np.load(scores / "economic_daily_residual.npy")
                assert np.isfinite(cardinal[active[rows]]).all()
            history = read(folder / "history.json")
            diag = read(folder / "diagnostics.json")
            records.append(
                {
                    "name": job["name"],
                    "panel": kind,
                    "manifest_sha256": sha256_file(folder / "run_manifest.json"),
                    "selected_epoch": manifest["selected_epoch"],
                    "epochs": manifest["epochs_completed"],
                    "stop_reason": manifest["stop_reason"],
                    "seconds": manifest["seconds_this_process"],
                    "peak_cuda_bytes": manifest["peak_cuda_bytes"],
                    "active_stock_days": int(active[rows].sum()),
                    "dates": [str(dates[rows[0]]), str(dates[rows[-1]])],
                    "selection_ic": manifest["selection_ic"],
                    "selected_fit_ic": diag["selected_clean_fit"]["mean_ic"],
                    "loss_scale_retries": sum(h["loss_scale_retries"] for h in history),
                }
            )
    books = []
    for area, expected in (("books", 336), ("continuous", 16)):
        paths = sorted((root / "phase3" / area).rglob("book.json"))
        assert len(paths) == expected, (area, len(paths))
        for path in paths:
            book = read(path)
            for name, digest in book["files"].items():
                assert sha256_file(path.parent / name) == digest, path.parent / name
                verified += 1
            assert max(book["dates"]) <= "2024-12-30"
            assert book["summary"]["max_absolute_reconciliation_error"] < 1e-10
            books.append(
                {
                    "path": str(path.relative_to(root)),
                    "sha256": sha256_file(path),
                    "summary": book["summary"],
                }
            )
    panels = {}
    for kind in ("screen", "confirmation"):
        summary = read(root / f"phase3/{kind}_summary.json")
        panels[kind] = {"survivors": summary["survivors"], "cells": {}}
        assert not summary["heldout_accessed"]
        for arm, cell in summary["cells"].items():
            panels[kind]["cells"][arm] = {k: cell[k] for k in ("gate", "contrasts")}
            panels[kind]["cells"][arm]["ensemble_forecasts"] = {
                k: v for k, v in cell["forecasts"].items() if k.endswith("/ensemble")
            }
    continuous = {
        arm: read(root / f"phase3/continuous/{arm}/comparison.json")
        for arm in ("TE_all", "C6")
    }
    offset = int(np.flatnonzero(dates == np.datetime64("2016-07-18"))[0])
    blend_weights = {}
    for index in range(1, 15):
        fold = f"F{index}"
        mapping = read(root / f"phase3/mappings/{fold}.json")
        panel = read(
            root
            / f"phase3/fits/TE_all/neutral/{fold}_seed_11/scores/score_manifest.json"
        )
        assert max(mapping["fit"]) < min(mapping["selection"])
        assert max(mapping["selection"]) + offset < min(
            panel["dataset"]["date_indices"]
        )
        blend_weights[fold] = mapping["blend"]["selected"]["te_weight"]
        for arm in ("TE_all", "C6"):
            for seed in (11, 29, 47):
                contracts = [
                    read(
                        root
                        / f"phase3/fits/{arm}/{variant}/{fold}_seed_{seed}/run_manifest.json"
                    )["contract"]
                    for variant in ("neutral", "economic")
                ]
                for contract in contracts:
                    contract.pop("economic_auxiliary")
                assert contracts[0] == contracts[1]
    result = {
        "passed": True,
        "root": str(root),
        "source": design["implementation"],
        "store_manifest_sha256": expected_store,
        "verified_artifact_hashes": verified,
        "fits": records,
        "books": books,
        "panels": panels,
        "continuous": continuous,
        "heldout_accessed": False,
        "additional_completion_checks": {
            "paired_contracts_identical_except_auxiliary": 84,
            "paired_date_and_population_equality": 84,
            "prior_only_mapping_boundaries_verified": 14,
            "te_blend_weights": blend_weights,
        },
    }
    write_json_atomic(output, result)
    print(
        json.dumps(
            {
                "passed": True,
                "fits": len(records),
                "books": len(books),
                "verified_hashes": verified,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(
        Path(read(Path("docs/v2_decision_run.json"))["root"]),
        Path(read(Path("docs/v2_data_inputs.json"))["store"]["root"]),
        args.output,
    )
