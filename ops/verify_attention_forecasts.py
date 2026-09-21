"""Reproduce saved attention forecasts eagerly on each checkpoint's own store."""

from functools import partial
import gc
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.characteristic_model import CharacteristicConfig, CharacteristicModel
from brazil_rv.v2.contract import HORIZONS
from brazil_rv.v2.data import V2DailyDataset, stage_name_count
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.round7_preprocessing import Round7Preprocessing
from brazil_rv.v2.round7_training import (
    autocast_dtype,
    forward,
    model_batch,
    sequential_batches,
)
from brazil_rv.v2.train import _cli_stage_indices, set_deterministic_seed
from scope_compiled_score_failure import contrast

PROJECT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1)
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    investigation = bound_json(run["scaling_investigation"])
    original = bound_json(investigation["old_plan"])
    roots = {
        "original": Path(original["foundation_root"]),
        "corrected": Path(run["stage_c_refit_root"]),
    }
    stores = {
        version: Path(bound_json(binding(root / "frozen_design.json"))["store"]["root"])
        for version, root in roots.items()
    }
    previous_path = (
        Path(run["stage_d_width_original_plan"]["path"]).parent
        / "scoring_failure/architecture_scope/report.json"
    )
    previous = bound_json(binding(previous_path))
    out = Path(run["scaling_investigation"]["path"]).parent / "forecast_verification"
    out.mkdir(exist_ok=True)
    plan_path = out / "plan.json"
    if not plan_path.exists():
        cases = []
        for version, root in roots.items():
            for arm in investigation["arms"]:
                for fold in investigation["screen_folds"]:
                    for seed in investigation["seeds"]:
                        fit = root / "fits" / arm / f"{fold}_seed_{seed}"
                        reused = version == "corrected" and fold == "F10" and seed == 11
                        if reused:
                            assert previous[arm]["checkpoint"] == binding(
                                fit / "selected.pt"
                            )
                        cases.append(
                            dict(
                                version=version,
                                arm=arm,
                                fold=fold,
                                seed=seed,
                                fit=binding(fit / "run_manifest.json"),
                                reused=reused,
                            )
                        )
        write_json_atomic(
            plan_path,
            dict(
                cases=cases,
                stores={
                    k: dict(root=str(v), manifest=binding(v / "manifest.json"))
                    for k, v in stores.items()
                },
                prior=binding(previous_path),
                rtol=0.02,
                atol=0.002,
                driver=binding(Path(__file__)),
                scope="All 48 original/corrected TE_full/TE_wide F checkpoints and original evaluation dates, own original conditioning/store, full population/full60. Reuse two already passed corrected F10/11 complete eager exports; 46 new eager exports. No optimizer, fit, changed score, new selection or held-out data. Preserve original tolerance and every failure; this tests inference reproducibility, not every historical training kernel.",
            ),
        )
    plan = bound_json(binding(plan_path))
    if plan["driver"] != binding(Path(__file__)):
        resume = bound_json(binding(out / "resume.json"))
        assert resume["original_plan"] == binding(plan_path)
        assert resume["driver"]["sha256"] == sha256_file(Path(__file__))
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    results = []
    for case in plan["cases"]:
        name = f"{case['version']}_{case['arm']}_{case['fold']}_{case['seed']}"
        path = out / name
        report_path = path / "report.json"
        if report_path.exists():
            results.append(bound_json(binding(report_path)))
            continue
        manifest = bound_json(case["fit"])
        fit = Path(case["fit"]["path"]).parent
        if case["reused"]:
            result = dict(
                case=case,
                reused=plan["prior"],
                comparison=previous[case["arm"]]["saved_vs_eager"],
                seconds=0,
            )
        else:
            started = perf_counter()
            checkpoint = fit / "selected.pt"
            assert sha256_file(checkpoint) == manifest["artifacts"]["selected.pt"]
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            contract = payload["contract"]
            assert json.loads(json.dumps(contract)) == manifest["contract"]
            store = stores[case["version"]]
            assert (
                contract["store_manifest_sha256"]
                == plan["stores"][case["version"]]["manifest"]["sha256"]
            )
            config = CharacteristicConfig(**contract["config"])
            _, _, rows, _ = _cli_stage_indices(store, "F", case["fold"])
            dataset = V2DailyDataset(
                store,
                rows,
                stage="evaluation",
                lookback=60,
                enabled_sidecars=tuple(k for k, _ in config.family_counts),
                include_fast=False,
                include_intraday=False,
                include_common_state=True,
                compact_names=True,
                purpose="evaluation",
            )
            prep = Round7Preprocessing.from_payload(contract["preprocessing"])
            loader = DataLoader(
                dataset,
                batch_sampler=sequential_batches(len(dataset)),
                collate_fn=partial(
                    prep.collate,
                    fixed_name_count=max(
                        contract["padded_name_count"], stage_name_count(dataset)
                    ),
                ),
            )
            set_deterministic_seed(case["seed"])
            model = CharacteristicModel(config).cuda().eval()
            model.load_state_dict(payload["model_state_dict"])
            values = np.zeros(
                (len(rows), len(dataset.store.isins), len(HORIZONS)), np.float32
            )
            valid = np.zeros_like(values, dtype=bool)
            offset = 0
            for cpu in loader:
                batch = model_batch(cpu, torch.device("cuda"))
                with (
                    torch.no_grad(),
                    torch.autocast("cuda", dtype=autocast_dtype(torch.device("cuda"))),
                ):
                    prediction = (
                        forward(model, batch, characteristic=True)
                        .float()
                        .mean(2)
                        .cpu()
                        .numpy()
                    )
                names, active = cpu["name_index"].numpy(), cpu["active_mask"].numpy()
                for i in range(len(names)):
                    for h, horizon in enumerate(config.horizons):
                        j = HORIZONS.index(horizon)
                        values[offset + i, names[i, active[i]], j] = prediction[
                            i, active[i], h
                        ]
                        valid[offset + i, names[i, active[i]], j] = True
                offset += len(names)
            saved = np.load(fit / "scores/scores.npy")
            np.testing.assert_array_equal(valid, np.load(fit / "scores/score_mask.npy"))
            path.mkdir(exist_ok=False)
            np.savez_compressed(
                path / "eager.npz", scores=values, valid=valid, date_indices=rows
            )
            result = dict(
                case=case,
                checkpoint=binding(checkpoint),
                original_scores=binding(fit / "scores/score_manifest.json"),
                comparison=contrast(saved, values, valid),
                seconds=perf_counter() - started,
            )
            dataset.store.close()
            del model, batch, loader, dataset, payload
            gc.collect()
            torch.cuda.empty_cache()
        path.mkdir(exist_ok=True)
        write_json_atomic(report_path, result)
        results.append(result)
        write_json_atomic(
            out / "progress.json",
            dict(completed=len(results), planned=len(plan["cases"])),
        )
        print(
            json.dumps(
                dict(
                    case=name,
                    comparison=result["comparison"],
                    seconds=result["seconds"],
                )
            ),
            flush=True,
        )
    report = dict(
        plan=binding(plan_path),
        results=results,
        passed=all(
            r["comparison"]["cells"] == r["comparison"]["finite"]
            and r["comparison"]["outside_tolerance"] == 0
            for r in results
        ),
    )
    write_json_atomic(out / "report.json", report)
    run["scaling_forecast_verification"] = binding(out / "report.json")
    write_json_atomic(pointer, run)


if __name__ == "__main__":
    main()
