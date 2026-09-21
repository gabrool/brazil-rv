"""Bound the two original attention exports with four FP16 tolerance crossings."""

from dataclasses import asdict
import json
from pathlib import Path
import pickle
from time import perf_counter

import numpy as np
import torch

from brazil_rv.execution.custody_fees import CustodyAssessment
from brazil_rv.execution.portfolio_policy import CalibratedPolicy, exact_replay
from brazil_rv.execution.spot_costs import MonthlySpotTariff
from brazil_rv.execution.stateful_ledger import LedgerConfig
from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.contract import HORIZONS
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.objective_readouts import calibration, rank_view
from brazil_rv.v2.portfolio_inputs import HEADS, normalized_ranks
from brazil_rv.v2.portfolio_readouts import save_book
from brazil_rv.v2.portfolio_training import windows
from brazil_rv.v2.train import rank_average_ensemble
from decompose_attention_reversal import nav_check

PROJECT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1)
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    investigation = bound_json(run["scaling_investigation"])
    root = Path(run["scaling_investigation"]["path"]).parent
    out = root / "forecast_precision"
    out.mkdir(exist_ok=False)
    original = bound_json(investigation["old_plan"])
    prior = Path(original["prior_root"])
    verification = bound_json(run["scaling_forecast_verification"])
    cases = {
        fold: [
            r
            for r in verification["results"]
            if r["case"]["version"] == "original"
            and r["case"]["arm"] == "TE_wide"
            and r["case"]["fold"] == fold
        ]
        for fold in ("F2", "F6")
    }
    plan = dict(
        driver=binding(Path(__file__)),
        investigation=run["scaling_investigation"],
        verification=run["scaling_forecast_verification"],
        cases=cases,
        scope="Two additional R10m neutral ensemble books using saved eager scores for all three ORIGINAL TE_wide seeds in F2/F6. Compare saved compiled-forecast/new-input/new-risk books from decomposition, same own original score support and new eligibility. Enumerate the four frozen-tolerance crossings; do not relax tolerance or replace scores. Economic contrast includes all eager/compiled rounding differences, not only four crossings. No source/model forward/fit or other book repeated.",
    )
    write_json_atomic(out / "plan.json", plan)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    inputs = bound_json(investigation["new_inputs"])
    cache = Path(inputs["cache"]["path"])
    assert sha256_file(cache) == inputs["cache"]["sha256"]
    with cache.open("rb") as stream:
        data = pickle.load(stream)
    values = bound_json(investigation["account"])["primary_config"].copy()
    values["initial_capital_brl"] = investigation["capital"]
    values["monthly_spot_tariffs"] = tuple(
        MonthlySpotTariff(**x) for x in values["monthly_spot_tariffs"]
    )
    values["custody_assessments"] = tuple(
        CustodyAssessment(**x) for x in values["custody_assessments"]
    )
    config = LedgerConfig(**values)
    results = []
    for fold, records in cases.items():
        started = perf_counter()
        rows = windows(prior, data, fold)["evaluation"]
        mapping_path = prior / "phase3/mappings" / f"{fold}.json"
        mapping = calibration(bound_json(binding(mapping_path))["arms"]["TE_all"])
        panels, common, crossings, sources = [], None, [], []
        for record in records:
            case = record["case"]
            eager_path = (
                root
                / "forecast_verification"
                / f"original_TE_wide_{fold}_{case['seed']}"
                / "eager.npz"
            )
            saved_path = Path(case["fit"]["path"]).parent / "scores"
            with np.load(eager_path) as z:
                eager, valid = z["scores"], z["valid"]
            saved = np.load(saved_path / "scores.npy")
            np.testing.assert_array_equal(valid, np.load(saved_path / "score_mask.npy"))
            for t, n, h in np.argwhere(
                valid & (np.abs(saved - eager) > 0.002 + 0.02 * np.abs(eager))
            ):
                crossings.append(
                    dict(
                        seed=case["seed"],
                        date=str(data.inputs.dates[rows[t]]),
                        isin=str(data.inputs.security_ids[n]),
                        horizon=HORIZONS[h],
                        saved=float(saved[t, n, h]),
                        eager=float(eager[t, n, h]),
                        tolerance=float(0.002 + 0.02 * abs(eager[t, n, h])),
                    )
                )
            head_mask = valid[..., HEADS]
            active = head_mask.all(-1) & data.inputs.active[rows]
            if common is not None:
                np.testing.assert_array_equal(common, active)
            common = active
            panels.append(
                normalized_ranks(
                    rank_average_ensemble([eager[..., HEADS]], head_mask), head_mask
                )
            )
            sources.append(binding(eager_path))
        assert len(crossings) == 2
        ranks = np.mean(panels, axis=0)
        view = rank_view(data, rows, ranks, common)
        start, stop = int(rows[0]), int(rows[-1]) + 1
        result, targets, previous = exact_replay(
            view, CalibratedPolicy(mapping), start, stop, config=config
        )
        target = out / fold
        book = save_book(
            target,
            view,
            result,
            targets,
            previous,
            start,
            start,
            dict(
                scenario="base",
                diagnostic="original_eager_forecast",
                plan=binding(out / "plan.json"),
                config=asdict(config),
                policy="equal_rank",
                mapping=binding(mapping_path),
                forecast_sources=sources,
                heldout_accessed=False,
            ),
        )
        control_path = (
            root
            / "decomposition/books/TE_wide"
            / fold
            / "old_forecast_new_risk/book.json"
        )
        control = bound_json(binding(control_path))
        with np.load(control_path.parent / "account.npz") as old:
            with np.load(target / "account.npz") as new:
                path_delta = (
                    (new["nav"] - old["nav"]) / investigation["capital"] * 10000
                )
        results.append(
            dict(
                fold=fold,
                crossings=crossings,
                control=binding(control_path),
                book=binding(target / "book.json"),
                quality=nav_check(target),
                net_bps_change=book["summary"]["mean"]["net_excess_bps"]
                - control["summary"]["mean"]["net_excess_bps"],
                final_nav_change_bps=float(path_delta[-1]),
                maximum_nav_change_bps=float(np.abs(path_delta).max()),
                seconds=perf_counter() - started,
            )
        )
        print(json.dumps(results[-1]), flush=True)
    write_json_atomic(
        out / "report.json",
        dict(
            plan=binding(out / "plan.json"),
            results=results,
            original_tolerance_passed=False,
        ),
    )
    run = json.loads(pointer.read_text())
    run["scaling_forecast_precision"] = binding(out / "report.json")
    write_json_atomic(pointer, run)


if __name__ == "__main__":
    main()
