"""Qualify the exact training prefix and evaluate the frozen parent contrast."""

import argparse
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
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.foundation_readouts import new_panel
from brazil_rv.v2.objective_readouts import calibration, forecast_readout, rank_view
from brazil_rv.v2.portfolio_inputs import HEADS, normalized_ranks
from brazil_rv.v2.portfolio_readouts import save_book
from brazil_rv.v2.portfolio_training import windows
from brazil_rv.v2.research_rounds import _score_artifact
from brazil_rv.v2.train import rank_average_ensemble
from decompose_attention_reversal import nav_check

PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    plan = bound_json(run["scaling_parent_patience_plan"])
    out = Path(run["scaling_parent_patience_plan"]["path"]).parent
    prefix_path = out / "prefix_qualification.json"
    if args.prefix:
        assert not prefix_path.exists()
        started = perf_counter()
        control = Path(plan["controls"]["P_seed_29"]["path"]).parent
        old_history = bound_json(binding(control / "history.json"))
        new_history = bound_json(binding(out / "P_seed_29/history.json"))
        assert len(old_history) == 9 and len(new_history) >= 9
        checks = []
        for i in range(9):
            a = {k: v for k, v in old_history[i].items() if k != "seconds"}
            b = {k: v for k, v in new_history[i].items() if k != "seconds"}
            assert a == b, (i + 1, "history")
            paths = [
                folder / "epochs" / f"epoch_{i + 1:03d}.pt"
                for folder in (control, out / "P_seed_29")
            ]
            old, new = [
                torch.load(p, map_location="cpu", weights_only=True) for p in paths
            ]
            changed = [
                k for k in old["contract"] if old["contract"][k] != new["contract"][k]
            ]
            assert set(changed) == {"code", "recipe"}, changed
            assert (
                old["contract"]["code"]["commit"]
                == "12c659d3e4454f16df4dc96b1a080b68a71e7b7c"
            )
            assert new["contract"]["code"] == plan["runtime"]["code"]
            assert {
                k
                for k in old["contract"]["recipe"]
                if old["contract"]["recipe"][k] != new["contract"]["recipe"][k]
            } == {"patience"}
            old_state, new_state = old["model_state_dict"], new["model_state_dict"]
            assert old_state.keys() == new_state.keys()
            assert all(torch.equal(v, new_state[k]) for k, v in old_state.items()), (
                i + 1
            )
            checks.append(
                dict(
                    epoch=i + 1,
                    cells=sum(v.numel() for v in old_state.values()),
                    exact_weights=True,
                    exact_history_except_seconds=True,
                    control=binding(paths[0]),
                    diagnostic=binding(paths[1]),
                )
            )
        write_json_atomic(
            prefix_path,
            dict(
                plan=run["scaling_parent_patience_plan"],
                checks=checks,
                seconds=perf_counter() - started,
                executed=binding(Path(__file__)),
                passed=True,
            ),
        )
        (out / "executed_prefix_qualifier.py").write_bytes(Path(__file__).read_bytes())
        evaluation = dict(
            plan=run["scaling_parent_patience_plan"],
            prefix=binding(prefix_path),
            members=["29", "ensemble"],
            scope="R10m corrected neutral F10 account, original calibration/new allocation risk. New seed29 child alone and ensemble replacing only seed29, reusing corrected seeds11/47 and both existing corrected controls. Two new books; raw checkpoints only. No new checkpoint or portfolio selection from evaluation outcomes. Preserve original-data seed29 reference separately, since it has different model data.",
            driver=binding(Path(__file__)),
        )
        write_json_atomic(out / "evaluation_plan.json", evaluation)
        run = json.loads(pointer.read_text())
        run["scaling_parent_patience_prefix"] = binding(prefix_path)
        run["scaling_parent_patience_evaluation_plan"] = binding(
            out / "evaluation_plan.json"
        )
        write_json_atomic(pointer, run)
        print(
            json.dumps(
                dict(
                    prefix_passed=True,
                    epochs=9,
                    cells=sum(r["cells"] for r in checks),
                    seconds=perf_counter() - started,
                )
            ),
            flush=True,
        )
        return
    evaluation = bound_json(run["scaling_parent_patience_evaluation_plan"])
    assert evaluation["driver"]["sha256"] == sha256_file(Path(__file__))
    assert bound_json(run["scaling_parent_patience_fits"])["status"] == "complete"
    investigation = bound_json(run["scaling_investigation"])
    original = bound_json(investigation["old_plan"])
    prior = Path(original["prior_root"])
    inputs = bound_json(investigation["new_inputs"])
    cache = Path(inputs["cache"]["path"])
    assert sha256_file(cache) == inputs["cache"]["sha256"]
    with cache.open("rb") as stream:
        data = pickle.load(stream)
    rows = windows(prior, data, "F10")["evaluation"]
    panels, valid, sources = new_panel(
        Path(run["stage_c_refit_root"]), data, "TE_wide", "F10", rows, "raw"
    )
    store = Path(plan["store"]["root"])
    schema = json.loads((store / "manifest.json").read_text())["metadata"][
        "feature_schema"
    ]["sha256"]
    fit = out / "F10_seed_29"
    values, mask = _score_artifact(
        fit / "scores",
        require_clean_transfer=True,
        expected_dates=np.asarray(data.inputs.dates, dtype="datetime64[D]")[rows],
        expected_isins=data.inputs.security_ids,
        expected_feature_schema_sha256=schema,
    )
    mask = mask[..., HEADS]
    np.testing.assert_array_equal(mask.all(-1), valid)
    new_rank = normalized_ranks(rank_average_ensemble([values[..., HEADS]], mask), mask)
    new_panels = {
        "29": new_rank,
        "ensemble": np.mean([panels["11"], new_rank, panels["47"]], axis=0),
    }
    mapping_path = prior / "phase3/mappings/F10.json"
    mapping = calibration(bound_json(binding(mapping_path))["arms"]["TE_all"])
    config_values = bound_json(investigation["account"])["primary_config"].copy()
    config_values["initial_capital_brl"] = investigation["capital"]
    config_values["monthly_spot_tariffs"] = tuple(
        MonthlySpotTariff(**x) for x in config_values["monthly_spot_tariffs"]
    )
    config_values["custody_assessments"] = tuple(
        CustodyAssessment(**x) for x in config_values["custody_assessments"]
    )
    config = LedgerConfig(**config_values)
    controls = bound_json(bound_json(investigation["new_result"])["books"])
    results = []
    for member in evaluation["members"]:
        started = perf_counter()
        view = rank_view(data, rows, new_panels[member], valid)
        start, stop = int(rows[0]), int(rows[-1]) + 1
        result, targets, previous = exact_replay(
            view, CalibratedPolicy(mapping), start, stop, config=config
        )
        target = out / "books" / member
        assert not target.exists()
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
                diagnostic="parent_patience20",
                plan=run["scaling_parent_patience_evaluation_plan"],
                config=asdict(config),
                policy="equal_rank",
                mapping=binding(mapping_path),
                forecast_sources=dict(
                    reused=sources, new=binding(fit / "scores/score_manifest.json")
                ),
                heldout_accessed=False,
            ),
        )
        control = next(
            r
            for r in controls
            if r["arm"] == "TE_wide"
            and r["fold"] == "F10"
            and str(r["member"]) == member
            and r["capital"] == investigation["capital"]
        )
        results.append(
            dict(
                member=member,
                book=binding(target / "book.json"),
                control=control,
                mean=book["summary"]["mean"],
                net_change_bps=book["summary"]["mean"]["net_excess_bps"]
                - control["mean"]["net_excess_bps"],
                quality=nav_check(target),
                forecast=forecast_readout(
                    data, rows, new_panels[member], valid, mapping
                ),
                seconds=perf_counter() - started,
            )
        )
        print(json.dumps(results[-1]), flush=True)
    fit_results = {
        name: {
            k: bound_json(binding(out / name / "run_manifest.json"))[k]
            for k in (
                "epochs_completed",
                "selected_epoch",
                "selection_ic",
                "stop_reason",
            )
        }
        for name in ("P_seed_29", "F10_seed_29")
    }
    write_json_atomic(
        out / "report.json",
        dict(
            plan=run["scaling_parent_patience_evaluation_plan"],
            prefix=binding(prefix_path),
            fits=fit_results,
            results=results,
            outcome_informed_diagnostic=True,
            adopted=False,
        ),
    )
    run = json.loads(pointer.read_text())
    run["scaling_parent_patience_results"] = binding(out / "report.json")
    write_json_atomic(pointer, run)


if __name__ == "__main__":
    main()
