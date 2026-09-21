"""Attribute the hedge arithmetic correction on the twelve frozen F3 source books."""

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
from brazil_rv.v2.objective_readouts import calibration, rank_view
from brazil_rv.v2.opportunity_research import benchmark_for
from brazil_rv.v2.performance import performance
from brazil_rv.v2.portfolio_readouts import save_book
from brazil_rv.v2.portfolio_training import windows
from brazil_rv.v2.research_rounds import _git_identity
import qualify_refit_books

PROJECT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1)
    code = _git_identity()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    ref = run["scaling_hedge_roundoff_plan"]
    plan = bound_json(ref)
    root = Path(ref["path"]).parent
    (root / "replay_executed.py").write_bytes(Path(__file__).read_bytes())
    primary = bound_json(plan["source_plan"])
    source = bound_json(primary["inputs"])["cache"]
    assert sha256_file(Path(source["path"])) == source["sha256"]
    with Path(source["path"]).open("rb") as f:
        data = pickle.load(f)
    progress = bound_json(plan["source_books"])
    assert len(progress["completed"]) == 12 and progress["status"] == "complete"
    completed = []
    if (root / "replays.json").exists():
        completed = bound_json(binding(root / "replays.json"))["completed"]
    done = {r["key"] for r in completed}
    panels = {}
    benchmark_root = Path(
        json.loads((PROJECT / "docs/v2_opportunity_run.json").read_text())["root"]
    )

    def save(status):
        write_json_atomic(
            root / "replays.json",
            dict(status=status, plan=ref, completed=completed, planned=12),
        )

    for base in progress["completed"]:
        key = base["key"]
        if key in done:
            continue
        tick = perf_counter()
        book = bound_json(base["book"])
        _, capital, arm, fold, member = key.split("/")
        rows = windows(Path(primary["prior_root"]), data, fold)["evaluation"]
        start, stop = int(rows[0]), int(rows[-1]) + 1
        if arm not in panels:
            panels[arm] = new_panel(
                Path(primary["fit_root"]), data, arm, fold, rows, "raw"
            )
        forecasts, valid, sources = panels[arm]
        assert sources == book["provenance"]["forecast_sources"]
        mapping = calibration(
            bound_json(book["provenance"]["mapping"])["arms"]["TE_all"]
        )
        view = rank_view(data, rows, forecasts[member], valid)
        cfg = book["provenance"]["config"].copy()
        cfg["monthly_spot_tariffs"] = tuple(
            MonthlySpotTariff(**x) for x in cfg["monthly_spot_tariffs"]
        )
        cfg["custody_assessments"] = tuple(
            CustodyAssessment(**x) for x in cfg["custody_assessments"]
        )
        config = LedgerConfig(**cfg)
        path = root / "books" / key
        assert not path.exists(), (
            "Preserve partial output and resume only missing readouts"
        )
        provenance = dict(
            book["provenance"],
            implementation=code,
            phase="hedge_roundoff",
            config=asdict(config),
            sensitivity_plan=ref,
            baseline=base["book"],
        )
        result, targets, previous = exact_replay(
            view, CalibratedPolicy(mapping), start, stop, config=config
        )
        save_book(path, view, result, targets, previous, start, start, provenance)
        bench = benchmark_for(
            benchmark_root, data.inputs.dates[start:stop], data.inputs.dates[start - 1]
        )
        write_json_atomic(
            path / "performance.json",
            performance(
                result.daily_net_return, data.inputs.cdi_returns[start:stop], bench
            ),
        )
        write_json_atomic(
            path / "loan_cash_payments.json",
            [asdict(p) for p in result.loan_cash_payments],
        )
        with np.load(Path(base["book"]["path"]).parent / "account.npz") as z:
            delta = (result.nav - z["nav"]) / int(capital) * 1e4
            target_distance = float(np.max(np.abs(targets - z["targets"])))
        completed.append(
            dict(
                key=key,
                book=binding(path / "book.json"),
                baseline=base,
                performance=binding(path / "performance.json"),
                loan_cash_payments=binding(path / "loan_cash_payments.json"),
                final_path_bps=float(delta[-1]),
                max_abs_path_bps=float(np.max(np.abs(delta))),
                target_distance=target_distance,
                economics_unresolved=bool(result.economics_unresolved),
                seconds=perf_counter() - tick,
            )
        )
        save("running")
        print(
            json.dumps(
                dict(
                    completed=key,
                    final_path_bps=completed[-1]["final_path_bps"],
                    seconds=completed[-1]["seconds"],
                )
            ),
            flush=True,
        )
    save("complete")
    context = root / "qualifier_context"
    (context / "docs").mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        root / "qualification_plan.json",
        dict(primary=plan["source_plan"], correction=ref),
    )
    write_json_atomic(
        context / "docs/v2_economic_data_scaling_run.json",
        dict(
            run,
            stage_c_refit_sensitivity_plan=binding(root / "qualification_plan.json"),
        ),
    )
    qualify_refit_books.PROJECT = context
    qualify_refit_books.main(sensitivities=True)
    run = json.loads(pointer.read_text())
    run["scaling_hedge_roundoff_books"] = binding(root / "replays.json")
    run["scaling_hedge_roundoff_qualification"] = binding(
        root / "qualification/manifest.json"
    )
    write_json_atomic(pointer, run)


if __name__ == "__main__":
    main()
