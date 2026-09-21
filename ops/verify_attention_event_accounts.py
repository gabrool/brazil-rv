"""Saved-intention differentiable-account check for the new cash revisions."""

import argparse
import json
from pathlib import Path
import pickle
from time import perf_counter

import numpy as np
import torch

from brazil_rv.execution.custody_fees import CustodyAssessment
from brazil_rv.execution.spot_costs import MonthlySpotTariff
from brazil_rv.execution.stateful_ledger import LedgerConfig
from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corrected-hedge", action="store_true")
    args = parser.parse_args()
    started = perf_counter()
    torch.set_num_threads(1)
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    plan = bound_json(run["scaling_expanded_source_plan"])
    root = Path(run["scaling_expanded_source_plan"]["path"]).parent
    books_root = (
        Path(run["scaling_hedge_roundoff_plan"]["path"]).parent
        if args.corrected_hedge
        else root
    )
    parent = books_root / "account_parity"
    out = parent if not parent.exists() else parent / "diagnostic"
    out.mkdir(exist_ok=False)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    source = bound_json(plan["inputs"])["cache"]
    assert sha256_file(Path(source["path"])) == source["sha256"]
    with Path(source["path"]).open("rb") as f:
        data = pickle.load(f)
    dates = np.asarray(data.inputs.dates).astype(str)
    reports = []
    progress = bound_json(binding(books_root / "replays.json"))
    assert progress["status"] == "complete" and len(progress["completed"]) == 12
    for record in progress["completed"]:
        book = bound_json(record["book"])
        indices = np.searchsorted(dates, book["state_dates"])
        np.testing.assert_array_equal(dates[indices], book["state_dates"])
        cfg = book["provenance"]["config"].copy()
        cfg["monthly_spot_tariffs"] = tuple(
            MonthlySpotTariff(**x) for x in cfg["monthly_spot_tariffs"]
        )
        cfg["custody_assessments"] = tuple(
            CustodyAssessment(**x) for x in cfg["custody_assessments"]
        )
        account = data.initial_account(int(indices[0]), LedgerConfig(**cfg))
        with np.load(Path(record["book"]["path"]).parent / "account.npz") as z:
            targets, nav = z["targets"], z["nav"]
        observed = []
        with torch.no_grad():
            for local, day in enumerate(indices):
                try:
                    row = data.step(
                        account,
                        torch.from_numpy(targets[local]),
                        int(day),
                        terminal=local == len(indices) - 1,
                    )
                except Exception as error:
                    trace = error.__traceback__
                    while trace.tb_next:
                        trace = trace.tb_next
                    state = trace.tb_frame.f_locals
                    detail = dict(
                        key=record["key"],
                        date=dates[day],
                        local=local,
                        error=str(error),
                        prefix_nav_error=float(
                            np.max(
                                np.abs(np.asarray(observed) - nav[:local]), initial=0
                            )
                        ),
                    )
                    if "exit_quantity" in state and "quantity" in state:
                        exits, entries = (
                            state[k].detach().numpy()
                            for k in ("exit_quantity", "quantity")
                        )
                        bad = np.flatnonzero(exits * entries < 0)
                        detail["opposite"] = [
                            dict(
                                axis=int(i),
                                exit=float(exits[i]),
                                entry=float(entries[i]),
                            )
                            for i in bad
                        ]
                    write_json_atomic(out / "failure.json", detail)
                    raise
                observed.append(float(row["nav"]))
                if abs(observed[-1] - nav[local]) >= 1e-7:
                    write_json_atomic(
                        out / "first_nav_difference.json",
                        dict(
                            key=record["key"],
                            date=dates[day],
                            local=local,
                            expected=float(nav[local]),
                            actual=observed[-1],
                        ),
                    )
                    raise AssertionError("first saved-intention NAV divergence")
        difference = np.asarray(observed) - nav
        report = dict(
            key=record["key"],
            book=record["book"],
            nav_count=len(nav),
            max_nav_difference_brl=float(np.max(np.abs(difference))),
        )
        write_json_atomic(out / (record["key"].replace("/", "_") + ".json"), report)
        assert report["max_nav_difference_brl"] < 1e-7, report
        reports.append(report)
    report = dict(
        passed=True,
        source=run["scaling_expanded_source_plan"],
        books=reports,
        seconds=perf_counter() - started,
        limits="Original independent books reused; identical saved intentions replayed only through the differentiable account to test the newly changed pending-cash revision interaction. No allocator/model/independent-account replay, no new numerical contrast. Existing independent Decimal and saved NAV proofs remain separate.",
    )
    write_json_atomic(out / "report.json", report)
    run = json.loads(pointer.read_text())
    key = (
        "scaling_hedge_roundoff_account_parity"
        if args.corrected_hedge
        else "scaling_expanded_event_account_parity"
    )
    run[key] = binding(out / "report.json")
    write_json_atomic(pointer, run)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
