"""Locate the first observed saved-intention account difference, without refitting."""

from dataclasses import asdict
import json
from pathlib import Path
import pickle

import numpy as np
import polars as pl
import torch

from brazil_rv.execution.custody_fees import CustodyAssessment
from brazil_rv.execution.loan_contracts import LoanContracts
from brazil_rv.execution.spot_costs import MonthlySpotTariff
from brazil_rv.execution.stateful_ledger import LedgerConfig
from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1)
    run = json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())
    plan = bound_json(run["scaling_expanded_source_plan"])
    root = Path(run["scaling_expanded_source_plan"]["path"]).parent
    out = root / "account_parity/first_difference"
    out.mkdir(exist_ok=False)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    with Path(bound_json(plan["inputs"])["cache"]["path"]).open("rb") as f:
        data = pickle.load(f)
    record = bound_json(binding(root / "replays.json"))["completed"][0]
    assert record["key"] == "data_refit/10000000/TE_full/F3/11"
    book = bound_json(record["book"])
    folder = Path(record["book"]["path"]).parent
    cfg = book["provenance"]["config"].copy()
    cfg["monthly_spot_tariffs"] = tuple(
        MonthlySpotTariff(**x) for x in cfg["monthly_spot_tariffs"]
    )
    cfg["custody_assessments"] = tuple(
        CustodyAssessment(**x) for x in cfg["custody_assessments"]
    )
    indices = np.searchsorted(
        np.asarray(data.inputs.dates).astype(str), book["state_dates"]
    )
    start = int(indices[0])
    account = data.initial_account(start, LedgerConfig(**cfg))
    original_accrue = LoanContracts.accrue
    charges = []

    def recorded(self, *args, **kwargs):
        assert "charges" not in kwargs
        return original_accrue(self, *args, **kwargs, charges=charges)

    LoanContracts.accrue = recorded
    old_charges = pl.read_parquet(folder / "loan_charges.parquet")
    with np.load(folder / "account.npz") as z:
        a = {k: z[k] for k in z.files}
    reports, states = [], {}
    try:
        with torch.no_grad():
            for local, day in enumerate(indices[:5]):
                begin = len(charges)
                row = data.step(
                    account, torch.from_numpy(a["targets"][local]), int(day)
                )
                current = [
                    dict(
                        asdict(c),
                        session=c.session - start,
                        opening_session=c.opening_session - start,
                    )
                    for c in charges[begin:]
                ]
                baseline = old_charges.filter(pl.col("session") == local).to_dicts()

                def grouped(values):
                    result = {}
                    for c in values:
                        key = c["security_index"], c["opening_session"]
                        value = result.setdefault(key, np.zeros(2))
                        value += [c["rent"], c["fee"]]
                    return result

                actual, expected = grouped(current), grouped(baseline)
                differences = []
                for key in actual.keys() | expected.keys():
                    delta = actual.get(key, np.zeros(2)) - expected.get(
                        key, np.zeros(2)
                    )
                    if np.max(np.abs(delta)) > 1e-9:
                        differences.append(
                            dict(
                                axis=key[0],
                                opened=key[1],
                                actual=actual.get(key, np.zeros(2)).tolist(),
                                expected=expected.get(key, np.zeros(2)).tolist(),
                            )
                        )
                share_delta = account.shares[:-1].numpy() - a["signed_shares"][local]
                reports.append(
                    dict(
                        date=book["state_dates"][local],
                        nav_error=float(row["nav"]) - float(a["nav"][local]),
                        max_share_error=float(np.max(np.abs(share_delta))),
                        charge_differences=differences,
                    )
                )
                for key in (
                    "name",
                    "opened",
                    "return_day",
                    "root",
                    "quantity",
                    "principal",
                    "minimum",
                    "root_fees",
                    "started",
                ):
                    value = getattr(account.loans, key)
                    states[f"{local}_{key}"] = (
                        value.detach().numpy()
                        if torch.is_tensor(value)
                        else value.copy()
                    )
    finally:
        LoanContracts.accrue = original_accrue
    np.savez_compressed(out / "loan_states.npz", **states)
    write_json_atomic(
        out / "report.json",
        dict(
            book=record["book"],
            days=reports,
            changes="Audit-only charge capture on the already observed five-day prefix; exact production method called once per day. No independent book, allocator, model or fit repeated.",
        ),
    )
    print(json.dumps(reports), flush=True)


if __name__ == "__main__":
    main()
