"""Capture the first unsaved guarded fraction-conversion state, then stop."""

from dataclasses import asdict
import inspect
import json
from pathlib import Path

import numpy as np
import torch

from brazil_rv.execution.loan_contracts import LoanContracts
from brazil_rv.v2.artifacts import write_json_atomic
import run_refit_sensitivities as producer

PROJECT = Path(__file__).resolve().parents[1]


def main():
    run = json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())
    root = (
        Path(run["stage_c_refit_sensitivity_plan"]["path"]).parent / "pending_fraction"
    )
    original_phase = producer.phase_data
    original = LoanContracts.provision_fractions
    current = {}

    def phase_data(data, phase):
        current.update(phase=phase, data=data)
        return original_phase(data, phase)

    def probe(self, name, ratio, day, auction):
        ids = np.flatnonzero(self.name == name)
        if np.any(self.return_day[ids] >= 0) or len(np.unique(self.root[ids])) != len(
            ids
        ):
            caller = inspect.currentframe().f_back
            ledger = caller.f_back.f_locals
            fields = {}
            for key, value in vars(self).items():
                if (
                    isinstance(value, (np.ndarray, torch.Tensor))
                    and value.shape[:1] == self.name.shape
                ):
                    fields[key] = value[ids].tolist()
            receipts = [
                dict(day=int(d), quantity=float(q[name]))
                for d, q in caller.f_locals["custody"].receipts
                if float(q[name]) != 0
            ]
            inputs = ledger["inputs"]
            write_json_atomic(
                root / "guard_state.json",
                dict(
                    phase=current["phase"],
                    name=name,
                    isin=current["data"].inputs.security_ids[name],
                    day=day,
                    date=str(inputs.dates[day]),
                    ratio=ratio,
                    auction=asdict(auction),
                    economic_shares=float(caller.f_locals["shares"][name]),
                    cohorts=fields,
                    custody_receipts=receipts,
                ),
            )
        return original(self, name, ratio, day, auction)

    producer.phase_data = phase_data
    LoanContracts.provision_fractions = probe
    producer.execute(run)


if __name__ == "__main__":
    main()
