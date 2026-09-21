"""Verify the three newly exposed SULA books using their saved intentions."""

from decimal import Decimal, ROUND_FLOOR
import json
from pathlib import Path
import pickle
import shutil
from time import perf_counter

import numpy as np
import torch

from brazil_rv.execution.custody_fees import CustodyAssessment
from brazil_rv.execution.loan_contracts import LoanContracts
from brazil_rv.execution.spot_costs import MonthlySpotTariff
from brazil_rv.execution.stateful_ledger import LedgerConfig
from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json
from run_economic_replays import phase_data

PROJECT = Path(__file__).resolve().parents[1]


def main():
    started = perf_counter()
    torch.set_num_threads(1)
    run = json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())
    plan = bound_json(run["stage_c_refit_sensitivity_plan"])
    primary = bound_json(plan["primary"])
    source = bound_json(primary["inputs"])["cache"]
    assert sha256_file(Path(source["path"])) == source["sha256"]
    with Path(source["path"]).open("rb") as handle:
        data = pickle.load(handle)
    root = Path(run["stage_c_refit_sensitivity_plan"]["path"]).parent
    out = root / "pending_fraction"
    assert not (out / "qualification.json").exists()
    shutil.copyfile(__file__, out / "executed_qualification.py")
    progress = bound_json(binding(root / "replays.json"))
    selected = [
        row
        for row in progress["completed"]
        if row["key"].startswith("loan_fraction_BRSULACDAM12/")
        and row["key"].endswith("/C6/F10/ensemble")
    ]
    assert len(selected) == 3
    original = LoanContracts.provision_fractions
    boundary = []

    def qualify(self, name, ratio, day, auction):
        ids = np.flatnonzero(self.name == name)
        due = ids[(self.return_day[ids] >= 0) & (self.return_day[ids] <= day)]
        active = ids[self.return_day[ids] < 0]
        if not len(due):
            return original(self, name, ratio, day, auction)
        assert data.inputs.security_ids[name] == "BRSULACDAM12"
        before = {
            field: getattr(self, field).clone()
            for field in ("quantity", "principal", "rent_due", "fees_due")
        }
        expected = sum(
            (value := Decimal(str(float(self.quantity[i]))) * Decimal(str(ratio)))
            - value.to_integral_value(rounding=ROUND_FLOOR)
            for i in active
        )
        result = original(self, name, ratio, day, auction)
        error = abs(float(result) - float(expected))
        assert error < 1e-10
        for field in ("principal", "rent_due", "fees_due"):
            torch.testing.assert_close(
                getattr(self, field), before[field], rtol=0, atol=0
            )
        torch.testing.assert_close(
            self.quantity[due], before["quantity"][due], rtol=0, atol=0
        )
        boundary.append(
            dict(
                date=str(data.inputs.dates[day]),
                ratio=ratio,
                physically_due_quantity=float(before["quantity"][due].sum()),
                unreturned_quantity=float(before["quantity"][active].sum()),
                retained_due_principal=float(before["principal"][due].sum()),
                retained_active_principal=float(before["principal"][active].sum()),
                fraction=float(result),
                decimal_fraction=str(expected),
                error=error,
            )
        )
        return result

    LoanContracts.provision_fractions = qualify
    records = []
    try:
        for record in selected:
            book = bound_json(record["book"])
            folder = Path(record["book"]["path"]).parent
            config = book["provenance"]["config"].copy()
            config["monthly_spot_tariffs"] = tuple(
                MonthlySpotTariff(**v) for v in config["monthly_spot_tariffs"]
            )
            config["custody_assessments"] = tuple(
                CustodyAssessment(**v) for v in config["custody_assessments"]
            )
            view = phase_data(data, book["provenance"]["phase_hypothesis"])
            start = list(map(str, data.inputs.dates)).index(book["state_dates"][0])
            account = view.initial_account(start, LedgerConfig(**config))
            with np.load(folder / "account.npz") as saved:
                targets, nav = saved["targets"], saved["nav"]
            maximum = 0.0
            before_count = len(boundary)
            with torch.no_grad():
                for local, target in enumerate(targets):
                    row = view.step(
                        account,
                        torch.from_numpy(target),
                        start + local,
                        terminal=local == len(targets) - 1,
                    )
                    maximum = max(maximum, abs(float(row["nav"]) - nav[local]))
            assert maximum < 1e-7
            assert len(boundary) == before_count + 1
            records.append(
                dict(
                    source=record["book"],
                    intentions=binding(folder / "account.npz"),
                    days=len(nav),
                    maximum_nav_error_brl=maximum,
                    boundary=boundary[-1],
                )
            )
    finally:
        LoanContracts.provision_fractions = original
    report = dict(
        passed=True,
        plan=binding(out / "repair_plan.json"),
        books=records,
        runtime=binding(PROJECT / "research/src/brazil_rv/execution/loan_contracts.py"),
        recipe=binding(out / "executed_qualification.py"),
        seconds=perf_counter() - started,
        scope="Three newly exposed books only, saved intentions through the differentiable account versus independent saved NAV. Decimal fraction oracle on unreturned cohorts; original physically due quantities/principal/accrued rent/fees remain exact at provision. Shared loan subledger is not an independent whole-loan reconstruction. Existing successful books are reused: their old guard guaranteed the selected cohorts are unchanged by this repair.",
        tests="Four distinct new boundaries and sixteen affected existing cases pass. First batch19pass/1fixturefail; corrected fixture recognizes an actual same-day successor cover, then1pass/10deselected. Production bytes unchanged by fixture correction.",
    )
    write_json_atomic(out / "qualification.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
