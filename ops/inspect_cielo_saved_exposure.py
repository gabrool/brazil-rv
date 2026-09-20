"""Inspect saved V32 Cielo inventory and available charge tails, never replay."""

import json
from pathlib import Path
import shutil

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding

PROJECT = Path(__file__).resolve().parents[1]


def main():
    run = json.loads(
        (PROJECT / "docs/v2_economic_data_scaling_run.json").read_text(encoding="utf8")
    )
    root = Path(run["root"])
    out = []
    for capital in [10000000, 1000000, 5000000]:
        for scenario in ["base", "denied", "recall2", "recall4"]:
            p = (
                root
                / "loan_return_notices/adaptive_bound"
                / f"2024-05-02_{capital}_{scenario}"
            )
            book = json.loads((p / "book.json").read_text(encoding="utf8"))
            dates = book["state_dates"]
            with np.load(p / "account.npz") as z:
                sh = z["signed_shares"][:, 235]
            negative = np.flatnonzero(sh < 0)
            row = dict(
                capital=capital,
                scenario=scenario,
                source_book=binding(p / "book.json"),
                last_negative_date=dates[negative[-1]] if len(negative) else None,
                min_shares=float(sh.min()),
                shares_aug29=float(sh[dates.index("2024-08-29")]),
                shares_aug30=float(sh[dates.index("2024-08-30")]),
                charge_tail_status="detailed_charge_file_not_saved_for_this_variant",
            )
            if (p / "loan_charges.parquet").exists():
                charges = pl.read_parquet(p / "loan_charges.parquet").filter(
                    pl.col("security_index") == 235
                )
                charged = charges.filter((pl.col("rent") != 0) | (pl.col("fee") != 0))
                last = charged["session"].max()
                row.update(
                    charge_tail_status="inspected",
                    charge_source=binding(p / "loan_charges.parquet"),
                    last_nonzero_charge=dates[last] if last is not None else None,
                    rows_at_or_after_aug29=charges.filter(
                        pl.col("session") >= dates.index("2024-08-29")
                    ).height,
                )
            out.append(row)
    target = root / "historical_spot"
    shutil.copyfile(__file__, target / "cielo_exposure_executed.py")
    write_json_atomic(
        target / "cielo_saved_exposure.json",
        dict(
            status="no_held_redemption_loan_case_established",
            security="BRCIELACNOR3",
            axis=235,
            scope="Twelve V32 2024 books only. All stock positions flat Aug29/30; only three base books have saved detailed charges. Nine variant pending-loan tails cannot be independently established from the absent source-specific charge files; absence is not zero. No replay or preference/rate change. Cielo cent bound remains open; not an exhaustive old-model exposure search.",
            cases=out,
        ),
    )
    print(json.dumps(out))


if __name__ == "__main__":
    main()
