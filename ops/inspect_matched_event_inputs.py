"""Inspect only exposed corporate boundaries in the accepted model store."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]
EVENTS = (
    ("LCAM", "BRLCAMACNOR3", "BRRENTACNOR4", "2022-07-04"),
    ("JSL_holding", "BRJSLGACNOR2", "BRSIMHACNOR0", "2020-09-18"),
    ("JSL_logistics", "BRJSLGA02OR1", "BRJSLGACNOR2", "2020-11-11"),
    ("RRRP", "BRRRRPACNOR5", "BRBRAVACNOR3", "2024-09-09"),
    ("SULA", "BRSULACDAM12", "BRRDORACNOR8", "2022-12-26"),
    ("VIVT", "BRVIVTACNPR7", "BRVIVTACNOR0", "2020-11-23"),
    ("MODL", "BRMODLCDAM13", "BRMODLACNOR2", "2022-09-19"),
    ("TIM", "BRTIMPACNOR1", "BRTIMSACNOR5", "2020-10-13"),
    ("BKBR", "BRBKBRACNOR4", "BRZAMPACNOR5", "2022-10-26"),
    ("VVAR", "BRVVARCDAM10", "BRVVARACNOR1", "2018-11-26"),
)
FIELDS = (
    "raw_close",
    "observed",
    "active",
    "entry_fill_allowed",
    "fast_present",
    "action_has_action",
    "action_session_resolved",
    "action_shares_per_prior_share",
    "action_cash_per_prior_share",
    "action_successor_index",
    "prior_reference_close",
    "shareholder_wealth_close",
    "target_scale_sigma",
)


def main():
    tick = perf_counter()
    run = json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())
    parent = bound_json(run["economic_refit_inputs"])["store"]
    store = Path(parent["root"])
    m = bound_json(
        dict(path=str(store / "manifest.json"), sha256=parent["manifest_sha256"])
    )
    out = Path(run["stage_c_root"]) / "event_input_scope"
    out.mkdir(exist_ok=False)
    write_json_atomic(
        out / "plan.json",
        dict(
            parent=parent,
            events=EVENTS,
            fields=FIELDS,
            scope="Two prior and three effect/following sessions per source/destination; successor first60 eligibility and dated M1 metadata. Enumerate later source actions to detect conflicts; no reducers or raw-source census.",
            exposure=run["stage_c_exposure_audit"],
        ),
    )
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    dates = np.load(store / "date_index.npy")
    names = np.load(store / "isin_index.npy").tolist()
    arrays = {k: np.load(store / m["arrays"][k]["path"], mmap_mode="r") for k in FIELDS}
    rows, scope = [], []
    selected = set()
    for label, pred, succ, day in EVENTS:
        start = int(np.searchsorted(dates, np.datetime64(day)))
        assert str(dates[start]) == day
        axes = []
        for isin in (pred, succ):
            if isin not in names:
                axes.append(dict(isin=isin, present=False))
                continue
            selected.add(isin)
            n = names.index(isin)
            for r in range(start - 2, min(start + 3, len(dates))):
                rows.append(
                    dict(
                        event=label,
                        date=str(dates[r]),
                        isin=isin,
                        axis=n,
                        **{k: v[r, n].item() for k, v in arrays.items()},
                    )
                )
            later = np.flatnonzero(arrays["action_has_action"][start:, n]) + start
            axes.append(
                dict(
                    isin=isin,
                    present=True,
                    axis=n,
                    first60_eligible=int(arrays["active"][start : start + 60, n].sum()),
                    first60_native=int(
                        arrays["fast_present"][start : start + 60, n].sum()
                    ),
                    later_actions=[
                        dict(
                            date=str(dates[r]),
                            q=float(arrays["action_shares_per_prior_share"][r, n]),
                            cash=float(arrays["action_cash_per_prior_share"][r, n]),
                        )
                        for r in later
                    ],
                )
            )
        scope.append(dict(event=label, effective_date=day, axes=axes))
    pl.DataFrame(rows).write_parquet(out / "boundaries.parquet")
    tables = {}
    for key in ("security_master", "native_fast_security_mapping", "issuer_identity"):
        p = store / m["tables"][key]["path"]
        schema = pl.read_parquet_schema(p)
        column = next(k for k in ("isin", "security_id") if k in schema)
        frame = (
            pl.scan_parquet(p).filter(pl.col(column).is_in(sorted(selected))).collect()
        )
        target = out / (key + ".parquet")
        frame.write_parquet(target)
        tables[key] = dict(
            source=binding(p),
            selected=binding(target),
            rows=frame.height,
            columns=frame.columns,
        )
    write_json_atomic(
        out / "manifest.json",
        dict(
            parent=parent,
            events=scope,
            tables=tables,
            boundaries=binding(out / "boundaries.parquet"),
            cells=len(rows) * len(FIELDS),
            seconds=perf_counter() - tick,
        ),
    )
    print(
        json.dumps(
            dict(
                events=scope,
                tables=tables,
                rows=len(rows),
                seconds=perf_counter() - tick,
            )
        )
    )


if __name__ == "__main__":
    main()
