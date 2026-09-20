"""Separate restored eligibility from inherited M1 scalar history and audit clocks."""

import json
from pathlib import Path
import numpy as np
import polars as pl
from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.round5_derived import verified
from brazil_rv.v2.contract import INTRADAY_DAILY_FEATURES
from brazil_rv.v2.feature_spec import feature_specs, transform_feature_panel_into
from brazil_rv.v2.decision_clock import load_session_schedule
from finalize_m1_scalars import targets

p = Path(__file__).resolve().parents[1]
r = json.loads((p / "docs/v2_economic_data_scaling_run.json").read_text())
root = Path(r["root"]) / "m1_scalar_assembly"
h = bound_json(r["rename_history_propagation"])
m = bound_json(r["rename_m1_propagation"])
s = Path(m["parent"]["root"])
active = np.load(verified(h["arrays"]["active"]), mmap_mode="r")
dates = np.load(s / "date_index.npy")
ids = np.load(s / "isin_index.npy")
links = pl.read_parquet(verified(h["history_mapping"]))
report = []
for rec in m["results"]:
    n = int(np.flatnonzero(ids == rec["isin"])[0])
    e = links.filter(pl.col("successor_index") == n)[0, "effective_index"]
    raw = np.load(root / "qualified_control" / str(dates[e]) / "raw_control.npz")
    rows = raw["date_indices"]
    take = np.flatnonzero(rows >= e - 1)
    out = np.load(root / "amendments" / (rec["isin"] + "_output.npz"))
    v = np.empty(out["intraday_values"].shape, dtype=np.float32)
    ok = np.empty_like(v, dtype=bool)
    transform_feature_panel_into(
        raw["values"],
        raw["valid"],
        active[rows],
        feature_specs("intraday", INTRADAY_DAILY_FEATURES),
        v,
        ok,
        source_rows=take,
    )
    prior_valid = np.load(s / "intraday_valid.npy", mmap_mode="r")[rows[take]]
    base_targets = targets(raw, active[rows])
    report.append(
        {
            "isin": rec["isin"],
            "eligibility_only_feature_gains": int((ok & ~prior_valid).sum()),
            "history_incremental_valid_gains": int((out["intraday_valid"] & ~ok).sum()),
            "history_incremental_valid_losses": int(
                (ok & ~out["intraday_valid"]).sum()
            ),
            "history_incremental_shared_value_changes": int(
                ((v != out["intraday_values"]) & ok & out["intraday_valid"]).sum()
            ),
            "eligibility_only_target_gains": int(
                (
                    base_targets["target_to_close_valid"][take]
                    & ~np.load(s / "target_to_close_valid.npy", mmap_mode="r")[
                        rows[take]
                    ]
                ).sum()
            ),
            "history_incremental_target_gains": int(
                (
                    out["target_to_close_valid"]
                    & ~base_targets["target_to_close_valid"][take]
                ).sum()
            ),
        }
    )
schedule = {x.trade_date: x for x in load_session_schedule(verified(m["schedule"]))}
valid = np.load(s / "target_to_close_valid.npy", mmap_mode="r")
res = np.load(s / "target_to_close_normalized_residual.npy", mmap_mode="r")


def minute(value):
    return value.hour * 60 + value.minute


groups = {}
for t, d in enumerate(dates.astype(object)):
    x = schedule[d]
    duration = minute(x.continuous_close) - minute(x.continuous_open)
    prefix = minute(x.decision_time) - minute(x.continuous_open)
    g = groups.setdefault(
        (duration, prefix),
        {
            "sessions": 0,
            "valid_targets": 0,
            "clipped_valid_residuals": 0,
            "first": str(d),
            "last": str(d),
        },
    )
    g["sessions"] += 1
    g["valid_targets"] += int(valid[t].sum())
    g["clipped_valid_residuals"] += int((valid[t] & (np.abs(res[t]) == 5)).sum())
    g["last"] = str(d)
write_json_atomic(
    root / "attribution_and_clock.json",
    {
        "rename_contrasts": report,
        "clock_groups": [
            dict(session_minutes=k[0], cutoff=k[1], **v) for k, v in groups.items()
        ],
        "current_target_defaults": {"session_minutes": 405, "cutoff": 345},
        "dated_schedule": m["schedule"],
        "disposition": "Dated entry/close bars are correct. Target normalization still uses fixed remaining/full-session ratio. Recompute from original prefix RSS before optional head admission; clipped residuals cannot recover raw normalization exactly. Changing prefix RSS to a daily volatility estimator would be a separate hypothesis, not part of a clock repair.",
        "source_code": binding(p / "research/src/brazil_rv/v2/targets.py"),
        "reproducer": binding(Path(__file__)),
    },
)
print(json.dumps(report))
print(
    json.dumps(
        [dict(session_minutes=k[0], cutoff=k[1], **v) for k, v in groups.items()]
    )
)
