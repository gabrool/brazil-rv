"""Read only saved four-period ensembles for actually billed roundoff hedge roots."""

import json
from pathlib import Path

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]
run = json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())
out = Path(run["scaling_hedge_roundoff_plan"]["path"]).parent / "prior_exposure"
out.mkdir(exist_ok=False)
(out / "executed.py").write_bytes(Path(__file__).read_bytes())
inputs = [
    run["stage_c_event_replays"],
    binding(Path(run["stage_c_data_replay_plan"]["path"]).parent / "replays.json"),
]
books = []
for ref in inputs:
    for record in bound_json(ref)["completed"]:
        phase, capital, arm, fold, member = record["key"].split("/")
        if (
            phase not in {"sources", "data_refit"}
            or capital != "10000000"
            or arm not in {"TE_full", "TE_wide"}
            or member != "ensemble"
        ):
            continue
        book = bound_json(record["book"])
        root = Path(record["book"]["path"]).parent
        fills = (
            pl.read_parquet(root / "fills.parquet")
            .filter(pl.col("purpose") == "hedge")
            .to_dicts()
        )
        charges = pl.read_parquet(root / "loan_charges.parquet").filter(
            pl.col("security_index") == 933
        )
        with np.load(root / "account.npz") as a:
            previous = np.r_[
                0, a["hedge_signed_shares"][:-1] * a["hedge_mark_price"][:-1]
            ]
            target = a["targets"][:, -1] * a["start_nav"]
        tiny = []
        for fill in fills:
            day = fill["fill_session"]
            scale = max(abs(previous[day]), abs(target[day]))
            if 0 < fill["gross_notional"] <= 8 * np.finfo(np.float64).eps * scale:
                rows = charges.filter(pl.col("opening_session") == day)
                tiny.append(
                    dict(
                        fill=fill,
                        relative_notional=fill["gross_notional"] / scale,
                        root_fee_brl=float(rows["fee"].sum()),
                        root_rent_brl=float(rows["rent"].sum()),
                    )
                )
        books.append(
            dict(
                key=record["key"],
                book=record["book"],
                tiny_fills=tiny,
                direct_tiny_root_fees_brl=sum(x["root_fee_brl"] for x in tiny),
                days=len(book["state_dates"]),
            )
        )
assert len(books) == 16, len(books)
report = dict(
    inputs=inputs,
    books=books,
    limits="Saved intentions/fills/charges only, no model/account replay. Relative threshold equals the frozen implementation correction. Opening-session charges identify actual tiny roots; this direct-fee inventory is not an adaptive total-path bound. All other account/source limitations remain.",
)
write_json_atomic(out / "report.json", report)
print(
    json.dumps(
        [
            {k: b[k] for k in ("key", "direct_tiny_root_fees_brl")}
            | dict(tiny_fills=len(b["tiny_fills"]))
            for b in books
        ]
    )
)
