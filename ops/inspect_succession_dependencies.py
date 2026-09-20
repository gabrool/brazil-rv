"""Locate the two surviving-company rename dependencies without rebuilding data."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]


def main():
    tick = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text(encoding="utf8"))
    parent = bound_json(run["economic_refit_inputs"])["store"]
    store = Path(parent["root"])
    m = bound_json(
        dict(path=str(store / "manifest.json"), sha256=parent["manifest_sha256"])
    )
    dates, isins = (
        np.load(store / "date_index.npy"),
        np.load(store / "isin_index.npy").tolist(),
    )
    out = Path(run["root"]) / "event_composition/identity_scope"
    out.mkdir(exist_ok=False)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    sources = Path(run["root"]) / "held_event_sources"
    links = pl.read_parquet(store / m["tables"]["isin_succession_links"]["path"])
    old_m1 = bound_json(run["rename_m1_propagation"])
    assignments = pl.read_parquet(old_m1["assignments"]["path"])
    candidates, rows = [], []
    arrays = {
        k: np.load(store / m["arrays"][k]["path"], mmap_mode="r")
        for k in (
            "observed",
            "active",
            "raw_close",
            "entry_fill_allowed",
            "slow_timestep_valid",
            "fast_present",
            "target_to_close_valid",
        )
    }
    for prefix, successor, effect, protocol in (
        ("BRSSBR", "BRALSOACNOR5", "2019-08-06", "703585"),
        ("BRARZZ", "BRAZZAACNOR9", "2024-08-01", "1264889"),
    ):
        predecessors = [s for s in isins if s.startswith(prefix)]
        assert len(predecessors) == 1
        predecessor = predecessors[0]
        n, j = isins.index(predecessor), isins.index(successor)
        t = int(np.searchsorted(dates, np.datetime64(effect)))
        assert not arrays["observed"][t:, n].any()
        assert not arrays["observed"][:t, j].any()
        assert not links.filter(
            (pl.col("predecessor_isin") == predecessor)
            | (pl.col("successor_isin") == successor)
        ).height
        # No model/source consumer read crosses the development boundary.
        stop = min(t + 126, len(dates))
        for s in (predecessor, successor):
            axis = isins.index(s)
            for day in range(t - 60, stop):
                rows.append(
                    dict(
                        isin=s,
                        date=str(dates[day]),
                        **{
                            k: (
                                float(a[day, axis])
                                if k == "raw_close" and np.isfinite(a[day, axis])
                                else None
                                if k == "raw_close"
                                else bool(a[day, axis])
                            )
                            for k, a in arrays.items()
                        },
                    )
                )
        selected_assignments = assignments.filter(
            pl.col("isin").is_in([predecessor, successor])
        )
        selected_assignments.write_parquet(out / (prefix + "_assignments.parquet"))
        text = (sources / (protocol + ".txt")).read_text(encoding="utf8")
        assert (
            "Sonae Sierra Brasil S.A." if prefix == "BRSSBR" else "Arezzo&Co"
        ) in text
        candidates.append(
            dict(
                predecessor_isin=predecessor,
                successor_isin=successor,
                effect=effect,
                issuer_original=binding(sources / (protocol + ".pdf")),
                issuer_text=binding(sources / (protocol + ".txt")),
                issuer_evidence="Surviving issuer explicitly renames its own shares; acquired-company exchange ratio belongs to ALSC/SOMA, not this same-class continuation.",
                share_quantity_continuation="1:1/no cash is a candidate same-class identity interpretation from surviving issuer rename; verify original source clocks and typed identity before admission. It is not the index-history merge of acquired-company turnover.",
                predecessor_last_observed=str(
                    dates[np.flatnonzero(arrays["observed"][:, n])[-1]]
                ),
                successor_first_observed=str(
                    dates[np.flatnonzero(arrays["observed"][:, j])[0]]
                ),
                successor_first_active=str(
                    dates[np.flatnonzero(arrays["active"][:, j])[0]]
                ),
                first60_successor_eligible=int(
                    arrays["active"][t : min(t + 60, len(dates)), j].sum()
                ),
                first60_successor_fast_present=int(
                    arrays["fast_present"][t : min(t + 60, len(dates)), j].sum()
                ),
                assignments=binding(out / (prefix + "_assignments.parquet")),
                next_required="Separate dated identity/source admission, account disposal/loan convention if actually exposed, then bounded eligibility/full60/252 and126-session daily/risk/peer, filing, lending-denominator, native/scalar and auxiliary propagation where actual dependencies change. Reuse accepted controls and earlier algorithms; no raw census, loan alias or acquired-issuer history pooling.",
            )
        )
    write_json_atomic(out / "selected_rows.json", rows)
    result = dict(
        status="bounded_dependency_inventory_no_new_links_or_data_admitted",
        parent=parent,
        old_link_table={
            **m["tables"]["isin_succession_links"],
            "path": str(store / m["tables"]["isin_succession_links"]["path"]),
        },
        candidates=candidates,
        selected_rows=binding(out / "selected_rows.json"),
        row_count=len(rows),
        fields_per_row=len(arrays),
        accepted_store_unchanged=True,
        raw_m1_or_heldout_observations_read=False,
        seconds=perf_counter() - tick,
    )
    write_json_atomic(out / "manifest.json", result)
    run["succession_identity_dependency_scope"] = binding(out / "manifest.json")
    write_json_atomic(pointer, run)
    print(
        json.dumps(
            {k: result[k] for k in ("status", "candidates", "row_count", "seconds")}
        )
    )


if __name__ == "__main__":
    main()
