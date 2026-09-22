"""Retain absent intraday streams and carry only actual predecessor source ages."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.feature_spec import observation_age_sessions_into
from repair_scaling_inputs import PROJECT, context, record


def main():
    tick = perf_counter()
    run, plan, root, m, _ = context()
    admission = bound_json(run["scaling_data_identity"])
    out = Path(bound_json(run["scaling_data_workspace"])["root"]) / "scalars"
    out.mkdir(exist_ok=False)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    assignment = next(
        r
        for r in m["sources"]
        if Path(r["path"]).name == "xp_accepted_source_assignments_v1.parquet"
    )
    assignments = pl.read_parquet(assignment["path"])
    dates, names = (
        np.load(root / "date_index.npy"),
        np.load(root / "isin_index.npy").tolist(),
    )
    old = np.load(root / "active.npy", mmap_mode="r")
    active = old.copy()
    with np.load(admission["deltas"]["path"]) as z:
        active[tuple(z["active__indices"].T)] = z["active__values"]
    membership_changes = active != old
    old_age = np.load(root / "intraday_age_sessions.npy", mmap_mode="r")
    age = old_age.copy()
    links = pl.read_parquet(admission["history_mapping"]["path"])
    write_json_atomic(
        out / "plan.json",
        dict(
            parent=plan["parent"],
            admission=run["scaling_data_identity"],
            assignments=assignment,
            implementation=binding(
                PROJECT / "research/src/brazil_rv/v2/intraday_features.py"
            ),
            contrast="All four new successors have no accepted M1 assignment through2024. Do not invent a stream or rerun missing scalar reducers. Scalar values/support/to-close inputs stay unchanged: added/retired membership cells have no current intraday support, and corrected scalar-action dates precede relevant M1 assignments. The WIZS source ages strictly before effect pass to WIZC after effect/knowledge and continue ageing without observations; other absent histories retain-1. Retired predecessor ages become-1. Guararapes native sigma normalization is independent of raw scalar/to-close formulas.",
        ),
    )
    controls = []
    for key in (
        "intraday_values",
        "intraday_valid",
        "fast_present",
        "target_to_close_valid",
    ):
        a = np.load(root / (key + ".npy"), mmap_mode="r")[membership_changes]
        assert not a.any(), key
        controls.append(dict(key=key, cells=a.size, zero_support_or_value=True))
    scopes = []
    for event in plan["history"]:
        p, n = names.index(event["isin"]), names.index(event["successor_isin"])
        edge = links.filter(
            (pl.col("predecessor_index") == p) & (pl.col("successor_index") == n)
        ).row(0, named=True)
        effect = edge["effective_index"]
        rows = np.arange(effect, len(dates))
        assert assignments.filter(
            (pl.col("isin") == event["successor_isin"])
            & (pl.col("first_overlap_date") <= "2024-12-30")
        ).is_empty()
        for key in ("intraday_valid", "fast_present", "target_to_close_valid"):
            assert not np.load(root / (key + ".npy"), mmap_mode="r")[rows, n].any()
        seed = old_age[effect - 1, [p, n]]
        mask = np.zeros((len(dates), 2, 20), bool)
        source_age = np.full(mask.shape, -1, np.float32)
        mask[effect - 1], source_age[effect - 1] = seed >= 0, seed
        rule = dict(edge, predecessor_index=0, successor_index=1)
        outputs = []
        for label, membership, rules in (
            ("control", old, ()),
            ("corrected", active, (rule,)),
        ):
            result = np.empty((len(rows), 2, 20), np.float32)
            observation_age_sessions_into(
                mask,
                membership[:, [p, n]],
                result,
                source_rows=rows,
                source_age_sessions=source_age,
                history_links=rules,
            )
            if label == "control":
                np.testing.assert_array_equal(result, old_age[rows][:, [p, n]])
            outputs.append(result)
        last = np.where(seed[0] >= 0, effect - 1 - seed[0], -1)
        expected = np.where(
            last[None, :] >= 0, rows[:, None] - last[None, :], -1
        ).astype(np.float32)
        gate = max(effect, edge["known_index"])
        expected[(rows < gate) | ~active[rows, n]] = -1
        np.testing.assert_array_equal(outputs[1][:, 1], expected)
        age[rows[:, None], [p, n]] = outputs[1]
        # Later predecessor prints can never refresh the successor's frozen source clock.
        mask[effect + 3 :, 0] = True
        source_age[effect + 3 :, 0] = 0
        future = np.empty_like(outputs[1])
        observation_age_sessions_into(
            mask,
            active[:, [p, n]],
            future,
            source_rows=rows,
            source_age_sessions=source_age,
            history_links=(rule,),
        )
        np.testing.assert_array_equal(future[:, 1], outputs[1][:, 1])
        np.savez_compressed(
            out / (event["id"] + "_ages.npz"),
            rows=rows,
            columns=[p, n],
            seed=seed,
            control=outputs[0],
            corrected=outputs[1],
        )
        scopes.append(
            dict(
                id=event["id"],
                seed_date=str(dates[effect - 1]),
                seed=seed.tolist(),
                control_cells=outputs[0].size,
                independent_cells=expected.size,
                future_predecessor_isolation=True,
            )
        )
    age[~active] = -1
    ix = np.argwhere(age != old_age)
    key = "intraday_age_sessions"
    np.savez_compressed(
        out / "deltas.npz",
        **{key + "__indices": ix, key + "__values": age[tuple(ix.T)]},
    )
    selected_names = list(
        {e[k] for e in plan["history"] for k in ("isin", "successor_isin")}
        | {e["isin"] for e in plan["scalars"]}
    )
    assignments.filter(pl.col("isin").is_in(selected_names)).write_parquet(
        out / "selected_assignment_metadata.parquet"
    )
    result = dict(
        status="qualified_scalar_clocks_no_new_intraday_observations",
        plan=binding(out / "plan.json"),
        scopes=scopes,
        controls=controls,
        changes=len(ix),
        new_eligible=int((active & ~old).sum()),
        scalar_validity_changes=0,
        to_close_changes=0,
        deltas=binding(out / "deltas.npz"),
        seconds=perf_counter() - tick,
    )
    record(run, "scaling_data_scalars", out / "manifest.json", result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "changes",
                    "new_eligible",
                    "scalar_validity_changes",
                    "to_close_changes",
                    "seconds",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
