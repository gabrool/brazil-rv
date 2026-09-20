"""Qualify the sparse account inputs and freeze only actually affected replays."""

from dataclasses import fields
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.corporate_replay import apply_corporate_replay, load_corporate_replay
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.portfolio_training import load_data

PROJECT = Path(__file__).resolve().parents[1]


def held_claims(root, dates, event, axis):
    path = root / "share_claim_positions.json"
    if not path.exists():
        return []
    return [
        c
        for c in json.loads(path.read_text())
        if c["successor_index"] == axis
        and c["signed_quantity"] != 0
        and str(dates[c["session"]]) >= event["effective_date"]
    ]


def extend_claim_scope():
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    root = Path(run["stage_c_root"]) / "event_replays"
    plan = bound_json(run["stage_c_event_replay_plan"])
    sources = bound_json(run["stage_c_event_source_admission"])
    parent = bound_json(run["economic_refit_inputs"])["store"]
    names = np.load(Path(parent["root"]) / "isin_index.npy").tolist()
    extra, reuse, receipts = [], [], []
    for rec in plan["reused"]:
        book = bound_json(rec["book"])
        path = Path(rec["book"]["path"]).parent
        selected = []
        for event in sources["events"]:
            claims = held_claims(
                path, book["state_dates"], event, names.index(event["isin"])
            )
            if claims:
                selected.append(dict(event=event["id"], claims=claims))
        if selected:
            extra.append(rec["key"])
            receipts.append(
                dict(
                    key=rec["key"],
                    source=binding(path / "share_claim_positions.json"),
                    selected=selected,
                )
            )
        else:
            reuse.append(rec)
    (root / "completed_initial_scope.json").write_bytes(
        (root / "replays.json").read_bytes()
    )
    update = {
        **plan,
        "included_keys": plan["included_keys"] + extra,
        "planned_new_books": plan["planned_new_books"] + len(extra),
        "reused": reuse,
        "supersedes_scope": run["stage_c_event_replay_plan"],
        "scope_correction": "Initial selection omitted claims whose underlying had already been sold. Actual ENAT/Brava interaction exposes this. Add every such saved claim from the62 reused books, independent of return sign; retain all98 completed corrected books.",
        "claim_exposure": receipts,
    }
    target = root / "claim_scope_plan.json"
    assert not target.exists()
    write_json_atomic(target, update)
    run["stage_c_event_replay_plan"] = binding(target)
    write_json_atomic(pointer, run)
    (root / "executed_claim_scope.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(additional_keys=extra, reused_books=len(reuse))))


def main():
    tick = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    out = Path(run["stage_c_root"]) / "event_replays"
    out.mkdir(exist_ok=False)
    (out / "executed_preparation.py").write_bytes(Path(__file__).read_bytes())
    previous = bound_json(run["stage_c_plan"])
    terms, calendar = load_corporate_replay(
        run["stage_c_event_candidate_terms"]["path"],
        run["stage_c_event_candidate_terms"]["sha256"],
    )
    data, cache = load_data(Path(previous["prior_root"]), "C6")
    inputs = data.inputs
    amended = apply_corporate_replay(
        inputs, terms, calendar, run["stage_c_event_candidate_terms"]["sha256"]
    )
    old_terms, _ = load_corporate_replay(
        run["corporate_replay"]["path"], run["corporate_replay"]["sha256"]
    )
    old = apply_corporate_replay(
        inputs, old_terms, calendar, run["corporate_replay"]["sha256"]
    )
    changes = {}
    unchanged = []
    for field in fields(inputs):
        k = field.name
        if isinstance(getattr(inputs, k), np.ndarray):
            if getattr(inputs, k) is getattr(amended, k):
                unchanged.append(k)
            elif getattr(old, k).shape == getattr(amended, k).shape:
                a, b = getattr(old, k), getattr(amended, k)
                changed = ~(np.equal(a, b) | (np.isnan(a) & np.isnan(b)))
                changes[k] = int(changed.sum())
    assert all(
        getattr(inputs, k) is getattr(amended, k)
        for k in (
            "active",
            "raw_close",
            "scores",
            "score_mask",
            "prior_feature_values",
            "scaled_midrank_targets",
            "target_scale_sigma",
        )
    )
    sources = bound_json(run["stage_c_event_source_admission"])
    names = inputs.security_ids
    # Existing terms are immutable aside from their per-manifest provenance.
    from dataclasses import replace

    for a, b in zip(old.share_distributions, amended.share_distributions, strict=False):
        assert replace(a, source=b.source) == b
    included, reused, exposure_rows = [], [], []
    saved = bound_json(run["stage_c_replays"])
    for record in saved["completed"]:
        book = bound_json(record["book"])
        root = Path(record["book"]["path"]).parent
        bdates = np.asarray(book["state_dates"], dtype="datetime64[D]")
        with np.load(root / "account.npz") as account:
            shares = account["signed_shares"]
            charges = pl.read_parquet(root / "loan_charges.parquet")
            reasons = []
            for e in sources["events"]:
                effect = np.datetime64(e["effective_date"])
                day = int(np.searchsorted(bdates, effect))
                if day >= len(bdates) or bdates[day] != effect:
                    continue
                n = names.index(e["isin"])
                held = 0.0 if day == 0 else float(shares[day - 1, n])
                loan_rows = charges.filter(
                    (pl.col("session") >= day) & (pl.col("security_index") == n)
                ).height
                # Include all later nonzero holdings too: legacy unresolved actions
                # can alter realized returns or an exit after the first effect close.
                later = bool(np.any(shares[day:, n] != 0))
                claims = held_claims(root, bdates, e, n)
                if held != 0 or loan_rows or later or claims:
                    reasons.append(
                        dict(
                            event=e["id"],
                            axis=n,
                            pre_effect_shares=held,
                            later_old_loan_charge_rows=loan_rows,
                            later_nonzero_inventory=later,
                            underlying_claim_rows=len(claims),
                        )
                    )
            if reasons:
                included.append(record["key"])
                exposure_rows.append(
                    dict(key=record["key"], reasons=reasons, previous=record["book"])
                )
            else:
                reused.append(record)
    plan = {
        **previous,
        "output_root": str(out),
        "corporate_terms": run["stage_c_event_candidate_terms"],
        "prior_replays": run["stage_c_replays"],
        "source_admission": run["stage_c_event_source_admission"],
        "included_keys": included,
        "planned_new_books": len(included),
        "reused": reused,
        "selection": "Any pre-effect inventory, later old-source inventory or post-effect source loan charge, across every saved book; no outcome/return gate. Unexposed old books remain reused. Data/forecast/static coordinates unchanged.",
        "status": "frozen_before_actual_event_corrected_books",
    }
    write_json_atomic(out / "plan.json", plan)
    write_json_atomic(
        out / "preparation.json",
        dict(
            cache=cache,
            source=run["stage_c_event_source_admission"],
            unchanged_original_arrays=unchanged,
            numeric_array_changes=changes,
            source_distribution_count=len(amended.share_distributions),
            scoped_replays=exposure_rows,
            reused_books=len(reused),
            new_books=len(included),
            seconds=perf_counter() - tick,
            limits="Sparse loader/pointer/control verification only, not actual book or model/data admission; retirement reopening and payment realization hypotheses retain separate qualification.",
        ),
    )
    run["stage_c_event_replay_plan"] = binding(out / "plan.json")
    run["stage_c_event_preparation"] = binding(out / "preparation.json")
    write_json_atomic(pointer, run)
    print(
        json.dumps(
            dict(
                new_books=len(included),
                reused_books=len(reused),
                changes=changes,
                seconds=perf_counter() - tick,
            )
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--extend-claims", action="store_true")
    args = parser.parse_args()
    if args.extend_claims:
        extend_claim_scope()
    else:
        main()
