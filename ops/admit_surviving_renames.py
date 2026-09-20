"""Admit two surviving-company identities and their strictly prior liquidity.

This writes source-bound intermediate data, never an accepted model store. The
acquired companies' exchange ratios/history and loan contracts are separate.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.build_store import _route_decision_known_continuations
from brazil_rv.v2.contract import (
    UNIVERSE_MIN_HISTORY,
    UNIVERSE_MIN_MEDIAN_VOLUME_BRL,
    UNIVERSE_MIN_PRIOR_CLOSE_BRL,
    UNIVERSE_MIN_TRADED,
    UNIVERSE_PRIOR_SESSIONS,
)
from brazil_rv.v2.data_foundation import load_isin_link_allowlist, slow_history_links
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.decision_clock import load_session_schedule
from brazil_rv.v2.universe import build_daily_universe

PROJECT = Path(__file__).resolve().parents[1]


def main():
    tick = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text(encoding="utf8"))
    scope = bound_json(run["succession_identity_dependency_scope"])
    parent = bound_json(run["economic_refit_inputs"])["store"]
    root = Path(parent["root"])
    m = bound_json(
        dict(path=str(root / "manifest.json"), sha256=parent["manifest_sha256"])
    )
    out = Path(run["root"]) / "surviving_renames"
    out.mkdir(exist_ok=False)
    (out / "executed_admission.py").write_bytes(Path(__file__).read_bytes())
    config = PROJECT / "research/configs/v2/isin_links_allowlist.csv"
    (out / "prior_allowlist.csv").write_bytes(config.read_bytes())
    write_json_atomic(
        out / "plan.json",
        {
            "parent": parent,
            "scope": run["succession_identity_dependency_scope"],
            "registration": binding(
                PROJECT / "research/preregistrations/v2_economic_data_scaling.md"
            ),
            "closeout": run["stage_a_closeout_plan"],
            "contrast": "Add only SSBR->ALSO and ARZZ->AZZA surviving-company ordinary-share 1:1/no-cash identity continuations at effect and minute-end public knowledge. Preserve earlier links and all accepted source assignments/stores/fits.",
            "downstream": "First bind typed identity, actual prior-session liquidity and full60 mapping. Then only evidenced daily252/risk60/monthly126-peer, issuer/financial/lending-denominator, native/scalar and auxiliary dependencies; recompose ready gross-event outcomes with final risks in a new complete contract.",
            "limits": "No acquired-company history pooling, new loan alias, inferred physical custody date, model inference or heldout market observations. Source identity admission is not StageA or complete-store acceptance.",
        },
    )
    dates = np.load(root / "date_index.npy")
    isins = np.load(root / "isin_index.npy").tolist()
    assert len(isins) == 933 and dates[-1] < np.datetime64("2025-01-01")
    schedule_record = next(
        r
        for r in m["sources"]
        if Path(r.get("path", "")).name == "b3_session_schedule_reconstructed_v1.csv"
    )
    schedule = load_session_schedule(Path(schedule_record["path"]))
    clocks = {s.trade_date: s.decision_at for s in schedule}
    decisions = [clocks[d.astype(object)] for d in dates]
    source_dir = Path(run["root"]) / "held_event_sources"
    contracts = []
    for candidate, ticker, protocol, expected in zip(
        scope["candidates"],
        ("ALSO3", "AZZA3"),
        ("703585", "1264889"),
        ("20190805 05/08/2019 18:19", "20240731 31/07/2024 18:20"),
        strict=True,
    ):
        receipt = binding(source_dir / f"{protocol}_receipt.json")
        saved = bound_json(receipt)
        assert saved["pdf"] == candidate["issuer_original"]
        row = next(x for x in saved["issuer_rows"] if x["protocol"] == protocol)
        assert row["receipt"] == expected
        stamp = datetime.strptime(expected[9:], "%d/%m/%Y %H:%M").replace(
            tzinfo=ZoneInfo("America/Sao_Paulo")
        )
        available = (stamp + timedelta(minutes=1)).astimezone(timezone.utc)
        assert available < clocks[np.datetime64(candidate["effect"]).astype(object)]
        contracts.append(
            {
                "ticker": ticker,
                "predecessor_isin": candidate["predecessor_isin"],
                "successor_isin": candidate["successor_isin"],
                "effective_date": candidate["effect"],
                "first_known_at": available.isoformat(),
                "shares_received_per_prior_share": 1.0,
                "cash_entitlement_per_prior_share": 0.0,
                "currency": "BRL",
                "receipt": receipt,
                "original": candidate["issuer_original"],
                "text": candidate["issuer_text"],
                "interpretation": "Same surviving legal issuer and ordinary shares under a new ticker/ISIN; the acquired-company issuance does not multiply existing survivor shares. Unit/no-cash identity follows the original rename, not the acquired company's exchange ratio or B3 index-history merge.",
            }
        )
    write_json_atomic(
        out / "source_admission.json",
        {
            "status": "admitted_surviving_company_spot_identity_only",
            "contracts": contracts,
            "source_visual_qualification_reused": run["held_event_source_audit"],
            "source_assignment_inventory_reused": run[
                "succession_identity_dependency_scope"
            ],
            "restrictions": [
                "No loan quote/rate/history alias",
                "No acquired-company feature history",
                "No new issuer vintage or unrestricted financial history",
                "No observed custody/loan conversion claim",
            ],
        },
    )
    evidence = binding(out / "source_admission.json")
    rows = [
        {
            **{
                k: c[k]
                for k in (
                    "ticker",
                    "predecessor_isin",
                    "successor_isin",
                    "effective_date",
                    "first_known_at",
                    "shares_received_per_prior_share",
                    "cash_entitlement_per_prior_share",
                    "currency",
                )
            },
            "source": evidence["path"],
            "evidence_sha256": evidence["sha256"],
        }
        for c in contracts
    ]
    pl.DataFrame(rows).write_csv(out / "additional_allowlist.csv")
    selected_isins = [
        c[k] for c in contracts for k in ("predecessor_isin", "successor_isin")
    ]
    # Reuse audited normalized annual sources, selecting only four identities.
    sources = [
        r
        for r in m["sources"]
        if Path(r.get("path", "")).name
        in ("equities_daily_2019.parquet", "equities_daily_2024.parquet")
    ]
    daily = pl.concat(
        [
            pl.scan_parquet(r["path"])
            .filter(
                pl.col("isin").is_in(selected_isins) & (pl.col("market_type") == 10)
            )
            .collect()
            for r in sources
        ]
    )
    daily.write_parquet(out / "identity_source_rows.parquet")
    added = load_isin_link_allowlist(out / "additional_allowlist.csv", daily)
    old_links = pl.read_parquet(root / m["tables"]["isin_succession_links"]["path"])
    links = pl.concat([old_links, added]).sort("successor_first_date")
    # Chronological continuation roots also update the later ALSO->ALOS chain.
    roots, resolved = {}, []
    for row in links.iter_rows(named=True):
        ancestor = roots.get(row["predecessor_isin"], row["predecessor_isin"])
        roots[row["successor_isin"]] = ancestor
        resolved.append(ancestor)
    links = links.with_columns(pl.Series("continuation_isin", resolved))
    unchanged = links.filter(
        ~pl.col("predecessor_isin").is_in([c["predecessor_isin"] for c in contracts])
    )
    assert unchanged.drop("continuation_isin").equals(
        old_links.sort("successor_first_date").drop("continuation_isin")
    )
    links.write_parquet(out / "isin_succession_links.parquet")
    slow_history_links(links, dates, isins, decisions).write_parquet(
        out / "slow_history_links.parquet"
    )

    def old(k):
        return np.load(root / m["arrays"][k]["path"], mmap_mode="r")

    packed, results = {}, []
    for row in added.iter_rows(named=True):
        pair = [row["predecessor_isin"], row["successor_isin"]]
        axes = [isins.index(s) for s in pair]
        t = int(np.searchsorted(dates, np.datetime64(row["effective_date"])))
        # 253 prior sessions preserve all liquidity/history thresholds; only 61
        # following sessions are needed for the new-universe difference/control.
        start, stop = t - 253, min(t + 61, len(dates))
        days = dates[start:stop]
        selected = np.arange(t - start, len(days))
        close = np.round(old("raw_close")[start:stop][:, axes].astype(np.float64), 2)
        volume = old("volume_brl")[start:stop][:, axes].astype(np.float64)
        quotes = daily.filter(
            pl.col("isin").is_in(pair)
            & pl.col("trade_date").is_between(
                days[0].astype(object), days[-1].astype(object)
            )
        )
        qi = np.searchsorted(days, quotes["trade_date"].to_numpy())
        qj = np.array([pair.index(s) for s in quotes["isin"]])
        volume[qi, qj] = quotes["volume_brl"].to_numpy()
        observed, traded, activity = (
            old(k)[start:stop][:, axes].copy()
            for k in ("observed", "trade_observed", "activity_valid")
        )
        kwargs = dict(
            trade_observed=traded,
            activity_valid=activity,
            source_session_complete=old("source_session_complete")[start:stop, 0],
        )
        control = build_daily_universe(close, volume, observed, **kwargs)
        np.testing.assert_array_equal(
            control.active[selected], old("active")[start:stop][:, axes][selected]
        )
        route = dict(
            dates=days,
            isins=pair,
            links=added.filter(pl.col("predecessor_isin") == pair[0]),
            decision_timestamps=decisions[start:stop],
            raw_close=close,
            volume_brl=volume,
            trades=old("trade_count")[start:stop][:, axes],
            observed=observed,
            trade_observed=traded,
            activity_valid=activity,
            ambiguous_action=np.zeros_like(observed),
            shareholder_wealth_arrays=(),
        )
        linked = _route_decision_known_continuations(**route)
        after = build_daily_universe(
            linked.close_brl,
            linked.volume_brl,
            linked.observed,
            trade_observed=linked.trade_observed,
            activity_valid=linked.activity_valid,
            source_session_complete=kwargs["source_session_complete"],
        )
        active = after.active & linked.claim_owner
        np.testing.assert_array_equal(
            active[: selected[0]], control.active[: selected[0]]
        )
        # Independent direct prior-window oracle: never consult today's print.
        for d in selected:
            n = UNIVERSE_PRIOR_SESSIONS
            for j in (0, 1):
                prior = np.flatnonzero(linked.observed[max(0, d - n) : d, j])
                first = np.flatnonzero(linked.observed[:, j])
                price = (
                    linked.close_brl[max(0, d - n) + prior[-1], j]
                    if len(prior)
                    else np.nan
                )
                expected = bool(
                    linked.claim_owner[d, j]
                    and linked.activity_valid[d - n : d, j].all()
                    and sum(linked.trade_observed[d - n : d, j]) >= UNIVERSE_MIN_TRADED
                    and np.median(linked.volume_brl[d - n : d, j])
                    >= UNIVERSE_MIN_MEDIAN_VOLUME_BRL
                    and price >= UNIVERSE_MIN_PRIOR_CLOSE_BRL
                    and d - first[0] >= UNIVERSE_MIN_HISTORY
                )
                assert bool(active[d, j]) == expected
        # Source knowledge moved past effect must not expose inherited history.
        late = route["links"].with_columns(
            pl.lit(decisions[t + 1]).alias("first_known_at")
        )
        delayed = _route_decision_known_continuations(**{**route, "links": late})
        np.testing.assert_array_equal(delayed.observed, observed)
        np.testing.assert_array_equal(delayed.close_brl, close)
        for key, arr in (
            ("active", active),
            ("prior_reference_close", after.prior_close_brl.astype(np.float32)),
        ):
            # Public successor coordinates remain empty before effect.
            prior = old(key)[start:stop][:, axes][selected]
            vals = arr[selected]
            ix = np.argwhere(~((prior == vals) | (np.isnan(prior) & np.isnan(vals))))
            packed.setdefault(key + "__indices", []).append(
                np.column_stack((ix[:, 0] + t, np.array(axes)[ix[:, 1]]))
            )
            packed.setdefault(key + "__values", []).append(vals[tuple(ix.T)])
        history = old("slow_timestep_valid")[start:stop][:, axes].copy()
        history[selected, 1] = True  # Prior predecessor observations exist.
        prior = old("slow_timestep_valid")[start:stop][:, axes][selected]
        ix = np.argwhere(prior != history[selected])
        packed.setdefault("slow_timestep_valid__indices", []).append(
            np.column_stack((ix[:, 0] + t, np.array(axes)[ix[:, 1]]))
        )
        packed.setdefault("slow_timestep_valid__values", []).append(
            history[selected][tuple(ix.T)]
        )
        np.savez_compressed(
            out / (pair[0] + "_liquidity.npz"),
            dates=days,
            global_rows=np.arange(start, stop),
            axes=axes,
            close=close,
            volume=volume,
            observed=observed,
            traded=traded,
            activity=activity,
            control=control.active,
            active=active,
            prior_close=after.prior_close_brl,
            prior_traded=after.prior_traded_sessions,
            median_volume=after.prior_median_volume_brl,
            history=after.history_sessions,
        )
        results.append(
            {
                "predecessor": pair[0],
                "successor": pair[1],
                "source_rows": len(days),
                "decisions": len(selected),
                "added": int((active[selected] & ~control.active[selected]).sum()),
                "retired": int((~active[selected] & control.active[selected]).sum()),
                "added_dates": [
                    str(x)
                    for x in days[selected][
                        active[selected, 1] & ~control.active[selected, 1]
                    ]
                ],
                "control_cells": int(control.active[selected].size),
            }
        )
    np.savez_compressed(
        out / "liquidity_deltas.npz",
        **{k: np.concatenate(v) for k, v in packed.items()},
    )
    # Advance the code allowlist only after the source, typed join and causal
    # liquidity checks pass. Old saved stores retain their bound old table.
    prior_csv = (out / "prior_allowlist.csv").read_text(encoding="utf8").rstrip()
    additional = (
        (out / "additional_allowlist.csv").read_text(encoding="utf8").splitlines()[1:]
    )
    config.write_text(prior_csv + "\n" + "\n".join(additional) + "\n", encoding="utf8")
    report = {
        "status": "source_and_liquidity_admitted_dependencies_pending",
        "parent": parent,
        "plan": binding(out / "plan.json"),
        "source_admission": evidence,
        "allowlist": binding(config),
        "links": binding(out / "isin_succession_links.parquet"),
        "history_mapping": binding(out / "slow_history_links.parquet"),
        "liquidity_deltas": binding(out / "liquidity_deltas.npz"),
        "normalized_sources": sources,
        "selected_source_rows": binding(out / "identity_source_rows.parquet"),
        "schedule": schedule_record,
        "cases": results,
        "seconds": perf_counter() - tick,
        "remaining": "Daily/peer/risk, issuer and auxiliary/native/scalar dependencies, final target composition and complete-store acceptance; Jan25 calendar and integratedA. No model result.",
    }
    write_json_atomic(out / "manifest.json", report)
    run["surviving_rename_admission"] = binding(out / "manifest.json")
    write_json_atomic(pointer, run)
    print(
        json.dumps(
            {"status": report["status"], "cases": results, "seconds": report["seconds"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
