"""Read sealed development books; attribute recorded P&L without replay or fitting."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np


REPO = Path(__file__).resolve().parents[1]
FOLDS = ("F2", "F6", "F10", "F14")
COMPONENTS = {
    "equity_gross": "gross_long_short_spread_pnl_bps",
    "hedge_gross": "hedge_gross_pnl_bps",
    "free_cash_interest": "free_cash_interest_bps",
    "short_proceeds_interest": "short_proceeds_interest_bps",
    "equity_borrow": "equity_borrow_observed_rate_bps",
    "borrow_fee": "equity_borrow_registration_fee_bps",
    "hedge_borrow": "hedge_borrow_bps",
    "hedge_trading": "hedge_cost_bps",
    "cash_benchmark": "cdi_benchmark_bps",
    "net": "net_excess_all_cash_bps",
    "turnover": "turnover_fraction_nav",
    "gross_exposure": "deployed_gross_fraction_nav",
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def describe(data, source, digest):
    economics = data["economics"]
    rows = economics["headline_audit"]["daily_state"]
    series = {
        key: np.array([r[field] for r in rows], dtype=float)
        for key, field in COMPONENTS.items()
    }
    series["equity_trading"] = np.array(
        [r["turnover_cost_bps"] - r["hedge_cost_bps"] for r in rows]
    )
    series["financing_net_benchmark"] = (
        series["free_cash_interest"]
        + series["short_proceeds_interest"]
        - series["cash_benchmark"]
    )
    reconstructed = (
        series["equity_gross"]
        + series["hedge_gross"]
        + series["financing_net_benchmark"]
        - sum(
            series[k]
            for k in [
                "equity_trading",
                "hedge_trading",
                "equity_borrow",
                "borrow_fee",
                "hedge_borrow",
            ]
        )
    )
    error = float(np.max(np.abs(reconstructed - series["net"])))
    if error > 1e-7 or not all(np.isfinite(v).all() for v in series.values()):
        raise ValueError(f"Saved book does not reconcile: {source}, {error}")
    summary = economics["headline"]
    holdings = economics["headline_audit"]["holding_age_distribution"]
    extras = [
        "mean_absolute_ex_ante_beta_after_hedge",
        "mean_gross_fraction_nav",
        "mean_gross_fraction_nav_including_hedge",
        "maximum_absolute_hedge_fraction_nav",
        "hedge_capped_session_count",
        "terminal_settlement_notional_fraction_nav",
        "economics_unresolved",
        "terminal_nav",
        "terminal_nav_settlement_haircut_scenario",
        "exit_instructions_by_cause",
        "blocked_entry_candidates_by_cause",
        "zero_entry_days_by_cause",
        "average_holding_sessions_approximation",
    ]
    return {
        "source": source,
        "sha256": digest,
        "dates": [r["date"] for r in rows],
        "primary_ic": data["mean_daily_primary_neutral_target_ic"],
        "means": {k: float(v.mean()) for k, v in series.items()},
        "daily": {k: v.tolist() for k, v in series.items()},
        "max_reconciliation_error_bps": error,
        "summary": {k: summary.get(k) for k in extras},
        "holding_age": {
            k: holdings[k]
            for k in [
                "mean_sessions",
                "median_sessions",
                "p95_sessions",
                "maximum_sessions",
            ]
        },
        "policy": economics["execution_policy"],
        "d5_only_net_bps": economics["d5_only_diagnostic"]["summary"][
            "mean_net_excess_bps_per_day"
        ],
        "contract": economics["contract"],
    }


def name_attribution(data):
    """Exact F2 gross mark/flow attribution; never infer a name from its ticker."""
    audit = data["economics"]["headline_audit"]
    holdings, fills, claims = (
        defaultdict(dict),
        defaultdict(lambda: defaultdict(float)),
        defaultdict(lambda: defaultdict(float)),
    )
    weights = defaultdict(list)
    for row in audit["holding_age_distribution"]["holdings"]:
        holdings[row["date"]][row["security"]] = row["signed_marked_value"]
        weights[row["security"]].append(row["signed_weight_fraction_nav"])
    for row in audit["fills"]:
        if row["purpose"] != "hedge":
            fills[row["fill_date"]][row["security"]] += row["gross_notional"] * (
                1 if row["side"] == "buy" else -1
            )
    for row in audit["claims_and_action_attribution"]["rows"]:
        if row["held_prior_to_action"]:
            if row["security"] != row["successor_security"]:
                raise ValueError(
                    "Name attribution needs an explicit cross-identity transfer"
                )
            claims[row["date"]][row["security"]] += row["cash_claim_signed"] or 0.0
    previous, contributions, errors = {}, defaultdict(float), []
    daily = audit["daily_state"]
    for row in daily:
        day = row["date"]
        current = holdings[day]
        names = (
            previous.keys() | current.keys() | fills[day].keys() | claims[day].keys()
        )
        pnl = {
            n: (
                current.get(n, 0)
                - previous.get(n, 0)
                - fills[day].get(n, 0)
                + claims[day].get(n, 0)
            )
            / row["start_nav"]
            * 1e4
            for n in names
        }
        errors.append(sum(pnl.values()) - row["gross_long_short_spread_pnl_bps"])
        for name, value in pnl.items():
            contributions[name] += value / len(daily)
        previous = current
    error = max(map(abs, errors))
    if error > 1e-7:
        raise ValueError(f"Name gross attribution does not reconcile: {error}")
    return {
        "max_daily_reconciliation_error_bps": error,
        "population": "F2 original positions, own NAV denominator each day; gross contributions are not net returns or counterfactual removals",
        "names": {
            n: {
                "gross_bps_per_day": v,
                "held_days": len(weights[n]),
                "mean_signed_weight_when_held": float(np.mean(weights[n])),
            }
            for n, v in sorted(contributions.items(), key=lambda x: -x[1])
        },
    }


def main():
    result = {
        "status": "completed",
        "interpretation": "recorded-book attribution, not counterfactual replay or a new experiment",
        "folds": list(FOLDS),
        "books": {},
        "sources": {},
        "comparisons": {},
    }
    raw_f2 = {}

    def accept(arm, fold, raw, source, expected=None):
        digest = hashlib.sha256(raw).hexdigest()
        if expected is not None and digest != expected:
            raise ValueError(
                f"Recovered member differs from sealed inventory: {source}"
            )
        data = json.loads(raw)
        result["books"].setdefault(arm, {})[fold] = describe(data, source, digest)
        if fold == "F2" and arm in [
            "R6_C6",
            "current_TE_all",
            "R6_C6_fresh_p",
            "current_S0",
        ]:
            raw_f2[arm] = data

    recovery = read(REPO / "docs/v2_post_data_recovery.json")
    selected = {
        f"aggregates/{a}/{f}/evaluation.json"
        for a in ["S0", "S0_common", "TE_all", "C1_all", "TL_slow"]
        for f in FOLDS
    }
    bindings = {}
    for archive in recovery["archives_in_restore_order"]:
        for item in read(archive["manifest"])["files"]:
            if item["path"] in selected:
                bindings[item["path"]] = (archive["archive"], item["sha256"])
    if set(bindings) != selected:
        raise ValueError("Requested current books absent from sealed recovery")
    grouped = defaultdict(set)
    for path, (archive, _) in bindings.items():
        grouped[archive].add(path)
    for archive, wanted in grouped.items():
        print(f"Reading {Path(archive).name}: {len(wanted)} books", flush=True)
        with tarfile.open(archive, "r|xz") as stream:
            for member in stream:
                if member.name not in wanted:
                    continue
                _, arm, fold, _ = member.name.split("/")
                accept(
                    "current_" + arm,
                    fold,
                    stream.extractfile(member).read(),
                    archive + "::" + member.name,
                    bindings[member.name][1],
                )

    ops = read(REPO / "docs/v2_round6_operations.json")
    r6 = Path(ops["complete_arm_recoveries"]["S0"]["local_root"])
    for arm, group in [
        ("S0", "session1"),
        ("C6", "session2"),
        ("C6_fresh_p", "session2"),
    ]:
        for fold in FOLDS:
            path = r6 / "aggregates" / group / arm / fold / "evaluation.json"
            accept("R6_" + arm, fold, path.read_bytes(), str(path))
    pointer = read(REPO / "docs/v2_round7_recovery.json")
    # Historical reproduction root is supplied by its own immutable recovery record.
    original = Path(pointer["local_root"]) / pointer["original"]["root"]
    for fold in FOLDS:
        path = original / "aggregates/seeds_11_29_47/A0" / fold / "evaluation.json"
        accept("R7_A0", fold, path.read_bytes(), str(path))

    for arm, folds in result["books"].items():
        result.setdefault("pooled", {})[arm] = {
            k: float(np.mean([x for f in FOLDS for x in folds[f]["daily"][k]]))
            for k in next(iter(folds.values()))["daily"]
        }
    for left, right in [
        ("R6_C6", "current_TE_all"),
        ("current_TE_all", "current_S0"),
        ("R6_C6", "R6_C6_fresh_p"),
        ("R7_A0", "current_S0"),
    ]:
        deltas = {}
        for f in FOLDS:
            a, b = result["books"][left][f], result["books"][right][f]
            if a["dates"] != b["dates"]:
                raise ValueError("Attribution dates differ")
            deltas[f] = {k: a["means"][k] - b["means"][k] for k in a["means"]}
        result["comparisons"][left + "_minus_" + right] = {
            "folds": deltas,
            "pooled": {
                k: result["pooled"][left][k] - result["pooled"][right][k]
                for k in deltas[FOLDS[0]]
            },
        }
    result["F2_name_attribution"] = {
        arm: name_attribution(data) for arm, data in raw_f2.items()
    }
    a = result["F2_name_attribution"]["R6_C6"]["names"]
    b = result["F2_name_attribution"]["current_TE_all"]["names"]
    result["F2_name_gross_gap_C6_minus_TE_all"] = dict(
        sorted(
            (
                (
                    n,
                    a.get(n, {}).get("gross_bps_per_day", 0)
                    - b.get(n, {}).get("gross_bps_per_day", 0),
                )
                for n in a.keys() | b.keys()
            ),
            key=lambda x: -x[1],
        )
    )
    # Keep a bounded local source excerpt for subsequent name/action investigation.
    output = REPO / "docs/v2_execution_reassessment_evidence.json"
    raw = (json.dumps(result, indent=2, allow_nan=False) + "\n").encode("utf-8")
    output.write_bytes(raw)
    output.with_suffix(".json.sha256").write_bytes(
        (hashlib.sha256(raw).hexdigest() + "\n").encode()
    )
    derived = Path("C:/quant-data/b3/interim/post_data_execution_review")
    derived.mkdir(exist_ok=True)
    for arm, data in raw_f2.items():
        (derived / f"{arm}_F2.json").write_bytes(json.dumps(data).encode("utf-8"))
    print(json.dumps(result["pooled"], indent=2), flush=True)


if __name__ == "__main__":
    main()
