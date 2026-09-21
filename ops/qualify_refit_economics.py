"""Qualify new portfolio-input composition without repeating data reducers."""

from collections import defaultdict
import json
from pathlib import Path
import pickle
from time import perf_counter

import numpy as np
import polars as pl

from brazil_rv.execution.portfolio_policy import ledger_arguments
from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.portfolio_training import windows

PROJECT = Path(__file__).resolve().parents[1]


def main():
    tick = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    source = bound_json(run["stage_c_refit_economics"])
    plan = bound_json(source["plan"])
    root = Path(source["store"]["root"])
    manifest = bound_json(
        dict(
            path=str(root / "manifest.json"), sha256=source["store"]["manifest_sha256"]
        )
    )
    out = Path(source["cache"]["path"]).parent / "qualification"
    out.mkdir(exist_ok=False)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    assert sha256_file(Path(source["cache"]["path"])) == source["cache"]["sha256"]
    with Path(source["cache"]["path"]).open("rb") as f:
        data = pickle.load(f)
    inputs = data.inputs
    indices = inputs.session_indices
    dates = np.load(root / "date_index.npy")
    names = np.load(root / "isin_index.npy").tolist()
    counts = defaultdict(int)

    def check(label, actual, expected):
        np.testing.assert_array_equal(actual, expected, err_msg=label)
        counts[label] += np.asarray(expected).size

    def array(key):
        return np.load(root / manifest["arrays"][key]["path"], mmap_mode="r")

    check("dates", np.asarray(inputs.dates, dtype="datetime64[D]"), dates[indices])
    check("identities", inputs.security_ids, names)
    assert dates[indices[-1]] <= np.datetime64("2024-12-30")
    for key in ("active", "raw_close", "entry_fill_allowed", "target_scale_sigma"):
        check("accepted_" + key, getattr(inputs, key), array(key)[indices])
    check(
        "accepted_prior_reference",
        data.references,
        array("prior_reference_close")[indices],
    )
    assert not inputs.score_mask.any() and not data.valid.any()
    assert np.count_nonzero(inputs.scores) == 0

    assert sha256_file(Path(source["risk"]["path"])) == source["risk"]["sha256"]
    with np.load(source["risk"]["path"]) as z:
        risk = {k: z[k] for k in z.files}
    for actual, key in (
        (data.beta, "resolved_beta"),
        (data.diagonal, "diagonal"),
        (data.factor, "factor"),
        (inputs.hedge_beta, "raw_beta"),
        (inputs.hedge_beta_valid, "raw_valid"),
    ):
        check("saved_" + key, actual, risk[key][indices])
    check(
        "volatility",
        data.volatility,
        np.sqrt(data.diagonal + data.beta**2 * data.factor[:, None]),
    )

    # Independent backward selection replaces the producer's forward state machine.
    # Sorted reverse-time edges cannot traverse the later JSL episode twice.
    links = pl.read_parquet(
        root / manifest["tables"]["slow_history_links"]["path"]
    ).to_dicts()
    edges = sorted(links, key=lambda r: r["effective_index"], reverse=True)
    first, stop = plan["decision_rows"]
    fallbacks = 0
    for t in range(first, stop):
        past = np.arange(t, max(first - 1, t - 21), -1)
        route = np.broadcast_to(np.arange(len(names)), (len(past), len(names))).copy()
        for edge in edges:
            e = edge["effective_index"]
            if max(e, edge["known_index"]) <= t:
                mask = (past[:, None] < e) & (route == edge["successor_index"])
                route[mask] = edge["predecessor_index"]
        valid = risk["raw_valid"][past[:, None], route]
        available = valid.any(0)
        latest = valid.argmax(0)
        expected = np.ones(len(names))
        expected[available] = risk["raw_beta"][
            past[latest[available]], route[latest[available], np.flatnonzero(available)]
        ]
        check("independent_20_session_fallback", risk["resolved_beta"][t], expected)
        fallbacks += int((~risk["raw_valid"][t] & available).sum())

    # Only the newly introduced ancestor interaction needs another raw oracle.
    # Ordinary formula windows/current-endpoint exclusion already passed once.
    wealth, seen = array("shareholder_wealth_close"), array("shareholder_wealth_valid")
    bova_manifest = bound_json(
        dict(
            path=str(Path(plan["bova11"]["root"]) / "manifest.json"),
            sha256=plan["bova11"]["manifest_sha256"],
        )
    )
    # The cached hedge closes are already qualified against the original source.
    hedge = inputs.bova11_close
    history_records = []
    for edge in links:
        for offset in (0, 20, 60):
            t, name = edge["effective_index"] + offset, edge["successor_index"]
            if not indices[0] + 61 <= t <= indices[-1]:
                continue
            rows = np.arange(t - 61, t)
            ancestors = []
            for day in rows:
                ancestor, ceiling = name, t + 1
                while True:
                    options = [
                        e
                        for e in links
                        if e["successor_index"] == ancestor
                        and day < e["effective_index"] < ceiling
                        and e["effective_index"] <= t
                        and e["known_index"] <= t
                    ]
                    if not options:
                        break
                    e = max(options, key=lambda x: x["effective_index"])
                    ancestor, ceiling = e["predecessor_index"], e["effective_index"]
                ancestors.append(ancestor)
            p = wealth[rows, ancestors].astype(float)
            m = seen[rows, ancestors]
            h = hedge[rows - indices[0]]
            good = (
                m[1:]
                & m[:-1]
                & (p[1:] >= 0)
                & (p[:-1] > 0)
                & np.isfinite(h[1:])
                & np.isfinite(h[:-1])
                & (h[1:] > 0)
                & (h[:-1] > 0)
            )
            y, x = p[1:][good] / p[:-1][good] - 1, h[1:][good] / h[:-1][good] - 1
            valid = len(x) >= 40 and np.var(x) > 0
            assert valid == bool(risk["raw_valid"][t, name])
            error = 0.0
            if valid:
                slope = np.linalg.lstsq(
                    np.column_stack((np.ones(len(x)), x)), y, rcond=None
                )[0][1]
                expected = np.clip(0.67 * slope + 0.33, -1, 3)
                error = abs(expected - risk["raw_beta"][t, name])
                assert error < 1e-10, (t, name, error)
            history_records.append(
                dict(
                    date=str(dates[t]),
                    isin=names[name],
                    source_rows=rows.tolist(),
                    source_axes=ancestors,
                    pairs=int(good.sum()),
                    valid=bool(valid),
                    beta_error=float(error),
                )
            )
    write_json_atomic(out / "history_selection.json", history_records)

    terms = bound_json(plan["terms"])

    def local(day):
        return int(np.searchsorted(dates, np.datetime64(day))) - int(indices[0])

    fields = (
        "action_shares_per_prior_share",
        "action_cash_per_prior_share",
        "action_has_action",
        "action_successor_index",
        "action_payment_session",
        "action_session_resolved",
    )
    expected = {
        k: np.array(array(k)[indices], dtype=getattr(inputs, k).dtype) for k in fields
    }
    pay = expected["action_payment_session"]
    pay[pay >= 0] -= indices[0]
    for event in source["source_actions_represented_once"]:
        t, n = local(event["date"]), names.index(event["isin"])
        expected["action_has_action"][t, n] = False
        expected["action_successor_index"][t, n] = n
    for event in terms["identity_actions"]:
        t, n = local(event["effective_date"]), names.index(event["predecessor_isin"])
        for key, value in (
            ("action_shares_per_prior_share", 1),
            ("action_cash_per_prior_share", 0),
            ("action_has_action", True),
            ("action_successor_index", names.index(event["successor_isin"])),
            ("action_payment_session", -1),
        ):
            expected[key][t, n] = value
        expected["action_session_resolved"][t:, n] = True
    assert len(inputs.share_distributions) == len(terms["share_distributions"]) == 18
    for event, actual in zip(
        terms["share_distributions"], inputs.share_distributions, strict=True
    ):
        t, n = local(event["effective_date"]), names.index(event["isin"])
        end = (
            local(event["source_reopens_date"])
            if event.get("source_reopens_date")
            else len(indices)
        )
        expected["action_session_resolved"][max(0, t) : end, n] = True
        assert (
            actual.source_index == n
            and actual.effective_session == t
            and actual.available_session == local(event["available_date"])
        )
        assert actual.cash_per_prior_share == event["cash_per_prior_share"]
        for leg, loaded in zip(event["legs"], actual.legs, strict=True):
            assert loaded.successor_index == names.index(leg["successor_isin"])
            assert loaded.shares_per_prior_share == leg["shares_per_prior_share"]
            assert loaded.delivery_session == (
                None if leg["delivery_date"] is None else local(leg["delivery_date"])
            )
        counts["distribution_objects"] += 1
    for event in terms["scalar_actions"]:
        t, n = local(event["effective_date"]), names.index(event["isin"])
        for key, value in (
            ("action_shares_per_prior_share", event["shares_per_prior_share"]),
            ("action_cash_per_prior_share", event["gross_cash_per_prior_share"]),
            ("action_has_action", True),
            ("action_session_resolved", True),
            (
                "action_payment_session",
                -1
                if not event["gross_cash_per_prior_share"]
                else local(event["payment_date"]),
            ),
        ):
            expected[key][t, n] = value
    for event in terms["cash_cancellations"]:
        t, n = local(event["recognition_date"]), names.index(event["isin"])
        for key, value in (
            ("action_shares_per_prior_share", 0),
            ("action_cash_per_prior_share", event["cash_per_share"]),
            ("action_has_action", True),
            ("action_payment_session", local(event["payment_date"])),
        ):
            expected[key][t, n] = value
        expected["action_session_resolved"][local(event["coverage_start"]) :, n] = True
    for key in fields:
        check("composed_" + key, getattr(inputs, key), expected[key])
    del expected
    for name in ("BRALSCACNOR0", "BRENATACNOR0"):
        event = next(
            e for e in inputs.share_distributions if e.source_index == names.index(name)
        )
        if name == "BRALSCACNOR0":
            assert event.legs[0].delivery_session is None
        else:
            auction = event.legs[0].fractional_auction
            assert (
                auction.available_session
                is auction.cash_per_share
                is auction.payment_session
                is None
            )

    cash = bound_json(run["cash_calendar"])["panel"]
    assert sha256_file(Path(cash["path"])) == cash["sha256"]
    with np.load(cash["path"]) as z:
        cdi = z["cdi_returns"]
    scope = bound_json(source["cash_scope"])
    covered = np.isfinite(cdi[indices])
    check("corrected_cash", inputs.cdi_returns[covered], cdi[indices][covered])
    assert (
        dates[indices][~covered].astype(str).tolist()
        == scope["uncovered_unused_prelude"]
        == ["2016-07-18"]
    )
    prior_cash = cdi[indices - 1].copy()
    for legacy_row in np.flatnonzero(~covered):
        prior_cash[indices - 1 == indices[legacy_row]] = inputs.cdi_returns[legacy_row]
    check("prior_cash", data.prior_cdi, prior_cash)
    loans = bound_json(run["loan_source_panels"])["panels"]
    assert sha256_file(Path(loans["path"])) == loans["sha256"]
    with np.load(loans["path"]) as z:
        check(
            "loan_references",
            inputs.loan_reference_prices,
            z["loan_reference_prices"][indices],
        )
        check(
            "hedge_borrow",
            inputs.hedge_annual_borrow_rate,
            z["hedge_annual_borrow_rate"][indices],
        )
    design = bound_json(run["stage_c_plan"])
    fold_records = []
    for fold in design["folds"]:
        rows = windows(Path(design["prior_root"]), data, fold)["evaluation"]
        a, b = int(rows[0]), int(rows[-1]) + 1
        assert covered[a:b].all() and np.isfinite(cdi[indices[a:b] - 1]).all()
        arguments = ledger_arguments(data, a, b)
        check("actual_ledger_beta", arguments["hedge_beta"], data.beta[a:b])
        check(
            "actual_ledger_reference",
            arguments["initial_reference_price"],
            data.references[a],
        )
        check("actual_ledger_cdi", arguments["cdi_returns"], cdi[indices[a:b]])
        assert arguments["hedge_beta_valid"].all()
        fold_records.append(
            dict(
                fold=fold,
                first=str(inputs.dates[a]),
                last=str(inputs.dates[b - 1]),
                sessions=b - a,
            )
        )
    result = dict(
        passed=True,
        inputs=run["stage_c_refit_economics"],
        counts=dict(counts),
        backward_fallback_cells=sum(
            counts[k] for k in counts if k == "independent_20_session_fallback"
        ),
        inherited_or_own_stale_beta_cells=fallbacks,
        history_selection=binding(out / "history_selection.json"),
        bova_source_schema=bova_manifest["schema"],
        reused_formula_checks=source["formula_checks"],
        reused_original_cache_controls=source["old_control_cells"],
        folds=fold_records,
        no_new_book_or_fit=True,
        no_repeated_source_reducer_or_full_store_audit=True,
        seconds=perf_counter() - tick,
    )
    write_json_atomic(out / "report.json", result)
    run["stage_c_refit_economics_qualification"] = binding(out / "report.json")
    write_json_atomic(pointer, run)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
