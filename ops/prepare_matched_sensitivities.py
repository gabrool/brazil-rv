"""Freeze one-factor model-book sensitivities, retaining unexposed skips."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.decision_clock import load_session_schedule

PROJECT = Path(__file__).resolve().parents[1]


def main():
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    root = Path(run["stage_c_root"])
    primary_plan = bound_json(run["stage_c_event_replay_plan"])
    proof = binding(root / "event_replays/qualified/report.json")
    assert bound_json(proof)["passed"]
    corrected = bound_json(binding(root / "event_replays/replays.json"))
    records = [*corrected["completed"], *primary_plan["reused"]]
    assert len(records) == len({r["key"] for r in records}) == 160
    primary_index = root / "event_replays/combined_primary.json"
    write_json_atomic(
        primary_index,
        dict(
            status="qualified_account_source_books_data_refits_and_sensitivities_pending",
            completed=records,
            corrected=100,
            reused=60,
            qualification=proof,
        ),
    )
    out = root / "sensitivities"
    out.mkdir(exist_ok=False)
    (out / "executed_preparation.py").write_bytes(Path(__file__).read_bytes())
    accepted = bound_json(run["economic_refit_inputs"])["store"]
    store = Path(accepted["root"])
    manifest = bound_json(
        dict(path=str(store / "manifest.json"), sha256=accepted["manifest_sha256"])
    )
    schedule = next(
        s
        for s in manifest["sources"]
        if Path(s.get("path", "")).name == "b3_session_schedule_reconstructed_v1.csv"
    )
    calendar = np.array(
        [s.trade_date for s in load_session_schedule(Path(schedule["path"]))],
        dtype="datetime64[D]",
    )
    names = np.load(store / "isin_index.npy").tolist()
    terms = bound_json(run["stage_c_event_candidate_terms"])
    account = bound_json(primary_plan["economic_account"])
    variants = [
        ("shortfall0", dict(execution_shortfall_bps=0), "trades"),
        ("shortfall2", dict(execution_shortfall_bps=2), "trades"),
        ("brokerhalf", dict(execution_brokerage_bps=0.5), "trades"),
        ("brokerone", dict(execution_brokerage_bps=1), "trades"),
        ("auction", dict(spot_execution_phase="auction"), "trades"),
        ("proceeds95", dict(short_proceeds_remuneration=0.95), "proceeds"),
        ("debit50", dict(annual_debit_spread=0.005), "debit"),
        ("debit100", dict(annual_debit_spread=0.01), "debit"),
        (
            "spot_invoice",
            dict(spot_invoice_convention="security_day_6dp_cent"),
            "trades",
        ),
        (
            "contract_nearest",
            dict(loan_invoice_convention="contract_nearest"),
            "loan_payment",
        ),
        (
            "contract_down",
            dict(loan_invoice_convention="contract_down"),
            "loan_payment",
        ),
        ("contract_up", dict(loan_invoice_convention="contract_up"), "loan_payment"),
        (
            "security_day_nearest",
            dict(loan_invoice_convention="security_day_nearest"),
            "loan_payment",
        ),
        ("minimum_pro_rata", dict(loan_minimum_allocation="pro_rata"), "old_minimum"),
        ("term30", dict(loan_term_sessions=30), "loans"),
        ("term126", dict(loan_term_sessions=126), "loans"),
        ("denied_renewal", dict(approve_loan_renewals=False), "loans"),
        ("custody_economic_long", dict(custody_base="economic_long"), "custody"),
        ("custody_exclude_rights", dict(custody_claim_fraction=0), "assessed_rights"),
        (
            "missing_tariff_lower",
            dict(unrecovered_spot_trading_bps=0.2),
            "missing_tariff",
        ),
    ]
    for lag in (3, 10):
        assessments = []
        for item in account["primary_config"]["custody_assessments"]:
            i = int(np.searchsorted(calendar, np.datetime64(item["date"])))
            assert str(calendar[i]) == item["date"]
            assessments.append(
                dict(date=item["date"], payment_date=str(calendar[i + lag]))
            )
        variants.append(
            (f"custody_payment{lag}", dict(custody_assessments=assessments), "custody")
        )
    groups = {
        "JSL": ["BRJSLGACNOR2"],
        "VIVT": ["BRVIVTACNPR7"],
        "MODL": ["BRMODLCDAM13", "BRMODLACNPR9"],
        "VVAR": ["BRVVARCDAM10", "BRVVARACNPR8"],
        "SULA": ["BRSULACDAM12", "BRSULAACNOR2", "BRSULAACNPR9"],
    }
    phases, included, cases, skips = {}, [], [], []
    for record in records:
        if not record["key"].startswith("sources/") or not record["key"].endswith(
            "/ensemble"
        ):
            continue
        book = bound_json(record["book"])
        _, capital, arm, fold, member = record["key"].split("/")
        capital = int(capital)
        folder = Path(record["book"]["path"]).parent
        with np.load(folder / "account.npz") as z:
            a = {
                k: z[k]
                for k in (
                    "signed_shares",
                    "loan_payment",
                    "loan_outstanding_principal",
                    "custody_fee",
                    "custody_claim_base",
                )
            }
        daily, days = book["daily"], book["state_dates"]
        cfg = book["provenance"]["config"]
        loans = bool(np.any(a["loan_outstanding_principal"] > 0))
        exposure = dict(
            trades=bool(np.any(np.array(daily["turnover"]) > 0)),
            proceeds=bool(np.any(np.array(daily["short_proceeds_income_bps"]) != 0)),
            debit=bool(np.any(np.array(daily["debit_financing_bps"]) != 0)),
            loan_payment=bool(np.any(a["loan_payment"] != 0)),
            loans=loans,
            old_minimum=loans and days[0] < "2020-10-26",
            custody=bool(np.any(a["custody_fee"] != 0)),
            assessed_rights=bool(
                np.any((a["custody_fee"] > 0) & (a["custody_claim_base"] > 0))
            ),
            missing_tariff=any(
                d < "2021-02-02"
                and not any(
                    t["valid_from"] <= d <= t["valid_to"]
                    for t in cfg["monthly_spot_tariffs"]
                )
                for d in days
            ),
        )

        def add(label, changed, exposed, reason):
            phase = dict(name=label, capital=capital, members="ensemble", **changed)
            phases[label, capital] = phase
            key = f"{label}/{capital}/{arm}/{fold}/{member}"
            entry = dict(key=key, baseline=record, exposure=reason)
            if exposed:
                included.append(key)
                cases.append(entry)
            else:
                skips.append(entry)

        for label, config, criterion in variants:
            add(label, dict(config=config), exposure[criterion], criterion)
        for deadline in (2, 4):
            add(
                f"recall{deadline}",
                dict(recall_deadline=deadline),
                loans,
                "Universal pre-decision notice at local session50, same frozen timing as qualified lifecycle engineering, includes hedge; no replacement loans until deadline/returns. Hypothesis, not observed notice.",
            )
        for group, isins in groups.items():
            exposed = False
            for e in terms["share_distributions"]:
                if e["isin"] in isins and e["effective_date"] in days:
                    t = days.index(e["effective_date"])
                    exposed |= (
                        t > 0 and a["signed_shares"][t - 1, names.index(e["isin"])] != 0
                    )
            add(
                "delivery2_" + group,
                dict(delivery_delays={s: 2 for s in isins}),
                exposed,
                "Actually held source at effect; shift only selected event delivery by two accepted sessions, keep sourced economic/knowledge/cash dates.",
            )
        for source in (
            "BRLCAMACNOR3",
            "BRSULACDAM12",
            "BRSULAACNOR2",
            "BRSULAACNPR9",
            "BRSOMAACNOR3",
            "BRENATACNOR0",
        ):
            e = next(e for e in terms["share_distributions"] if e["isin"] == source)
            t = days.index(e["effective_date"]) if e["effective_date"] in days else 0
            exposed = t > 0 and a["signed_shares"][t - 1, names.index(source)] < 0
            add(
                "loan_fraction_" + source,
                dict(
                    loan_fraction_conventions={
                        source: not e["legs"][0]["fractional_auction"].get(
                            "provision_loan_fractions", False
                        )
                    }
                ),
                exposed,
                "Negative net source at effect; switch between per-original-loan fractional provision and continuous quantities. SOMA primary is provisioned; other selected events primary continuous. No hypothetical forced exposure.",
            )
    plan = {
        **deepcopy(primary_plan),
        "output_root": str(out),
        "primary_books": binding(primary_index),
        "phases": list(phases.values()),
        "included_keys": included,
        "planned_new_books": len(included),
        "cases": cases,
        "skipped": skips,
        "reused": [],
        "schedule": schedule,
        "status": "frozen_before_sensitivity_outcomes",
        "contrast": "One factor at a time on the48 source-corrected ensemble books: four original models/folds at R10m/R1m/R5m. Reuse every primary. No new neural scoring or seed fit; retain static old coordinates, allocator/calibration and original forecasts. Outcomes are total adaptive paths. Grouping attribution is security-day-nearest minus contract-nearest, not versus unrounded. No combined worst-case/interior-extrema inference.",
        "conditional_skips": "No screen fold intersects January25 2017, Natura2019, BRML/DMMO/Copel2023. SULA fraction auction/payment first known2023 lies beyond F10. No actual Cielo loan cash payment in corrected/reused books; cent scenario stays conditional. Existing unexposed skips do not establish general bounds. Source-identity/loan histories and corrected data refits remain separate.",
        "preserved_adaptive_uncertainty": "All prior V32-V41 measured fixed-minimum two-account path uncertainties remain, including0.29492822018219156bp at R1m. These are numerical total-path differences, not daily alpha; tiny contrasts are unresolved below the relevant bound.",
    }
    write_json_atomic(out / "plan.json", plan)
    run["stage_c_event_replays"] = binding(primary_index)
    run["stage_c_event_qualification"] = proof
    run["stage_c_sensitivity_plan"] = binding(out / "plan.json")
    write_json_atomic(pointer, run)
    print(
        json.dumps(
            dict(
                books=len(included),
                skipped=len(skips),
                phases=len(phases),
                primary_books=160,
                ensemble_primaries=48,
            )
        )
    )


if __name__ == "__main__":
    main()
