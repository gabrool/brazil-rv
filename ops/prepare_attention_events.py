"""Bound the actually held 2019 transitions without changing model coordinates."""

from copy import copy, deepcopy
from dataclasses import fields, replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import pickle
import re
from time import perf_counter
from zoneinfo import ZoneInfo

import numpy as np

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.corporate_replay import apply_corporate_replay, load_corporate_replay
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.decision_clock import load_session_schedule

PROJECT = Path(__file__).resolve().parents[1]
PAGES = {
    "653816": [1, 2, 3],
    "659020": [1, 2],
    "659133": [1],
    "659958": [1, 2],
    "662680": [1, 2],
    "680451": [1],
    "673331": [1],
    "685074": [1],
    "680551": [1],
    "681261": [1],
}


def main():
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    base = bound_json(run["scaling_expanded_evaluation_plan"])
    root = Path(run["scaling_expanded_evaluation_plan"]["path"]).parent
    sources, out = root / "event_sources", root / "source_replays"
    out.mkdir(exist_ok=False)
    (out / "executed_preparation.py").write_bytes(Path(__file__).read_bytes())
    store = Path(base["store"]["root"])
    manifest = bound_json(
        dict(path=str(store / "manifest.json"), sha256=base["store"]["manifest_sha256"])
    )
    schedule = load_session_schedule(
        Path(
            next(
                s["path"]
                for s in manifest["sources"]
                if Path(s.get("path", "")).name
                == "b3_session_schedule_reconstructed_v1.csv"
            )
        )
    )
    originals = {}
    for protocol, pages in PAGES.items():
        receipt = bound_json(binding(sources / f"{protocol}_receipt.json"))
        stamps = {
            re.search(r"\d{2}/\d{2}/\d{4} \d{2}:\d{2}", r["receipt"])[0]
            for r in receipt["issuer_rows"]
        }
        assert len(stamps) == 1
        known = (
            datetime.strptime(stamps.pop(), "%d/%m/%Y %H:%M").replace(
                tzinfo=ZoneInfo("America/Sao_Paulo")
            )
            + timedelta(minutes=1)
        ).astimezone(timezone.utc)
        originals[protocol] = dict(
            pdf=receipt["pdf"],
            receipt=binding(sources / f"{protocol}_receipt.json"),
            available_at=known.isoformat(),
            available_date=str(
                next(s.trade_date for s in schedule if s.decision_at >= known)
            ),
            visually_qualified_pages=[
                dict(
                    page=p, image=binding(sources / "rendered" / f"{protocol}_p{p}.png")
                )
                for p in pages
            ],
        )
    parent_terms = bound_json(base["terms"])
    terms = dict(
        calendar=parent_terms["calendar"],
        sources=[v["pdf"] for v in originals.values()],
        identity_actions=[],
        scalar_actions=[],
        share_distributions=[],
        cash_cancellations=[],
        loan_cash_settlements=[],
    )
    terms["identity_actions"].append(
        dict(
            predecessor_isin="BRQGEPACNOR8",
            successor_isin="BRENATACNOR0",
            effective_date="2019-04-24",
            available_date=originals["681261"]["available_date"],
        )
    )
    for isin, effect, protocol, q, cash, payment, delivery in (
        ("BRQGEPACNOR8", "2019-04-22", "680551", 1, 1.90684828735, "2019-05-07", None),
        ("BRGUARACNOR4", "2019-05-02", "685074", 8, 0, None, "2019-05-07"),
    ):
        terms["scalar_actions"].append(
            dict(
                isin=isin,
                effective_date=effect,
                available_date=originals[protocol]["available_date"],
                shares_per_prior_share=q,
                gross_cash_per_prior_share=cash,
                payment_date=payment,
                bonus_delivery_date=delivery,
                withholding_rate=0,
                short_cash_fraction=1,
            )
        )
    for isin, successor, effect, protocol, ratio, cash, payment, delivery, auction in (
        (
            "BRFIBRACNOR9",
            "BRSUZBACNOR0",
            "2019-01-04",
            "659133",
            0.4613,
            50.12,
            "2019-01-14",
            "2019-01-09",
            dict(
                available_date=originals["673331"]["available_date"],
                cash_per_share=43.85864640290,
                payment_date="2019-03-29",
                provision_loan_fractions=False,
            ),
        ),
        (
            "BRGUARACNPR1",
            "BRGUARACNOR4",
            "2019-02-07",
            "662680",
            1,
            0,
            None,
            "2019-02-07",
            None,
        ),
    ):
        terms["share_distributions"].append(
            dict(
                isin=isin,
                effective_date=effect,
                available_date=originals[protocol]["available_date"],
                cash_per_prior_share=cash,
                payment_date=payment,
                carry_source_value=False,
                legs=[
                    dict(
                        successor_isin=successor,
                        shares_per_prior_share=ratio,
                        delivery_date=delivery,
                        fractional_auction=auction,
                        loan_principal_fraction=1.0,
                    )
                ],
            )
        )
    terms["share_distributions"][0]["cash_values"] = [
        dict(
            available_date=originals["659958"]["available_date"],
            cash_per_prior_share=50.20,
        )
    ]
    terms["hypotheses"] = [
        "Fibria credit at January8 market close is usable from January9 decision in this discrete-session primary. Separate January8 arrival and January4 when-issued owned-disposal hypotheses bound the timing. Actual source date is not relabelled January9.",
        "GUAR PN/ON conversion uses effect-day custody as a hypothesis; separate one/two-session later delivery. May2 split is exactly8, with May6 closing credit represented as May7 decision usability; May6 is the separate arrival endpoint.",
        "Fibria cash retains last-announced50.12 until final50.20 is known January10, recognized after current intentions; no future amount/rate backdating. Separate causal-DI valuation may be tested before the final announcement. Original cash principal pays once January14.",
        "Corporate pre-income-tax BRL returns; no ordinary-CNPJ income-tax exemption claimed. Domestic shareholder terms and signed gross lender compensation remain separate from unobserved event-specific lender instructions. Continuous loan shares/conversion at selected custody is primary; provisioned fractions separate when exposed.",
        "No QGEP/ENAT loan-source alias or new locate; only already held cohorts follow legal identity. No source quote, shareholder history pooling, source census, accepted store change or fit rerun.",
    ]
    write_json_atomic(out / "incremental_terms.json", terms)
    write_json_atomic(
        out / "source_admission.json",
        dict(
            originals=originals,
            events=terms,
            exposure=run["scaling_expanded_qualification"],
            cash_revision_plan=binding(sources / "cash_revision_plan.json"),
            limitations="Purpose-limited account admission. The two disproved inferred gross factors and QGEP identity also imply model-data dependencies; these are not silently repaired in the fixed-store comparison. No source-completeness claim.",
        ),
    )
    save_source_overlay(
        base, terms, parent_terms, out, run, ["F3"], "scaling_expanded_source_plan"
    )


def save_source_overlay(base, terms, parent_terms, out, run, folds, pointer_key):
    """Share the unchanged shallow-cache assembly across exposed event groups."""
    started = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    original = bound_json(base["inputs"])
    with Path(original["cache"]["path"]).open("rb") as f:
        data = pickle.load(f)
    old = data.inputs
    provenance = dict(old.source_artifact_hashes)
    previous = provenance.pop("corporate_replay")
    provenance["corporate_replay_parent"] = previous
    loaded, calendar = load_corporate_replay(
        str(out / "incremental_terms.json"),
        binding(out / "incremental_terms.json")["sha256"],
    )
    amended = apply_corporate_replay(
        replace(old, source_artifact_hashes=provenance),
        loaded,
        calendar,
        binding(out / "incremental_terms.json")["sha256"],
    )
    data = copy(data)
    data.inputs = amended
    unchanged, changes = [], {}
    for field in fields(old):
        a, b = getattr(old, field.name), getattr(amended, field.name)
        if isinstance(a, np.ndarray):
            if a is b:
                unchanged.append(field.name)
            else:
                changes[field.name] = int(
                    np.count_nonzero(~(np.equal(a, b) | (np.isnan(a) & np.isnan(b))))
                )
    # All non-account coordinates remain the original objects, including risks.
    for field in ("raw_close", "active", "scores", "hedge_beta"):
        assert getattr(old, field) is getattr(amended, field)
    cache = out / "policy_data.pkl"
    with cache.open("xb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    combined = deepcopy(parent_terms)
    for field in ("identity_actions", "scalar_actions", "share_distributions"):
        combined.setdefault(field, []).extend(terms[field])
    combined["sources"].extend(terms["sources"])
    combined["hypotheses"] = terms["hypotheses"]
    write_json_atomic(out / "terms.json", combined)
    write_json_atomic(
        out / "inputs.json",
        dict(
            original,
            cache=binding(cache),
            parent=base["inputs"],
            account_terms=binding(out / "incremental_terms.json"),
            unchanged_objects=unchanged,
            changed_cells=changes,
            scope="Shallow PolicyData copy; model forecasts/static policy/risk/eligibility/history preserved. Only separately attributed sourced account events.",
        ),
    )
    plan = dict(
        base,
        folds=folds,
        planned_books=12 * len(folds),
        inputs=binding(out / "inputs.json"),
        terms=binding(out / "terms.json"),
        scope="Separately attributed source-only books compared to saved matching controls; existing eight-period fits stay on fixed accepted coordinates.",
        baseline=run["scaling_expanded_evaluation_plan"],
        source_admission=binding(out / "source_admission.json"),
    )
    write_json_atomic(out / "plan.json", plan)
    write_json_atomic(
        out / "preparation.json",
        dict(
            seconds=perf_counter() - started,
            changes=changes,
            unchanged=unchanged,
            plan=binding(out / "plan.json"),
            source_admission=binding(out / "source_admission.json"),
        ),
    )
    run = json.loads(pointer.read_text())
    run[pointer_key] = binding(out / "plan.json")
    write_json_atomic(pointer, run)
    print(
        json.dumps(dict(changes=changes, seconds=perf_counter() - started)), flush=True
    )


if __name__ == "__main__":
    main()
