"""Record completed composition evidence and precise remaining closeout work."""

from collections import Counter
import json
from pathlib import Path

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]


def main():
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text(encoding="utf8"))
    base = Path(run["root"])
    folder = base / "event_composition"
    source = base / "calendar_issuer_notice"
    receipt = bound_json(binding(source / "528672_receipt.json"))
    text = (source / "528672.txt").read_text(encoding="utf8")
    assert (
        "30 de dezembro de" in text
        and "não haverá negociação e liquidação nos mercados da BM&FBOVESPA" in text
    )
    calendar = dict(
        status="dated_issuer_original_corrobates_dec30_closure_jan25_remains",
        date="2016-12-30",
        disposition="closed primary supported by the original issuer erratum plus dated B3 nontrading calendar; not an observed equity settlement receipt or a new B3 circular",
        issuer="Elektro Eletricidade e Serviços S.A., CNPJ02.328.280/0001-97; current RAD label Elektro Redes is only an archive label",
        original=receipt["pdf"],
        receipt=binding(source / "528672_receipt.json"),
        selected_index=binding(source / "index.json"),
        text=binding(source / "528672.txt"),
        visual_page=binding(source / "528672_page1.png"),
        pages_visually_qualified=[1],
        source_receipt="2016-08-19 17:51 America/Sao_Paulo; minute-end17:52; notice refers to prior Aug18 newspaper publication, original erratum has no separate printed signature date",
        selected_lead="The two original Elektro August2016 notices were selected from existing2016 RAD responses; only replacement528672 downloaded. Prior B3 calendar/manual search and unrelated source censuses not repeated.",
        earlier_calendar_evidence=run["closeout_calendar_sources"],
        migration_evidence=run["clearing_calendar_resolution"],
        money_only_dates_supported=22,
        total_money_only_dates=23,
        remaining_date="2017-01-25",
        arrays_changed=False,
        rent_scope="No new loan-rent convention follows from cash-market closure. Existing B3-session rent hypothesis retained; January25 cash/physical delivery and rent hypotheses must remain distinct.",
        search_limit="No disclosure completeness/revision-completeness claim. Narrow web exact-date queries yielded no additional admitted evidence; original recovered via existing RAD receipt, not secondary transcription.",
    )
    write_json_atomic(source / "qualification.json", calendar)
    (source / "executed_qualification.py").write_bytes(Path(__file__).read_bytes())
    run["december_clearing_resolution"] = binding(source / "qualification.json")
    attempts = dict(
        production_or_numerical_failure=False,
        composition="First invocation passed,2.2790655s;1,693,560 accepted control cells; no books, neural inference or store modification.",
        consumer="First invocation passed,10.3649059s;33new crossing-date views/full933/full60,4,310,460 target/mask sample and collator cells. No repeated earlier consumer window.",
        identity_scope="First invocation passed,.0708237s;700bounded predecessor/successor rows,seven fields. Candidate identity implications only, no links/eligibility/history or loan aliases changed.",
        initial_source_retrieval=dict(
            code=binding(source / "initial_failed_528672.py"),
            viewer=binding(source / "initial_failed_528672_viewer.html"),
            command="uv run --project research --no-sync python ops/recover_calendar_issuer_notice.py --protocol 528672",
            failure="Wrong ID query and embedded-base64 assumption returned viewer HTML but no original PDF. ValueError: No embedded original PDF in saved viewer. Exact failed code/viewer retained; this prose/command record is reconstructed from the tool transcript, not a separately retained original stdout file.",
        ),
        qualified_source_retrieval="Reused two-row index; corrected existing public NumeroProtocoloEntrega/ExibirPDF route after explicit captcha-disabled viewer. One original PDF, .5576635s; no access control bypass or repeat source census.",
        read_only_attempts="Initial tool output exceeded context; bounded rereads followed. rg used nonexistent producer paths and a Windows wildcard once; no output artifacts changed. Descriptions reconstructed, not exact shell transcripts.",
        formatting="Ruff passed four new producers/qualifiers; no research implementation change or new mechanical test matrix required.",
        rendering="Poppler emitted optional-font warnings; the single page is fully legible and visually checked.",
    )
    write_json_atomic(folder / "attempts.json", attempts)
    audit = bound_json(run["event_source_composition"])
    qualified = bound_json(run["event_composition_qualification"])
    oracles = bound_json(audit["endpoint_oracles"])
    by_name = Counter(x["isin"] for x in oracles if x["valid"])
    progress = dict(
        status="partial_corporate_calendar_source_block_no_stage_a_acceptance",
        composition=run["event_source_composition"],
        qualification=run["event_composition_qualification"],
        primary_terms=run["composed_primary_event_terms"],
        identity_scope=run["succession_identity_dependency_scope"],
        december_resolution=run["december_clearing_resolution"],
        attempts=binding(folder / "attempts.json"),
        new_valid_endpoints=dict(by_name),
        neutral_numeric_changes=qualified["virtual_primary_numeric_changes"],
        target_data_status="Sparse source-only attribution qualified on accepted Natura risks; not a new accepted full store. Recompose with final repaired risks/history if surviving-company rename propagation changes them.",
        parent_store=audit["parent"],
        all_accepted_stores_and_old_fits_immutable=True,
        reused_proofs="V31-V41, historical-cost/corporate and invoice blocks unchanged; no new account books, repeated source scan, GPU or corrected model outcome.",
        remaining=[
            "January25 2017 old-equity clearing disposition or distinct cash/delivery and rent hypotheses",
            "Source-admit and propagate SSBR->ALSO and ARZZ->AZZA surviving-company identities with original clocks; preserve every eligible name/full60 and source assignments",
            "Compose the72new gross endpoints/entry barriers with final data risks and qualify a new complete refit store without overwriting either accepted store",
            "Integrate primary event/cost/cash/loan contract and exposure-conditional Cielo/daytrade branches for StageA admission",
            "Registered C then conditional D",
        ],
        prior_recovery_dependencies={
            k: run[k]
            for k in (
                "spot_invoice_recovery",
                "historical_cost_recovery",
                "enat_settlement_recovery",
                "natura_store_recovery",
                "held_event_source_recovery",
            )
        },
    )
    write_json_atomic(folder / "progress.json", progress)
    run["event_composition_progress"] = binding(folder / "progress.json")
    report = PROJECT / "docs/v2_EVENT_SOURCE_COMPOSITION.md"
    report.write_text(
        """# Event/source composition and remaining data scope

2026-09-20. This records progress inside the existing corporate/calendar/source
delivery block. It is not a new Stage A acceptance or a completed delivery block.

The composed primary table reuses all seven qualified distributions, two Natura
scalar settlements and Cielo shareholder/loan cash. Actual loader comparisons
confirm identical numeric terms and arrays; only provenance and stale limitation
prose change. Existing evidence remains authoritative. No historical book was
replayed, no new loan/issuer alias was added, and the old corporate_replay pointer
remains unchanged pending integrated admission. ALSC whole claims remain locked;
its recovered auction gross amount is not an invented net receipt. ENAT's unknown
auction remains null, with signed residual claims. All earlier one-factor terms
and measured adaptive uncertainty remain dependencies.

The separate sparse model-data attribution covers NATU, ALSC, SOMA and ENAT economic
exchanges on the accepted Natura risk coordinates. Thirty-three crossing dates
reconstruct1,693,560 accepted control cells exactly. Independent Decimal arithmetic
checks84 source endpoints:21 valid each for NATU/ALSC/SOMA and9 for ENAT. Twelve ENAT
endpoints at/after delivery remain unsupported under the unknown-auction contract.
There are72 new valid gross endpoints and no losses. Physical rank changes10,234
cells, normalized residual9,424, shareholder rank10,228 and price rank10,225;
these counts overlap. The actual virtual neutral target changes10,205 values and
gains72 supported cells, with no losses. Independent SVD arithmetic and the actual
dataset/collator agree on4,310,460 target/mask cells over33full933/full60 views;
all33 revoked endpoint windows expose no target and feature reads stay causal.
These are target/slow-history audit views, not a repeated native/auxiliary audit or
an accepted complete model store. Producer2.2790655s, consumer10.3649059s, first-pass.

All24 post-effect source eligibility cells stay eligible;2,806 new-source entry
permissions close on max(effect,knowledge), including those24. All already lack
observed source prices. The amendment changes no OHLC, wealth, risk, eligibility,
history, native or auxiliary predictor. Exact successor closing entitlements stay
audit-only; ENAT post-credit lot-dependent unit marks are omitted. Account opening
continuity values never become observed endpoint prices. The sparse amendment
must be composed once with final repaired risks before a new refit-store acceptance.

The specific Elektro erratum lead is now recovered as original CVM528672, received
August19 2016 17:51 local, SHA34d65a7ff055903139c02df38d11cf1445fff19692e58da184d84bb5664b8e20.
Its single page was visually checked: the issuer moves the last dividend payment
from December30 to29 because BM&FBOVESPA markets will have no trading or settlement
on December30. Combined with the dated B3 calendar, this supports the existing
closed-day treatment. This is a dated issuer-original corroboration, not a new B3
circular or an observed client settlement receipt.22of23 monetary-only dates now
have supported dispositions without calendar-array changes. January25 2017 remains
ambiguous. No loan-rent clock change is inferred from delivery closure. The existing
two-row August2016 issuer selection supplied this original; no B3 original retrieval or
source census was repeated; narrow web queries added no admitted evidence. The failed initial viewer request and exact recipe are
preserved; the qualified original retrieval took.5576635s. Publication/revision
completeness remains unknown.

The remaining succession-data scope is material. Existing issuer originals703585
and1264889 identify the surviving companies' renames, distinct from the acquired
ALSC/SOMA exchange ratios. The accepted store has no SSBR->ALSO or ARZZ->AZZA links.
BRSSBRACNOR1 last prints August5 2019 and BRALSOACNOR5 first prints August6;
BRARZZACNOR3 last prints July31 2024 and BRAZZAACNOR9 first prints August1.
Both successor axes have zero eligible dates in their first60 sessions, first
becoming active October29 2019/October24 2024. This inventory does not yet claim
that all60 will qualify under the unchanged thresholds. The bounded700-row,
seven-field inspection and assignment metadata took.0708237s and changed nothing.
Original source clocks/typed identity, then bounded affected history/risk/native/
auxiliary dependencies need admission. B3's acquired-company index-history rule
is not permission to pool issuer histories or create loan aliases. Reuse the
existing propagation algorithms and controls, rather than rebuilding raw sources.

Both accepted full stores and all old fits remain immutable. Canonical refit inputs
still select the accepted Natura store. C/D are unstarted and no corrected model
profitability or new neural forward exists. Recovery binds the new sparse outputs,
views, source original and exact recipes while depending on prior immutable stores.
""",
        encoding="utf8",
    )
    run["event_composition_report"] = binding(report)
    write_json_atomic(pointer, run)
    print(
        json.dumps(
            {
                "progress": run["event_composition_progress"],
                "calendar": run["december_clearing_resolution"],
                "report": run["event_composition_report"],
            }
        )
    )


if __name__ == "__main__":
    main()
