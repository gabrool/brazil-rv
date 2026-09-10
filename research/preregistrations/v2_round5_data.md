# Round 5 data registration

Status: registered before any Round-5 derived build, replay or fit. CPU only. The current source commit is 97635435aed191725cb1f32ec8b29981e925f9ca; the supplied d6679e1 reference differs only by the subsequent research-proposal document. This round adopts the supplied data-round scope, not that proposal's unaccepted experiments. The accompanying JSON binds immutable inputs and planned families. Acquisition probes and source-semantic audits are permitted before family admission; no source is assumed usable merely because it is listed below.

## Resolutions of source assumptions and internal conflicts

These resolutions implement the user's instruction to avoid both leakage and unnecessary delay, using the user's standing authority to choose the recommended technically correct interpretation.

- Market information is available at its actual fixing/print time, not the later free archive upload. This requires evidence that the historical field corresponds to that contemporaneously observable measurement. Assessed prices, adjusted closes, revised rates, daily aggregates and continuous futures must not silently be relabelled immutable prints. For uncertain measurement/publication semantics, retain an explicit unavailable result instead of a guessed lag.
- For minute-resolution timestamps use the upper bound of the minute (+1 minute), then the first decision at or after availability. Thus a source stamped 15:44 is available at 15:45; a strict before-15:44 condition would add an unnecessary day. Date-only financial receipts remain next-session when intraday availability cannot be recovered. Document every fallback and never double-lag an already available date.
- Latest-vintage or retrospectively classified observations are retained in audited source archives, labeled separately and excluded from the verified point-in-time consumer family unless historical validity can be demonstrated. A guessed extra lag does not repair revised values. Revision shares are measured where versions exist; where not identifiable, report unknown, not an invented estimate. Only evidence-backed revision bounds support a sensitivity.
- FCA ticker/issuer records require a dated COTAHIST ISIN join if they do not themselves contain ISIN. Both knowledge date and effective identity bounds apply. Total issuer shares are not free float; no utilization/free-float feature is fabricated. Issuer capitalization requires appropriately valued share classes and unit ratios, not total shares times an arbitrary class price. Unsupported features remain masked with an explicit reason.
- Store construction preserves all existing slow/intraday/native-fast, labels, axes, masks, raw economic inputs and unrelated arrays byte-for-byte. The enumerated Round-5 sidecar families may be replaced/extended in the new root, including previously empty families and backfilled lending. Existing odd-lot fields remain exact. Old stores and their field definitions remain immutable for historical runs; S0 reuse is justified only by its exact consumed input identity.
- Hedge repair and observed lending rates live in fresh economic sidecars and ledger-only replays. Source predictions, labels, populations and non-ledger readouts stay fixed. Borrow-balance model inputs and the observed-rate economic replacement are separate treatments. The BOVA-only, lending-only and joint impacts are reported separately where feasible, with causal affected-fold attribution. No failed historical panel is relabelled passed.
- Magnitudes are stored in economically meaningful unnormalized units with masks; fit-dependent clipping is fitted by each training consumer using only its fit data and recorded in its fit artifact. One globally clipped store cannot represent fourteen different fit-window thresholds without leakage.
- Group-B forward capture is explicitly raw/quarantined and outside the development store. Historical acquisitions and every fitting/scoring/replay consumer remain bounded through 2024-12-30; do not request annual 2025/2026 historical payloads. Metadata/source documentation are not held-out outcomes. The explicit forward-capture request is not permission to evaluate a held-out period.
- Screens are the expressly requested CPU GBDT information readouts (14 folds, five seeds), not neural fits. Zero-valid families receive an unavailable result, not a meaningless fitted screen. Their results cannot select a network arm or reject a null family. Group-B produces timing tables and the specifically requested forward capture, no model arrays.
- Store capacity is not promised to eliminate every possible future data rebuild. This round creates one complete admitted-family development root after acquisition/semantics are resolved, measuring peak RSS below 8 GiB. No repeated full-source/M1 rebuild is needed where copying verified unchanged arrays expresses the identical contract.

## Supplied scope

The following is the supplied document verbatim, retained as requested scope and source assertions to verify. The resolutions above govern where assertions conflict with verified source semantics. The original byte SHA-256 is recorded in the protocol JSON.

# v2 Round 5 — the data round: a wide point-in-time dataset, built once and correctly

Audience: the coding model (repo `main` at `d6679e1`). Round 5 is CPU-only and contains
no neural fit, no arm, no blend and no 2025 byte. Its product is a rebuilt development
store carrying every candidate information family as its own masked sidecar with a
verified availability rule, plus the fixes carried from the Round-4 readout, so that
Round 6 and later rounds are pure arms against S0 on a store that never needs to be
rebuilt again for data reasons. Register as `v2_round5_data.md` before any build; freeze
against the repaired store (`db4f751d…`), the Round-4 roots and this document.

The two reviews this synthesises agree on what the model lacks (information it does not
have) and disagree on how to get at it. Blending S0 with GBDT or momentum (the coding
model's E2) is not adopted: a combination with a control is a diagnostic, and the
program's object is the model. Cost-aware trade-versus-hold calibration on frozen scores
(E1) and the cost-aware training objective belong to a portfolio round after the
continuous-book evaluator exists. The fine-tuning-multiplier test (E4), the magnitude
channels (E6), the simple-model comparator (E8) and attention (E8) are Round-6 arms and
are listed in §7 so their data prerequisites are built now. Everything else below is
data work, and the rule that governs all of it is stated first.

---

## 0. The availability rule: no leak, no nerf

Every observation enters the feature set at the first 15:45 decision at or after the
moment the market could have known it, and not one session later. Concretely:

- **Market prices** (FX, rates, oil, iron ore, VIX, US closes, ADRs): the information
  exists the instant it prints. The free archive's posting lag (EIA posts daily spot
  prices weekly; SGS posts yesterday's rate this morning) is an artefact of the archive,
  not of the market, and is *not* applied. The rule is: a series whose day-t value is
  fixed before 15:45 BRT on t (Asian sessions, PTAX at ~13:10, the previous US close) is
  available on t; a series whose day-t value is fixed after 15:45 (the B3 close, the US
  close, DI settlements) is available on t+1 as its t value. Daily features therefore
  use t−1 closes for B3/US series and day-t settlements for Asia and PTAX, and the
  manifest says which. Values that are never revised need no vintage handling.
- **Published statistics and filings** (lending balances and rates, open interest,
  financial statements, material facts, index previews, fund reports, foreign flow):
  availability is the publication time, taken from the source's own timestamp where one
  exists (RAD/ENET to the minute; CVM `DT_RECEB` to the day), otherwise from the
  documented publication schedule verified against file evidence. A publication before
  15:45 on t is available on t; after, on t+1. Where only the date is known and the
  documented practice is after-hours release (financial statements), the observation is
  available on the next session and the share of filings treated this way is reported.
- **Revisions**: where a source keeps versions (CVM statements carry `VERSAO` and a
  receipt date per version), the point-in-time value at t is the latest version received
  at or before t, so first publications are used first and restatements enter only when
  they were received. Where only a latest vintage exists (ONS corrections, a current
  sector file), the family is labelled `latest_vintage`, the share of values that could
  differ from first publication is estimated, and a sensitivity is reported.
- **Identity**: every family is keyed by dated security identity through the store's
  existing ISIN axis; issuer-level sources (CVM) map through a CNPJ ↔ ISIN bridge built
  from the CVM FCA cadastre with dated validity, so a share-class or ticker change does
  not lose or double-count an issuer.
- **Proof**: each family extends the joined builder fixture — mutate one source value at
  its reference date; every feature at decisions before its availability is bit-identical,
  and the first changed decision is exactly the registered availability. Each manifest
  records `availability_rule`, its evidence, and the first and last usable dates.

No family enters the store without this proof. A family whose publication timing cannot
be established stays `source-semantics unavailable` with the reason, rather than being
admitted under a guessed lag in either direction.

---

## 1. Fixes carried into this round

1. **BOVA11 2019 gap.** `bova11.py` keeps BDI 14 only; the ETF was filed under 02 for
   part of 2019 and every 2019H2 book ran unhedged for 92 sessions. Accept BDI 02 and 14
   for ISIN `BRBOVACTF003`, market type 10, spec `CI`; test that every session from 2010
   with any COTAHIST row for the ISIN has a close; rebuild the hedge-beta sidecar;
   ledger-only replays of every sealed Round-4 book and the 126 CPU panels in fresh roots,
   before/after for F4/F5, other folds exact.
2. **Lending backfill, 2019-10 → 2022-03.** B3's era-1 API (`LendingOpenPositionFile`,
   `LoanBalanceFile`, 2019-10-25 → 2024-04-01, `arquivos.b3.com.br/api/download/
   requestname`) carries per-ticker open balances and min/avg/max donor and taker rates.
   The store's balances currently start 2022-03 and observed rates 2023-07-11. Backfill
   both: balances make the lending family informative on six more folds; observed rates
   replace the 2% placeholder in the ledger's borrow cost from 2019-10, which is a
   realism gain for every sealed book's economics in F4–F11. Stitch and level-audit the
   seams (2022-03 with the PDF archive; 2024-04 with era 2), report the rate
   distribution by year, replay the sealed books under the observed rates, and report the
   before/after economics with the placeholder era now reduced to 2018-01 → 2019-10.
3. **Continuous-book walk-forward** (second priority, after §§2–4): the book carries
   across consecutive folds; the model changes at the boundary; one chronological wealth
   path; the fold-reset readout kept as the historical comparator; no invented scores on
   fold-end dates (audit saved score coverage first). This is the prerequisite for the
   portfolio round and for the overlay's replay.
4. Process rules from the Round-4 readout stand: gates bind registered panels only;
   confirmation seeds for close calls and before any 2025 read.

---

## 2. Families to build now (Group A)

Each family: source, history, availability rule, the features, what the acceptance table
must show. Features are named here so Round 6 can refer to them; exact formulas go in
`feature_spec.py` with descriptions that enter the store identity as today.

### 2.1 Events — filings and material facts (all names, 1998 →)

Source: RAD/ENET `ListarDocumentos` delivery timestamps (minute precision) for ITR, DFP
and *fato relevante*, already harvested for v1 through 2024-12; IPE yearly archives for
the categorised event history (dividends, buybacks, offerings, meetings). Availability:
timestamp + 1 minute, same-session if before 15:44. Features: `sessions_since_financial_
filing` (exists), `filing_is_dfp` flag, `sessions_since_material_fact`, `material_fact_
count_20`, `dividend_announcement_age`, `offering_or_buyback_flag`, and `sessions_until_
expected_filing` defined as the issuer's same-quarter filing lag in the prior year
applied to the current quarter end (labelled an expectation, not a schedule; reported
accuracy by year). Acceptance: issuers covered per year, filings per issuer per year,
share of timestamps before 15:44, the fixture proof.

### 2.2 Fundamentals — point-in-time CVM statements (2011 →)

Source: CVM open data `itr_cia_aberta` and `dfp_cia_aberta` annual archives (DRE, BPA,
BPP, DFC), all versions, with `DT_RECEB` and `VERSAO`; RAD timestamps joined for
intraday precision. Verify on ten representative issuers that every historical version
is present in the archives (if only latest versions survive, the family is
`latest_vintage` and says so). Semantics: consolidated statements where they exist,
individual otherwise, flagged; year-to-date flows converted to standalone quarters using
only versions available at the time; Q4 = DFP − 9M ITR only when both are received;
banks and insurers on their own line items (net interest income in place of gross
profit) and flagged, never forced into the non-financial template. Shares outstanding
from the capital-composition table by receipt date. Availability: receipt timestamp; if
time-of-day is unknown, next session. Features: `log_market_cap` (shares × t−1 close),
`book_to_market`, `earnings_yield_ttm`, `gross_profitability`, `liabilities_to_assets`,
`accruals_to_assets`, `revenue_growth_yoy`, `sue` = seasonal earnings change scaled by
the trailing eight-quarter standard deviation of that change (accounting surprise, no
consensus), `statement_age_sessions`. Acceptance: issuers and name-days covered by year,
share of values from first versus later versions, share with unknown receipt time,
reconciliation of a sample against the published PDF.

### 2.3 Lending (backfilled; §1.2) — 2019-10 →

Existing five features extended backward, plus `utilization_proxy` = balance / free
float where shares outstanding exist (§2.2), `new_loan_volume_surprise` from the
`LoanBalanceFile` contract flow, and the rate level and change already present.
Availability: next session at 15:45 (files published at the open with D−1 data; era 2
identical).

### 2.4 Options activity (optionable names, ~20–40)

Source: option records already inside COTAHIST (TPMERC 070/080: premiums, volumes,
strikes, expiries, since 1986); `DerivativesOpenPositionFile` for per-series OI with the
covered/uncovered split (2019-11 →); `InstrumentsConsolidatedFile` for series →
underlying mapping. Availability: next session. Features: `option_to_stock_volume_20`,
`put_call_volume_ratio_5`, `put_call_oi_log_ratio` (exists, unmapped), `delta_oi_to_
volume_1`, `uncovered_call_share`. No implied volatility in this round (it needs a
pricing step with the DI curve and dividends; register as a later addition). Masked
where the name has no listed options; acceptance reports covered names by year.

### 2.5 Odd-lot (in the store)

Already materialised and complete; confirm the two features and add `oddlot_net_
imbalance_5` if the fractional-market records carry buy/sell direction (they carry
volume and trades only → then report that the imbalance is not derivable and keep two
features).

### 2.6 Cross-market and macro, entering through exposures

A common scalar cannot re-rank a cross-section — the C arm showed that — so this family
enters as per-name exposures and their products with the day's shock, alongside the
shocks themselves as common state for conditioning.

Series and availability (Group A only):

| Series | Source | History | Day-t value available at 15:45 on |
| --- | --- | --- | --- |
| BRL/USD PTAX | BCB SGS (series 1 or the PTAX bulletin API) | 1990s → | t (published ~13:10) |
| Brazil DI curve: swap DI×pré 30/90/180/360/720/1080d | BCB SGS (the *taxa referencial de swaps DI×pré* series) | 2000s → | t+1 as the t close (use t−1 at decision t) |
| US Treasury par curve 3m/2y/5y/10y | US Treasury daily yield-curve CSV | 1990 → | t as the t−1 close |
| Brent | FRED `DCOILBRENTEU` (EIA) as the archive; rule is market availability | 1987 → | t as the t−1 close |
| Iron ore, rebar, HRC, pulp | DCE/SHFE daily settlements (Sina daily klines `I0`, SHFE `kxYYYYMMDD.dat`), mirrored | 2013 → | t (day session closes 04:00 BRT) |
| VIX | Cboe daily history | 1990 → | t as the t−1 close |
| EWZ and the ~25 Brazilian ADR closes | Stooq daily | 2000s → | t as the t−1 US close (after the B3 close: overnight information) |
| Foreign investor net flow on B3 | B3 daily investor-participation publication | verify | the publication date (documented D+2); the audit establishes it |

Features: per-name trailing 120-session OLS exposures of excess return to each shock,
shrunk toward the name's sector (§2.8) mean (`exposure_fx`, `exposure_rates_br`,
`exposure_rates_us`, `exposure_oil`, `exposure_iron`, `exposure_vix`); the shocks as
common-state values (1- and 5-session changes); the products `exposure × shock_1` and
`exposure × shock_5`; `adr_premium_close` = ADR-implied BRL price over the B3 close for
dual-listed names (masked elsewhere); `foreign_flow_5` and its interaction with
`log_volume_mean_20`. Acceptance: each series' first and last date, gaps, the DST
calendar used for US/Asia alignment, the fixture proof for the t versus t+1 rule per
series, and the exposure distributions by sector.

### 2.7 Index rebalance (audit, then build or defer)

The v1 archive has IBOV/IBXX/SMLL preview additions, deletions, signed weight changes,
pressure, pre-effective ramp and post-effective reversal fields for 2023–2024 (45,978
name-days). Build only if each preview's *publication date* is recoverable from B3's
announcements; the effective-date membership must never be available before its
announcement. If publication dates cannot be established for the history, keep the
family `source-semantics unavailable`, and start the daily capture of composition and
preview snapshots now so a future family has an archive. One compact signal if built:
`index_pressure` (signed weight change × days-to-effective, scaled by ADV) and
`index_event_age`.

### 2.8 Sector classification (enabling)

B3's sector/segment classification of listed companies, joined to ISIN through the
bridge; historical changes reconstructed from FRE `setor de atividade` by year where
available, otherwise `latest_vintage` with disclosure (reclassifications are rare).
Used for exposure shrinkage (§2.6), sector-relative features (`name_minus_sector_return_
5/21`, `sector_momentum_12_1`), and later for a sector-neutral target experiment. Not a
model input by itself in this round.

### 2.9 Magnitude channels (enabling E6)

Four causal raw-scale features beside the ranked set: `return_1_over_vol_20`,
`daily_vol_20_raw`, `log_traded_value_20`, `economic_beta_60` (the hedge-beta sidecar's
Blume-adjusted slope, already causal). Stored as their own small family with a declared
clipping estimated on the fit window, never the full panel, so Round 6 can test E6
without a rebuild.

### 2.10 Microstructure (cheap, low priority)

B3 trade-count and after-hours files (2019-10 →): `avg_trade_size_20`, `after_hours_
volume_share_5`. Next-session availability. Build last in Group A; drop if the API
archive does not reach 2019.

## 3. Capture now, use later (Group B)

Index composition and preview snapshots (no archive exists; daily job from today);
DCE/SHFE night-session minute data (forward-only; for the risk track, not alpha); CVM
fund daily reports and CDA holdings — coverage and timing table only (publication delays
and confidentiality windows decide whether flow pressure is ever timely); ONS stored
energy and CCEE PLD — timing table only, with the sign-heterogeneity caveat across
utilities; Focus survey — weekly publication, low novelty, not built. Each produces a
one-page acceptance table and nothing in the store.

## 4. Not in this round

News or text; anything hourly for the alpha model; pre-2010 history; blends (E2); the
calibrated trade-versus-hold policy (E1) and the cost-aware objective (portfolio round,
after §1.3); attention and hyperparameters (Round 6); the 2025 read.

---

## 5. The rebuild and its acceptance

One store build as a new root with every Group-A family as `sidecar_<family>_values/
valid/age_sessions`, the same clean-commit path, memory guards and 8-GiB measured
invariant as `db4f751d…` (the added families are roughly 933 × 3,717 × ~45 floats plus
masks, about 0.7 GiB; report the measured peak). Acceptance: every array outside the new
families bit-identical to `db4f751d…` (the same comparison as the last rebuild); the
calendar still ends 2024-12-30 and no 2025 row reaches any consumer array although the
raw archives extend past it; the per-family acceptance tables of §§2–3; the fixture proof
per family; the existing survivorship, provider-invariance and causality audits.

Because S0 consumes only the slow family, which is bit-identical, the sealed Round-4 S0
panels remain valid on the new store by hash-bound statement; no refit. The 126 CPU
panels are replayed ledger-only under §1.1–1.2 (not because of the new families).

**Sidecar input contract for the network**, implemented and tested in this round so
Round 6 arms are one line each: each family enters through its own zero-initialised
projection gated by its validity mask (a name-day with an invalid family contributes
exactly the parent's representation), the family's parameters receive gradient only from
rows where it is valid, and a test shows that with every family invalid the S0 forward
pass and batches are reproduced bit-for-bit. This is the mechanism the S0 result asks
for: partial coverage must not become noise.

**GBDT information screens** (CPU, readouts, no selection weight for the network):
`a_slow` plus each Group-A family on the fourteen folds, five seeds, paired versus
`a_slow` on the family's informative folds, with TreeSHAP of the family's fields. The
GBDT is a weak learner here (it did not see lending's gain while the network did), so a
null screen does not demote a family; a clearly positive screen orders the Round-6 arms.

---

## 6. Order, budget and stops

1. Registration (`v2_round5_data.md` with §§0–5; the protocol JSON gains the family
   list, each `availability_rule`, the bridge, the sidecar contract). Fixes §1.1–1.2 with
   their replays and before/after reports.
2. Bridge (CNPJ ↔ ISIN, dated) and the RAD timestamp harvest refreshed through 2024-12;
   then families in the order 2.1, 2.2, 2.3, 2.6, 2.4, 2.8, 2.9, 2.7 (audit), 2.10; each
   lands as an immutable, hashed archive root with its acceptance table before the build.
3. One store build; acceptance; S0 validity statement; sidecar contract with tests.
4. GBDT screens; §3 timing tables; §1.3 continuous-book evaluator.
5. `docs/v2_ROUND5_DATA.md`: the family table (source, history, availability rule and
   evidence, coverage by year, revision share, first usable fold), the fixes'
   before/after, the screens, the store manifest, and the Round-6 arm order it implies.
   Stop.

Budget: CPU only; several days of acquisition and parsing (CVM statements are the long
pole), one build of about two hours, screens overnight. No paid instance. The overlay
track waits for this round to finish unless Gabriel says otherwise, because both compete
for the same hands.

Stops: the 8-GiB invariant; any 2025/2026 row in a consumer array; any array outside the
new families changed by the build; a family admitted without its availability proof; a
replay changing a non-ledger field. A source whose timing cannot be established is
recorded as unavailable, which is a result, not a stop.

---

## 7. Round 6, so the data is built for it

Arms on S0 through the sidecar contract, one family each, ordered by the §5 screens and
the literature prior (events and fundamentals first, then lending-backfilled, cross-market
exposures, options, odd-lot, index if built), then the combination of the families that
helped; alongside, the cheap non-data arms: fine-tuning multiplier 1.0 versus 0.3 (E4),
the magnitude channels (E6), time decay 756, and the point-in-time MLP comparator (E8)
that tells us whether the GRU's temporal modelling is doing work. Attention and the
hyperparameter pass follow in Round 7 on whichever parent Round 6 produces. The
portfolio round (continuous-book evaluator → cost-aware policy → cost-aware objective)
runs when §1.3 exists and the meaningful-capital cost analysis has set what turnover
costs.
