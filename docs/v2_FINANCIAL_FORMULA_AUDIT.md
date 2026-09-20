# Financial numerators, trailing periods and share denominators

Stage B now independently reconciles the selected financial account values,
financial formulas and share denominators used by the current identity overlay.
There is no remaining discrepancy at these audited boundaries and no resulting
change to source data, financial values, accepted stores or fitted coordinates.
Resolve the evidence through `v2_economic_data_scaling_run.json` and
`v2_financial_formula_acceptance.json`.

## Original amounts and units

All 22,210 cached own-version filings, including the 427 newly linked filings,
have their selected accounts checked against the original published rows. The
273,625 observations comprise 240,434 annual-CSV rows, 33,086 original HTML rows
and 105 original ZIP/XML rows. Amounts, accounting basis, source code, description,
start/end periods and BRL unit scales agree at source numerical precision.
The audit uses independent CSV/HTML/XML extraction, not the production account
loader. It verifies 21,431 consumed source-file hashes in 81.05 seconds; it does
not repeat the earlier capital/header/source census.

The 90 previously reviewed currency-unit dispositions are also checked against
their complete required after-state. A narrow qualification confirms that a missed
unit correction is rejected even if an unscaled source number matches. This reuses
the existing interpretation of each own-filing note, rather than claiming a new
independent reading of all notes. The initial executed audit and qualification
remain separately recoverable. No accepted account values changed.

All 427 incremental own-version capital tables reconcile in their independent
share units, including treasury subtraction. The earlier 21,783-document capital
audit is reused. Capital snapshots remain dated by their economic reference, and
their availability remains their own filing receipt.

## Formula and availability reconstruction

The independent reconstruction checks all 949,539 rows of the current financial
overlay: 7,596,312 value/missingness comparisons across eight fields, plus dependency
and update ages, for 19,451,154 comparisons in the initial full pass. It evaluates
33,084 as-of states. The fields are market capitalization, book to market, earnings
yield, gross profitability, liabilities/assets, accruals/assets, revenue growth
and standardized earnings surprise.

Annual flows are observed directly; interim trailing-year values use current YTD
plus the preceding annual period minus the corresponding prior YTD. Sources must
share the fiscal start, accounting basis and correct period endpoints. Each source
uses its latest already-public version. Parent-company earnings/equity and explicit
minority subtraction remain separate from consolidated totals. Accruals use the
average of current and prior-year assets. Surprise scales use the previous eight
consecutive seasonal changes, excluding the current change. Incomplete filings
retain coherent earlier calculations, with their actual dependency clocks.

Market capitalization sums each outstanding class at its own previous observed
close. Positive reported classes need an unambiguous dated identity; treasury is
excluded. Known capital events and observed unit boundaries invalidate stale share
denominators. These checks preserve the original market-input contract; they do
not certify the still-pending repaired wealth/history store.

The initial full pass took 482.60 seconds and reported 1,896 discrepancies, all
for Camil. They came from the audit's prior-year date calculation, which lost the
February month-end across leap years. Production already handled this correctly.
After fixing the independent audit, only Camil's complete 1,406-row issuer history
was rechecked (15.22 seconds); all discrepancies resolve. Its largest arithmetic
difference is 8.89e-16 in surprise, from independent standard-deviation arithmetic.
The passed remainder was not repeated. Initial results and executed source bytes
are preserved; the qualified receipt explicitly supersedes those discrepancies.

## Free float and the actual utilization consumer

All 8,991 FRE circulating-share records exactly reproduce from the bound original
annual files, with matching ID, issuer, version, class and measurement date. There
are 4,104 exact-minute receipt records; the remainder use date-only upper bounds.
Together with the incremental capital check, this takes 1.89 seconds.

Seven records carry a measurement date after their receipt. Their printed dates
are retained. The independent utilization reconstruction excludes future
measurements and reproduces all 29,943 valid ratios/ages exactly, including the
original Float32 storage contract, from 242,232 candidate loan-balance rows.
Four future-measurement candidates are excluded where relevant. No selected
denominator is below 10,000 shares. Identity/class ambiguity, source timing,
capital changes and observed share-unit boundaries explain the other rejected
candidates. This consumer check takes 4.98 seconds; it does not repeat lending
PDF acquisition or parsing.

## Interpretation and remaining program

These audits establish correctness of selected source values and their implemented
arithmetic. They do not prove that every useful unselected account was admitted,
that issuer disclosures are free from errors, or that a single retrieved vintage
preserves all historical revisions. Public float and lending balances are not
executable locate capacity. No model bps/day or predictive improvement follows from
passing these checks. The earlier identity coverage gains remain unchanged.

Stage B still needs the remaining upstream-family clocks/revisions/units, full
ALLOS/ISA history and universe/warm-up/wealth/label propagation, separately
attributed recovered-lending features and actual tensors in an accepted derived
store. Stage A's execution and source-sensitivity bounds remain open. C/D have not
started; there are no corrected profitability results or new GPU fits. All timings
above are CPU audit observations, not neural training estimates.
