# v2 semantic migration

The current multi-day contract is intentionally incompatible with earlier v2
stores and runs. This is a semantic break, not a filename-only revision. Do not
rename, copy, or reinterpret an old artifact to make it load.

## What changed

| Earlier artifact assumption | Current contract |
|---|---|
| Stage-specific consumers shifted slow rows inconsistently | every stage reads canonical decision row `t`; its daily market fields end at `t-1` while decision-available publications may be current on `t` |
| v1 26-channel minute tensors, old slow context, and 12 empty clock patches defined the fast branch | native seven-channel five-minute prefixes use real dated-session patches and per-channel masks |
| a v1 fast checkpoint was a normal dependency | native fast weights initialize freshly; the isolated legacy adapter is explicitly contaminated and diagnostic-only |
| inferred/fixed legacy session axes | a versioned B3 schedule defines sessions, decision time, and full-session boundaries |
| realized price ratios and synthetic event neutralization could define adjusted economics | verified contractual shares/cash/claim mappings define shareholder wealth; unresolved actions stay unresolved |
| one ambiguous adjusted/raw target family | primary median-adjusted/scaled shareholder labels plus independent raw shareholder and price-return families |
| one row/readiness flag could stand in for missingness | per-feature slow/current masks, a separate calendar/padding mask, per-channel fast validity, fast presence, eligibility, and independent outcome masks |
| compulsory to-close loss at weight 0.5 | auxiliary default is 0.0; the only registered enabled weight is 0.2 |
| evaluation could average changing horizon/name support | D1/D2/D3/D5 use one common >=20-name population per date; D10 is a separate diagnostic |
| closing outcomes could determine replacement names or liquidation | intended orders precede fills; unresolved signed inventory and claims remain in the ledger |

Superseded adjusted and synthetic-return fields such as
`adjusted_{open,high,low,close}`, `price_adjustment_factor`,
`neutralized_log_return*`, and `target_raw_*` are not part of
`BRAZIL_RV_V2_DAILY_STORE_V3`. Their nearest current arrays are not
byte-compatible replacements: raw OHLC, shareholder-wealth OHLC,
contractual-action arrays, `target_primary*`, `target_shareholder_*`, and
`target_price_*` have different meanings and masks.

## Required migration

There is no in-place data migration:

1. Keep old stores, checkpoints, scores, and experiment roots immutable as
   historical evidence.
2. Build a new `BRAZIL_RV_V2_DAILY_STORE_V3` store from immutable raw sources
   with the current clock/calendar, FeatureSpec, identity links, contractual
   actions, M1 assignments, and source hashes.
3. Run the engineering acceptance checks on that sealed store.
4. Train new P/F/J checkpoints under the current input and chronology schemas.
5. Create new scores, evaluation outputs, run identities, and—before research
   scoring—a revised preregistration.

An old store is rejected at open time with a diagnostic naming the decision
row, action/return, mask, and native-fast incompatibility. Checkpoints and
scores additionally bind their schema, store manifest, feature schema/order,
feature-age and decision-row contracts, stage/fold/date access, lookback, and
model configuration. Resume and runner
paths require exact manifest/result hashes, so a stale partial run cannot be
continued under new semantics.

The optional `legacy_v1_contaminated` model adapter does not migrate an
artifact. It exists only for explicitly labelled historical diagnostics,
requires exact file identity and ancestor chronology, and cannot make unknown
temporal provenance admissible for clean or official evaluation.
