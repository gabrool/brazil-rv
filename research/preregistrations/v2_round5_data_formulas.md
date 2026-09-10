# Round 5 formula and source interpretation amendment

This amendment supplies exact formulas and resolves source assumptions under the
user's standing instruction to choose the technically correct interpretation.
It is recorded before the final family archives, store extension and information
screens. It does not change S0, the 465 sealed replay inputs, their score-derived
readouts, the development boundary, or the prohibition on neural fits. The
original registration and replay bindings remain intact.

- CVM annual account files often omit earlier versions even when headers retain
  them. Recover original ENET document tables; never substitute a later version.
  Annual DFP directly establishes annual TTM. Interim TTM is current YTD plus
  prior annual DFP minus prior-year matching YTD, each using its own currently
  known version and the same fiscal period/accounting basis. Standalone quarters
  and accounting surprise retain their required adjacent-quarter inputs.
- SUE is current seasonal earnings change divided by sample deviation of the
  previous eight consecutive seasonal changes. It does not include the current
  surprise in its scale. Financial, consolidated and single-class-valuation flags
  accompany the financial fields requested in the original scope.
- Before FCA ticker columns become usable, exact normalized historical legal-name
  matches may connect a dated FCA issuer to dated COTAHIST identity. Require one
  contemporaneous issuer and an already observed security. No later survivor or
  current-sector confirmation is required. An old FCA ticker cannot label a new
  ISIN unless the FCA explicitly preannounced its matching first listing session.
  A shared legal CNPJ root additionally requires the same CVM issuer code.
- FRE's January reference date can be a filing-year label. Free float is measured
  at its actual `Data_Ultima_Assembleia`, enters only after receipt, and remains
  separate from total issued shares. Capital-boundary tests use that measurement
  date. Unsupported classes and units are masked rather than priced as one class.
- For common market state, retain all retrieved DI vertices and Treasury tenors;
  do not reduce acquisition to the two regression benchmarks. The six exposures
  use PTAX, DI360, Treasury10y, Brent, iron ore and VIX. Rates change in percentage
  points; other shocks are log returns. One/five-session shocks refer to source
  sessions. Futures use linked exact-contract returns selected using prior OI.
  A source holiday retains the last known shock with its true age, and never
  inserts another print or repeats a return in the five-session sum.
- Exposure OLS includes an intercept and uses reference-date matched wealth log
  return minus `log1p(CDI)` from the prior 120 B3 sessions, with at least 60 observed
  pairs. Both measurements must already be public. The current stock outcome is
  excluded. A same-day Asian/PTAX shock can therefore influence the current
  interaction, while the exposure estimate still ends at t-1.
- Sector shrinkage uses currently known sectors. Collapse peer share classes to
  one issuer mean. With at least three other issuers, use their mean slope as a
  prior, and `tau²=max(sample_variance(peer_slopes)-mean(peer_estimation_variance),0)`.
  The posterior slope is `(tau²*own_slope+own_variance*peer_mean)/(tau²+own_variance)`.
  Without enough sector peers, retain the observed OLS slope. Other own share
  classes never count as independent peers. Sector-relative returns/momentum
  require at least two other issuers and use equal issuer weights.
- Common state, exposures, their products and magnitude channels keep physical
  units rather than cross-sectional ranks. This prevents a common scalar from
  becoming identically zero. Magnitudes alone have their registered per-fit
  0.5/99.5 percentile clipping; selection/scoring reuse those exact bounds. A
  feature absent in the fit window cannot acquire bounds from later observations.
- Actual US early closes can be available at the same B3 decision. Convert through
  historical `America/New_York` and `America/Sao_Paulo` rules. VIX's final
  dissemination extends one minute from 2021-09-27; half-day availability follows
  Cboe's corresponding shortened session. Historical pre-2011 PTAX observations
  use their own timestamp rather than a modern lunchtime assumption.
- BVBG.086 `OpnIntrst` is opening-of-session D, equivalent to the prior closing
  position. A proven after-hours D report admitted at D+1 has source age two;
  its volume denominator also ends at the position date. Price-report option
  trading volume still describes D. Omitted OI is not zero without completeness
  evidence. Missing files with an empty archive response are described as
  unretrievable, not as proof that no historical publication existed.
- Compact index pressure compares current and preview weights from the same
  disclosed snapshot. Sum weight delta in percentage points times remaining
  sessions, divided by ADV in BRL millions. This is a deterministic unit choice,
  not an assumed fund AUM. Dated BDI opening-table footnotes can establish the
  snapshot's availability even when a cached attachment was uploaded later.
- A source remains unavailable when its measurement or first-publication semantics
  cannot be verified. Retain retrieved raw evidence and distinguish download
  failure, missing original vintage, identity ambiguity and timing ambiguity.
  No guessed extra lag cures these problems.

Exact ordered field descriptions and transforms are in `feature_spec.py` and are
hashed into the new store. Joined fixtures bind first admissible decisions to the
actual per-source clocks. The final build plan binds this amendment, each admitted
family archive, its proof, and the unchanged base store.
