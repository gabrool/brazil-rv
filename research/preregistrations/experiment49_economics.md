# Experiment 49 — 15-minute economics and bounce robustness

Frozen before any Experiment-49 number exists. This is a zero-GPU,
discovery-fold-only study. It may read raw prices through 2025-06-30 only to
estimate execution costs; it may not read official-validation model predictions
or any held-out-test data.

## Sources and scope

- Canonical raw M1 sources resolved through the canonical feature-store manifest.
- Immutable Experiment-48 Part-C four-head Fold A/B/C prediction archives and
  frozen cross-fit Raw Patience-3 realizations.
- Immutable Experiment-41 store-v2 Fold A/B/C comparator archives and the exact
  parent replays already bound by Experiment 48.
- The canonical causal target-scale array used by Experiment 48.

All resolved paths and SHA-256 identities are frozen in the runtime design. No
training, official prediction read, test read, pool score, or deployment change
is allowed.

## Part 1 — Roll effective spreads

For each permanent security ID and calendar quarter, form close-to-close log M1
returns only from consecutive observed bars within the same session. A usable
lag pair requires three consecutive observed closes. Use the sample covariance
of `(r_t, r_{t-1})`; when it is strictly negative, measured full effective
spread is `2 * sqrt(-covariance)`, otherwise it is missing. Liquidity groups are
top 40, ranks 41–100, and ranks 101–158 based on the quarter median of causal
20-session trailing ADV ranks. ADV for a date is the mean of the previous 20
observed sessions' daily `close * real_volume`, excluding that date.

The executable schedule uses the exact security-quarter estimate when present,
then the same security's most recent prior valid quarter, then the current
quarter liquidity-group median, then the current-quarter market median. Every
fallback is recorded. The schedule reports full spread and per-side half-spread
in fractions and basis points, plus liquidity rank/group. It is compared with
the documented modeled tick-spread table when that table is present; absence of
that non-canonical document is reported rather than synthesized.

## Part 2 — alternative labels

Evaluate archived predictions against exactly two alternatives while retaining
the original causal target scale, cross-sectional median removal, minimum-active
rule, and centered midranks:

- open-to-open horizon `h`: `log(open[T+h] / open[T])`;
- mid-proxy horizon `h`: `log(m[T+h] / m[T])`, where
  `m[t] = mean(close[t-1], close[t])`.

Every endpoint must be observed and within the same session; missing endpoints
are masked, never filled. Horizons are 15/30/60/120 minutes for the Experiment-48
candidate and 30/60/120 for the comparator. Report raw IC, alternative IC, and
`alternative/raw` retention by fold/head/variant. Raw IC is recomputed directly
from the frozen archived observation contract.

## Part 3 — spread localization

Within each fold selection window, split securities into terciles using the
median executable Roll full-spread schedule over that window. Freeze the tercile
membership before computing IC. Report each candidate and comparator head's raw,
open-to-open, and mid-proxy IC within each tercile.

## Part 4 — executable books

Use the causal trailing ADV top 80 at each decision date, intersected with valid
labels. Construct centered rank-linear dollar-neutral weights and normalize
`sum(abs(weight)) = 2`. The 15-minute candidate book rebalances at decision
indices `0,3,6,...`; the comparator 30-minute book at `0,6,12,...`. Entry is
`open[T]`, exit is `open[T+h]`. Each independent horizon holding pays entry and
exit turnover; per-unit side cost is measured Roll half-spread plus 2.0 bps.
Impact is zero. Report gross/net bps per day, annualized daily net Sharpe,
turnover, per-rebalance net alpha, and daily 15m/30m net-PnL correlation.

The combined book uses equal risk. On each date, weights are inverse-volatility
from each component's prior daily gross PnL only; before 20 prior finite dates it
uses 50/50. The two risk weights are normalized to sum to one, so no future day
affects a weight.

## Frozen verdict

KEEP the 15-minute head iff all three conditions hold:

1. mid-proxy 15m retention is at least `0.60` on at least two of three folds;
2. 15m net expected daily return is strictly positive on every fold;
3. combined net Sharpe is at least the 30m-only net Sharpe on at least two folds.

Otherwise DROP. The verdict is written before Experiment 50 is realized. In
either case, record the 30/60/120 retention table as pre-test calibration. A
dedicated 15m model remains an unbuilt future option.

All manifests, archived-source inventories, derived tables, configuration,
analysis, verdict, and final audit are retained and SHA-256 inventoried.
`official_validation_accessed=false` and `test_accessed=false` everywhere.

