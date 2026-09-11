# Round 6 CPU acceptance and fixed experiment plan

No new neural fit or model score was used to build this acceptance. The supplied
registration and the source, model, return-basis and economics amendments are
fixed under `research/preregistrations/v2_round6*`. The authoritative input
record is [v2_round6_inputs.json](v2_round6_inputs.json).

The final store has 3,717 sessions and 933 permanent ISINs, ending 2024-12-30.
All 94 protected arrays, two axes and 36 tables match the accepted Round-5 store.
Only cross-market, lending and rebalance are replaced. Peak builder RSS was
5,536,452,608 bytes. The previous unscored candidates remain immutable evidence.

| Preflight | Outcome |
| --- | --- |
| Cross-market | Brent admitted next B3 decision; same-endpoint BRL ADR and EWZ gaps; dated program/class identity; published MTD flow differences and interactions. |
| Foreign flow | 687 bounded archive requests, 496 recovered participation tables. Missing tables remain missing. D-2 reference dates are separate from publication; date-only bulletins enter the next decision, without an extra lag. |
| Index releases | 44 events, including 25 newly recovered 2018-2022 releases with full portfolios. A missing effective cycle is not silently bridged. |
| Lending | Original observations preserved; 6,483 positive latest-vintage additions, with 1,535 exact overlaps. Latest-vintage status remains explicit. |
| Fundamental/event signs | Nineteen fields, pooled and by year; no selection weight. Earnings yield and SUE positive; leverage and accruals negative. Profitability and size retain their adverse-era results. Thirteen bounded TTM, unit, receipt and valuation tests passed; no independently demonstrated sign/alignment error. |
| Historical rate probe | Wayback: 287 captures; seven download captures with four unique fixed-width payloads. Four isolated 2016-2017 reference days were recovered. This is useful sparse evidence, not a continuous first-vintage 2012-2023 archive. No training admission. |
| Parent transfer | All three sealed S0 Stage-P checkpoints transferred with protected parameter values exact and new residual projection zero. MLP uses three architecture-matched P runs. |
| S0 evaluation integration | One reused F1 panel: seed ensemble, primary IC and economic headline exact. Four daily decomposition fields differ by at most 1.42e-14 bps from floating-point arithmetic; all other daily fields exact. No parent refit. |

Informative folds require positive active input support in both fit and
evaluation; selection counts are reported separately. Lending remains F6-F14,
sector F2-F14, rebalance F3-F10/F12-F14, and the other families all fourteen.
Whole folds are compared; no outcome-based name/day subset is selected.

The ADR adjustment diagnostic found 1,205 material changes in the US
adjustment-versus-close return difference across the covered ADR/EWZ series:
837 matched a retained B3 distribution date, 179 lay within seven days, and 189
were unmatched in that archive. These counts are descriptive, not proof of a
vendor error or a trading premium. Dated depositary timing, fees, taxes and
missing provider events remain possible explanations.

The extreme-gap review also identified the inherited U2 inferred-action problem
described in the [return-basis amendment](../research/preregistrations/v2_round6_return_basis_amendment.md).
The new ADR and oil calculations exclude 6,200 inferred-action return cells
across the underlying panel. This source-semantic correction is independent of
IC and gap-size thresholds. S0 inputs, targets, old fields and economics remain
fixed; this information round does not certify the absolute wealth ledger.

The tested `placeholder_v2` sensitivity uses n/(n+20) shrinkage of each name's
2023-2024 median observed rate toward its liquidity-quintile median. Names
without a usable observed rate take the quintile median; missing liquidity uses
the pooled median. It changes only explicit flat-rate placeholder cells. It is
hindsight on earlier books, has zero promotion weight, and is never a feature.
There are 356 calibration names with observed rates. The final economics tables
will show this and the separately labelled latest-vintage rate replay beside
the causal headline. Detailed stale-name settlement research remains a readout
task and does not gate neural fits under the registration.

The staged runner performs ten score-free session-one smokes (nine F settings
and MLP P), three MLP P fits, then 378 F fits using seeds 11/29/47. Each enabled
family is forced invalid in a separate inference-only score artifact; multi-family
arms also get the joint ablation. A completed fit is not marked ready until
required scoring finishes. Session two is locked from the complete session-one
informative-fold paired-IC rule. All experiments keep the original epochs,
patience, expanding folds and registered evaluation rules. Benchmark concurrent
throughput before estimating wall time. No forward capture or 2025 consumer
rows, and no Round-7 launch.
