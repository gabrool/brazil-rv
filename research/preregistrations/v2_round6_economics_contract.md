# Round 6 economics and attribution details fixed before new scores

The headline uses the repaired Round-5 BOVA hedge inputs, the original admitted
borrow-rate archive, the adopted buffer-9/equal-weight D3/D5/D10 policy, and A1
full-calendar fold-reset accounting. Preserve the sealed S0 scores. A changed
feature-store manifest never authorizes a change to any parent outcome array.

`placeholder_v2` changes only flat-2% placeholder cells. Estimate each permanent
ISIN's median of distinct observed 2023-2024 annual taker-rate sessions from the
original plus separately admitted latest-vintage extension. Use n/(n+20) weight
on that name median and 20/(n+20) on the liquidity-quintile median; n is the number
of observed sessions, fixed independently of any model result. Quintile medians
give each observed name one vote, not each rate row. Determine calibration
quintiles by each name's median positive prior-20-session traded value during
2023-2024. At each replay decision classify the name using its already available
prior-20-session traded value against those fixed calibration boundaries. A name
without a usable observed rate uses that quintile's median. If liquidity is
unavailable, use the pooled median and label the fallback; an empty quintile
also uses the pooled median. No observations means the sensitivity is unavailable,
not a invented zero rate.

The 2023-2024 estimates and boundaries are hindsight on earlier years. This
scenario is a cost sensitivity only and has zero promotion weight. Keep causal
observed/recent and cross-sectional-imputed rates unchanged; replace only cells
explicitly marked as the historical flat-rate placeholder. A separate
`latest_vintage` ledger uses the extended rate archive's same original pricing
rules and does not masquerade as retained first-vintage history. All scenarios
are actual ledger replays so changes to NAV, trade sizes and costs are consistent.

Inference attribution uses every enabled family forced invalid individually and,
for multi-family models, all enabled families jointly. Score the same selected
checkpoint on the same fold and active population; no refitting and no change to
training-derived magnitude clipping. Attribution panels are descriptive and never
new promotion candidates. Paired informative folds, all-fold results, fixed
leave-one-seed-out choices, and registered confirmation rules remain unchanged.
