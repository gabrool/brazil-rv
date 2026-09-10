# Round 5 observed-series OI amendment

Registered before the activity-family archive, store build and information screens.
The user's standing authorization permits the recommended source resolution.
The complete-market OI fields keep their existing strict completeness requirement.

The public BVBG.086 archive exposes an optional OI value for each reported option
series. No authoritative omitted-value-equals-zero rule has been recovered. That
prevents a claim of complete underlying-level OI; it does not erase positions
actually reported for identified series. Across the first 890 successfully parsed
source days (2019-11-01 to 2023-06-07), 140,379 underlying/date rows have known
listings, only 382 have complete reported OI, and 117,918 have positive reported
call and put positions. These are source-support counts, with no target, screen
or economic outcome inspected. Evidence is
`D:/quant-data/b3/interim/round5_data_20260910T140442Z/observed_subset_oi_support_audit.json`.

Add two explicitly labeled source features to the options family:

- `observed_series_put_call_oi_log_ratio`: natural log of the sum of **reported**
  put OI divided by the sum of **reported** call OI. Both sums must be strictly
  positive; an absent side is unavailable. There is no zero imputation or
  pseudocount. Changing series composition is a known interpretation limit.
- `observed_series_oi_coverage`: count of listed series with an explicit OI field
  divided by the dated known listed-series count. A known listing population with
  no OI reports has support zero; this is an observed support statistic, not zero
  economic OI. Unknown listings remain unavailable.

Both features use only the original PR version published by the historical 15:45
decision. BVBG.086's opening-D positions represent the D-1 close. An after-hours-D
publication consumed at D+1 consequently has age two for these OI statistics.
No second lag is applied at the family join. The original source creation times,
identity intervals and reported fields remain bound in the archive manifests.

`put_call_oi_log_ratio` and `delta_oi_to_volume_1` retain strict complete-market
semantics. `uncovered_call_share` remains unavailable without its separate source.
No observed-subset OI difference is added: the current aggregate does not establish
an unchanged matched-contract population between adjacent reports.

The fixture proves both-positive support, unavailable missing sides, explicit
coverage and its age, and unchanged decisions before publication with the first
change at the exact eligible decision. The screen treats these as additional
registered information fields; they cannot be relabeled complete-market OI.
