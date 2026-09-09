# Intraday repair: archive-only 2024 coverage

The repaired archive audit covers 251 sessions and 158 M1 identities, with 39,658 possible name-days (32,991 active). It uses no model scores or 2025/2026 payload. Peak RSS was 2.17 GiB. The immutable source store is the canonical `8021e42` store.

All fractions below use the same 158-name denominator. The active-only coverage is included separately; transformed coverage also requires the existing 20-name cross-sectional support.

| Feature | Old transformed | Repaired raw | Repaired active-only | Repaired transformed | Sessions with 20 active observations |
|---|---:|---:|---:|---:|---:|
| overnight_return | 0.0% | 59.5% | 69.5% | 57.8% | 231 |
| intraday_return_1545 | 64.2% | 67.0% | 77.2% | 64.2% | 251 |
| overnight_return_sum_5 | 0.0% | 43.2% | 51.1% | 42.5% | 202 |
| overnight_return_sum_20 | 0.0% | 23.8% | 28.4% | 21.8% | 171 |
| intraday_return_sum_5 | 30.0% | 57.9% | 68.0% | 56.6% | 251 |
| intraday_return_sum_20 | 0.1% | 48.3% | 57.1% | 47.5% | 251 |
| overnight_minus_intraday | 0.0% | 59.5% | 69.5% | 57.8% | 231 |
| overnight_minus_intraday_mean_20 | 0.0% | 23.8% | 28.4% | 21.8% | 171 |
| last_30_minute_return_share_lag1 | 0.0% | 56.9% | 67.1% | 55.8% | 231 |
| last_hour_volume_share_lag1 | 25.8% | 24.7% | 29.5% | 24.6% | 229 |
| close_vwap_deviation_lag1 | 0.0% | 24.7% | 29.5% | 24.5% | 229 |
| vwap_deviation_1545 | 27.0% | 27.3% | 32.6% | 27.0% | 249 |
| realized_vol_5m_1 | 70.6% | 71.7% | 84.8% | 70.6% | 251 |
| realized_vol_5m_5 | 59.7% | 68.1% | 81.1% | 67.4% | 251 |
| realized_vol_5m_20 | 49.1% | 65.1% | 77.8% | 64.7% | 251 |
| realized_skew_5m_20 | 70.7% | 71.2% | 84.9% | 70.7% | 251 |
| roll_spread_20 | 56.1% | 56.5% | 67.5% | 56.1% | 251 |
| corwin_schultz_spread_20 | 60.4% | 64.8% | 77.4% | 64.4% | 251 |
| intraday_range_1545 | 75.7% | 77.0% | 91.0% | 75.7% | 251 |
| volume_1545_relative_median_20 | 0.0% | 3.5% | 4.3% | 0.1% | 2 |

Return consistency uses adjacent exact M1 closes against shareholder returns at the unchanged 0.005 log-return tolerance. Price levels remain in their own source units. Same-day close validation and inferred close-based actions cannot suppress that day's 15:45 prefix. Rolling sums/means use at least 4/5 or 16/20 actual observations, without extrapolation. Supporting fractions and underlying observation ages are retained.

The relative-volume feature remains nearly unavailable after transformation: only two sessions have 20 active supported names. Complete-prefix volume is known for a small subset of the sparse archive, and its 16-of-20 denominator further reduces support. Missing minutes remain unknown volume. This is an observed data limitation, not a reason to fill missing activity or relax the cross-sectional transform.

The odd-lot archive was independently re-derived from COTAHIST 2009-2024 in annual batches: 1,253,007 raw rows at 0.413 GiB peak RSS, with exact ISIN identity. Source dates are published next session; the final consumed source date is 2024-12-27, available 2024-12-30. The December 30 source archive is scanned but contributes no consumer row inside the bounded calendar.

The new lending archive's `annual_taker_rate` is already decimal. Its loader projection now preserves that column; no inversion or scaling is applied. Thirty-two targeted sidecar/archive/native-fast tests passed. Ninety earlier targeted intraday/store tests passed, including provider invariance and same-day close-mutation causality.

This audit precedes the one combined store rebuild and is not a model-performance result. The initial archive attempt stopped on an empty bounded source frame; `29f3238` corrected that input handling and this audit uses a fresh root.

Machine-readable evidence and exact archive/manifest hashes: [coverage evidence](v2_intraday_coverage_evidence.json).
