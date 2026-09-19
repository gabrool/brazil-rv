# Auxiliary producer-to-tensor audit

All ten auxiliary families reconcile exactly with the accepted 2010–2024 store:
113 fields, 41,942,683 valid active field observations, zero value, validity-mask
or age mismatches. No source, accepted store, feature coordinate or fit changed.
The run pointer binds `auxiliary_tensor_audit` and `oddlot_tensor_audit` receipts.

The nine decision-dated family audit independently joins exact date/ISIN keys and
implements the five typed transforms from the sealed schema. It compares all
stored cells, including invalid/inactive cells and ages. It does not call the
production aligner or transform. All 111 fields reconcile; no source-domain
rejections were found. Source rows from inactive names remain in the physical
family archives, rather than being fabricated into observed neural inputs.

| Family | Fields | Valid active field observations |
|---|---:|---:|
| Cross-market | 65 | 29,708,824 |
| Events | 7 | 2,133,384 |
| Fundamentals | 14 | 3,622,656 |
| Lending | 7 | 663,384 |
| Magnitudes | 4 | 2,260,277 |
| Microstructure | 2 | 818,526 |
| Options | 7 | 866,806 |
| Rebalance | 2 | 130,544 |
| Sector | 3 | 601,853 |
| Odd-lot | 2 | 1,136,429 |

Odd-lot requires a different independent oracle: BRL odd volume divided by total
regular-plus-odd volume, an exact exchange-session lag-five join, and a check that
the earlier observation was already published. All 1,253,007 archived rows have
exact next-session availability. The reconstruction retains 2009 warm-up and
finds 125 valid names in each field on the first store session. It does not shift
five observed rows across missing sessions or reuse a future revision. This was
a new derived-volume-to-tensor audit; completed COTAHIST/lending censuses were not
repeated.

Nine family parquets omit original publication timestamps. The audit verifies
their bound upstream manifest/proof identities, and uses the repaired financial
family rather than the superseded financial parquet. The older evidence records
specific clocks: own-version CVM receipts and minute upper bounds; previous-session
US closes; BDI next-decision dates; distinct BVBG opening/closing-position dates;
and point-in-time sector mappings. Those receipts provide the next source-level
audit targets, not blanket proof from a final parquet date. Several market and
lending archives explicitly retain unknown historical revision share.

This completes the auxiliary producer-to-sealed-store boundary, not Stage B.
Original publication/revision audits, units and denominators, complete identity
propagation, corporate wealth/labels, fit-only conditioning and actual final neural
tensors remain. No forecasting or profitability inference follows from exact
reconstruction. Single CPU observations were 16.51 seconds for nine families and
.61 seconds for odd-lot, not neural-fit estimates.
