# Dated auxiliary target normalization and AERI final-row qualification

The optional to-close target now takes dated continuous-session and completed-prefix
minute counts. The producer no longer supplies the fixed 405/345 convention. The
registered formula remains `sqrt((total-prefix)/total)`, with the existing
pre-decision five-minute-return RSS, cross-sectional median, clipping at ±5 and
midrank transform. Changing that volatility estimator or the denominator from
total to prefix would be a separate hypothesis. This repair does not activate the
optional loss, which was inactive in the foundation parent.

Resolve `to_close_clock`, `to_close_clock_input_audit`, `aeri_boundary` and
`aeri_wealth_qualification` in `v2_economic_data_scaling_run.json`. These are
explicit derived amendments, not replacement accepted inputs or results from a
new fit. Both accounting paths and frozen learned-policy coordinates are unchanged.

## Target evidence and effects

The bounded recovery reused 17,817 already reconstructed target inputs from the
qualified rename cross-sections. For the remaining 72,060 admitted name/date
outcomes it read 4,963,736 required rows from 144 original physical files: completed
five-minute closes before the dated decision, the exact entry open and exact
continuous close. Reads stay within accepted per-ISIN assignment dates and end by
2024-12-30. Missing minutes are not filled. The unchanged 80% return-support rule
is checked. No OHLCV/native/scalar census or passed rename reconstruction was
repeated. The original file hashes remain bound by the sealed M1 audit.

Recovered inputs reproduce all four old-convention arrays exactly on the full
3717-session/933-name axes, including all 89,877 admitted outcomes: 89,799 sealed
plus the previously accepted 78 rename additions. Float32 input-storage semantics
are preserved before the float64 target arithmetic. This directly recovers the
inputs to all 65 clipped residuals; no clipped value is inverted or rescaled.

| Clock-only effect versus rename-admitted old convention | Count |
| --- | ---: |
| Valid outcomes added/lost | 0 / 0 |
| Existing raw log returns changed | 0 |
| Normalized residuals changed | 88,311 |
| Midrank targets changed | 12 |
| Residuals at clipping boundary, before → after | 65 → 19 |

The 12 rank changes occur on four dates: 2023-02-22, 2023-12-13, 2024-02-14 and
2024-12-11. A common positive scale preserves order except where clipping ties
change. The actual changed indices/values are in the sparse delta artifact.
An independently coded endpoint/median/clipping/tie-rank oracle matches all
89,877 outcomes. Deleting later inputs/clocks preserves the earlier prefix.

The actual dataset/collator verifies 18 full-933-name, 60-session samples,
covering supported schedule variants and every changed-rank date: 67,176 target
array/CPU tensor cells, zero mismatches. Their audit views retain the completed
native/history amendments and are not accepted training stores. Recovery took
23.68 seconds, complete arithmetic attribution 24.37 seconds including recovery,
and consumer verification 6.02 seconds; none is a neural-fit ETA.

## AERI source lineage and actual amendment

AERI3 / BRAERIACNOR4 on 2024-12-30 was an inherited pre-Round-7 U2 inference,
with an inferred 1.4553415061295973 factor first available on December 31.
The saved Round-7 `u2_events.json` explicitly classifies this as
`large_move_no_action`, with no corroboration. Its manifest explicitly says that
inherited audit tables describe the earlier store. Accepted action arrays already
use q1/cash0 and preserve the raw closing move from R$8.31 to R$5.71. The missing
term in the accepted terms table was therefore deliberate, not a forgotten
issuer-confirmed bonus.

The narrow qualification reproduces the old four wealth values exactly from
the rejected factor, the prior Float32 wealth and decimal-cent raw prices. It
then removes that factor under the existing recurrence:

| Final AERI wealth coordinate | Before | After |
| --- | ---: | ---: |
| Open | 0.6838021874 | 0.4698568285 |
| High | 0.6854518652 | 0.4709903896 |
| Low | 0.4322223663 | 0.2969903350 |
| Close | 0.4709903896 | 0.3236287832 |

One retrospective M1 consistency bit becomes true; its corresponding diagnostic
row becomes `completed_action_boundary=false`. The patch is confined to the
last date, preserving every earlier coordinate. Daily features use market data
through t−1, so the first potential consuming decision is January 2, 2025,
outside this program's consumer axis. No 2025 observations were read. The five
previously valid primary-horizon shareholder outcomes ending on December 30
already equal the actual price-only returns and remain exact. The to-close target
stays unsupported: the current prefix has insufficient five-minute return support,
`fast_present=false` and RSS NaN, independently of the old diagnostic. Exact
entry/close endpoints do not override this support requirement.

Two original CVM issuer notices were archived and visually checked. The
[December 9 notice](https://www.rad.cvm.gov.br/ENET/frmExibirArquivoIPEExterno.aspx?NumeroProtocoloEntrega=1312607)
calls December 30 debenture meetings concerning covenants and payment timing.
The [December 10 notice](https://www.rad.cvm.gov.br/ENET/frmExibirArquivoIPEExterno.aspx?NumeroProtocoloEntrega=1314227)
describes preliminary talks about controlling shareholders' holdings. Neither
provides equity bonus/conversion terms. Their preserved CVM receipt clocks are
December 9 07:39 and December 10 23:03, respectively. This evidence does not prove
completeness of all issuer disclosures or establish the cause of the price move.

## Verification, retained attempts and limits

Eighteen affected target/store-audit tests passed in 0.41 seconds; the actual
raw-to-store future-mutation fixture passed separately in 4.58 seconds. The new
fixture covers dated normalization, changed clipping ties and exact earlier
prefixes. Ruff passed. No accounting test suite, benchmark, financial audit,
source census, neural forward or forecast scoring was repeated.

Retained attempts include the AERI audit's initial date-column lookup failure,
the renderer not being on the uv PATH, and the bundled Poppler lacking pdftotext.
The successful retrieval was reused; existing pypdf supplies text and Poppler
supplies the viewed pages. Initial and qualified executed bytes remain archived.
An initial test command named a nonexistent test file and ran zero tests; the
correct affected files are the passing batches above. Web search results outside
the requested historical evidence were not admitted or used in any consumer.

Current amendments still require composition with every earlier native/scalar,
daily/peer, issuer/auxiliary and lending amendment in a separately accepted full
store. Remaining microstructure/options/rebalance/odd-lot histories, upstream
clocks/revisions, other contractual wealth/labels and final tensor/refit contracts
are open. Stage A's adaptive-book lifecycle/execution sensitivities are also open;
C/D have not started. The program and heartbeat remain active.
