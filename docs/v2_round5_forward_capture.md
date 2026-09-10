# Round 5 forward capture stopped

The user explicitly stopped all forward capture on September 10, 2026. This
instruction supersedes the capture requests in the original Round-5 registration
and its supplied scope. No current/future snapshots or manual capture runs are
authorized. At the stop, no capture process or matching Windows scheduled task
was active. The existing Codex task was changed to historical Round-5 continuation
only and will pause on completion; it must not restore daily capture.

The first capture completed in 18.1 seconds on September 10, 2026. It is a raw archive for future work, excluded from the development store and all Round-5 model screens. No paid instance or continuous process is required.

| Family | Captured | First-availability rule | Limit |
|---|---|---|---|
| IBOV, IBXX, SMLL | Configuration, current-day composition, theoretical portfolio and preview; 12 responses | Per-response UTC collection time | Current API cannot reconstruct historical publication vintages. Nonempty preview data can be inactive. |
| DCE iron ore | 5 exact unexpired contracts, 1,023 one-minute rows each | Raw vendor timestamps preserved; collection time bounds first capture | Vendor roster is a subset of exchange contracts; no `I0` stitching. |
| SHFE rebar | 6 exact contracts, 1,023 rows each | Same | Raw OHLCV/OI, including night-session timestamps. |
| SHFE HRC | 5 exact contracts, 1,023 rows each | Same | No filling or assumed completed last bar. |
| SHFE pulp | 4 exact contracts, 1,023 rows each | Same | Exchange trade-date/night-session assignment still needs a future consumer audit. |

The source endpoints were verified from the [official B3 index application](https://sistemaswebb3-listados.b3.com.br/indexPage/day/IBOV?language=pt-br) and the [vendor's futures data SDK](https://finance.sina.com.cn/sinafinancesdk/js/datas/k.js). Contract rosters come from the corresponding vendor product pages, such as [iron ore](https://finance.sina.com.cn/futures/quotes/I0.shtml) and [rebar](https://finance.sina.com.cn/futures/quotes/RB0.shtml). Continuous and expired symbols are excluded from minute requests. Raw source HTML/JS and proof responses are retained in the Round-5 index audit's `capture_discovery` directory.

The first output is `D:\quant-data\b3\interim\round5_data_20260910T140442Z\forward_capture\20260910T152200661738Z`. Its manifest SHA-256 is `a5548985698b677ba89a881de135e6be8a08190dc58f6f4e113ed943fec70d63`. The 36 source responses preserve raw bytes, URLs, SHA-256 hashes and UTC retrieval timestamps. All three current index configurations had inactive preview flags even though their preview endpoints returned 76/100/104 old constituent rows. The archive preserves that distinction.

The initial minute responses span September 8 morning through September 10 at
23:00 Shanghai, with some illiquid contracts reaching further back. That one
snapshot does not claim complete announcement or intraday history. Four focused
tests originally checked exact-contract roster selection, missing-minute
preservation, complete index pagination and inactive preview interpretation.
The first snapshot and its source hashes remain immutable in their original
location. No additional snapshot is scheduled or requested.
