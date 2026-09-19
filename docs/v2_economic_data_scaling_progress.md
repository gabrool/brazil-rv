# Economic accounting, data audit and scaling progress

2026-09-19: user authorized accounting repairs, a deep source-to-model audit,
additional historical B3 retrieval and economic-primary attention/GRU capacity
follow-ups. Read `research/preregistrations/v2_economic_data_scaling.md` after each
stage. Foundation outcomes remain sealed at the canonical foundation pointer.

Initial inspection: the account supports only one scalar successor and rejects a
successor already held; multi-leg Copel delivery and mixed contractual consideration
need explicit support. Borrow charges currently use each day's marked exposure and
latest causal archive rate, rather than carrying a fixed contract reference/rate.
These are model assumptions to reconcile with the dated B3 contract, not yet fixed.
Published outstanding lending balances are used as shortability evidence and are
not proof of free locate capacity. The user specified a likely retail-adjacent
account and zero brokerage; exchange and borrowing costs remain separate.

No new GPU fit has launched. Next: source-backed accounting/borrow specification,
source-to-store audit map and prioritized reproducible defect evidence, then fixes
with matched replay. Preserve held-out consumer restrictions and immutable stores.

First source-backed repair: BDI registered-loan extraction now reconstructs printed
wrapped dates, tickers and ISIN check digits, and unambiguously separated thousands
groups where quantity/BRL columns touch. Inspection of 2024-11-27 page 67 confirmed
the physical table; the complete bulletin reconciles 1,632 printed numerical rows.
The arbitrary 100-row minimum is replaced by printed-row reconciliation so small
complete tables survive and partial modality extraction cannot silently bias rates.
Four regression cases plus existing lending tests pass (14 targeted tests).

The active Round-6 economics pointer still references the original lending archive:
no new Round-5 rate vintage was admitted. That archive starts actual rates on
2023-07-10, with a 2% pre-history placeholder in the consumer despite contradictory
older manifest prose. Existing books retain that source. `ops/audit_lending_sources.py`
reparses the admitted-rate era and all cached later-2024 PDFs, plus quarterly earlier
source checks, into a separate audit directory with per-PDF hashes and failures.
Recovered rows are not yet admitted into features or accounting. Zero-flow printed
carried rates do not become fresh trades. Brokerage is assumed zero; published
taker/donor rates remain separately observable proxies.
