from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl
from numpy.typing import NDArray


SCALAR_ACTION_TYPES = frozenset(
    {
        "split",
        "reverse_split",
        "bonus",
        "simple_conversion",
        "dividend",
        "jcp",
        "cash_distribution",
        "cancellation",
    }
)
COMPLEX_ACTION_TYPES = frozenset(
    {"subscription_rights", "rights", "spinoff", "merger", "multi_claim_merger"}
)
ACTION_TYPES = SCALAR_ACTION_TYPES | COMPLEX_ACTION_TYPES
VERIFIED_ACTION_SCHEMA = {
    "action_type": pl.String,
    "isin": pl.String,
    "issuer_id": pl.String,
    "effective_date": pl.Date,
    "ex_date": pl.Date,
    "payment_date": pl.Date,
    "announced_at": pl.Datetime("us", "UTC"),
    "available_at": pl.Datetime("us", "UTC"),
    "shares_per_prior_share": pl.Float64,
    "cash_per_prior_share": pl.Float64,
    "currency": pl.String,
    "source": pl.String,
    "evidence": pl.String,
    "coverage_status": pl.String,
    "resulting_isin": pl.String,
    "sequence": pl.Int32,
    "resolved": pl.Boolean,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class DetectedActionResult:
    """Audit-only COTAHIST action classification on the daily panel.

    These realized price/quantity and DISMES diagnostics must never supply an
    economic adjustment factor or a cash entitlement.
    """

    event_candidate: NDArray[np.bool_]
    split_event: NDArray[np.bool_]
    cash_event: NDArray[np.bool_]
    ambiguous_event: NDArray[np.bool_]
    price_jump_anomaly_mask: NDArray[np.bool_]
    price_ratio: NDArray[np.float64]
    quantity_ratio: NDArray[np.float64]


@dataclass(frozen=True)
class VerifiedActionTerm:
    """One evidenced contractual action, expressed in pre-event share units.

    ``shares_per_prior_share`` (q) and ``cash_per_prior_share`` (d) apply once
    on the ex/effective session.  ``sequence`` documents the order when more
    than one action applies to the same claim on the same session.  A later
    term's q and d are expressed per share produced by earlier terms.
    """

    action_type: str
    isin: str
    issuer_id: str | None
    effective_date: date
    ex_date: date
    payment_date: date | None
    announced_at: datetime | None
    available_at: datetime
    shares_per_prior_share: float
    cash_per_prior_share: float
    currency: str
    source: str
    evidence: str
    coverage_status: str
    resulting_isin: str | None = None
    sequence: int = 0
    resolved: bool = True

    def __post_init__(self) -> None:
        if self.action_type not in ACTION_TYPES:
            raise ValueError(f"unsupported action type: {self.action_type}")
        if not self.isin or not self.source or not self.evidence:
            raise ValueError("verified actions require ISIN, source, and evidence")
        if self.currency not in {"BRL", "R$"} or not self.coverage_status:
            raise ValueError(
                "verified actions require BRL currency and coverage status"
            )
        if self.sequence < 0:
            raise ValueError("action sequence must be non-negative")
        _require_aware_timestamp(self.available_at, "available_at")
        if self.announced_at is not None:
            _require_aware_timestamp(self.announced_at, "announced_at")
            if self.announced_at > self.available_at:
                raise ValueError("announced_at cannot follow available_at")
        if self.payment_date is not None and self.payment_date < self.ex_date:
            raise ValueError("payment_date cannot precede ex_date")
        q = float(self.shares_per_prior_share)
        d = float(self.cash_per_prior_share)
        if not np.isfinite(q) or q < 0.0 or not np.isfinite(d) or d < 0.0:
            raise ValueError("q and d must be finite and non-negative")
        if q == 0.0 and self.action_type != "cancellation":
            raise ValueError("only a verified cancellation may have q=0")
        if self.action_type in COMPLEX_ACTION_TYPES and self.resolved:
            raise ValueError("complex actions require an explicit non-scalar mapping")
        if self.resolved and self.action_type == "simple_conversion":
            if not self.resulting_isin or self.resulting_isin == self.isin:
                raise ValueError("simple conversions require a distinct resulting ISIN")
        elif self.resulting_isin is not None:
            raise ValueError("resulting_isin is valid only for a simple conversion")
        if not self.resolved and (q != 1.0 or d != 0.0):
            raise ValueError("unresolved actions cannot carry economic fallback terms")
        if (
            self.resolved
            and (q != 1.0 or self.action_type == "simple_conversion")
            and d != 0.0
            and self.effective_date != self.ex_date
        ):
            raise ValueError(
                "combined conversion/share and cash terms require one common event session"
            )


@dataclass(frozen=True)
class AlignedActionTerms:
    """Verified scalar terms aligned to an exchange-session/name grid."""

    shares_per_prior_share: NDArray[np.float64]
    cash_per_prior_share: NDArray[np.float64]
    session_resolved: NDArray[np.bool_]
    has_action: NDArray[np.bool_]
    successor_index: NDArray[np.int64] | None = None


@dataclass(frozen=True)
class ShareholderWealthOHLC:
    """Coherent reinvest-at-close wealth-index OHLC for volatility inputs."""

    open: NDArray[np.float64]
    high: NDArray[np.float64]
    low: NDArray[np.float64]
    close: NDArray[np.float64]
    valid: NDArray[np.bool_]


def _require_aware_timestamp(value: datetime, field: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")


def validate_verified_action_terms(
    terms: Sequence[VerifiedActionTerm],
) -> tuple[VerifiedActionTerm, ...]:
    """Validate ordering/uniqueness and return a deterministic action sequence."""

    checked = tuple(
        sorted(
            terms,
            key=lambda term: (
                term.isin,
                min(term.ex_date, term.effective_date),
                term.sequence,
                term.action_type,
            ),
        )
    )
    keys: set[tuple[str, date, int]] = set()
    for term in checked:
        event_date = (
            term.effective_date
            if term.shares_per_prior_share != 1.0
            or term.action_type == "simple_conversion"
            else term.ex_date
        )
        key = (term.isin, event_date, term.sequence)
        if key in keys:
            raise ValueError("action sequence must be unique per ISIN/event session")
        keys.add(key)
    return checked


def verified_action_terms_to_table(
    terms: Sequence[VerifiedActionTerm],
) -> pl.DataFrame:
    """Serialize verified terms without dropping evidence or timing fields."""

    rows = [
        {
            "action_type": term.action_type,
            "isin": term.isin,
            "issuer_id": term.issuer_id,
            "effective_date": term.effective_date,
            "ex_date": term.ex_date,
            "payment_date": term.payment_date,
            "announced_at": (
                term.announced_at.astimezone(timezone.utc)
                if term.announced_at is not None
                else None
            ),
            "available_at": term.available_at.astimezone(timezone.utc),
            "shares_per_prior_share": term.shares_per_prior_share,
            "cash_per_prior_share": term.cash_per_prior_share,
            "currency": term.currency,
            "source": term.source,
            "evidence": term.evidence,
            "coverage_status": term.coverage_status,
            "resulting_isin": term.resulting_isin,
            "sequence": term.sequence,
            "resolved": term.resolved,
        }
        for term in validate_verified_action_terms(terms)
    ]
    return pl.DataFrame(rows, schema=VERIFIED_ACTION_SCHEMA)


def verified_action_terms_from_table(
    actions: pl.DataFrame,
) -> tuple[VerifiedActionTerm, ...]:
    """Load the complete canonical schema and re-run contractual validation."""

    missing = set(VERIFIED_ACTION_SCHEMA) - set(actions.columns)
    if missing:
        raise ValueError(f"verified action columns missing: {sorted(missing)}")
    checked = actions.select(list(VERIFIED_ACTION_SCHEMA)).cast(VERIFIED_ACTION_SCHEMA)
    return validate_verified_action_terms(
        tuple(VerifiedActionTerm(**row) for row in checked.iter_rows(named=True))
    )


def provider_actions_to_verified_terms(
    actions: pl.DataFrame,
) -> tuple[VerifiedActionTerm, ...]:
    """Convert the sealed provider action rows into explicit contractual terms.

    The historical acquisition archive does not contain announcement or payment
    timestamps.  Its ``fetched_at`` field is an acquisition timestamp (stored by
    the legacy writer as naive UTC), not evidence that the event was known at the
    historical decision.  It is therefore preserved only as ``available_at``
    provenance.  The terms may settle retrospective outcomes; they never rewrite
    decision-time features before their recorded availability.

    Scalar split/cash observations are usable as provider-evidenced terms.
    Complex actions, cancellations, and conversions without a resulting claim
    remain unresolved rather than being forced through a scalar formula.
    """

    checked = validate_action_table(actions)
    terms: list[VerifiedActionTerm] = []
    for sequence, row in enumerate(checked.iter_rows(named=True)):
        fetched = row.get("fetched_at")
        if not isinstance(fetched, datetime):
            raise ValueError("provider action rows require fetched_at provenance")
        if fetched.tzinfo is None or fetched.utcoffset() is None:
            fetched = fetched.replace(tzinfo=timezone.utc)
        else:
            fetched = fetched.astimezone(timezone.utc)
        action_type = str(row["action_type"])
        unresolved = bool(row["unresolved"])
        scalar_supported = action_type in {
            "split",
            "reverse_split",
            "bonus",
            "dividend",
            "jcp",
            "cash_distribution",
        }
        resolved = scalar_supported and not unresolved
        q = (
            float(row["split_factor"])
            if resolved and action_type in {"split", "reverse_split", "bonus"}
            else 1.0
        )
        d = (
            float(row["cash_distribution_brl"])
            if resolved and action_type in {"dividend", "jcp", "cash_distribution"}
            else 0.0
        )
        source = str(row.get("source") or "sealed_provider_actions")
        ticker = str(row.get("source_ticker") or "unknown")
        ex_date = _as_date(row["ex_date"])
        terms.append(
            VerifiedActionTerm(
                action_type=action_type,
                isin=str(row["isin"]),
                issuer_id=None,
                effective_date=ex_date,
                ex_date=ex_date,
                payment_date=None,
                announced_at=None,
                available_at=fetched,
                shares_per_prior_share=q,
                cash_per_prior_share=d,
                currency="BRL",
                source=source,
                evidence=f"sealed provider action row; source_ticker={ticker}",
                coverage_status="provider_segment_observed",
                sequence=sequence,
                resolved=resolved,
            )
        )
    return validate_verified_action_terms(terms)


def verified_conversion_terms_from_links(
    links: pl.DataFrame,
) -> tuple[VerifiedActionTerm, ...]:
    """Convert source-audited ISIN-link rows into contractual action terms.

    The identity foundation is responsible for candidate/allowlist matching.
    This boundary preserves the verified conversion ratio, cash, effective
    session, first-known timestamp, and evidence in the one canonical action
    representation used by targets and accounting.
    """

    if links.is_empty():
        return ()
    required = {
        "predecessor_isin",
        "successor_isin",
        "effective_date",
        "first_known_at",
        "shares_received_per_prior_share",
        "cash_entitlement_per_prior_share",
        "currency",
        "source",
        "evidence_sha256",
    }
    missing = required - set(links.columns)
    if missing:
        raise ValueError(
            f"verified ISIN conversions are missing columns: {sorted(missing)}"
        )
    terms: list[VerifiedActionTerm] = []
    for row in links.sort(
        "effective_date", "predecessor_isin", "successor_isin"
    ).iter_rows(named=True):
        first_known = row["first_known_at"]
        effective = row["effective_date"]
        evidence_sha256 = str(row["evidence_sha256"])
        if not isinstance(first_known, datetime):
            raise ValueError("verified ISIN conversions require first_known_at")
        _require_aware_timestamp(first_known, "first_known_at")
        if not isinstance(effective, date):
            raise ValueError("verified ISIN conversions require effective_date")
        if len(evidence_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in evidence_sha256
        ):
            raise ValueError("verified ISIN conversion evidence must be a SHA-256")
        terms.append(
            VerifiedActionTerm(
                action_type="simple_conversion",
                isin=str(row["predecessor_isin"]),
                issuer_id=None,
                effective_date=effective,
                ex_date=effective,
                payment_date=None,
                announced_at=None,
                available_at=first_known,
                shares_per_prior_share=float(row["shares_received_per_prior_share"]),
                cash_per_prior_share=float(row["cash_entitlement_per_prior_share"]),
                currency=str(row["currency"]),
                source=str(row["source"]),
                evidence=f"sha256:{evidence_sha256}",
                coverage_status="verified_isin_conversion_allowlist",
                resulting_isin=str(row["successor_isin"]),
            )
        )
    return validate_verified_action_terms(terms)


def action_coverage_resolved_mask(
    acquisition_audit: pl.DataFrame,
    dates: Sequence[date | np.datetime64],
    isins: Sequence[str],
) -> NDArray[np.bool_]:
    """Map *economically complete* source segments onto the session/name grid.

    A successful provider request proves only that its advertised action
    taxonomy was queried.  It does not prove that every split, bonus, JCP,
    right, conversion, or cash entitlement is represented.  Consequently a
    segment resolves no-action cells only when its acquisition audit carries
    ``economic_terms_complete=True`` from independently verified source
    evidence.  Missing/false flags are the safe default and leave the segment
    unresolved.  Explicit verified action rows remain usable on their own
    event cells when aligned below.
    """

    required = {"isin", "first_date", "last_date", "status"}
    if not required.issubset(acquisition_audit.columns):
        raise ValueError(
            "action acquisition audit columns missing: "
            f"{sorted(required - set(acquisition_audit.columns))}"
        )
    normalized_dates = np.asarray(dates, dtype="datetime64[D]")
    if normalized_dates.ndim != 1:
        raise ValueError("action coverage dates must be one-dimensional")
    isin_lookup = {value: index for index, value in enumerate(isins)}
    success = np.zeros((len(normalized_dates), len(isins)), dtype=np.bool_)
    failure = np.zeros_like(success)
    for row in acquisition_audit.iter_rows(named=True):
        name = isin_lookup.get(str(row["isin"]))
        if name is None:
            continue
        first = np.datetime64(_as_date(row["first_date"]), "D")
        last = np.datetime64(_as_date(row["last_date"]), "D")
        if last < first:
            raise ValueError("action acquisition segment ends before it starts")
        covered = (normalized_dates >= first) & (normalized_dates <= last)
        is_failure = str(row["status"]).casefold() == "failed"
        economically_complete = bool(row.get("economic_terms_complete") or False)
        if is_failure:
            failure[covered, name] = True
        elif economically_complete:
            success[covered, name] = True
    return success & ~failure


def align_verified_action_terms(
    terms: Sequence[VerifiedActionTerm],
    dates: Sequence[date | np.datetime64],
    isins: Sequence[str],
    *,
    coverage_resolved: NDArray[np.bool_],
) -> AlignedActionTerms:
    """Align retrospective contractual terms for outcomes and accounting.

    ``coverage_resolved`` is mandatory: an empty action table alone does not
    prove that a name/session had no action.  Multiple terms are applied in
    their declared sequence and collapsed to q/d in the session's opening
    share units.  This retrospective alignment may use terms acquired after an
    event, so it must not be used for historical decision-time features; use
    :func:`align_decision_known_action_terms` for that purpose.
    """

    return _align_verified_action_terms(
        terms,
        dates,
        isins,
        coverage_resolved=coverage_resolved,
        decision_timestamps=None,
    )


def align_decision_known_action_terms(
    terms: Sequence[VerifiedActionTerm],
    dates: Sequence[date | np.datetime64],
    isins: Sequence[str],
    *,
    coverage_resolved: NDArray[np.bool_],
    decision_timestamps: Sequence[datetime],
) -> AlignedActionTerms:
    """Align only action components known by each historical decision.

    A term acquired after its event is deliberately absent from that earlier
    snapshot.  Retrospective settlement callers use
    :func:`align_verified_action_terms` instead.  Keeping the two entry points
    distinct prevents later action evidence from rewriting historical model
    inputs or their masks.
    """

    cutoffs = tuple(decision_timestamps)
    if len(cutoffs) != len(dates):
        raise ValueError("decision timestamps must align the action date axis")
    for cutoff in cutoffs:
        _require_aware_timestamp(cutoff, "decision timestamp")
    if any(left >= right for left, right in zip(cutoffs, cutoffs[1:], strict=False)):
        raise ValueError("decision timestamps must be strictly chronological")
    return _align_verified_action_terms(
        terms,
        dates,
        isins,
        coverage_resolved=coverage_resolved,
        decision_timestamps=cutoffs,
    )


def _align_verified_action_terms(
    terms: Sequence[VerifiedActionTerm],
    dates: Sequence[date | np.datetime64],
    isins: Sequence[str],
    *,
    coverage_resolved: NDArray[np.bool_],
    decision_timestamps: Sequence[datetime] | None,
) -> AlignedActionTerms:

    normalized_dates = tuple(
        value.astype(object) if isinstance(value, np.datetime64) else value
        for value in dates
    )
    shape = (len(normalized_dates), len(isins))
    resolved = np.asarray(coverage_resolved, dtype=np.bool_)
    if resolved.shape != shape:
        raise ValueError("coverage_resolved must align [date, name]")
    resolved = resolved.copy()
    date_lookup = {value: index for index, value in enumerate(normalized_dates)}
    isin_lookup = {value: index for index, value in enumerate(isins)}
    grouped: dict[tuple[int, int], list[tuple[int, float, float, bool, int]]] = {}
    has_action = np.zeros(shape, dtype=np.bool_)
    for term in validate_verified_action_terms(terms):
        name = isin_lookup.get(term.isin)
        if name is None:
            continue
        successor = (
            isin_lookup.get(term.resulting_isin)
            if term.resulting_isin is not None
            else name
        )
        if term.resulting_isin is not None and successor is None:
            raise ValueError("a resolved conversion successor is outside the ISIN axis")
        components: list[tuple[date, float, float, int]] = []
        if term.action_type == "simple_conversion":
            components.append(
                (
                    term.effective_date,
                    term.shares_per_prior_share,
                    term.cash_per_prior_share,
                    int(successor),
                )
            )
        elif term.shares_per_prior_share != 1.0 and term.cash_per_prior_share != 0.0:
            components.append(
                (
                    term.effective_date,
                    term.shares_per_prior_share,
                    term.cash_per_prior_share,
                    name,
                )
            )
        else:
            if term.shares_per_prior_share != 1.0:
                components.append(
                    (
                        term.effective_date,
                        term.shares_per_prior_share,
                        0.0,
                        int(successor),
                    )
                )
            if term.cash_per_prior_share != 0.0:
                components.append((term.ex_date, 1.0, term.cash_per_prior_share, name))
            if not components:
                components.append((term.ex_date, 1.0, 0.0, int(successor)))
        for event_date, q, d, next_name in components:
            day = date_lookup.get(event_date)
            if day is None:
                continue
            if (
                decision_timestamps is not None
                and term.available_at > decision_timestamps[day]
            ):
                # The retrospective archive proves that an action affected this
                # session, but its contractual terms were unavailable at the
                # historical decision.  Keep it out of the decision-known
                # action signal while explicitly breaking the unit/wealth
                # chain; treating the absent terms as q=1, d=0 would silently
                # manufacture a known cross-session return.
                resolved[day, name] = False
                continue
            has_action[day, name] = True
            grouped.setdefault((day, name), []).append(
                (term.sequence, float(q), float(d), term.resolved, next_name)
            )

    q_aligned = np.ones(shape, dtype=np.float64)
    d_aligned = np.zeros(shape, dtype=np.float64)
    successor_aligned = np.broadcast_to(
        np.arange(len(isins), dtype=np.int64), shape
    ).copy()
    for (day, name), rows in grouped.items():
        # A source-verified term resolves this event even when the broader
        # acquisition audit cannot prove that every no-action session in the
        # surrounding segment was observed.  Coverage remains the authority
        # only for cells without an explicit term.
        resolved[day, name] = True
        q_total = 1.0
        d_total = 0.0
        next_name = name
        for _, q, d, row_resolved, row_successor in sorted(rows):
            if not row_resolved:
                resolved[day, name] = False
                continue
            if row_successor != name:
                if next_name != name:
                    raise ValueError("multiple same-session conversions are not scalar")
                next_name = row_successor
            d_total += q_total * d
            q_total *= q
        q_aligned[day, name] = q_total
        d_aligned[day, name] = d_total
        successor_aligned[day, name] = next_name
    return AlignedActionTerms(
        q_aligned, d_aligned, resolved, has_action, successor_aligned
    )


def align_action_payment_sessions(
    terms: Sequence[VerifiedActionTerm],
    dates: Sequence[date | np.datetime64],
    isins: Sequence[str],
) -> NDArray[np.int64]:
    """Align each cash-bearing action to its first payable session.

    Values are absolute positions on ``dates``. ``-1`` means that the source
    does not establish a payment date. A known payment after the available
    calendar is represented by ``len(dates)`` so a bounded evaluation window
    retains the claim rather than silently treating it as unknown or paid.

    The scalar action panel collapses same-session terms into one cash amount.
    That is only lossless when those terms share the same payment date; mixed
    payment dates are therefore rejected instead of sweeping unrelated claims
    together later in the ledger.
    """

    normalized_dates = np.asarray(dates, dtype="datetime64[D]")
    if normalized_dates.ndim != 1 or normalized_dates.size == 0:
        raise ValueError("payment alignment requires a nonempty date axis")
    if np.any(normalized_dates[1:] <= normalized_dates[:-1]):
        raise ValueError("payment alignment dates must be strictly chronological")
    isin_lookup = {value: index for index, value in enumerate(isins)}
    if len(isin_lookup) != len(isins):
        raise ValueError("payment alignment ISIN axis must be unique")

    output = np.full((normalized_dates.size, len(isins)), -1, dtype=np.int64)
    payment_by_event: dict[tuple[int, int], date | None] = {}
    for term in validate_verified_action_terms(terms):
        if not term.resolved or term.cash_per_prior_share == 0.0:
            continue
        name = isin_lookup.get(term.isin)
        if name is None:
            continue
        event = int(
            np.searchsorted(
                normalized_dates, np.datetime64(term.ex_date, "D"), side="left"
            )
        )
        if event >= normalized_dates.size or normalized_dates[event] != np.datetime64(
            term.ex_date, "D"
        ):
            continue
        key = (event, name)
        prior = payment_by_event.get(key, term.payment_date)
        if key in payment_by_event and prior != term.payment_date:
            raise ValueError(
                "same-session cash actions with different payment dates cannot "
                "be collapsed into one scalar action cell"
            )
        payment_by_event[key] = term.payment_date

    for (event, name), payment_date in payment_by_event.items():
        if payment_date is None:
            continue
        payment = int(
            np.searchsorted(
                normalized_dates,
                np.datetime64(payment_date, "D"),
                side="left",
            )
        )
        if payment < event:
            raise ValueError("action payment session cannot precede its ex session")
        output[event, name] = payment
    return output


def apply_contractual_action(
    shares: float | NDArray[np.floating],
    cash: float | NDArray[np.floating],
    *,
    shares_per_prior_share: float | NDArray[np.floating],
    cash_per_prior_share: float | NDArray[np.floating],
) -> tuple[float | NDArray[np.float64], float | NDArray[np.float64]]:
    """Apply one scalar action to signed holdings; payment later is a no-op.

    Cash entitlement is computed from pre-action signed shares, so a short
    position naturally incurs the corresponding distribution obligation.
    """

    share_array, cash_array, q, d = np.broadcast_arrays(
        np.asarray(shares, dtype=np.float64),
        np.asarray(cash, dtype=np.float64),
        np.asarray(shares_per_prior_share, dtype=np.float64),
        np.asarray(cash_per_prior_share, dtype=np.float64),
    )
    if (
        not np.isfinite(share_array).all()
        or not np.isfinite(cash_array).all()
        or not np.isfinite(q).all()
        or not np.isfinite(d).all()
        or (q < 0.0).any()
        or (d < 0.0).any()
    ):
        raise ValueError(
            "action holdings and terms must be finite, with q/d non-negative"
        )
    new_cash = cash_array + share_array * d
    new_shares = share_array * q
    if new_shares.ndim == 0:
        return float(new_shares), float(new_cash)
    return new_shares, new_cash


def build_shareholder_wealth_ohlc(
    raw_open: NDArray[np.floating],
    raw_high: NDArray[np.floating],
    raw_low: NDArray[np.floating],
    raw_close: NDArray[np.floating],
    observed: NDArray[np.bool_],
    actions: AlignedActionTerms,
) -> ShareholderWealthOHLC:
    """Construct economically coherent OHLC for Yang--Zhang inputs.

    On session u the recurrence is
    ``W_X,u = W_C,u-1 * (q_u * X_u + d_u) / P_C,u-1``.  A missing or
    unresolved transition is left invalid rather than bridged; the index may
    restart on a later ordinary session so subsequent local returns remain
    usable.  The caller performs the single decision-time lag of the resulting
    volatility estimate.
    """

    shape = np.asarray(raw_close).shape
    outputs = [np.empty(shape, dtype=np.float64) for _ in range(4)]
    valid = np.empty(shape, dtype=np.bool_)
    build_shareholder_wealth_ohlc_into(
        raw_open,
        raw_high,
        raw_low,
        raw_close,
        observed,
        actions,
        wealth_open=outputs[0],
        wealth_high=outputs[1],
        wealth_low=outputs[2],
        wealth_close=outputs[3],
        wealth_valid=valid,
    )
    return ShareholderWealthOHLC(*outputs, valid)


def build_shareholder_wealth_ohlc_into(
    raw_open: NDArray[np.floating],
    raw_high: NDArray[np.floating],
    raw_low: NDArray[np.floating],
    raw_close: NDArray[np.floating],
    observed: NDArray[np.bool_],
    actions: AlignedActionTerms,
    *,
    wealth_open: NDArray[np.floating],
    wealth_high: NDArray[np.floating],
    wealth_low: NDArray[np.floating],
    wealth_close: NDArray[np.floating],
    wealth_valid: NDArray[np.bool_],
) -> None:
    """Write shareholder-wealth OHLC directly to caller-owned storage."""

    open_, high, low, close = (
        np.asarray(value) for value in (raw_open, raw_high, raw_low, raw_close)
    )
    seen = np.asarray(observed, dtype=np.bool_)
    shape = close.shape
    action_arrays = (
        actions.shares_per_prior_share,
        actions.cash_per_prior_share,
        actions.session_resolved,
    )
    if close.ndim != 2 or any(
        value.shape != shape for value in (open_, high, low, seen)
    ):
        raise ValueError("raw OHLC and observed must align [date, name]")
    if any(np.asarray(value).shape != shape for value in action_arrays):
        raise ValueError("aligned action arrays do not match OHLC")
    q = np.asarray(actions.shares_per_prior_share, dtype=np.float64)
    d = np.asarray(actions.cash_per_prior_share, dtype=np.float64)
    resolved = np.asarray(actions.session_resolved, dtype=np.bool_)
    successor = (
        np.broadcast_to(np.arange(shape[1], dtype=np.int64), shape)
        if actions.successor_index is None
        else np.asarray(actions.successor_index, dtype=np.int64)
    )
    if (
        successor.shape != shape
        or (successor < 0).any()
        or (successor >= shape[1]).any()
    ):
        raise ValueError("action successor indices are invalid")
    raw_valid = (
        seen
        & np.isfinite(open_)
        & np.isfinite(high)
        & np.isfinite(low)
        & np.isfinite(close)
        & (low > 0.0)
        & (low <= open_)
        & (low <= close)
        & (high >= open_)
        & (high >= close)
    )
    invalid_terms = resolved & (
        ~np.isfinite(q) | ~np.isfinite(d) | (q < 0.0) | (d < 0.0)
    )
    if invalid_terms.any():
        raise ValueError("resolved action terms must have finite non-negative q/d")

    outputs = [
        np.asarray(value)
        for value in (wealth_open, wealth_high, wealth_low, wealth_close)
    ]
    valid = np.asarray(wealth_valid)
    if (
        any(
            value.shape != shape or not np.issubdtype(value.dtype, np.floating)
            for value in outputs
        )
        or valid.shape != shape
        or valid.dtype != np.bool_
    ):
        raise ValueError("shareholder-wealth destinations are misaligned")
    for output in outputs:
        output[...] = 0.0
    valid[...] = False
    nontrivial = (q != 1.0) | (d != 0.0)
    for name in range(shape[1]):
        claim = name
        for day in range(shape[0]):
            prior_claim = claim
            if not resolved[day, prior_claim]:
                continue
            claim = int(successor[day, prior_claim])
            if not raw_valid[day, claim]:
                continue
            if day == 0 or not valid[day - 1, name]:
                if nontrivial[day, prior_claim] or claim != prior_claim:
                    continue
                for output, raw in zip(outputs, (open_, high, low, close), strict=True):
                    output[day, name] = raw[day, claim]
                valid[day, name] = True
                continue
            prior_price = close[day - 1, prior_claim]
            prior_wealth = outputs[3][day - 1, name]
            if not np.isfinite(prior_price) or prior_price <= 0.0:
                continue
            scale = prior_wealth / prior_price
            candidates = []
            for raw in (open_, high, low, close):
                candidates.append(
                    scale
                    * (q[day, prior_claim] * raw[day, claim] + d[day, prior_claim])
                )
            if not all(np.isfinite(value) and value > 0.0 for value in candidates):
                continue
            for output, raw in zip(outputs, (open_, high, low, close), strict=True):
                output[day, name] = scale * (
                    q[day, prior_claim] * raw[day, claim] + d[day, prior_claim]
                )
            valid[day, name] = True


def normalize_yfinance_actions(
    actions: pl.DataFrame,
    *,
    isin: str,
    ticker: str,
    fetched_at: datetime,
) -> pl.DataFrame:
    """Convert yfinance's dividends/splits frame into the canonical action rows."""

    date_column = "Date" if "Date" in actions.columns else "date"
    dividend_column = "Dividends" if "Dividends" in actions.columns else "dividends"
    split_column = (
        "Stock Splits" if "Stock Splits" in actions.columns else "stock_splits"
    )
    required = {date_column, dividend_column, split_column}
    if not required.issubset(actions.columns):
        raise ValueError("yfinance action frame lacks Date/Dividends/Stock Splits")
    rows: list[dict[str, object]] = []
    for raw_date, raw_dividend, raw_split in actions.select(
        date_column, dividend_column, split_column
    ).iter_rows():
        ex_date = raw_date.date() if isinstance(raw_date, datetime) else raw_date
        dividend = _finite_or_zero(raw_dividend)
        split = _finite_or_zero(raw_split)
        if split:
            rows.append(
                {
                    "isin": isin,
                    "ex_date": ex_date,
                    "action_type": "split" if split >= 1 else "reverse_split",
                    "split_factor": split,
                    "cash_distribution_brl": 0.0,
                    "provider_cash_distribution_brl": 0.0,
                    "cash_unit_adjustment_factor": 1.0,
                    "known_date": ex_date,
                    "source_ticker": ticker,
                    "source": "yfinance.actions",
                    "fetched_at": fetched_at,
                    "unresolved": False,
                }
            )
        if dividend:
            rows.append(
                {
                    "isin": isin,
                    "ex_date": ex_date,
                    "action_type": "dividend",
                    "split_factor": 1.0,
                    "cash_distribution_brl": dividend,
                    "provider_cash_distribution_brl": dividend,
                    "cash_unit_adjustment_factor": 1.0,
                    "known_date": ex_date,
                    "source_ticker": ticker,
                    "source": "yfinance.actions",
                    "fetched_at": fetched_at,
                    "unresolved": False,
                }
            )
    schema = {
        "isin": pl.String,
        "ex_date": pl.Date,
        "action_type": pl.String,
        "split_factor": pl.Float64,
        "cash_distribution_brl": pl.Float64,
        "provider_cash_distribution_brl": pl.Float64,
        "cash_unit_adjustment_factor": pl.Float64,
        "known_date": pl.Date,
        "source_ticker": pl.String,
        "source": pl.String,
        "fetched_at": pl.Datetime("us"),
        "unresolved": pl.Boolean,
    }
    return pl.DataFrame(rows, schema=schema)


def normalize_cached_action_schema(
    frame: pl.DataFrame,
    *,
    isin: str,
    ticker: str,
    fetched_at: datetime,
) -> pl.DataFrame:
    """Upgrade legacy immutable cache rows in memory without rewriting them."""

    raw_columns = {"date", "dividends", "stock_splits"}
    provider_columns = {"Date", "Dividends", "Stock Splits"}
    if raw_columns.issubset(frame.columns) or provider_columns.issubset(frame.columns):
        return normalize_yfinance_actions(
            frame, isin=isin, ticker=ticker, fetched_at=fetched_at
        )
    required = {
        "isin",
        "ex_date",
        "action_type",
        "split_factor",
        "cash_distribution_brl",
    }
    if not required.issubset(frame.columns):
        raise ValueError(
            "legacy action cache columns missing: "
            f"{sorted(required - set(frame.columns))}"
        )
    additions: list[pl.Expr] = []
    if "provider_cash_distribution_brl" not in frame.columns:
        additions.append(
            pl.col("cash_distribution_brl")
            .cast(pl.Float64)
            .alias("provider_cash_distribution_brl")
        )
    if "cash_unit_adjustment_factor" not in frame.columns:
        additions.append(pl.lit(1.0).alias("cash_unit_adjustment_factor"))
    if "known_date" not in frame.columns:
        additions.append(pl.col("ex_date").cast(pl.Date).alias("known_date"))
    if "source_ticker" not in frame.columns:
        additions.append(pl.lit(ticker).alias("source_ticker"))
    if "source" not in frame.columns:
        additions.append(pl.lit("yfinance.actions").alias("source"))
    if "fetched_at" not in frame.columns:
        additions.append(pl.lit(fetched_at).alias("fetched_at"))
    if "unresolved" not in frame.columns:
        additions.append(pl.lit(False).alias("unresolved"))
    upgraded = frame.with_columns(additions) if additions else frame
    columns = (
        "isin",
        "ex_date",
        "action_type",
        "split_factor",
        "cash_distribution_brl",
        "provider_cash_distribution_brl",
        "cash_unit_adjustment_factor",
        "known_date",
        "source_ticker",
        "source",
        "fetched_at",
        "unresolved",
    )
    return upgraded.select(columns).cast(
        {
            "isin": pl.String,
            "ex_date": pl.Date,
            "action_type": pl.String,
            "split_factor": pl.Float64,
            "cash_distribution_brl": pl.Float64,
            "provider_cash_distribution_brl": pl.Float64,
            "cash_unit_adjustment_factor": pl.Float64,
            "known_date": pl.Date,
            "source_ticker": pl.String,
            "source": pl.String,
            "fetched_at": pl.Datetime("us"),
            "unresolved": pl.Boolean,
        }
    )


def _empty_yfinance_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "date": pl.Date,
            "dividends": pl.Float64,
            "stock_splits": pl.Float64,
            "price_observed": pl.Boolean,
        }
    )


def _finite_or_zero(value: object) -> float:
    try:
        number = float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0
    return number if np.isfinite(number) else 0.0


def _extract_yfinance_actions(raw: Any, symbol: str) -> pl.DataFrame | None:
    """Extract one symbol from either orientation of yfinance MultiIndex output."""

    if raw is None or raw.empty:
        return None
    frame = raw
    columns = raw.columns
    if getattr(columns, "nlevels", 1) > 1:
        level_zero = set(columns.get_level_values(0))
        level_one = set(columns.get_level_values(1))
        if symbol in level_zero:
            frame = raw[symbol]
        elif symbol in level_one:
            frame = raw.xs(symbol, axis=1, level=1)
        else:
            return None
    if frame.empty:
        return None
    price_columns = [
        column
        for column in ("Open", "High", "Low", "Close", "Adj Close")
        if column in frame.columns
    ]
    price_values = (
        np.asarray(frame[price_columns].to_numpy(), dtype=np.float64)
        if price_columns
        else np.empty((len(frame), 0), dtype=np.float64)
    )
    if not price_columns or not np.isfinite(price_values).any():
        # yfinance keeps failed symbols in a batch MultiIndex, filling their
        # complete price frame with NaN.  That is a provider failure, not proof
        # that a live symbol had genuinely zero actions.
        return None
    rows: list[dict[str, object]] = []
    price_observed = np.isfinite(price_values).any(axis=1)
    for (timestamp, record), has_price in zip(
        frame.iterrows(), price_observed, strict=True
    ):
        if not has_price:
            continue
        dividend = _finite_or_zero(record.get("Dividends", 0.0))
        split = _finite_or_zero(record.get("Stock Splits", 0.0))
        rows.append(
            {
                "date": timestamp.date(),
                "dividends": dividend,
                "stock_splits": split,
                "price_observed": True,
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "date": pl.Date,
            "dividends": pl.Float64,
            "stock_splits": pl.Float64,
            "price_observed": pl.Boolean,
        },
    )


def unadjust_yfinance_cash_distributions(actions: pl.DataFrame) -> pl.DataFrame:
    """Restore Yahoo cash actions to contemporaneous pre-later-split units."""

    checked = validate_action_table(actions)
    if checked.is_empty():
        return checked
    rows: list[dict[str, object]] = []
    for group in checked.partition_by("isin", maintain_order=True):
        group_rows = list(group.sort("ex_date", "action_type").iter_rows(named=True))
        split_by_date: dict[date, float] = {}
        for row in group_rows:
            ex_date = _as_date(row["ex_date"])
            split_by_date[ex_date] = split_by_date.get(ex_date, 1.0) * float(
                row["split_factor"]
            )
        later_factor_by_date: dict[date, float] = {}
        later_factor = 1.0
        for ex_date in sorted(split_by_date, reverse=True):
            later_factor_by_date[ex_date] = later_factor
            later_factor *= split_by_date[ex_date]
        for row in group_rows:
            output = dict(row)
            output.setdefault(
                "provider_cash_distribution_brl", row["cash_distribution_brl"]
            )
            output.setdefault("cash_unit_adjustment_factor", 1.0)
            provider_cash = float(
                row.get("provider_cash_distribution_brl")
                or row["cash_distribution_brl"]
            )
            if str(row["action_type"]) in {"dividend", "jcp"}:
                unit_factor = later_factor_by_date[_as_date(row["ex_date"])]
                output["provider_cash_distribution_brl"] = provider_cash
                output["cash_unit_adjustment_factor"] = unit_factor
                output["cash_distribution_brl"] = provider_cash * unit_factor
            rows.append(output)
    return validate_action_table(pl.DataFrame(rows, infer_schema_length=None))


def _download_yfinance_batch(
    tickers: Sequence[str], *, start: date, end: date
) -> dict[str, pl.DataFrame]:
    """Download one bounded symbol batch; callers isolate and record failures."""

    import yfinance as yf

    symbols = [f"{ticker}.SA" for ticker in tickers]
    raw = yf.download(
        symbols,
        start=start.isoformat(),
        end=end.isoformat(),
        actions=True,
        auto_adjust=False,
        group_by="ticker",
        threads=True,
        progress=False,
    )
    extracted = {
        ticker: _extract_yfinance_actions(raw, symbol)
        for ticker, symbol in zip(tickers, symbols, strict=True)
    }
    return {ticker: frame for ticker, frame in extracted.items() if frame is not None}


def acquire_yfinance_actions(
    security_master: pl.DataFrame,
    cache_dir: Path,
    *,
    refresh: bool = False,
    fetched_at: datetime | None = None,
    batch_size: int = 64,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Fetch and immutably cache yfinance actions for every ticker segment.

    The segment-specific cache key prevents a historical ticker from being bound
    outside its observed ISIN interval. Existing cache files are reused unless
    ``refresh`` is explicit. The returned audit records exact cache hashes.
    """

    required = {"isin", "ticker", "first_date", "last_date"}
    if not required.issubset(security_master.columns):
        raise ValueError(
            f"security master columns missing: {sorted(required - set(security_master.columns))}"
        )
    if (
        security_master.select(
            pl.struct("isin", "ticker", "first_date", "last_date").n_unique()
        ).item()
        != security_master.height
    ):
        raise ValueError("security master contains duplicate ticker segments")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    cache = Path(cache_dir).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    timestamp = fetched_at or datetime.now(timezone.utc)
    timestamp_key = timestamp.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    frames: list[pl.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    rows = list(
        security_master.sort("isin", "first_date", "ticker").iter_rows(named=True)
    )
    pending: list[dict[str, object]] = []
    cached_paths: dict[str, Path] = {}
    identities: dict[str, str] = {}
    for row in rows:
        identity = "|".join(
            (
                str(row["isin"]),
                str(row["ticker"]),
                row["first_date"].isoformat(),
                row["last_date"].isoformat(),
            )
        )
        key = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        identities[identity] = key
        existing = sorted(cache.glob(f"{row['ticker']}_{key}_*.parquet"))
        legacy = cache / f"{row['ticker']}_{key}.parquet"
        if legacy.exists():
            existing.append(legacy)
        if existing and not refresh:
            cached_paths[identity] = sorted(existing)[-1]
        else:
            pending.append(row)

    downloaded: dict[str, pl.DataFrame] = {}
    failures: dict[str, str] = {}
    unique_tickers = sorted({str(row["ticker"]) for row in pending})
    bounds = {
        ticker: (
            min(row["first_date"] for row in pending if row["ticker"] == ticker),
            max(row["last_date"] for row in pending if row["ticker"] == ticker),
        )
        for ticker in unique_tickers
    }
    for offset in range(0, len(unique_tickers), batch_size):
        batch = unique_tickers[offset : offset + batch_size]
        start = min(bounds[ticker][0] for ticker in batch)
        end = max(bounds[ticker][1] for ticker in batch) + timedelta(days=1)
        try:
            downloaded.update(_download_yfinance_batch(batch, start=start, end=end))
        except Exception as error:  # provider failures are recorded per symbol
            for ticker in batch:
                failures[ticker] = f"{type(error).__name__}: {error}"[:1000]

    # A batch may succeed while silently omitting one failed symbol. Retry every
    # missing symbol individually and retain the last provider error in the audit.
    for ticker in unique_tickers:
        if ticker in downloaded:
            continue
        start, last = bounds[ticker]
        for attempt in range(2):
            try:
                result = _download_yfinance_batch(
                    [ticker], start=start, end=last + timedelta(days=1)
                )
                if ticker in result:
                    downloaded[ticker] = result[ticker]
                    failures.pop(ticker, None)
                    break
                failures[ticker] = "symbol missing from provider response"
            except Exception as error:
                failures[ticker] = (
                    f"attempt {attempt + 1}: {type(error).__name__}: {error}"
                )[:1000]

    current_ticker = {
        str(group[0, "isin"]): str(
            group.sort("last_date", descending=True)[0, "ticker"]
        )
        for group in security_master.partition_by("isin", maintain_order=True)
    }
    fallback_rows = [
        row for row in pending if current_ticker[str(row["isin"])] != str(row["ticker"])
    ]
    fallback_downloads: dict[tuple[str, str], pl.DataFrame] = {}
    for isin in sorted({str(row["isin"]) for row in fallback_rows}):
        alias = current_ticker[isin]
        related = [row for row in fallback_rows if str(row["isin"]) == isin]
        start = min(row["first_date"] for row in related)
        end = max(row["last_date"] for row in related) + timedelta(days=1)
        for attempt in range(2):
            try:
                result = _download_yfinance_batch([alias], start=start, end=end)
                if alias in result:
                    fallback_downloads[(isin, alias)] = result[alias]
                    break
            except Exception as error:
                for row in related:
                    failures[str(row["ticker"])] = (
                        f"current-ticker {alias} attempt {attempt + 1}: "
                        f"{type(error).__name__}: {error}"
                    )[:1000]

    for row in rows:
        identity = "|".join(
            (
                str(row["isin"]),
                str(row["ticker"]),
                row["first_date"].isoformat(),
                row["last_date"].isoformat(),
            )
        )
        ticker = str(row["ticker"])
        path = cached_paths.get(identity)
        status = "cache_hit"
        error = None
        query_ticker = ticker
        if path is None:
            raw_frame = downloaded.get(ticker)
            alias = current_ticker[str(row["isin"])]
            if raw_frame is not None:
                raw_frame = raw_frame.filter(
                    pl.col("date").is_between(
                        row["first_date"], row["last_date"], closed="both"
                    )
                )
            alias_frame = fallback_downloads.get((str(row["isin"]), alias))
            if (raw_frame is None or raw_frame.is_empty()) and alias_frame is not None:
                raw_frame = alias_frame
                raw_frame = raw_frame.filter(
                    pl.col("date").is_between(
                        row["first_date"], row["last_date"], closed="both"
                    )
                )
                query_ticker = alias
                status = "downloaded_current_ticker"
            if raw_frame is None:
                error = failures.get(ticker, "symbol missing from provider response")
                audit_rows.append(
                    {
                        "isin": row["isin"],
                        "ticker": ticker,
                        "query_ticker": query_ticker,
                        "first_date": row["first_date"],
                        "last_date": row["last_date"],
                        "cache_path": None,
                        "cache_sha256": None,
                        "action_rows": 0,
                        "status": "failed",
                        "error": error,
                    }
                )
                continue
            if raw_frame.is_empty():
                audit_rows.append(
                    {
                        "isin": row["isin"],
                        "ticker": ticker,
                        "query_ticker": query_ticker,
                        "first_date": row["first_date"],
                        "last_date": row["last_date"],
                        "cache_path": None,
                        "cache_sha256": None,
                        "action_rows": 0,
                        "status": "failed",
                        "error": "no finite provider price inside segment window",
                    }
                )
                continue
            frame = normalize_yfinance_actions(
                raw_frame,
                isin=str(row["isin"]),
                ticker=query_ticker,
                fetched_at=timestamp,
            )
            key = identities[identity]
            path = cache / f"{ticker}_{key}_{timestamp_key}.parquet"
            if path.exists():
                raise FileExistsError(f"immutable action cache already exists: {path}")
            temporary = path.with_suffix(".parquet.tmp")
            frame.write_parquet(temporary)
            temporary.replace(path)
            if status == "downloaded_current_ticker":
                if frame.is_empty():
                    status = "zero_actions_current_ticker"
            else:
                status = "zero_actions" if frame.is_empty() else "downloaded"
        frame = normalize_cached_action_schema(
            pl.read_parquet(path),
            isin=str(row["isin"]),
            ticker=query_ticker,
            fetched_at=timestamp,
        )
        frames.append(frame)
        audit_rows.append(
            {
                "isin": row["isin"],
                "ticker": ticker,
                "query_ticker": query_ticker,
                "first_date": row["first_date"],
                "last_date": row["last_date"],
                "cache_path": str(path),
                "cache_sha256": _sha256(path),
                "action_rows": frame.height,
                "status": status,
                "error": error,
            }
        )
    if frames:
        combined = pl.concat(frames, how="vertical_relaxed")
    else:
        combined = normalize_yfinance_actions(
            _empty_yfinance_frame(),
            isin="BR0000000000",
            ticker="EMPTY",
            fetched_at=timestamp,
        )
    # Yahoo request success is useful acquisition provenance, but this source
    # does not certify a complete Brazilian corporate-action taxonomy.  Keep
    # the distinction machine-readable so downstream economics cannot promote
    # a provider zero to a verified economic no-action interval.
    acquisition = pl.DataFrame(audit_rows).with_columns(
        pl.lit(False, dtype=pl.Boolean).alias("economic_terms_complete")
    )
    return unadjust_yfinance_cash_distributions(combined), acquisition


def validate_action_table(actions: pl.DataFrame) -> pl.DataFrame:
    required = {
        "isin",
        "ex_date",
        "action_type",
        "split_factor",
        "cash_distribution_brl",
        "unresolved",
    }
    if not required.issubset(actions.columns):
        raise ValueError(
            f"action columns missing: {sorted(required - set(actions.columns))}"
        )
    invalid_types = set(actions.get_column("action_type").unique()) - ACTION_TYPES
    if invalid_types:
        raise ValueError(f"unknown action types: {sorted(invalid_types)}")
    if actions.filter(
        (pl.col("split_factor") <= 0)
        | ~pl.col("split_factor").is_finite()
        | (pl.col("cash_distribution_brl") < 0)
        | ~pl.col("cash_distribution_brl").is_finite()
    ).height:
        raise ValueError("action factors/distributions must be finite and non-negative")
    if (
        "known_date" in actions.columns
        and actions.filter(
            pl.col("known_date").is_not_null()
            & (pl.col("known_date") > pl.col("ex_date"))
        ).height
    ):
        raise ValueError("an action cannot be treated as known after its ex-date")
    return actions.sort("isin", "ex_date", "action_type")


def align_action_arrays(
    actions: pl.DataFrame,
    dates: Sequence[date | np.datetime64],
    isins: Sequence[str],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Aggregate action rows onto a date/ISIN axis."""

    checked = validate_action_table(actions)
    normalized_dates = tuple(
        value.astype(object) if isinstance(value, np.datetime64) else value
        for value in dates
    )
    date_lookup = {value: index for index, value in enumerate(normalized_dates)}
    isin_lookup = {value: index for index, value in enumerate(isins)}
    shape = (len(normalized_dates), len(isins))
    split = np.ones(shape, dtype=np.float64)
    cash = np.zeros(shape, dtype=np.float64)
    unresolved = np.zeros(shape, dtype=np.bool_)
    for row in checked.iter_rows(named=True):
        date_index = date_lookup.get(row["ex_date"])
        isin_index = isin_lookup.get(row["isin"])
        if date_index is None or isin_index is None:
            continue
        split[date_index, isin_index] *= float(row["split_factor"])
        cash[date_index, isin_index] += float(row["cash_distribution_brl"])
        unresolved[date_index, isin_index] |= bool(row["unresolved"])
    return split, cash, unresolved


def action_calendar_alignment_table(
    actions: pl.DataFrame,
    dates: Sequence[date | np.datetime64],
    isins: Sequence[str],
) -> pl.DataFrame:
    """Account for action rows not represented on the aligned trading calendar."""

    checked = validate_action_table(actions)
    calendar = {_as_date(value) for value in dates}
    names = set(isins)
    rows = [
        {
            "isin": str(row["isin"]),
            "ex_date": row["ex_date"],
            "action_type": str(row["action_type"]),
            "reason": (
                "off_calendar_ex_date"
                if row["ex_date"] not in calendar
                else "outside_isin_axis"
            ),
        }
        for row in checked.iter_rows(named=True)
        if row["ex_date"] not in calendar or str(row["isin"]) not in names
    ]
    return pl.DataFrame(
        rows,
        schema={
            "isin": pl.String,
            "ex_date": pl.Date,
            "action_type": pl.String,
            "reason": pl.String,
        },
    )


def detect_distribution_changes(
    distribution_number: NDArray[np.floating],
    observed: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    """Mark a COTAHIST distribution-number change at the next observed row."""

    distribution = np.asarray(distribution_number, dtype=np.float64)
    seen = np.asarray(observed, dtype=np.bool_)
    if distribution.shape != seen.shape or distribution.ndim != 2:
        raise ValueError("distribution-number inputs must align [date, name]")
    changed = np.zeros(seen.shape, dtype=np.bool_)
    for name in range(distribution.shape[1]):
        prior: float | None = None
        for day in np.flatnonzero(seen[:, name]):
            current = distribution[day, name]
            if not np.isfinite(current):
                continue
            if prior is not None and current != prior:
                changed[day, name] = True
            prior = float(current)
    return changed


def detect_cotahist_actions(
    raw_close: NDArray[np.floating],
    quantity: NDArray[np.floating],
    distribution_number: NDArray[np.floating],
    observed: NDArray[np.bool_],
    *,
    median_sessions: int = 3,
    split_log_price_threshold: float = 0.08,
    ambiguous_log_price_threshold: float = 0.04,
    volume_continuity_tolerance: float = 0.15,
    large_ratio_tolerance: float = 0.35,
    undocumented_split_fallback: bool = False,
) -> DetectedActionResult:
    """Classify COTAHIST action boundaries causally from ``DISMES`` changes.

    Event ratios compare the current session with up to three strictly prior
    observed sessions. Ordinary price jumps are audit anomalies, never cash
    events. The optional strict fallback promotes only an inverse price/quantity
    jump of at least 30 percent to a split.
    """

    close = np.asarray(raw_close, dtype=np.float64)
    qty = np.asarray(quantity, dtype=np.float64)
    distribution = np.asarray(distribution_number, dtype=np.float64)
    seen = np.asarray(observed, dtype=np.bool_)
    if (
        close.ndim != 2
        or qty.shape != close.shape
        or distribution.shape != close.shape
        or seen.shape != close.shape
    ):
        raise ValueError("COTAHIST action inputs must align [date, name]")
    if median_sessions < 1:
        raise ValueError("median_sessions must be positive")
    if not (
        0.0 < ambiguous_log_price_threshold < split_log_price_threshold
        and volume_continuity_tolerance > 0.0
        and large_ratio_tolerance > volume_continuity_tolerance
    ):
        raise ValueError("COTAHIST action thresholds are invalid")

    distribution_changed = detect_distribution_changes(distribution, seen)
    immediate_price = np.full(close.shape, np.nan, dtype=np.float64)
    adjacent_price = (
        seen[1:]
        & seen[:-1]
        & np.isfinite(close[1:])
        & np.isfinite(close[:-1])
        & (close[1:] > 0)
        & (close[:-1] > 0)
    )
    np.divide(close[1:], close[:-1], out=immediate_price[1:], where=adjacent_price)
    with np.errstate(divide="ignore", invalid="ignore"):
        immediate_log_price = np.log(immediate_price)
    jump_without_distribution = (
        np.isfinite(immediate_log_price)
        & (np.abs(immediate_log_price) > ambiguous_log_price_threshold)
        & ~distribution_changed
    )

    price_ratio = np.full(close.shape, np.nan, dtype=np.float64)
    quantity_ratio = np.full(close.shape, np.nan, dtype=np.float64)
    ratio_candidates = distribution_changed | jump_without_distribution
    for name in range(close.shape[1]):
        prior_close: list[float] = []
        prior_quantity: list[float] = []
        for day in range(close.shape[0]):
            current_valid = bool(
                seen[day, name]
                and np.isfinite(close[day, name])
                and close[day, name] > 0
                and np.isfinite(qty[day, name])
                and qty[day, name] > 0
            )
            if ratio_candidates[day, name] and current_valid and prior_close:
                before_close = float(np.median(prior_close))
                before_quantity = float(np.median(prior_quantity))
                price_ratio[day, name] = close[day, name] / before_close
                quantity_ratio[day, name] = qty[day, name] / before_quantity
            if current_valid:
                prior_close.append(float(close[day, name]))
                prior_quantity.append(float(qty[day, name]))
                if len(prior_close) > median_sessions:
                    del prior_close[0]
                    del prior_quantity[0]

    with np.errstate(divide="ignore", invalid="ignore"):
        log_price = np.log(price_ratio)
        log_quantity = np.log(quantity_ratio)
    comparable = np.isfinite(log_price) & np.isfinite(log_quantity)
    opposite = comparable & (log_price * log_quantity < 0.0)
    tolerance = np.where(
        np.abs(log_price) <= 0.30,
        volume_continuity_tolerance,
        large_ratio_tolerance,
    )
    split_like = (
        comparable
        & (np.abs(log_price) > split_log_price_threshold)
        & opposite
        & (np.abs(log_price + log_quantity) < tolerance)
    )
    documented_split = distribution_changed & split_like
    fallback_split = (
        jump_without_distribution
        & comparable
        & (np.abs(log_price) >= 0.30)
        & opposite
        & (np.abs(log_price + log_quantity) < large_ratio_tolerance)
    )
    if not undocumented_split_fallback:
        fallback_split[...] = False
    split_event = documented_split | fallback_split
    ambiguous_event = (
        distribution_changed
        & ~split_event
        & comparable
        & (np.abs(log_price) > ambiguous_log_price_threshold)
        & (np.abs(log_price) <= split_log_price_threshold)
    )
    cash_event = distribution_changed & ~split_event & ~ambiguous_event
    event_candidate = distribution_changed | fallback_split
    price_jump_anomaly_mask = jump_without_distribution & ~fallback_split
    return DetectedActionResult(
        event_candidate=event_candidate,
        split_event=split_event,
        cash_event=cash_event,
        ambiguous_event=ambiguous_event,
        price_jump_anomaly_mask=price_jump_anomaly_mask,
        price_ratio=price_ratio,
        quantity_ratio=quantity_ratio,
    )


def cotahist_action_classification_table(
    dates: Sequence[date | np.datetime64],
    isins: Sequence[str],
    distribution_changed: NDArray[np.bool_],
    result: DetectedActionResult,
) -> pl.DataFrame:
    changed = np.asarray(distribution_changed, dtype=np.bool_)
    shape = (len(dates), len(isins))
    arrays = (
        result.event_candidate,
        result.split_event,
        result.cash_event,
        result.ambiguous_event,
        result.price_jump_anomaly_mask,
        result.price_ratio,
        result.quantity_ratio,
    )
    if changed.shape != shape or any(
        np.asarray(value).shape != shape for value in arrays
    ):
        raise ValueError("COTAHIST action classification axes are misaligned")
    rows = [
        {
            "date": _as_date(dates[day]),
            "isin": isins[name],
            "distribution_number_changed": bool(changed[day, name]),
            "split_event": bool(result.split_event[day, name]),
            "cash_event": bool(result.cash_event[day, name]),
            "ambiguous_event": bool(result.ambiguous_event[day, name]),
            "price_jump_anomaly": bool(result.price_jump_anomaly_mask[day, name]),
            "price_ratio": (
                float(result.price_ratio[day, name])
                if np.isfinite(result.price_ratio[day, name])
                else None
            ),
            "quantity_ratio": (
                float(result.quantity_ratio[day, name])
                if np.isfinite(result.quantity_ratio[day, name])
                else None
            ),
        }
        for day, name in np.argwhere(
            result.event_candidate | result.price_jump_anomaly_mask
        )
    ]
    return pl.DataFrame(
        rows,
        schema={
            "date": pl.Date,
            "isin": pl.String,
            "distribution_number_changed": pl.Boolean,
            "split_event": pl.Boolean,
            "cash_event": pl.Boolean,
            "ambiguous_event": pl.Boolean,
            "price_jump_anomaly": pl.Boolean,
            "price_ratio": pl.Float64,
            "quantity_ratio": pl.Float64,
        },
    )


def split_review_table(
    dates: Sequence[date],
    isins: Sequence[str],
    detected: NDArray[np.bool_],
    recorded_split_factor: NDArray[np.floating],
) -> pl.DataFrame:
    detected = np.asarray(detected, dtype=np.bool_)
    recorded = np.asarray(recorded_split_factor, dtype=np.float64)
    if detected.shape != recorded.shape or detected.shape != (len(dates), len(isins)):
        raise ValueError("split review axes are misaligned")
    rows: list[dict[str, object]] = []
    for date_index, isin_index in np.argwhere(detected | (recorded != 1.0)):
        has_record = bool(recorded[date_index, isin_index] != 1.0)
        has_detection = bool(detected[date_index, isin_index])
        rows.append(
            {
                "date": dates[date_index],
                "isin": isins[isin_index],
                "detected_jump": has_detection,
                "recorded_split_factor": float(recorded[date_index, isin_index]),
                "agreement": has_detection == has_record,
                "unresolved": has_detection != has_record,
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "date": pl.Date,
            "isin": pl.String,
            "detected_jump": pl.Boolean,
            "recorded_split_factor": pl.Float64,
            "agreement": pl.Boolean,
            "unresolved": pl.Boolean,
        },
    )


def provider_split_detection_audit(
    dates: Sequence[date | np.datetime64],
    isins: Sequence[str],
    detected_split: NDArray[np.bool_],
    provider_actions: pl.DataFrame,
    acquisition_audit: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Measure COTAHIST split detection only where the provider was available."""

    normalized_dates = np.asarray(dates, dtype="datetime64[D]")
    detected = np.asarray(detected_split, dtype=np.bool_)
    shape = (len(normalized_dates), len(isins))
    if detected.shape != shape:
        raise ValueError("provider split audit axes are misaligned")
    provider_factor, _, _ = align_action_arrays(provider_actions, dates, isins)
    provider_split = provider_factor != 1.0
    covered = np.zeros(shape, dtype=np.bool_)
    isin_lookup = {isin: index for index, isin in enumerate(isins)}
    if acquisition_audit is None:
        for isin in provider_actions.get_column("isin").unique().to_list():
            name = isin_lookup.get(str(isin))
            if name is not None:
                covered[:, name] = True
    else:
        required = {"isin", "first_date", "last_date", "status"}
        if not required.issubset(acquisition_audit.columns):
            raise ValueError("corporate-action acquisition audit has wrong schema")
        for row in acquisition_audit.filter(pl.col("status") != "failed").iter_rows(
            named=True
        ):
            name = isin_lookup.get(str(row["isin"]))
            if name is None:
                continue
            first = np.datetime64(_as_date(row["first_date"]), "D")
            last = np.datetime64(_as_date(row["last_date"]), "D")
            covered[:, name] |= (normalized_dates >= first) & (normalized_dates <= last)

    years = normalized_dates.astype("datetime64[Y]").astype(np.int64) + 1970
    periods: list[tuple[str, NDArray[np.bool_]]] = [("all", covered)]
    periods.extend(
        (str(int(year)), covered & (years == year)[:, None])
        for year in sorted(set(years.tolist()))
    )
    rows: list[dict[str, object]] = []
    for period, eligible in periods:
        predicted = detected & eligible
        actual = provider_split & eligible
        true_positive = int((predicted & actual).sum())
        predicted_count = int(predicted.sum())
        actual_count = int(actual.sum())
        rows.append(
            {
                "period": period,
                "covered_name_days": int(eligible.sum()),
                "detected_split_count": predicted_count,
                "provider_split_count": actual_count,
                "true_positive_count": true_positive,
                "precision": (
                    true_positive / predicted_count if predicted_count else None
                ),
                "recall": true_positive / actual_count if actual_count else None,
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "period": pl.String,
            "covered_name_days": pl.Int64,
            "detected_split_count": pl.Int64,
            "provider_split_count": pl.Int64,
            "true_positive_count": pl.Int64,
            "precision": pl.Float64,
            "recall": pl.Float64,
        },
    )


def action_coverage_table(
    actions: pl.DataFrame,
    dates: Sequence[date | np.datetime64],
    isins: Sequence[str],
    acquisition_audit: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Report action counts and segment-level provider status per ISIN-year.

    A successful query with no rows is reported as a true provider zero; it is
    never conflated with a failed ticker-segment request.  The source taxonomy
    is disclosed because Yahoo's free actions surface does not separately
    guarantee bonus, JCP, or subscription-right classifications.
    """

    checked = validate_action_table(actions)
    years = sorted({_as_date(value).year for value in dates})
    counts: dict[tuple[str, int], tuple[int, int]] = {}
    for row in checked.iter_rows(named=True):
        key = (str(row["isin"]), row["ex_date"].year)
        total, unresolved = counts.get(key, (0, 0))
        counts[key] = (total + 1, unresolved + int(bool(row["unresolved"])))
    audit_rows = (
        list(acquisition_audit.iter_rows(named=True))
        if acquisition_audit is not None
        else []
    )
    if acquisition_audit is not None and not {
        "isin",
        "first_date",
        "last_date",
        "status",
    }.issubset(acquisition_audit.columns):
        raise ValueError("corporate-action acquisition audit has wrong schema")
    rows: list[dict[str, object]] = []
    for isin in isins:
        for year in years:
            segments = [
                row
                for row in audit_rows
                if str(row["isin"]) == isin
                and _as_date(row["first_date"]).year <= year
                and _as_date(row["last_date"]).year >= year
            ]
            failed = sum(str(row["status"]) == "failed" for row in segments)
            succeeded = len(segments) - failed
            zeros = sum(
                str(row["status"]) != "failed" and int(row.get("action_rows") or 0) == 0
                for row in segments
            )
            action_count, unresolved_count = counts.get((isin, year), (0, 0))
            if failed and succeeded:
                status = "partial_provider_failure"
            elif failed:
                status = "provider_failure"
            elif succeeded and action_count:
                status = "covered_actions"
            elif succeeded:
                status = "covered_zero_actions"
            else:
                status = "no_acquisition_segment"
            rows.append(
                {
                    "isin": isin,
                    "year": year,
                    "action_count": action_count,
                    "unresolved_count": unresolved_count,
                    "ticker_segment_count": len(segments),
                    "successful_segment_count": succeeded,
                    "zero_action_segment_count": zeros,
                    "failed_segment_count": failed,
                    "acquisition_status": status,
                    "provider_taxonomy": "yfinance dividends and stock splits",
                    "source_limitations": (
                        "bonus, JCP, and subscription-right taxonomy is not "
                        "guaranteed by the provider; query success therefore does "
                        "not resolve economic no-action cells"
                    ),
                }
            )
    return pl.DataFrame(
        rows,
        schema={
            "isin": pl.String,
            "year": pl.Int32,
            "action_count": pl.Int32,
            "unresolved_count": pl.Int32,
            "ticker_segment_count": pl.Int32,
            "successful_segment_count": pl.Int32,
            "zero_action_segment_count": pl.Int32,
            "failed_segment_count": pl.Int32,
            "acquisition_status": pl.String,
            "provider_taxonomy": pl.String,
            "source_limitations": pl.String,
        },
    )


def cash_unit_adjustment_audit(actions: pl.DataFrame) -> pl.DataFrame:
    """Summarize the direction and magnitude of restored Yahoo cash units."""

    checked = validate_action_table(actions)
    cash = checked.filter(pl.col("cash_distribution_brl") > 0)
    if cash.is_empty():
        return pl.DataFrame(
            schema={
                "isin": pl.String,
                "cash_action_count": pl.Int32,
                "adjusted_cash_action_count": pl.Int32,
                "minimum_factor": pl.Float64,
                "maximum_factor": pl.Float64,
                "direction_proof": pl.String,
            }
        )
    factor = (
        pl.col("cash_unit_adjustment_factor").cast(pl.Float64)
        if "cash_unit_adjustment_factor" in cash.columns
        else pl.lit(1.0)
    )
    return (
        cash.with_columns(factor.alias("factor"))
        .group_by("isin")
        .agg(
            pl.len().cast(pl.Int32).alias("cash_action_count"),
            (pl.col("factor") != 1.0)
            .sum()
            .cast(pl.Int32)
            .alias("adjusted_cash_action_count"),
            pl.col("factor").min().alias("minimum_factor"),
            pl.col("factor").max().alias("maximum_factor"),
        )
        .with_columns(
            pl.lit(
                "provider cash multiplied by the product of strictly later "
                "split factors to restore contemporaneous raw-share units"
            ).alias("direction_proof")
        )
        .sort("isin")
    )


def dividend_close_drop_audit(
    dates: Sequence[date | np.datetime64],
    isins: Sequence[str],
    cash_distribution_brl: NDArray[np.floating],
    raw_close: NDArray[np.floating],
    observed: NDArray[np.bool_],
    *,
    outlier_absolute_error: float = 0.20,
) -> pl.DataFrame:
    """Compare ex-date close-to-close returns with the mechanical cash yield."""

    cash = np.asarray(cash_distribution_brl, dtype=np.float64)
    close = np.asarray(raw_close, dtype=np.float64)
    seen = np.asarray(observed, dtype=np.bool_)
    shape = (len(dates), len(isins))
    if cash.shape != shape or close.shape != shape or seen.shape != shape:
        raise ValueError("dividend audit axes are misaligned")
    rows: list[dict[str, object]] = []
    for name, isin in enumerate(isins):
        returns: list[float] = []
        mechanical: list[float] = []
        for day in np.flatnonzero(cash[:, name] > 0):
            prior = next(
                (
                    index
                    for index in range(day - 1, -1, -1)
                    if seen[index, name]
                    and np.isfinite(close[index, name])
                    and close[index, name] > 0
                ),
                None,
            )
            if (
                prior is None
                or not seen[day, name]
                or not np.isfinite(close[day, name])
                or close[day, name] <= 0
            ):
                continue
            returns.append(float(close[day, name] / close[prior, name] - 1.0))
            mechanical.append(float(-cash[day, name] / close[prior, name]))
        if not returns:
            continue
        mean_return = float(np.mean(returns))
        mean_mechanical = float(np.mean(mechanical))
        error = mean_return - mean_mechanical
        rows.append(
            {
                "isin": isin,
                "comparable_cash_action_count": len(returns),
                "mean_ex_date_close_return": mean_return,
                "mean_negative_cash_yield": mean_mechanical,
                "mean_difference": error,
                "outlier": abs(error) > outlier_absolute_error,
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "isin": pl.String,
            "comparable_cash_action_count": pl.Int32,
            "mean_ex_date_close_return": pl.Float64,
            "mean_negative_cash_yield": pl.Float64,
            "mean_difference": pl.Float64,
            "outlier": pl.Boolean,
        },
    ).sort("isin")


def audit_m1_adjustment_status(
    dates: Sequence[date | np.datetime64],
    isins: Sequence[str],
    m1_session_close: NDArray[np.floating],
    raw_daily_close: NDArray[np.floating],
    adjusted_daily_close: NDArray[np.floating],
    split_factor: NDArray[np.floating],
    cash_distribution_brl: NDArray[np.floating],
    *,
    relative_tolerance: float = 2e-3,
) -> pl.DataFrame:
    """Classify M1 units using both level and pre/post event-day ratios."""

    minute = np.asarray(m1_session_close, dtype=np.float64)
    raw = np.asarray(raw_daily_close, dtype=np.float64)
    adjusted = np.asarray(adjusted_daily_close, dtype=np.float64)
    split = np.asarray(split_factor, dtype=np.float64)
    cash = np.asarray(cash_distribution_brl, dtype=np.float64)
    shape = (len(dates), len(isins))
    if any(value.shape != shape for value in (minute, raw, adjusted, split, cash)):
        raise ValueError("M1 adjustment-audit axes are misaligned")
    rows: list[dict[str, object]] = []
    for day, name in np.argwhere((split != 1.0) | (cash > 0)):
        prior_day = next(
            (
                candidate
                for candidate in range(day - 1, -1, -1)
                if all(
                    np.isfinite(value[candidate, name]) and value[candidate, name] > 0
                    for value in (minute, raw, adjusted)
                )
            ),
            None,
        )
        comparable = (
            np.isfinite(minute[day, name])
            and np.isfinite(raw[day, name])
            and np.isfinite(adjusted[day, name])
            and minute[day, name] > 0
            and raw[day, name] > 0
            and adjusted[day, name] > 0
        )
        raw_error = (
            abs(minute[day, name] / raw[day, name] - 1.0) if comparable else np.nan
        )
        adjusted_error = (
            abs(minute[day, name] / adjusted[day, name] - 1.0) if comparable else np.nan
        )
        ratio_comparable = comparable and prior_day is not None
        m1_event_ratio = (
            minute[day, name] / minute[prior_day, name] if ratio_comparable else np.nan
        )
        raw_event_ratio = (
            raw[day, name] / raw[prior_day, name] if ratio_comparable else np.nan
        )
        adjusted_event_ratio = (
            adjusted[day, name] / adjusted[prior_day, name]
            if ratio_comparable
            else np.nan
        )
        raw_ratio_error = (
            abs(m1_event_ratio / raw_event_ratio - 1.0)
            if ratio_comparable and raw_event_ratio > 0
            else np.nan
        )
        adjusted_ratio_error = (
            abs(m1_event_ratio / adjusted_event_ratio - 1.0)
            if ratio_comparable and adjusted_event_ratio > 0
            else np.nan
        )
        if not comparable:
            status = "missing"
        elif (
            ratio_comparable
            and raw_ratio_error <= relative_tolerance
            and raw_ratio_error <= adjusted_ratio_error
        ):
            status = "raw_unadjusted"
        elif ratio_comparable and adjusted_ratio_error <= relative_tolerance:
            status = "price_adjusted"
        elif raw_error <= relative_tolerance and raw_error <= adjusted_error:
            status = "raw_unadjusted"
        elif adjusted_error <= relative_tolerance:
            status = "price_adjusted"
        else:
            status = "mismatch"
        rows.append(
            {
                "trade_date": _as_date(dates[day]),
                "isin": isins[name],
                "split_factor": float(split[day, name]),
                "cash_distribution_brl": float(cash[day, name]),
                "m1_close": float(minute[day, name]),
                "raw_daily_close": float(raw[day, name]),
                "adjusted_daily_close": float(adjusted[day, name]),
                "raw_relative_error": float(raw_error),
                "adjusted_relative_error": float(adjusted_error),
                "prior_trade_date": (
                    _as_date(dates[prior_day]) if prior_day is not None else None
                ),
                "m1_pre_post_ratio": float(m1_event_ratio),
                "raw_daily_pre_post_ratio": float(raw_event_ratio),
                "adjusted_daily_pre_post_ratio": float(adjusted_event_ratio),
                "raw_ratio_relative_error": float(raw_ratio_error),
                "adjusted_ratio_relative_error": float(adjusted_ratio_error),
                "status": status,
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "trade_date": pl.Date,
            "isin": pl.String,
            "split_factor": pl.Float64,
            "cash_distribution_brl": pl.Float64,
            "m1_close": pl.Float64,
            "raw_daily_close": pl.Float64,
            "adjusted_daily_close": pl.Float64,
            "raw_relative_error": pl.Float64,
            "adjusted_relative_error": pl.Float64,
            "prior_trade_date": pl.Date,
            "m1_pre_post_ratio": pl.Float64,
            "raw_daily_pre_post_ratio": pl.Float64,
            "adjusted_daily_pre_post_ratio": pl.Float64,
            "raw_ratio_relative_error": pl.Float64,
            "adjusted_ratio_relative_error": pl.Float64,
            "status": pl.String,
        },
    )


def m1_cotahist_mismatch_by_year(
    dates: Sequence[date | np.datetime64],
    m1_session_close: NDArray[np.floating],
    m1_session_close_valid: NDArray[np.bool_],
    cotahist_close: NDArray[np.floating],
    cotahist_observed: NDArray[np.bool_],
    *,
    maximum_log_mismatch: float = 0.005,
) -> pl.DataFrame:
    """Report the exact close-unit gate used before COTAHIST anchoring."""

    minute = np.asarray(m1_session_close, dtype=np.float64)
    minute_valid = np.asarray(m1_session_close_valid, dtype=np.bool_)
    official = np.asarray(cotahist_close, dtype=np.float64)
    official_valid = np.asarray(cotahist_observed, dtype=np.bool_)
    if (
        minute.ndim != 2
        or minute_valid.shape != minute.shape
        or official.shape != minute.shape
        or official_valid.shape != minute.shape
        or len(dates) != minute.shape[0]
    ):
        raise ValueError("M1/COTAHIST mismatch axes are misaligned")
    comparable = (
        minute_valid
        & official_valid
        & np.isfinite(minute)
        & np.isfinite(official)
        & (minute > 0)
        & (official > 0)
    )
    error = np.full(minute.shape, np.nan, dtype=np.float64)
    error[comparable] = np.abs(np.log(minute[comparable] / official[comparable]))
    mismatch = comparable & (error > maximum_log_mismatch)
    years = np.asarray(dates, dtype="datetime64[Y]").astype(np.int64) + 1970
    rows: list[dict[str, object]] = []
    for year in sorted(set(years.tolist())):
        selected = (years == year)[:, None]
        comparable_count = int((selected & comparable).sum())
        mismatch_count = int((selected & mismatch).sum())
        finite_error = error[selected & comparable]
        rows.append(
            {
                "year": int(year),
                "comparable_name_days": comparable_count,
                "mismatch_name_days": mismatch_count,
                "unavailable_name_days": int((selected & ~comparable).sum()),
                "mismatch_rate": (
                    mismatch_count / comparable_count if comparable_count else None
                ),
                "median_absolute_log_ratio": (
                    float(np.median(finite_error)) if finite_error.size else None
                ),
                "maximum_allowed_absolute_log_ratio": maximum_log_mismatch,
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "year": pl.Int32,
            "comparable_name_days": pl.Int64,
            "mismatch_name_days": pl.Int64,
            "unavailable_name_days": pl.Int64,
            "mismatch_rate": pl.Float64,
            "median_absolute_log_ratio": pl.Float64,
            "maximum_allowed_absolute_log_ratio": pl.Float64,
        },
    )


def _as_date(value: date | np.datetime64) -> date:
    if isinstance(value, np.datetime64):
        return value.astype("datetime64[D]").astype(object)
    return value


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Acquire cached ISIN-bound yfinance corporate actions"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--security-master", type=Path)
    source.add_argument("--cotahist-root", type=Path)
    parser.add_argument("--v1-assignments", type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> None:
    args = parse_args(arguments)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    input_paths: list[Path]
    if args.security_master:
        security_master = pl.read_parquet(args.security_master)
        input_paths = [args.security_master]
    else:
        from brazil_rv.preprocessing.contract import EXPECTED_EQUITIES

        from .contract import COTAHIST_YEARS
        from .data_foundation import (
            build_security_master,
            filter_cash_equities,
            load_cotahist,
        )

        if args.v1_assignments is None:
            raise ValueError(
                "--cotahist-root acquisition requires --v1-assignments for "
                "the frozen cash-equity exceptions"
            )
        if args.v1_assignments.is_dir():
            from brazil_rv.preprocessing.io import load_assignments

            assignments = load_assignments(args.v1_assignments)
            assignment_path = (
                args.v1_assignments / "xp_accepted_source_assignments_v1.parquet"
            )
        else:
            assignments = pl.read_parquet(args.v1_assignments)
            assignment_path = args.v1_assignments
        if (
            assignments.height != EXPECTED_EQUITIES
            or assignments.get_column("isin").n_unique() != EXPECTED_EQUITIES
        ):
            raise ValueError(
                f"v1 assignments must bind exactly {EXPECTED_EQUITIES} ISINs"
            )

        input_paths = sorted(args.cotahist_root.glob("year=*/equities_daily_*.parquet"))
        v1_isins = tuple(assignments.get_column("isin").to_list())
        daily = load_cotahist(input_paths, v1_isins=v1_isins).filter(
            pl.col("trade_date").dt.year().is_in(COTAHIST_YEARS)
        )
        daily = filter_cash_equities(daily, v1_isins=v1_isins)
        security_master = build_security_master(daily)
        input_paths.append(assignment_path)
    actions, audit = acquire_yfinance_actions(
        security_master,
        args.cache_dir,
        refresh=args.refresh,
    )
    master_path = output / "security_master.parquet"
    actions_path = output / "corporate_actions.parquet"
    audit_path = output / "yfinance_acquisition_audit.parquet"
    cash_unit_path = output / "cash_unit_adjustment_audit.parquet"
    security_master.write_parquet(master_path)
    actions.write_parquet(actions_path)
    audit.write_parquet(audit_path)
    cash_unit_adjustment_audit(actions).write_parquet(cash_unit_path)
    manifest = {
        "schema": "V2_CORPORATE_ACTIONS_V2",
        "provider_taxonomy": "yfinance dividends and stock splits",
        "source_limitations": (
            "bonus, JCP, and subscription-right taxonomy is not guaranteed by "
            "the provider; failed segments are resolved only around independently "
            "detected event candidates, and cash uncertainty is total-return-only"
        ),
        "acquisition_status_counts": {
            str(row["status"]): int(row["len"])
            for row in audit.group_by("status").len().iter_rows(named=True)
        },
        "security_master": {
            "path": master_path.name,
            "rows": security_master.height,
            "sha256": _sha256(master_path),
        },
        "actions": {
            "path": actions_path.name,
            "rows": actions.height,
            "sha256": _sha256(actions_path),
        },
        "acquisition_audit": {
            "path": audit_path.name,
            "rows": audit.height,
            "sha256": _sha256(audit_path),
        },
        "cash_unit_adjustment_audit": {
            "path": cash_unit_path.name,
            "rows": pl.read_parquet(cash_unit_path).height,
            "sha256": _sha256(cash_unit_path),
        },
        "inputs": [
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in input_paths
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
