

"""Apply supported customer events to a mutable `TwinState`."""

from __future__ import annotations

from typing import Callable, Dict

from twin_engine.events.event import Event, EventType
from twin_engine.state.twin_state import TwinState

CLAIM_OUTCOMES = ["approved", "rejected", "pending"]


def _apply_payment_missed(state: TwinState, event: Event) -> str:
    """Increase the trailing late-payment count by the requested increment."""
    increment = int(event.payload.get("count", 1))
    state.late_payment_count_12m += increment
    return f"Payment missed (late_payment_count_12m -> {state.late_payment_count_12m})"


def _apply_claim_created(state: TwinState, event: Event) -> str:
    """Accumulate a claim in the trailing-12-month aggregate.

    Unknown outcomes default to approved. Approved claims contribute their
    amount times `payout_fraction` to payouts; the default is full payment.
    """
    amount = float(event.payload.get("claim_amount", 0.0))
    outcome = event.payload.get("outcome", "approved")
    if outcome not in CLAIM_OUTCOMES:
        outcome = "approved"
    settlement_days = int(event.payload.get("settlement_time_days", state.avg_settlement_time_days))
    payout_fraction = float(event.payload.get("payout_fraction", 1.0 if outcome == "approved" else 0.0))

    previous_total_claims = state.num_claims_12m
    state.num_claims_12m += 1
    if outcome == "approved":
        state.num_approved_claims_12m += 1
        state.total_payout_amount_12m += amount * payout_fraction
    elif outcome == "rejected":
        state.num_rejected_claims_12m += 1
    else:
        state.num_pending_claims_12m += 1

    state.total_claim_amount_12m += amount
    state.avg_claim_amount = (
        state.total_claim_amount_12m / state.num_claims_12m if state.num_claims_12m else 0.0
    )
    state.avg_settlement_time_days = settlement_days
    state.days_since_last_claim = 0

    return (
        f"New claim filed (amount={amount:.2f}, outcome={outcome}); "
        f"num_claims_12m {previous_total_claims} -> {state.num_claims_12m}"
    )


def _apply_premium_changed(state: TwinState, event: Event) -> str:
    """Set the premium directly or apply a relative change to it."""
    old_premium = state.current_premium
    if "current_premium" in event.payload:
        new_premium = float(event.payload["current_premium"])
    elif "change_pct" in event.payload:
        new_premium = old_premium * (1.0 + float(event.payload["change_pct"]))
    else:
        new_premium = old_premium

    new_premium = max(0.0, new_premium)
    state.current_premium = new_premium
    if new_premium > old_premium:
        state.num_price_increases_last_3y += 1

    return f"Premium changed ({old_premium:.2f} -> {new_premium:.2f})"


def _apply_policy_renewed(state: TwinState, event: Event) -> str:
    """Start a new reporting window at renewal.

    The current premium becomes the comparison baseline and all trailing-
    12-month counters reset. This matches the aggregate shape of the model
    data, although a production claims ledger may carry history across renewal.
    """
    state.premium_last_year = state.current_premium
    state.late_payment_count_12m = 0
    state.num_claims_12m = 0
    state.num_approved_claims_12m = 0
    state.num_rejected_claims_12m = 0
    state.num_pending_claims_12m = 0
    state.total_claim_amount_12m = 0.0
    state.total_payout_amount_12m = 0.0
    state.avg_claim_amount = 0.0
    state.num_contacts_12m = 0
    state.complaint_flag = 0
    state.complaint_resolution_days = 0
    return "Policy renewed (trailing 12-month counters reset for new period)"


def _apply_engagement_changed(state: TwinState, event: Event) -> str:
    """Adjust contact volume and record quote shopping when supplied."""
    delta = int(event.payload.get("contact_delta", 1))
    state.num_contacts_12m = max(0, state.num_contacts_12m + delta)
    description = f"Customer engagement changed (num_contacts_12m -> {state.num_contacts_12m})"
    if event.payload.get("quote_requested"):
        state.quote_requested_flag = 1
        description += "; quote requested"
    return description


def _apply_coverage_downgraded(state: TwinState, event: Event) -> str:
    """Reduce coverage by an explicit amount or percentage, never below zero."""
    old_coverage = state.coverage_amount
    if "coverage_amount" in event.payload:
        new_coverage = float(event.payload["coverage_amount"])
    elif "reduction_pct" in event.payload:
        new_coverage = old_coverage * (1.0 - float(event.payload["reduction_pct"]))
    else:
        new_coverage = old_coverage * 0.8

    state.coverage_amount = max(0.0, new_coverage)
    state.coverage_downgrade_flag = 1
    return f"Coverage downgraded ({old_coverage:.2f} -> {state.coverage_amount:.2f})"


def _apply_complaint_lodged(state: TwinState, event: Event) -> str:
    """Mark a complaint and record its resolution time, defaulting to unresolved."""
    state.complaint_flag = 1
    state.complaint_resolution_days = int(event.payload.get("resolution_days", 0))
    return f"Complaint lodged (resolution_days={state.complaint_resolution_days})"


_HANDLERS: Dict[EventType, Callable[[TwinState, Event], str]] = {
    EventType.PAYMENT_MISSED: _apply_payment_missed,
    EventType.CLAIM_CREATED: _apply_claim_created,
    EventType.PREMIUM_CHANGED: _apply_premium_changed,
    EventType.POLICY_RENEWED: _apply_policy_renewed,
    EventType.ENGAGEMENT_CHANGED: _apply_engagement_changed,
    EventType.COVERAGE_DOWNGRADED: _apply_coverage_downgraded,
    EventType.COMPLAINT_LODGED: _apply_complaint_lodged,
}


class EventTransitionHandler:
    """Apply an event in place and return the same state object.

    Callers that must preserve the real Twin must pass a cloned state.
    """

    def apply(self, state: TwinState, event: Event) -> TwinState:
        event_type = event.event_type
        if isinstance(event_type, str):
            event_type = EventType(event_type)

        handler = _HANDLERS.get(event_type)
        if handler is None:
            raise ValueError(f"No transition handler registered for event type: {event_type}")

        description = handler(state, event)
        state.record_event(event_id=event.event_id, event_type=event_type.value, payload=event.payload, description=description)
        state.touch()
        return state


event_transition_handler = EventTransitionHandler()
