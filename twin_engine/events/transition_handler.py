"""
Event Transition Handler.

Implements S_(t+1) = f(S_t, E_t): receives the current Twin state and an
event, and returns the updated Twin state. This is the ONLY place in the
codebase that mutates a real (non-cloned) Twin state in response to an
event - the Scenario Transformer (twin_engine/simulation) reuses the same
per-event-type logic but always applies it to a cloned state instead.

Each event type below documents:
  Event type | Affected Twin state | Affected ML features | Expected state change

All fields mutated here are dataset-grounded (data/customer_churn.csv /
docs/dataset-mapping.md) - no event invents a Twin state field that
doesn't correspond to a real model feature.
"""

from __future__ import annotations

from typing import Callable, Dict

from twin_engine.events.event import Event, EventType
from twin_engine.state.twin_state import TwinState

CLAIM_OUTCOMES = ["approved", "rejected", "pending"]


def _apply_payment_missed(state: TwinState, event: Event) -> str:
    """
    Event: payment_missed
    Affected Twin state: late_payment_count_12m
    Affected ML features: late_payment_count_12m, (derived) missed_payment_flag
    Expected state change: late_payment_count_12m += 1 (or += payload
        "count" if given). missed_payment_flag is a derived property
        (>= 4) so it updates automatically - see TwinState.missed_payment_flag.
    """
    increment = int(event.payload.get("count", 1))
    state.late_payment_count_12m += increment
    return f"Payment missed (late_payment_count_12m -> {state.late_payment_count_12m})"


def _apply_claim_created(state: TwinState, event: Event) -> str:
    """
    Event: claim_created
    Affected Twin state: num_claims_12m, num_{approved,rejected,pending}_claims_12m,
        total_claim_amount_12m, avg_claim_amount, total_payout_amount_12m,
        avg_settlement_time_days, days_since_last_claim
    Affected ML features: all of the above, plus (derived) payout_ratio_12m
    Expected state change: a new claim is filed with a given amount,
        outcome ("approved"/"rejected"/"pending"), and settlement time.
        Running 12m totals/counts are incremented; averages are
        recomputed; days_since_last_claim resets to 0. If the claim is
        approved, its amount contributes to total_payout_amount_12m
        (using the payload's `payout_fraction`, default 1.0 - i.e. fully
        paid out - a documented MVP simplification of real claims
        adjustment).
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
    """
    Event: premium_changed
    Affected Twin state: current_premium, num_price_increases_last_3y
    Affected ML features: current_premium, num_price_increases_last_3y,
        (derived) premium_change_pct, premium_to_coverage_ratio
    Expected state change: current_premium moves to a new value - either
        given directly (`current_premium`) or as a relative change
        (`change_pct`, e.g. 0.15 for +15%) against the current premium.
        num_price_increases_last_3y is incremented if the new premium is
        higher than the old one.
    """
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
    """
    Event: policy_renewed
    Affected Twin state: premium_last_year, and a fresh trailing-12-month
        window: late_payment_count_12m, num_claims_12m,
        num_{approved,rejected,pending}_claims_12m, total_claim_amount_12m,
        total_payout_amount_12m, avg_claim_amount, num_contacts_12m,
        complaint_flag, complaint_resolution_days
    Affected ML features: all of the above, plus every derived feature
        that depends on them (premium_change_pct, missed_payment_flag,
        payout_ratio_12m)
    Expected state change: renewal rolls the current premium into
        `premium_last_year` (so the next `premium_changed` event compares
        against it) and resets the "last 12 months" trailing counters to
        zero, representing the start of a fresh policy period.

    ASSUMPTION: real complaint/claims history could persist across a
    renewal in a real system; this MVP resets them to model a fresh
    trailing-12-month reporting window, matching how the dataset's "_12m"
    columns are framed. Documented explicitly, not hidden.
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
    """
    Event: engagement_changed
    Affected Twin state: num_contacts_12m, quote_requested_flag
    Affected ML features: num_contacts_12m, quote_requested_flag
    Expected state change: num_contacts_12m += 1 (or += payload
        `contact_delta`); optionally sets quote_requested_flag if the
        payload's `quote_requested` is true (a customer shopping around
        for a quote is a meaningful, dataset-grounded signal distinct from
        a routine contact).
    """
    delta = int(event.payload.get("contact_delta", 1))
    state.num_contacts_12m = max(0, state.num_contacts_12m + delta)
    description = f"Customer engagement changed (num_contacts_12m -> {state.num_contacts_12m})"
    if event.payload.get("quote_requested"):
        state.quote_requested_flag = 1
        description += "; quote requested"
    return description


def _apply_coverage_downgraded(state: TwinState, event: Event) -> str:
    """
    Event: coverage_downgraded
    Affected Twin state: coverage_amount, coverage_downgrade_flag
    Affected ML features: coverage_amount, coverage_downgrade_flag,
        (derived) premium_to_coverage_ratio
    Expected state change: coverage_amount decreases (given directly via
        `coverage_amount`, or as a fractional cut via `reduction_pct`,
        e.g. 0.2 for a 20% reduction); coverage_downgrade_flag -> 1.
    """
    old_coverage = state.coverage_amount
    if "coverage_amount" in event.payload:
        new_coverage = float(event.payload["coverage_amount"])
    elif "reduction_pct" in event.payload:
        new_coverage = old_coverage * (1.0 - float(event.payload["reduction_pct"]))
    else:
        new_coverage = old_coverage * 0.8  # default: a 20% downgrade

    state.coverage_amount = max(0.0, new_coverage)
    state.coverage_downgrade_flag = 1
    return f"Coverage downgraded ({old_coverage:.2f} -> {state.coverage_amount:.2f})"


def _apply_complaint_lodged(state: TwinState, event: Event) -> str:
    """
    Event: complaint_lodged
    Affected Twin state: complaint_flag, complaint_resolution_days
    Affected ML features: complaint_flag, complaint_resolution_days
    Expected state change: complaint_flag -> 1; complaint_resolution_days
        set from the payload (default: 0, i.e. not yet resolved/unknown).
    """
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
    """
    Applies an event to a Twin state: S_(t+1) = f(S_t, E_t).

    `apply` mutates and returns the SAME state object it was given. Callers
    that must not mutate the real Twin (e.g. the Scenario Transformer) are
    responsible for passing in a cloned state.
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
