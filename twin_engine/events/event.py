"""
Customer-domain events.

Event flow (see docs/event-model.md for full detail):

    Event Generator -> Event -> Event Transition Handler -> Twin State Store
        -> Risk recalculation

Only event types grounded in the actual dataset (data/customer_churn.csv,
the insurance policyholder churn dataset) are implemented. Each is
documented with the Twin state field(s) it affects, the ML feature(s) it
therefore affects, and the expected state change - see
twin_engine/events/transition_handler.py and docs/event-model.md.

`Event` has no dependency on Kafka or any transport mechanism - it is a
plain, transport-agnostic data object, produced today by the local
EventGenerator or the POST /events endpoint. The same object could equally
be the payload consumed off a Kafka topic later, without changing anything
downstream of this class.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict


class EventType(str, Enum):
    """
    Supported customer-domain events.

    Dataset grounding (see docs/event-model.md for the full contract of each):
      PAYMENT_MISSED         -> late_payment_count_12m (+ derived missed_payment_flag)
      CLAIM_CREATED          -> num_claims_12m, num_{approved,rejected,pending}_claims_12m,
                                 total_claim_amount_12m, avg_claim_amount,
                                 total_payout_amount_12m, avg_settlement_time_days,
                                 days_since_last_claim
      PREMIUM_CHANGED        -> current_premium, num_price_increases_last_3y
                                 (+ derived premium_change_pct, premium_to_coverage_ratio)
      POLICY_RENEWED         -> premium_last_year, and a fresh trailing-12-month
                                 window for late payments/claims/complaints/contacts
      ENGAGEMENT_CHANGED     -> num_contacts_12m, quote_requested_flag
      COVERAGE_DOWNGRADED    -> coverage_amount, coverage_downgrade_flag
                                 (+ derived premium_to_coverage_ratio)
      COMPLAINT_LODGED       -> complaint_flag, complaint_resolution_days
    """

    PAYMENT_MISSED = "payment_missed"
    CLAIM_CREATED = "claim_created"
    PREMIUM_CHANGED = "premium_changed"
    POLICY_RENEWED = "policy_renewed"
    ENGAGEMENT_CHANGED = "engagement_changed"
    COVERAGE_DOWNGRADED = "coverage_downgraded"
    COMPLAINT_LODGED = "complaint_lodged"


@dataclass
class Event:
    """A single customer-domain event, E_t."""

    customer_id: str
    event_type: EventType
    payload: Dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: str = "unspecified"
    occurred_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "customer_id": self.customer_id,
            "event_type": self.event_type.value if isinstance(self.event_type, EventType) else self.event_type,
            "payload": self.payload,
            "source": self.source,
            "occurred_at": self.occurred_at,
        }
