

"""Transport-agnostic customer event types and envelopes."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict


class EventType(str, Enum):
    """Customer-domain events defined in `docs/event-model.md`."""

    PAYMENT_MISSED = "payment_missed"
    CLAIM_CREATED = "claim_created"
    PREMIUM_CHANGED = "premium_changed"
    POLICY_RENEWED = "policy_renewed"
    ENGAGEMENT_CHANGED = "engagement_changed"
    COVERAGE_DOWNGRADED = "coverage_downgraded"
    COMPLAINT_LODGED = "complaint_lodged"


@dataclass
class Event:
    """Event envelope shared by API, generator, and future transport adapters."""

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
