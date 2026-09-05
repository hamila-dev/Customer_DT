# Digital Twin Engine Snapshot

Generated from the current repository files. The source and documentation sections below are verbatim file contents.

## `twin_engine/state/__init__.py`

```python twin_engine/state/__init__.py

```

## `twin_engine/state/state_store.py`

```python twin_engine/state/state_store.py
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from twin_engine.state.twin_state import TwinState

import config


class TwinStateStore:
    """Simple thread-safe, JSON-file-persisted store of TwinState objects."""

    def __init__(self, storage_path: Path = config.TWIN_STORE_PATH):
        self._storage_path = storage_path
        self._lock = threading.RLock()
        self._states: Dict[str, TwinState] = {}
        self._load_from_disk()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _load_from_disk(self) -> None:
        if not self._storage_path.exists():
            return
        try:
            raw = json.loads(self._storage_path.read_text())
        except (json.JSONDecodeError, OSError):
            return
        with self._lock:
            for customer_id, state_dict in raw.items():
                self._states[customer_id] = TwinState.from_dict(state_dict)

    def _flush_to_disk(self) -> None:
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {cid: state.to_dict() for cid, state in self._states.items()}
        tmp_path = self._storage_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(serializable, indent=2, default=str))
        tmp_path.replace(self._storage_path)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------
    def save(self, state: TwinState) -> None:
        with self._lock:
            self._states[state.customer_id] = state
            self._flush_to_disk()

    def bulk_save(self, states: Iterable[TwinState]) -> None:
        with self._lock:
            for state in states:
                self._states[state.customer_id] = state
            self._flush_to_disk()

    def get(self, customer_id: str) -> Optional[TwinState]:
        with self._lock:
            return self._states.get(customer_id)

    def exists(self, customer_id: str) -> bool:
        with self._lock:
            return customer_id in self._states

    def list_all(self) -> List[TwinState]:
        with self._lock:
            return list(self._states.values())

    def count(self) -> int:
        with self._lock:
            return len(self._states)

    def is_empty(self) -> bool:
        with self._lock:
            return len(self._states) == 0


# Module-level singleton used across the app (simple, explicit, no DI framework
# needed for an MVP of this size).
twin_state_store = TwinStateStore()

```

## `twin_engine/state/twin_state.py`

```python twin_engine/state/twin_state.py
from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TwinEventRecord:
    """A single entry in the Twin's event history."""

    event_id: str
    event_type: str
    occurred_at: str
    payload: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TwinState:
    """Represents S_t - the current state of one policyholder's Digital Twin."""

    # ---------------- Identity / static profile (never mutated by events) ----------------
    customer_id: str
    age: int
    region_name: str
    marital_status: str
    customer_tenure_months: int
    multi_policy_flag: int
    num_policies: int
    policy_type: str
    renewal_month: int
    payment_frequency: str
    autopay_enabled: int

    # ---------------- Dynamic (mutated by events) ----------------
    current_premium: float
    premium_last_year: float
    num_price_increases_last_3y: int
    coverage_amount: float
    late_payment_count_12m: int
    num_claims_12m: int
    num_approved_claims_12m: int
    num_rejected_claims_12m: int
    num_pending_claims_12m: int
    total_claim_amount_12m: float
    total_payout_amount_12m: float
    avg_claim_amount: float
    avg_settlement_time_days: int
    days_since_last_claim: int
    num_contacts_12m: int
    complaint_flag: int
    complaint_resolution_days: int
    quote_requested_flag: int
    coverage_downgrade_flag: int

    # ---------------- ML-only (no dataset-grounded event mutates this in the MVP) ----------------
    payment_method_change_flag: int = 0

    # ---------------- Twin-engine bookkeeping (not raw dataset columns) ----------------
    version: int = 0
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    event_history: List[TwinEventRecord] = field(default_factory=list)

    # ---------------- Ground truth label, reference only, never an ML feature ----------------
    historical_churn_label: Optional[int] = None

    # ------------------------------------------------------------------
    # Derived properties - recomputed, never stored independently
    # ------------------------------------------------------------------
    @property
    def premium_change_pct(self) -> float:
        """(current_premium - premium_last_year) / premium_last_year.

        NOTE: in the source training data this column carries some
        independent noise beyond this exact formula (observed deviation up
        to ~0.22 in a sample check). For Twin simulation purposes we
        recompute it exactly from current_premium/premium_last_year so a
        simulated premium change always produces an internally consistent
        feature vector - this is a documented MVP modeling choice, not a
        claim that it reproduces the original data-generating noise.
        """
        if self.premium_last_year == 0:
            return 0.0
        return (self.current_premium - self.premium_last_year) / self.premium_last_year

    @property
    def premium_to_coverage_ratio(self) -> float:
        """current_premium / coverage_amount - an exact match to how this
        column is derived in the source dataset (verified during
        integration, floating-point-rounding aside)."""
        if self.coverage_amount == 0:
            return 0.0
        return self.current_premium / self.coverage_amount

    @property
    def payout_ratio_12m(self) -> float:
        """total_payout_amount_12m / total_claim_amount_12m.

        ASSUMPTION: when total_claim_amount_12m is 0 (no claims filed),
        the source dataset still carries a baseline value in roughly the
        0.75-0.85 range rather than an undefined 0/0. We use a documented
        MVP default of 0.75 in that case rather than 0.0, to avoid
        implying "no claims" is equivalent to "claims are never paid out."
        """
        if self.total_claim_amount_12m == 0:
            return 0.75
        return self.total_payout_amount_12m / self.total_claim_amount_12m

    @property
    def missed_payment_flag(self) -> int:
        """1 if late_payment_count_12m >= 4, else 0 - matches the exact
        rule documented in the source data dictionary
        ("1 if missed payments flag (>=4 late payments), else 0"),
        verified against the real dataset during integration."""
        return 1 if self.late_payment_count_12m >= 4 else 0

    # ------------------------------------------------------------------
    def clone(self) -> "TwinState":
        """Deep, fully independent copy. Used by the Scenario Transformer
        so what-if simulations can never mutate the real Twin state."""
        return copy.deepcopy(self)

    def record_event(self, event_id: str, event_type: str, payload: Dict[str, Any], description: str = "") -> None:
        self.event_history.append(
            TwinEventRecord(event_id=event_id, event_type=event_type, occurred_at=_now_iso(), payload=payload, description=description)
        )
        if len(self.event_history) > 50:
            self.event_history = self.event_history[-50:]

    def touch(self) -> None:
        self.version += 1
        self.updated_at = _now_iso()

    def to_feature_dict(self) -> Dict[str, Any]:
        """
        The full feature vector fed to preprocessing.joblib. Field names
        and set intentionally match model/feature_schema.json and
        model/model_metadata.json's feature_columns exactly - see
        risk_intelligence/feature_mapper.py.
        """
        return {
            "age": self.age,
            "customer_tenure_months": self.customer_tenure_months,
            "multi_policy_flag": self.multi_policy_flag,
            "num_policies": self.num_policies,
            "renewal_month": self.renewal_month,
            "current_premium": self.current_premium,
            "premium_last_year": self.premium_last_year,
            "premium_change_pct": self.premium_change_pct,
            "num_price_increases_last_3y": self.num_price_increases_last_3y,
            "coverage_amount": self.coverage_amount,
            "premium_to_coverage_ratio": self.premium_to_coverage_ratio,
            "autopay_enabled": self.autopay_enabled,
            "late_payment_count_12m": self.late_payment_count_12m,
            "missed_payment_flag": self.missed_payment_flag,
            "payment_method_change_flag": self.payment_method_change_flag,
            "num_claims_12m": self.num_claims_12m,
            "num_approved_claims_12m": self.num_approved_claims_12m,
            "num_rejected_claims_12m": self.num_rejected_claims_12m,
            "num_pending_claims_12m": self.num_pending_claims_12m,
            "avg_claim_amount": self.avg_claim_amount,
            "total_claim_amount_12m": self.total_claim_amount_12m,
            "total_payout_amount_12m": self.total_payout_amount_12m,
            "payout_ratio_12m": self.payout_ratio_12m,
            "avg_settlement_time_days": self.avg_settlement_time_days,
            "days_since_last_claim": self.days_since_last_claim,
            "num_contacts_12m": self.num_contacts_12m,
            "complaint_flag": self.complaint_flag,
            "complaint_resolution_days": self.complaint_resolution_days,
            "quote_requested_flag": self.quote_requested_flag,
            "coverage_downgrade_flag": self.coverage_downgrade_flag,
            "region_name": self.region_name,
            "marital_status": self.marital_status,
            "policy_type": self.policy_type,
            "payment_frequency": self.payment_frequency,
        }

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["premium_change_pct"] = self.premium_change_pct
        d["premium_to_coverage_ratio"] = self.premium_to_coverage_ratio
        d["payout_ratio_12m"] = self.payout_ratio_12m
        d["missed_payment_flag"] = self.missed_payment_flag
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TwinState":
        data = dict(data)
        for derived_field in ("premium_change_pct", "premium_to_coverage_ratio", "payout_ratio_12m", "missed_payment_flag"):
            data.pop(derived_field, None)
        history = data.pop("event_history", [])
        state = cls(**data)
        state.event_history = [TwinEventRecord(**h) if isinstance(h, dict) else h for h in history]
        return state

```

## `twin_engine/events/transition_handler.py`

```python twin_engine/events/transition_handler.py


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

```

## `twin_engine/events/event.py`

```python twin_engine/events/event.py


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

```

## `twin_engine/events/event_generator.py`

```python twin_engine/events/event_generator.py


from __future__ import annotations

import logging
import random
import threading
import time
from typing import List, Optional

from twin_engine.events.event import Event, EventType
from twin_engine.state.state_store import TwinStateStore, twin_state_store
from twin_engine.synchronization.synchronizer import StateSynchronizer, state_synchronizer

import config

logger = logging.getLogger(__name__)

CLAIM_OUTCOMES = ["approved", "approved", "rejected", "pending"]  # weighted toward "approved"


def _random_event_for_customer(customer_id: str, scenarios: List[str]) -> Event:
    event_type = EventType(random.choice(scenarios))
    payload = {}

    if event_type == EventType.CLAIM_CREATED:
        payload = {
            "claim_amount": round(random.uniform(200, 8000), 2),
            "outcome": random.choice(CLAIM_OUTCOMES),
            "settlement_time_days": random.randint(3, 30),
        }
    elif event_type == EventType.PREMIUM_CHANGED:
        payload = {"change_pct": round(random.uniform(-0.05, 0.20), 3)}
    elif event_type == EventType.ENGAGEMENT_CHANGED:
        payload = {"contact_delta": random.randint(1, 3), "quote_requested": random.random() < 0.2}
    elif event_type == EventType.COVERAGE_DOWNGRADED:
        payload = {"reduction_pct": round(random.uniform(0.1, 0.3), 2)}
    elif event_type == EventType.COMPLAINT_LODGED:
        payload = {"resolution_days": random.randint(1, 21)}
    elif event_type == EventType.PAYMENT_MISSED:
        payload = {"count": 1}
    # POLICY_RENEWED needs no extra payload.

    return Event(customer_id=customer_id, event_type=event_type, payload=payload, source="event_generator")


class EventGenerator:
    """
    Background thread that generates simulated events for the customers
    currently in the Twin State Store, at a configurable interval, using a
    configurable set of scenario event types.
    """

    def __init__(
        self,
        store: TwinStateStore = twin_state_store,
        synchronizer: StateSynchronizer = state_synchronizer,
        interval_seconds: float = config.EVENT_GENERATOR_DEFAULT_INTERVAL_SECONDS,
        scenarios: Optional[List[str]] = None,
    ):
        self._store = store
        self._synchronizer = synchronizer
        self.interval_seconds = interval_seconds
        self.scenarios = scenarios or list(config.EVENT_GENERATOR_SCENARIOS)
        self._thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
        self._events_generated = 0

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def events_generated(self) -> int:
        return self._events_generated

    def _run_loop(self) -> None:
        logger.info("Event generator started (interval=%ss, scenarios=%s)", self.interval_seconds, self.scenarios)
        while not self._stop_flag.is_set():
            customers = self._store.list_all()
            if customers:
                target = random.choice(customers)
                event = _random_event_for_customer(target.customer_id, self.scenarios)
                try:
                    self._synchronizer.process_event(event)
                    self._events_generated += 1
                except Exception:
                    logger.exception("Event generator failed to process event %s", event.event_id)
            self._stop_flag.wait(self.interval_seconds)
        logger.info("Event generator stopped")

    def start(self) -> None:
        if self.is_running:
            return
        self._stop_flag.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="event-generator")
        self._thread.start()

    def stop(self) -> None:
        self._stop_flag.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_seconds + 2)
        self._thread = None

    def configure(self, interval_seconds: Optional[float] = None, scenarios: Optional[List[str]] = None) -> None:
        if interval_seconds is not None:
            self.interval_seconds = interval_seconds
        if scenarios is not None:
            self.scenarios = scenarios


# Module-level singleton, started/stopped via the API (POST /events control
# endpoints) or standalone via `python -m twin_engine.events.event_generator`.
event_generator = EventGenerator()


def run_standalone() -> None:
    """Entry point for running the generator as its own local process,
    independent of the FastAPI app (python -m twin_engine.events.event_generator)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    generator = EventGenerator()
    generator.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        generator.stop()


if __name__ == "__main__":
    run_standalone()

```

## `twin_engine/events/__init__.py`

```python twin_engine/events/__init__.py

```

## `twin_engine/synchronization/__init__.py`

```python twin_engine/synchronization/__init__.py

```

## `twin_engine/synchronization/synchronizer.py`

```python twin_engine/synchronization/synchronizer.py

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Callable, Optional

from twin_engine.events.event import Event
from twin_engine.events.transition_handler import EventTransitionHandler, event_transition_handler
from twin_engine.state.state_store import TwinStateStore, twin_state_store
from twin_engine.state.twin_state import TwinState

import config

logger = logging.getLogger(__name__)

# Callback signature: (customer_id, updated_state) -> None
RiskRecalcCallback = Callable[[str, TwinState], None]


class StateSynchronizer:
    def __init__(
        self,
        store: TwinStateStore = twin_state_store,
        transition_handler: EventTransitionHandler = event_transition_handler,
        on_state_updated: Optional[RiskRecalcCallback] = None,
    ):
        self._store = store
        self._transition_handler = transition_handler
        self._on_state_updated = on_state_updated
        config.EVENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    def set_risk_recalc_callback(self, callback: RiskRecalcCallback) -> None:
        self._on_state_updated = callback

    def _append_event_log(self, event: Event) -> None:
        """Simple append-only local event log (JSON lines) for auditability/demo."""
        try:
            with config.EVENT_LOG_PATH.open("a") as f:
                f.write(json.dumps(event.to_dict()) + "\n")
        except OSError:
            logger.exception("Failed to append event to event log")

    def process_event(self, event: Event) -> TwinState:
        """
        Apply a single event to the current Twin state, persist it, log it,
        and trigger risk recalculation.
        """
        state = self._store.get(event.customer_id)
        if state is None:
            raise KeyError(f"No Twin state found for customer_id={event.customer_id}")

        updated_state = self._transition_handler.apply(state, event)
        self._store.save(updated_state)
        self._append_event_log(event)

        if self._on_state_updated is not None:
            self._on_state_updated(event.customer_id, updated_state)

        logger.info(
            "Synchronized event %s (%s) for customer %s at %s",
            event.event_id,
            event.event_type,
            event.customer_id,
            datetime.now(timezone.utc).isoformat(),
        )
        return updated_state


# Module-level singleton wired up at API startup (see api/main.py) with the
# real risk-recalculation callback.
state_synchronizer = StateSynchronizer()

```

## `twin_engine/simulation/scenario_transformer.py`

```python twin_engine/simulation/scenario_transformer.py


from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from twin_engine.events.event import Event, EventType
from twin_engine.events.transition_handler import EventTransitionHandler, event_transition_handler
from twin_engine.state.twin_state import TwinState


@dataclass
class Scenario:
    """A named what-if scenario: an event type plus its parameters."""

    name: str
    event_type: EventType
    parameters: Dict[str, Any]

    @classmethod
    def from_request(cls, scenario_name: str, parameters: Dict[str, Any]) -> "Scenario":
        try:
            event_type = EventType(scenario_name)
        except ValueError as exc:
            valid = ", ".join(e.value for e in EventType)
            raise ValueError(f"Unknown scenario '{scenario_name}'. Valid scenarios: {valid}") from exc
        return cls(name=scenario_name, event_type=event_type, parameters=parameters)


class ScenarioTransformer:
    """
    Clones a Twin state and applies a hypothetical scenario to the clone.
    NEVER mutates the real Twin.
    """

    def __init__(self, transition_handler: EventTransitionHandler = event_transition_handler):
        self._transition_handler = transition_handler

    def transform(self, state: TwinState, scenario: Scenario) -> TwinState:
        """Returns S'_t - a brand-new TwinState object, independent of `state`."""
        cloned_state = state.clone()

        synthetic_event = Event(
            customer_id=cloned_state.customer_id,
            event_type=scenario.event_type,
            payload=scenario.parameters,
            source="scenario_transformer",
        )
        self._transition_handler.apply(cloned_state, synthetic_event)
        return cloned_state


scenario_transformer = ScenarioTransformer()

```

## `twin_engine/simulation/monte_carlo.py`

```python twin_engine/simulation/monte_carlo.py


from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from risk_intelligence.predictor import ChurnPredictor, churn_predictor
from twin_engine.simulation.scenario_transformer import Scenario, ScenarioTransformer, scenario_transformer
from twin_engine.state.twin_state import TwinState

import config

# Scenario parameters treated as "numeric" for perturbation purposes -
# the monetary/percentage payload fields used by the dataset-grounded
# events in twin_engine/events/transition_handler.py.
NUMERIC_SCENARIO_PARAMETERS = {
    "claim_amount",
    "current_premium",
    "change_pct",
    "coverage_amount",
    "reduction_pct",
}


@dataclass
class MonteCarloResult:
    customer_id: str
    scenario_name: str
    trials: int
    mean: float
    median: float
    p10: float
    p90: float
    std_dev: float
    distribution: List[float]
    assumptions: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "customer_id": self.customer_id,
            "scenario_name": self.scenario_name,
            "trials": self.trials,
            "mean_churn_probability": round(self.mean, 4),
            "median_churn_probability": round(self.median, 4),
            "p10_churn_probability": round(self.p10, 4),
            "p90_churn_probability": round(self.p90, 4),
            "std_dev": round(self.std_dev, 4),
            # Distribution is returned rounded and (for large N) downsampled
            # for a reasonably sized JSON payload / histogram rendering.
            "distribution_sample": [round(v, 4) for v in self.distribution],
            "assumptions": self.assumptions,
        }


class MonteCarloEngine:
    def __init__(
        self,
        transformer: ScenarioTransformer = scenario_transformer,
        predictor: ChurnPredictor = churn_predictor,
    ):
        self._transformer = transformer
        self._predictor = predictor

    def _perturb_parameters(self, parameters: Dict[str, Any], noise_std: float, rng: random.Random) -> Dict[str, Any]:
        perturbed = dict(parameters)
        for key, value in parameters.items():
            if key in NUMERIC_SCENARIO_PARAMETERS and isinstance(value, (int, float)):
                noise_factor = rng.gauss(1.0, noise_std)
                perturbed_value = value * noise_factor
                # Only floor genuinely non-negative monetary quantities at 0.
                # "change_pct" is a signed percentage (a premium DECREASE is
                # legitimate and must stay negative) so it is perturbed but
                # never clamped.
                if key in ("claim_amount", "current_premium", "coverage_amount"):
                    perturbed_value = max(0.0, perturbed_value)
                perturbed[key] = perturbed_value
        return perturbed

    def run(
        self,
        state: TwinState,
        scenario: Scenario,
        trials: int = config.MONTE_CARLO_DEFAULT_TRIALS,
        numeric_noise_std: float = config.MONTE_CARLO_NUMERIC_NOISE_STD,
        max_distribution_points: int = 200,
        random_seed: int = None,
    ) -> MonteCarloResult:
        rng = random.Random(random_seed)
        simulated_states: List[TwinState] = []

        for _ in range(trials):
            perturbed_params = self._perturb_parameters(scenario.parameters, numeric_noise_std, rng)
            perturbed_scenario = Scenario(
                name=scenario.name,
                event_type=scenario.event_type,
                parameters=perturbed_params,
            )
            # transformer clones `state` internally - the real Twin is untouched.
            simulated_states.append(self._transformer.transform(state, perturbed_scenario))

        probabilities = self._predictor.predict_batch(simulated_states)
        arr = np.array(probabilities, dtype=float)

        # Downsample the returned distribution sample for payload size, while
        # keeping full-precision summary statistics computed over all trials.
        if len(arr) > max_distribution_points:
            sample_idx = np.linspace(0, len(arr) - 1, max_distribution_points).astype(int)
            distribution_sample = arr[sample_idx].tolist()
        else:
            distribution_sample = arr.tolist()

        return MonteCarloResult(
            customer_id=state.customer_id,
            scenario_name=scenario.name,
            trials=trials,
            mean=float(np.mean(arr)),
            median=float(np.median(arr)),
            p10=float(np.percentile(arr, 10)),
            p90=float(np.percentile(arr, 90)),
            std_dev=float(np.std(arr)),
            distribution=distribution_sample,
            assumptions={
                "source_of_stochasticity": (
                    "Numeric scenario parameters (e.g. claim_amount, current_premium, "
                    "change_pct, coverage_amount, reduction_pct) are "
                    "perturbed per trial with multiplicative Gaussian noise "
                    f"(mean=1.0, std={numeric_noise_std}) before being applied to a "
                    "cloned Twin state and re-scored by the Random Forest. This "
                    "represents uncertainty in how the hypothetical scenario plays "
                    "out, NOT statistical uncertainty in the model itself."
                ),
                "numeric_noise_std": numeric_noise_std,
                "note": "These are MVP simulation assumptions, not empirically fitted uncertainty parameters.",
            },
        )


monte_carlo_engine = MonteCarloEngine()

```

## `risk_intelligence/predictor.py`

```python risk_intelligence/predictor.py
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from risk_intelligence.feature_mapper import FeatureMappingError, twin_state_to_feature_row
from twin_engine.state.twin_state import TwinState

import config

logger = logging.getLogger(__name__)


class ModelNotAvailableError(RuntimeError):
    """Raised when a request needs the trained model but artifacts are missing."""


@dataclass
class RiskResult:
    customer_id: str
    churn_probability: float
    risk_level: str
    model_version: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "customer_id": self.customer_id,
            "churn_probability": round(self.churn_probability, 4),
            "risk_level": self.risk_level,
            "model_version": self.model_version,
        }


class ChurnPredictor:
    """
    Loads model/churn_model.joblib, model/preprocessing.joblib, and
    model/model_metadata.json if present, and exposes a `predict` method.

    Safe to construct even when artifacts are missing - `is_available`
    will be False and `predict` will raise ModelNotAvailableError with
    instructions, rather than crashing the whole application at import
    time. This lets the rest of the API run (customers list, Twin views,
    events) even before you've dropped in a trained model.
    """

    MISSING_ARTIFACT_MESSAGE = (
        "Trained model artifacts were not found. This endpoint requires exactly:\n"
        f"  - {config.MODEL_PATH}\n"
        f"  - {config.PREPROCESSING_PATH}\n"
        f"  - {config.MODEL_METADATA_PATH}\n"
        f"  - {config.FEATURE_SCHEMA_PATH}\n"
        "These are consumed exactly as provided - the application performs "
        "INFERENCE ONLY and never trains or retrains a model itself. "
        "See docs/dataset-mapping.md and model/README.md for details."
    )

    def __init__(
        self,
        model_path: Path = config.MODEL_PATH,
        preprocessing_path: Path = config.PREPROCESSING_PATH,
        metadata_path: Path = config.MODEL_METADATA_PATH,
        feature_schema_path: Path = config.FEATURE_SCHEMA_PATH,
    ):
        self._model_path = model_path
        self._preprocessing_path = preprocessing_path
        self._metadata_path = metadata_path
        self._feature_schema_path = feature_schema_path

        self.model = None
        self.preprocessing = None
        self.metadata: Dict[str, Any] = {}
        self.feature_schema: Dict[str, Any] = {}

        self._try_load()

    def _try_load(self) -> None:
        if not (self._model_path.exists() and self._preprocessing_path.exists()):
            logger.warning("Model artifacts not found yet. %s", self.MISSING_ARTIFACT_MESSAGE)
            return

        import joblib  # local import: keeps joblib optional if never used

        try:
            self.model = joblib.load(self._model_path)
            self.preprocessing = joblib.load(self._preprocessing_path)
        except Exception:
            logger.exception("Failed to load model artifacts from %s / %s", self._model_path, self._preprocessing_path)
            self.model = None
            self.preprocessing = None
            return

        if self._metadata_path.exists():
            try:
                self.metadata = json.loads(self._metadata_path.read_text())
            except (json.JSONDecodeError, OSError):
                logger.exception("Failed to read model metadata at %s", self._metadata_path)
                self.metadata = {}
        else:
            logger.warning("model_metadata.json not found at %s - proceeding without it.", self._metadata_path)

        if self._feature_schema_path.exists():
            try:
                self.feature_schema = json.loads(self._feature_schema_path.read_text())
            except (json.JSONDecodeError, OSError):
                logger.exception("Failed to read feature schema at %s", self._feature_schema_path)
                self.feature_schema = {}
        else:
            logger.warning("feature_schema.json not found at %s - proceeding without it.", self._feature_schema_path)

    def reload(self) -> None:
        """Re-attempt loading artifacts (e.g. after you drop them in without restarting)."""
        self._try_load()

    @property
    def is_available(self) -> bool:
        return self.model is not None and self.preprocessing is not None

    @property
    def model_version(self) -> str:
        """
        The shipped model_metadata.json (see model/README.md for the
        expected shape) may or may not include an explicit "model_version"
        string. If it doesn't, we build a readable identifier from
        whatever real metadata fields ARE present (model type + training
        timestamp) rather than reporting a meaningless "unversioned".
        """
        if "model_version" in self.metadata:
            return self.metadata["model_version"]
        model_type = self.metadata.get("model_type", "model")
        trained_at = self.metadata.get("training_timestamp")
        return f"{model_type}@{trained_at}" if trained_at else model_type

    @property
    def evaluation_metrics(self) -> Dict[str, Any]:
        """Real, held-out evaluation metrics as recorded in model_metadata.json
        at training time - never computed or invented here."""
        return self.metadata.get("evaluation_metrics", {})

    def _require_model(self) -> None:
        if not self.is_available:
            raise ModelNotAvailableError(self.MISSING_ARTIFACT_MESSAGE)

    def _predict_proba_for_frame(self, feature_frame: pd.DataFrame):
        self._require_model()
        transformed = self.preprocessing.transform(feature_frame)
        # Standard scikit-learn binary classifier convention: column 1 = positive class.
        # model_metadata.json should record which class index/label corresponds to "churn".
        proba = self.model.predict_proba(transformed)
        churn_class_index = self.metadata.get("churn_class_index", 1)
        return proba[:, churn_class_index]

    def predict(self, state: TwinState) -> RiskResult:
        feature_row = twin_state_to_feature_row(state)
        churn_probability = float(self._predict_proba_for_frame(feature_row)[0])
        risk_level = config.risk_level_from_probability(churn_probability)
        return RiskResult(
            customer_id=state.customer_id,
            churn_probability=churn_probability,
            risk_level=risk_level,
            model_version=self.model_version,
        )

    def predict_batch(self, states) -> list:
        """Used by the Monte Carlo engine - one preprocessing/model call for N cloned states."""
        from risk_intelligence.feature_mapper import twin_states_to_feature_frame

        states = list(states)
        if not states:
            return []
        feature_frame = twin_states_to_feature_frame(states)
        probabilities = self._predict_proba_for_frame(feature_frame)
        return [float(p) for p in probabilities]


# Module-level singleton. Constructed once; if artifacts are absent at
# import time, `reload()` can pick them up later without restarting the app.
churn_predictor = ChurnPredictor()

```

## `recommendation_engine/effect_estimator.py`

```python recommendation_engine/effect_estimator.py


from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from recommendation_engine.action_lookup import CandidateAction

import config


@dataclass
class EstimatedEffect:
    action_id: str
    assumed_risk_reduction: float  # absolute reduction in churn probability, e.g. 0.05 = 5 points
    basis: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "assumed_risk_reduction": round(self.assumed_risk_reduction, 4),
            "basis": self.basis,
        }


class EffectEstimator:
    def __init__(self, assumed_effects: Dict[str, float] = None):
        self._assumed_effects = assumed_effects or config.ASSUMED_ACTION_EFFECT

    def estimate(self, action: CandidateAction) -> EstimatedEffect:
        assumed_reduction = self._assumed_effects.get(action.action_id, 0.02)
        return EstimatedEffect(
            action_id=action.action_id,
            assumed_risk_reduction=assumed_reduction,
            basis=(
                "MVP simulation assumption (config.ASSUMED_ACTION_EFFECT) - not "
                "derived from an intervention-outcome dataset. Replace with a "
                "learned uplift model once real intervention data is available."
            ),
        )


effect_estimator = EffectEstimator()

```

## `config.py`

```python config.py


from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
STORAGE_DIR = BASE_DIR / "storage"  # local JSON persistence (created at runtime)

CUSTOMER_DATA_CSV = DATA_DIR / "customer_churn.csv"

MODEL_PATH = MODEL_DIR / "churn_model.joblib"
PREPROCESSING_PATH = MODEL_DIR / "preprocessing.joblib"
MODEL_METADATA_PATH = MODEL_DIR / "model_metadata.json"
FEATURE_SCHEMA_PATH = MODEL_DIR / "feature_schema.json"

TWIN_STORE_PATH = STORAGE_DIR / "twin_states.json"
EVENT_LOG_PATH = STORAGE_DIR / "event_log.json"
RISK_HISTORY_PATH = STORAGE_DIR / "risk_history.json"

# --------------------------------------------------------------------------
# Dataset loading
# --------------------------------------------------------------------------
# How many rows of the public/synthetic CSV to load into the local demo
# store on startup. This is a prototype convenience limit only - the CSV
# itself has 50,000 rows, far more than an MVP dashboard needs to display.
MAX_CUSTOMERS_TO_LOAD = 300

# --------------------------------------------------------------------------
# Risk levels
# ASSUMPTION: these churn-probability cut points are MVP BUSINESS
# thresholds, chosen by looking at the trained model's predicted-probability
# distribution over the real reference dataset (roughly: median ~0.34,
# 75th percentile ~0.56, 90th percentile ~0.74) so that HIGH/MEDIUM/LOW
# produce a usable three-way split. They are NOT scientifically validated,
# NOT derived from a cost/benefit study, and should be revisited once real
# business impact data exists.
# --------------------------------------------------------------------------
RISK_THRESHOLDS = {
    "high": 0.60,    # churn_probability >= 0.60  -> HIGH
    "medium": 0.35,  # 0.35 <= churn_probability < 0.60 -> MEDIUM
    # churn_probability < 0.35 -> LOW
}


def risk_level_from_probability(probability: float) -> str:
    if probability >= RISK_THRESHOLDS["high"]:
        return "HIGH"
    if probability >= RISK_THRESHOLDS["medium"]:
        return "MEDIUM"
    return "LOW"


# --------------------------------------------------------------------------
# Driver identification
# --------------------------------------------------------------------------
# Number of top drivers to surface per customer.
TOP_N_DRIVERS = 3

# --------------------------------------------------------------------------
# Monte Carlo simulation assumptions
# ASSUMPTION: all values below define *simulation uncertainty*, i.e. how
# much a scenario's numeric parameters are randomly perturbed across Monte
# Carlo trials. They are NOT statements about real-world variance and are
# clearly surfaced to the user as configurable assumptions.
# --------------------------------------------------------------------------
MONTE_CARLO_DEFAULT_TRIALS = 300

# Relative (fractional) standard deviation applied to numeric scenario
# parameters during Monte Carlo perturbation, e.g. 0.1 = +/-10% noise
# around the scenario-transformed value.
MONTE_CARLO_NUMERIC_NOISE_STD = 0.10

# --------------------------------------------------------------------------
# Recommendation Engine - Action Lookup
# ASSUMPTION: this rule table is an MVP prototype rule set, written by hand
# from the dataset's available features. It is not a learned policy.
# --------------------------------------------------------------------------
## Keys MUST match the raw dataset column names in
## risk_intelligence.feature_mapper.FEATURE_COLUMNS exactly - the Driver
## Identifier ranks drivers using those same column names. Only columns
## with a plausible, dataset-grounded administrator action are given a
## dedicated rule; everything else falls back to "default".
ACTION_RULES = {
    "missed_payment_flag": {
        "action": "payment_plan_review",
        "label": "Payment plan review",
        "description": (
            "The customer has missed multiple payments in the last 12 "
            "months. Offer a payment plan review (e.g. switch to monthly "
            "autopay) before the next renewal."
        ),
    },
    "late_payment_count_12m": {
        "action": "payment_plan_review",
        "label": "Payment plan review",
        "description": (
            "A rising count of late payments contributes strongly to this "
            "customer's risk assessment; a payment plan or autopay "
            "conversation may help before it escalates."
        ),
    },
    "premium_change_pct": {
        "action": "premium_review",
        "label": "Premium review",
        "description": (
            "A recent premium increase contributes strongly to this "
            "customer's risk assessment; offer a premium/coverage review."
        ),
    },
    "current_premium": {
        "action": "premium_review",
        "label": "Premium review",
        "description": "Offer a premium/coverage review to check the policy still fits the customer's budget.",
    },
    "premium_to_coverage_ratio": {
        "action": "premium_review",
        "label": "Premium review",
        "description": (
            "This customer's premium is high relative to their coverage "
            "amount compared to peers; review pricing and coverage fit."
        ),
    },
    "num_price_increases_last_3y": {
        "action": "premium_review",
        "label": "Premium review",
        "description": "Repeated premium increases over the last 3 years contribute strongly to this customer's risk assessment.",
    },
    "complaint_flag": {
        "action": "service_recovery_outreach",
        "label": "Service recovery outreach",
        "description": (
            "The customer has an open or recent complaint; proactively "
            "follow up to confirm resolution and rebuild confidence."
        ),
    },
    "complaint_resolution_days": {
        "action": "service_recovery_outreach",
        "label": "Service recovery outreach",
        "description": "A slow complaint resolution contributes strongly to this customer's risk assessment; follow up personally.",
    },
    "num_claims_12m": {
        "action": "claims_review_outreach",
        "label": "Claims review & proactive outreach",
        "description": "Contact the customer to review recent claim activity and confirm satisfaction with claims handling.",
    },
    "num_rejected_claims_12m": {
        "action": "claims_review_outreach",
        "label": "Claims review & proactive outreach",
        "description": (
            "The customer has had claim(s) rejected recently, a common "
            "source of dissatisfaction; review the claim decision with them."
        ),
    },
    "avg_settlement_time_days": {
        "action": "claims_review_outreach",
        "label": "Claims review & proactive outreach",
        "description": "Slow claim settlement contributes strongly to this customer's risk assessment; check in on their most recent claim.",
    },
    "payout_ratio_12m": {
        "action": "claims_review_outreach",
        "label": "Claims review & proactive outreach",
        "description": "A low payout-to-claim ratio contributes strongly to this customer's risk assessment; review claims handling with them.",
    },
    "coverage_downgrade_flag": {
        "action": "coverage_review",
        "label": "Coverage review",
        "description": (
            "The customer recently downgraded coverage, often a sign of "
            "price sensitivity; review whether current coverage still "
            "meets their needs."
        ),
    },
    "quote_requested_flag": {
        "action": "retention_offer_review",
        "label": "Retention offer review",
        "description": (
            "The customer has recently requested a quote (a common "
            "shopping-around signal); consider a proactive retention offer."
        ),
    },
    "num_contacts_12m": {
        "action": "engagement_outreach",
        "label": "Customer engagement outreach",
        "description": "This customer's contact/engagement pattern contributes strongly to their risk assessment; reach out to check in.",
    },
    "payment_method_change_flag": {
        "action": "payment_plan_review",
        "label": "Payment plan review",
        "description": "A recent payment-method change contributes strongly to this customer's risk assessment; confirm their new payment details are working smoothly.",
    },
    "default": {
        "action": "general_account_review",
        "label": "General account review",
        "description": "No specific dominant driver identified; perform a general account check-in.",
    },
}

# --------------------------------------------------------------------------
# Recommendation Engine - Effect Estimator
# ASSUMPTION: these are MVP simulation assumptions about how much an action
# is assumed to reduce churn probability. There is no intervention-outcome
# dataset backing these numbers. They exist so the Expected Value Ranker has
# a documented, configurable, and clearly-labelled input to work with, and
# so they can be swapped for a learned uplift/treatment-effect model later.
# --------------------------------------------------------------------------
ASSUMED_ACTION_EFFECT = {
    "payment_plan_review": 0.09,        # assumed absolute churn-probability reduction
    "premium_review": 0.06,
    "service_recovery_outreach": 0.10,
    "claims_review_outreach": 0.07,
    "coverage_review": 0.05,
    "retention_offer_review": 0.08,
    "engagement_outreach": 0.03,
    "general_account_review": 0.02,
}

# --------------------------------------------------------------------------
# Recommendation Engine - Expected Value Ranker
# ASSUMPTION: prototype placeholders for customer lifetime value and action
# cost. Real values should come from Finance / CRM once available.
# --------------------------------------------------------------------------
DEFAULT_CUSTOMER_VALUE = 3000.0  # placeholder assumed customer value (currency units, NZD)
# Chosen as roughly 3x the reference dataset's average current_premium
# (~NZD 1,048/year) as a rough stand-in for a few years of retained premium
# revenue - an MVP placeholder, not a real Finance-supplied CLV figure.

ASSUMED_ACTION_COST = {
    "payment_plan_review": 8.0,
    "premium_review": 10.0,
    "service_recovery_outreach": 20.0,
    "claims_review_outreach": 15.0,
    "coverage_review": 10.0,
    "retention_offer_review": 25.0,
    "engagement_outreach": 5.0,
    "general_account_review": 5.0,
}

# --------------------------------------------------------------------------
# Event generator
# --------------------------------------------------------------------------
EVENT_GENERATOR_DEFAULT_INTERVAL_SECONDS = 5
EVENT_GENERATOR_SCENARIOS = [
    "payment_missed",
    "claim_created",
    "premium_changed",
    "policy_renewed",
    "engagement_changed",
    "coverage_downgraded",
    "complaint_lodged",
]

# --------------------------------------------------------------------------
# API / server
# --------------------------------------------------------------------------
API_HOST = "0.0.0.0"
API_PORT = 8000
CORS_ALLOW_ORIGINS = ["*"]  # MVP only - tighten before any real deployment

```

# Documentation

## `docs/architecture.md`

```markdown docs/architecture.md
# Architecture

## What this MVP is, and isn't

This is a simplified, local, single-process implementation of the
Customer Twin concept from the technical design document. It keeps the
**conceptual architecture** (Digital Twin Engine at the center; event flow
producer → transition handler → state store → risk recalculation;
Kafka-replaceable event ingestion) while deliberately removing
infrastructure that isn't needed to demonstrate that architecture:
no Kafka, no Docker, no Kubernetes, no Redis, no microservices, no
authentication, no PostgreSQL. See the technical design document's own
Section "Design Principles" for the production version of this system —
this repository intentionally implements a scaled-down version of it.

## Component diagram (as implemented)

```
                 REAL-TIME EVENT GENERATOR (local Python thread)
                          |  produces Event objects
                          v
        POST /events  ---+
                          v
            +------------------------------+
            |     DIGITAL TWIN ENGINE       |
            |  (twin_engine/)               |
            |                               |
            |  Event Transition Handler     |  S_(t+1) = f(S_t, E_t)
            |            |                  |
            |            v                  |
            |  Twin State Store             |  local JSON file, source of truth
            |            |                  |
            |  State Synchronizer           |  sequences: apply -> persist -> recalc
            |                               |
            |  Scenario Transformer         |  S'_t = T(S_t, theta), clone-only
            |            |                  |
            |            v                  |
            |  Monte Carlo Engine           |  outcome distribution, real Twin untouched
            +---------------+---------------+
                            |
                            v
               +----------------------+
               |  RISK INTELLIGENCE   |   (risk_intelligence/)
               |                      |
               |  Feature Mapper      |
               |  Random Forest       |   loaded from model/*.joblib — never trained here
               |  Churn Probability   |
               |  Risk Level          |
               |  Driver Identifier   |   real feature_importances_, real dataset stats
               +----------+-----------+
                          |
                          v
            +-----------------------------+
            |   RECOMMENDATION ENGINE      |  (recommendation_engine/)
            |                              |
            | Driver Identifier (shared)   |
            | Action Lookup                |
            | Effect Estimator             |
            | Expected Value Ranker        |
            +-------------+---------------+
                          |
                          v
            +-----------------------------+
            |      SIMPLE FRONTEND        |   (frontend/) — vanilla HTML/CSS/JS
            |      INSURISE-STYLE         |   served by FastAPI's StaticFiles
            +-----------------------------+
                          ^
                          |
                     FastAPI (api/main.py) — thin handlers only
```

## Why this stays swappable for the "real" (Kafka) architecture later

- `Event` (`twin_engine/events/event.py`) is a plain, transport-agnostic
  dataclass. It doesn't know whether it was constructed by the local
  `EventGenerator`, the `POST /events` endpoint, or (later) a Kafka
  consumer. Swapping the event source for a real Kafka consumer means
  writing a new adapter that constructs the same `Event` objects and
  calls the same `StateSynchronizer.process_event` — nothing in
  `twin_engine/`, `risk_intelligence/`, or `recommendation_engine/` needs
  to change.
- The Twin State Store's public interface (`get` / `save` / `bulk_save` /
  `list_all`) is the seam a future PostgreSQL-backed implementation would
  preserve.
- The Scenario Transformer and Monte Carlo Engine never call
  `TwinStateStore.save()` — only the `StateSynchronizer`, driven by real
  events, is allowed to persist. This is what makes "the real Twin is
  never modified by a what-if simulation" an enforced invariant rather
  than a convention.

## Module map

| Module | Responsibility |
|---|---|
| `twin_engine/state/` | `TwinState` (S_t) and the `TwinStateStore` (source of truth) |
| `twin_engine/events/` | `Event`, `EventTransitionHandler` (f(S_t, E_t)), `EventGenerator` |
| `twin_engine/synchronization/` | `StateSynchronizer` — sequences event → persist → risk recalc |
| `twin_engine/simulation/` | `ScenarioTransformer` (S'_t = T(S_t, theta)), `MonteCarloEngine` |
| `risk_intelligence/` | Feature mapping, Random Forest loading/inference, driver identification |
| `recommendation_engine/` | Action lookup, effect estimation, expected-value ranking |
| `api/` | FastAPI app — thin request/response handlers only |
| `frontend/` | Static HTML/CSS/JS admin portal, served by the same FastAPI app |
| `model/` | Where you place your trained `churn_model.joblib` / `preprocessing.joblib` / `model_metadata.json` |
| `data/` | `customer_churn.csv` — the public prototype dataset |
| `docs/` | This documentation |

## Persistence (see also docs/simulation.md and dataset-mapping.md)

No database is used. `storage/twin_states.json` (created at runtime) is
the entire Twin State Store; `storage/event_log.json` is a simple
append-only audit log of every event processed. Both are plain local
JSON, deliberately simple per the MVP simplification rules. A future
production version would replace `TwinStateStore`'s internals with
PostgreSQL (JSONB for the twin snapshot, relational tables for
customer/policy/claim) without changing its public interface — see the
technical design document's own Database Architecture section for that
target design.

## Explicitly out of scope for this MVP (see technical design doc for the production path)

Kafka, Docker/Kubernetes, Redis, microservices, GraphQL/gRPC,
authentication/authorization, cloud infrastructure, OpenTelemetry,
automated MLOps/model registry/retraining, a full automated test suite,
and production deployment infrastructure. All of these are described in
the original technical design document as the production-scale
evolution of this same conceptual architecture — none of them change how
the Digital Twin Engine itself is structured.

```

## `docs/twin-engine.md`

```markdown docs/twin-engine.md
# Digital Twin Engine

The Digital Twin Engine (`twin_engine/`) is the core of this application.
Everything else (Risk Intelligence, Recommendation Engine, frontend)
reads from or reacts to it — it does not depend on them.

## Twin State Store (`twin_engine/state/`)

`TwinState` (`twin_state.py`) represents S_t — one customer's current
virtual state. Its fields map directly onto
`data/customer_churn.csv` and the trained model's exact feature schema
(`model/feature_schema.json`, `model/model_metadata.json`) — see
`docs/dataset-mapping.md` for the full column mapping. Fields fall into
four categories, documented at the top of `twin_state.py`:

- **Static** (identity/profile, never mutated by any event in this MVP):
  `age`, `region_name`, `marital_status`, `customer_tenure_months`,
  `multi_policy_flag`, `num_policies`, `policy_type`, `renewal_month`,
  `payment_frequency`, `autopay_enabled`.
- **Dynamic** (mutated by one or more of the 7 events — see
  `docs/event-model.md`): `current_premium`, `premium_last_year`,
  `num_price_increases_last_3y`, `coverage_amount`,
  `late_payment_count_12m`, the claims counters/totals, `num_contacts_12m`,
  `complaint_flag`, `complaint_resolution_days`, `quote_requested_flag`,
  `coverage_downgrade_flag`.
- **ML-only** (fed to the model, present in Twin state, but no
  dataset-grounded event mutates it in this MVP): `payment_method_change_flag`.
- **Derived** (recomputed properties, never stored independently, so they
  can never silently drift out of sync with their inputs):
  `premium_change_pct`, `premium_to_coverage_ratio`, `payout_ratio_12m`,
  `missed_payment_flag`.

`TwinStateStore` (`state_store.py`) is the single source of truth for
every customer's current Twin. For this MVP it's an in-memory dictionary
persisted to a local JSON file (`storage/twin_states.json`) — no database
infrastructure, per the simplification rules. Its public interface
(`get`, `save`, `bulk_save`, `list_all`, `exists`) is intentionally the
only seam the rest of the app touches, so it can be swapped for a
PostgreSQL-backed implementation later without touching any calling code.

`TwinState.clone()` performs a deep copy — this is what makes the
Scenario Transformer's "never mutate the real Twin" guarantee possible.

## Event Transition Handler (`twin_engine/events/transition_handler.py`)

Implements S_(t+1) = f(S_t, E_t). Each of the 7 supported event types is a
small, pure function that receives the current state and an `Event` and
mutates specific fields — documented inline with its affected Twin state
field(s), affected ML feature(s), and expected state change. Full
per-event contract: **`docs/event-model.md`**.

Every applied event is appended to `TwinState.event_history` (capped at
the last 50 entries for the MVP) with a human-readable description, and
`TwinState.version` / `updated_at` are bumped.

The 7 event types exist because they're grounded in the actual dataset —
see `docs/dataset-mapping.md` for the full Dataset → Twin → ML feature →
Event mapping table.

## State Synchronization (`twin_engine/synchronization/synchronizer.py`)

`StateSynchronizer.process_event(event)` is the one place that sequences:

```
Event -> Twin state update (Event Transition Handler)
      -> Persist/update current state (Twin State Store)
      -> Risk recalculation (callback into Risk Intelligence)
```

Both the `POST /events` endpoint and the local `EventGenerator` call this
same method — there is exactly one path an event can take through the
system, whatever produced it. No distributed synchronization is
implemented (no locks, no distributed consensus) — this is explicitly
out of scope for the MVP; the Twin State Store is the single process's
single source of truth.

## Scenario Transformer (`twin_engine/simulation/scenario_transformer.py`)

Implements S'_t = T(S_t, theta):

1. `state.clone()` — a deep copy, fully independent of the real state.
2. Apply the scenario (reusing the same `EventTransitionHandler` logic a
   real event of that type would use) to the clone.
3. Return the transformed clone.

**Enforced invariant:** this class never calls `TwinStateStore.save()`.
Only `StateSynchronizer`, acting on real events, is allowed to persist
state. This is what guarantees a what-if scenario can never leak into the
real Twin, rather than merely being a convention callers are expected to
follow.

## Monte Carlo Engine (`twin_engine/simulation/monte_carlo.py`)

See `docs/simulation.md` for the full explanation of deterministic vs.
Monte Carlo simulation and the uncertainty assumptions involved. In brief:
the Monte Carlo Engine runs many independent scenario transformations
(each on its own clone of the real state, with the scenario's numeric
monetary/percentage parameters perturbed by configurable Gaussian noise),
scores each resulting clone with the Random Forest in a single batched
call, and reports the resulting distribution (mean, median, P10, P90,
std dev).

## Real-Time Event Generator (`twin_engine/events/event_generator.py`)

A daemon background thread that periodically picks a random customer from
the Twin State Store and a random scenario from
`config.EVENT_GENERATOR_SCENARIOS`, builds an `Event`, and calls
`StateSynchronizer.process_event` — the exact same path `POST /events`
uses. It never touches the ML model or a risk score directly; only the
Random Forest (via Risk Intelligence) determines the resulting risk. This
keeps it a drop-in replacement target for a real Kafka consumer later:
whatever produces the event, the rest of the pipeline behaves identically.

Controlled via `POST /api/event-generator/start` /
`POST /api/event-generator/stop` / `GET /api/event-generator/status`, or
by running `python -m twin_engine.events.event_generator` as its own
standalone local process (see the README for exact commands).

```

## `docs/event-model.md`

```markdown docs/event-model.md
# Event Model

Every event supported by this MVP is grounded in an actual column of
`data/customer_churn.csv` — see `docs/dataset-mapping.md` for the full
column analysis. This document lists each event's contract.

## `payment_missed`

- **Affected Twin state:** `late_payment_count_12m`
- **Affected ML feature(s):** `late_payment_count_12m`, derived `missed_payment_flag`
- **Expected state change:** `late_payment_count_12m += payload.count`
  (default 1). `missed_payment_flag` updates automatically (it's a derived
  property: `1` once `late_payment_count_12m >= 4`, matching the exact
  rule in `data/data_dictionary.csv`).
- **Payload:** `{"count": <int, optional, default 1>}`

## `claim_created`

- **Affected Twin state:** `num_claims_12m`,
  `num_approved_claims_12m`/`num_rejected_claims_12m`/`num_pending_claims_12m`
  (whichever matches `outcome`), `total_claim_amount_12m`,
  `avg_claim_amount`, `total_payout_amount_12m` (if approved),
  `avg_settlement_time_days`, `days_since_last_claim`
- **Affected ML feature(s):** all of the above, plus derived `payout_ratio_12m`
- **Expected state change:** a new claim is filed and accumulates into
  the running 12-month totals (this dataset's claims columns are
  trailing-12-month aggregates, so `claim_created` increments/accumulates
  rather than replacing — see `docs/dataset-mapping.md`).
  `avg_claim_amount` is recomputed as `total_claim_amount_12m /
  num_claims_12m`. `days_since_last_claim` resets to `0`.
- **Payload:** `{"claim_amount": <number>, "outcome": "approved"|"rejected"|"pending" (default "approved"), "settlement_time_days": <int, optional>, "payout_fraction": <float, optional, default 1.0 if approved else 0.0>}`

## `premium_changed`

- **Affected Twin state:** `current_premium`, `num_price_increases_last_3y`
- **Affected ML feature(s):** `current_premium`,
  `num_price_increases_last_3y`, derived `premium_change_pct` and
  `premium_to_coverage_ratio`
- **Expected state change:** `current_premium` moves to a new value,
  given either directly (`current_premium`) or as a relative change
  (`change_pct`, e.g. `0.15` for +15%) against the current premium.
  `num_price_increases_last_3y` increments only if the new premium is
  higher than the old one.
- **Payload:** `{"current_premium": <number>}` **or**
  `{"change_pct": <number>}` (e.g. `-0.05` for a 5% decrease)

## `policy_renewed`

- **Affected Twin state:** `premium_last_year`, and a fresh
  trailing-12-month window: `late_payment_count_12m`, `num_claims_12m`,
  `num_approved_claims_12m`/`num_rejected_claims_12m`/`num_pending_claims_12m`,
  `total_claim_amount_12m`, `total_payout_amount_12m`, `avg_claim_amount`,
  `num_contacts_12m`, `complaint_flag`, `complaint_resolution_days`
- **Affected ML feature(s):** all of the above, plus every derived
  feature that depends on them (`premium_change_pct`,
  `missed_payment_flag`, `payout_ratio_12m`)
- **Expected state change:** `current_premium` rolls into
  `premium_last_year` (so a subsequent `premium_changed` event compares
  against it), and every trailing-12-month counter resets to zero —
  see the ASSUMPTION in `docs/dataset-mapping.md` about why.
- **Payload:** none required.

## `engagement_changed`

- **Affected Twin state:** `num_contacts_12m`, `quote_requested_flag`
- **Affected ML feature(s):** `num_contacts_12m`, `quote_requested_flag`
- **Expected state change:** `num_contacts_12m += payload.contact_delta`
  (default 1, floored at 0). If `payload.quote_requested` is true,
  `quote_requested_flag -> 1` (a customer shopping around for a quote is a
  distinct, meaningful signal from a routine contact).
- **Payload:** `{"contact_delta": <int, optional, default 1>, "quote_requested": <bool, optional>}`

## `coverage_downgraded`

- **Affected Twin state:** `coverage_amount`, `coverage_downgrade_flag`
- **Affected ML feature(s):** `coverage_amount`,
  `coverage_downgrade_flag`, derived `premium_to_coverage_ratio`
- **Expected state change:** `coverage_amount` decreases, given either
  directly (`coverage_amount`) or as a fractional cut (`reduction_pct`,
  e.g. `0.2` for -20%; defaults to a 20% cut if neither is given).
  `coverage_downgrade_flag -> 1`.
- **Payload:** `{"coverage_amount": <number>}` **or**
  `{"reduction_pct": <number>}`

## `complaint_lodged`

- **Affected Twin state:** `complaint_flag`, `complaint_resolution_days`
- **Affected ML feature(s):** `complaint_flag`, `complaint_resolution_days`
- **Expected state change:** `complaint_flag -> 1`;
  `complaint_resolution_days` set from the payload (default 0 = not yet
  resolved/unknown).
- **Payload:** `{"resolution_days": <int, optional, default 0>}`

## Event flow

```
Event Generator (local thread)  --\
                                    +--> Event --> Event Transition Handler --> Twin State Store --> Risk recalculation
POST /api/events (manual/API)   --/
```

Both producers construct the same `Event` dataclass
(`twin_engine/events/event.py`) and hand it to
`StateSynchronizer.process_event` — there is exactly one code path an
event travels through regardless of source. This is deliberate: it's the
seam that would let a real Kafka consumer replace the local generator
later without touching the Event Transition Handler, Twin State Store, or
anything downstream (see docs/architecture.md).

Every processed event is appended (in order) to that customer's
`TwinState.event_history` (visible via `GET /api/customers/{id}/events`
and the Digital Twin frontend view) and to the flat audit log at
`storage/event_log.json`.

```

## `docs/dataset-mapping.md`

```markdown docs/dataset-mapping.md

# Dataset Mapping

## Source

`data/customer_churn.csv` — the **insurance policyholder churn dataset**
(50,000 rows, 39 columns), provided together with the trained model
artifacts in `model/`. Per `model/model_metadata.json`'s own notes: *"a
public/synthetic proxy dataset shaped like Insurise's domain, not real
policyholder records. Treat metrics as directional; re-train on real
Insurise data before production use."* This is **not real Insurise
customer data**. `data/data_dictionary.csv` (also provided) documents
every column's intended meaning and is the authoritative source for the
column descriptions below.

> This supersedes the previous prototype dataset (`randomdata.csv`, a much
> smaller 11-column synthetic file used earlier in this project's history).
> If you see references to that older schema (`Claim Reason`, `BMI`,
> `Data confidentiality`, ...) elsewhere, they describe a prior integration
> and no longer apply — the current Twin schema is entirely driven by this
> dataset and `model/feature_schema.json`.

## Columns actually present

39 columns total. `customer_id` is a real per-row identifier (unlike the
previous dataset). Five columns are explicitly excluded from the model
per `model/model_metadata.json`'s `excluded_features`:
`customer_id`, `as_of_date`, `age_band`, `churn_type`,
`churn_probability_true`. The remaining 34 columns are the model's exact
feature schema (30 numerical + 4 categorical — see `model/README.md`).
`churn_flag` is the target.

## Identified relationships (real analysis, run against the actual file and the actual loaded model)

- **`missed_payment_flag` is an exact rule on `late_payment_count_12m`**:
  every row with `missed_payment_flag == 1` has
  `late_payment_count_12m >= 4`, and every row with `== 0` has
  `late_payment_count_12m < 4` (mean 4.24 vs 0.48) — this matches the data
  dictionary's own description verbatim ("1 if missed payments flag
  (>=4 late payments), else 0"). `TwinState.missed_payment_flag` is
  implemented as a derived property using this exact rule rather than
  stored independently, so it can never drift out of sync.
- **`num_claims_12m` is an exact sum**:
  `num_claims_12m == num_approved_claims_12m + num_rejected_claims_12m +
  num_pending_claims_12m` for all 50,000 rows (verified, zero mismatches).
- **`total_payout_amount_12m` is an exact product**:
  `total_payout_amount_12m == total_claim_amount_12m * payout_ratio_12m`
  wherever `total_claim_amount_12m > 0` (verified to within floating-point
  rounding). Where `total_claim_amount_12m == 0` (86.6% of rows —
  customers with no claims in the trailing 12 months), `payout_ratio_12m`
  still carries a baseline value in the 0.75–0.85 range rather than an
  undefined `0/0`. `TwinState.payout_ratio_12m` is a derived property
  implementing this exact relationship, with a documented `0.75` default
  for the zero-claim case (see `twin_engine/state/twin_state.py`).
- **`premium_to_coverage_ratio` is an exact ratio**:
  `premium_to_coverage_ratio == current_premium / coverage_amount`
  (verified to ~2.5e-6, floating-point rounding). Implemented as a derived
  `TwinState` property, never stored independently.
- **`premium_change_pct` is NOT a pure derived duplicate of
  `current_premium`/`premium_last_year`** — the source data carries
  independent noise beyond the exact formula `(current - last) / last`
  (observed deviation up to ~0.22 in a sample check). For Twin simulation
  purposes, `TwinState.premium_change_pct` recomputes the exact formula
  from `current_premium`/`premium_last_year` so a simulated premium change
  always produces an internally consistent feature vector — a documented
  MVP modeling choice, not a claim that it reproduces the original noise.
- **`current_premium` and `premium_last_year` are highly correlated**
  (Pearson r ≈ 0.98) — expected (premiums don't usually swing wildly
  year-over-year) but worth knowing when reading driver rankings that
  involve both.
- **`age_band` is a strict, deterministic bucketing of `age`** (18–24,
  25–34, ..., 75+) — correctly excluded from the model's feature set
  already (per `model_metadata.json`); not stored in the Twin at all.

**No BMI-style leakage this time.** Unlike the previous prototype
dataset, this one's feature-target relationships look realistic: e.g.
`missed_payment_flag=1` customers churn at 89.8% vs. 29.8% for
`missed_payment_flag=0` (a strong, plausible signal — missing payments is
a textbook churn precursor, not a data artifact); `complaint_flag=1`
customers churn at 55.1% vs. 29.1%; and churn rate rises smoothly with
`premium_change_pct` (17.4% for a >5% premium cut, up to 45.3% for a >10%
increase). The trained model's held-out metrics (accuracy 0.749, ROC-AUC
0.786 — see `model/README.md`) are realistic for this kind of problem,
not suspiciously perfect.

## Mapping: Dataset feature → Twin state → ML feature → Event

| Dataset feature | Twin state field | Category | Triggering event(s) |
|---|---|---|---|
| `customer_id` | `customer_id` | static, identity only (not an ML feature) | — |
| `age` | `age` | static, ML feature | — |
| `region_name` | `region_name` | static, ML feature | — |
| `marital_status` | `marital_status` | static, ML feature | — |
| `customer_tenure_months` | `customer_tenure_months` | static, ML feature | — |
| `multi_policy_flag` | `multi_policy_flag` | static, ML feature | — |
| `num_policies` | `num_policies` | static, ML feature | — |
| `policy_type` | `policy_type` | static, ML feature | — |
| `renewal_month` | `renewal_month` | static, ML feature | — |
| `payment_frequency` | `payment_frequency` | static, ML feature | — |
| `autopay_enabled` | `autopay_enabled` | static, ML feature | — |
| `current_premium` | `current_premium` | dynamic, ML feature | `premium_changed`, `policy_renewed` |
| `premium_last_year` | `premium_last_year` | dynamic, ML feature | `policy_renewed` (rolls `current_premium` forward) |
| `premium_change_pct` | *(derived property)* | derived, ML feature | recomputed whenever `current_premium`/`premium_last_year` change |
| `num_price_increases_last_3y` | `num_price_increases_last_3y` | dynamic, ML feature | `premium_changed` (if the new premium is higher) |
| `coverage_amount` | `coverage_amount` | dynamic, ML feature | `coverage_downgraded` |
| `premium_to_coverage_ratio` | *(derived property)* | derived, ML feature | recomputed whenever `current_premium`/`coverage_amount` change |
| `late_payment_count_12m` | `late_payment_count_12m` | dynamic, ML feature | `payment_missed`; reset by `policy_renewed` |
| `missed_payment_flag` | *(derived property)* | derived, ML feature | recomputed from `late_payment_count_12m >= 4` |
| `payment_method_change_flag` | `payment_method_change_flag` | **ML-only** (fed to model; no dataset-grounded event mutates it in this MVP) | — |
| `num_claims_12m`, `num_approved_claims_12m`, `num_rejected_claims_12m`, `num_pending_claims_12m` | same names | dynamic, ML features | `claim_created`; reset by `policy_renewed` |
| `avg_claim_amount` | `avg_claim_amount` | dynamic, ML feature (recomputed as `total_claim_amount_12m / num_claims_12m` on each claim) | `claim_created`; reset by `policy_renewed` |
| `total_claim_amount_12m` | `total_claim_amount_12m` | dynamic, ML feature | `claim_created`; reset by `policy_renewed` |
| `total_payout_amount_12m` | `total_payout_amount_12m` | dynamic, ML feature | `claim_created` (approved claims only); reset by `policy_renewed` |
| `payout_ratio_12m` | *(derived property)* | derived, ML feature | recomputed from `total_payout_amount_12m`/`total_claim_amount_12m` |
| `avg_settlement_time_days` | `avg_settlement_time_days` | dynamic, ML feature (overwritten by the latest claim's settlement time — a simplification; see `docs/event-model.md`) | `claim_created` |
| `days_since_last_claim` | `days_since_last_claim` | dynamic, ML feature | `claim_created` (resets to 0) |
| `num_contacts_12m` | `num_contacts_12m` | dynamic, ML feature | `engagement_changed`; reset by `policy_renewed` |
| `complaint_flag` | `complaint_flag` | dynamic, ML feature | `complaint_lodged`; reset by `policy_renewed` |
| `complaint_resolution_days` | `complaint_resolution_days` | dynamic, ML feature | `complaint_lodged`; reset by `policy_renewed` |
| `quote_requested_flag` | `quote_requested_flag` | dynamic, ML feature | `engagement_changed` (optional payload flag) |
| `coverage_downgrade_flag` | `coverage_downgrade_flag` | dynamic, ML feature | `coverage_downgraded` |
| `as_of_date` | *(not stored)* | excluded | — |
| `age_band` | *(not stored)* | excluded (derived from `age`, redundant) | — |
| `churn_type` | *(not stored)* | excluded (post-hoc label metadata) | — |
| `churn_probability_true` | *(not stored)* | excluded (ground-truth generation artifact) | — |
| `churn_flag` | `historical_churn_label` | reference/display only, **never an ML feature** | — |

## Assumptions made explicit

1. **The dataset represents one running claim ledger, not per-claim
   records.** `claim_created` increments counts/totals and recomputes
   averages rather than replacing them — a more realistic accumulation
   model than the previous dataset integration allowed for, since this
   dataset's claims columns are genuinely trailing-12-month aggregates.
2. **`policy_renewed` resets every trailing-12-month counter** (late
   payments, claims, complaints, contacts) to model the start of a fresh
   reporting period. A real system might carry some of this history
   forward across a renewal; this MVP resets it to match how the
   dataset's "_12m" columns are framed. Documented, not hidden.
3. **`avg_settlement_time_days` is overwritten, not averaged**, by each
   new claim's settlement time — a simplification that avoids
   implementing full historical claim-by-claim tracking for the MVP.
4. **`payment_method_change_flag` has no dedicated event** in this MVP —
   it's fed to the model as an "ML-only" feature (see the mapping table)
   that starts at its bootstrap value from the dataset and doesn't change
   during a demo session. A future iteration could add a
   `payment_method_changed` event following the same pattern as the other
   seven.

## What is NOT invented

- The four model artifacts (`churn_model.joblib`, `preprocessing.joblib`,
  `model_metadata.json`, `feature_schema.json`) were provided pre-trained
  and are used exactly as given — this repository never trains, retrains,
  or modifies them (see `model/README.md`).
- `model_metadata.json`'s `evaluation_metrics` (accuracy 0.7489, precision
  0.5873, recall 0.5641, F1 0.5755, ROC-AUC 0.7856) are the real,
  reported held-out metrics from the training run — not recomputed,
  invented, or adjusted anywhere in this codebase.
- No customer value, action cost, or action-effect number in `config.py`
  is claimed to be real business data — each is labelled as an "MVP
  simulation assumption" in the code, this documentation, and the API
  responses that use it (see `docs/recommendation-engine.md`).

```

## `docs/simulation.md`

```markdown docs/simulation.md
# Simulation

Two kinds of what-if simulation are implemented: **deterministic**
(`POST /api/customers/{id}/simulate`) and **Monte Carlo**
(`POST /api/customers/{id}/simulate/monte-carlo`). Both share the same
guarantee: **the real Twin state is never modified.**

## Why the real Twin is never modified

`ScenarioTransformer.transform(state, scenario)` always calls
`state.clone()` (a deep copy) before applying anything, and never calls
`TwinStateStore.save(...)`. Only `StateSynchronizer`, driven by real
events (`POST /events` or the Event Generator), is allowed to persist
state. So a what-if simulation literally has no code path available to it
that could write back to the store — this isn't just a convention, it's
enforced by which objects have a reference to the store at all.

Every simulation response also includes `"real_twin_modified": false`, and
the automated verification described in this repository's development
process re-fetched the Twin state after both a deterministic and a Monte
Carlo simulation and confirmed the version number and every field were
byte-for-byte unchanged.

## Deterministic simulation

1. Read the customer's current Twin state and current risk (`before`).
2. Clone the state and apply the requested scenario
   (`ScenarioTransformer`).
3. Score the clone with the Random Forest (`after`).
4. Return `before`, `after`, and `difference` (after − before).

Example request:

```json
POST /api/customers/C000010/simulate
{
  "scenario": "premium_changed",
  "parameters": { "change_pct": 0.15 }
}
```

The result comes from the actual loaded Random Forest — nothing here is
hard-coded. If no model is loaded, this endpoint returns `503` with an
explanation instead of a fabricated number.

## Monte Carlo simulation

The Digital Twin Engine's Monte Carlo Engine
(`twin_engine/simulation/monte_carlo.py`) runs the scenario many times
(default `config.MONTE_CARLO_DEFAULT_TRIALS = 300`, configurable per
request) and reports the resulting **outcome distribution**, not a single
number: mean, median, P10, P90, standard deviation, and a (downsampled, for
payload size) sample of the raw trial-level probabilities for histogram
rendering.

### Separating ML prediction from simulation uncertainty

This separation is deliberate and is surfaced directly in the API
response's `assumptions` field:

1. **ML prediction** — for one exact feature vector, the Random Forest's
   `churn_probability` is a real, deterministic model output.
2. **Simulation uncertainty** — the Random Forest itself does not define a
   distribution over future states. For this MVP, the distribution comes
   entirely from a documented, configurable perturbation: numeric scenario
   parameters (e.g. `claim_amount`, `current_premium`, `change_pct`,
   `coverage_amount`, `reduction_pct`) are perturbed per trial
   with multiplicative Gaussian noise (`mean=1.0`,
   `std=config.MONTE_CARLO_NUMERIC_NOISE_STD`, default `0.10`) before being
   applied to a cloned Twin state and re-scored.

This is an **MVP simulation assumption about how a hypothetical scenario
might play out**, not a statistical property of the customer or the model.
The API response says so explicitly (`assumptions.source_of_stochasticity`
and `assumptions.note`) so nobody mistakes the resulting spread for a
scientifically-derived confidence interval.

Example request:

```json
POST /api/customers/C000010/simulate/monte-carlo
{
  "scenario": "premium_changed",
  "parameters": { "change_pct": 0.15 },
  "trials": 300,
  "numeric_noise_std": 0.10
}
```

### Why batched

`MonteCarloEngine.run` builds all `trials` cloned/transformed states first,
then calls `ChurnPredictor.predict_batch(...)` once — a single
`preprocessing.transform(...)` + `model.predict_proba(...)` call over an
N-row DataFrame, rather than N separate calls. This matters more as
`trials` grows; scikit-learn's vectorized inference is materially faster
batched than looped.

```

# Storage format sample

The following is the actual first two customer entries from `storage/twin_states.json`, pretty-printed without redaction.

```json storage/twin_states.json
{
    "C000001":  {
                    "customer_id":  "C000001",
                    "age":  24,
                    "region_name":  "Manawatu-Whanganui",
                    "marital_status":  "Married",
                    "customer_tenure_months":  128,
                    "multi_policy_flag":  1,
                    "num_policies":  4,
                    "policy_type":  "Auto",
                    "renewal_month":  8,
                    "payment_frequency":  "Monthly",
                    "autopay_enabled":  1,
                    "current_premium":  1884.511789357767,
                    "premium_last_year":  1946.8096997497594,
                    "num_price_increases_last_3y":  8,
                    "coverage_amount":  2379.59229340225,
                    "late_payment_count_12m":  0,
                    "num_claims_12m":  0,
                    "num_approved_claims_12m":  0,
                    "num_rejected_claims_12m":  0,
                    "num_pending_claims_12m":  0,
                    "total_claim_amount_12m":  0.0,
                    "total_payout_amount_12m":  0.0,
                    "avg_claim_amount":  0.0,
                    "avg_settlement_time_days":  18,
                    "days_since_last_claim":  0,
                    "num_contacts_12m":  0,
                    "complaint_flag":  0,
                    "complaint_resolution_days":  0,
                    "quote_requested_flag":  1,
                    "coverage_downgrade_flag":  1,
                    "payment_method_change_flag":  0,
                    "version":  55,
                    "created_at":  "2026-08-24T11:12:01.049569+00:00",
                    "updated_at":  "2026-09-01T09:29:54.803709+00:00",
                    "event_history":  [
                                          {
                                              "event_id":  "d9891638-77b3-44a5-ac7d-1e2295d4678d",
                                              "event_type":  "policy_renewed",
                                              "occurred_at":  "2026-08-31T09:27:20.660410+00:00",
                                              "payload":  {

                                                          },
                                              "description":  "Policy renewed (trailing 12-month counters reset for new period)"
                                          },
                                          {
                                              "event_id":  "7d829fc3-04da-4d28-97ee-aa09945fa3a8",
                                              "event_type":  "premium_changed",
                                              "occurred_at":  "2026-08-31T09:35:27.813975+00:00",
                                              "payload":  {
                                                              "change_pct":  -0.036
                                                          },
                                              "description":  "Premium changed (1017.52 -\u003e 980.89)"
                                          },
                                          {
                                              "event_id":  "b12601b3-cccc-4969-840b-1bfdaad66a41",
                                              "event_type":  "premium_changed",
                                              "occurred_at":  "2026-08-31T09:40:40.902239+00:00",
                                              "payload":  {
                                                              "change_pct":  -0.037
                                                          },
                                              "description":  "Premium changed (980.89 -\u003e 944.60)"
                                          },
                                          {
                                              "event_id":  "1fbf1465-242e-4993-bb86-0f3583fa05de",
                                              "event_type":  "payment_missed",
                                              "occurred_at":  "2026-08-31T10:36:08.362280+00:00",
                                              "payload":  {
                                                              "count":  1
                                                          },
                                              "description":  "Payment missed (late_payment_count_12m -\u003e 1)"
                                          },
                                          {
                                              "event_id":  "2bf5e0d2-669f-4f45-a406-c3d9993406cc",
                                              "event_type":  "payment_missed",
                                              "occurred_at":  "2026-08-31T10:49:56.123254+00:00",
                                              "payload":  {
                                                              "count":  1
                                                          },
                                              "description":  "Payment missed (late_payment_count_12m -\u003e 2)"
                                          },
                                          {
                                              "event_id":  "29c078bf-d7b6-41b3-b603-4ab0108c4a55",
                                              "event_type":  "coverage_downgraded",
                                              "occurred_at":  "2026-08-31T10:53:06.100115+00:00",
                                              "payload":  {
                                                              "reduction_pct":  0.21
                                                          },
                                              "description":  "Coverage downgraded (8924.09 -\u003e 7050.03)"
                                          },
                                          {
                                              "event_id":  "81c27c4e-ffb4-4c20-aa85-bd5160349da3",
                                              "event_type":  "policy_renewed",
                                              "occurred_at":  "2026-08-31T11:14:42.939389+00:00",
                                              "payload":  {

                                                          },
                                              "description":  "Policy renewed (trailing 12-month counters reset for new period)"
                                          },
                                          {
                                              "event_id":  "c9152f2a-472a-4505-bae3-c6d05912a7de",
                                              "event_type":  "claim_created",
                                              "occurred_at":  "2026-08-31T11:38:40.390823+00:00",
                                              "payload":  {
                                                              "claim_amount":  5284.1,
                                                              "outcome":  "approved",
                                                              "settlement_time_days":  22
                                                          },
                                              "description":  "New claim filed (amount=5284.10, outcome=approved); num_claims_12m 0 -\u003e 1"
                                          },
                                          {
                                              "event_id":  "3fe2f6dc-4be6-4b30-b86b-1617b197ded2",
                                              "event_type":  "claim_created",
                                              "occurred_at":  "2026-08-31T12:00:28.469843+00:00",
                                              "payload":  {
                                                              "claim_amount":  6773.58,
                                                              "outcome":  "rejected",
                                                              "settlement_time_days":  17
                                                          },
                                              "description":  "New claim filed (amount=6773.58, outcome=rejected); num_claims_12m 1 -\u003e 2"
                                          },
                                          {
                                              "event_id":  "e8f9cbee-7076-4d9b-b33c-d714b1477c1f",
                                              "event_type":  "complaint_lodged",
                                              "occurred_at":  "2026-08-31T12:51:52.110738+00:00",
                                              "payload":  {
                                                              "resolution_days":  4
                                                          },
                                              "description":  "Complaint lodged (resolution_days=4)"
                                          },
                                          {
                                              "event_id":  "14582b4d-1ce9-4089-9889-a786641281da",
                                              "event_type":  "complaint_lodged",
                                              "occurred_at":  "2026-08-31T12:56:32.538676+00:00",
                                              "payload":  {
                                                              "resolution_days":  11
                                                          },
                                              "description":  "Complaint lodged (resolution_days=11)"
                                          },
                                          {
                                              "event_id":  "49f0f9cc-9c14-486b-94e8-9173c5f9ecf1",
                                              "event_type":  "premium_changed",
                                              "occurred_at":  "2026-08-31T13:11:02.615279+00:00",
                                              "payload":  {
                                                              "change_pct":  0.136
                                                          },
                                              "description":  "Premium changed (944.60 -\u003e 1073.07)"
                                          },
                                          {
                                              "event_id":  "be0e27dc-cda1-496f-a09d-b8cdb1d33c99",
                                              "event_type":  "claim_created",
                                              "occurred_at":  "2026-08-31T13:18:32.832140+00:00",
                                              "payload":  {
                                                              "claim_amount":  5624.37,
                                                              "outcome":  "approved",
                                                              "settlement_time_days":  26
                                                          },
                                              "description":  "New claim filed (amount=5624.37, outcome=approved); num_claims_12m 2 -\u003e 3"
                                          },
                                          {
                                              "event_id":  "0655f54e-ce0b-4a5d-83ae-be5c3cab14a2",
                                              "event_type":  "premium_changed",
                                              "occurred_at":  "2026-08-31T13:21:28.311388+00:00",
                                              "payload":  {
                                                              "change_pct":  0.084
                                                          },
                                              "description":  "Premium changed (1073.07 -\u003e 1163.20)"
                                          },
                                          {
                                              "event_id":  "c813a03b-45d0-4486-93e1-ce28478b10f9",
                                              "event_type":  "claim_created",
                                              "occurred_at":  "2026-08-31T13:23:38.840189+00:00",
                                              "payload":  {
                                                              "claim_amount":  4831.46,
                                                              "outcome":  "approved",
                                                              "settlement_time_days":  5
                                                          },
                                              "description":  "New claim filed (amount=4831.46, outcome=approved); num_claims_12m 3 -\u003e 4"
                                          },
                                          {
                                              "event_id":  "f7a54f55-e0ae-415e-a44e-b1abe1f661e3",
                                              "event_type":  "policy_renewed",
                                              "occurred_at":  "2026-08-31T13:28:05.191615+00:00",
                                              "payload":  {

                                                          },
                                              "description":  "Policy renewed (trailing 12-month counters reset for new period)"
                                          },
                                          {
                                              "event_id":  "22e2b8da-8a45-45c4-82be-40802ac3b0a0",
                                              "event_type":  "claim_created",
                                              "occurred_at":  "2026-08-31T13:33:41.196659+00:00",
                                              "payload":  {
                                                              "claim_amount":  1693.5,
                                                              "outcome":  "rejected",
                                                              "settlement_time_days":  17
                                                          },
                                              "description":  "New claim filed (amount=1693.50, outcome=rejected); num_claims_12m 0 -\u003e 1"
                                          },
                                          {
                                              "event_id":  "f35d4447-5630-4aa6-8517-705441c19ee8",
                                              "event_type":  "payment_missed",
                                              "occurred_at":  "2026-08-31T13:35:33.074960+00:00",
                                              "payload":  {
                                                              "count":  1
                                                          },
                                              "description":  "Payment missed (late_payment_count_12m -\u003e 1)"
                                          },
                                          {
                                              "event_id":  "ac061d3c-01b9-4b5d-9826-1bc5f9655335",
                                              "event_type":  "complaint_lodged",
                                              "occurred_at":  "2026-08-31T13:53:25.103550+00:00",
                                              "payload":  {
                                                              "resolution_days":  11
                                                          },
                                              "description":  "Complaint lodged (resolution_days=11)"
                                          },
                                          {
                                              "event_id":  "b8a8e80e-a5a8-46cb-9088-4760ac10a618",
                                              "event_type":  "coverage_downgraded",
                                              "occurred_at":  "2026-08-31T13:55:19.680054+00:00",
                                              "payload":  {
                                                              "reduction_pct":  0.21
                                                          },
                                              "description":  "Coverage downgraded (7050.03 -\u003e 5569.52)"
                                          },
                                          {
                                              "event_id":  "6ca87afe-848d-4d1b-9675-d54c30613579",
                                              "event_type":  "policy_renewed",
                                              "occurred_at":  "2026-08-31T14:01:56.927589+00:00",
                                              "payload":  {

                                                          },
                                              "description":  "Policy renewed (trailing 12-month counters reset for new period)"
                                          },
                                          {
                                              "event_id":  "66de6769-7311-406b-9fda-914e51907399",
                                              "event_type":  "engagement_changed",
                                              "occurred_at":  "2026-08-31T14:08:55.691674+00:00",
                                              "payload":  {
                                                              "contact_delta":  3,
                                                              "quote_requested":  true
                                                          },
                                              "description":  "Customer engagement changed (num_contacts_12m -\u003e 3); quote requested"
                                          },
                                          {
                                              "event_id":  "df14447e-9562-4bb0-9533-f92b1c10ec59",
                                              "event_type":  "premium_changed",
                                              "occurred_at":  "2026-08-31T14:18:56.191792+00:00",
                                              "payload":  {
                                                              "change_pct":  0.168
                                                          },
                                              "description":  "Premium changed (1163.20 -\u003e 1358.62)"
                                          },
                                          {
                                              "event_id":  "ed8dd3d3-6512-446b-92db-6669f2c1aecb",
                                              "event_type":  "premium_changed",
                                              "occurred_at":  "2026-08-31T14:23:28.029748+00:00",
                                              "payload":  {
                                                              "change_pct":  -0.05
                                                          },
                                              "description":  "Premium changed (1358.62 -\u003e 1290.69)"
                                          },
                                          {
                                              "event_id":  "f936f445-07b8-4992-9195-398a59bc6ec4",
                                              "event_type":  "premium_changed",
                                              "occurred_at":  "2026-08-31T14:24:19.305114+00:00",
                                              "payload":  {
                                                              "change_pct":  -0.017
                                                          },
                                              "description":  "Premium changed (1290.69 -\u003e 1268.75)"
                                          },
                                          {
                                              "event_id":  "a659c590-467a-4be4-a5fd-3b1b6abf3130",
                                              "event_type":  "claim_created",
                                              "occurred_at":  "2026-08-31T14:30:59.977930+00:00",
                                              "payload":  {
                                                              "claim_amount":  2222.84,
                                                              "outcome":  "approved",
                                                              "settlement_time_days":  13
                                                          },
                                              "description":  "New claim filed (amount=2222.84, outcome=approved); num_claims_12m 0 -\u003e 1"
                                          },
                                          {
                                              "event_id":  "2e81228d-eeb5-4de8-8d2d-7ec84d99a69b",
                                              "event_type":  "policy_renewed",
                                              "occurred_at":  "2026-08-31T14:41:42.458430+00:00",
                                              "payload":  {

                                                          },
                                              "description":  "Policy renewed (trailing 12-month counters reset for new period)"
                                          },
                                          {
                                              "event_id":  "43061b59-3068-489d-ae1c-90117fe0fe2f",
                                              "event_type":  "complaint_lodged",
                                              "occurred_at":  "2026-08-31T14:44:10.233613+00:00",
                                              "payload":  {
                                                              "resolution_days":  8
                                                          },
                                              "description":  "Complaint lodged (resolution_days=8)"
                                          },
                                          {
                                              "event_id":  "968881b5-9f76-4521-b2e9-2f8277a68a5f",
                                              "event_type":  "coverage_downgraded",
                                              "occurred_at":  "2026-08-31T14:48:23.661836+00:00",
                                              "payload":  {
                                                              "reduction_pct":  0.12
                                                          },
                                              "description":  "Coverage downgraded (5569.52 -\u003e 4901.18)"
                                          },
                                          {
                                              "event_id":  "20c9fd98-fc7b-4e0f-b080-0f93f8b276ca",
                                              "event_type":  "engagement_changed",
                                              "occurred_at":  "2026-08-31T14:49:43.423558+00:00",
                                              "payload":  {
                                                              "contact_delta":  2,
                                                              "quote_requested":  true
                                                          },
                                              "description":  "Customer engagement changed (num_contacts_12m -\u003e 2); quote requested"
                                          },
                                          {
                                              "event_id":  "251d75a3-480e-4de8-a932-81bc90c1c7f9",
                                              "event_type":  "engagement_changed",
                                              "occurred_at":  "2026-08-31T18:32:03.162044+00:00",
                                              "payload":  {
                                                              "contact_delta":  1,
                                                              "quote_requested":  true
                                                          },
                                              "description":  "Customer engagement changed (num_contacts_12m -\u003e 3); quote requested"
                                          },
                                          {
                                              "event_id":  "c0e6726d-0918-41e0-b2b1-2b6da5fd6d42",
                                              "event_type":  "policy_renewed",
                                              "occurred_at":  "2026-08-31T18:32:37.401787+00:00",
                                              "payload":  {

                                                          },
                                              "description":  "Policy renewed (trailing 12-month counters reset for new period)"
                                          },
                                          {
                                              "event_id":  "9292475a-338a-4661-b002-bb9e64878a2b",
                                              "event_type":  "coverage_downgraded",
                                              "occurred_at":  "2026-08-31T18:43:22.584270+00:00",
                                              "payload":  {
                                                              "reduction_pct":  0.26
                                                          },
                                              "description":  "Coverage downgraded (4901.18 -\u003e 3626.87)"
                                          },
                                          {
                                              "event_id":  "433855f7-aa29-4ba8-bf50-e2e601e5a8a7",
                                              "event_type":  "engagement_changed",
                                              "occurred_at":  "2026-08-31T19:53:03.968140+00:00",
                                              "payload":  {
                                                              "contact_delta":  2,
                                                              "quote_requested":  true
                                                          },
                                              "description":  "Customer engagement changed (num_contacts_12m -\u003e 2); quote requested"
                                          },
                                          {
                                              "event_id":  "b264b623-a629-415e-855f-a4959befac42",
                                              "event_type":  "complaint_lodged",
                                              "occurred_at":  "2026-08-31T20:24:13.553054+00:00",
                                              "payload":  {
                                                              "resolution_days":  17
                                                          },
                                              "description":  "Complaint lodged (resolution_days=17)"
                                          },
                                          {
                                              "event_id":  "0bcd7e21-06ee-4e9a-a913-48b0fea55c72",
                                              "event_type":  "payment_missed",
                                              "occurred_at":  "2026-08-31T20:45:44.141707+00:00",
                                              "payload":  {
                                                              "count":  1
                                                          },
                                              "description":  "Payment missed (late_payment_count_12m -\u003e 1)"
                                          },
                                          {
                                              "event_id":  "281f60c2-6085-41d4-9aaa-7fe3eebc2930",
                                              "event_type":  "coverage_downgraded",
                                              "occurred_at":  "2026-08-31T20:58:15.704284+00:00",
                                              "payload":  {
                                                              "reduction_pct":  0.19
                                                          },
                                              "description":  "Coverage downgraded (3626.87 -\u003e 2937.77)"
                                          },
                                          {
                                              "event_id":  "ed05ba6f-2b33-46e1-a006-0a047ebcb64d",
                                              "event_type":  "coverage_downgraded",
                                              "occurred_at":  "2026-08-31T21:01:23.260354+00:00",
                                              "payload":  {
                                                              "reduction_pct":  0.19
                                                          },
                                              "description":  "Coverage downgraded (2937.77 -\u003e 2379.59)"
                                          },
                                          {
                                              "event_id":  "a0af621a-83a1-4bb3-9175-e5e8bc46de2c",
                                              "event_type":  "engagement_changed",
                                              "occurred_at":  "2026-08-31T21:06:24.351520+00:00",
                                              "payload":  {
                                                              "contact_delta":  1,
                                                              "quote_requested":  false
                                                          },
                                              "description":  "Customer engagement changed (num_contacts_12m -\u003e 3)"
                                          },
                                          {
                                              "event_id":  "c4bb6d25-41b3-4a82-a5fa-cc839779ec2f",
                                              "event_type":  "premium_changed",
                                              "occurred_at":  "2026-08-31T21:16:56.700985+00:00",
                                              "payload":  {
                                                              "change_pct":  -0.034
                                                          },
                                              "description":  "Premium changed (1268.75 -\u003e 1225.61)"
                                          },
                                          {
                                              "event_id":  "567ed132-2d1c-4e64-a31d-7ce483a553c2",
                                              "event_type":  "premium_changed",
                                              "occurred_at":  "2026-08-31T21:31:25.656611+00:00",
                                              "payload":  {
                                                              "change_pct":  0.198
                                                          },
                                              "description":  "Premium changed (1225.61 -\u003e 1468.28)"
                                          },
                                          {
                                              "event_id":  "fd88ae5c-83f6-4aad-9c69-bb099fcb3d00",
                                              "event_type":  "claim_created",
                                              "occurred_at":  "2026-08-31T21:41:47.941564+00:00",
                                              "payload":  {
                                                              "claim_amount":  6681.9,
                                                              "outcome":  "pending",
                                                              "settlement_time_days":  17
                                                          },
                                              "description":  "New claim filed (amount=6681.90, outcome=pending); num_claims_12m 0 -\u003e 1"
                                          },
                                          {
                                              "event_id":  "007ac69a-5659-40b1-ae1c-4f7a70c7eca5",
                                              "event_type":  "complaint_lodged",
                                              "occurred_at":  "2026-09-01T00:06:17.413520+00:00",
                                              "payload":  {
                                                              "resolution_days":  14
                                                          },
                                              "description":  "Complaint lodged (resolution_days=14)"
                                          },
                                          {
                                              "event_id":  "b7e6ecd1-a1c8-4549-afe4-6e99c5dee795",
                                              "event_type":  "claim_created",
                                              "occurred_at":  "2026-09-01T00:09:45.934692+00:00",
                                              "payload":  {
                                                              "claim_amount":  1086.73,
                                                              "outcome":  "pending",
                                                              "settlement_time_days":  18
                                                          },
                                              "description":  "New claim filed (amount=1086.73, outcome=pending); num_claims_12m 1 -\u003e 2"
                                          },
                                          {
                                              "event_id":  "8e42b74d-56b2-4b00-be1d-8fdd79d3fa0a",
                                              "event_type":  "premium_changed",
                                              "occurred_at":  "2026-09-01T00:12:09.619706+00:00",
                                              "payload":  {
                                                              "change_pct":  0.158
                                                          },
                                              "description":  "Premium changed (1468.28 -\u003e 1700.27)"
                                          },
                                          {
                                              "event_id":  "8130b401-fc3f-4ac8-bf4d-cab85a066cf0",
                                              "event_type":  "premium_changed",
                                              "occurred_at":  "2026-09-01T09:11:43.950693+00:00",
                                              "payload":  {
                                                              "change_pct":  0.145
                                                          },
                                              "description":  "Premium changed (1700.27 -\u003e 1946.81)"
                                          },
                                          {
                                              "event_id":  "adcf9265-2373-4c38-880b-9ec3fcb769cc",
                                              "event_type":  "payment_missed",
                                              "occurred_at":  "2026-09-01T09:15:02.113611+00:00",
                                              "payload":  {
                                                              "count":  1
                                                          },
                                              "description":  "Payment missed (late_payment_count_12m -\u003e 2)"
                                          },
                                          {
                                              "event_id":  "c19ce144-650e-41d9-b440-6fb899cc8a39",
                                              "event_type":  "policy_renewed",
                                              "occurred_at":  "2026-09-01T09:16:32.901603+00:00",
                                              "payload":  {

                                                          },
                                              "description":  "Policy renewed (trailing 12-month counters reset for new period)"
                                          },
                                          {
                                              "event_id":  "ca59a35b-9e17-477f-9322-6022d1ce7b87",
                                              "event_type":  "policy_renewed",
                                              "occurred_at":  "2026-09-01T09:21:39.417979+00:00",
                                              "payload":  {

                                                          },
                                              "description":  "Policy renewed (trailing 12-month counters reset for new period)"
                                          },
                                          {
                                              "event_id":  "1a464590-4878-4948-83b7-deea75a40f5b",
                                              "event_type":  "premium_changed",
                                              "occurred_at":  "2026-09-01T09:29:54.803709+00:00",
                                              "payload":  {
                                                              "change_pct":  -0.032
                                                          },
                                              "description":  "Premium changed (1946.81 -\u003e 1884.51)"
                                          }
                                      ],
                    "historical_churn_label":  0,
                    "premium_change_pct":  -0.03200000000000008,
                    "premium_to_coverage_ratio":  0.7919473409721646,
                    "payout_ratio_12m":  0.75,
                    "missed_payment_flag":  0
                },
    "C000002":  {
                    "customer_id":  "C000002",
                    "age":  70,
                    "region_name":  "Auckland",
                    "marital_status":  "Married",
                    "customer_tenure_months":  76,
                    "multi_policy_flag":  1,
                    "num_policies":  3,
                    "policy_type":  "Auto",
                    "renewal_month":  3,
                    "payment_frequency":  "Monthly",
                    "autopay_enabled":  1,
                    "current_premium":  2238.2376965469502,
                    "premium_last_year":  2238.2376965469502,
                    "num_price_increases_last_3y":  9,
                    "coverage_amount":  4815.660248101422,
                    "late_payment_count_12m":  0,
                    "num_claims_12m":  0,
                    "num_approved_claims_12m":  0,
                    "num_rejected_claims_12m":  0,
                    "num_pending_claims_12m":  0,
                    "total_claim_amount_12m":  0.0,
                    "total_payout_amount_12m":  0.0,
                    "avg_claim_amount":  0.0,
                    "avg_settlement_time_days":  8,
                    "days_since_last_claim":  0,
                    "num_contacts_12m":  0,
                    "complaint_flag":  0,
                    "complaint_resolution_days":  0,
                    "quote_requested_flag":  0,
                    "coverage_downgrade_flag":  1,
                    "payment_method_change_flag":  0,
                    "version":  55,
                    "created_at":  "2026-08-24T11:12:01.049887+00:00",
                    "updated_at":  "2026-09-01T09:31:29.064033+00:00",
                    "event_history":  [
                                          {
                                              "event_id":  "5d65210a-18de-4db6-81b9-454d66826114",
                                              "event_type":  "coverage_downgraded",
                                              "occurred_at":  "2026-08-31T09:42:39.070449+00:00",
                                              "payload":  {
                                                              "reduction_pct":  0.28
                                                          },
                                              "description":  "Coverage downgraded (30404.19 -\u003e 21891.02)"
                                          },
                                          {
                                              "event_id":  "623ff8d6-d9f5-46d9-92eb-904474972fc7",
                                              "event_type":  "payment_missed",
                                              "occurred_at":  "2026-08-31T09:50:07.584059+00:00",
                                              "payload":  {
                                                              "count":  1
                                                          },
                                              "description":  "Payment missed (late_payment_count_12m -\u003e 2)"
                                          },
                                          {
                                              "event_id":  "789fe884-8c0b-4b59-bdff-86314ee4c4ca",
                                              "event_type":  "policy_renewed",
                                              "occurred_at":  "2026-08-31T10:31:36.767584+00:00",
                                              "payload":  {

                                                          },
                                              "description":  "Policy renewed (trailing 12-month counters reset for new period)"
                                          },
                                          {
                                              "event_id":  "8ec57ec5-ff21-49c7-a674-4fd32be2948b",
                                              "event_type":  "payment_missed",
                                              "occurred_at":  "2026-08-31T10:32:02.376001+00:00",
                                              "payload":  {
                                                              "count":  1
                                                          },
                                              "description":  "Payment missed (late_payment_count_12m -\u003e 1)"
                                          },
                                          {
                                              "event_id":  "e1a9e179-9cd6-46d3-b0a6-e24367ea232f",
                                              "event_type":  "policy_renewed",
                                              "occurred_at":  "2026-08-31T11:24:04.125527+00:00",
                                              "payload":  {

                                                          },
                                              "description":  "Policy renewed (trailing 12-month counters reset for new period)"
                                          },
                                          {
                                              "event_id":  "4080278a-25ba-4f21-8a66-c202047f2ccd",
                                              "event_type":  "payment_missed",
                                              "occurred_at":  "2026-08-31T11:32:38.797591+00:00",
                                              "payload":  {
                                                              "count":  1
                                                          },
                                              "description":  "Payment missed (late_payment_count_12m -\u003e 1)"
                                          },
                                          {
                                              "event_id":  "3dd00c95-c588-4a59-88ce-07f39e2df14e",
                                              "event_type":  "complaint_lodged",
                                              "occurred_at":  "2026-08-31T11:38:50.679145+00:00",
                                              "payload":  {
                                                              "resolution_days":  4
                                                          },
                                              "description":  "Complaint lodged (resolution_days=4)"
                                          },
                                          {
                                              "event_id":  "b997c09b-462f-403e-b977-d3d9a324deaf",
                                              "event_type":  "payment_missed",
                                              "occurred_at":  "2026-08-31T13:14:13.547972+00:00",
                                              "payload":  {
                                                              "count":  1
                                                          },
                                              "description":  "Payment missed (late_payment_count_12m -\u003e 2)"
                                          },
                                          {
                                              "event_id":  "ea9cd04a-4660-4061-aeca-1d9eec347d05",
                                              "event_type":  "engagement_changed",
                                              "occurred_at":  "2026-08-31T13:15:51.122398+00:00",
                                              "payload":  {
                                                              "contact_delta":  1,
                                                              "quote_requested":  false
                                                          },
                                              "description":  "Customer engagement changed (num_contacts_12m -\u003e 1)"
                                          },
                                          {
                                              "event_id":  "b0205a89-13fd-4b2b-8151-1ccdee31df8f",
                                              "event_type":  "engagement_changed",
                                              "occurred_at":  "2026-08-31T13:20:39.983529+00:00",
                                              "payload":  {
                                                              "contact_delta":  1,
                                                              "quote_requested":  false
                                                          },
                                              "description":  "Customer engagement changed (num_contacts_12m -\u003e 2)"
                                          },
                                          {
                                              "event_id":  "7b6afb85-fb7d-4058-822a-9ba7351fdf6c",
                                              "event_type":  "claim_created",
                                              "occurred_at":  "2026-08-31T13:25:09.721359+00:00",
                                              "payload":  {
                                                              "claim_amount":  3315.83,
                                                              "outcome":  "rejected",
                                                              "settlement_time_days":  11
                                                          },
                                              "description":  "New claim filed (amount=3315.83, outcome=rejected); num_claims_12m 0 -\u003e 1"
                                          },
                                          {
                                              "event_id":  "7441d379-0d4c-4c21-8d26-6e5c39ea1ef2",
                                              "event_type":  "policy_renewed",
                                              "occurred_at":  "2026-08-31T13:26:56.623210+00:00",
                                              "payload":  {

                                                          },
                                              "description":  "Policy renewed (trailing 12-month counters reset for new period)"
                                          },
                                          {
                                              "event_id":  "3536cd16-6d52-4577-9af6-8aa8aa9239cb",
                                              "event_type":  "premium_changed",
                                              "occurred_at":  "2026-08-31T13:28:04.017684+00:00",
                                              "payload":  {
                                                              "change_pct":  0.131
                                                          },
                                              "description":  "Premium changed (1250.86 -\u003e 1414.72)"
                                          },
                                          {
                                              "event_id":  "e0b58e30-24e6-47eb-8e3e-6ed5a02d615b",
                                              "event_type":  "policy_renewed",
                                              "occurred_at":  "2026-08-31T13:28:23.517884+00:00",
                                              "payload":  {

                                                          },
                                              "description":  "Policy renewed (trailing 12-month counters reset for new period)"
                                          },
                                          {
                                              "event_id":  "06d5a026-88dc-4ac1-b562-e27a21495379",
                                              "event_type":  "coverage_downgraded",
                                              "occurred_at":  "2026-08-31T13:29:45.010133+00:00",
                                              "payload":  {
                                                              "reduction_pct":  0.25
                                                          },
                                              "description":  "Coverage downgraded (21891.02 -\u003e 16418.26)"
                                          },
                                          {
                                              "event_id":  "e3643861-397f-450d-83c6-5b5cb8595b78",
                                              "event_type":  "policy_renewed",
                                              "occurred_at":  "2026-08-31T13:33:59.168960+00:00",
                                              "payload":  {

                                                          },
                                              "description":  "Policy renewed (trailing 12-month counters reset for new period)"
                                          },
                                          {
                                              "event_id":  "abfe7ce9-0785-4d5c-8ab7-fd7845757d44",
                                              "event_type":  "payment_missed",
                                              "occurred_at":  "2026-08-31T13:37:12.995810+00:00",
                                              "payload":  {
                                                              "count":  1
                                                          },
                                              "description":  "Payment missed (late_payment_count_12m -\u003e 1)"
                                          },
                                          {
                                              "event_id":  "16704285-34a8-48dd-8e1d-336648efa978",
                                              "event_type":  "claim_created",
                                              "occurred_at":  "2026-08-31T13:39:48.813482+00:00",
                                              "payload":  {
                                                              "claim_amount":  6799.85,
                                                              "outcome":  "pending",
                                                              "settlement_time_days":  20
                                                          },
                                              "description":  "New claim filed (amount=6799.85, outcome=pending); num_claims_12m 0 -\u003e 1"
                                          },
                                          {
                                              "event_id":  "dcc3635d-b801-4699-b3df-89da00a255ed",
                                              "event_type":  "premium_changed",
                                              "occurred_at":  "2026-08-31T13:43:52.035050+00:00",
                                              "payload":  {
                                                              "change_pct":  0.18
                                                          },
                                              "description":  "Premium changed (1414.72 -\u003e 1669.37)"
                                          },
                                          {
                                              "event_id":  "065345c7-6b8b-4deb-af96-41732ef3fd11",
                                              "event_type":  "premium_changed",
                                              "occurred_at":  "2026-08-31T13:44:43.923129+00:00",
                                              "payload":  {
                                                              "change_pct":  0.157
                                                          },
                                              "description":  "Premium changed (1669.37 -\u003e 1931.46)"
                                          },
                                          {
                                              "event_id":  "c3f60cfb-b332-4dc9-af96-e3ad19501fb4",
                                              "event_type":  "complaint_lodged",
                                              "occurred_at":  "2026-08-31T13:49:30.962862+00:00",
                                              "payload":  {
                                                              "resolution_days":  18
                                                          },
                                              "description":  "Complaint lodged (resolution_days=18)"
                                          },
                                          {
                                              "event_id":  "3140bbd4-4c6c-49f9-bed3-489fdc956ddc",
                                              "event_type":  "coverage_downgraded",
                                              "occurred_at":  "2026-08-31T13:51:42.350614+00:00",
                                              "payload":  {
                                                              "reduction_pct":  0.22
                                                          },
                                              "description":  "Coverage downgraded (16418.26 -\u003e 12806.24)"
                                          },
                                          {
                                              "event_id":  "0b2260d8-e769-4bcf-bc06-0bd03ae5b4ae",
                                              "event_type":  "premium_changed",
                                              "occurred_at":  "2026-08-31T14:09:20.349684+00:00",
                                              "payload":  {
                                                              "change_pct":  0.001
                                                          },
                                              "description":  "Premium changed (1931.46 -\u003e 1933.39)"
                                          },
                                          {
                                              "event_id":  "7ffd1a2d-c0eb-45df-a512-860b6d6eb9f0",
                                              "event_type":  "payment_missed",
                                              "occurred_at":  "2026-08-31T14:09:28.973115+00:00",
                                              "payload":  {
                                                              "count":  1
                                                          },
                                              "description":  "Payment missed (late_payment_count_12m -\u003e 2)"
                                          },
                                          {
                                              "event_id":  "a7d6aba8-f41d-476f-ba52-cc795cd4e1f4",
                                              "event_type":  "policy_renewed",
                                              "occurred_at":  "2026-08-31T14:16:53.121800+00:00",
                                              "payload":  {

                                                          },
                                              "description":  "Policy renewed (trailing 12-month counters reset for new period)"
                                          },
                                          {
                                              "event_id":  "8147d638-1609-4b94-9323-3860a92ce3e5",
                                              "event_type":  "policy_renewed",
                                              "occurred_at":  "2026-08-31T14:17:33.622638+00:00",
                                              "payload":  {

                                                          },
                                              "description":  "Policy renewed (trailing 12-month counters reset for new period)"
                                          },
                                          {
                                              "event_id":  "6e18059e-5b2f-49f5-a0d9-254b901f198d",
                                              "event_type":  "engagement_changed",
                                              "occurred_at":  "2026-08-31T14:22:06.731751+00:00",
                                              "payload":  {
                                                              "contact_delta":  3,
                                                              "quote_requested":  false
                                                          },
                                              "description":  "Customer engagement changed (num_contacts_12m -\u003e 3)"
                                          },
                                          {
                                              "event_id":  "76ae7389-e93c-48ab-a561-0ee0f36143a6",
                                              "event_type":  "engagement_changed",
                                              "occurred_at":  "2026-08-31T18:42:50.884676+00:00",
                                              "payload":  {
                                                              "contact_delta":  3,
                                                              "quote_requested":  false
                                                          },
                                              "description":  "Customer engagement changed (num_contacts_12m -\u003e 6)"
                                          },
                                          {
                                              "event_id":  "a57be5b5-85fb-4a9c-b6f6-c20a32ed7326",
                                              "event_type":  "premium_changed",
                                              "occurred_at":  "2026-08-31T19:53:00.675679+00:00",
                                              "payload":  {
                                                              "change_pct":  -0.015
                                                          },
                                              "description":  "Premium changed (1933.39 -\u003e 1904.39)"
                                          },
                                          {
                                              "event_id":  "6da6b8a7-9db5-4fe8-b0d2-bb863b3c12a2",
                                              "event_type":  "policy_renewed",
                                              "occurred_at":  "2026-08-31T20:24:32.871805+00:00",
                                              "payload":  {

                                                          },
                                              "description":  "Policy renewed (trailing 12-month counters reset for new period)"
                                          },
                                          {
                                              "event_id":  "e0f5163d-08c5-4a1b-b48c-ade8e7fe3123",
                                              "event_type":  "policy_renewed",
                                              "occurred_at":  "2026-08-31T20:25:06.737765+00:00",
                                              "payload":  {

                                                          },
                                              "description":  "Policy renewed (trailing 12-month counters reset for new period)"
                                          },
                                          {
                                              "event_id":  "e3aa8a37-7483-476f-a57d-d3d8a13794c5",
                                              "event_type":  "complaint_lodged",
                                              "occurred_at":  "2026-08-31T20:30:14.586305+00:00",
                                              "payload":  {
                                                              "resolution_days":  19
                                                          },
                                              "description":  "Complaint lodged (resolution_days=19)"
                                          },
                                          {
                                              "event_id":  "1f10aeb6-5aa4-442e-aa01-cb4da9ebf8bc",
                                              "event_type":  "claim_created",
                                              "occurred_at":  "2026-08-31T20:32:45.403306+00:00",
                                              "payload":  {
                                                              "claim_amount":  5796.84,
                                                              "outcome":  "rejected",
                                                              "settlement_time_days":  24
                                                          },
                                              "description":  "New claim filed (amount=5796.84, outcome=rejected); num_claims_12m 0 -\u003e 1"
                                          },
                                          {
                                              "event_id":  "dbdcf927-6753-4501-a3a1-64ce422b0591",
                                              "event_type":  "coverage_downgraded",
                                              "occurred_at":  "2026-08-31T20:43:57.238147+00:00",
                                              "payload":  {
                                                              "reduction_pct":  0.15
                                                          },
                                              "description":  "Coverage downgraded (12806.24 -\u003e 10885.31)"
                                          },
                                          {
                                              "event_id":  "ece3ef71-254a-42d3-a6d5-c2c91ee4ec24",
                                              "event_type":  "claim_created",
                                              "occurred_at":  "2026-08-31T21:02:23.032290+00:00",
                                              "payload":  {
                                                              "claim_amount":  6971.15,
                                                              "outcome":  "rejected",
                                                              "settlement_time_days":  12
                                                          },
                                              "description":  "New claim filed (amount=6971.15, outcome=rejected); num_claims_12m 1 -\u003e 2"
                                          },
                                          {
                                              "event_id":  "53184aa3-a231-4b64-a32e-2083620eebd8",
                                              "event_type":  "payment_missed",
                                              "occurred_at":  "2026-08-31T21:11:24.653913+00:00",
                                              "payload":  {
                                                              "count":  1
                                                          },
                                              "description":  "Payment missed (late_payment_count_12m -\u003e 1)"
                                          },
                                          {
                                              "event_id":  "0c4a6f89-acb2-437e-b96b-0f522109e0c3",
                                              "event_type":  "complaint_lodged",
                                              "occurred_at":  "2026-08-31T21:25:19.360014+00:00",
                                              "payload":  {
                                                              "resolution_days":  2
                                                          },
                                              "description":  "Complaint lodged (resolution_days=2)"
                                          },
                                          {
                                              "event_id":  "853df54c-2499-45f5-b1d2-71b98cadf0b0",
                                              "event_type":  "coverage_downgraded",
                                              "occurred_at":  "2026-08-31T21:37:57.574611+00:00",
                                              "payload":  {
                                                              "reduction_pct":  0.2
                                                          },
                                              "description":  "Coverage downgraded (10885.31 -\u003e 8708.25)"
                                          },
                                          {
                                              "event_id":  "97033494-ec62-4679-acea-48c89d424ba8",
                                              "event_type":  "payment_missed",
                                              "occurred_at":  "2026-08-31T21:41:53.647786+00:00",
                                              "payload":  {
                                                              "count":  1
                                                          },
                                              "description":  "Payment missed (late_payment_count_12m -\u003e 2)"
                                          },
                                          {
                                              "event_id":  "66378f93-5036-4e06-ad05-99495cefc8f9",
                                              "event_type":  "payment_missed",
                                              "occurred_at":  "2026-09-01T00:06:10.074836+00:00",
                                              "payload":  {
                                                              "count":  1
                                                          },
                                              "description":  "Payment missed (late_payment_count_12m -\u003e 3)"
                                          },
                                          {
                                              "event_id":  "421d1002-ac8b-4d8d-8eb2-d1c76d2c46d5",
                                              "event_type":  "coverage_downgraded",
                                              "occurred_at":  "2026-09-01T00:09:55.624426+00:00",
                                              "payload":  {
                                                              "reduction_pct":  0.21
                                                          },
                                              "description":  "Coverage downgraded (8708.25 -\u003e 6879.51)"
                                          },
                                          {
                                              "event_id":  "cd09d218-5b9d-4d55-ba67-9f2cf084c61b",
                                              "event_type":  "premium_changed",
                                              "occurred_at":  "2026-09-01T00:12:00.293219+00:00",
                                              "payload":  {
                                                              "change_pct":  -0.034
                                                          },
                                              "description":  "Premium changed (1904.39 -\u003e 1839.64)"
                                          },
                                          {
                                              "event_id":  "07c48962-dc70-411d-95eb-89862718d1bf",
                                              "event_type":  "engagement_changed",
                                              "occurred_at":  "2026-09-01T00:16:49.776204+00:00",
                                              "payload":  {
                                                              "contact_delta":  2,
                                                              "quote_requested":  false
                                                          },
                                              "description":  "Customer engagement changed (num_contacts_12m -\u003e 2)"
                                          },
                                          {
                                              "event_id":  "bdf5678f-a4f6-458f-a316-c03e1c6b8776",
                                              "event_type":  "premium_changed",
                                              "occurred_at":  "2026-09-01T09:16:54.614481+00:00",
                                              "payload":  {
                                                              "change_pct":  0.171
                                                          },
                                              "description":  "Premium changed (1839.64 -\u003e 2154.22)"
                                          },
                                          {
                                              "event_id":  "6ba19f50-ae5b-487d-9300-345531ae8a6e",
                                              "event_type":  "premium_changed",
                                              "occurred_at":  "2026-09-01T09:25:28.142324+00:00",
                                              "payload":  {
                                                              "change_pct":  0.039
                                                          },
                                              "description":  "Premium changed (2154.22 -\u003e 2238.24)"
                                          },
                                          {
                                              "event_id":  "b08709a7-e7e9-4d90-b8aa-6593254b9c67",
                                              "event_type":  "claim_created",
                                              "occurred_at":  "2026-09-01T09:27:48.514531+00:00",
                                              "payload":  {
                                                              "claim_amount":  4692.1,
                                                              "outcome":  "approved",
                                                              "settlement_time_days":  8
                                                          },
                                              "description":  "New claim filed (amount=4692.10, outcome=approved); num_claims_12m 2 -\u003e 3"
                                          },
                                          {
                                              "event_id":  "580aea7a-c5ee-4adc-a48c-134873e86da7",
                                              "event_type":  "coverage_downgraded",
                                              "occurred_at":  "2026-09-01T09:27:52.784403+00:00",
                                              "payload":  {
                                                              "reduction_pct":  0.3
                                                          },
                                              "description":  "Coverage downgraded (6879.51 -\u003e 4815.66)"
                                          },
                                          {
                                              "event_id":  "20a2f62d-b0e9-4ba3-9606-49bb37b87a6b",
                                              "event_type":  "engagement_changed",
                                              "occurred_at":  "2026-09-01T09:28:33.657217+00:00",
                                              "payload":  {
                                                              "contact_delta":  1,
                                                              "quote_requested":  false
                                                          },
                                              "description":  "Customer engagement changed (num_contacts_12m -\u003e 3)"
                                          },
                                          {
                                              "event_id":  "f746b46f-6a53-4e17-8837-803d29b18ef9",
                                              "event_type":  "complaint_lodged",
                                              "occurred_at":  "2026-09-01T09:30:00.130323+00:00",
                                              "payload":  {
                                                              "resolution_days":  18
                                                          },
                                              "description":  "Complaint lodged (resolution_days=18)"
                                          },
                                          {
                                              "event_id":  "25123305-b573-44c8-87b0-a9853583bea2",
                                              "event_type":  "policy_renewed",
                                              "occurred_at":  "2026-09-01T09:31:29.064033+00:00",
                                              "payload":  {

                                                          },
                                              "description":  "Policy renewed (trailing 12-month counters reset for new period)"
                                          }
                                      ],
                    "historical_churn_label":  0,
                    "premium_change_pct":  0.0,
                    "premium_to_coverage_ratio":  0.4647831410925173,
                    "payout_ratio_12m":  0.75,
                    "missed_payment_flag":  0
                }
}
```
# Known gaps vs a full event-sourced design

## Latest state versus state/event history

`TwinStateStore` persists one current `TwinState` per customer. Its `save` method assigns `self._states[state.customer_id] = state` and then rewrites the complete JSON snapshot, so prior state versions are not retained. `TwinState.record_event` does retain embedded event history, but caps it at the latest 50 records. This is bounded history inside the current snapshot, not an append-only state-version store.

## Audit trail and event log

Yes. `storage/event_log.json` exists and `StateSynchronizer._append_event_log` opens it in append mode and writes `json.dumps(event.to_dict()) + "\n"`, producing timestamped JSON Lines event envelopes. It is an audit trail of processed events. There is no separate timestamped state-history table or file in `storage/`; `config.py` defines `RISK_HISTORY_PATH`, but no `storage/risk_history.json` file is present in this repository snapshot.

## Twin state schema categories

`TwinState` is a dataclass with explicit groups: identity/static profile fields, dynamic event-mutated fields, an ML-only `payment_method_change_flag`, bookkeeping fields (`version`, timestamps, and bounded `event_history`), and a reference-only `historical_churn_label`. Derived ML fields (`premium_change_pct`, `premium_to_coverage_ratio`, `payout_ratio_12m`, and `missed_payment_flag`) are properties rather than independently stored inputs. The full class definition appears above in `twin_engine/state/twin_state.py`; the persisted JSON is still one snapshot structure with nested event history, not separate raw/derived/prediction tables. Predicted churn is returned separately by the `RiskResult` dataclass in `risk_intelligence/predictor.py`.

## Real versus simulated isolation

`scenario_transformer.py` calls `state.clone()` before constructing and applying a synthetic event, and it has no store reference or save call. This is an implementation-level clone-and-no-save isolation path. However, there is no automated test in this repository proving that the real state is never mutated: no test files were found. `docs/simulation.md` describes an automated verification, but that verification is not represented by a checked-in test file in the current repository.

### Evidence excerpts

```python twin_engine/state/state_store.py
    def save(self, state: TwinState) -> None:
        with self._lock:
            self._states[state.customer_id] = state
            self._flush_to_disk()
```

```python twin_engine/state/twin_state.py
    def record_event(self, event_id: str, event_type: str, payload: Dict[str, Any], description: str = "") -> None:
        self.event_history.append(
            TwinEventRecord(event_id=event_id, event_type=event_type, occurred_at=_now_iso(), payload=payload, description=description)
        )
        if len(self.event_history) > 50:
            self.event_history = self.event_history[-50:]
```

```python twin_engine/synchronization/synchronizer.py
    def _append_event_log(self, event: Event) -> None:
        with config.EVENT_LOG_PATH.open("a") as f:
            f.write(json.dumps(event.to_dict()) + "\n")
```

```python twin_engine/simulation/scenario_transformer.py
        cloned_state = state.clone()
        self._transition_handler.apply(cloned_state, synthetic_event)
        return cloned_state
```