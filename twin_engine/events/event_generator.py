"""
Real-Time Event Generator.

There is no Kafka in this MVP. This module is the local stand-in: a
background thread that periodically produces simulated customer-domain
events and pushes them through the exact same path a real event source
would use (Event -> Event Transition Handler -> Twin State Store -> risk
recalculation), via the State Synchronizer.

Design note (see docs/architecture.md): this generator only ever
constructs `Event` objects and hands them to `StateSynchronizer.process_event`.
It never touches the ML model or a risk score directly - the Random Forest,
via Risk Intelligence, is what determines the resulting risk. This keeps
the class swappable for a real Kafka consumer later without redesigning
anything downstream.
"""

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
