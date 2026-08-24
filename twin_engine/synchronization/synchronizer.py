"""
State Synchronizer.

Responsibility (kept intentionally simple for the MVP - no distributed
synchronization):

    Event -> Twin state update -> Persist/update current state
        -> Risk recalculation

The Twin State Store is the single source of truth for the current
simulated Twin. This class is the one place that sequences "apply event,
then persist, then ask Risk Intelligence to recalculate" so that every
entry point (API POST /events, the local Event Generator) goes through the
exact same path.
"""

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
