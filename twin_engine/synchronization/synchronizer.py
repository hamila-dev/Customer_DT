
"""Apply real events, persist the resulting Twin, and trigger recalculation."""

from __future__ import annotations

import json
import logging
import time
from typing import Callable, Optional

from twin_engine.events.event import Event
from twin_engine.events.transition_handler import EventTransitionHandler, event_transition_handler
from twin_engine.state.state_store import TwinStateStore, twin_state_store
from twin_engine.state.twin_state import TwinState

import config

logger = logging.getLogger(__name__)

# Called after persistence so risk recalculation sees the stored state.
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

    def _append_event_log(
        self,
        event: Event,
        prev_state: TwinState,
        new_state: TwinState,
        processing_time_ms: float,
        model_version: Optional[str] = None,
    ) -> None:
        """Append event, before/after state, timing, and model metadata as JSONL."""
        try:
            record = event.to_dict()
            record["prev_state"] = prev_state.to_dict()
            record["new_state"] = new_state.to_dict()
            record["processing_time_ms"] = round(processing_time_ms, 3)
            record["model_version"] = model_version
            with config.EVENT_LOG_PATH.open("a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except OSError:
            logger.exception("Failed to append event to event log")

    def process_event(self, event: Event) -> TwinState:
        """Process the only supported path for mutating and persisting a real Twin."""
        state = self._store.get(event.customer_id)
        if state is None:
            raise KeyError(f"No Twin state found for customer_id={event.customer_id}")

        prev_state = state.clone()
        start = time.perf_counter()
        updated_state = self._transition_handler.apply(state, event)
        elapsed_ms = (time.perf_counter() - start) * 1000
        self._store.save(updated_state)

        model_version = None
        if self._on_state_updated is not None:
            self._on_state_updated(event.customer_id, updated_state)
            from risk_intelligence.predictor import churn_predictor

            if churn_predictor.is_available:
                model_version = churn_predictor.model_version

        self._append_event_log(event, prev_state, updated_state, elapsed_ms, model_version)

        logger.info(
            "Synchronized event %s (%s) for customer %s in %.2fms",
            event.event_id,
            event.event_type,
            event.customer_id,
            elapsed_ms,
        )
        return updated_state


# Shared synchronizer wired to the API's risk callback at startup.
state_synchronizer = StateSynchronizer()
