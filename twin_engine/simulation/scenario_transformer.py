"""
Scenario Transformer.

Implements S'_t = T(S_t, theta):
  1. Clone the current Twin state.
  2. Apply hypothetical scenario parameters (theta) to the clone.
  3. Return the transformed (cloned) state.

CRITICAL INVARIANT: the real Twin state must never be modified by a
what-if scenario. This is enforced here by always operating on
`state.clone()` (a deep copy - see TwinState.clone) and by this module
never calling TwinStateStore.save() itself. Only the StateSynchronizer
(driven by real events) is allowed to persist changes to the store.

Supported scenarios reuse the same per-event-type transition logic as real
events (via EventTransitionHandler), applied to the clone instead of the
real state. This guarantees "what a scenario does" and "what the
equivalent real event does" can never silently drift apart.
"""

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
