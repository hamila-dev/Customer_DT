

"""Apply hypothetical events to isolated copies of real Twin states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from twin_engine.events.event import Event, EventType
from twin_engine.events.transition_handler import EventTransitionHandler, event_transition_handler
from twin_engine.state.twin_state import TwinState


@dataclass
class Scenario:
    """Named what-if event and its parameters."""

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
    """Transform a cloned state without access to persistence."""

    def __init__(self, transition_handler: EventTransitionHandler = event_transition_handler):
        self._transition_handler = transition_handler

    def transform(self, state: TwinState, scenario: Scenario) -> TwinState:
        """Return a transformed clone; the supplied state is never persisted or mutated."""
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
