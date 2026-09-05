import copy

from twin_engine.events.event import EventType
from twin_engine.simulation.scenario_transformer import Scenario, scenario_transformer
from twin_engine.state.state_store import twin_state_store


def _scenario() -> Scenario:
    return Scenario(
        name="premium_changed",
        event_type=EventType.PREMIUM_CHANGED,
        parameters={"change_pct": 0.15},
    )


def test_scenario_does_not_mutate_real_state():
    """A what-if scenario leaves the persisted real Twin unchanged."""
    customers = twin_state_store.list_all()
    assert customers, "No customers in the store - run bootstrap.py first."

    customer_id = customers[0].customer_id
    before = twin_state_store.get(customer_id)
    before_snapshot = copy.deepcopy(before.to_dict())

    scenario_transformer.transform(before, _scenario())

    after_snapshot = twin_state_store.get(customer_id).to_dict()
    assert before_snapshot == after_snapshot


def test_scenario_returns_independent_clone():
    """The transformed result is independent from the real Twin object."""
    customers = twin_state_store.list_all()
    assert customers, "No customers in the store - run bootstrap.py first."

    real_state = twin_state_store.get(customers[0].customer_id)
    simulated_state = scenario_transformer.transform(real_state, _scenario())

    assert simulated_state is not real_state
    assert simulated_state.current_premium != real_state.current_premium