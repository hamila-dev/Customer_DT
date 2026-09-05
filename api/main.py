"""FastAPI handlers for the Twin, risk, recommendation, and simulation APIs."""

from __future__ import annotations

import json
import logging
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import bootstrap
import config
from api.schemas import EventGeneratorConfigIn, EventIn, MonteCarloIn, SimulateIn
from recommendation_engine.engine import recommendation_engine
from risk_intelligence.feature_mapper import FeatureMappingError
from risk_intelligence.predictor import ModelNotAvailableError, churn_predictor
from twin_engine.events.event import Event, EventType
from twin_engine.events.event_generator import event_generator
from twin_engine.simulation.monte_carlo import monte_carlo_engine
from twin_engine.simulation.scenario_transformer import Scenario, scenario_transformer
from twin_engine.state.state_store import twin_state_store
from twin_engine.synchronization.synchronizer import state_synchronizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Customer Digital Twin — Insurise MVP",
    description=(
        "Prototype Digital Twin Engine for churn risk. Uses a public/synthetic "
        "insurance policyholder churn dataset as a stand-in for real Insurise "
        "data - see docs/dataset-mapping.md."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(FeatureMappingError)
async def feature_mapping_error_handler(request, exc: FeatureMappingError):
    """Return 422 when a Twin cannot satisfy the model feature schema."""
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=422, content={"detail": str(exc)})


def _risk_recalc_callback(customer_id: str, state) -> None:
    """Re-score persisted event updates when model artifacts are available."""
    if churn_predictor.is_available:
        try:
            result = churn_predictor.predict(state)
            logger.info("Risk recalculated for %s: %s (%.3f)", customer_id, result.risk_level, result.churn_probability)
        except ModelNotAvailableError:
            pass


state_synchronizer.set_risk_recalc_callback(_risk_recalc_callback)


@app.on_event("startup")
def on_startup() -> None:
    loaded = bootstrap.load_initial_customers(twin_state_store)
    if loaded:
        logger.info("Bootstrap: loaded %d customers.", loaded)
    else:
        logger.info("Bootstrap: store already populated with %d customers.", twin_state_store.count())
    if not churn_predictor.is_available:
        logger.warning("Model artifacts not found - risk/recommendation endpoints will return 503 until you add them to model/.")


def _get_state_or_404(customer_id: str):
    state = twin_state_store.get(customer_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Customer '{customer_id}' not found")
    return state


def _model_unavailable_response(exc: ModelNotAvailableError):
    raise HTTPException(status_code=503, detail=str(exc))


@app.get("/api/dashboard/summary")
def dashboard_summary():
    states = twin_state_store.list_all()
    summary = {
        "total_customers": len(states),
        "high_risk": 0,
        "medium_risk": 0,
        "low_risk": 0,
        "unscored": 0,
        "model_available": churn_predictor.is_available,
        "event_generator_running": event_generator.is_running,
        "events_generated": event_generator.events_generated,
    }
    high_risk_customers = []

    if churn_predictor.is_available:
        # Batch scoring avoids one Random Forest call per customer.
        try:
            probabilities = churn_predictor.predict_batch(states)
        except ModelNotAvailableError:
            probabilities = [None] * len(states)

        for state, probability in zip(states, probabilities):
            if probability is None:
                summary["unscored"] += 1
                continue
            risk_level = config.risk_level_from_probability(probability)
            if risk_level == "HIGH":
                summary["high_risk"] += 1
                high_risk_customers.append(
                    {
                        "customer_id": state.customer_id,
                        "region_name": state.region_name,
                        "policy_type": state.policy_type,
                        "churn_probability": round(probability, 4),
                    }
                )
            elif risk_level == "MEDIUM":
                summary["medium_risk"] += 1
            else:
                summary["low_risk"] += 1
    else:
        summary["unscored"] = len(states)

    high_risk_customers.sort(key=lambda c: c["churn_probability"], reverse=True)
    summary["high_risk_customers"] = high_risk_customers[:20]

    # The dashboard displays the newest recent events first.
    recent_events = []
    for state in states:
        for record in state.event_history[-3:]:
            recent_events.append(
                {
                    "customer_id": state.customer_id,
                    "region_name": state.region_name,
                    "policy_type": state.policy_type,
                    **record.to_dict(),
                }
            )
    recent_events.sort(key=lambda e: e["occurred_at"], reverse=True)
    summary["recent_events"] = recent_events[:25]

    return summary


@app.get("/api/customers")
def list_customers():
    states = twin_state_store.list_all()

    probabilities = [None] * len(states)
    if churn_predictor.is_available:
        try:
            probabilities = churn_predictor.predict_batch(states)
        except ModelNotAvailableError:
            pass

    results = []
    for state, probability in zip(states, probabilities):
        entry = {
            "customer_id": state.customer_id,
            "region_name": state.region_name,
            "policy_type": state.policy_type,
            "age": state.age,
        }
        if probability is not None:
            entry["churn_probability"] = round(probability, 4)
            entry["risk_level"] = config.risk_level_from_probability(probability)
        else:
            entry["churn_probability"] = None
            entry["risk_level"] = None
        results.append(entry)
    return {"customers": results, "total": len(results)}


@app.get("/api/customers/{customer_id}")
def get_customer(customer_id: str):
    state = _get_state_or_404(customer_id)
    return state.to_dict()


@app.get("/api/customers/{customer_id}/twin")
def get_customer_twin(customer_id: str):
    state = _get_state_or_404(customer_id)
    return {
        "customer_id": state.customer_id,
        "state": state.to_dict(),
        "feature_vector": state.to_feature_dict(),
        "version": state.version,
        "updated_at": state.updated_at,
    }


@app.get("/api/customers/{customer_id}/risk")
def get_customer_risk(customer_id: str):
    state = _get_state_or_404(customer_id)
    try:
        result = churn_predictor.predict(state)
    except ModelNotAvailableError as exc:
        _model_unavailable_response(exc)
    return result.to_dict()


@app.get("/api/customers/{customer_id}/events")
def get_customer_events(customer_id: str):
    state = _get_state_or_404(customer_id)
    return {"customer_id": customer_id, "events": [e.to_dict() for e in state.event_history]}


@app.get("/api/customers/{customer_id}/trace")
def get_customer_trace(customer_id: str, limit: int = 100):
    """Return enriched transition records for one customer."""
    if not config.EVENT_LOG_PATH.exists():
        raise HTTPException(status_code=404, detail="No event log found yet.")
    if limit < 1:
        raise HTTPException(status_code=400, detail="limit must be at least 1")

    records = []
    with config.EVENT_LOG_PATH.open() as event_log:
        for line in event_log:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("customer_id") == customer_id:
                records.append(record)

    return {"customer_id": customer_id, "count": len(records), "trace": records[-limit:]}


@app.get("/api/customers/{customer_id}/state-at")
def get_customer_state_at(customer_id: str, timestamp: str):
    """Return a customer's reconstructed state at an ISO 8601 timestamp."""
    from twin_engine.state.time_travel import get_state_at

    try:
        state = get_state_at(customer_id, timestamp)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid ISO 8601 timestamp: {exc}") from exc
    if state is None:
        raise HTTPException(status_code=404, detail=f"No state found for {customer_id}")
    return state.to_dict()


@app.get("/api/customers/{customer_id}/recommendations")
def get_customer_recommendations(customer_id: str):
    state = _get_state_or_404(customer_id)
    try:
        result = recommendation_engine.recommend(state)
    except ModelNotAvailableError as exc:
        _model_unavailable_response(exc)
    return result.to_dict()


@app.post("/api/events")
def post_event(event_in: EventIn):
    _get_state_or_404(event_in.customer_id)
    try:
        event_type = EventType(event_in.event_type)
    except ValueError:
        valid = ", ".join(e.value for e in EventType)
        raise HTTPException(status_code=400, detail=f"Unknown event_type '{event_in.event_type}'. Valid: {valid}")

    event = Event(
        customer_id=event_in.customer_id,
        event_type=event_type,
        payload=event_in.payload,
        source=event_in.source,
    )
    updated_state = state_synchronizer.process_event(event)
    return {"status": "ok", "customer_id": event_in.customer_id, "twin_version": updated_state.version}


@app.get("/api/event-generator/status")
def event_generator_status():
    return {
        "running": event_generator.is_running,
        "interval_seconds": event_generator.interval_seconds,
        "scenarios": event_generator.scenarios,
        "events_generated": event_generator.events_generated,
    }


@app.post("/api/event-generator/start")
def event_generator_start(cfg: EventGeneratorConfigIn = None):
    if cfg:
        event_generator.configure(interval_seconds=cfg.interval_seconds, scenarios=cfg.scenarios)
    event_generator.start()
    return event_generator_status()


@app.post("/api/event-generator/stop")
def event_generator_stop():
    event_generator.stop()
    return event_generator_status()


@app.post("/api/customers/{customer_id}/simulate")
def simulate_customer(customer_id: str, sim_in: SimulateIn):
    state = _get_state_or_404(customer_id)
    if not churn_predictor.is_available:
        _model_unavailable_response(ModelNotAvailableError(churn_predictor.MISSING_ARTIFACT_MESSAGE))

    try:
        scenario = Scenario.from_request(sim_in.scenario, sim_in.parameters)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    before_risk = churn_predictor.predict(state)
    simulated_state = scenario_transformer.transform(state, scenario)
    after_risk = churn_predictor.predict(simulated_state)

    return {
        "customer_id": customer_id,
        "scenario": sim_in.scenario,
        "parameters": sim_in.parameters,
        "before": {"churn_probability": round(before_risk.churn_probability, 4), "risk_level": before_risk.risk_level},
        "after": {"churn_probability": round(after_risk.churn_probability, 4), "risk_level": after_risk.risk_level},
        "difference": round(after_risk.churn_probability - before_risk.churn_probability, 4),
        "real_twin_modified": False,
        "simulated_state": simulated_state.to_dict(),
    }


@app.post("/api/customers/{customer_id}/simulate/monte-carlo")
def simulate_customer_monte_carlo(customer_id: str, mc_in: MonteCarloIn):
    state = _get_state_or_404(customer_id)
    if not churn_predictor.is_available:
        _model_unavailable_response(ModelNotAvailableError(churn_predictor.MISSING_ARTIFACT_MESSAGE))

    try:
        scenario = Scenario.from_request(mc_in.scenario, mc_in.parameters)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    kwargs = {}
    if mc_in.trials is not None:
        kwargs["trials"] = mc_in.trials
    if mc_in.numeric_noise_std is not None:
        kwargs["numeric_noise_std"] = mc_in.numeric_noise_std

    result = monte_carlo_engine.run(state, scenario, **kwargs)
    payload = result.to_dict()
    payload["real_twin_modified"] = False
    return payload


app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
