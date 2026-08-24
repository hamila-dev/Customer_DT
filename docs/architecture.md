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
