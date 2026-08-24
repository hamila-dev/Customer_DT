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
