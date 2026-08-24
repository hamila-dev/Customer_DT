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
