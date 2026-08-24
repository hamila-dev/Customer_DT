# Recommendation Engine

```
Risk Intelligence
       |
       v
Driver Identifier
       |
       v
Action Lookup
       |
       v
Effect Estimator
       |
       v
Expected Value Ranker
       |
       v
Recommended Action
```

`recommendation_engine/engine.py`'s `RecommendationEngine.recommend(state)`
is a thin orchestrator that calls these four components in order. Each is
independently testable/replaceable.

## Driver Identifier (`risk_intelligence/driver_identifier.py`)

The original technical design document sketched driver identification for
a logistic-regression-style model (`argmax |w_j * S_j|`). **That does not
apply here — the model is a Random Forest.** Instead:

1. Read the Random Forest's real, trained `feature_importances_` (never
   invented).
2. Aggregate those importances back onto the *original* raw dataset
   columns — a one-hot-encoded categorical column's several transformed
   importances are summed back together, using the fitted preprocessing
   pipeline's `get_feature_names_out()` when available.
3. Combine each column's global importance with how unusual the specific
   customer's value is for that column, computed from the real dataset
   (`data/customer_churn.csv`) — a numeric column's percentile rank, or a
   categorical column's empirical churn-rate deviation from the overall
   rate. Nothing here is invented; both the importances and the
   reference statistics come from real, loaded data.
4. Rank columns by `importance * salience`, return the top N
   (`config.TOP_N_DRIVERS`, default 3).

Language used is deliberately non-causal: *"is an important model
feature for this customer's risk assessment"*, never *"causes churn."*
Certain feature groups in this dataset are closely/exactly related by
construction (e.g. `num_claims_12m` = approved + rejected + pending
exactly; `total_payout_amount_12m` = `total_claim_amount_12m` ×
`payout_ratio_12m` exactly — see `docs/dataset-mapping.md`). When one of
these ranks as a top driver, the explanation text attaches a brief
multicollinearity note so it isn't read as three independent signals.

## Action Lookup (`recommendation_engine/action_lookup.py`)

A simple, deterministic, hand-written rule table
(`config.ACTION_RULES`, keyed by raw dataset column name) mapping a driver
to a candidate action:

| Driver | Candidate action |
|---|---|
| `missed_payment_flag`, `late_payment_count_12m`, `payment_method_change_flag` | Payment plan review |
| `premium_change_pct`, `current_premium`, `premium_to_coverage_ratio`, `num_price_increases_last_3y` | Premium review |
| `complaint_flag`, `complaint_resolution_days` | Service recovery outreach |
| `num_claims_12m`, `num_rejected_claims_12m`, `avg_settlement_time_days`, `payout_ratio_12m` | Claims review & proactive outreach |
| `coverage_downgrade_flag` | Coverage review |
| `quote_requested_flag` | Retention offer review |
| `num_contacts_12m` | Customer engagement outreach |
| *(no strong driver match)* | General account review |

This is **not** a learned policy — it's an MVP prototype rule set written
by hand from the features actually present in the dataset, and it does
not claim these actions are proven to reduce churn.

## Effect Estimator (`recommendation_engine/effect_estimator.py`)

There is no real intervention-outcome dataset for this MVP. This
component returns fixed, documented assumption values
(`config.ASSUMED_ACTION_EFFECT`) — e.g. "claims review outreach is assumed
to reduce churn probability by 0.08" — explicitly labelled as **MVP
simulation assumptions**, not empirical facts. The class's interface is
structured so it can be swapped for a learned uplift/treatment-effect
model later without changing its caller.

## Expected Value Ranker (`recommendation_engine/expected_value_ranker.py`)

Implements, for the MVP:

```
EV(a|S) = P(churn) * tau(a,S) * Value(S) - Cost(a)
```

- `P(churn)` — the customer's real churn probability (Risk Intelligence)
- `tau(a,S)` — the assumed risk reduction for action `a` (Effect Estimator)
- `Value(S)` — a configurable assumed customer value
  (`config.DEFAULT_CUSTOMER_VALUE`)
- `Cost(a)` — a configurable assumed action cost
  (`config.ASSUMED_ACTION_COST`)

Candidate actions are ranked by `EV(a|S)` descending; the top-ranked
action is the `recommended_action` in
`GET /api/customers/{id}/recommendations`. `Value(S)` and `Cost(a)` are
prototype placeholders, not real Finance/CRM figures — see `config.py`.

## What this component intentionally does not claim

- It does not claim any action is proven to reduce churn.
- It does not claim `Value(S)` or `Cost(a)` are real business figures.
- It does not claim the Driver Identifier's ranking is a causal
  explanation of *why* a customer might churn.

Every one of these caveats is attached directly to the relevant API
response fields (`basis`, `description`, `explanation`), not just this
document, so an Administrator using the frontend sees them in context.
