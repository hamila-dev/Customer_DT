# Model Artifacts

**These artifacts are fixed and provided. The application performs
INFERENCE ONLY — it never trains, retrains, replaces, or modifies the
model.** If you want a different model, retrain it yourself outside this
repository and replace these four files with the new versions (keeping
the same schema described below).

```
model/
├── churn_model.joblib       # trained sklearn RandomForestClassifier
├── preprocessing.joblib     # fitted sklearn ColumnTransformer
├── model_metadata.json      # training run metadata (real, computed - see below)
└── feature_schema.json      # authoritative feature contract (dtype/kind/allowed values)
```

`risk_intelligence/predictor.py` loads all four at startup. If any of
`churn_model.joblib` / `preprocessing.joblib` are missing, every endpoint
that needs a prediction returns a clear `503` explaining what's missing
instead of a fake prediction; the rest of the API (customers list, Twin
state, events) still works.

## What's actually in these files (verified during integration)

- **`preprocessing.joblib`** is a `sklearn.compose.ColumnTransformer` with
  two branches: `num` (median imputation on the 30 numerical columns) and
  `cat` (most-frequent imputation + one-hot encoding on the 4 categorical
  columns). It exposes `get_feature_names_out()`, which
  `risk_intelligence/driver_identifier.py` uses to map the Random
  Forest's `feature_importances_` back onto the original raw columns.
- **`churn_model.joblib`** is a `sklearn.ensemble.RandomForestClassifier`
  (`n_estimators=300`, `min_samples_leaf=5`, `class_weight="balanced"`),
  `n_features_in_ = 46` (30 numeric + 16 one-hot categorical columns —
  matches `preprocessing.transform(...)`'s output exactly), `classes_ =
  [0, 1]` (class index `1` = churned, matching `churn_class_index` default
  in `risk_intelligence/predictor.py`).
- **`model_metadata.json`** records real training-run metadata: feature
  columns, excluded columns, class distribution, and held-out evaluation
  metrics. **Nothing in it is invented** — these are the numbers reported
  at training time:

  | Metric | Value |
  |---|---|
  | Accuracy | 0.7489 |
  | Precision | 0.5873 |
  | Recall | 0.5641 |
  | F1 | 0.5755 |
  | ROC-AUC | 0.7856 |

  (40,000 train rows / 10,000 test rows, `random_state=42`.) These are
  realistic numbers for a synthetic-but-plausible dataset — nowhere near
  the suspiciously perfect scores the *previous* prototype dataset
  produced (see `docs/dataset-mapping.md` for that history and why this
  dataset doesn't have the same issue).
- **`feature_schema.json`** is the authoritative feature contract:
  `risk_intelligence/feature_mapper.py` loads `FEATURE_COLUMNS` and
  `CATEGORICAL_COLUMNS` directly from `model_metadata.json`/
  `feature_schema.json` at import time rather than hand-duplicating them,
  and validates every categorical value against this file's
  `allowed_values` before scoring.

## Feature schema the Twin must supply

34 features, exactly matching `model/feature_schema.json`. See
`docs/dataset-mapping.md` for the full Dataset → Twin → ML feature → Event
mapping table.

30 numerical: `age`, `customer_tenure_months`, `multi_policy_flag`,
`num_policies`, `renewal_month`, `current_premium`, `premium_last_year`,
`premium_change_pct`, `num_price_increases_last_3y`, `coverage_amount`,
`premium_to_coverage_ratio`, `autopay_enabled`, `late_payment_count_12m`,
`missed_payment_flag`, `payment_method_change_flag`, `num_claims_12m`,
`num_approved_claims_12m`, `num_rejected_claims_12m`,
`num_pending_claims_12m`, `avg_claim_amount`, `total_claim_amount_12m`,
`total_payout_amount_12m`, `payout_ratio_12m`, `avg_settlement_time_days`,
`days_since_last_claim`, `num_contacts_12m`, `complaint_flag`,
`complaint_resolution_days`, `quote_requested_flag`,
`coverage_downgrade_flag`.

4 categorical (with `feature_schema.json`-enforced allowed values):
`region_name` (7 NZ regions), `marital_status` (`Married`/`Single`),
`policy_type` (`Auto`/`Health`/`Home`/`Life`/`Travel`), `payment_frequency`
(`Annual`/`Monthly`).

## If a Twin state can't satisfy this schema

`risk_intelligence/feature_mapper.py`'s `build_feature_row` never
substitutes a default for a missing or invalid feature. It raises
`FeatureMappingError`, which the API surfaces as an HTTP `422` with a
specific message naming the missing field or invalid category — see the
`@app.exception_handler(FeatureMappingError)` in `api/main.py`.

## Excluded from the model (per `model_metadata.json`'s `excluded_features`)

`customer_id`, `as_of_date`, `age_band` (a strict bucketing of `age`),
`churn_type`, `churn_probability_true` — none of these are fed to the
model; `age_band`/`churn_type`/`churn_probability_true` aren't stored in
the Twin at all, and `customer_id`/`as_of_date` are identity/reference
only.

## Proxy data, not real Insurise data

Per `model_metadata.json`'s own notes field: *"Trained on a public/synthetic
proxy dataset shaped like Insurise's domain, not real policyholder
records. Treat metrics as directional; re-train on real Insurise data
before production use."* This repository does not alter that framing —
see `docs/dataset-mapping.md`.
