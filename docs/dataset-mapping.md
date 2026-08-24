# Dataset Mapping

## Source

`data/customer_churn.csv` — the **insurance policyholder churn dataset**
(50,000 rows, 39 columns), provided together with the trained model
artifacts in `model/`. Per `model/model_metadata.json`'s own notes: *"a
public/synthetic proxy dataset shaped like Insurise's domain, not real
policyholder records. Treat metrics as directional; re-train on real
Insurise data before production use."* This is **not real Insurise
customer data**. `data/data_dictionary.csv` (also provided) documents
every column's intended meaning and is the authoritative source for the
column descriptions below.

> This supersedes the previous prototype dataset (`randomdata.csv`, a much
> smaller 11-column synthetic file used earlier in this project's history).
> If you see references to that older schema (`Claim Reason`, `BMI`,
> `Data confidentiality`, ...) elsewhere, they describe a prior integration
> and no longer apply — the current Twin schema is entirely driven by this
> dataset and `model/feature_schema.json`.

## Columns actually present

39 columns total. `customer_id` is a real per-row identifier (unlike the
previous dataset). Five columns are explicitly excluded from the model
per `model/model_metadata.json`'s `excluded_features`:
`customer_id`, `as_of_date`, `age_band`, `churn_type`,
`churn_probability_true`. The remaining 34 columns are the model's exact
feature schema (30 numerical + 4 categorical — see `model/README.md`).
`churn_flag` is the target.

## Identified relationships (real analysis, run against the actual file and the actual loaded model)

- **`missed_payment_flag` is an exact rule on `late_payment_count_12m`**:
  every row with `missed_payment_flag == 1` has
  `late_payment_count_12m >= 4`, and every row with `== 0` has
  `late_payment_count_12m < 4` (mean 4.24 vs 0.48) — this matches the data
  dictionary's own description verbatim ("1 if missed payments flag
  (>=4 late payments), else 0"). `TwinState.missed_payment_flag` is
  implemented as a derived property using this exact rule rather than
  stored independently, so it can never drift out of sync.
- **`num_claims_12m` is an exact sum**:
  `num_claims_12m == num_approved_claims_12m + num_rejected_claims_12m +
  num_pending_claims_12m` for all 50,000 rows (verified, zero mismatches).
- **`total_payout_amount_12m` is an exact product**:
  `total_payout_amount_12m == total_claim_amount_12m * payout_ratio_12m`
  wherever `total_claim_amount_12m > 0` (verified to within floating-point
  rounding). Where `total_claim_amount_12m == 0` (86.6% of rows —
  customers with no claims in the trailing 12 months), `payout_ratio_12m`
  still carries a baseline value in the 0.75–0.85 range rather than an
  undefined `0/0`. `TwinState.payout_ratio_12m` is a derived property
  implementing this exact relationship, with a documented `0.75` default
  for the zero-claim case (see `twin_engine/state/twin_state.py`).
- **`premium_to_coverage_ratio` is an exact ratio**:
  `premium_to_coverage_ratio == current_premium / coverage_amount`
  (verified to ~2.5e-6, floating-point rounding). Implemented as a derived
  `TwinState` property, never stored independently.
- **`premium_change_pct` is NOT a pure derived duplicate of
  `current_premium`/`premium_last_year`** — the source data carries
  independent noise beyond the exact formula `(current - last) / last`
  (observed deviation up to ~0.22 in a sample check). For Twin simulation
  purposes, `TwinState.premium_change_pct` recomputes the exact formula
  from `current_premium`/`premium_last_year` so a simulated premium change
  always produces an internally consistent feature vector — a documented
  MVP modeling choice, not a claim that it reproduces the original noise.
- **`current_premium` and `premium_last_year` are highly correlated**
  (Pearson r ≈ 0.98) — expected (premiums don't usually swing wildly
  year-over-year) but worth knowing when reading driver rankings that
  involve both.
- **`age_band` is a strict, deterministic bucketing of `age`** (18–24,
  25–34, ..., 75+) — correctly excluded from the model's feature set
  already (per `model_metadata.json`); not stored in the Twin at all.

**No BMI-style leakage this time.** Unlike the previous prototype
dataset, this one's feature-target relationships look realistic: e.g.
`missed_payment_flag=1` customers churn at 89.8% vs. 29.8% for
`missed_payment_flag=0` (a strong, plausible signal — missing payments is
a textbook churn precursor, not a data artifact); `complaint_flag=1`
customers churn at 55.1% vs. 29.1%; and churn rate rises smoothly with
`premium_change_pct` (17.4% for a >5% premium cut, up to 45.3% for a >10%
increase). The trained model's held-out metrics (accuracy 0.749, ROC-AUC
0.786 — see `model/README.md`) are realistic for this kind of problem,
not suspiciously perfect.

## Mapping: Dataset feature → Twin state → ML feature → Event

| Dataset feature | Twin state field | Category | Triggering event(s) |
|---|---|---|---|
| `customer_id` | `customer_id` | static, identity only (not an ML feature) | — |
| `age` | `age` | static, ML feature | — |
| `region_name` | `region_name` | static, ML feature | — |
| `marital_status` | `marital_status` | static, ML feature | — |
| `customer_tenure_months` | `customer_tenure_months` | static, ML feature | — |
| `multi_policy_flag` | `multi_policy_flag` | static, ML feature | — |
| `num_policies` | `num_policies` | static, ML feature | — |
| `policy_type` | `policy_type` | static, ML feature | — |
| `renewal_month` | `renewal_month` | static, ML feature | — |
| `payment_frequency` | `payment_frequency` | static, ML feature | — |
| `autopay_enabled` | `autopay_enabled` | static, ML feature | — |
| `current_premium` | `current_premium` | dynamic, ML feature | `premium_changed`, `policy_renewed` |
| `premium_last_year` | `premium_last_year` | dynamic, ML feature | `policy_renewed` (rolls `current_premium` forward) |
| `premium_change_pct` | *(derived property)* | derived, ML feature | recomputed whenever `current_premium`/`premium_last_year` change |
| `num_price_increases_last_3y` | `num_price_increases_last_3y` | dynamic, ML feature | `premium_changed` (if the new premium is higher) |
| `coverage_amount` | `coverage_amount` | dynamic, ML feature | `coverage_downgraded` |
| `premium_to_coverage_ratio` | *(derived property)* | derived, ML feature | recomputed whenever `current_premium`/`coverage_amount` change |
| `late_payment_count_12m` | `late_payment_count_12m` | dynamic, ML feature | `payment_missed`; reset by `policy_renewed` |
| `missed_payment_flag` | *(derived property)* | derived, ML feature | recomputed from `late_payment_count_12m >= 4` |
| `payment_method_change_flag` | `payment_method_change_flag` | **ML-only** (fed to model; no dataset-grounded event mutates it in this MVP) | — |
| `num_claims_12m`, `num_approved_claims_12m`, `num_rejected_claims_12m`, `num_pending_claims_12m` | same names | dynamic, ML features | `claim_created`; reset by `policy_renewed` |
| `avg_claim_amount` | `avg_claim_amount` | dynamic, ML feature (recomputed as `total_claim_amount_12m / num_claims_12m` on each claim) | `claim_created`; reset by `policy_renewed` |
| `total_claim_amount_12m` | `total_claim_amount_12m` | dynamic, ML feature | `claim_created`; reset by `policy_renewed` |
| `total_payout_amount_12m` | `total_payout_amount_12m` | dynamic, ML feature | `claim_created` (approved claims only); reset by `policy_renewed` |
| `payout_ratio_12m` | *(derived property)* | derived, ML feature | recomputed from `total_payout_amount_12m`/`total_claim_amount_12m` |
| `avg_settlement_time_days` | `avg_settlement_time_days` | dynamic, ML feature (overwritten by the latest claim's settlement time — a simplification; see `docs/event-model.md`) | `claim_created` |
| `days_since_last_claim` | `days_since_last_claim` | dynamic, ML feature | `claim_created` (resets to 0) |
| `num_contacts_12m` | `num_contacts_12m` | dynamic, ML feature | `engagement_changed`; reset by `policy_renewed` |
| `complaint_flag` | `complaint_flag` | dynamic, ML feature | `complaint_lodged`; reset by `policy_renewed` |
| `complaint_resolution_days` | `complaint_resolution_days` | dynamic, ML feature | `complaint_lodged`; reset by `policy_renewed` |
| `quote_requested_flag` | `quote_requested_flag` | dynamic, ML feature | `engagement_changed` (optional payload flag) |
| `coverage_downgrade_flag` | `coverage_downgrade_flag` | dynamic, ML feature | `coverage_downgraded` |
| `as_of_date` | *(not stored)* | excluded | — |
| `age_band` | *(not stored)* | excluded (derived from `age`, redundant) | — |
| `churn_type` | *(not stored)* | excluded (post-hoc label metadata) | — |
| `churn_probability_true` | *(not stored)* | excluded (ground-truth generation artifact) | — |
| `churn_flag` | `historical_churn_label` | reference/display only, **never an ML feature** | — |

## Assumptions made explicit

1. **The dataset represents one running claim ledger, not per-claim
   records.** `claim_created` increments counts/totals and recomputes
   averages rather than replacing them — a more realistic accumulation
   model than the previous dataset integration allowed for, since this
   dataset's claims columns are genuinely trailing-12-month aggregates.
2. **`policy_renewed` resets every trailing-12-month counter** (late
   payments, claims, complaints, contacts) to model the start of a fresh
   reporting period. A real system might carry some of this history
   forward across a renewal; this MVP resets it to match how the
   dataset's "_12m" columns are framed. Documented, not hidden.
3. **`avg_settlement_time_days` is overwritten, not averaged**, by each
   new claim's settlement time — a simplification that avoids
   implementing full historical claim-by-claim tracking for the MVP.
4. **`payment_method_change_flag` has no dedicated event** in this MVP —
   it's fed to the model as an "ML-only" feature (see the mapping table)
   that starts at its bootstrap value from the dataset and doesn't change
   during a demo session. A future iteration could add a
   `payment_method_changed` event following the same pattern as the other
   seven.

## What is NOT invented

- The four model artifacts (`churn_model.joblib`, `preprocessing.joblib`,
  `model_metadata.json`, `feature_schema.json`) were provided pre-trained
  and are used exactly as given — this repository never trains, retrains,
  or modifies them (see `model/README.md`).
- `model_metadata.json`'s `evaluation_metrics` (accuracy 0.7489, precision
  0.5873, recall 0.5641, F1 0.5755, ROC-AUC 0.7856) are the real,
  reported held-out metrics from the training run — not recomputed,
  invented, or adjusted anywhere in this codebase.
- No customer value, action cost, or action-effect number in `config.py`
  is claimed to be real business data — each is labelled as an "MVP
  simulation assumption" in the code, this documentation, and the API
  responses that use it (see `docs/recommendation-engine.md`).
