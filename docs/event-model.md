# Event Model

Every event supported by this MVP is grounded in an actual column of
`data/customer_churn.csv` — see `docs/dataset-mapping.md` for the full
column analysis. This document lists each event's contract.

## `payment_missed`

- **Affected Twin state:** `late_payment_count_12m`
- **Affected ML feature(s):** `late_payment_count_12m`, derived `missed_payment_flag`
- **Expected state change:** `late_payment_count_12m += payload.count`
  (default 1). `missed_payment_flag` updates automatically (it's a derived
  property: `1` once `late_payment_count_12m >= 4`, matching the exact
  rule in `data/data_dictionary.csv`).
- **Payload:** `{"count": <int, optional, default 1>}`

## `claim_created`

- **Affected Twin state:** `num_claims_12m`,
  `num_approved_claims_12m`/`num_rejected_claims_12m`/`num_pending_claims_12m`
  (whichever matches `outcome`), `total_claim_amount_12m`,
  `avg_claim_amount`, `total_payout_amount_12m` (if approved),
  `avg_settlement_time_days`, `days_since_last_claim`
- **Affected ML feature(s):** all of the above, plus derived `payout_ratio_12m`
- **Expected state change:** a new claim is filed and accumulates into
  the running 12-month totals (this dataset's claims columns are
  trailing-12-month aggregates, so `claim_created` increments/accumulates
  rather than replacing — see `docs/dataset-mapping.md`).
  `avg_claim_amount` is recomputed as `total_claim_amount_12m /
  num_claims_12m`. `days_since_last_claim` resets to `0`.
- **Payload:** `{"claim_amount": <number>, "outcome": "approved"|"rejected"|"pending" (default "approved"), "settlement_time_days": <int, optional>, "payout_fraction": <float, optional, default 1.0 if approved else 0.0>}`

## `premium_changed`

- **Affected Twin state:** `current_premium`, `num_price_increases_last_3y`
- **Affected ML feature(s):** `current_premium`,
  `num_price_increases_last_3y`, derived `premium_change_pct` and
  `premium_to_coverage_ratio`
- **Expected state change:** `current_premium` moves to a new value,
  given either directly (`current_premium`) or as a relative change
  (`change_pct`, e.g. `0.15` for +15%) against the current premium.
  `num_price_increases_last_3y` increments only if the new premium is
  higher than the old one.
- **Payload:** `{"current_premium": <number>}` **or**
  `{"change_pct": <number>}` (e.g. `-0.05` for a 5% decrease)

## `policy_renewed`

- **Affected Twin state:** `premium_last_year`, and a fresh
  trailing-12-month window: `late_payment_count_12m`, `num_claims_12m`,
  `num_approved_claims_12m`/`num_rejected_claims_12m`/`num_pending_claims_12m`,
  `total_claim_amount_12m`, `total_payout_amount_12m`, `avg_claim_amount`,
  `num_contacts_12m`, `complaint_flag`, `complaint_resolution_days`
- **Affected ML feature(s):** all of the above, plus every derived
  feature that depends on them (`premium_change_pct`,
  `missed_payment_flag`, `payout_ratio_12m`)
- **Expected state change:** `current_premium` rolls into
  `premium_last_year` (so a subsequent `premium_changed` event compares
  against it), and every trailing-12-month counter resets to zero —
  see the ASSUMPTION in `docs/dataset-mapping.md` about why.
- **Payload:** none required.

## `engagement_changed`

- **Affected Twin state:** `num_contacts_12m`, `quote_requested_flag`
- **Affected ML feature(s):** `num_contacts_12m`, `quote_requested_flag`
- **Expected state change:** `num_contacts_12m += payload.contact_delta`
  (default 1, floored at 0). If `payload.quote_requested` is true,
  `quote_requested_flag -> 1` (a customer shopping around for a quote is a
  distinct, meaningful signal from a routine contact).
- **Payload:** `{"contact_delta": <int, optional, default 1>, "quote_requested": <bool, optional>}`

## `coverage_downgraded`

- **Affected Twin state:** `coverage_amount`, `coverage_downgrade_flag`
- **Affected ML feature(s):** `coverage_amount`,
  `coverage_downgrade_flag`, derived `premium_to_coverage_ratio`
- **Expected state change:** `coverage_amount` decreases, given either
  directly (`coverage_amount`) or as a fractional cut (`reduction_pct`,
  e.g. `0.2` for -20%; defaults to a 20% cut if neither is given).
  `coverage_downgrade_flag -> 1`.
- **Payload:** `{"coverage_amount": <number>}` **or**
  `{"reduction_pct": <number>}`

## `complaint_lodged`

- **Affected Twin state:** `complaint_flag`, `complaint_resolution_days`
- **Affected ML feature(s):** `complaint_flag`, `complaint_resolution_days`
- **Expected state change:** `complaint_flag -> 1`;
  `complaint_resolution_days` set from the payload (default 0 = not yet
  resolved/unknown).
- **Payload:** `{"resolution_days": <int, optional, default 0>}`

## Event flow

```
Event Generator (local thread)  --\
                                    +--> Event --> Event Transition Handler --> Twin State Store --> Risk recalculation
POST /api/events (manual/API)   --/
```

Both producers construct the same `Event` dataclass
(`twin_engine/events/event.py`) and hand it to
`StateSynchronizer.process_event` — there is exactly one code path an
event travels through regardless of source. This is deliberate: it's the
seam that would let a real Kafka consumer replace the local generator
later without touching the Event Transition Handler, Twin State Store, or
anything downstream (see docs/architecture.md).

Every processed event is appended (in order) to that customer's
`TwinState.event_history` (visible via `GET /api/customers/{id}/events`
and the Digital Twin frontend view) and to the flat audit log at
`storage/event_log.json`.
