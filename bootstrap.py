"""
Bootstrap loader.

Reads data/customer_churn.csv (the insurance policyholder churn dataset -
see docs/dataset-mapping.md) and initializes the Twin State Store with one
TwinState per customer row, up to config.MAX_CUSTOMERS_TO_LOAD (default
300) rows.

This dataset is a PUBLIC/SYNTHETIC PROTOTYPE DATASET, not real Insurise
production data (see docs/dataset-mapping.md and the README). It DOES
include a real `customer_id` column (unlike the previous prototype
dataset), which is used directly as the Twin's customer_id.

Only runs (populates the store) if the store is currently empty, so
restarting the API does not silently reset events you've already
generated during a demo session (state persists in storage/twin_states.json).
"""

from __future__ import annotations

import logging

import pandas as pd

from twin_engine.state.state_store import TwinStateStore, twin_state_store
from twin_engine.state.twin_state import TwinState

import config

logger = logging.getLogger(__name__)


def _row_to_twin_state(row) -> TwinState:
    return TwinState(
        customer_id=f"C{int(row['customer_id']):06d}",
        age=int(row["age"]),
        region_name=str(row["region_name"]),
        marital_status=str(row["marital_status"]),
        customer_tenure_months=int(row["customer_tenure_months"]),
        multi_policy_flag=int(row["multi_policy_flag"]),
        num_policies=int(row["num_policies"]),
        policy_type=str(row["policy_type"]),
        renewal_month=int(row["renewal_month"]),
        payment_frequency=str(row["payment_frequency"]),
        autopay_enabled=int(row["autopay_enabled"]),
        current_premium=float(row["current_premium"]),
        premium_last_year=float(row["premium_last_year"]),
        num_price_increases_last_3y=int(row["num_price_increases_last_3y"]),
        coverage_amount=float(row["coverage_amount"]),
        late_payment_count_12m=int(row["late_payment_count_12m"]),
        num_claims_12m=int(row["num_claims_12m"]),
        num_approved_claims_12m=int(row["num_approved_claims_12m"]),
        num_rejected_claims_12m=int(row["num_rejected_claims_12m"]),
        num_pending_claims_12m=int(row["num_pending_claims_12m"]),
        total_claim_amount_12m=float(row["total_claim_amount_12m"]),
        total_payout_amount_12m=float(row["total_payout_amount_12m"]),
        avg_claim_amount=float(row["avg_claim_amount"]),
        avg_settlement_time_days=int(row["avg_settlement_time_days"]),
        days_since_last_claim=int(row["days_since_last_claim"]),
        num_contacts_12m=int(row["num_contacts_12m"]),
        complaint_flag=int(row["complaint_flag"]),
        complaint_resolution_days=int(row["complaint_resolution_days"]),
        quote_requested_flag=int(row["quote_requested_flag"]),
        coverage_downgrade_flag=int(row["coverage_downgrade_flag"]),
        payment_method_change_flag=int(row["payment_method_change_flag"]),
        historical_churn_label=int(row["churn_flag"]),
    )


def load_initial_customers(store: TwinStateStore = twin_state_store, force: bool = False) -> int:
    """
    Populate `store` from data/customer_churn.csv.
    Returns the number of customers loaded (0 if skipped because the store
    was already populated and force=False).
    """
    if not force and not store.is_empty():
        logger.info("Twin State Store already populated (%d customers) - skipping bootstrap load.", store.count())
        return 0

    if not config.CUSTOMER_DATA_CSV.exists():
        raise FileNotFoundError(
            f"Dataset not found at {config.CUSTOMER_DATA_CSV}. "
            "Place the insurance policyholder churn CSV there before starting the API."
        )

    df = pd.read_csv(config.CUSTOMER_DATA_CSV)
    df = df.head(config.MAX_CUSTOMERS_TO_LOAD)

    states = [_row_to_twin_state(row) for _, row in df.iterrows()]
    store.bulk_save(states)

    logger.info("Loaded %d customers from %s into the Twin State Store.", len(states), config.CUSTOMER_DATA_CSV)
    return len(states)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_initial_customers()
