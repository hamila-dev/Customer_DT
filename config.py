

"""Shared paths and business assumptions for the Customer Twin application."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
STORAGE_DIR = BASE_DIR / "storage"  # local JSON persistence

CUSTOMER_DATA_CSV = DATA_DIR / "customer_churn.csv"

MODEL_PATH = MODEL_DIR / "churn_model.joblib"
PREPROCESSING_PATH = MODEL_DIR / "preprocessing.joblib"
MODEL_METADATA_PATH = MODEL_DIR / "model_metadata.json"
FEATURE_SCHEMA_PATH = MODEL_DIR / "feature_schema.json"

TWIN_STORE_PATH = STORAGE_DIR / "twin_states.json"
EVENT_LOG_PATH = STORAGE_DIR / "event_log.json"
RISK_HISTORY_PATH = STORAGE_DIR / "risk_history.json"

# Keep startup responsive; the source CSV contains 50,000 rows.
MAX_CUSTOMERS_TO_LOAD = 300

# Business thresholds chosen from the reference prediction distribution;
# they are not validated decision boundaries.
RISK_THRESHOLDS = {
    "high": 0.60,
    "medium": 0.35,
}


def risk_level_from_probability(probability: float) -> str:
    if probability >= RISK_THRESHOLDS["high"]:
        return "HIGH"
    if probability >= RISK_THRESHOLDS["medium"]:
        return "MEDIUM"
    return "LOW"


# Number of drivers shown per customer.
TOP_N_DRIVERS = 3

# These values describe simulation perturbations, not measured real-world variance.
MONTE_CARLO_DEFAULT_TRIALS = 300

# Relative standard deviation applied to numeric scenario parameters.
MONTE_CARLO_NUMERIC_NOISE_STD = 0.10

# Keys must match raw dataset feature names in feature_mapper.FEATURE_COLUMNS.
ACTION_RULES = {
    "missed_payment_flag": {
        "action": "payment_plan_review",
        "label": "Payment plan review",
        "description": (
            "The customer has missed multiple payments in the last 12 "
            "months. Offer a payment plan review (e.g. switch to monthly "
            "autopay) before the next renewal."
        ),
    },
    "late_payment_count_12m": {
        "action": "payment_plan_review",
        "label": "Payment plan review",
        "description": (
            "A rising count of late payments contributes strongly to this "
            "customer's risk assessment; a payment plan or autopay "
            "conversation may help before it escalates."
        ),
    },
    "premium_change_pct": {
        "action": "premium_review",
        "label": "Premium review",
        "description": (
            "A recent premium increase contributes strongly to this "
            "customer's risk assessment; offer a premium/coverage review."
        ),
    },
    "current_premium": {
        "action": "premium_review",
        "label": "Premium review",
        "description": "Offer a premium/coverage review to check the policy still fits the customer's budget.",
    },
    "premium_to_coverage_ratio": {
        "action": "premium_review",
        "label": "Premium review",
        "description": (
            "This customer's premium is high relative to their coverage "
            "amount compared to peers; review pricing and coverage fit."
        ),
    },
    "num_price_increases_last_3y": {
        "action": "premium_review",
        "label": "Premium review",
        "description": "Repeated premium increases over the last 3 years contribute strongly to this customer's risk assessment.",
    },
    "complaint_flag": {
        "action": "service_recovery_outreach",
        "label": "Service recovery outreach",
        "description": (
            "The customer has an open or recent complaint; proactively "
            "follow up to confirm resolution and rebuild confidence."
        ),
    },
    "complaint_resolution_days": {
        "action": "service_recovery_outreach",
        "label": "Service recovery outreach",
        "description": "A slow complaint resolution contributes strongly to this customer's risk assessment; follow up personally.",
    },
    "num_claims_12m": {
        "action": "claims_review_outreach",
        "label": "Claims review & proactive outreach",
        "description": "Contact the customer to review recent claim activity and confirm satisfaction with claims handling.",
    },
    "num_rejected_claims_12m": {
        "action": "claims_review_outreach",
        "label": "Claims review & proactive outreach",
        "description": (
            "The customer has had claim(s) rejected recently, a common "
            "source of dissatisfaction; review the claim decision with them."
        ),
    },
    "avg_settlement_time_days": {
        "action": "claims_review_outreach",
        "label": "Claims review & proactive outreach",
        "description": "Slow claim settlement contributes strongly to this customer's risk assessment; check in on their most recent claim.",
    },
    "payout_ratio_12m": {
        "action": "claims_review_outreach",
        "label": "Claims review & proactive outreach",
        "description": "A low payout-to-claim ratio contributes strongly to this customer's risk assessment; review claims handling with them.",
    },
    "coverage_downgrade_flag": {
        "action": "coverage_review",
        "label": "Coverage review",
        "description": (
            "The customer recently downgraded coverage, often a sign of "
            "price sensitivity; review whether current coverage still "
            "meets their needs."
        ),
    },
    "quote_requested_flag": {
        "action": "retention_offer_review",
        "label": "Retention offer review",
        "description": (
            "The customer has recently requested a quote (a common "
            "shopping-around signal); consider a proactive retention offer."
        ),
    },
    "num_contacts_12m": {
        "action": "engagement_outreach",
        "label": "Customer engagement outreach",
        "description": "This customer's contact/engagement pattern contributes strongly to their risk assessment; reach out to check in.",
    },
    "payment_method_change_flag": {
        "action": "payment_plan_review",
        "label": "Payment plan review",
        "description": "A recent payment-method change contributes strongly to this customer's risk assessment; confirm their new payment details are working smoothly.",
    },
    "default": {
        "action": "general_account_review",
        "label": "General account review",
        "description": "No specific dominant driver identified; perform a general account check-in.",
    },
}

# Assumed absolute probability reductions; no intervention-outcome data backs them.
ASSUMED_ACTION_EFFECT = {
    "payment_plan_review": 0.09,
    "premium_review": 0.06,
    "service_recovery_outreach": 0.10,
    "claims_review_outreach": 0.07,
    "coverage_review": 0.05,
    "retention_offer_review": 0.08,
    "engagement_outreach": 0.03,
    "general_account_review": 0.02,
}

# Placeholder customer value and action costs until Finance/CRM data exists.
DEFAULT_CUSTOMER_VALUE = 3000.0

ASSUMED_ACTION_COST = {
    "payment_plan_review": 8.0,
    "premium_review": 10.0,
    "service_recovery_outreach": 20.0,
    "claims_review_outreach": 15.0,
    "coverage_review": 10.0,
    "retention_offer_review": 25.0,
    "engagement_outreach": 5.0,
    "general_account_review": 5.0,
}

EVENT_GENERATOR_DEFAULT_INTERVAL_SECONDS = 5
EVENT_GENERATOR_SCENARIOS = [
    "payment_missed",
    "claim_created",
    "premium_changed",
    "policy_renewed",
    "engagement_changed",
    "coverage_downgraded",
    "complaint_lodged",
]

API_HOST = "0.0.0.0"
API_PORT = 8000
CORS_ALLOW_ORIGINS = ["*"]  # Restrict before production deployment.
