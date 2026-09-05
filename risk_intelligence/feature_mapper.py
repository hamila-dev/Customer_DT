

"""Map `TwinState` objects to the trained model's feature schema."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List

import pandas as pd

from twin_engine.state.twin_state import TwinState

import config


class FeatureMappingError(ValueError):
    """Raised when a Twin state cannot be mapped to the model's required feature schema."""


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _load_feature_schema() -> List[Dict[str, Any]]:
    if not config.FEATURE_SCHEMA_PATH.exists():
        return []
    return _load_json(config.FEATURE_SCHEMA_PATH)["features"]


def _load_model_metadata() -> Dict[str, Any]:
    if not config.MODEL_METADATA_PATH.exists():
        return {}
    return _load_json(config.MODEL_METADATA_PATH)


_FEATURE_SCHEMA = _load_feature_schema()
_MODEL_METADATA = _load_model_metadata()

# Prefer the shipped metadata, then the feature schema, so preprocessing sees
# the same column order used during training. The fallback keeps the app
# inspectable before model artifacts are installed.
_FALLBACK_FEATURE_COLUMNS = [
    "age", "customer_tenure_months", "multi_policy_flag", "num_policies", "renewal_month",
    "current_premium", "premium_last_year", "premium_change_pct", "num_price_increases_last_3y",
    "coverage_amount", "premium_to_coverage_ratio", "autopay_enabled", "late_payment_count_12m",
    "missed_payment_flag", "payment_method_change_flag", "num_claims_12m", "num_approved_claims_12m",
    "num_rejected_claims_12m", "num_pending_claims_12m", "avg_claim_amount", "total_claim_amount_12m",
    "total_payout_amount_12m", "payout_ratio_12m", "avg_settlement_time_days", "days_since_last_claim",
    "num_contacts_12m", "complaint_flag", "complaint_resolution_days", "quote_requested_flag",
    "coverage_downgrade_flag", "region_name", "marital_status", "policy_type", "payment_frequency",
]

FEATURE_COLUMNS: List[str] = (
    _MODEL_METADATA.get("feature_columns")
    or [f["name"] for f in _FEATURE_SCHEMA]
    or _FALLBACK_FEATURE_COLUMNS
)

CATEGORICAL_COLUMNS: List[str] = (
    _MODEL_METADATA.get("categorical_features")
    or [f["name"] for f in _FEATURE_SCHEMA if f.get("kind") == "categorical"]
    or ["region_name", "marital_status", "policy_type", "payment_frequency"]
)

NUMERIC_COLUMNS: List[str] = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]

# Categorical values are checked before they reach the preprocessing pipeline.
_ALLOWED_VALUES: Dict[str, List[str]] = {
    f["name"]: f["allowed_values"] for f in _FEATURE_SCHEMA if f.get("allowed_values")
}


def _validate_row(feature_dict: Dict[str, Any]) -> None:
    missing = [c for c in FEATURE_COLUMNS if c not in feature_dict]
    if missing:
        raise FeatureMappingError(
            f"Twin state is missing required model feature(s): {missing}. "
            f"TwinState.to_feature_dict() must supply every column in "
            f"model/model_metadata.json's feature_columns - this indicates "
            f"the Twin schema and the model schema have drifted apart. "
            f"See docs/dataset-mapping.md."
        )

    for column, allowed in _ALLOWED_VALUES.items():
        if column in feature_dict and feature_dict[column] not in allowed:
            raise FeatureMappingError(
                f"Twin state field '{column}' has value {feature_dict[column]!r}, "
                f"which is not one of the allowed values in model/feature_schema.json: "
                f"{allowed}. Refusing to silently pass an out-of-schema category to "
                f"the model."
            )


def build_feature_row(state: TwinState) -> pd.DataFrame:
    """Build a validated single-row frame for model preprocessing."""
    feature_dict = state.to_feature_dict()
    _validate_row(feature_dict)
    return pd.DataFrame([{col: feature_dict[col] for col in FEATURE_COLUMNS}])


def build_feature_frame(states: Iterable[TwinState]) -> pd.DataFrame:
    """Build a validated multi-row frame for batch scoring."""
    states = list(states)
    if not states:
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    rows = []
    for state in states:
        feature_dict = state.to_feature_dict()
        _validate_row(feature_dict)
        rows.append({col: feature_dict[col] for col in FEATURE_COLUMNS})
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


# Preserve the names used by existing callers.
twin_state_to_feature_row = build_feature_row
twin_states_to_feature_frame = build_feature_frame
