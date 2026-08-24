"""
Feature Mapper.

Converts a Twin state into the EXACT feature vector the trained model
expects, per model/model_metadata.json (`feature_columns`,
`numerical_features`, `categorical_features`) and model/feature_schema.json
(dtype, kind, and - for categoricals - allowed_values for every feature).

This module does not invent values. If a Twin state is missing a required
feature, or a categorical value isn't one of the schema's allowed values,
`build_feature_row` raises `FeatureMappingError` with a specific,
actionable message rather than silently substituting a default.

The column set/order here is loaded directly from the shipped
model/model_metadata.json and model/feature_schema.json (not
hand-duplicated), so this file cannot silently drift out of sync with the
artifacts in model/ - if those files change, this module's behavior
changes with them automatically.
"""

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

# Authoritative feature column order/set: model/model_metadata.json's
# `feature_columns` if present, else derived from feature_schema.json,
# else (only if neither artifact is present yet) a hard-coded fallback
# matching the schema this MVP was integrated against - so the app can
# still start and describe its expected schema before you've added the
# model/ artifacts.
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

# name -> allowed_values (categorical only), for validation
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
    """Single-row DataFrame for one customer, ready to hand to preprocessing.joblib.

    Raises FeatureMappingError if the Twin state cannot satisfy the
    model's required feature schema - never silently substitutes a value.
    """
    feature_dict = state.to_feature_dict()
    _validate_row(feature_dict)
    return pd.DataFrame([{col: feature_dict[col] for col in FEATURE_COLUMNS}])


def build_feature_frame(states: Iterable[TwinState]) -> pd.DataFrame:
    """Multi-row DataFrame, e.g. for Monte Carlo batches or bulk scoring."""
    states = list(states)
    if not states:
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    rows = []
    for state in states:
        feature_dict = state.to_feature_dict()
        _validate_row(feature_dict)
        rows.append({col: feature_dict[col] for col in FEATURE_COLUMNS})
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


# Backwards-compatible aliases used elsewhere in the codebase
# (risk_intelligence.driver_identifier, scripts/train_model_example.py).
twin_state_to_feature_row = build_feature_row
twin_states_to_feature_frame = build_feature_frame
