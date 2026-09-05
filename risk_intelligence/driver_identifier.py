
"""Rank customer risk drivers using model importance and dataset salience."""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from risk_intelligence.feature_mapper import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS
from risk_intelligence.predictor import ChurnPredictor, ModelNotAvailableError, churn_predictor
from twin_engine.state.twin_state import TwinState

import config

logger = logging.getLogger(__name__)

# Caveats attached to explanations when related features rank as drivers.
MULTICOLLINEARITY_NOTES: Dict[str, str] = {
    "num_claims_12m": "Note: this is the exact sum of approved + rejected + pending claims (see docs/dataset-mapping.md).",
    "num_approved_claims_12m": "Note: closely tied to num_claims_12m by construction (see docs/dataset-mapping.md).",
    "total_payout_amount_12m": "Note: total_payout_amount_12m = total_claim_amount_12m x payout_ratio_12m exactly (see docs/dataset-mapping.md).",
    "payout_ratio_12m": "Note: derived exactly from total_payout_amount_12m / total_claim_amount_12m (see docs/dataset-mapping.md).",
    "premium_to_coverage_ratio": "Note: derived exactly from current_premium / coverage_amount (see docs/dataset-mapping.md).",
    "current_premium": "Note: highly correlated with premium_last_year (~0.98) in the reference dataset (see docs/dataset-mapping.md).",
    "premium_last_year": "Note: highly correlated with current_premium (~0.98) in the reference dataset (see docs/dataset-mapping.md).",
}


@dataclass
class Driver:
    feature: str
    raw_value: Any
    global_importance: float
    salience: float
    combined_score: float
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature": self.feature,
            "value": self.raw_value,
            "global_importance": round(self.global_importance, 4),
            "salience": round(self.salience, 4),
            "combined_score": round(self.combined_score, 4),
            "explanation": self.explanation,
        }


@functools.lru_cache(maxsize=1)
def _reference_dataframe() -> pd.DataFrame:
    """Load the reference distribution once for salience calculations."""
    return pd.read_csv(config.CUSTOMER_DATA_CSV)


def _numeric_salience(column: str, value: float) -> float:
    """Score how far a numeric value is from the reference median."""
    df = _reference_dataframe()
    if column not in df.columns:
        return 0.5
    series = df[column].dropna()
    if series.empty:
        return 0.5
    percentile = (series < value).mean()
    return abs(percentile - 0.5) * 2


def _categorical_salience(column: str, value: str) -> float:
    """Score a category by its normalized deviation from overall churn."""
    df = _reference_dataframe()
    if column not in df.columns or "churn_flag" not in df.columns:
        return 0.5
    overall_rate = df["churn_flag"].mean()
    subset = df[df[column] == value]
    if subset.empty:
        return 0.5
    category_rate = subset["churn_flag"].mean()
    max_possible_deviation = max(overall_rate, 1 - overall_rate)
    if max_possible_deviation == 0:
        return 0.5
    return min(abs(category_rate - overall_rate) / max_possible_deviation, 1.0)


def _aggregate_importances_by_raw_column(predictor: ChurnPredictor) -> Dict[str, float]:
    """Aggregate transformed model importances back to raw feature columns."""
    importances = np.asarray(predictor.model.feature_importances_)

    transformed_names = None
    get_names_fn = getattr(predictor.preprocessing, "get_feature_names_out", None)
    if callable(get_names_fn):
        try:
            transformed_names = list(get_names_fn())
        except Exception:
            logger.warning("preprocessing.get_feature_names_out() failed; falling back to raw column alignment.")
            transformed_names = None

    aggregated = {col: 0.0 for col in FEATURE_COLUMNS}

    if transformed_names is not None and len(transformed_names) == len(importances):
        for name, importance in zip(transformed_names, importances):
            matched_column = None
            # Match longer names first so a short name cannot match a substring.
            for raw_col in sorted(FEATURE_COLUMNS, key=len, reverse=True):
                if raw_col in name:
                    matched_column = raw_col
                    break
            if matched_column is not None:
                aggregated[matched_column] += float(importance)
            else:
                logger.debug("Could not map transformed feature '%s' back to a raw column; ignoring.", name)
    elif len(importances) == len(FEATURE_COLUMNS):
        for col, importance in zip(FEATURE_COLUMNS, importances):
            aggregated[col] = float(importance)
    else:
        logger.warning(
            "Could not align %d model feature importances to %d raw columns; "
            "driver ranking will fall back to equal weighting.",
            len(importances),
            len(FEATURE_COLUMNS),
        )
        aggregated = {col: 1.0 / len(FEATURE_COLUMNS) for col in FEATURE_COLUMNS}

    return aggregated


class DriverIdentifier:
    """Combine model importance with customer-specific feature salience."""

    def __init__(self, predictor: ChurnPredictor = churn_predictor):
        self._predictor = predictor

    def identify(self, state: TwinState, top_n: int = config.TOP_N_DRIVERS) -> List[Driver]:
        if not self._predictor.is_available:
            raise ModelNotAvailableError(ChurnPredictor.MISSING_ARTIFACT_MESSAGE)

        aggregated_importance = _aggregate_importances_by_raw_column(self._predictor)
        feature_values = state.to_feature_dict()

        drivers: List[Driver] = []
        for column in FEATURE_COLUMNS:
            value = feature_values[column]
            importance = aggregated_importance.get(column, 0.0)

            if column in NUMERIC_COLUMNS:
                salience = _numeric_salience(column, float(value))
            else:
                salience = _categorical_salience(column, str(value))

            combined_score = importance * salience

            explanation = (
                f"'{column}' (current value: {value}) is an important model feature for "
                f"this customer's risk assessment."
            )
            if column in MULTICOLLINEARITY_NOTES:
                explanation += " " + MULTICOLLINEARITY_NOTES[column]

            drivers.append(
                Driver(
                    feature=column,
                    raw_value=value,
                    global_importance=importance,
                    salience=salience,
                    combined_score=combined_score,
                    explanation=explanation,
                )
            )

        drivers.sort(key=lambda d: d.combined_score, reverse=True)
        return drivers[:top_n]


driver_identifier = DriverIdentifier()
