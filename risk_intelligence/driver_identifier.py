"""
Driver Identifier — Random Forest-compatible risk driver identification.

The model is a Random Forest, so this module does NOT use logistic
regression coefficients (argmax |w_j * S_j|), as the original technical
design document sketched for a different model type. Instead:

  1. Reads the Random Forest's real, trained `feature_importances_`
     (never invented) - a global measure of how much each transformed
     feature reduces impurity across the forest.
  2. Aggregates those importances back onto the ORIGINAL raw dataset
     columns (a categorical column one-hot-encoded into several transformed
     columns has its importance summed back together), using
     `preprocessing.get_feature_names_out()` (the fitted ColumnTransformer
     supports this).
  3. Combines each raw column's global importance with how unusual /
     salient the CURRENT customer's value is for that column, computed
     from the real reference dataset (data/customer_churn.csv) - not
     invented statistics.
  4. Ranks raw columns by (importance * salience) and returns the top N.

This produces a transparent, reproducible ranking grounded entirely in
the real trained model and the real dataset. It explicitly does NOT claim
causality - see the phrasing in `explanation` below ("important model
feature", never "cause of churn").

Known multicollinearity in this dataset (see docs/dataset-mapping.md):
several feature groups are closely related by construction (e.g.
num_claims_12m = num_approved + num_rejected + num_pending exactly;
total_payout_amount_12m = total_claim_amount_12m * payout_ratio_12m
exactly; premium_to_coverage_ratio is an exact function of current_premium
and coverage_amount). When one of these ranks as a top driver, the
explanation attaches a brief caveat so an Administrator doesn't read three
near-redundant columns as three independent signals.
"""

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

# Brief caveats for feature groups known (from real analysis of the
# dataset - see docs/dataset-mapping.md) to be closely/exactly related to
# other features in the schema. Attached to the explanation text when the
# feature ranks as a top driver, so it's never presented as an
# independent signal without context.
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
    """Real reference distribution, loaded once, used only to judge how
    unusual a customer's numeric/categorical value is (never used to
    invent model metrics)."""
    return pd.read_csv(config.CUSTOMER_DATA_CSV)


def _numeric_salience(column: str, value: float) -> float:
    """0 (typical/median value) to 1 (extreme value) based on the real
    reference distribution's percentile rank."""
    df = _reference_dataframe()
    if column not in df.columns:
        return 0.5
    series = df[column].dropna()
    if series.empty:
        return 0.5
    percentile = (series < value).mean()
    return abs(percentile - 0.5) * 2


def _categorical_salience(column: str, value: str) -> float:
    """How much this category's empirical churn rate deviates from the
    dataset's overall churn rate, normalized to roughly [0, 1]."""
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
    """
    Maps the fitted preprocessing's transformed feature names back onto the
    raw dataset columns in FEATURE_COLUMNS, summing feature_importances_
    for every transformed feature that derives from each raw column.

    Falls back to treating feature_importances_ as already aligned 1:1
    with FEATURE_COLUMNS if the preprocessing object exposes no
    `get_feature_names_out` (e.g. a bespoke transformer).
    """
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
            # Prefer the longest matching raw column name to avoid a short
            # column name (e.g. "age") false-matching inside a longer
            # transformed name that happens to contain it as a substring.
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
