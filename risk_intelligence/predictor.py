"""
Risk Intelligence — Random Forest churn predictor.

This module NEVER trains a model and NEVER invents a prediction. It only
loads pre-trained artifacts (produced by you, separately - see
model/README.md) and, if present, uses them to:

  1. Receive a Twin state.
  2. Convert it into the model feature vector (risk_intelligence.feature_mapper).
  3. Apply the saved preprocessing.
  4. Run the Random Forest.
  5. Return churn probability.
  6. Convert probability into a configurable risk level (config.py).

If the artifacts are missing, every public method raises
ModelNotAvailableError with a clear, actionable message instead of
fabricating a result.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from risk_intelligence.feature_mapper import FeatureMappingError, twin_state_to_feature_row
from twin_engine.state.twin_state import TwinState

import config

logger = logging.getLogger(__name__)


class ModelNotAvailableError(RuntimeError):
    """Raised when a request needs the trained model but artifacts are missing."""


@dataclass
class RiskResult:
    customer_id: str
    churn_probability: float
    risk_level: str
    model_version: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "customer_id": self.customer_id,
            "churn_probability": round(self.churn_probability, 4),
            "risk_level": self.risk_level,
            "model_version": self.model_version,
        }


class ChurnPredictor:
    """
    Loads model/churn_model.joblib, model/preprocessing.joblib, and
    model/model_metadata.json if present, and exposes a `predict` method.

    Safe to construct even when artifacts are missing - `is_available`
    will be False and `predict` will raise ModelNotAvailableError with
    instructions, rather than crashing the whole application at import
    time. This lets the rest of the API run (customers list, Twin views,
    events) even before you've dropped in a trained model.
    """

    MISSING_ARTIFACT_MESSAGE = (
        "Trained model artifacts were not found. This endpoint requires exactly:\n"
        f"  - {config.MODEL_PATH}\n"
        f"  - {config.PREPROCESSING_PATH}\n"
        f"  - {config.MODEL_METADATA_PATH}\n"
        f"  - {config.FEATURE_SCHEMA_PATH}\n"
        "These are consumed exactly as provided - the application performs "
        "INFERENCE ONLY and never trains or retrains a model itself. "
        "See docs/dataset-mapping.md and model/README.md for details."
    )

    def __init__(
        self,
        model_path: Path = config.MODEL_PATH,
        preprocessing_path: Path = config.PREPROCESSING_PATH,
        metadata_path: Path = config.MODEL_METADATA_PATH,
        feature_schema_path: Path = config.FEATURE_SCHEMA_PATH,
    ):
        self._model_path = model_path
        self._preprocessing_path = preprocessing_path
        self._metadata_path = metadata_path
        self._feature_schema_path = feature_schema_path

        self.model = None
        self.preprocessing = None
        self.metadata: Dict[str, Any] = {}
        self.feature_schema: Dict[str, Any] = {}

        self._try_load()

    def _try_load(self) -> None:
        if not (self._model_path.exists() and self._preprocessing_path.exists()):
            logger.warning("Model artifacts not found yet. %s", self.MISSING_ARTIFACT_MESSAGE)
            return

        import joblib  # local import: keeps joblib optional if never used

        try:
            self.model = joblib.load(self._model_path)
            self.preprocessing = joblib.load(self._preprocessing_path)
        except Exception:
            logger.exception("Failed to load model artifacts from %s / %s", self._model_path, self._preprocessing_path)
            self.model = None
            self.preprocessing = None
            return

        if self._metadata_path.exists():
            try:
                self.metadata = json.loads(self._metadata_path.read_text())
            except (json.JSONDecodeError, OSError):
                logger.exception("Failed to read model metadata at %s", self._metadata_path)
                self.metadata = {}
        else:
            logger.warning("model_metadata.json not found at %s - proceeding without it.", self._metadata_path)

        if self._feature_schema_path.exists():
            try:
                self.feature_schema = json.loads(self._feature_schema_path.read_text())
            except (json.JSONDecodeError, OSError):
                logger.exception("Failed to read feature schema at %s", self._feature_schema_path)
                self.feature_schema = {}
        else:
            logger.warning("feature_schema.json not found at %s - proceeding without it.", self._feature_schema_path)

    def reload(self) -> None:
        """Re-attempt loading artifacts (e.g. after you drop them in without restarting)."""
        self._try_load()

    @property
    def is_available(self) -> bool:
        return self.model is not None and self.preprocessing is not None

    @property
    def model_version(self) -> str:
        """
        The shipped model_metadata.json (see model/README.md for the
        expected shape) may or may not include an explicit "model_version"
        string. If it doesn't, we build a readable identifier from
        whatever real metadata fields ARE present (model type + training
        timestamp) rather than reporting a meaningless "unversioned".
        """
        if "model_version" in self.metadata:
            return self.metadata["model_version"]
        model_type = self.metadata.get("model_type", "model")
        trained_at = self.metadata.get("training_timestamp")
        return f"{model_type}@{trained_at}" if trained_at else model_type

    @property
    def evaluation_metrics(self) -> Dict[str, Any]:
        """Real, held-out evaluation metrics as recorded in model_metadata.json
        at training time - never computed or invented here."""
        return self.metadata.get("evaluation_metrics", {})

    def _require_model(self) -> None:
        if not self.is_available:
            raise ModelNotAvailableError(self.MISSING_ARTIFACT_MESSAGE)

    def _predict_proba_for_frame(self, feature_frame: pd.DataFrame):
        self._require_model()
        transformed = self.preprocessing.transform(feature_frame)
        # Standard scikit-learn binary classifier convention: column 1 = positive class.
        # model_metadata.json should record which class index/label corresponds to "churn".
        proba = self.model.predict_proba(transformed)
        churn_class_index = self.metadata.get("churn_class_index", 1)
        return proba[:, churn_class_index]

    def predict(self, state: TwinState) -> RiskResult:
        feature_row = twin_state_to_feature_row(state)
        churn_probability = float(self._predict_proba_for_frame(feature_row)[0])
        risk_level = config.risk_level_from_probability(churn_probability)
        return RiskResult(
            customer_id=state.customer_id,
            churn_probability=churn_probability,
            risk_level=risk_level,
            model_version=self.model_version,
        )

    def predict_batch(self, states) -> list:
        """Used by the Monte Carlo engine - one preprocessing/model call for N cloned states."""
        from risk_intelligence.feature_mapper import twin_states_to_feature_frame

        states = list(states)
        if not states:
            return []
        feature_frame = twin_states_to_feature_frame(states)
        probabilities = self._predict_proba_for_frame(feature_frame)
        return [float(p) for p in probabilities]


# Module-level singleton. Constructed once; if artifacts are absent at
# import time, `reload()` can pick them up later without restarting the app.
churn_predictor = ChurnPredictor()
