"""Load the shipped churn model and expose single- and batch-prediction APIs."""

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
    """Raised when prediction is requested before model artifacts are available."""


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
    """Load model artifacts when present and fail prediction explicitly when absent."""

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

        # Keep joblib optional for API features that do not need prediction.
        import joblib

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
        """Return the explicit metadata version or a stable metadata-derived label."""
        if "model_version" in self.metadata:
            return self.metadata["model_version"]
        model_type = self.metadata.get("model_type", "model")
        trained_at = self.metadata.get("training_timestamp")
        return f"{model_type}@{trained_at}" if trained_at else model_type

    @property
    def evaluation_metrics(self) -> Dict[str, Any]:
        """Return held-out metrics recorded with the shipped model."""
        return self.metadata.get("evaluation_metrics", {})

    def _require_model(self) -> None:
        if not self.is_available:
            raise ModelNotAvailableError(self.MISSING_ARTIFACT_MESSAGE)

    def _predict_proba_for_frame(self, feature_frame: pd.DataFrame):
        self._require_model()
        transformed = self.preprocessing.transform(feature_frame)
        # Metadata selects the positive-class column; binary classifiers conventionally use 1.
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
        """Score many states in one preprocessing/model call."""
        from risk_intelligence.feature_mapper import twin_states_to_feature_frame

        states = list(states)
        if not states:
            return []
        feature_frame = twin_states_to_feature_frame(states)
        probabilities = self._predict_proba_for_frame(feature_frame)
        return [float(p) for p in probabilities]


# Reload can pick up artifacts added after process startup.
churn_predictor = ChurnPredictor()
