"""
Effect Estimator.

There is no real intervention-outcome dataset for this MVP, so this
component returns FIXED, DOCUMENTED assumption values
(config.ASSUMED_ACTION_EFFECT) describing how much each action is assumed
to reduce a customer's churn probability.

These values are explicitly "MVP simulation assumptions" - NOT empirical
facts, NOT derived from any A/B test or historical intervention log. The
interface is structured so this class can later be swapped for a learned
uplift/treatment-effect model (e.g. a T-learner or causal forest) without
changing any caller (recommendation_engine/expected_value_ranker.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from recommendation_engine.action_lookup import CandidateAction

import config


@dataclass
class EstimatedEffect:
    action_id: str
    assumed_risk_reduction: float  # absolute reduction in churn probability, e.g. 0.05 = 5 points
    basis: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "assumed_risk_reduction": round(self.assumed_risk_reduction, 4),
            "basis": self.basis,
        }


class EffectEstimator:
    def __init__(self, assumed_effects: Dict[str, float] = None):
        self._assumed_effects = assumed_effects or config.ASSUMED_ACTION_EFFECT

    def estimate(self, action: CandidateAction) -> EstimatedEffect:
        assumed_reduction = self._assumed_effects.get(action.action_id, 0.02)
        return EstimatedEffect(
            action_id=action.action_id,
            assumed_risk_reduction=assumed_reduction,
            basis=(
                "MVP simulation assumption (config.ASSUMED_ACTION_EFFECT) - not "
                "derived from an intervention-outcome dataset. Replace with a "
                "learned uplift model once real intervention data is available."
            ),
        )


effect_estimator = EffectEstimator()
