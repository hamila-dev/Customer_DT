"""
Recommendation Engine.

Architecture (see docs/recommendation-engine.md):

    Risk Intelligence
           |
           v
    Driver Identifier
           |
           v
    Action Lookup
           |
           v
    Effect Estimator
           |
           v
    Expected Value Ranker
           |
           v
    Recommended Action

This module is the thin orchestrator that calls the four components in
order and returns the ranked list plus the top recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from recommendation_engine.action_lookup import ActionLookup, action_lookup
from recommendation_engine.expected_value_ranker import ExpectedValueRanker, RankedAction, expected_value_ranker
from risk_intelligence.driver_identifier import Driver, DriverIdentifier, driver_identifier
from risk_intelligence.predictor import ChurnPredictor, RiskResult, churn_predictor
from twin_engine.state.twin_state import TwinState

import config


@dataclass
class RecommendationResult:
    customer_id: str
    risk: RiskResult
    top_drivers: List[Driver]
    ranked_actions: List[RankedAction]

    @property
    def top_recommendation(self) -> RankedAction:
        return self.ranked_actions[0]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "customer_id": self.customer_id,
            "risk": self.risk.to_dict(),
            "top_drivers": [d.to_dict() for d in self.top_drivers],
            "ranked_actions": [a.to_dict() for a in self.ranked_actions],
            "recommended_action": self.top_recommendation.to_dict(),
        }


class RecommendationEngine:
    def __init__(
        self,
        predictor: ChurnPredictor = churn_predictor,
        identifier: DriverIdentifier = driver_identifier,
        lookup: ActionLookup = action_lookup,
        ranker: ExpectedValueRanker = expected_value_ranker,
    ):
        self._predictor = predictor
        self._identifier = identifier
        self._lookup = lookup
        self._ranker = ranker

    def recommend(self, state: TwinState, customer_value: float = config.DEFAULT_CUSTOMER_VALUE) -> RecommendationResult:
        risk = self._predictor.predict(state)
        drivers = self._identifier.identify(state)
        candidate_actions = self._lookup.actions_for_drivers(drivers)
        ranked_actions = self._ranker.rank(
            candidate_actions,
            churn_probability=risk.churn_probability,
            customer_value=customer_value,
        )
        return RecommendationResult(
            customer_id=state.customer_id,
            risk=risk,
            top_drivers=drivers,
            ranked_actions=ranked_actions,
        )


recommendation_engine = RecommendationEngine()
