"""Rank candidate retention actions by assumed expected value."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from recommendation_engine.action_lookup import CandidateAction
from recommendation_engine.effect_estimator import EffectEstimator, effect_estimator

import config


@dataclass
class RankedAction:
    """Candidate action with its risk, value, cost, and ranking inputs."""

    action: CandidateAction
    churn_probability: float
    assumed_risk_reduction: float
    customer_value: float
    action_cost: float
    expected_value: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.action.to_dict(),
            "expected_value": round(self.expected_value, 2),
            "inputs": {
                "churn_probability": round(self.churn_probability, 4),
                "assumed_risk_reduction": round(self.assumed_risk_reduction, 4),
                "customer_value_assumption": self.customer_value,
                "action_cost_assumption": self.action_cost,
            },
        }


class ExpectedValueRanker:
    """Rank actions using assumed benefit minus action cost."""

    def __init__(self, estimator: EffectEstimator = effect_estimator):
        self._estimator = estimator

    def rank(
        self,
        candidate_actions: List[CandidateAction],
        churn_probability: float,
        customer_value: float = config.DEFAULT_CUSTOMER_VALUE,
    ) -> List[RankedAction]:
        """Use churn probability * assumed reduction * value, less cost."""
        ranked: List[RankedAction] = []

        for action in candidate_actions:
            effect = self._estimator.estimate(action)
            cost = config.ASSUMED_ACTION_COST.get(action.action_id, 5.0)

            expected_value = (
                churn_probability * effect.assumed_risk_reduction * customer_value
            ) - cost

            ranked.append(
                RankedAction(
                    action=action,
                    churn_probability=churn_probability,
                    assumed_risk_reduction=effect.assumed_risk_reduction,
                    customer_value=customer_value,
                    action_cost=cost,
                    expected_value=expected_value,
                )
            )

        ranked.sort(key=lambda r: r.expected_value, reverse=True)
        return ranked


expected_value_ranker = ExpectedValueRanker()
