

"""Translate ranked risk drivers into deduplicated candidate actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from risk_intelligence.driver_identifier import Driver

import config


@dataclass
class CandidateAction:
    """Action selected from a driver-to-action rule."""

    action_id: str
    label: str
    description: str
    driver_feature: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "label": self.label,
            "description": self.description,
            "driver_feature": self.driver_feature,
        }


class ActionLookup:
    """Apply the configured action rules to ranked drivers."""

    def __init__(self, rules: Dict[str, Dict[str, str]] = None):
        self._rules = rules or config.ACTION_RULES

    def actions_for_drivers(self, drivers: List[Driver]) -> List[CandidateAction]:
        """One candidate action per (deduplicated) driver-derived rule."""
        seen_action_ids = set()
        candidates: List[CandidateAction] = []

        for driver in drivers:
            rule = self._rules.get(driver.feature, self._rules["default"])
            if rule["action"] in seen_action_ids:
                continue
            seen_action_ids.add(rule["action"])
            candidates.append(
                CandidateAction(
                    action_id=rule["action"],
                    label=rule["label"],
                    description=rule["description"],
                    driver_feature=driver.feature,
                )
            )

        if not candidates:
            default_rule = self._rules["default"]
            candidates.append(
                CandidateAction(
                    action_id=default_rule["action"],
                    label=default_rule["label"],
                    description=default_rule["description"],
                    driver_feature="none",
                )
            )
        return candidates


action_lookup = ActionLookup()
