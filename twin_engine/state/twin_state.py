from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TwinEventRecord:
    event_id: str
    event_type: str
    occurred_at: str
    payload: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TwinState:
    """Current state of one policyholder's Digital Twin."""

    # Static profile fields are not changed by event transitions.
    customer_id: str
    age: int
    region_name: str
    marital_status: str
    customer_tenure_months: int
    multi_policy_flag: int
    num_policies: int
    policy_type: str
    renewal_month: int
    payment_frequency: str
    autopay_enabled: int

    # Trailing-period and policy values changed by event transitions.
    current_premium: float
    premium_last_year: float
    num_price_increases_last_3y: int
    coverage_amount: float
    late_payment_count_12m: int
    num_claims_12m: int
    num_approved_claims_12m: int
    num_rejected_claims_12m: int
    num_pending_claims_12m: int
    total_claim_amount_12m: float
    total_payout_amount_12m: float
    avg_claim_amount: float
    avg_settlement_time_days: int
    days_since_last_claim: int
    num_contacts_12m: int
    complaint_flag: int
    complaint_resolution_days: int
    quote_requested_flag: int
    coverage_downgrade_flag: int

    # Fed to the model but not changed by the current event model.
    payment_method_change_flag: int = 0

    # Engine metadata, not raw dataset columns.
    version: int = 0
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    event_history: List[TwinEventRecord] = field(default_factory=list)

    # Reference label; never included in the model feature vector.
    historical_churn_label: Optional[int] = None

    # Derived properties are recomputed from their source fields.
    @property
    def premium_change_pct(self) -> float:
        """Compute the premium change used by the simulated feature vector.

        The source dataset contains independent noise in this column, but
        simulation needs the feature to stay consistent with its two inputs.
        """
        if self.premium_last_year == 0:
            return 0.0
        return (self.current_premium - self.premium_last_year) / self.premium_last_year

    @property
    def premium_to_coverage_ratio(self) -> float:
        """Return the premium-to-coverage ratio used by the model."""
        if self.coverage_amount == 0:
            return 0.0
        return self.current_premium / self.coverage_amount

    @property
    def payout_ratio_12m(self) -> float:
        """Return the payout-to-claim ratio, using the dataset's no-claim baseline."""
        if self.total_claim_amount_12m == 0:
            # The source data uses a 0.75-0.85 baseline when no claim exists;
            # 0.75 avoids treating no claims as no payouts.
            return 0.75
        return self.total_payout_amount_12m / self.total_claim_amount_12m

    @property
    def missed_payment_flag(self) -> int:
        """Apply the dataset rule: four or more late payments means missed payment."""
        return 1 if self.late_payment_count_12m >= 4 else 0

    def clone(self) -> "TwinState":
        """Return an independent copy for what-if simulation."""
        return copy.deepcopy(self)

    def record_event(self, event_id: str, event_type: str, payload: Dict[str, Any], description: str = "") -> None:
        """Append an event record, retaining only the latest 50 entries."""
        self.event_history.append(
            TwinEventRecord(event_id=event_id, event_type=event_type, occurred_at=_now_iso(), payload=payload, description=description)
        )
        if len(self.event_history) > 50:
            self.event_history = self.event_history[-50:]

    def touch(self) -> None:
        self.version += 1
        self.updated_at = _now_iso()

    def to_feature_dict(self) -> Dict[str, Any]:
        """Build the feature vector required by the trained preprocessing pipeline."""
        return {
            "age": self.age,
            "customer_tenure_months": self.customer_tenure_months,
            "multi_policy_flag": self.multi_policy_flag,
            "num_policies": self.num_policies,
            "renewal_month": self.renewal_month,
            "current_premium": self.current_premium,
            "premium_last_year": self.premium_last_year,
            "premium_change_pct": self.premium_change_pct,
            "num_price_increases_last_3y": self.num_price_increases_last_3y,
            "coverage_amount": self.coverage_amount,
            "premium_to_coverage_ratio": self.premium_to_coverage_ratio,
            "autopay_enabled": self.autopay_enabled,
            "late_payment_count_12m": self.late_payment_count_12m,
            "missed_payment_flag": self.missed_payment_flag,
            "payment_method_change_flag": self.payment_method_change_flag,
            "num_claims_12m": self.num_claims_12m,
            "num_approved_claims_12m": self.num_approved_claims_12m,
            "num_rejected_claims_12m": self.num_rejected_claims_12m,
            "num_pending_claims_12m": self.num_pending_claims_12m,
            "avg_claim_amount": self.avg_claim_amount,
            "total_claim_amount_12m": self.total_claim_amount_12m,
            "total_payout_amount_12m": self.total_payout_amount_12m,
            "payout_ratio_12m": self.payout_ratio_12m,
            "avg_settlement_time_days": self.avg_settlement_time_days,
            "days_since_last_claim": self.days_since_last_claim,
            "num_contacts_12m": self.num_contacts_12m,
            "complaint_flag": self.complaint_flag,
            "complaint_resolution_days": self.complaint_resolution_days,
            "quote_requested_flag": self.quote_requested_flag,
            "coverage_downgrade_flag": self.coverage_downgrade_flag,
            "region_name": self.region_name,
            "marital_status": self.marital_status,
            "policy_type": self.policy_type,
            "payment_frequency": self.payment_frequency,
        }

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["premium_change_pct"] = self.premium_change_pct
        d["premium_to_coverage_ratio"] = self.premium_to_coverage_ratio
        d["payout_ratio_12m"] = self.payout_ratio_12m
        d["missed_payment_flag"] = self.missed_payment_flag
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TwinState":
        data = dict(data)
        for derived_field in ("premium_change_pct", "premium_to_coverage_ratio", "payout_ratio_12m", "missed_payment_flag"):
            data.pop(derived_field, None)
        history = data.pop("event_history", [])
        state = cls(**data)
        state.event_history = [TwinEventRecord(**h) if isinstance(h, dict) else h for h in history]
        return state
