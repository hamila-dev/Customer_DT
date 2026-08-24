"""
Twin State — the customer's current virtual state (S_t).

Fields here map directly onto data/customer_churn.csv (the insurance
policyholder churn dataset) and model/feature_schema.json /
model/model_metadata.json (the trained Random Forest's exact input
contract). See docs/dataset-mapping.md for the full mapping and the
Static / Dynamic / ML-only / Event-driven / Simulation-parameter
classification of every field.

Field categories (see docs/dataset-mapping.md for the full table):

  STATIC (identity/profile, never mutated by any event in this MVP):
    customer_id, age, region_name, marital_status, customer_tenure_months,
    multi_policy_flag, num_policies, policy_type, renewal_month,
    payment_frequency, autopay_enabled

  DYNAMIC (mutated by events):
    current_premium, premium_last_year, num_price_increases_last_3y,
    coverage_amount, late_payment_count_12m, num_claims_12m,
    num_approved_claims_12m, num_rejected_claims_12m,
    num_pending_claims_12m, total_claim_amount_12m,
    total_payout_amount_12m, avg_claim_amount, avg_settlement_time_days,
    days_since_last_claim, num_contacts_12m, complaint_flag,
    complaint_resolution_days, quote_requested_flag,
    coverage_downgrade_flag

  ML-ONLY (fed to the model, present in Twin state, but no dataset-grounded
  event mutates them in this MVP - they start at their bootstrap value and
  stay there until a future event is added):
    payment_method_change_flag

  DERIVED (recomputed properties, never stored independently, so they can
  never silently drift out of sync with their inputs):
    premium_change_pct, premium_to_coverage_ratio, payout_ratio_12m,
    missed_payment_flag

Excluded entirely from the Twin (per model/model_metadata.json's
`excluded_features`, or not needed for the MVP demo): as_of_date,
age_band (a strict bucketing of age), churn_type, churn_probability_true.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TwinEventRecord:
    """A single entry in the Twin's event history."""

    event_id: str
    event_type: str
    occurred_at: str
    payload: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TwinState:
    """Represents S_t - the current state of one policyholder's Digital Twin."""

    # ---------------- Identity / static profile (never mutated by events) ----------------
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

    # ---------------- Dynamic (mutated by events) ----------------
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

    # ---------------- ML-only (no dataset-grounded event mutates this in the MVP) ----------------
    payment_method_change_flag: int = 0

    # ---------------- Twin-engine bookkeeping (not raw dataset columns) ----------------
    version: int = 0
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    event_history: List[TwinEventRecord] = field(default_factory=list)

    # ---------------- Ground truth label, reference only, never an ML feature ----------------
    historical_churn_label: Optional[int] = None

    # ------------------------------------------------------------------
    # Derived properties - recomputed, never stored independently
    # ------------------------------------------------------------------
    @property
    def premium_change_pct(self) -> float:
        """(current_premium - premium_last_year) / premium_last_year.

        NOTE: in the source training data this column carries some
        independent noise beyond this exact formula (observed deviation up
        to ~0.22 in a sample check). For Twin simulation purposes we
        recompute it exactly from current_premium/premium_last_year so a
        simulated premium change always produces an internally consistent
        feature vector - this is a documented MVP modeling choice, not a
        claim that it reproduces the original data-generating noise.
        """
        if self.premium_last_year == 0:
            return 0.0
        return (self.current_premium - self.premium_last_year) / self.premium_last_year

    @property
    def premium_to_coverage_ratio(self) -> float:
        """current_premium / coverage_amount - an exact match to how this
        column is derived in the source dataset (verified during
        integration, floating-point-rounding aside)."""
        if self.coverage_amount == 0:
            return 0.0
        return self.current_premium / self.coverage_amount

    @property
    def payout_ratio_12m(self) -> float:
        """total_payout_amount_12m / total_claim_amount_12m.

        ASSUMPTION: when total_claim_amount_12m is 0 (no claims filed),
        the source dataset still carries a baseline value in roughly the
        0.75-0.85 range rather than an undefined 0/0. We use a documented
        MVP default of 0.75 in that case rather than 0.0, to avoid
        implying "no claims" is equivalent to "claims are never paid out."
        """
        if self.total_claim_amount_12m == 0:
            return 0.75
        return self.total_payout_amount_12m / self.total_claim_amount_12m

    @property
    def missed_payment_flag(self) -> int:
        """1 if late_payment_count_12m >= 4, else 0 - matches the exact
        rule documented in the source data dictionary
        ("1 if missed payments flag (>=4 late payments), else 0"),
        verified against the real dataset during integration."""
        return 1 if self.late_payment_count_12m >= 4 else 0

    # ------------------------------------------------------------------
    def clone(self) -> "TwinState":
        """Deep, fully independent copy. Used by the Scenario Transformer
        so what-if simulations can never mutate the real Twin state."""
        return copy.deepcopy(self)

    def record_event(self, event_id: str, event_type: str, payload: Dict[str, Any], description: str = "") -> None:
        self.event_history.append(
            TwinEventRecord(event_id=event_id, event_type=event_type, occurred_at=_now_iso(), payload=payload, description=description)
        )
        if len(self.event_history) > 50:
            self.event_history = self.event_history[-50:]

    def touch(self) -> None:
        self.version += 1
        self.updated_at = _now_iso()

    def to_feature_dict(self) -> Dict[str, Any]:
        """
        The full feature vector fed to preprocessing.joblib. Field names
        and set intentionally match model/feature_schema.json and
        model/model_metadata.json's feature_columns exactly - see
        risk_intelligence/feature_mapper.py.
        """
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
