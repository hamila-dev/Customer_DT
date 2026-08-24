"""Pydantic schemas for API requests/responses. Handlers stay thin; these
just describe the shapes going over the wire."""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class EventIn(BaseModel):
    customer_id: str
    event_type: str = Field(..., description="One of the EventType values, e.g. 'claim_created'")
    payload: Dict[str, Any] = Field(default_factory=dict)
    source: str = "manual"


class SimulateIn(BaseModel):
    scenario: str = Field(..., description="Scenario/event type name, e.g. 'premium_adjusted'")
    parameters: Dict[str, Any] = Field(default_factory=dict)


class MonteCarloIn(BaseModel):
    scenario: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    trials: Optional[int] = None
    numeric_noise_std: Optional[float] = None


class EventGeneratorConfigIn(BaseModel):
    interval_seconds: Optional[float] = None
    scenarios: Optional[list] = None
