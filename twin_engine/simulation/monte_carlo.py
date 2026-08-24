"""
Monte Carlo Engine.

The Digital Twin Engine is the primary focus of this project, and Monte
Carlo simulation is a first-class part of it: rather than a single
simulated churn probability, this engine produces an outcome DISTRIBUTION
by running the scenario-transformed Twin state through the Random Forest
many times under documented, configurable uncertainty assumptions.

CRITICAL SEPARATION (see docs/simulation.md):
  1. ML prediction - the Random Forest's churn_probability for one exact
     feature vector. This is a real, deterministic model output.
  2. Simulation uncertainty - for the MVP, uncertainty is introduced by
     perturbing the scenario's NUMERIC parameters with configurable
     Gaussian noise (config.MONTE_CARLO_NUMERIC_NOISE_STD) before applying
     them, then re-running the Random Forest on each perturbed clone.

The Random Forest itself does not define a future-state probability
distribution - the distribution here comes entirely from the documented
perturbation assumption in (2), which is clearly labelled as a simulation
assumption, not a scientific/statistical fact about the customer.

The real Twin is never modified: every trial clones the real state via
ScenarioTransformer before touching it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from risk_intelligence.predictor import ChurnPredictor, churn_predictor
from twin_engine.simulation.scenario_transformer import Scenario, ScenarioTransformer, scenario_transformer
from twin_engine.state.twin_state import TwinState

import config

# Scenario parameters treated as "numeric" for perturbation purposes -
# the monetary/percentage payload fields used by the dataset-grounded
# events in twin_engine/events/transition_handler.py.
NUMERIC_SCENARIO_PARAMETERS = {
    "claim_amount",
    "current_premium",
    "change_pct",
    "coverage_amount",
    "reduction_pct",
}


@dataclass
class MonteCarloResult:
    customer_id: str
    scenario_name: str
    trials: int
    mean: float
    median: float
    p10: float
    p90: float
    std_dev: float
    distribution: List[float]
    assumptions: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "customer_id": self.customer_id,
            "scenario_name": self.scenario_name,
            "trials": self.trials,
            "mean_churn_probability": round(self.mean, 4),
            "median_churn_probability": round(self.median, 4),
            "p10_churn_probability": round(self.p10, 4),
            "p90_churn_probability": round(self.p90, 4),
            "std_dev": round(self.std_dev, 4),
            # Distribution is returned rounded and (for large N) downsampled
            # for a reasonably sized JSON payload / histogram rendering.
            "distribution_sample": [round(v, 4) for v in self.distribution],
            "assumptions": self.assumptions,
        }


class MonteCarloEngine:
    def __init__(
        self,
        transformer: ScenarioTransformer = scenario_transformer,
        predictor: ChurnPredictor = churn_predictor,
    ):
        self._transformer = transformer
        self._predictor = predictor

    def _perturb_parameters(self, parameters: Dict[str, Any], noise_std: float, rng: random.Random) -> Dict[str, Any]:
        perturbed = dict(parameters)
        for key, value in parameters.items():
            if key in NUMERIC_SCENARIO_PARAMETERS and isinstance(value, (int, float)):
                noise_factor = rng.gauss(1.0, noise_std)
                perturbed_value = value * noise_factor
                # Only floor genuinely non-negative monetary quantities at 0.
                # "change_pct" is a signed percentage (a premium DECREASE is
                # legitimate and must stay negative) so it is perturbed but
                # never clamped.
                if key in ("claim_amount", "current_premium", "coverage_amount"):
                    perturbed_value = max(0.0, perturbed_value)
                perturbed[key] = perturbed_value
        return perturbed

    def run(
        self,
        state: TwinState,
        scenario: Scenario,
        trials: int = config.MONTE_CARLO_DEFAULT_TRIALS,
        numeric_noise_std: float = config.MONTE_CARLO_NUMERIC_NOISE_STD,
        max_distribution_points: int = 200,
        random_seed: int = None,
    ) -> MonteCarloResult:
        rng = random.Random(random_seed)
        simulated_states: List[TwinState] = []

        for _ in range(trials):
            perturbed_params = self._perturb_parameters(scenario.parameters, numeric_noise_std, rng)
            perturbed_scenario = Scenario(
                name=scenario.name,
                event_type=scenario.event_type,
                parameters=perturbed_params,
            )
            # transformer clones `state` internally - the real Twin is untouched.
            simulated_states.append(self._transformer.transform(state, perturbed_scenario))

        probabilities = self._predictor.predict_batch(simulated_states)
        arr = np.array(probabilities, dtype=float)

        # Downsample the returned distribution sample for payload size, while
        # keeping full-precision summary statistics computed over all trials.
        if len(arr) > max_distribution_points:
            sample_idx = np.linspace(0, len(arr) - 1, max_distribution_points).astype(int)
            distribution_sample = arr[sample_idx].tolist()
        else:
            distribution_sample = arr.tolist()

        return MonteCarloResult(
            customer_id=state.customer_id,
            scenario_name=scenario.name,
            trials=trials,
            mean=float(np.mean(arr)),
            median=float(np.median(arr)),
            p10=float(np.percentile(arr, 10)),
            p90=float(np.percentile(arr, 90)),
            std_dev=float(np.std(arr)),
            distribution=distribution_sample,
            assumptions={
                "source_of_stochasticity": (
                    "Numeric scenario parameters (e.g. claim_amount, current_premium, "
                    "change_pct, coverage_amount, reduction_pct) are "
                    "perturbed per trial with multiplicative Gaussian noise "
                    f"(mean=1.0, std={numeric_noise_std}) before being applied to a "
                    "cloned Twin state and re-scored by the Random Forest. This "
                    "represents uncertainty in how the hypothetical scenario plays "
                    "out, NOT statistical uncertainty in the model itself."
                ),
                "numeric_noise_std": numeric_noise_std,
                "note": "These are MVP simulation assumptions, not empirically fitted uncertainty parameters.",
            },
        )


monte_carlo_engine = MonteCarloEngine()
