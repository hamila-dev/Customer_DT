

"""Run stochastic what-if scenarios against isolated Twin-state clones."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from risk_intelligence.predictor import ChurnPredictor, churn_predictor
from twin_engine.simulation.scenario_transformer import Scenario, ScenarioTransformer, scenario_transformer
from twin_engine.state.twin_state import TwinState

import config

# Only numeric payloads with meaningful scenario uncertainty are perturbed.
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
            # Keep the API payload bounded while summary statistics use all trials.
            "distribution_sample": [round(v, 4) for v in self.distribution],
            "assumptions": self.assumptions,
        }


class MonteCarloEngine:
    """Transform and score independent scenario clones in one batch."""

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
                # A percentage change may be negative; monetary amounts and
                # coverage cannot be below zero.
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
            simulated_states.append(self._transformer.transform(state, perturbed_scenario))

        probabilities = self._predictor.predict_batch(simulated_states)
        arr = np.array(probabilities, dtype=float)

        # Preserve full statistics while limiting the histogram payload.
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
