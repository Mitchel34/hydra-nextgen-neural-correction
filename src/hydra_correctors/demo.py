"""Deterministic synthetic demonstration; no optimizer or physical simulator."""

import json
import math

from .kernels import Model, Scaler, dumps, loads, zero_state
from .evaluation import ForecastKey, Observation, issue_available_value, score_paired


def run_demo():
    # Analytic coefficients demonstrate arithmetic only. They are never fitted.
    scaler = Scaler((2e-8, 2.0, 0.4), (1e-7, 3.0, 0.5))
    weights = tuple(0.08 * math.sin(i * 0.17) for i in range(3489))
    model = loads(dumps(Model("gru32", weights, scaler)))
    origin = zero_state()
    dry = model.forward((2e-8, 0.0, 0.3), origin)
    wet = model.forward((2e-8, 12.0, 0.3), origin)
    repeated = model.forward((2e-8, 0.0, 0.3), origin)
    assert dry == repeated and origin == zero_state()

    # Availability is declared metadata, not a verified publication vintage.
    row = Observation("synthetic-gauge", 3600, 7200, 10.0, True, "synthetic-v1")
    accepted = issue_available_value(row, 7200)
    try:
        issue_available_value(row, 3600)
    except ValueError:
        future_rejected = True
    else:
        raise AssertionError("Future observation was accepted")

    # Values are in m^3/s. No neural skill or physical response is being scored.
    keys = tuple(ForecastKey("synthetic-gauge", 0, lead) for lead in (1, 2, 3))
    observations = dict(zip(keys, (10.0, 12.0, 14.0)))
    illustrative_predictions = dict(zip(keys, (11.0, 13.0, 15.0)))
    score = score_paired(keys, observations, illustrative_predictions, {"synthetic-gauge": 2.0})
    return {
        "data": "synthetic; analytic untrained coefficients; no hydrologic simulation",
        "scientific_skill_claim": False,
        "model_id": model.id,
        "internal_kernel": {
            "architecture": "gru32",
            "parameters": len(weights),
            "dry_factor": dry.factor,
            "wet_factor": wet.factor,
            "factor_range": [0.5, 2.0],
            "repeat_is_exact": dry == repeated,
            "origin_unchanged": origin == zero_state(),
            "state_is_candidate_only": True,
        },
        "observation_contract": {
            "accepted_age_hours": accepted["age_hours"],
            "future_value_rejected": future_rejected,
        },
        "illustrative_fixed_support_score": score,
    }


def main():
    print(json.dumps(run_demo(), indent=2, sort_keys=True, allow_nan=False))
    return 0
