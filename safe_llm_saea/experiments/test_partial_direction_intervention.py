import numpy as np
import pytest

from safe_llm_saea.experiments import run_partial_direction_intervention_development as partial
from safe_llm_saea.paper_ready_safe_llm_saea import ExperimentConfig, LLMAssistedDE


def test_partial_intervention_protocol_is_frozen_and_never_replaces_worst() -> None:
    protocol = partial._protocol(["first", "second"])

    assert protocol["intervention"]["alphas"] == [0.25, 0.5, 0.75]
    assert protocol["intervention"]["diversity_floor"] == 0.95
    assert protocol["intervention"]["no_worst_individual_replacement"] is True
    assert protocol["future_boundary"].startswith("Do not capture instances 14--21")


def test_one_shot_directional_injection_validates_host_contract() -> None:
    config = ExperimentConfig(
        algorithm_name="injection-contract", problem_name="unit", dimension=2,
        max_evaluations=20, population_size=6, llm_enhancement=False,
        gating_enabled=False, enable_sniper=False, use_surrogate=False,
    )
    optimizer = LLMAssistedDE(config, api_key="disabled", use_api=False)

    optimizer.configure_one_shot_directional_injection(np.asarray([0.1, -0.2]), 0.5, 0.95)
    assert np.allclose(optimizer.pending_directional_delta, [0.1, -0.2])
    assert optimizer.pending_directional_alpha == 0.5
    assert optimizer.pending_directional_diversity_floor == 0.95

    with pytest.raises(ValueError):
        optimizer.configure_one_shot_directional_injection(np.asarray([0.1]), 0.5)
    with pytest.raises(ValueError):
        optimizer.configure_one_shot_directional_injection(np.asarray([0.1, 0.2]), 0.0)
