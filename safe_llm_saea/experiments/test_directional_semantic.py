from __future__ import annotations

import json

import numpy as np
import pytest

from safe_llm_saea.directional_semantic import (
    build_directional_prompt,
    construct_directional_candidates,
    parse_directional_response,
    summarize_de_state,
    summarize_pso_state,
)


BOUNDS = [(-100.0, 100.0)] * 3


def test_backbone_prompts_are_distinct_and_direction_only() -> None:
    population = [[0, 0, 0], [1, 2, 3], [-1, -2, -3]]
    values = [0.0, 1.0, 2.0]
    de = summarize_de_state(population, values, BOUNDS, 0.3)
    pso = summarize_pso_state(population, values, population, BOUNDS, 0.3)
    de_system, _, _ = build_directional_prompt(de)
    pso_system, _, _ = build_directional_prompt(pso)
    assert "differential evolution" in de_system
    assert "particle swarm optimization" in pso_system
    assert "not absolute solutions" in de_system


def test_parser_and_host_constructor_enforce_direction_contract() -> None:
    content = json.dumps({"proposals": [
        {"direction": [1, 0, 0], "step_scale": 0.05, "intent": "a"},
        {"direction": [0, -1, 0], "step_scale": 0.10, "intent": "b"},
        {"direction": [0, 0, 0.5], "step_scale": 0.25, "intent": "c"},
    ]})
    instructions = parse_directional_response(content, 3)
    candidates, lineage, records = construct_directional_candidates(
        [95, -95, 0], instructions, BOUNDS
    )
    assert len(candidates) == len(lineage) == len(records) == 3
    assert all(np.all(candidate <= 100) and np.all(candidate >= -100) for candidate in candidates)


def test_parser_rejects_absolute_or_malformed_output() -> None:
    bad = json.dumps({"proposals": [
        {"direction": [200, 0, 0], "step_scale": 0.05},
        {"direction": [0, -1, 0], "step_scale": 0.10},
        {"direction": [0, 0, 1], "step_scale": 0.25},
    ]})
    with pytest.raises(ValueError):
        parse_directional_response(bad, 3)
