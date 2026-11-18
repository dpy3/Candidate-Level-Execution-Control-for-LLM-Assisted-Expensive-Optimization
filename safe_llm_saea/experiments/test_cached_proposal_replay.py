from __future__ import annotations

import json

import numpy as np

from safe_llm_saea.paper_ready_safe_llm_saea import ExperimentConfig, LLMAssistedDE


def test_replay_returns_identical_vectors_and_usage(tmp_path):
    source = {
        "diagnostics": {
            "proposal_trace": [
                {
                    "raw_candidates": 2,
                    "returned_candidate_vectors": [[0.1, 0.2], [0.3, 0.4]],
                    "parameters_applied": {"mode": "balanced", "F_base": 0.5, "CR_base": 0.9},
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120, "cost_usd": 0.01},
                }
            ]
        }
    }
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps(source), encoding="utf-8")
    config = ExperimentConfig(
        dimension=2,
        llm_enhancement=True,
        proposal_trace_mode="replay",
        proposal_replay_path=str(trace_path),
    )
    optimizer = LLMAssistedDE(config=config, use_api=False)
    optimizer.population = np.array([[0.0, 0.0], [1.0, 1.0]])
    optimizer.fitness = np.array([0.0, 1.0])
    optimizer.best_idx = 0
    optimizer.best_fit = 0.0
    optimizer.bounds = [(0.0, 1.0), (0.0, 1.0)]
    suggestions = optimizer._get_llm_suggestions(optimizer.bounds, gen=0)
    assert [item.tolist() for item in suggestions] == [[0.1, 0.2], [0.3, 0.4]]
    assert optimizer.total_tokens == 120
    assert optimizer.total_cost_usd == 0.01
    assert optimizer.proposal_replay_cursor == 1
    assert optimizer.admission_events[0]["source_call_index"] == 0


def test_full_and_weak_replay_share_stream_budget_and_seed(tmp_path):
    source = {
        "diagnostics": {
            "proposal_trace": [
                {
                    "raw_candidates": 2,
                    "returned_candidate_vectors": [[0.2] * 5, [0.8] * 5],
                    "parameters_applied": {"mode": "balanced", "F_base": 0.5, "CR_base": 0.9},
                    "usage": {"prompt_tokens": 80, "completion_tokens": 20, "total_tokens": 100, "cost_usd": 0.005},
                }
            ]
        }
    }
    trace_path = tmp_path / "paired_trace.json"
    trace_path.write_text(json.dumps(source), encoding="utf-8")
    results = []
    for weak_control in (False, True):
        np.random.seed(1234)
        config = ExperimentConfig(
            dimension=5,
            population_size=6,
            max_evaluations=30,
            llm_enhancement=True,
            max_llm_calls_per_run=1,
            max_llm_calls_stage1=1,
            max_llm_calls_stage2=0,
            min_remaining_budget=0,
            min_call_interval=1,
            force_initial_llm_call=True,
            enable_sniper=False,
            use_surrogate=True,
            weak_admission_control=weak_control,
            proposal_trace_mode="replay",
            proposal_replay_path=str(trace_path),
            enforce_full_evaluation_budget=True,
        )
        optimizer = LLMAssistedDE(config=config, use_api=False)
        result = optimizer.optimize(lambda x: float(np.sum(np.asarray(x) ** 2)), [(0.0, 1.0)] * 5)
        results.append(result)
    assert [result["evaluations_used"] for result in results] == [30, 30]
    assert [result["stage1_calls"] + result["stage2_calls"] for result in results] == [1, 1]
    assert results[0]["diagnostics"]["proposal_replay_source_sha256"] == results[1]["diagnostics"]["proposal_replay_source_sha256"]
    assert results[0]["diagnostics"]["proposal_trace"][0]["returned_candidate_vectors"] == results[1]["diagnostics"]["proposal_trace"][0]["returned_candidate_vectors"]
    for result in results:
        manifest = result["diagnostics"]["trace_manifest"]
        assert manifest["event_count"] == len(result["diagnostics"]["csa_trace"])
        assert manifest["semantic_evaluations"] == sum(
            len(event["evaluated"]) for event in result["diagnostics"]["csa_trace"]
        )
        for event in result["diagnostics"]["admission_events"]:
            assert event["runtime_authorization_enforced"]
            assert len(event["evaluation_authorizations"]) == event["evaluated_candidates"]


def test_fail_closed_policy_does_not_evaluate_without_surrogate():
    config = ExperimentConfig(
        dimension=2,
        population_size=2,
        max_evaluations=10,
        use_surrogate=True,
        surrogate_unavailable_policy="reject",
    )
    optimizer = LLMAssistedDE(config=config, use_api=False)
    optimizer.population = np.array([[0.0, 0.0], [1.0, 1.0]])
    optimizer.fitness = np.array([0.0, 2.0])
    optimizer.best_idx = 0
    optimizer.best_fit = 0.0
    optimizer.bounds = [(0.0, 1.0), (0.0, 1.0)]
    optimizer._new_admission_event(gen=0, progress=0.0)

    objective_calls = 0

    def objective(candidate):
        nonlocal objective_calls
        objective_calls += 1
        return float(np.sum(np.asarray(candidate) ** 2))

    optimizer._integrate_suggestions([np.array([0.2, 0.2]), np.array([0.8, 0.8])], objective)

    event = optimizer.admission_events[0]
    assert objective_calls == 0
    assert event["evaluated_candidates"] == 0
    assert event["surrogate_fallback_reason"] == "surrogate_not_ready_rejected"
    assert event["call_outcome"] == "surrogate_screened_all"
