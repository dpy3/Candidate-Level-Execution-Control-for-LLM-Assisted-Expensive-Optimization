from safe_llm_saea.application_benchmark.run_csa_development_lock import _selection_key
from safe_llm_saea.application_benchmark.uav.build_uav_development_split import geometry_sha256


def test_geometry_hash_ignores_case_metadata():
    base = {"start": [0, 0, 0], "goal": [1, 1, 1], "bounds": {}, "path_parameterization": {}, "obstacles": [], "reference_path": []}
    left = {**base, "case_id": "left"}
    right = {**base, "case_id": "right", "difficulty_tags": ["other"]}
    assert geometry_sha256(left) == geometry_sha256(right)


def test_selection_prioritizes_feasibility_then_cost_then_evaluations():
    weak_feasibility = {"score": {"strict_feasible_rate": 0.9, "mean_system_cost": 1.0, "mean_semantic_evaluations": 20}, "surrogate_min_history": 24, "surrogate_acceptance_margin": 0.01}
    strong_feasibility = {"score": {"strict_feasible_rate": 1.0, "mean_system_cost": 10.0, "mean_semantic_evaluations": 1}, "surrogate_min_history": 12, "surrogate_acceptance_margin": 0.0}
    assert min([weak_feasibility, strong_feasibility], key=_selection_key) is strong_feasibility
