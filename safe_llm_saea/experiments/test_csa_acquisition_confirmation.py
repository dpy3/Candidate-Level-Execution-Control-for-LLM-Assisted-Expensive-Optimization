from safe_llm_saea.application_benchmark.run_csa_acquisition_confirmation import _selection_key


def test_acquisition_selection_prioritizes_feasibility_then_cost():
    safer = {"score": {"strict_feasible_rate": 1.0, "mean_system_cost": 20.0, "semantic_evaluations": 10}, "surrogate_admission_policy": "rank_top", "surrogate_min_history": 16}
    cheaper_but_infeasible = {"score": {"strict_feasible_rate": 0.9, "mean_system_cost": 1.0, "semantic_evaluations": 1}, "surrogate_admission_policy": "improvement_threshold", "surrogate_min_history": 24}
    assert min([safer, cheaper_but_infeasible], key=_selection_key) is safer


def test_acquisition_selection_uses_cost_before_evaluation_count():
    cheaper = {"score": {"strict_feasible_rate": 1.0, "mean_system_cost": 10.0, "semantic_evaluations": 20}, "surrogate_admission_policy": "rank_top", "surrogate_min_history": 16}
    selective = {"score": {"strict_feasible_rate": 1.0, "mean_system_cost": 11.0, "semantic_evaluations": 1}, "surrogate_admission_policy": "improvement_threshold", "surrogate_min_history": 24}
    assert min([cheaper, selective], key=_selection_key) is cheaper
