from safe_llm_saea.experiments.run_csa_fault_injection import FAULTS, run_fault_injection


def test_fault_injection_detects_every_requested_violation():
    report = run_fault_injection(trials_per_fault=5, seed=20260714)
    assert report["false_positive_rate"] == 0.0
    assert report["overall_detection_rate"] == 1.0
    assert report["expected_code_rate"] == 1.0
    assert set(report["fault_results"]) == set(FAULTS)
