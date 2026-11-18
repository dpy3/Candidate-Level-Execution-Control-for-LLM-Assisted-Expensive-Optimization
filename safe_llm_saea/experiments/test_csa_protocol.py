import pytest
from concurrent.futures import ThreadPoolExecutor

from safe_llm_saea.csa_protocol import (
    AuthorizationError,
    RuntimeAdmissionGate,
    audit_csa_compliance,
    candidate_id,
    make_trace_event,
)


def test_compliant_trace_passes():
    vector = [0.1, 0.2]
    payload = {
        "strict_zero_call": False,
        "csa_trace": [
            make_trace_event(
                [vector],
                [vector],
                [vector],
                [vector],
                [vector],
                population_accepted_vectors=[vector],
                best_update_vectors=[vector],
            )
        ],
    }
    report = audit_csa_compliance(payload)
    assert report["csa_compliant"]
    assert report["trace_completeness"] == 1.0


def _violation_types(report):
    return {item["type"] for item in report["violations"]}


def test_missing_validated_state_is_detected():
    vector = [0.1, 0.2]
    event = make_trace_event([vector], [], [vector], [vector], [vector])
    report = audit_csa_compliance({"strict_zero_call": False, "csa_trace": [event]})
    assert not report["csa_compliant"]
    assert "selected_without_validated" in _violation_types(report)


def test_evaluation_without_admission_is_detected():
    vector = [0.1, 0.2]
    event = make_trace_event([vector], [vector], [vector], [], [vector])
    report = audit_csa_compliance({"strict_zero_call": False, "csa_trace": [event]})
    assert not report["csa_compliant"]
    assert "evaluated_without_admitted" in _violation_types(report)


def test_update_without_evaluation_fails():
    vector = [0.1, 0.2]
    event = make_trace_event([vector], [vector], [vector], [vector], [], incumbent_update_ids=["not-evaluated"])
    report = audit_csa_compliance({"strict_zero_call": False, "csa_trace": [event]})
    assert not report["csa_compliant"]
    assert not report["update_closure"]
    assert "update_without_evaluation" in _violation_types(report)


def test_strict_zero_requires_empty_path():
    vector = [0.1, 0.2]
    event = make_trace_event([vector], [], [], [], [])
    report = audit_csa_compliance({"strict_zero_call": True, "csa_trace": [event]})
    assert not report["zero_call_closure"]
    assert "strict_zero_nonempty_semantic_path" in _violation_types(report)


def test_resource_mismatch_is_detected():
    payload = {
        "strict_zero_call": True,
        "csa_trace": [],
        "resource_audit": {
            "expected": {"seed": 17, "max_fes": 200, "evaluations_used": 200},
            "observed": {"seed": 18, "max_fes": 200, "evaluations_used": 199},
        },
    }
    report = audit_csa_compliance(payload)
    assert not report["csa_compliant"]
    assert not report["resource_consistency"]
    assert set(report["resource_mismatches"]) == {"seed", "evaluations_used"}
    assert "resource_mismatch" in _violation_types(report)


def test_candidate_id_tampering_is_detected():
    vector = [0.1, 0.2]
    event = make_trace_event([vector], [vector], [vector], [vector], [vector])
    event["evaluated"][0]["candidate_id"] = "forged-id"
    report = audit_csa_compliance({"strict_zero_call": False, "csa_trace": [event]})
    assert not report["csa_compliant"]
    assert not report["candidate_identity_valid"]
    assert "candidate_identity_mismatch" in _violation_types(report)


def test_trace_schema_tampering_is_detected():
    vector = [0.1, 0.2]
    event = make_trace_event([vector], [vector], [vector], [vector], [vector])
    event["schema"] = "returned-evaluated/v0"
    report = audit_csa_compliance({"strict_zero_call": False, "csa_trace": [event]})
    assert not report["csa_compliant"]
    assert not report["trace_schema_valid"]
    assert "trace_schema_mismatch" in _violation_types(report)


def test_population_acceptance_requires_evaluation():
    vector = [0.1, 0.2]
    event = make_trace_event(
        [vector], [vector], [vector], [vector], [], population_accepted_vectors=[vector]
    )
    report = audit_csa_compliance({"strict_zero_call": False, "csa_trace": [event]})
    assert not report["population_acceptance_closure"]
    assert "population_accepted_without_evaluated" in _violation_types(report)


def test_best_update_requires_population_acceptance():
    vector = [0.1, 0.2]
    event = make_trace_event(
        [vector], [vector], [vector], [vector], [vector], best_update_vectors=[vector]
    )
    report = audit_csa_compliance({"strict_zero_call": False, "csa_trace": [event]})
    assert not report["best_update_closure"]
    assert "best_update_without_population_accepted" in _violation_types(report)


def test_host_derived_candidate_is_traceable_through_parent_lineage():
    returned = [0.1, 0.2]
    fused = [0.15, 0.25]
    event = make_trace_event(
        [returned],
        [fused],
        [fused],
        [fused],
        [fused],
        candidate_lineage={candidate_id(fused): [candidate_id(returned)]},
    )
    report = audit_csa_compliance({"strict_zero_call": False, "csa_trace": [event]})
    assert report["csa_compliant"]
    assert report["trace_completeness"] == 1.0


def test_runtime_gate_requires_admission_and_blocks_certificate_replay():
    vector = [0.1, 0.2]
    gate = RuntimeAdmissionGate(
        [vector], [vector], remaining_budget=1, event_id="event-1", policy="rank_top"
    )
    certificate = gate.issue(vector)
    calls = []
    value = gate.evaluate(certificate, vector, lambda item: calls.append(item) or 3.5)
    assert value == 3.5
    assert len(calls) == 1
    with pytest.raises(AuthorizationError, match="already been consumed"):
        gate.evaluate(certificate, vector, lambda item: 0.0)

    unadmitted_gate = RuntimeAdmissionGate(
        [vector], [], remaining_budget=1, event_id="event-2", policy="rank_top"
    )
    with pytest.raises(AuthorizationError, match="not admitted"):
        unadmitted_gate.issue(vector)


def test_runtime_gate_rejects_admission_beyond_remaining_budget():
    vectors = [[0.1, 0.2], [0.3, 0.4]]
    with pytest.raises(AuthorizationError, match="exceed"):
        RuntimeAdmissionGate(
            vectors,
            vectors,
            remaining_budget=1,
            event_id="event-budget",
            policy="rank_top",
        )


def test_runtime_authorization_certificate_is_audited():
    vector = [0.1, 0.2]
    gate = RuntimeAdmissionGate(
        [vector], [vector], remaining_budget=1, event_id="event-3", policy="rank_top"
    )
    certificate = gate.issue(vector).as_dict()
    event = make_trace_event(
        [vector],
        [vector],
        [vector],
        [vector],
        [vector],
        runtime_authorization_enforced=True,
        evaluation_authorizations=[certificate],
    )
    clean = audit_csa_compliance({"strict_zero_call": False, "csa_trace": [event]})
    assert clean["csa_compliant"]
    assert clean["runtime_authorization_consistency"]

    event["evaluation_authorizations"][0]["token"] = "forged-token"
    forged = audit_csa_compliance({"strict_zero_call": False, "csa_trace": [event]})
    assert not forged["csa_compliant"]
    assert not forged["runtime_authorization_consistency"]
    assert "invalid_runtime_authorization_certificate" in _violation_types(forged)


def test_trace_manifest_detects_whole_event_truncation():
    vector = [0.1, 0.2]
    event = make_trace_event([vector], [vector], [vector], [vector], [vector])
    payload = {
        "strict_zero_call": False,
        "csa_trace": [event],
        "trace_manifest": {
            "event_count": 1,
            "semantic_evaluations": 1,
            "runtime_authorization_events": 0,
        },
    }
    assert audit_csa_compliance(payload)["csa_compliant"]
    payload["csa_trace"] = []
    report = audit_csa_compliance(payload)
    assert not report["trace_manifest_consistency"]
    assert "trace_manifest_mismatch" in _violation_types(report)


def test_runtime_gate_consumes_token_before_objective_exception():
    vector = [0.1, 0.2]
    gate = RuntimeAdmissionGate(
        [vector], [vector], remaining_budget=1, event_id="exception", policy="rank_top"
    )
    certificate = gate.issue(vector)
    with pytest.raises(RuntimeError, match="objective failed"):
        gate.evaluate(certificate, vector, lambda _: (_ for _ in ()).throw(RuntimeError("objective failed")))
    assert gate.remaining_budget == 0
    with pytest.raises(AuthorizationError, match="already been consumed"):
        gate.evaluate(certificate, vector, lambda _: 0.0)


def test_runtime_gate_blocks_concurrent_double_consumption():
    vector = [0.1, 0.2]
    gate = RuntimeAdmissionGate(
        [vector], [vector], remaining_budget=1, event_id="concurrent", policy="rank_top"
    )
    certificate = gate.issue(vector)
    objective_calls = []

    def attempt():
        try:
            return gate.evaluate(certificate, vector, lambda _: objective_calls.append(1) or 1.0)
        except AuthorizationError:
            return "blocked"

    with ThreadPoolExecutor(max_workers=8) as executor:
        outcomes = list(executor.map(lambda _: attempt(), range(32)))
    assert outcomes.count(1.0) == 1
    assert outcomes.count("blocked") == 31
    assert len(objective_calls) == 1
