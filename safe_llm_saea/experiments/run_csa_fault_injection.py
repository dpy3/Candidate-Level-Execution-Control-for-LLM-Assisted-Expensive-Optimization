from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np

from safe_llm_saea.application_benchmark.common.io_utils import save_json
from safe_llm_saea.csa_protocol import (
    RuntimeAdmissionGate,
    audit_csa_compliance,
    make_trace_event,
)


Payload = Dict[str, Any]
Injector = Callable[[Payload, np.random.Generator], None]


def _clean_payload(vector: List[float]) -> Payload:
    gate = RuntimeAdmissionGate(
        [vector], [vector], remaining_budget=1, event_id="fault-event", policy="rank_top"
    )
    certificate = gate.issue(vector).as_dict()
    return {
        "strict_zero_call": False,
        "csa_trace": [
            make_trace_event(
                [vector],
                [vector],
                [vector],
                [vector],
                [vector],
                runtime_authorization_enforced=True,
                evaluation_authorizations=[certificate],
            )
        ],
        "resource_audit": {
            "expected": {"seed": 17, "max_fes": 120, "evaluations_used": 120},
            "observed": {"seed": 17, "max_fes": 120, "evaluations_used": 120},
        },
    }


def _forged_state(payload: Payload, rng: np.random.Generator) -> None:
    forged = [float(value) for value in rng.normal(size=4)]
    forged_item = make_trace_event([], [], [], [forged], [])['admitted'][0]
    payload["csa_trace"][0]["admitted"].append(forged_item)


def _skipped_admission(payload: Payload, rng: np.random.Generator) -> None:
    del rng
    payload["csa_trace"][0]["admitted"] = []


def _budget_mismatch(payload: Payload, rng: np.random.Generator) -> None:
    del rng
    payload["resource_audit"]["observed"]["evaluations_used"] = 119


def _candidate_id_replacement(payload: Payload, rng: np.random.Generator) -> None:
    payload["csa_trace"][0]["evaluated"][0]["candidate_id"] = (
        f"replaced-{int(rng.integers(0, 2**31))}"
    )


def _missing_predecessor(payload: Payload, rng: np.random.Generator) -> None:
    del rng
    payload["csa_trace"][0]["validated"] = []


def _forged_authorization_token(payload: Payload, rng: np.random.Generator) -> None:
    payload["csa_trace"][0]["evaluation_authorizations"][0]["token"] = (
        f"forged-{int(rng.integers(0, 2**31))}"
    )


FAULTS: Dict[str, tuple[Injector, str]] = {
    "forged_state": (_forged_state, "admitted_without_selected"),
    "skipped_admission": (_skipped_admission, "evaluated_without_admitted"),
    "budget_mismatch": (_budget_mismatch, "resource_mismatch"),
    "candidate_id_replacement": (
        _candidate_id_replacement,
        "candidate_identity_mismatch",
    ),
    "missing_predecessor": (_missing_predecessor, "selected_without_validated"),
    "forged_authorization_token": (
        _forged_authorization_token,
        "invalid_runtime_authorization_certificate",
    ),
}


def run_fault_injection(trials_per_fault: int = 100, seed: int = 20260714) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    clean_passes = 0
    results: Dict[str, Any] = {}
    for trial in range(trials_per_fault):
        vector = [float(value) for value in rng.normal(size=4)]
        report = audit_csa_compliance(_clean_payload(vector))
        clean_passes += int(bool(report["csa_compliant"]))

    for fault_name, (injector, expected_type) in FAULTS.items():
        detected = 0
        expected_code_detected = 0
        observed_codes: Dict[str, int] = {}
        for trial in range(trials_per_fault):
            vector = [float(value) for value in rng.normal(size=4)]
            payload = deepcopy(_clean_payload(vector))
            injector(payload, rng)
            report = audit_csa_compliance(payload)
            codes = {str(item["type"]) for item in report["violations"]}
            detected += int(not bool(report["csa_compliant"]))
            expected_code_detected += int(expected_type in codes)
            for code in codes:
                observed_codes[code] = observed_codes.get(code, 0) + 1
        results[fault_name] = {
            "trials": trials_per_fault,
            "expected_violation": expected_type,
            "detected": detected,
            "detection_rate": detected / trials_per_fault,
            "expected_code_detected": expected_code_detected,
            "expected_code_rate": expected_code_detected / trials_per_fault,
            "observed_violation_counts": observed_codes,
        }

    total_faults = trials_per_fault * len(FAULTS)
    total_detected = sum(int(item["detected"]) for item in results.values())
    total_expected = sum(int(item["expected_code_detected"]) for item in results.values())
    return {
        "protocol_id": "CSA-CHECKER-FAULT-INJECTION-RUNTIME-GATE-2026-07-15",
        "seed": seed,
        "trials_per_fault": trials_per_fault,
        "clean_trials": trials_per_fault,
        "clean_passes": clean_passes,
        "false_positive_rate": 1.0 - clean_passes / trials_per_fault,
        "fault_trials": total_faults,
        "faults_detected": total_detected,
        "overall_detection_rate": total_detected / total_faults,
        "expected_codes_detected": total_expected,
        "expected_code_rate": total_expected / total_faults,
        "fault_results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic CSA checker fault injection.")
    parser.add_argument("--trials_per_fault", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run_fault_injection(args.trials_per_fault, args.seed)
    output = Path(args.output)
    save_json(report, output)
    print(report)


if __name__ == "__main__":
    main()
