from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable, Dict, List

import numpy as np

from safe_llm_saea.csa_protocol import AuthorizationError, RuntimeAdmissionGate, audit_csa_compliance, make_trace_event
from safe_llm_saea.experiments.run_csa_fault_injection import run_fault_injection


PROTOCOL_ID = "CSA-RUNTIME-STRESS-AND-FAULT-BLOCKING-2026-07-15"


def _save(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _vectors(count: int, dimension: int, seed: int) -> List[List[float]]:
    return np.random.default_rng(seed).uniform(-1.0, 1.0, size=(count, dimension)).tolist()


def _stress_point(count: int, dimension: int, repetitions: int, seed: int) -> Dict[str, Any]:
    construction_ns = []
    issue_evaluate_ns = []
    checker_ns = []
    serialized_bytes = []
    for repetition in range(repetitions):
        vectors = _vectors(count, dimension, seed + repetition)
        started = time.perf_counter_ns()
        gate = RuntimeAdmissionGate(vectors, vectors, remaining_budget=count, event_id=f"stress-{dimension}-{count}-{repetition}", policy="stress")
        construction_ns.append(time.perf_counter_ns() - started)

        certificates = []
        started = time.perf_counter_ns()
        for vector in vectors:
            certificate = gate.issue(vector)
            certificates.append(certificate.as_dict())
            gate.evaluate(certificate, vector, lambda _: 0.0)
        issue_evaluate_ns.append(time.perf_counter_ns() - started)

        event = make_trace_event(
            vectors, vectors, vectors, vectors, vectors,
            population_accepted_vectors=vectors,
            runtime_authorization_enforced=True,
            evaluation_authorizations=certificates,
        )
        payload = {
            "strict_zero_call": False,
            "csa_trace": [event],
            "trace_manifest": {"event_count": 1, "semantic_evaluations": count, "runtime_authorization_events": 1},
        }
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        serialized_bytes.append(len(encoded))
        started = time.perf_counter_ns()
        audit = audit_csa_compliance(payload)
        checker_ns.append(time.perf_counter_ns() - started)
        if not audit["csa_compliant"]:
            raise RuntimeError(f"Stress trace failed compliance at D={dimension}, N={count}")

    per_certificate_us = [value / count / 1000.0 for value in issue_evaluate_ns]
    return {
        "dimension": dimension,
        "semantic_candidates": count,
        "repetitions": repetitions,
        "median_gate_construction_us": median(construction_ns) / 1000.0,
        "median_issue_evaluate_us_per_candidate": median(per_certificate_us),
        "p95_issue_evaluate_us_per_candidate": float(np.percentile(per_certificate_us, 95)),
        "median_checker_ms": median(checker_ns) / 1_000_000.0,
        "median_serialized_trace_bytes": int(median(serialized_bytes)),
        "bytes_per_candidate": mean(serialized_bytes) / count,
    }


def run_stress(repetitions: int, seed: int) -> Dict[str, Any]:
    grid = []
    for dimension in (2, 10, 50, 100, 500, 1000):
        for count in (1, 10, 100, 1000):
            if dimension * count > 100_000:
                continue
            grid.append(_stress_point(count, dimension, repetitions, seed + dimension * 10000 + count))
    return {
        "repetitions_per_point": repetitions,
        "grid": grid,
        "max_dimension": max(item["dimension"] for item in grid),
        "max_candidate_count": max(item["semantic_candidates"] for item in grid),
        "worst_median_issue_evaluate_us_per_candidate": max(item["median_issue_evaluate_us_per_candidate"] for item in grid),
        "worst_median_checker_ms": max(item["median_checker_ms"] for item in grid),
        "largest_serialized_trace_bytes": max(item["median_serialized_trace_bytes"] for item in grid),
    }


def _blocked(action: Callable[[], Any]) -> bool:
    try:
        action()
    except AuthorizationError:
        return True
    return False


def _runtime_trial(name: str, trial: int) -> bool:
    vector = [0.1 + trial * 1e-9, 0.2]
    other = [0.3 + trial * 1e-9, 0.4]
    if name == "unselected issuance":
        gate = RuntimeAdmissionGate([vector], [vector], remaining_budget=1, event_id="u", policy="test")
        return _blocked(lambda: gate.issue(other))
    if name == "unadmitted issuance":
        gate = RuntimeAdmissionGate([vector], [], remaining_budget=1, event_id="a", policy="test")
        return _blocked(lambda: gate.issue(vector))
    if name == "admission beyond budget":
        return _blocked(lambda: RuntimeAdmissionGate([vector, other], [vector, other], remaining_budget=1, event_id="b", policy="test"))
    if name == "token replay":
        gate = RuntimeAdmissionGate([vector], [vector], remaining_budget=1, event_id="r", policy="test")
        certificate = gate.issue(vector)
        gate.evaluate(certificate, vector, lambda _: 0.0)
        return _blocked(lambda: gate.evaluate(certificate, vector, lambda _: 0.0))
    if name == "candidate replacement":
        gate = RuntimeAdmissionGate([vector], [vector], remaining_budget=1, event_id="i", policy="test")
        certificate = gate.issue(vector)
        return _blocked(lambda: gate.evaluate(certificate, other, lambda _: 0.0))
    if name == "forged certificate":
        gate = RuntimeAdmissionGate([vector], [vector], remaining_budget=1, event_id="f", policy="test")
        forged = replace(gate.issue(vector), token="forged")
        return _blocked(lambda: gate.evaluate(forged, vector, lambda _: 0.0))
    if name == "foreign-gate restart":
        old_gate = RuntimeAdmissionGate([vector], [vector], remaining_budget=1, event_id="old", policy="test")
        certificate = old_gate.issue(vector)
        new_gate = RuntimeAdmissionGate([vector], [vector], remaining_budget=1, event_id="new", policy="test")
        return _blocked(lambda: new_gate.evaluate(certificate, vector, lambda _: 0.0))
    if name == "concurrent duplicate consumption":
        gate = RuntimeAdmissionGate([vector], [vector], remaining_budget=1, event_id="c", policy="test")
        certificate = gate.issue(vector)
        objective_calls = []
        def attempt() -> str:
            try:
                gate.evaluate(certificate, vector, lambda _: objective_calls.append(1) or 0.0)
                return "allowed"
            except AuthorizationError:
                return "blocked"
        with ThreadPoolExecutor(max_workers=8) as executor:
            outcomes = list(executor.map(lambda _: attempt(), range(32)))
        return outcomes.count("allowed") == 1 and len(objective_calls) == 1
    if name == "concurrent issuance budget cap":
        vectors = _vectors(4, 2, trial + 77)
        gate = RuntimeAdmissionGate(vectors, vectors, remaining_budget=4, event_id="ci", policy="test")
        def attempt(index: int) -> str:
            try:
                gate.issue(vectors[index % len(vectors)])
                return "issued"
            except AuthorizationError:
                return "blocked"
        with ThreadPoolExecutor(max_workers=16) as executor:
            outcomes = list(executor.map(attempt, range(32)))
        return outcomes.count("issued") == 4 and outcomes.count("blocked") == 28
    if name == "objective exception consumes token":
        gate = RuntimeAdmissionGate([vector], [vector], remaining_budget=1, event_id="e", policy="test")
        certificate = gate.issue(vector)
        try:
            gate.evaluate(certificate, vector, lambda _: (_ for _ in ()).throw(RuntimeError("failure")))
        except RuntimeError:
            pass
        return gate.remaining_budget == 0 and _blocked(lambda: gate.evaluate(certificate, vector, lambda _: 0.0))
    raise KeyError(name)


def run_runtime_faults(trials: int) -> Dict[str, Any]:
    names = (
        "unselected issuance", "unadmitted issuance", "admission beyond budget", "token replay",
        "candidate replacement", "forged certificate", "foreign-gate restart",
        "concurrent duplicate consumption", "concurrent issuance budget cap", "objective exception consumes token",
    )
    results = {}
    for name in names:
        blocked = sum(_runtime_trial(name, trial) for trial in range(trials))
        results[name] = {"trials": trials, "blocked": blocked, "blocked_rate": blocked / trials}

    bypass_calls = []
    for trial in range(trials):
        objective = lambda value, calls=bypass_calls: calls.append(value) or 0.0
        objective([trial, 0.0])
    return {
        "enforced_fault_classes": len(names),
        "enforced_trials": len(names) * trials,
        "blocked": sum(item["blocked"] for item in results.values()),
        "blocked_rate": sum(item["blocked"] for item in results.values()) / (len(names) * trials),
        "results": results,
        "malicious_host_direct_bypass_control": {
            "trials": trials,
            "objective_calls": len(bypass_calls),
            "blocked_rate": 0.0,
            "interpretation": "Direct objective access outside the gate is outside the declared trusted-host guarantee.",
        },
    }


def _clean_trace(trial: int) -> Dict[str, Any]:
    vector = [0.1 + trial * 1e-9, 0.2]
    gate = RuntimeAdmissionGate([vector], [vector], remaining_budget=1, event_id=f"checker-{trial}", policy="test")
    certificate = gate.issue(vector)
    gate.evaluate(certificate, vector, lambda _: 0.0)
    event = make_trace_event(
        [vector], [vector], [vector], [vector], [vector],
        population_accepted_vectors=[vector],
        runtime_authorization_enforced=True,
        evaluation_authorizations=[certificate.as_dict()],
    )
    return {
        "strict_zero_call": False,
        "csa_trace": [event],
        "trace_manifest": {"event_count": 1, "semantic_evaluations": 1, "runtime_authorization_events": 1},
    }


def run_checker_extensions(trials: int) -> Dict[str, Any]:
    mutations = {
        "whole-event truncation": lambda payload: payload.update({"csa_trace": []}),
        "missing runtime certificate": lambda payload: payload["csa_trace"][0].update({"evaluation_authorizations": []}),
        "duplicate runtime certificate": lambda payload: payload["csa_trace"][0]["evaluation_authorizations"].append(deepcopy(payload["csa_trace"][0]["evaluation_authorizations"][0])),
    }
    results = {}
    clean_passes = 0
    for trial in range(trials):
        clean_passes += int(audit_csa_compliance(_clean_trace(trial))["csa_compliant"])
    for name, mutate in mutations.items():
        detected = 0
        observed_codes: Dict[str, int] = {}
        for trial in range(trials):
            payload = _clean_trace(trial)
            mutate(payload)
            audit = audit_csa_compliance(payload)
            detected += int(not audit["csa_compliant"])
            for violation in audit["violations"]:
                code = str(violation["type"])
                observed_codes[code] = observed_codes.get(code, 0) + 1
        results[name] = {"trials": trials, "detected": detected, "detection_rate": detected / trials, "observed_violation_counts": observed_codes}
    return {
        "additional_fault_classes": len(mutations),
        "additional_fault_trials": len(mutations) * trials,
        "additional_faults_detected": sum(item["detected"] for item in results.values()),
        "clean_trials": trials,
        "clean_passes": clean_passes,
        "false_positive_rate": 1.0 - clean_passes / trials,
        "results": results,
    }


def execute(args: argparse.Namespace) -> Dict[str, Any]:
    legacy_checker = run_fault_injection(args.trials, args.seed)
    report = {
        "protocol_id": PROTOCOL_ID,
        "stress": run_stress(args.repetitions, args.seed),
        "runtime_fault_blocking": run_runtime_faults(args.trials),
        "checker_legacy_six_classes": legacy_checker,
        "checker_extensions": run_checker_extensions(args.trials),
    }
    report["aggregate"] = {
        "runtime_enforced_faults_blocked": report["runtime_fault_blocking"]["blocked"],
        "runtime_enforced_fault_trials": report["runtime_fault_blocking"]["enforced_trials"],
        "checker_faults_detected": legacy_checker["faults_detected"] + report["checker_extensions"]["additional_faults_detected"],
        "checker_fault_trials": legacy_checker["fault_trials"] + report["checker_extensions"]["additional_fault_trials"],
        "checker_false_positive_rate": report["checker_extensions"]["false_positive_rate"],
        "malicious_host_bypass_blocked": False,
    }
    _save(report, Path(args.output))
    print(json.dumps(report["aggregate"], indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260715)
    execute(parser.parse_args())


if __name__ == "__main__":
    main()
