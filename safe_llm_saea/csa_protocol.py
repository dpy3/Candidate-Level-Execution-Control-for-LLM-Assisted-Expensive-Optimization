from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from collections import Counter
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence

import numpy as np


CSA_PIPELINE_STATES = ("returned", "validated", "selected", "admitted", "evaluated")
CSA_OUTCOME_STATES = ("population_accepted", "best_update")
CSA_STATES = CSA_PIPELINE_STATES + CSA_OUTCOME_STATES
TRACE_SCHEMA = (
    "returned-validated-selected-admitted-evaluated-"
    "population_accepted-best_update/v2"
)
LEGACY_TRACE_SCHEMA = "returned-validated-selected-admitted-evaluated/v1"
SUPPORTED_TRACE_SCHEMAS = {TRACE_SCHEMA, LEGACY_TRACE_SCHEMA}


class AuthorizationError(RuntimeError):
    """Raised when a semantic objective call lacks valid host authorization."""


def candidate_id(vector: Sequence[float]) -> str:
    normalized = [round(float(value), 12) for value in vector]
    payload = json.dumps(normalized, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class CSAContract:
    proposal_source: str
    validator: str
    numerical_ranker: str
    admission_gate: str
    evaluation_authority: str
    update_authority: str
    trace_schema: str = TRACE_SCHEMA

    def as_dict(self) -> Dict[str, str]:
        return {
            "proposal_source": self.proposal_source,
            "validator": self.validator,
            "numerical_ranker": self.numerical_ranker,
            "admission_gate": self.admission_gate,
            "evaluation_authority": self.evaluation_authority,
            "update_authority": self.update_authority,
            "trace_schema": self.trace_schema,
        }


def _authorization_token(fields: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(fields), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class EvaluationAuthorization:
    candidate_id: str
    event_id: str
    policy: str
    budget_before: int
    serial: int
    issuer: str
    token: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "event_id": self.event_id,
            "policy": self.policy,
            "budget_before": self.budget_before,
            "serial": self.serial,
            "issuer": self.issuer,
            "token": self.token,
        }


def verify_authorization_certificate(certificate: Mapping[str, Any]) -> bool:
    required = {
        "candidate_id",
        "event_id",
        "policy",
        "budget_before",
        "serial",
        "issuer",
    }
    if not required.issubset(certificate):
        return False
    try:
        fields = {
            "candidate_id": str(certificate["candidate_id"]),
            "event_id": str(certificate["event_id"]),
            "policy": str(certificate["policy"]),
            "budget_before": int(certificate["budget_before"]),
            "serial": int(certificate["serial"]),
            "issuer": str(certificate["issuer"]),
        }
    except (TypeError, ValueError):
        return False
    return str(certificate.get("token", "")) == _authorization_token(fields)


class RuntimeAdmissionGate:
    """One-use host gate for semantic objective calls within one admission event."""

    def __init__(
        self,
        selected_vectors: Iterable[Sequence[float]],
        admitted_vectors: Iterable[Sequence[float]],
        *,
        remaining_budget: int,
        event_id: str,
        policy: str,
        issuer: str = "host-runtime",
    ) -> None:
        self.selected_ids = {candidate_id(vector) for vector in selected_vectors}
        self.admitted_ids = {candidate_id(vector) for vector in admitted_vectors}
        self.remaining_budget = int(remaining_budget)
        self.event_id = str(event_id)
        self.policy = str(policy)
        self.issuer = str(issuer)
        self._serial = 0
        self._issued: Dict[str, EvaluationAuthorization] = {}
        self._consumed_tokens: set[str] = set()
        self._lock = threading.Lock()

        if self.remaining_budget < 0:
            raise AuthorizationError("remaining budget cannot be negative")
        if not self.admitted_ids.issubset(self.selected_ids):
            raise AuthorizationError("admitted candidates must be numerically selected")
        if len(self.admitted_ids) > self.remaining_budget:
            raise AuthorizationError("admitted candidates exceed the remaining objective budget")

    def issue(self, vector: Sequence[float]) -> EvaluationAuthorization:
        with self._lock:
            identifier = candidate_id(vector)
            if identifier not in self.selected_ids:
                raise AuthorizationError("candidate was not numerically selected")
            if identifier not in self.admitted_ids:
                raise AuthorizationError("candidate was not admitted by the host budget gate")
            if self.remaining_budget <= 0:
                raise AuthorizationError("objective budget is exhausted")
            outstanding = len(self._issued) - len(self._consumed_tokens)
            if outstanding >= self.remaining_budget:
                raise AuthorizationError("outstanding authorizations exhaust the objective budget")

            fields = {
                "candidate_id": identifier,
                "event_id": self.event_id,
                "policy": self.policy,
                "budget_before": self.remaining_budget,
                "serial": self._serial,
                "issuer": self.issuer,
            }
            certificate = EvaluationAuthorization(token=_authorization_token(fields), **fields)
            self._issued[certificate.token] = certificate
            self._serial += 1
            return certificate

    def evaluate(
        self,
        certificate: EvaluationAuthorization,
        vector: Sequence[float],
        objective_function: Callable[[Sequence[float]], Any],
    ) -> Any:
        with self._lock:
            issued = self._issued.get(str(certificate.token))
            if issued != certificate or not verify_authorization_certificate(certificate.as_dict()):
                raise AuthorizationError("authorization certificate is invalid or was not issued by this gate")
            if certificate.token in self._consumed_tokens:
                raise AuthorizationError("authorization certificate has already been consumed")
            if candidate_id(vector) != certificate.candidate_id:
                raise AuthorizationError("candidate identity changed after authorization")
            if self.remaining_budget <= 0:
                raise AuthorizationError("objective budget is exhausted")

            self._consumed_tokens.add(certificate.token)
            self.remaining_budget -= 1
        return objective_function(vector)


def make_trace_event(
    returned_vectors: Iterable[Sequence[float]],
    validated_vectors: Iterable[Sequence[float]],
    selected_vectors: Iterable[Sequence[float]],
    admitted_vectors: Iterable[Sequence[float]],
    evaluated_vectors: Iterable[Sequence[float]],
    incumbent_update_ids: Iterable[str] = (),
    population_accepted_vectors: Iterable[Sequence[float]] = (),
    best_update_vectors: Iterable[Sequence[float]] = (),
    **metadata: Any,
) -> Dict[str, Any]:
    groups = {
        "returned": list(returned_vectors),
        "validated": list(validated_vectors),
        "selected": list(selected_vectors),
        "admitted": list(admitted_vectors),
        "evaluated": list(evaluated_vectors),
        "population_accepted": list(population_accepted_vectors),
        "best_update": list(best_update_vectors),
    }
    event: Dict[str, Any] = {"schema": TRACE_SCHEMA}
    for state, vectors in groups.items():
        event[state] = [
            {"candidate_id": candidate_id(vector), "vector": [float(value) for value in vector]}
            for vector in vectors
        ]
    event["incumbent_update_ids"] = list(incumbent_update_ids)
    event.update(metadata)
    return event


def _ids(event: Mapping[str, Any], state: str) -> List[str]:
    return [str(item.get("candidate_id", "")) for item in event.get(state, [])]


def _has_predecessor(
    child_id: str, parent_ids: set[str], lineage: Mapping[str, Any]
) -> bool:
    if child_id in parent_ids:
        return True
    recorded = lineage.get(child_id, [])
    if isinstance(recorded, str):
        recorded = [recorded]
    return any(str(parent_id) in parent_ids for parent_id in recorded)


def audit_csa_compliance(payload: Mapping[str, Any]) -> Dict[str, Any]:
    events = list(payload.get("csa_trace", []))
    strict_zero = bool(payload.get("strict_zero_call", False))
    violations: List[Dict[str, Any]] = []
    traced_evaluations = 0
    complete_evaluations = 0
    runtime_authorization_enforced_events = 0

    for event_index, event in enumerate(events):
        if event.get("schema") not in SUPPORTED_TRACE_SCHEMAS:
            violations.append({
                "event": event_index,
                "type": "trace_schema_mismatch",
                "candidate_ids": [],
                "expected": TRACE_SCHEMA,
                "observed": event.get("schema"),
            })
        for state in CSA_STATES:
            for item in event.get(state, []):
                try:
                    observed_id = str(item.get("candidate_id", ""))
                    expected_id = candidate_id(item["vector"])
                except (KeyError, TypeError, ValueError):
                    observed_id = str(item.get("candidate_id", "")) if isinstance(item, Mapping) else ""
                    expected_id = "unparseable-vector"
                if observed_id != expected_id:
                    violations.append({
                        "event": event_index,
                        "type": "candidate_identity_mismatch",
                        "state": state,
                        "candidate_ids": [observed_id],
                        "expected_candidate_id": expected_id,
                    })
        state_ids = {state: _ids(event, state) for state in CSA_STATES}
        state_sets = {state: set(values) for state, values in state_ids.items()}
        lineage = event.get("candidate_lineage", {})
        if not isinstance(lineage, Mapping):
            lineage = {}
        transitions = (
            ("validated", "returned"),
            ("selected", "validated"),
            ("admitted", "selected"),
            ("evaluated", "admitted"),
            ("population_accepted", "evaluated"),
            ("best_update", "population_accepted"),
        )
        for child, parent in transitions:
            unauthorized = sorted(
                candidate
                for candidate in state_sets[child]
                if not _has_predecessor(candidate, state_sets[parent], lineage)
            )
            if unauthorized:
                violations.append({"event": event_index, "type": f"{child}_without_{parent}", "candidate_ids": unauthorized})
        if bool(event.get("runtime_authorization_enforced", False)):
            runtime_authorization_enforced_events += 1
            certificates = list(event.get("evaluation_authorizations", []))
            valid_certificates = [
                certificate
                for certificate in certificates
                if isinstance(certificate, Mapping)
                and verify_authorization_certificate(certificate)
            ]
            if len(valid_certificates) != len(certificates):
                violations.append({
                    "event": event_index,
                    "type": "invalid_runtime_authorization_certificate",
                    "candidate_ids": [],
                })
            certificate_ids = [
                str(certificate.get("candidate_id", ""))
                for certificate in valid_certificates
            ]
            certificate_tokens = [
                str(certificate.get("token", ""))
                for certificate in valid_certificates
            ]
            duplicate_tokens = sorted({
                token for token in certificate_tokens if certificate_tokens.count(token) > 1
            })
            if duplicate_tokens:
                violations.append({
                    "event": event_index,
                    "type": "duplicate_runtime_authorization",
                    "candidate_ids": [],
                })
            unauthorized_certificates = sorted(set(certificate_ids) - state_sets["admitted"])
            if unauthorized_certificates:
                violations.append({
                    "event": event_index,
                    "type": "authorization_without_admission",
                    "candidate_ids": unauthorized_certificates,
                })
            certificate_counts = Counter(certificate_ids)
            evaluated_counts = Counter(state_ids["evaluated"])
            missing_certificates = sorted(
                identifier
                for identifier, count in evaluated_counts.items()
                if certificate_counts[identifier] < count
            )
            if missing_certificates:
                violations.append({
                    "event": event_index,
                    "type": "evaluated_without_runtime_authorization",
                    "candidate_ids": missing_certificates,
                })
        update_ids = set(str(value) for value in event.get("incumbent_update_ids", []))
        update_ids.update(state_sets["best_update"])
        unevaluated_updates = sorted(update_ids - state_sets["evaluated"])
        if unevaluated_updates:
            violations.append({"event": event_index, "type": "update_without_evaluation", "candidate_ids": unevaluated_updates})
        traced_evaluations += len(state_ids["evaluated"])
        complete_evaluations += sum(
            1 for value in state_ids["evaluated"]
            if value in state_sets["admitted"]
            and value in state_sets["selected"]
            and value in state_sets["validated"]
            and _has_predecessor(value, state_sets["returned"], lineage)
        )

    nonempty_semantic_path = any(_ids(event, state) for event in events for state in CSA_STATES)
    zero_call_closure = (not strict_zero) or (not nonempty_semantic_path)
    if not zero_call_closure:
        violations.append({"event": None, "type": "strict_zero_nonempty_semantic_path", "candidate_ids": []})

    trace_manifest_consistency = True
    trace_manifest_mismatches: Dict[str, Dict[str, Any]] = {}
    trace_manifest = payload.get("trace_manifest")
    if isinstance(trace_manifest, Mapping):
        observed_manifest = {
            "event_count": len(events),
            "semantic_evaluations": traced_evaluations,
            "runtime_authorization_events": runtime_authorization_enforced_events,
        }
        for key, observed in observed_manifest.items():
            expected = trace_manifest.get(key)
            if expected is not None and int(expected) != int(observed):
                trace_manifest_mismatches[key] = {"expected": int(expected), "observed": int(observed)}
        trace_manifest_consistency = not trace_manifest_mismatches
        if not trace_manifest_consistency:
            violations.append({
                "event": None,
                "type": "trace_manifest_mismatch",
                "candidate_ids": [],
                "fields": trace_manifest_mismatches,
            })

    resource_audit = payload.get("resource_audit")
    resource_consistency = True
    resource_mismatches: Dict[str, Dict[str, Any]] = {}
    if isinstance(resource_audit, Mapping):
        expected = resource_audit.get("expected", {})
        observed = resource_audit.get("observed", {})
        if isinstance(expected, Mapping) and isinstance(observed, Mapping):
            for key in sorted(set(expected) | set(observed)):
                if expected.get(key) != observed.get(key):
                    resource_mismatches[str(key)] = {
                        "expected": expected.get(key),
                        "observed": observed.get(key),
                    }
        else:
            resource_mismatches["resource_audit"] = {
                "expected": "mapping",
                "observed": type(resource_audit).__name__,
            }
        resource_consistency = not resource_mismatches
        if not resource_consistency:
            violations.append({
                "event": None,
                "type": "resource_mismatch",
                "candidate_ids": [],
                "fields": resource_mismatches,
            })

    proposal_separation = not any(item["type"] == "validated_without_returned" for item in violations)
    numerical_authorization = not any(
        item["type"] in {"selected_without_validated", "admitted_without_selected"} for item in violations
    )
    evaluation_closure = not any(item["type"] == "evaluated_without_admitted" for item in violations)
    population_acceptance_closure = not any(
        item["type"] == "population_accepted_without_evaluated" for item in violations
    )
    best_update_closure = not any(
        item["type"] == "best_update_without_population_accepted" for item in violations
    )
    update_closure = not any(item["type"] == "update_without_evaluation" for item in violations)
    trace_schema_valid = not any(item["type"] == "trace_schema_mismatch" for item in violations)
    candidate_identity_valid = not any(item["type"] == "candidate_identity_mismatch" for item in violations)
    runtime_authorization_consistency = not any(
        item["type"] in {
            "invalid_runtime_authorization_certificate",
            "duplicate_runtime_authorization",
            "authorization_without_admission",
            "evaluated_without_runtime_authorization",
        }
        for item in violations
    )
    trace_completeness = 1.0 if traced_evaluations == 0 else complete_evaluations / traced_evaluations
    compliant = all(
        (
            proposal_separation,
            numerical_authorization,
            evaluation_closure,
            population_acceptance_closure,
            best_update_closure,
            update_closure,
            zero_call_closure,
            resource_consistency,
            trace_schema_valid,
            candidate_identity_valid,
            runtime_authorization_consistency,
            trace_manifest_consistency,
        )
    ) and trace_completeness == 1.0
    return {
        "proposal_separation": proposal_separation,
        "numerical_authorization": numerical_authorization,
        "evaluation_closure": evaluation_closure,
        "population_acceptance_closure": population_acceptance_closure,
        "best_update_closure": best_update_closure,
        "update_closure": update_closure,
        "zero_call_closure": zero_call_closure,
        "resource_consistency": resource_consistency,
        "resource_mismatches": resource_mismatches,
        "trace_schema_valid": trace_schema_valid,
        "candidate_identity_valid": candidate_identity_valid,
        "runtime_authorization_consistency": runtime_authorization_consistency,
        "runtime_authorization_enforced_events": runtime_authorization_enforced_events,
        "trace_manifest_consistency": trace_manifest_consistency,
        "trace_manifest_mismatches": trace_manifest_mismatches,
        "trace_completeness": trace_completeness,
        "semantic_evaluations": traced_evaluations,
        "violations": violations,
        "csa_compliant": compliant,
    }
