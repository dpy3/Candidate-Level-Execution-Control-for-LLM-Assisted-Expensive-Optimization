from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from safe_llm_saea.csa_protocol import audit_csa_compliance
from safe_llm_saea.experiments.run_csa_calibrated_regret_study import run_pso
from safe_llm_saea.experiments.run_csa_prospective_f02_f10 import (
    DEDICATED_BASE_URL_ENV,
    DEDICATED_KEY_ENV,
    EXPOSED_KEY_ACK_ENV,
    ROTATION_ACK_ENV,
    _execution_manifest,
    _load_and_verify_protocol,
    _load_or_run,
    _require_rotated_credentials,
    _run_de,
    _sanitize_capture_event,
    _sha256_json,
    _verify_cache,
)


PROTOCOL_PATH = Path(
    "paper/q1/experiment_outputs/csa_regret_crossfit_20260717_complete/"
    "prospective_unseen_f02_f10_protocol.json"
)


def _protocol() -> dict:
    return _load_and_verify_protocol(PROTOCOL_PATH)


def test_protocol_hash_tampering_is_rejected(tmp_path: Path) -> None:
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    payload["runs_per_case"] += 1
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="protocol SHA-256 mismatch"):
        _load_and_verify_protocol(path)


def test_capture_requires_dedicated_rotated_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(DEDICATED_KEY_ENV, raising=False)
    monkeypatch.delenv(DEDICATED_BASE_URL_ENV, raising=False)
    monkeypatch.delenv(ROTATION_ACK_ENV, raising=False)
    monkeypatch.delenv(EXPOSED_KEY_ACK_ENV, raising=False)
    with pytest.raises(RuntimeError, match="requires dedicated"):
        _require_rotated_credentials(_protocol())

    monkeypatch.setenv(DEDICATED_KEY_ENV, "new-dedicated-key")
    monkeypatch.setenv(DEDICATED_BASE_URL_ENV, "https://example.invalid/v1")
    with pytest.raises(RuntimeError, match="explicit user risk acceptance"):
        _require_rotated_credentials(_protocol())

    monkeypatch.setenv(EXPOSED_KEY_ACK_ENV, "YES")
    key, base_url, model_id = _require_rotated_credentials(_protocol())
    assert key == "new-dedicated-key"
    assert base_url == "https://example.invalid/v1"
    assert model_id == "gpt-5.4"


def test_sanitized_event_strips_labels_state_raw_response_and_unknown_nested_fields() -> None:
    event = {
        "source_call_index": 0,
        "returned_candidate_vectors": [[1] * 10],
        "usage": {"total_tokens": 12, "raw_response": "secret"},
        "parameters_applied": {"mode": "explore", "incumbent": 4.2},
        "evaluated_candidate_objectives": [1.0],
        "incumbent_at_admission": 2.0,
        "raw_response": "secret",
    }
    sanitized = _sanitize_capture_event(event)
    serialized = json.dumps(sanitized)
    assert sanitized["usage"] == {"total_tokens": 12}
    assert sanitized["parameters_applied"] == {"mode": "explore"}
    for forbidden in ("objective", "incumbent", "raw_response"):
        assert forbidden not in serialized


def test_stale_or_tampered_cache_is_rejected(tmp_path: Path) -> None:
    protocol = _protocol()
    manifest = _execution_manifest(protocol)
    payload = {
        "protocol_sha256": protocol["protocol_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "labels_persisted": False,
        "proposal_vectors": [[0.0] * 10],
        "diagnostics": {"proposal_trace": []},
    }
    payload["artifact_sha256"] = _sha256_json(payload)
    path = tmp_path / "cache.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    _verify_cache(path, protocol, manifest)

    tampered = copy.deepcopy(payload)
    tampered["proposal_vectors"][0][0] = 1.0
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        _verify_cache(path, protocol, manifest)


def test_tampered_run_artifact_is_rejected(tmp_path: Path) -> None:
    protocol = _protocol()
    manifest = _execution_manifest(protocol)
    payload = {
        "protocol_sha256": protocol["protocol_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "objective": 1.0,
    }
    payload["artifact_sha256"] = _sha256_json(payload)
    path = tmp_path / "run.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert _load_or_run(path, lambda: {}, protocol, manifest)["objective"] == 1.0

    payload["objective"] = 2.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        _load_or_run(path, lambda: {}, protocol, manifest)


def test_strict_zero_empty_trace_is_compliant() -> None:
    audit = audit_csa_compliance({"strict_zero_call": True, "csa_trace": []})
    assert audit["csa_compliant"] is True


def test_forced_de_replay_stops_at_frozen_call_budget(tmp_path: Path) -> None:
    protocol = _protocol()
    manifest = _execution_manifest(protocol)
    events = []
    for call_index in range(2):
        vectors = [
            [float(call_index + candidate_index)] * 10
            for candidate_index in range(3)
        ]
        events.append({
            "source_call_index": call_index,
            "raw_candidates": 3,
            "dimension_valid_candidates": 3,
            "returned_candidates": 3,
            "returned_candidate_vectors": vectors,
            "parameters_applied": {},
            "usage": {},
            "call_outcome": "success",
        })
    path = tmp_path / "replay.json"
    path.write_text(json.dumps({"diagnostics": {"proposal_trace": events}}), encoding="utf-8")

    payload = _run_de(2, 0, 20260917, 80, "rank_top", path, protocol, manifest)
    assert len(payload["events"]) == 2
    assert payload["compliance"]["csa_compliant"] is True


def test_projected_pso_candidates_preserve_source_lineage() -> None:
    objective = lambda vector: float(sum(value * value for value in vector))
    proposals = [[20.0] * 10, [-20.0] * 10, [5.0] * 10]
    payload = run_pso(
        objective, 20260718, 50, proposals, "rank_top", 1.0, 1.0, 0.0,
        projection_alphas=[0.5], commit_policy="incumbent_improvement",
    )
    assert payload["compliance"]["csa_compliant"] is True
    assert payload["csa_trace"][0]["candidate_lineage"]
