from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from safe_llm_saea.cec2022 import get_bounds, get_function
from safe_llm_saea.csa_protocol import audit_csa_compliance
from safe_llm_saea.experiments.run_csa_calibrated_regret_study import (
    BOUNDS,
    DIMENSION,
    _candidate_records,
    _performance_row,
    _regret_model_metrics,
    _save_json,
    _sha256_json,
    run_pso,
)
from safe_llm_saea.experiments.run_csa_regret_crossfit import _summarize
from safe_llm_saea.paper_ready_safe_llm_saea import ExperimentConfig, LLMAssistedDE


EXECUTION_MANIFEST_ID = "CSA-PROSPECTIVE-F02-F10-EXECUTION-MANIFEST-2026-07-17"
DEDICATED_KEY_ENV = "CSA_PROSPECTIVE_API_KEY"
DEDICATED_BASE_URL_ENV = "CSA_PROSPECTIVE_BASE_URL"
ROTATION_ACK_ENV = "CSA_PROSPECTIVE_KEY_ROTATED"
EXPOSED_KEY_ACK_ENV = "CSA_PROSPECTIVE_EXPOSED_KEY_ACCEPTED"
SANITIZED_EVENT_KEYS = (
    "source_call_index",
    "raw_candidates",
    "dimension_valid_candidates",
    "returned_candidates",
    "returned_candidate_vectors",
    "parameters_applied",
    "usage",
    "call_outcome",
)
SANITIZED_PARAMETER_KEYS = ("mode", "F_base", "CR_base")
SANITIZED_USAGE_KEYS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cost_usd",
)


def _load_and_verify_protocol(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = str(payload.get("protocol_sha256", ""))
    actual = _sha256_json({key: value for key, value in payload.items() if key != "protocol_sha256"})
    if not claimed or claimed != actual:
        raise RuntimeError("prospective protocol SHA-256 mismatch")
    if payload.get("status") != "frozen_not_executed":
        raise RuntimeError("prospective protocol is not in frozen_not_executed state")
    if payload.get("unseen_test_cases") != ["CEC2022 F02", "CEC2022 F10"]:
        raise RuntimeError("unexpected prospective test cases")
    return payload


def _execution_manifest(protocol: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "manifest_id": EXECUTION_MANIFEST_ID,
        "parent_protocol_sha256": protocol["protocol_sha256"],
        "status": "frozen_before_capture",
        "capture": {
            "host_backbone": "DE",
            "dimension": DIMENSION,
            "evaluation_budget": 80,
            "population_size": 12,
            "calls_per_run": int(protocol["proposal_capture"]["calls_per_run"]),
            "force_initial_call": True,
            "force_call_each_generation_until_budget": True,
            "surrogate_min_history": 12,
            "surrogate_policy": "rank_top",
            "surrogate_unavailable_policy": "reject",
            "persist_objective_labels": False,
            "persist_incumbent_state": False,
            "persist_raw_response": False,
        },
        "execution": {
            "test_fids": [2, 10],
            "runs_per_case": int(protocol["runs_per_case"]),
            "seed_base": 20260717,
            "de_population_size": 12,
            "de_budget": int(protocol["budgets"]["DE"]),
            "pso_population_size": 10,
            "pso_budget": int(protocol["budgets"]["PSO"]),
            "policies": ["strict-zero", "rank_top", "calibrated_regret"],
            "proposal_reuse": "same sanitized stream for all policies and both backbones",
            "resume_rule": "reuse only artifacts with matching protocol and manifest hashes",
        },
        "credential_boundary": {
            "key_env": DEDICATED_KEY_ENV,
            "base_url_env": DEDICATED_BASE_URL_ENV,
            "rotation_ack_env": ROTATION_ACK_ENV,
            "exposed_key_risk_ack_env": EXPOSED_KEY_ACK_ENV,
            "accepted_ack_modes": ["rotated", "user-accepted-exposed-key-risk"],
            "generic historical key variables are ignored": True,
            "credentials are never serialized": True,
        },
        "analysis": {
            "unit": "paired run",
            "coverage_source": "offline scoring of each frozen proposal pool after all declared runs",
            "no_early_stopping": True,
            "no_threshold_refit": True,
        },
    }
    payload["manifest_sha256"] = _sha256_json(payload)
    return payload


def _load_or_create_manifest(
    output_root: Path, protocol: dict[str, Any]
) -> dict[str, Any]:
    path = output_root / "execution_manifest.json"
    expected = _execution_manifest(protocol)
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        claimed = current.get("manifest_sha256")
        actual = _sha256_json({key: value for key, value in current.items() if key != "manifest_sha256"})
        if claimed != actual or claimed != expected["manifest_sha256"]:
            raise RuntimeError("execution manifest differs from the frozen expected manifest")
        return current
    _save_json(expected, path)
    return expected


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _seed_for(protocol: dict[str, Any], fid: int, run_id: int) -> int:
    if protocol.get("seed_rule") != "20260717 + fid * 100 + run_id":
        raise RuntimeError("unsupported frozen seed rule")
    return 20260717 + int(fid) * 100 + int(run_id)


def _generic_de_config(
    fid: int,
    enabled: bool,
    max_fes: int,
    calls: int,
    policy: str,
    replay_path: str = "",
    protocol: dict[str, Any] | None = None,
) -> ExperimentConfig:
    config = ExperimentConfig(
        algorithm_name=(
            f"L-ADE-CSA-prospective-{policy}" if enabled else "L-ADE-CSA-prospective-strict-zero"
        ),
        problem_name=f"CEC2022_F{fid:02d}",
        dimension=DIMENSION,
        max_evaluations=max_fes,
        population_size=12,
        runs=1,
        llm_enhancement=enabled,
        gating_enabled=enabled,
        max_llm_calls_per_run=calls if enabled else 0,
        max_llm_cost_usd_per_run=0.05,
        fusion_weight=0.84,
        de_F=0.50,
        de_CR=0.90,
        min_remaining_budget=24 if max_fes > 80 else 0,
        min_stagnation_generations=0,
        min_uncertainty_threshold=0.0,
        min_call_interval=1,
        min_confidence=0.0,
        force_initial_llm_call=enabled,
        force_llm_every_gen=enabled and max_fes <= 80,
        max_llm_calls_stage1=calls if enabled else 0,
        max_llm_calls_stage2=0,
        llm_suggestions_k=3,
        problem_description=(
            f"CEC 2022 F{fid:02d}, {DIMENSION}D bounded minimization; unseen prospective test"
        ),
        use_surrogate=True,
        enable_sniper=False,
        surrogate_min_history=12,
        surrogate_admission_policy=policy,
        surrogate_unavailable_policy="reject",
        enforce_full_evaluation_budget=True,
        proposal_trace_mode="replay" if replay_path else "capture",
        proposal_replay_path=replay_path,
        proposal_replay_strict=True,
    )
    if protocol is not None and policy == "calibrated_regret":
        model = protocol["regret_model"]
        config.regret_model_feature_mean = tuple(model["feature_mean"])
        config.regret_model_feature_scale = tuple(model["feature_scale"])
        config.regret_model_coefficients = tuple(model["coefficients"])
        config.regret_model_residual_radius = float(protocol["conformal_residual_radius"])
        config.regret_uncertainty_scale = 1.0
        config.regret_risk_multiplier = float(protocol["risk_multiplier"])
        config.regret_abstention_threshold = float(protocol["abstention_threshold"])
    return config


def _sanitize_capture_event(event: dict[str, Any]) -> dict[str, Any]:
    sanitized = {key: event.get(key) for key in SANITIZED_EVENT_KEYS}
    vectors = sanitized.get("returned_candidate_vectors") or []
    sanitized["returned_candidate_vectors"] = [
        [float(value) for value in vector] for vector in vectors
    ]
    usage = dict(sanitized.get("usage") or {})
    sanitized["usage"] = {
        key: usage[key] for key in SANITIZED_USAGE_KEYS if key in usage
    }
    parameters = dict(sanitized.get("parameters_applied") or {})
    sanitized["parameters_applied"] = {
        key: parameters[key] for key in SANITIZED_PARAMETER_KEYS if key in parameters
    }
    return sanitized


def _require_rotated_credentials(protocol: dict[str, Any]) -> tuple[str, str, str]:
    api_key = os.getenv(DEDICATED_KEY_ENV, "").strip()
    base_url = os.getenv(DEDICATED_BASE_URL_ENV, "").strip().rstrip("/")
    rotation_ack = os.getenv(ROTATION_ACK_ENV, "").strip().upper()
    exposed_key_ack = os.getenv(EXPOSED_KEY_ACK_ENV, "").strip().upper()
    if not api_key or not base_url:
        raise RuntimeError(
            f"capture requires dedicated {DEDICATED_KEY_ENV} and {DEDICATED_BASE_URL_ENV}"
        )
    if rotation_ack != "YES" and exposed_key_ack != "YES":
        raise RuntimeError(
            f"set {ROTATION_ACK_ENV}=YES after rotation or "
            f"{EXPOSED_KEY_ACK_ENV}=YES after explicit user risk acceptance"
        )
    model_id = str(protocol["proposal_capture"]["model_id"])
    return api_key, base_url, model_id


def _capture_path(output_root: Path, fid: int, run_id: int) -> Path:
    return output_root / "proposal_cache" / f"cec_f{fid:02d}_10d_r{run_id:02d}.json"


def capture_proposals(
    output_root: Path,
    protocol: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    api_key, base_url, model_id = _require_rotated_credentials(protocol)
    calls = int(protocol["proposal_capture"]["calls_per_run"])
    for fid in manifest["execution"]["test_fids"]:
        objective = get_function(int(fid), DIMENSION)
        for run_id in range(int(protocol["runs_per_case"])):
            path = _capture_path(output_root, int(fid), run_id)
            if path.exists():
                _verify_cache(path, protocol, manifest)
                continue
            seed = _seed_for(protocol, int(fid), run_id)
            _seed(seed)
            config = _generic_de_config(
                int(fid), True, int(manifest["capture"]["evaluation_budget"]),
                calls, "rank_top",
            )
            silent = io.StringIO()
            with contextlib.redirect_stdout(silent), contextlib.redirect_stderr(silent):
                optimizer = LLMAssistedDE(
                    config=config,
                    api_key=api_key,
                    base_url=base_url,
                    use_api=True,
                    llm_model=model_id,
                )
                result = optimizer.optimize(objective, BOUNDS)
            events = result.get("diagnostics", {}).get("proposal_trace", [])
            successful = int(result.get("api_success_calls", 0))
            if successful != calls or len(events) != calls:
                raise RuntimeError(
                    f"capture F{fid:02d} run {run_id} produced {successful}/{calls} successful calls"
                )
            sanitized = [_sanitize_capture_event(event) for event in events]
            vectors = [
                vector for event in sanitized
                for vector in event.get("returned_candidate_vectors", [])
            ]
            if not vectors:
                raise RuntimeError(f"capture F{fid:02d} run {run_id} returned no candidates")
            payload = {
                "protocol_sha256": protocol["protocol_sha256"],
                "manifest_sha256": manifest["manifest_sha256"],
                "fid": int(fid),
                "dimension": DIMENSION,
                "run_id": run_id,
                "seed": seed,
                "model_id": model_id,
                "base_url": base_url,
                "successful_calls": successful,
                "capture_cost_usd": float(result.get("total_cost_usd", 0.0)),
                "capture_total_tokens": int(result.get("total_tokens", 0)),
                "labels_persisted": False,
                "proposal_vectors": vectors,
                "diagnostics": {"proposal_trace": sanitized},
            }
            payload["artifact_sha256"] = _sha256_json(payload)
            _save_json(payload, path)


def _verify_cache(
    path: Path, protocol: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = payload.get("artifact_sha256")
    actual = _sha256_json({key: value for key, value in payload.items() if key != "artifact_sha256"})
    if claimed != actual:
        raise RuntimeError(f"proposal cache checksum mismatch: {path}")
    if payload.get("protocol_sha256") != protocol["protocol_sha256"]:
        raise RuntimeError(f"proposal cache protocol mismatch: {path}")
    if payload.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise RuntimeError(f"proposal cache manifest mismatch: {path}")
    if payload.get("labels_persisted") is not False:
        raise RuntimeError(f"proposal cache label boundary missing: {path}")
    forbidden = {
        "evaluated_candidate_objectives", "evaluated_candidate_regrets",
        "incumbent_at_admission", "incumbent_vector_at_admission",
        "best_update_candidate_vectors", "raw_response",
    }
    for event in payload.get("diagnostics", {}).get("proposal_trace", []):
        if forbidden.intersection(event):
            raise RuntimeError(f"proposal cache persists forbidden labels/state: {path}")
    return payload


def _run_de(
    fid: int,
    run_id: int,
    seed: int,
    max_fes: int,
    policy: str,
    cache_path: Path | None,
    protocol: dict[str, Any],
    manifest: dict[str, Any],
    commit_policy: str = "population_improvement",
    risk_model: dict[str, Any] | None = None,
    risk_probability_threshold: float = 0.5,
    calls: int | None = None,
    allow_control_updates: bool = True,
    projection_alpha: float = 0.0,
) -> dict[str, Any]:
    objective = get_function(fid, DIMENSION)
    enabled = policy != "strict-zero"
    replay_path = str(cache_path) if enabled and cache_path is not None else ""
    config = _generic_de_config(
        fid, enabled, max_fes,
        int(calls if calls is not None else protocol["proposal_capture"]["calls_per_run"]),
        policy if enabled else "rank_top",
        replay_path=replay_path,
        protocol=protocol,
    )
    config.semantic_commit_policy = commit_policy
    config.allow_llm_control_updates = bool(allow_control_updates)
    config.semantic_projection_alpha = float(projection_alpha)
    if risk_model is not None:
        config.risk_model_feature_mean = tuple(risk_model["feature_mean"])
        config.risk_model_feature_scale = tuple(risk_model["feature_scale"])
        config.risk_model_coefficients = tuple(risk_model["coefficients"])
        config.risk_model_intercept = float(risk_model["intercept"])
        config.risk_probability_threshold = float(risk_probability_threshold)
    _seed(seed)
    silent = io.StringIO()
    with contextlib.redirect_stdout(silent), contextlib.redirect_stderr(silent):
        optimizer = LLMAssistedDE(config, use_api=False, llm_model="gpt-5.4")
        result = optimizer.optimize(objective, BOUNDS)
    diagnostics = result.get("diagnostics", {})
    events = diagnostics.get("admission_events", [])
    payload = {
        "protocol_sha256": protocol["protocol_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "algorithm": "DE",
        "policy": policy,
        "fid": fid,
        "run_id": run_id,
        "seed": seed,
        "objective": float(result["best_fitness"]),
        "evaluations": int(result["evaluations_used"]),
        "semantic_evaluations": sum(int(event.get("evaluated_candidates", 0)) for event in events),
        "candidate_events": sum(int(event.get("validated_candidates", 0)) > 0 for event in events),
        "abstained_events": sum(bool(event.get("regret_abstained", False)) for event in events),
        "events": events,
        "csa_trace": diagnostics.get("csa_trace", []),
    }
    payload["compliance"] = audit_csa_compliance({
        "strict_zero_call": policy == "strict-zero",
        "csa_trace": payload["csa_trace"],
    })
    return payload


def _run_path(output_root: Path, fid: int, run_id: int, backbone: str, policy: str) -> Path:
    return output_root / "runs" / f"f{fid:02d}_r{run_id:02d}" / f"{backbone}_{policy}.json"


def _load_or_run(
    path: Path,
    producer: Any,
    protocol: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        claimed = payload.get("artifact_sha256")
        actual = _sha256_json({
            key: value for key, value in payload.items() if key != "artifact_sha256"
        })
        if claimed != actual:
            raise RuntimeError(f"run artifact checksum mismatch: {path}")
        if (
            payload.get("protocol_sha256") != protocol["protocol_sha256"]
            or payload.get("manifest_sha256") != manifest["manifest_sha256"]
        ):
            raise RuntimeError(f"stale run artifact: {path}")
        return payload
    payload = producer()
    if (
        payload.get("protocol_sha256") != protocol["protocol_sha256"]
        or payload.get("manifest_sha256") != manifest["manifest_sha256"]
    ):
        raise RuntimeError(f"producer returned an unbound run artifact: {path}")
    payload["artifact_sha256"] = _sha256_json(payload)
    _save_json(payload, path)
    return payload


def execute(
    output_root: Path,
    protocol: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    de_rows = {"rank_top": [], "calibrated_regret": []}
    pso_rows = {"rank_top": [], "calibrated_regret": []}
    de_candidate_records = []
    pso_candidate_records = []
    capture_cost = 0.0
    capture_tokens = 0
    for fid in manifest["execution"]["test_fids"]:
        objective = get_function(int(fid), DIMENSION)
        for run_id in range(int(protocol["runs_per_case"])):
            cache_path = _capture_path(output_root, int(fid), run_id)
            if not cache_path.exists():
                raise RuntimeError(f"missing proposal cache: {cache_path}")
            cache = _verify_cache(cache_path, protocol, manifest)
            capture_cost += float(cache.get("capture_cost_usd", 0.0))
            capture_tokens += int(cache.get("capture_total_tokens", 0))
            proposals = cache["proposal_vectors"]
            seed = _seed_for(protocol, int(fid), run_id)

            de_payloads = {}
            for policy in ("strict-zero", "rank_top", "calibrated_regret"):
                path = _run_path(output_root, int(fid), run_id, "de", policy)
                de_payloads[policy] = _load_or_run(
                    path,
                    lambda policy=policy: _run_de(
                        int(fid), run_id, seed, int(protocol["budgets"]["DE"]),
                        policy, cache_path, protocol, manifest,
                    ),
                    protocol, manifest,
                )
                if not de_payloads[policy]["compliance"]["csa_compliant"]:
                    raise RuntimeError(f"DE compliance failure F{fid:02d} run {run_id} {policy}")
            strict_de = float(de_payloads["strict-zero"]["objective"])
            for policy in ("rank_top", "calibrated_regret"):
                row = _performance_row("DE", policy, run_id, de_payloads[policy], strict_de)
                row["fid"] = int(fid)
                de_rows[policy].append(row)
            de_candidate_records.extend(_candidate_records({
                "fid": int(fid), "case": f"F{fid:02d}-10D", "run_id": run_id,
                "events": de_payloads["rank_top"]["events"],
            }))

            pso_payloads = {}
            for policy in ("native", "strict-zero", "rank_top", "calibrated_regret"):
                path = _run_path(output_root, int(fid), run_id, "pso", policy)
                pso_payloads[policy] = _load_or_run(
                    path,
                    lambda policy=policy: {
                        **run_pso(
                            objective,
                            seed,
                            int(protocol["budgets"]["PSO"]),
                            [] if policy == "native" else proposals,
                            policy,
                            1.0,
                            float(protocol["risk_multiplier"]),
                            float(protocol["abstention_threshold"]),
                            regret_model=(
                                protocol["regret_model"] if policy == "calibrated_regret" else None
                            ),
                            residual_radius=float(protocol["conformal_residual_radius"]),
                        ),
                        "protocol_sha256": protocol["protocol_sha256"],
                        "manifest_sha256": manifest["manifest_sha256"],
                        "fid": int(fid),
                        "run_id": run_id,
                    },
                    protocol, manifest,
                )
            if pso_payloads["native"]["trajectory_sha256"] != pso_payloads["strict-zero"]["trajectory_sha256"]:
                raise RuntimeError(f"PSO strict-zero trajectory mismatch F{fid:02d} run {run_id}")
            for policy in ("strict-zero", "rank_top", "calibrated_regret"):
                if not pso_payloads[policy]["compliance"]["csa_compliant"]:
                    raise RuntimeError(f"PSO compliance failure F{fid:02d} run {run_id} {policy}")
            for policy in ("rank_top", "calibrated_regret"):
                row = _performance_row(
                    "PSO", policy, run_id, pso_payloads[policy],
                    float(pso_payloads["strict-zero"]["best_fitness"]),
                )
                row["fid"] = int(fid)
                pso_rows[policy].append(row)
            pso_candidate_records.extend(_candidate_records({
                "fid": int(fid), "case": f"F{fid:02d}-10D", "run_id": run_id,
                "events": pso_payloads["rank_top"]["csa_trace"],
            }))

    metrics = {
        "DE_rank_top": _summarize(de_rows["rank_top"], 2026071701),
        "DE_calibrated_regret": _summarize(de_rows["calibrated_regret"], 2026071702),
        "PSO_rank_top": _summarize(pso_rows["rank_top"], 2026071703),
        "PSO_calibrated_regret": _summarize(pso_rows["calibrated_regret"], 2026071704),
    }
    coverage = {
        "DE": _regret_model_metrics(
            de_candidate_records, protocol["regret_model"],
            float(protocol["conformal_residual_radius"]),
            float(protocol["target_coverage"]),
        ),
        "PSO": _regret_model_metrics(
            pso_candidate_records, protocol["regret_model"],
            float(protocol["conformal_residual_radius"]),
            float(protocol["target_coverage"]),
        ),
    }
    per_function = {}
    for fid in manifest["execution"]["test_fids"]:
        per_function[str(fid)] = {}
        for backbone, rows_by_policy in (("DE", de_rows), ("PSO", pso_rows)):
            for policy, rows in rows_by_policy.items():
                selected = [row for row in rows if int(row["fid"]) == int(fid)]
                per_function[str(fid)][f"{backbone}_{policy}"] = _summarize(
                    selected, 20260717 + int(fid) * 10 + len(policy)
                )
    nondegenerate = {
        f"{backbone}_{fid}": (
            per_function[str(fid)][f"{backbone}_calibrated_regret"]["semantic_evaluations"] > 0
            and per_function[str(fid)][f"{backbone}_calibrated_regret"]["abstention_rate"] < 1.0
        )
        for backbone in ("DE", "PSO")
        for fid in manifest["execution"]["test_fids"]
    }
    report = {
        "protocol_sha256": protocol["protocol_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "status": "complete",
        "proposal_capture": {
            "calls": int(protocol["runs_per_case"]) * len(manifest["execution"]["test_fids"])
            * int(protocol["proposal_capture"]["calls_per_run"]),
            "total_cost_usd": capture_cost,
            "total_tokens": capture_tokens,
            "labels_persisted": False,
        },
        "coverage": coverage,
        "aggregate": metrics,
        "per_function": per_function,
        "nondegenerate_admission": nondegenerate,
        "general_acquisition_claim_eligible": bool(
            all(nondegenerate.values())
            and all(value["empirical_coverage"] >= 0.80 for value in coverage.values())
        ),
        "claim_boundary": protocol["claim_rule"],
    }
    report["artifact_sha256"] = _sha256_json(report)
    _save_json(report, output_root / "prospective_report.json")
    return report


def validate(
    output_root: Path,
    protocol: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    function_checks = {}
    for fid in manifest["execution"]["test_fids"]:
        objective = get_function(int(fid), DIMENSION)
        probe = float(objective(np.zeros(DIMENSION, dtype=float)))
        function_checks[f"F{fid:02d}"] = {
            "available": bool(np.isfinite(probe)),
            "probe_finite": bool(np.isfinite(probe)),
        }
    key_present = bool(os.getenv(DEDICATED_KEY_ENV, "").strip())
    base_url_present = bool(os.getenv(DEDICATED_BASE_URL_ENV, "").strip())
    rotation_ack = os.getenv(ROTATION_ACK_ENV, "").strip().upper() == "YES"
    exposed_key_ack = os.getenv(EXPOSED_KEY_ACK_ENV, "").strip().upper() == "YES"
    expected_cache_paths = [
        _capture_path(output_root, int(fid), run_id)
        for fid in manifest["execution"]["test_fids"]
        for run_id in range(int(protocol["runs_per_case"]))
    ]
    valid_cache_count = 0
    invalid_cache_files = []
    for path in expected_cache_paths:
        if not path.exists():
            continue
        try:
            _verify_cache(path, protocol, manifest)
            valid_cache_count += 1
        except RuntimeError:
            invalid_cache_files.append(str(path))
    report = {
        "protocol_sha256_valid": True,
        "manifest_sha256_valid": True,
        "function_checks": function_checks,
        "dedicated_credentials": {
            "key_present": key_present,
            "base_url_present": base_url_present,
            "rotation_acknowledged": rotation_ack,
            "exposed_key_risk_acknowledged": exposed_key_ack,
            "capture_ready": (
                key_present and base_url_present and (rotation_ack or exposed_key_ack)
            ),
        },
        "proposal_cache": {
            "expected": len(expected_cache_paths),
            "present": len(list((output_root / "proposal_cache").glob("*.json"))),
            "valid_expected": valid_cache_count,
            "invalid_expected_files": invalid_cache_files,
        },
        "live_api_called": False,
        "execution_ready_without_capture": False,
    }
    report["execution_ready_without_capture"] = (
        report["proposal_cache"]["valid_expected"] == report["proposal_cache"]["expected"]
        and not report["proposal_cache"]["invalid_expected_files"]
    )
    _save_json(report, output_root / "validation_report.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("validate", "capture", "execute", "all"), default="validate")
    parser.add_argument(
        "--protocol",
        default=(
            "paper/q1/experiment_outputs/csa_regret_crossfit_20260717_complete/"
            "prospective_unseen_f02_f10_protocol.json"
        ),
    )
    parser.add_argument(
        "--output_root",
        default="paper/q1/experiment_outputs/csa_regret_prospective_f02_f10_20260717",
    )
    args = parser.parse_args()
    protocol = _load_and_verify_protocol(Path(args.protocol))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = _load_or_create_manifest(output_root, protocol)
    if args.mode == "validate":
        print(json.dumps(validate(output_root, protocol, manifest), indent=2))
        return
    if args.mode in {"capture", "all"}:
        capture_proposals(output_root, protocol, manifest)
    if args.mode in {"execute", "all"}:
        report = execute(output_root, protocol, manifest)
        print(json.dumps({
            "status": report["status"],
            "aggregate": report["aggregate"],
            "coverage": report["coverage"],
            "general_acquisition_claim_eligible": report["general_acquisition_claim_eligible"],
        }, indent=2))


if __name__ == "__main__":
    main()
