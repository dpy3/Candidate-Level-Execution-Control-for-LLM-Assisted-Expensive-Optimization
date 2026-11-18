from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

import numpy as np
from scipy.stats import wilcoxon

from safe_llm_saea.application_benchmark.common.io_utils import (
    load_json,
    make_run_filename,
    save_json,
)
from safe_llm_saea.application_benchmark.run_cached_proposal_replay import (
    CAPTURE_LABEL,
    STRICT_ZERO_LABEL,
    WEAK_REPLAY_LABEL,
    _budget,
    _llm_config,
    _validate_trace,
)
from safe_llm_saea.application_benchmark.run_lade_proof_experiments import (
    _proof_lade_variant,
)
from safe_llm_saea.application_benchmark.uav.aggregate_uav_results import (
    aggregate_uav_pipeline,
)
from safe_llm_saea.application_benchmark.uav.run_uav_benchmark import (
    run_single_uav_case,
)
from safe_llm_saea.csa_protocol import (
    audit_csa_compliance,
    verify_authorization_certificate,
)


FAIL_CLOSED_LABEL = "L-ADE-CSA-fail-closed-rank-top"


def _source_path(source_raw_dir: Path, case_id: str, run_id: int) -> Path:
    return source_raw_dir / make_run_filename("uav", case_id, CAPTURE_LABEL, run_id)


def _authorization_audit(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    evaluated = 0
    authorized = 0
    events = 0
    rejected_unavailable_events = 0
    runtime_authorized = 0
    violations: List[Dict[str, Any]] = []
    for record in records:
        record_id = str(record.get("record_id") or record.get("output_path") or "unknown")
        for event_index, event in enumerate(
            record.get("diagnostics", {}).get("admission_events", [])
        ):
            events += 1
            n_evaluated = int(event.get("evaluated_candidates", 0))
            evaluated += n_evaluated
            fallback_reason = str(event.get("surrogate_fallback_reason") or "")
            if fallback_reason.endswith("_rejected"):
                rejected_unavailable_events += 1
            scores_present = all(
                len(event.get(key) or []) >= int(event.get("selected_candidates", 0))
                for key in ("surrogate_mu", "surrogate_sigma", "surrogate_lcb")
            )
            numerically_authorized = (
                bool(event.get("surrogate_guidance_ready"))
                and str(event.get("surrogate_unavailable_policy")) == "reject"
                and not fallback_reason
                and scores_present
            )
            if n_evaluated and numerically_authorized:
                authorized += n_evaluated
            certificates = list(event.get("evaluation_authorizations", []))
            runtime_authorization_valid = (
                bool(event.get("runtime_authorization_enforced", False))
                and len(certificates) == n_evaluated
                and all(
                    isinstance(certificate, dict)
                    and verify_authorization_certificate(certificate)
                    for certificate in certificates
                )
            )
            if n_evaluated and runtime_authorization_valid:
                runtime_authorized += n_evaluated
            if n_evaluated and not numerically_authorized:
                violations.append(
                    {
                        "record": record_id,
                        "event": event_index,
                        "evaluated_candidates": n_evaluated,
                        "surrogate_guidance_ready": bool(
                            event.get("surrogate_guidance_ready")
                        ),
                        "surrogate_unavailable_policy": event.get(
                            "surrogate_unavailable_policy"
                        ),
                        "surrogate_fallback_reason": fallback_reason,
                        "scores_present": scores_present,
                    }
                )
            if n_evaluated and not runtime_authorization_valid:
                violations.append(
                    {
                        "record": record_id,
                        "event": event_index,
                        "type": "runtime_authorization_missing_or_invalid",
                        "evaluated_candidates": n_evaluated,
                        "certificates": len(certificates),
                    }
                )
    return {
        "admission_events": events,
        "semantic_evaluations": evaluated,
        "numerically_authorized_evaluations": authorized,
        "runtime_authorized_evaluations": runtime_authorized,
        "numerical_authorization_rate": 1.0 if evaluated == 0 else authorized / evaluated,
        "rejected_unavailable_events": rejected_unavailable_events,
        "violations": violations,
        "all_semantic_evaluations_numerically_authorized": not violations,
        "all_semantic_evaluations_runtime_authorized": (
            runtime_authorized == evaluated and not violations
        ),
    }


def _trace_audit(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    failures = []
    completeness = []
    for record in records:
        expected_resources = {
            "seed": int(record["seed"]),
            "max_fes": int(record["budget"]["max_fes"]),
            "evaluations_used": int(record["budget"]["max_fes"]),
        }
        observed_resources = {
            "seed": int(record["seed"]),
            "max_fes": int(record["budget"]["max_fes"]),
            "evaluations_used": int(record["runtime"]["objective_eval_count"]),
        }
        report = audit_csa_compliance(
            {
                "strict_zero_call": False,
                "csa_trace": record.get("diagnostics", {}).get("csa_trace", []),
                "resource_audit": {
                    "expected": expected_resources,
                    "observed": observed_resources,
                },
            }
        )
        completeness.append(float(report["trace_completeness"]))
        if not report["csa_compliant"]:
            failures.append(
                {
                    "case_id": record["case_id"],
                    "run_id": record["run_id"],
                    "violations": report["violations"],
                }
            )
    return {
        "records_checked": len(records),
        "mean_trace_completeness": mean(completeness) if completeness else 1.0,
        "failed_records": failures,
        "all_v2_traces_compliant": not failures,
    }


def _case_means(records: List[Dict[str, Any]], label: str) -> Dict[str, float]:
    grouped: Dict[str, List[float]] = {}
    for record in records:
        if str(record.get("algorithm")) != label:
            continue
        grouped.setdefault(str(record["case_id"]), []).append(
            float(record["metrics"]["system_cost"])
        )
    return {case_id: mean(values) for case_id, values in grouped.items()}


def _load_reference_records(raw_dir: Path, label: str) -> List[Dict[str, Any]]:
    return [
        load_json(path)
        for path in sorted(raw_dir.glob(f"uav_*_{label}_r*.json"))
    ]


def _fairness_audit(
    fail_closed: List[Dict[str, Any]], weak: List[Dict[str, Any]]
) -> Dict[str, Any]:
    weak_by_key = {
        (str(record["case_id"]), int(record["run_id"])): record for record in weak
    }
    mismatches: List[Dict[str, Any]] = []
    checked = 0
    for record in fail_closed:
        key = (str(record["case_id"]), int(record["run_id"]))
        reference = weak_by_key.get(key)
        if reference is None:
            mismatches.append({"case_id": key[0], "run_id": key[1], "field": "missing_pair"})
            continue
        checked += 1
        comparisons = {
            "seed": (record.get("seed"), reference.get("seed")),
            "budget": (record.get("budget"), reference.get("budget")),
            "objective_eval_count": (
                record.get("runtime", {}).get("objective_eval_count"),
                reference.get("runtime", {}).get("objective_eval_count"),
            ),
            "total_calls": (
                record.get("llm_stats", {}).get("total_calls"),
                reference.get("llm_stats", {}).get("total_calls"),
            ),
            "prompt_tokens": (
                record.get("llm_stats", {}).get("prompt_tokens"),
                reference.get("llm_stats", {}).get("prompt_tokens"),
            ),
            "completion_tokens": (
                record.get("llm_stats", {}).get("completion_tokens"),
                reference.get("llm_stats", {}).get("completion_tokens"),
            ),
            "proposal_source_sha256": (
                record.get("diagnostics", {}).get("proposal_replay_source_sha256"),
                reference.get("diagnostics", {}).get("proposal_replay_source_sha256"),
            ),
            "returned_candidate_vectors": (
                [
                    event.get("returned_candidate_vectors", [])
                    for event in record.get("diagnostics", {}).get("admission_events", [])
                ],
                [
                    event.get("returned_candidate_vectors", [])
                    for event in reference.get("diagnostics", {}).get("admission_events", [])
                ],
            ),
        }
        for field, (observed, expected) in comparisons.items():
            if observed != expected:
                mismatches.append(
                    {
                        "case_id": key[0],
                        "run_id": key[1],
                        "field": field,
                        "observed": observed,
                        "expected": expected,
                    }
                )
    return {
        "pairs_checked": checked,
        "mismatches": mismatches,
        "all_resources_and_proposals_matched": checked == len(fail_closed) and not mismatches,
    }


def _comparison(
    fail_closed: List[Dict[str, Any]], weak: List[Dict[str, Any]], strict_zero: List[Dict[str, Any]]
) -> Dict[str, Any]:
    fail_case = _case_means(fail_closed, FAIL_CLOSED_LABEL)
    weak_case = _case_means(weak, WEAK_REPLAY_LABEL)
    zero_case = _case_means(strict_zero, STRICT_ZERO_LABEL)
    common = sorted(set(fail_case) & set(weak_case) & set(zero_case))
    if not common:
        raise ValueError("No common UAV cases across fail-closed, weak, and strict-zero records")
    fail_values = np.asarray([fail_case[key] for key in common], dtype=float)
    weak_values = np.asarray([weak_case[key] for key in common], dtype=float)
    zero_values = np.asarray([zero_case[key] for key in common], dtype=float)
    test = wilcoxon(fail_values, weak_values, alternative="less", method="auto")
    return {
        "n_cases": len(common),
        "n_runs_per_setting": len(fail_closed),
        "mean_system_cost": {
            "fail_closed": float(np.mean(fail_values)),
            "weak_control": float(np.mean(weak_values)),
            "strict_zero": float(np.mean(zero_values)),
        },
        "case_mean_wins": {
            "fail_closed_vs_weak": int(np.sum(fail_values < weak_values)),
            "weak_vs_fail_closed": int(np.sum(weak_values < fail_values)),
            "ties": int(np.sum(fail_values == weak_values)),
        },
        "wilcoxon_fail_closed_better_than_weak": {
            "statistic": float(test.statistic),
            "pvalue": float(test.pvalue),
            "unit": "case mean",
        },
        "case_means": {
            key: {
                "fail_closed": fail_case[key],
                "weak_control": weak_case[key],
                "strict_zero": zero_case[key],
            }
            for key in common
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay cached UAV proposals under fail-closed numerical admission."
    )
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--source_raw_dir", required=True)
    parser.add_argument("--reference_raw_dir", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed_base", type=int, default=20260407)
    parser.add_argument("--max_fes", type=int, default=120)
    parser.add_argument("--population_size", type=int, default=12)
    parser.add_argument("--max_iter", type=int, default=8)
    parser.add_argument("--llm_call_budget", type=int, default=2)
    parser.add_argument("--llm_cost_budget_usd", type=float, default=0.05)
    parser.add_argument("--surrogate_min_history", type=int, default=16)
    parser.add_argument("--uav_proof_profile", choices=["default", "engaged"], default="default")
    parser.add_argument("--api_key", default="")
    parser.add_argument("--base_url", default="https://api.deepseek.com")
    parser.add_argument("--llm_model", default="deepseek-chat")
    args = parser.parse_args()

    manifest = load_json(args.manifest_path)
    scenarios = list(manifest.get("scenarios", []))
    seeds = [int(args.seed_base) + offset for offset in range(int(args.runs))]
    source_raw_dir = Path(args.source_raw_dir)
    reference_raw_dir = Path(args.reference_raw_dir)
    output_root = Path(args.output_root)
    raw_dir = output_root / "raw"
    records: List[Dict[str, Any]] = []

    for scenario in scenarios:
        case_id = str(scenario["case_id"])
        for run_id, seed in enumerate(seeds):
            source_path = _source_path(source_raw_dir, case_id, run_id)
            source = _validate_trace(source_path)
            if int(source.get("seed", -1)) != seed:
                raise ValueError(f"Seed mismatch for {source_path}")
            if dict(source.get("budget", {})) != _budget(args):
                raise ValueError(f"Budget mismatch for {source_path}")
            output_path = raw_dir / make_run_filename(
                "uav", case_id, FAIL_CLOSED_LABEL, run_id
            )
            if output_path.exists():
                records.append(load_json(output_path))
                continue
            variant = _proof_lade_variant(
                "uav",
                str(args.uav_proof_profile),
                weak_admission_control=False,
                proposal_trace_mode="replay",
                proposal_replay_path=str(source_path.resolve()),
                proposal_replay_strict=True,
                enforce_full_evaluation_budget=True,
                surrogate_min_history=int(args.surrogate_min_history),
                surrogate_acceptance_margin=0.01,
                surrogate_admission_policy="rank_top",
                surrogate_unavailable_policy="reject",
                use_api=False,
            )
            records.append(
                run_single_uav_case(
                    scenario=scenario,
                    algorithm_name="L-ADE-Sniper",
                    run_id=run_id,
                    seed=seed,
                    budget_config=_budget(args),
                    llm_config=_llm_config(args, use_api=False),
                    output_dir=str(raw_dir),
                    algorithm_label=FAIL_CLOSED_LABEL,
                    algorithm_extra_config={"lade_variant": variant},
                )
            )

    if len(records) != len(scenarios) * len(seeds):
        raise RuntimeError("Fail-closed replay did not produce the complete run matrix")
    if any(
        int(record.get("runtime", {}).get("objective_eval_count", -1))
        != int(args.max_fes)
        for record in records
    ):
        raise RuntimeError("Fail-closed replay did not preserve the real-evaluation budget")

    authorization = _authorization_audit(records)
    if not authorization["all_semantic_evaluations_numerically_authorized"]:
        raise RuntimeError("Fail-closed replay contains a semantic evaluation without numerical authorization")
    trace_audit = _trace_audit(records)
    if not trace_audit["all_v2_traces_compliant"]:
        raise RuntimeError("Fail-closed replay contains a non-compliant v2 candidate trace")

    weak = _load_reference_records(reference_raw_dir, WEAK_REPLAY_LABEL)
    strict_zero = _load_reference_records(reference_raw_dir, STRICT_ZERO_LABEL)
    fairness = _fairness_audit(records, weak)
    if not fairness["all_resources_and_proposals_matched"]:
        raise RuntimeError("Fail-closed and weak-control resources or proposals are not paired")
    comparison = _comparison(records, weak, strict_zero)
    aggregate_uav_pipeline(
        raw_dir=str(raw_dir),
        suite_output=str(output_root / "suite.json"),
        manifest_path=str(args.manifest_path),
        analysis_output=str(output_root / "analysis.json"),
        figures_dir=str(output_root / "figures"),
    )
    report = {
        "protocol_id": "CSA-UAV-FAIL-CLOSED-RUNTIME-GATE-2026-07-15",
        "policy": {
            "surrogate_admission_policy": "rank_top",
            "surrogate_unavailable_policy": "reject",
            "surrogate_min_history": int(args.surrogate_min_history),
        },
        "n_cases": len(scenarios),
        "n_runs": len(records),
        "seeds": seeds,
        "budget": _budget(args),
        "proposal_source": str(source_raw_dir.resolve()),
        "authorization_audit": authorization,
        "trace_audit": trace_audit,
        "fairness_audit": fairness,
        "semantic_evaluations": authorization["semantic_evaluations"],
        "comparison": comparison,
    }
    save_json(report, output_root / "fail_closed_report.json")
    print(report)


if __name__ == "__main__":
    main()
