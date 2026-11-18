from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np

from safe_llm_saea.application_benchmark.common.io_utils import load_json, save_json
from safe_llm_saea.application_benchmark.uav.run_uav_benchmark import run_single_uav_case
from safe_llm_saea.application_benchmark.wsn.run_wsn_benchmark import run_single_wsn_case
from safe_llm_saea.cec2022 import get_bounds, get_function
from safe_llm_saea.csa_protocol import audit_csa_compliance
from safe_llm_saea.paper_ready_safe_llm_saea import ExperimentConfig, LLMAssistedDE


CEC_DESCRIPTIONS = {
    1: "Shifted Zakharov function (unimodal).",
    6: "Hybrid function combining heterogeneous landscape components.",
    9: "Composition function with multiple attraction basins.",
}


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _admission_summary(diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    events = [event for diagnostic in diagnostics for event in diagnostic.get("admission_events", [])]
    evaluated = sum(int(event.get("evaluated_candidates", 0)) for event in events)
    authorized = 0
    runtime_authorized = 0
    for event in events:
        count = int(event.get("evaluated_candidates", 0))
        selected = int(event.get("selected_candidates", 0))
        scores_present = all(
            len(event.get(key) or []) >= selected
            for key in ("surrogate_mu", "surrogate_sigma", "surrogate_lcb")
        )
        if (
            count
            and bool(event.get("surrogate_guidance_ready"))
            and str(event.get("surrogate_unavailable_policy")) == "reject"
            and not str(event.get("surrogate_fallback_reason") or "")
            and scores_present
        ):
            authorized += count
        certificates = list(event.get("evaluation_authorizations") or [])
        if count and bool(event.get("runtime_authorization_enforced")) and len(certificates) == count:
            runtime_authorized += count
    traces = [trace for diagnostic in diagnostics for trace in diagnostic.get("csa_trace", [])]
    trace_report = audit_csa_compliance({"strict_zero_call": False, "csa_trace": traces})
    return {
        "events": len(events),
        "semantic_evaluations": evaluated,
        "numerically_authorized": authorized,
        "runtime_authorized": runtime_authorized,
        "all_numerically_authorized": None if evaluated == 0 else evaluated == authorized,
        "all_runtime_authorized": None if evaluated == 0 else evaluated == runtime_authorized,
        "trace_compliant": bool(trace_report.get("csa_compliant", False)),
        "trace_violations": trace_report.get("violations", []),
    }


def _cec_config(fid: int, *, enabled: bool, max_fes: int, calls: int) -> ExperimentConfig:
    return ExperimentConfig(
        algorithm_name="L-ADE-CSA-closure" if enabled else "L-ADE-CSA-strict-zero",
        problem_name=f"CEC2022_F{fid:02d}",
        dimension=10,
        max_evaluations=max_fes,
        population_size=12,
        runs=1,
        llm_enhancement=enabled,
        gating_enabled=enabled,
        max_llm_calls_per_run=calls if enabled else 0,
        max_llm_cost_usd_per_run=0.02,
        fusion_weight=0.84,
        de_F=0.50,
        de_CR=0.90,
        min_remaining_budget=24,
        min_stagnation_generations=3,
        min_uncertainty_threshold=0.0,
        min_call_interval=3,
        min_confidence=0.0,
        force_initial_llm_call=enabled,
        max_llm_calls_stage1=calls if enabled else 0,
        max_llm_calls_stage2=0,
        llm_suggestions_k=3,
        problem_description=CEC_DESCRIPTIONS[fid],
        use_surrogate=True,
        enable_sniper=True,
        surrogate_min_history=12,
        surrogate_admission_policy="rank_top",
        surrogate_unavailable_policy="reject",
        enforce_full_evaluation_budget=True,
    )


def _run_cec(args: argparse.Namespace, llm_config: dict[str, Any], raw_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for fid in args.cec_fids:
        function = get_function(fid, 10)
        lo, hi = get_bounds(10)
        bounds = [(lo, hi)] * 10
        for run_id in range(args.runs):
            seed = args.seed_base + run_id
            for policy, enabled in (("fail_closed", True), ("strict_zero", False)):
                _seed(seed)
                optimizer = LLMAssistedDE(
                    _cec_config(fid, enabled=enabled, max_fes=args.cec_fes, calls=args.calls),
                    api_key=llm_config["api_key"],
                    base_url=llm_config["base_url"],
                    use_api=enabled,
                    llm_model=llm_config["model_id"],
                )
                result = optimizer.optimize(function, bounds)
                row = {
                    "domain": "CEC2022",
                    "case": f"F{fid:02d}-10D",
                    "run_id": run_id,
                    "seed": seed,
                    "policy": policy,
                    "objective": float(result["best_fitness"]),
                    "evaluations": int(result["evaluations_used"]),
                    "calls": int(result["stage1_calls"] + result["stage2_calls"]),
                    "successful_calls": int(result["api_success_calls"]),
                    "cost_usd": float(result["total_cost_usd"]),
                    "diagnostics": result["diagnostics"],
                }
                save_json(row, raw_dir / f"cec_f{fid:02d}_{policy}_r{run_id:02d}.json")
                rows.append(row)
    return rows


def _run_applications(args: argparse.Namespace, llm_config: dict[str, Any], raw_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    app_specs = [
        ("WSN", load_json(args.wsn_manifest)["cases"][: args.app_cases]),
        ("UAV", load_json(args.uav_manifest)["scenarios"][: args.app_cases]),
    ]
    budget = {
        "max_fes": args.app_fes,
        "population_size": 12,
        "max_iter": 10,
        "llm_call_budget": args.calls,
        "llm_cost_budget_usd": 0.02,
    }
    for domain, cases in app_specs:
        for case in cases:
            for run_id in range(args.runs):
                seed = args.seed_base + run_id
                for policy, enabled in (("fail_closed", True), ("strict_zero", False)):
                    variant = {
                        "strict_zero_call": not enabled,
                        "llm_enabled": enabled,
                        "surrogate_min_history": 12,
                        "surrogate_admission_policy": "rank_top",
                        "surrogate_unavailable_policy": "reject",
                        "min_remaining_budget": 24,
                        "force_initial_llm_call": enabled,
                        "max_unknown_nodes": args.wsn_unknowns,
                        "enforce_full_evaluation_budget": True,
                    }
                    label = f"L-ADE-CSA-{policy}"
                    if domain == "WSN":
                        record = run_single_wsn_case(
                            case_data=case,
                            algorithm_name="L-ADE-Sniper",
                            run_id=run_id,
                            seed=seed,
                            budget_config=budget,
                            llm_config={**llm_config, "enabled_algorithms": {"L-ADE-Sniper"}, "use_api": enabled},
                            output_dir=str(raw_dir),
                            algorithm_label=label,
                            algorithm_extra_config={"lade_variant": variant, "max_unknown_nodes": args.wsn_unknowns},
                        )
                        diagnostics = [item.get("diagnostics", {}) for item in record["artifacts"]["node_results"]]
                        objective = float(record["metrics"]["ale"])
                    else:
                        record = run_single_uav_case(
                            scenario=case,
                            algorithm_name="L-ADE-Sniper",
                            run_id=run_id,
                            seed=seed,
                            budget_config=budget,
                            llm_config={**llm_config, "enabled_algorithms": {"L-ADE-Sniper"}, "use_api": enabled},
                            output_dir=str(raw_dir),
                            algorithm_label=label,
                            algorithm_extra_config={"lade_variant": variant},
                        )
                        diagnostics = [record.get("diagnostics", {})]
                        objective = float(record["metrics"]["system_cost"])
                    rows.append({
                        "domain": domain,
                        "case": str(record["case_id"]),
                        "run_id": run_id,
                        "seed": seed,
                        "policy": policy,
                        "objective": objective,
                        "evaluations": int(record["runtime"]["objective_eval_count"]),
                        "calls": int(record["llm_stats"]["total_calls"]),
                        "successful_calls": int(record["llm_stats"].get("success_calls", 0)),
                        "cost_usd": float(record["llm_stats"].get("token_cost_usd", 0.0)),
                        "diagnostics": diagnostics,
                    })
    return rows


def _report(rows: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[float]]] = {}
    diagnostic_sets: list[dict[str, Any]] = []
    diagnostics_by_domain: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["domain"], {}).setdefault(row["policy"], []).append(float(row["objective"]))
        if row["policy"] == "fail_closed":
            diagnostics = row["diagnostics"] if isinstance(row["diagnostics"], list) else [row["diagnostics"]]
            diagnostic_sets.extend(diagnostics)
            diagnostics_by_domain.setdefault(row["domain"], []).extend(diagnostics)
    summary = {
        domain: {
            policy: {"n": len(values), "mean_objective": mean(values)}
            for policy, values in policies.items()
        }
        for domain, policies in grouped.items()
    }
    admission = _admission_summary(diagnostic_sets)
    domain_admission = {
        domain: _admission_summary(diagnostics)
        for domain, diagnostics in diagnostics_by_domain.items()
    }
    row_pairs: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["domain"]), str(row["case"]), int(row["run_id"]))
        row_pairs.setdefault(key, {})[str(row["policy"])] = row
    paired_summary: dict[str, dict[str, Any]] = {}
    for (domain, _, _), pair in row_pairs.items():
        if "fail_closed" not in pair or "strict_zero" not in pair:
            continue
        positive = float(pair["fail_closed"]["objective"])
        strict_zero = float(pair["strict_zero"]["objective"])
        item = paired_summary.setdefault(domain, {
            "paired_runs": 0,
            "wins": 0,
            "ties": 0,
            "losses": 0,
            "positive_objectives": [],
            "strict_zero_objectives": [],
            "paired_relative_changes_pct": [],
            "successful_calls": 0,
            "attempted_calls": 0,
            "api_cost_usd": 0.0,
        })
        item["paired_runs"] += 1
        item["positive_objectives"].append(positive)
        item["strict_zero_objectives"].append(strict_zero)
        item["paired_relative_changes_pct"].append(
            100.0 * (positive - strict_zero) / max(abs(strict_zero), 1e-12)
        )
        if positive < strict_zero - 1e-9:
            item["wins"] += 1
        elif positive > strict_zero + 1e-9:
            item["losses"] += 1
        else:
            item["ties"] += 1
        item["successful_calls"] += int(pair["fail_closed"]["successful_calls"])
        item["attempted_calls"] += int(pair["fail_closed"]["calls"])
        item["api_cost_usd"] += float(pair["fail_closed"]["cost_usd"])
    for domain, item in paired_summary.items():
        positive_values = item.pop("positive_objectives")
        strict_zero_values = item.pop("strict_zero_objectives")
        relative_changes = item.pop("paired_relative_changes_pct")
        item["positive_mean_objective"] = mean(positive_values)
        item["strict_zero_mean_objective"] = mean(strict_zero_values)
        item["mean_absolute_change"] = mean(
            positive - strict_zero
            for positive, strict_zero in zip(positive_values, strict_zero_values)
        )
        item["mean_relative_change_pct"] = 100.0 * (
            item["positive_mean_objective"] - item["strict_zero_mean_objective"]
        ) / max(abs(item["strict_zero_mean_objective"]), 1e-12)
        item["median_paired_relative_change_pct"] = median(relative_changes)
        item["admission"] = domain_admission[domain]
    closed = bool(
        admission["semantic_evaluations"] > 0
        and admission["all_numerically_authorized"]
        and admission["all_runtime_authorized"]
        and admission["trace_compliant"]
    )
    return {
        "closure_status": "closed" if closed else "blocked",
        "closure_blocker": "" if closed else "No successful positive-call semantic evaluation was available for authorization audit.",
        "protocol": protocol,
        "summary": summary,
        "paired_summary": paired_summary,
        "admission": admission,
        "total_api_cost_usd": sum(float(row["cost_usd"]) for row in rows),
        "records": [{key: value for key, value in row.items() if key != "diagnostics"} for row in rows],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Representative three-domain final-protocol closure.")
    parser.add_argument("--output_root", default="paper/q1/experiment_outputs/csa_representative_closure_v1")
    parser.add_argument("--wsn_manifest", default="safe_llm_saea/tmp/app_benchmarks/lade_proof_20260413_stage4_live/wsn/manifest.json")
    parser.add_argument("--uav_manifest", default="tmp/app_benchmarks/lade_proof_20260414_formal_default/uav/manifest.json")
    parser.add_argument("--cec_fids", nargs="+", type=int, default=[1, 6, 9])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--app_cases", type=int, default=2)
    parser.add_argument("--wsn_unknowns", type=int, default=4)
    parser.add_argument("--seed_base", type=int, default=20260716)
    parser.add_argument("--cec_fes", type=int, default=240)
    parser.add_argument("--app_fes", type=int, default=120)
    parser.add_argument("--calls", type=int, default=2)
    parser.add_argument("--base_url", default=os.getenv("Q1_GPT_AGENT_BASE_URL", "https://gpt-agent.cc/v1"))
    parser.add_argument("--model_id", default=os.getenv("Q1_GPT_AGENT_MODEL", "gpt-5.4"))
    parser.add_argument("--provider", default=os.getenv("Q1_LLM_PROVIDER", "OpenAI-compatible gateway"))
    args = parser.parse_args()

    api_key = os.getenv("GPT_AGENT_API_KEY", "")
    if not api_key:
        raise RuntimeError("GPT_AGENT_API_KEY is required for the positive-call closure runs")
    output_root = Path(args.output_root)
    raw_dir = output_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    llm_config = {"api_key": api_key, "base_url": args.base_url, "model_id": args.model_id, "llm_model": args.model_id}
    protocol = {
        "provider": args.provider,
        "base_url": args.base_url,
        "model_id": args.model_id,
        "api": "POST /v1/chat/completions",
        "temperature": 0.2,
        "max_tokens": 4096,
        "timeout_seconds": 60,
        "calls_per_optimizer_instance": args.calls,
        "wsn_optimizer_instances_per_case_run": args.wsn_unknowns,
        "surrogate_admission_policy": "rank_top",
        "surrogate_unavailable_policy": "reject",
        "runtime_authorization": "one-use host certificate",
    }
    rows = _run_cec(args, llm_config, raw_dir)
    rows.extend(_run_applications(args, llm_config, raw_dir))
    report = _report(rows, protocol)
    save_json(report, output_root / "representative_closure_report.json")
    print(json.dumps({"closure_status": report["closure_status"], "summary": report["summary"], "admission": report["admission"], "total_api_cost_usd": report["total_api_cost_usd"]}, indent=2))
    if report["closure_status"] != "closed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
