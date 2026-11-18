from __future__ import annotations

import argparse
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from safe_llm_saea.application_benchmark.common.algorithm_registry import run_algorithm
from safe_llm_saea.application_benchmark.common.io_utils import load_json, make_run_filename, save_json
from safe_llm_saea.application_benchmark.uav.aggregate_uav_results import aggregate_uav_pipeline

_UAV_ALGORITHM_PRESETS = {
    "recommended_rrt": ["L-ADE-Sniper", "LLM-DE", "LLM-SAEA", "DE", "PSO", "SHADE", "RRT"],
    "recommended_astar": ["L-ADE-Sniper", "LLM-DE", "LLM-SAEA", "DE", "PSO", "SHADE", "A*"],
    "llm_plus_classic": ["L-ADE-Sniper", "LLM-DE", "LLM-SAEA", "DE", "PSO", "SHADE"],
    "classic_baselines": ["DE", "PSO", "SHADE", "RRT"],
}


def load_uav_suite_manifest(manifest_path: str) -> Dict[str, Any]:
    manifest = load_json(manifest_path)
    if manifest.get("application") != "uav":
        raise ValueError("Manifest is not a UAV suite")
    return manifest


def build_uav_objective_bundle(scenario: Dict[str, Any]) -> Dict[str, Any]:
    reference_path = scenario.get("reference_path") or [scenario["start"], scenario["goal"]]
    metrics = evaluate_uav_path(reference_path, scenario)
    return {
        "application": "uav",
        "case_id": scenario["case_id"],
        "scenario": scenario,
        "placeholder_objective": metrics["path_length"],
        "default_artifacts": {"path_points": reference_path},
        "reference_metrics": metrics,
    }


def evaluate_uav_path(path_points: List[List[float]], scenario: Dict[str, Any]) -> Dict[str, Any]:
    if len(path_points) < 2:
        path_points = [scenario["start"], scenario["goal"]]
    points = np.asarray(path_points, dtype=float)
    path_length = 0.0
    min_clearance = float("inf")
    collision_count = 0
    collision_depth = 0.0
    for left, right in zip(points[:-1], points[1:]):
        path_length += math.dist(left.tolist(), right.tolist())
        segment_length = float(np.linalg.norm(right - left))
        sample_count = max(3, int(segment_length / 6.0) + 1)
        for alpha in np.linspace(0.0, 1.0, sample_count):
            point = left * (1.0 - alpha) + right * alpha
            for obstacle in scenario.get("obstacles", []):
                if obstacle.get("type") == "sphere":
                    center = np.asarray(obstacle["center"], dtype=float)
                    radius = float(obstacle["radius"])
                    clearance = float(np.linalg.norm(point - center) - radius)
                    min_clearance = min(min_clearance, clearance)
                    if clearance < 0:
                        collision_count += 1
                        collision_depth += abs(clearance)
    if min_clearance == float("inf"):
        min_clearance = 10.0
    smoothness_cost = 0.0
    for index in range(1, len(points) - 1):
        left = points[index] - points[index - 1]
        right = points[index + 1] - points[index]
        left_norm = float(np.linalg.norm(left))
        right_norm = float(np.linalg.norm(right))
        if left_norm < 1e-12 or right_norm < 1e-12:
            continue
        cosine = float(np.clip(np.dot(left, right) / (left_norm * right_norm), -1.0, 1.0))
        smoothness_cost += math.acos(cosine) ** 2
    curvature_cost = smoothness_cost / max(1, len(path_points) - 2)
    altitude_variation = float(np.max(points[:, 2]) - np.min(points[:, 2]))
    straight_line_length = float(math.dist(points[0].tolist(), points[-1].tolist()))
    path_efficiency = float(path_length / max(straight_line_length, 1e-6))
    clearance_penalty = max(0.0, 3.0 - min_clearance) ** 2
    system_cost = (
        path_length
        + 28.0 * smoothness_cost
        + 6.0 * altitude_variation
        + 140.0 * clearance_penalty
        + 1e6 * collision_count
        + 5e4 * collision_depth
    )
    return {
        "system_cost": float(system_cost),
        "path_length": float(path_length),
        "straight_line_length": straight_line_length,
        "path_efficiency": path_efficiency,
        "collision_count": int(collision_count),
        "min_clearance": float(min_clearance),
        "clearance_deficit": float(max(0.0, 3.0 - min_clearance)),
        "smoothness_cost": float(smoothness_cost),
        "curvature_cost": float(curvature_cost),
        "altitude_variation": float(altitude_variation),
        "time_to_first_feasible_eval": 0 if collision_count == 0 else int(1e9),
        "success_under_strict_rule": bool(collision_count == 0),
        "success_under_viz_rule": bool(collision_count < 5),
    }


def _estimate_first_feasible_eval(
    path_history: List[List[List[float]]],
    scenario: Dict[str, Any],
    objective_eval_count: int,
    final_metrics: Dict[str, Any],
) -> int:
    if path_history:
        history_length = max(1, len(path_history))
        for index, path_points in enumerate(path_history):
            if evaluate_uav_path(path_points, scenario)["success_under_strict_rule"]:
                return max(1, int(math.ceil((index + 1) * max(1, objective_eval_count) / history_length)))
    if final_metrics.get("success_under_strict_rule"):
        return 0
    return int(1e9)


def run_single_uav_case(
    scenario: Dict[str, Any],
    algorithm_name: str,
    run_id: int,
    seed: int,
    budget_config: Dict[str, Any],
    llm_config: Dict[str, Any],
    output_dir: str,
    algorithm_label: str | None = None,
    algorithm_extra_config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    objective_bundle = build_uav_objective_bundle(scenario)
    algorithm_result = run_algorithm(
        algorithm_name=algorithm_name,
        objective_bundle=objective_bundle,
        budget_config=budget_config,
        seed=seed,
        llm_config=llm_config,
        extra_config={
            "application": "uav",
            "case_id": scenario["case_id"],
            **deepcopy(algorithm_extra_config or {}),
        },
    )
    path_points = deepcopy(algorithm_result.get("artifacts", {}).get("path_points", objective_bundle["default_artifacts"]["path_points"]))
    best_path_history = deepcopy(algorithm_result.get("artifacts", {}).get("best_path_history", []))
    metrics = evaluate_uav_path(path_points, scenario)
    metrics["time_to_first_feasible_eval"] = int(
        _estimate_first_feasible_eval(
            path_history=best_path_history,
            scenario=scenario,
            objective_eval_count=int(algorithm_result["runtime"].get("objective_eval_count", 0)),
            final_metrics=metrics,
        )
    )
    record = {
        "application": "uav",
        "suite_name": f"uav_{scenario['level']}_suite",
        "case_id": scenario["case_id"],
        "level": scenario["level"],
        "scenario_id": scenario["scenario_id"],
        "algorithm": algorithm_label or algorithm_name,
        "algorithm_family": "llm_ea" if llm_config.get("enabled_algorithms") and algorithm_name in llm_config["enabled_algorithms"] else "classic_or_reference",
        "run_id": int(run_id),
        "seed": int(seed),
        "budget": deepcopy(budget_config),
        "status": algorithm_result["status"],
        "feasible": metrics["success_under_strict_rule"],
        "best_objective": float(metrics["system_cost"]),
        "metrics": metrics,
        "runtime": algorithm_result["runtime"],
        "llm_stats": algorithm_result["llm_stats"],
        "diagnostics": deepcopy(algorithm_result.get("diagnostics", {})),
        "convergence_curve": algorithm_result["convergence_curve"],
        "artifacts": {
            "path_points": path_points,
            "control_points": path_points[1:-1],
            "best_solution": deepcopy(algorithm_result.get("artifacts", {}).get("best_solution", [])),
            "best_solution_history": deepcopy(algorithm_result.get("artifacts", {}).get("best_solution_history", [])),
            "best_path_history": best_path_history,
            "scenario_meta_path": "",
        },
    }
    filename = make_run_filename("uav", scenario["case_id"], algorithm_label or algorithm_name, run_id)
    save_json(record, f"{output_dir}/{filename}")
    return record


def run_uav_benchmark(
    manifest_path: str,
    algorithms: List[str],
    seeds: List[int],
    budget_config: Dict[str, Any],
    llm_config: Dict[str, Any],
    output_dir: str,
) -> List[Dict[str, Any]]:
    manifest = load_uav_suite_manifest(manifest_path)
    records: List[Dict[str, Any]] = []
    output_path = Path(output_dir)
    for scenario in manifest["scenarios"]:
        for algorithm_name in algorithms:
            for run_id, seed in enumerate(seeds):
                filename = make_run_filename("uav", scenario["case_id"], algorithm_name, run_id)
                if (output_path / filename).exists():
                    continue
                records.append(
                    run_single_uav_case(
                        scenario=scenario,
                        algorithm_name=algorithm_name,
                        run_id=run_id,
                        seed=seed,
                        budget_config=budget_config,
                        llm_config=llm_config,
                        output_dir=output_dir,
                    )
                )
    return records


def resolve_uav_algorithms(algorithms: List[str] | None, algorithm_preset: str | None) -> List[str]:
    ordered: List[str] = []
    for algorithm_name in (algorithms or []) + _UAV_ALGORITHM_PRESETS.get(str(algorithm_preset), []):
        if algorithm_name not in ordered:
            ordered.append(algorithm_name)
    if not ordered:
        raise ValueError("Provide --algorithms or --algorithm_preset for UAV benchmarking")
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--algorithms", nargs="+")
    parser.add_argument("--algorithm_preset", choices=sorted(_UAV_ALGORITHM_PRESETS))
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed_base", type=int, default=20260404)
    parser.add_argument("--max_fes", type=int, default=300)
    parser.add_argument("--population_size", type=int, default=20)
    parser.add_argument("--max_iter", type=int, default=15)
    parser.add_argument("--llm_call_budget", type=int, default=4)
    parser.add_argument("--llm_cost_budget_usd", type=float, default=0.01)
    parser.add_argument("--use_api", action="store_true")
    parser.add_argument("--api_key", default="")
    parser.add_argument("--base_url", default="https://api.deepseek.com")
    parser.add_argument("--llm_model", default="deepseek-chat")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--suite_output", default="")
    parser.add_argument("--analysis_output", default="")
    parser.add_argument("--figures_dir", default="")
    args = parser.parse_args()

    algorithms = resolve_uav_algorithms(args.algorithms, args.algorithm_preset)
    seeds = [args.seed_base + i for i in range(args.runs)]
    budget_config = {
        "max_fes": args.max_fes,
        "population_size": args.population_size,
        "max_iter": args.max_iter,
        "llm_call_budget": args.llm_call_budget,
        "llm_cost_budget_usd": args.llm_cost_budget_usd,
    }
    llm_config = {
        "enabled_algorithms": {"L-ADE-Sniper", "LLM-DE", "LLM-SAEA"},
        "use_api": bool(args.use_api),
        "api_key": args.api_key,
        "base_url": args.base_url,
        "llm_model": args.llm_model,
    }
    run_uav_benchmark(
        manifest_path=args.manifest,
        algorithms=algorithms,
        seeds=seeds,
        budget_config=budget_config,
        llm_config=llm_config,
        output_dir=args.output_dir,
    )
    if args.suite_output or args.analysis_output or args.figures_dir:
        aggregate_uav_pipeline(
            raw_dir=args.output_dir,
            suite_output=args.suite_output or f"{args.output_dir}/suite.json",
            manifest_path=args.manifest,
            analysis_output=args.analysis_output,
            figures_dir=args.figures_dir,
        )


if __name__ == "__main__":
    main()
