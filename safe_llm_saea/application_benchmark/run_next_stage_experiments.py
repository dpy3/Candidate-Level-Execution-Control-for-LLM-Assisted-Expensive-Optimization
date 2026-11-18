from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from safe_llm_saea.application_benchmark.common.io_utils import save_json
from safe_llm_saea.application_benchmark.uav.aggregate_uav_results import aggregate_uav_pipeline
from safe_llm_saea.application_benchmark.uav.build_uav_suite import build_uav_suite_manifest
from safe_llm_saea.application_benchmark.uav.run_uav_benchmark import run_uav_benchmark
from safe_llm_saea.application_benchmark.wsn.aggregate_wsn_results import aggregate_wsn_pipeline
from safe_llm_saea.application_benchmark.wsn.build_wsn_suite import build_wsn_suite_manifest
from safe_llm_saea.application_benchmark.wsn.run_wsn_benchmark import run_wsn_benchmark

_DEFAULT_WSN_ALGORITHMS = ["L-ADE-Sniper", "LLM-DE", "LLM-SAEA", "DE", "PSO", "SHADE", "DV-Hop"]
_DEFAULT_UAV_ALGORITHMS = ["L-ADE-Sniper", "LLM-DE", "LLM-SAEA", "DE", "PSO", "SHADE", "RRT"]
_DEFAULT_BOUNDS = {"x": [-10.0, 190.0], "y": [-10.0, 190.0], "z": [5.0, 40.0]}


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_output_root() -> Path:
    return _repository_root() / "tmp" / "app_benchmarks" / "next_stage"


def _default_uav_source_dir() -> Path:
    return _repository_root() / "safe_llm_saea" / "paper_results"


def _budget_config(
    max_fes: int,
    population_size: int,
    max_iter: int,
    llm_call_budget: int,
    llm_cost_budget_usd: float,
) -> Dict[str, Any]:
    return {
        "max_fes": int(max_fes),
        "population_size": int(population_size),
        "max_iter": int(max_iter),
        "llm_call_budget": int(llm_call_budget),
        "llm_cost_budget_usd": float(llm_cost_budget_usd),
    }


def _llm_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "enabled_algorithms": {"L-ADE-Sniper", "LLM-DE", "LLM-SAEA"},
        "use_api": bool(args.use_api),
        "api_key": str(args.api_key),
        "base_url": str(args.base_url),
        "llm_model": str(args.llm_model),
    }


def _extract_wsn_key_metrics(analysis: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    key_metrics: Dict[str, Dict[str, float]] = {}
    for algorithm, metrics in analysis.get("algorithm_summary", {}).items():
        key_metrics[str(algorithm)] = {
            "mean_ale": float(metrics.get("mean_ale", 0.0)),
            "mean_rmse": float(metrics.get("mean_rmse", 0.0)),
            "mean_max_error": float(metrics.get("mean_max_error", 0.0)),
            "mean_localized_ratio": float(metrics.get("mean_localized_ratio", 0.0)),
            "mean_anchor_distance_rmse": float(metrics.get("mean_anchor_distance_rmse", 0.0)),
            "mean_anchor_distance_mae": float(metrics.get("mean_anchor_distance_mae", 0.0)),
            "mean_relative_distance_error": float(metrics.get("mean_relative_distance_error", 0.0)),
            "nearest_anchor_match_ratio": float(metrics.get("nearest_anchor_match_ratio", 0.0)),
            "one_hop_anchor_violation_rate": float(metrics.get("one_hop_anchor_violation_rate", 0.0)),
            "multi_hop_anchor_shortcut_rate": float(metrics.get("multi_hop_anchor_shortcut_rate", 0.0)),
            "finite_anchor_support_ratio": float(metrics.get("finite_anchor_support_ratio", 0.0)),
        }
    return key_metrics


def _extract_uav_key_metrics(analysis: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    key_metrics: Dict[str, Dict[str, float]] = {}
    for algorithm, metrics in analysis.get("algorithm_summary", {}).items():
        key_metrics[str(algorithm)] = {
            "mean_system_cost": float(metrics.get("mean_system_cost", 0.0)),
            "success_rate": float(metrics.get("success_rate", 0.0)),
            "collision_free_rate": float(metrics.get("collision_free_rate", 0.0)),
            "avg_collision_count": float(metrics.get("avg_collision_count", 0.0)),
            "mean_min_clearance": float(metrics.get("mean_min_clearance", 0.0)),
            "mean_clearance_deficit": float(metrics.get("mean_clearance_deficit", 0.0)),
            "mean_time_to_first_feasible_eval": float(metrics.get("mean_time_to_first_feasible_eval", 0.0)),
            "mean_path_length": float(metrics.get("mean_path_length", 0.0)),
            "mean_path_efficiency": float(metrics.get("mean_path_efficiency", 0.0)),
            "mean_curvature_cost": float(metrics.get("mean_curvature_cost", 0.0)),
            "mean_altitude_variation": float(metrics.get("mean_altitude_variation", 0.0)),
        }
    return key_metrics


def run_next_stage_experiments(args: argparse.Namespace) -> Dict[str, Any]:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    budget_config = _budget_config(
        max_fes=args.max_fes,
        population_size=args.population_size,
        max_iter=args.max_iter,
        llm_call_budget=args.llm_call_budget,
        llm_cost_budget_usd=args.llm_cost_budget_usd,
    )
    llm_config = _llm_config(args)
    seeds = [int(args.seed_base) + offset for offset in range(int(args.runs))]
    summary: Dict[str, Any] = {
        "applications": [str(name) for name in args.applications],
        "runs": int(args.runs),
        "seed_base": int(args.seed_base),
        "budget_config": budget_config,
        "llm_config": {
            "enabled_algorithms": sorted(llm_config["enabled_algorithms"]),
            "use_api": bool(llm_config["use_api"]),
            "base_url": llm_config["base_url"],
            "llm_model": llm_config["llm_model"],
        },
    }

    if "wsn" in args.applications:
        wsn_root = output_root / "wsn"
        manifest_path = wsn_root / "manifest.json"
        raw_dir = wsn_root / "raw"
        suite_output = wsn_root / "suite.json"
        analysis_output = wsn_root / "analysis.json"
        figures_dir = wsn_root / "figures"
        build_wsn_suite_manifest(
            node_counts=[int(value) for value in args.wsn_node_counts],
            anchor_counts=[int(value) for value in args.wsn_anchor_counts],
            comm_radii=[float(value) for value in args.wsn_comm_radii],
            cases_per_setting=int(args.wsn_cases_per_setting),
            area_size=float(args.wsn_area_size),
            seed_base=int(args.seed_base),
            output_path=str(manifest_path),
        )
        run_wsn_benchmark(
            manifest_path=str(manifest_path),
            algorithms=[str(name) for name in args.wsn_algorithms],
            seeds=seeds,
            budget_config=budget_config,
            llm_config=llm_config,
            output_dir=str(raw_dir),
        )
        wsn_result = aggregate_wsn_pipeline(
            raw_dir=str(raw_dir),
            suite_output=str(suite_output),
            manifest_path=str(manifest_path),
            analysis_output=str(analysis_output),
            figures_dir=str(figures_dir),
        )
        wsn_analysis = wsn_result.get("analysis") or {}
        summary["wsn"] = {
            "manifest_path": str(manifest_path),
            "raw_dir": str(raw_dir),
            "suite_output": str(suite_output),
            "analysis_output": str(analysis_output),
            "figures_dir": str(figures_dir),
            "algorithms": [str(name) for name in args.wsn_algorithms],
            "settings": {
                "node_counts": [int(value) for value in args.wsn_node_counts],
                "anchor_counts": [int(value) for value in args.wsn_anchor_counts],
                "comm_radii": [float(value) for value in args.wsn_comm_radii],
                "cases_per_setting": int(args.wsn_cases_per_setting),
                "area_size": float(args.wsn_area_size),
            },
            "suite_summary": wsn_analysis.get("suite_summary", {}),
            "pairwise_significance": wsn_analysis.get("pairwise_significance", {}),
            "key_metrics": _extract_wsn_key_metrics(wsn_analysis),
        }

    if "uav" in args.applications:
        uav_root = output_root / "uav"
        manifest_path = uav_root / "manifest.json"
        raw_dir = uav_root / "raw"
        suite_output = uav_root / "suite.json"
        analysis_output = uav_root / "analysis.json"
        figures_dir = uav_root / "figures"
        build_uav_suite_manifest(
            source_dir=str(args.uav_source_dir),
            output_path=str(manifest_path),
            levels=[str(level) for level in args.uav_levels],
            control_point_dim=int(args.uav_control_point_dim),
            bounds={axis: [float(value) for value in bounds] for axis, bounds in _DEFAULT_BOUNDS.items()},
            synthetic_hard_cases=int(args.uav_synthetic_hard_cases),
        )
        run_uav_benchmark(
            manifest_path=str(manifest_path),
            algorithms=[str(name) for name in args.uav_algorithms],
            seeds=seeds,
            budget_config=budget_config,
            llm_config=llm_config,
            output_dir=str(raw_dir),
        )
        uav_result = aggregate_uav_pipeline(
            raw_dir=str(raw_dir),
            suite_output=str(suite_output),
            manifest_path=str(manifest_path),
            analysis_output=str(analysis_output),
            figures_dir=str(figures_dir),
        )
        uav_analysis = uav_result.get("analysis") or {}
        summary["uav"] = {
            "manifest_path": str(manifest_path),
            "raw_dir": str(raw_dir),
            "suite_output": str(suite_output),
            "analysis_output": str(analysis_output),
            "figures_dir": str(figures_dir),
            "algorithms": [str(name) for name in args.uav_algorithms],
            "settings": {
                "source_dir": str(args.uav_source_dir),
                "levels": [str(level) for level in args.uav_levels],
                "control_point_dim": int(args.uav_control_point_dim),
                "synthetic_hard_cases": int(args.uav_synthetic_hard_cases),
            },
            "suite_summary": uav_analysis.get("suite_summary", {}),
            "system_cost_suite_summary": uav_analysis.get("system_cost_suite_summary", {}),
            "pairwise_significance": uav_analysis.get("pairwise_significance", {}),
            "safety_significance": uav_analysis.get("safety_significance", {}),
            "safety_priority_advantage": uav_analysis.get("safety_priority_advantage", {}),
            "key_metrics": _extract_uav_key_metrics(uav_analysis),
        }

    summary_path = output_root / "next_stage_summary.json"
    summary["summary_path"] = str(summary_path)
    save_json(summary, summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_root",
        default=str(_default_output_root()),
    )
    parser.add_argument(
        "--applications",
        nargs="+",
        choices=["wsn", "uav"],
        default=["wsn", "uav"],
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed_base", type=int, default=20260407)
    parser.add_argument("--max_fes", type=int, default=300)
    parser.add_argument("--population_size", type=int, default=20)
    parser.add_argument("--max_iter", type=int, default=15)
    parser.add_argument("--llm_call_budget", type=int, default=4)
    parser.add_argument("--llm_cost_budget_usd", type=float, default=0.01)
    parser.add_argument("--use_api", action="store_true")
    parser.add_argument("--api_key", default="")
    parser.add_argument("--base_url", default="https://api.deepseek.com")
    parser.add_argument("--llm_model", default="deepseek-chat")
    parser.add_argument("--wsn_node_counts", nargs="+", type=int, default=[30, 50, 70])
    parser.add_argument("--wsn_anchor_counts", nargs="+", type=int, default=[6, 8, 10])
    parser.add_argument("--wsn_comm_radii", nargs="+", type=float, default=[25.0, 30.0, 35.0])
    parser.add_argument("--wsn_cases_per_setting", type=int, default=2)
    parser.add_argument("--wsn_area_size", type=float, default=100.0)
    parser.add_argument("--wsn_algorithms", nargs="+", default=list(_DEFAULT_WSN_ALGORITHMS))
    parser.add_argument("--uav_source_dir", default=str(_default_uav_source_dir()))
    parser.add_argument("--uav_levels", nargs="+", default=["easy", "medium", "hard"])
    parser.add_argument("--uav_control_point_dim", type=int, default=24)
    parser.add_argument("--uav_synthetic_hard_cases", type=int, default=8)
    parser.add_argument("--uav_algorithms", nargs="+", default=list(_DEFAULT_UAV_ALGORITHMS))
    args = parser.parse_args()
    run_next_stage_experiments(args)


if __name__ == "__main__":
    main()
