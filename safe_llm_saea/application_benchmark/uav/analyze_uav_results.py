from __future__ import annotations

import argparse
from collections import defaultdict
from statistics import mean
from typing import Any, Dict

from safe_llm_saea.application_benchmark.common.io_utils import load_json, save_json
from safe_llm_saea.application_benchmark.common.statistics import (
    build_rank_table,
    build_reference_advantage,
    build_suite_summary,
    run_pairwise_significance,
    summarize_by_case,
)


def _build_algorithm_summary(records: list[Dict[str, Any]], algorithms: list[str]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for algorithm in algorithms:
        algo_records = [record for record in records if record["algorithm"] == algorithm]
        if not algo_records:
            continue
        collision_free_count = sum(1 for record in algo_records if int(record["metrics"].get("collision_count", 0)) == 0)
        summary[algorithm] = {
            "mean_system_cost": float(mean(record["metrics"]["system_cost"] for record in algo_records)),
            "mean_path_length": float(mean(record["metrics"].get("path_length", 0.0) for record in algo_records)),
            "mean_path_efficiency": float(mean(record["metrics"].get("path_efficiency", 0.0) for record in algo_records)),
            "mean_min_clearance": float(mean(record["metrics"].get("min_clearance", 0.0) for record in algo_records)),
            "mean_clearance_deficit": float(mean(record["metrics"].get("clearance_deficit", 0.0) for record in algo_records)),
            "mean_curvature_cost": float(mean(record["metrics"].get("curvature_cost", 0.0) for record in algo_records)),
            "mean_altitude_variation": float(mean(record["metrics"].get("altitude_variation", 0.0) for record in algo_records)),
            "mean_time_to_first_feasible_eval": float(
                mean(record["metrics"].get("time_to_first_feasible_eval", 0) for record in algo_records)
            ),
            "success_rate": float(
                sum(1 for record in algo_records if record["metrics"]["success_under_strict_rule"]) / len(algo_records)
            ),
            "collision_free_rate": float(collision_free_count / len(algo_records)),
            "avg_collision_count": float(mean(record["metrics"].get("collision_count", 0) for record in algo_records)),
            "avg_runtime_sec": float(mean(record["runtime"].get("wall_clock_sec", 0.0) for record in algo_records)),
            "avg_eval_count": float(mean(record["runtime"].get("objective_eval_count", 0) for record in algo_records)),
            "avg_llm_calls": float(mean(record["llm_stats"].get("total_calls", 0) for record in algo_records)),
        }
    return summary


def _build_level_breakdown(
    records: list[Dict[str, Any]],
    algorithms: list[str],
    levels: list[str],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    breakdown: Dict[str, Dict[str, Dict[str, float]]] = {}
    for level in levels:
        level_records = [record for record in records if record.get("level") == level]
        breakdown[level] = _build_algorithm_summary(level_records, algorithms)
    return breakdown


def _build_per_case_mean_cost(summary_by_case: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    per_case_cost: Dict[str, Dict[str, float]] = {}
    for case_id, case_summary in summary_by_case.items():
        per_case_cost[case_id] = {
            algorithm: float(metrics["mean"])
            for algorithm, metrics in case_summary.items()
        }
    return per_case_cost


def _group_case_algorithm_records(records: list[Dict[str, Any]]) -> Dict[str, Dict[str, list[Dict[str, Any]]]]:
    grouped: Dict[str, Dict[str, list[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        grouped[str(record["case_id"])][str(record["algorithm"])].append(record)
    return grouped


def _build_safety_priority_case_summary(records: list[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    grouped = _group_case_algorithm_records(records)
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for case_id, algorithm_records in grouped.items():
        summary[case_id] = {}
        for algorithm, algo_records in algorithm_records.items():
            success_rate = float(
                sum(1 for record in algo_records if record["metrics"].get("success_under_strict_rule")) / len(algo_records)
            )
            collision_free_rate = float(
                sum(1 for record in algo_records if int(record["metrics"].get("collision_count", 0)) == 0) / len(algo_records)
            )
            avg_collision_count = float(mean(record["metrics"].get("collision_count", 0) for record in algo_records))
            mean_clearance_deficit = float(mean(record["metrics"].get("clearance_deficit", 0.0) for record in algo_records))
            mean_min_clearance = float(mean(record["metrics"].get("min_clearance", 0.0) for record in algo_records))
            mean_time_to_first_feasible_eval = float(
                mean(record["metrics"].get("time_to_first_feasible_eval", 0.0) for record in algo_records)
            )
            mean_system_cost = float(mean(record["metrics"].get("system_cost", 0.0) for record in algo_records))
            mean_path_efficiency = float(mean(record["metrics"].get("path_efficiency", 0.0) for record in algo_records))
            mean_path_length = float(mean(record["metrics"].get("path_length", 0.0) for record in algo_records))
            summary[case_id][algorithm] = {
                "strict_failure_rate": float(1.0 - success_rate),
                "collision_free_rate": collision_free_rate,
                "avg_collision_count": avg_collision_count,
                "mean_clearance_deficit": mean_clearance_deficit,
                "mean_min_clearance": mean_min_clearance,
                "mean_time_to_first_feasible_eval": mean_time_to_first_feasible_eval,
                "mean_system_cost": mean_system_cost,
                "mean_path_efficiency": mean_path_efficiency,
                "mean_path_length": mean_path_length,
            }
    return summary


def _safety_priority_sort_key(metrics: Dict[str, float]) -> tuple[float, float, float, float, float, float, float, float]:
    return (
        float(metrics.get("strict_failure_rate", 1.0)),
        float(metrics.get("avg_collision_count", 0.0)),
        float(metrics.get("mean_clearance_deficit", 0.0)),
        float(metrics.get("mean_system_cost", 0.0)),
        float(metrics.get("mean_time_to_first_feasible_eval", 0.0)),
        float(metrics.get("mean_path_efficiency", 0.0)),
        float(metrics.get("mean_path_length", 0.0)),
        -float(metrics.get("mean_min_clearance", 0.0)),
    )


def _build_safety_priority_rank_table(
    safety_summary_by_case: Dict[str, Dict[str, Dict[str, float]]]
) -> Dict[str, Any]:
    case_ranks: Dict[str, Dict[str, float]] = {}
    average_rank_map: Dict[str, list[float]] = defaultdict(list)
    for case_id, case_summary in safety_summary_by_case.items():
        ordered = sorted(
            case_summary.items(),
            key=lambda item: _safety_priority_sort_key(item[1]),
        )
        if not ordered:
            continue
        case_ranks[case_id] = {}
        for rank_index, (algorithm, _) in enumerate(ordered, start=1):
            case_ranks[case_id][algorithm] = float(rank_index)
            average_rank_map[algorithm].append(float(rank_index))
    average_rank = {
        algorithm: float(mean(ranks))
        for algorithm, ranks in average_rank_map.items()
        if ranks
    }
    return {
        "primary_metric": "safety_priority_order",
        "ranking_policy": [
            "strict_failure_rate",
            "avg_collision_count",
            "mean_clearance_deficit",
            "mean_system_cost",
            "mean_time_to_first_feasible_eval",
            "mean_path_efficiency",
            "mean_path_length",
            "mean_min_clearance_desc",
        ],
        "case_ranks": case_ranks,
        "average_rank": average_rank,
    }


def _build_safety_priority_advantage(
    safety_summary_by_case: Dict[str, Dict[str, Dict[str, float]]],
    reference_algorithm: str,
) -> Dict[str, Any]:
    comparisons: Dict[str, Any] = {}
    algorithms = sorted(
        {
            algorithm
            for case_summary in safety_summary_by_case.values()
            for algorithm in case_summary.keys()
            if algorithm != reference_algorithm
        }
    )
    for algorithm in algorithms:
        wins = 0
        ties = 0
        losses = 0
        clearance_gains: list[float] = []
        deficit_gains: list[float] = []
        collision_gains: list[float] = []
        failure_gains: list[float] = []
        compared_cases = 0
        for case_summary in safety_summary_by_case.values():
            if reference_algorithm not in case_summary or algorithm not in case_summary:
                continue
            compared_cases += 1
            reference_metrics = case_summary[reference_algorithm]
            challenger_metrics = case_summary[algorithm]
            reference_key = _safety_priority_sort_key(reference_metrics)
            challenger_key = _safety_priority_sort_key(challenger_metrics)
            if reference_key < challenger_key:
                wins += 1
            elif reference_key > challenger_key:
                losses += 1
            else:
                ties += 1
            clearance_gains.append(
                float(reference_metrics.get("mean_min_clearance", 0.0)) - float(challenger_metrics.get("mean_min_clearance", 0.0))
            )
            deficit_gains.append(
                float(challenger_metrics.get("mean_clearance_deficit", 0.0)) - float(reference_metrics.get("mean_clearance_deficit", 0.0))
            )
            collision_gains.append(
                float(challenger_metrics.get("avg_collision_count", 0.0)) - float(reference_metrics.get("avg_collision_count", 0.0))
            )
            failure_gains.append(
                float(challenger_metrics.get("strict_failure_rate", 0.0)) - float(reference_metrics.get("strict_failure_rate", 0.0))
            )
        if compared_cases == 0:
            continue
        comparisons[algorithm] = {
            "n_cases": int(compared_cases),
            "wins": int(wins),
            "ties": int(ties),
            "losses": int(losses),
            "win_rate": float(wins / compared_cases),
            "avg_min_clearance_gain": float(mean(clearance_gains)) if clearance_gains else 0.0,
            "avg_clearance_deficit_reduction": float(mean(deficit_gains)) if deficit_gains else 0.0,
            "avg_collision_count_reduction": float(mean(collision_gains)) if collision_gains else 0.0,
            "avg_failure_rate_reduction": float(mean(failure_gains)) if failure_gains else 0.0,
        }
    return {
        "reference_algorithm": reference_algorithm,
        "ranking_policy": "safety_priority_order",
        "comparisons": comparisons,
    }


def analyze_uav_suite(suite_results_path: str, output_path: str) -> Dict[str, Any]:
    suite = load_json(suite_results_path)
    records = suite["records"]
    algorithms = list(suite["algorithms"])
    levels = list(suite.get("levels", []))
    reference_algorithm = "L-ADE-Sniper"
    system_cost_metric = "metrics.system_cost"
    summary_by_case = summarize_by_case(records, system_cost_metric)
    system_cost_rank_table = build_rank_table(summary_by_case, system_cost_metric)
    system_cost_suite_summary = build_suite_summary(summary_by_case, system_cost_rank_table)
    significance = run_pairwise_significance(records, system_cost_metric, reference_algorithm=reference_algorithm)
    lade_advantage = build_reference_advantage(summary_by_case, reference_algorithm=reference_algorithm, lower_is_better=True)
    safety_priority_summary_by_case = _build_safety_priority_case_summary(records)
    safety_priority_rank_table = _build_safety_priority_rank_table(safety_priority_summary_by_case)
    safety_priority_suite_summary = build_suite_summary(safety_priority_summary_by_case, safety_priority_rank_table)
    safety_priority_advantage = _build_safety_priority_advantage(
        safety_priority_summary_by_case,
        reference_algorithm=reference_algorithm,
    )
    safety_significance = {
        "collision_count": run_pairwise_significance(records, "metrics.collision_count", reference_algorithm=reference_algorithm),
        "clearance_deficit": run_pairwise_significance(records, "metrics.clearance_deficit", reference_algorithm=reference_algorithm),
        "time_to_first_feasible_eval": run_pairwise_significance(
            records,
            "metrics.time_to_first_feasible_eval",
            reference_algorithm=reference_algorithm,
        ),
    }
    algorithm_summary = _build_algorithm_summary(records, algorithms)
    success_rate = {algorithm: metrics["success_rate"] for algorithm, metrics in algorithm_summary.items()}
    llm_efficiency = {
        algorithm: {
            "avg_llm_calls": metrics["avg_llm_calls"],
            "avg_runtime_sec": metrics["avg_runtime_sec"],
            "avg_eval_count": metrics["avg_eval_count"],
        }
        for algorithm, metrics in algorithm_summary.items()
    }
    collision_rate = {
        algorithm: {
            "collision_free_rate": metrics["collision_free_rate"],
            "avg_collision_count": metrics["avg_collision_count"],
        }
        for algorithm, metrics in algorithm_summary.items()
    }
    safety_margin = {
        algorithm: {
            "mean_min_clearance": metrics["mean_min_clearance"],
            "mean_clearance_deficit": metrics["mean_clearance_deficit"],
            "mean_time_to_first_feasible_eval": metrics["mean_time_to_first_feasible_eval"],
        }
        for algorithm, metrics in algorithm_summary.items()
    }
    path_quality = {
        algorithm: {
            "mean_path_length": metrics["mean_path_length"],
            "mean_path_efficiency": metrics["mean_path_efficiency"],
            "mean_curvature_cost": metrics["mean_curvature_cost"],
            "mean_altitude_variation": metrics["mean_altitude_variation"],
        }
        for algorithm, metrics in algorithm_summary.items()
    }
    analysis = {
        "application": "uav",
        "primary_metric": "safety_priority_order",
        "system_cost_metric": system_cost_metric,
        "algorithms": algorithms,
        "levels": levels,
        "summary_by_case": summary_by_case,
        "safety_priority_summary_by_case": safety_priority_summary_by_case,
        "per_case_mean_cost": _build_per_case_mean_cost(summary_by_case),
        "rank_table": safety_priority_rank_table,
        "suite_summary": safety_priority_suite_summary,
        "system_cost_rank_table": system_cost_rank_table,
        "system_cost_suite_summary": system_cost_suite_summary,
        "pairwise_significance": significance,
        "safety_significance": safety_significance,
        "lade_advantage": lade_advantage,
        "safety_priority_advantage": safety_priority_advantage,
        "success_rate": success_rate,
        "collision_rate": collision_rate,
        "safety_margin": safety_margin,
        "path_quality": path_quality,
        "algorithm_summary": algorithm_summary,
        "level_breakdown": _build_level_breakdown(records, algorithms, levels),
        "llm_efficiency": llm_efficiency,
    }
    save_json(analysis, output_path)
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite_results", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    analyze_uav_suite(suite_results_path=args.suite_results, output_path=args.output)


if __name__ == "__main__":
    main()
