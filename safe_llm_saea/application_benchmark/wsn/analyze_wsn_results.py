from __future__ import annotations

import argparse
import re
from statistics import mean
from typing import Any, Dict, Iterable, List, Tuple

from safe_llm_saea.application_benchmark.common.io_utils import load_json, save_json
from safe_llm_saea.application_benchmark.common.statistics import (
    build_rank_table,
    build_reference_advantage,
    build_suite_summary,
    run_pairwise_significance,
    summarize_by_case,
)


_CASE_PATTERN = re.compile(r"N(?P<node_count>\d+)_K(?P<anchor_count>\d+)_R(?P<comm_radius>\d+)_S(?P<seed>\d+)")


def _safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(mean(values))


def _extract_case_metadata(record: Dict[str, Any]) -> Dict[str, float | int]:
    metadata = {
        "node_count": record.get("node_count"),
        "anchor_count": record.get("anchor_count"),
        "comm_radius": record.get("comm_radius"),
        "topology_seed": record.get("topology_seed", record.get("seed")),
    }
    if all(value is not None for value in metadata.values()):
        return {
            "node_count": int(metadata["node_count"]),
            "anchor_count": int(metadata["anchor_count"]),
            "comm_radius": float(metadata["comm_radius"]),
            "topology_seed": int(metadata["topology_seed"]),
        }
    match = _CASE_PATTERN.fullmatch(str(record.get("case_id", "")))
    if match is None:
        return {"node_count": 0, "anchor_count": 0, "comm_radius": 0.0, "topology_seed": 0}
    return {
        "node_count": int(match.group("node_count")),
        "anchor_count": int(match.group("anchor_count")),
        "comm_radius": float(match.group("comm_radius")),
        "topology_seed": int(match.group("seed")),
    }


def _build_case_metadata(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, float | int]]:
    metadata: Dict[str, Dict[str, float | int]] = {}
    for record in records:
        case_id = str(record["case_id"])
        if case_id not in metadata:
            metadata[case_id] = _extract_case_metadata(record)
    return metadata


def _build_algorithm_summary(
    suite_algorithms: List[str],
    records: List[Dict[str, Any]],
    average_rank: Dict[str, float],
    suite_summary: Dict[str, Any],
) -> Dict[str, Dict[str, float]]:
    algorithm_summary: Dict[str, Dict[str, float]] = {}
    wins = suite_summary.get("wins", {})
    for algorithm in suite_algorithms:
        algo_records = [record for record in records if record["algorithm"] == algorithm]
        if not algo_records:
            continue
        algorithm_summary[algorithm] = {
            "mean_ale": _safe_mean(record["metrics"]["ale"] for record in algo_records),
            "mean_rmse": _safe_mean(record["metrics"]["rmse"] for record in algo_records),
            "mean_max_error": _safe_mean(record["metrics"]["max_error"] for record in algo_records),
            "mean_localized_ratio": _safe_mean(record["metrics"]["localized_ratio"] for record in algo_records),
            "mean_anchor_distance_rmse": _safe_mean(
                record["metrics"].get("geometry_consistency", {}).get("anchor_distance_rmse", 0.0)
                for record in algo_records
            ),
            "mean_anchor_distance_mae": _safe_mean(
                record["metrics"].get("geometry_consistency", {}).get("anchor_distance_mae", 0.0)
                for record in algo_records
            ),
            "mean_relative_distance_error": _safe_mean(
                record["metrics"].get("geometry_consistency", {}).get("relative_distance_error", 0.0)
                for record in algo_records
            ),
            "nearest_anchor_match_ratio": _safe_mean(
                record["metrics"].get("geometry_consistency", {}).get("nearest_anchor_match_ratio", 0.0)
                for record in algo_records
            ),
            "one_hop_anchor_violation_rate": _safe_mean(
                record["metrics"].get("geometry_consistency", {}).get("one_hop_anchor_violation_rate", 0.0)
                for record in algo_records
            ),
            "multi_hop_anchor_shortcut_rate": _safe_mean(
                record["metrics"].get("geometry_consistency", {}).get("multi_hop_anchor_shortcut_rate", 0.0)
                for record in algo_records
            ),
            "finite_anchor_support_ratio": _safe_mean(
                record["metrics"].get("geometry_consistency", {}).get("finite_anchor_support_ratio", 0.0)
                for record in algo_records
            ),
            "average_rank": float(average_rank.get(algorithm, 0.0)),
            "wins": float(wins.get(algorithm, 0.0)),
            "avg_runtime_sec": _safe_mean(record["runtime"].get("wall_clock_sec", 0.0) for record in algo_records),
            "avg_eval_count": _safe_mean(record["runtime"].get("objective_eval_count", 0) for record in algo_records),
            "avg_llm_calls": _safe_mean(record["llm_stats"].get("total_calls", 0) for record in algo_records),
        }
    return algorithm_summary


def _build_factor_breakdown(
    records: List[Dict[str, Any]],
    suite_algorithms: List[str],
    case_metadata: Dict[str, Dict[str, float | int]],
) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    factor_aliases: List[Tuple[str, str]] = [
        ("node_count", "node_count"),
        ("anchor_count", "anchor_count"),
        ("comm_radius", "comm_radius"),
    ]
    factor_breakdown: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for factor_key, metadata_key in factor_aliases:
        value_groups: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        for record in records:
            case_id = str(record["case_id"])
            factor_value = case_metadata.get(case_id, {}).get(metadata_key, 0)
            value_key = str(int(factor_value) if float(factor_value).is_integer() else float(factor_value))
            value_groups.setdefault(value_key, {}).setdefault(record["algorithm"], []).append(record)
        ordered_groups = sorted(value_groups.items(), key=lambda item: float(item[0]))
        factor_breakdown[factor_key] = {}
        for value_key, grouped_records in ordered_groups:
            factor_breakdown[factor_key][value_key] = {}
            for algorithm in suite_algorithms:
                algo_records = grouped_records.get(algorithm, [])
                if not algo_records:
                    continue
                factor_breakdown[factor_key][value_key][algorithm] = {
                    "count": float(len(algo_records)),
                    "mean_ale": _safe_mean(record["metrics"]["ale"] for record in algo_records),
                    "mean_rmse": _safe_mean(record["metrics"]["rmse"] for record in algo_records),
                    "mean_max_error": _safe_mean(record["metrics"]["max_error"] for record in algo_records),
                    "mean_localized_ratio": _safe_mean(record["metrics"]["localized_ratio"] for record in algo_records),
                    "mean_anchor_distance_rmse": _safe_mean(
                        record["metrics"].get("geometry_consistency", {}).get("anchor_distance_rmse", 0.0)
                        for record in algo_records
                    ),
                    "mean_anchor_distance_mae": _safe_mean(
                        record["metrics"].get("geometry_consistency", {}).get("anchor_distance_mae", 0.0)
                        for record in algo_records
                    ),
                    "mean_relative_distance_error": _safe_mean(
                        record["metrics"].get("geometry_consistency", {}).get("relative_distance_error", 0.0)
                        for record in algo_records
                    ),
                    "nearest_anchor_match_ratio": _safe_mean(
                        record["metrics"].get("geometry_consistency", {}).get("nearest_anchor_match_ratio", 0.0)
                        for record in algo_records
                    ),
                    "one_hop_anchor_violation_rate": _safe_mean(
                        record["metrics"].get("geometry_consistency", {}).get("one_hop_anchor_violation_rate", 0.0)
                        for record in algo_records
                    ),
                    "multi_hop_anchor_shortcut_rate": _safe_mean(
                        record["metrics"].get("geometry_consistency", {}).get("multi_hop_anchor_shortcut_rate", 0.0)
                        for record in algo_records
                    ),
                    "finite_anchor_support_ratio": _safe_mean(
                        record["metrics"].get("geometry_consistency", {}).get("finite_anchor_support_ratio", 0.0)
                        for record in algo_records
                    ),
                    "avg_runtime_sec": _safe_mean(record["runtime"].get("wall_clock_sec", 0.0) for record in algo_records),
                    "avg_eval_count": _safe_mean(record["runtime"].get("objective_eval_count", 0) for record in algo_records),
                    "avg_llm_calls": _safe_mean(record["llm_stats"].get("total_calls", 0) for record in algo_records),
                }
    return factor_breakdown


def analyze_wsn_suite(suite_results_path: str, output_path: str) -> Dict[str, Any]:
    suite = load_json(suite_results_path)
    records = suite["records"]
    algorithms = suite["algorithms"]
    reference_algorithm = "L-ADE-Sniper"
    primary_metric = "metrics.ale"
    summary_by_case = summarize_by_case(records, primary_metric)
    rank_table = build_rank_table(summary_by_case, primary_metric)
    suite_summary = build_suite_summary(summary_by_case, rank_table)
    significance = run_pairwise_significance(records, primary_metric, reference_algorithm=reference_algorithm)
    lade_advantage = build_reference_advantage(summary_by_case, reference_algorithm=reference_algorithm, lower_is_better=True)
    case_metadata = _build_case_metadata(records)
    localization_ratio = {}
    llm_efficiency = {}
    for algorithm in algorithms:
        algo_records = [record for record in records if record["algorithm"] == algorithm]
        if not algo_records:
            continue
        localization_ratio[algorithm] = sum(record["metrics"]["localized_ratio"] for record in algo_records) / len(algo_records)
        llm_efficiency[algorithm] = {
            "avg_llm_calls": float(mean(record["llm_stats"].get("total_calls", 0) for record in algo_records)),
            "avg_runtime_sec": float(mean(record["runtime"].get("wall_clock_sec", 0.0) for record in algo_records)),
            "avg_eval_count": float(mean(record["runtime"].get("objective_eval_count", 0) for record in algo_records)),
        }
    algorithm_summary = _build_algorithm_summary(
        suite_algorithms=algorithms,
        records=records,
        average_rank=rank_table.get("average_rank", {}),
        suite_summary=suite_summary,
    )
    factor_breakdown = _build_factor_breakdown(records=records, suite_algorithms=algorithms, case_metadata=case_metadata)
    geometry_consistency = {
        algorithm: {
            "mean_anchor_distance_rmse": metrics["mean_anchor_distance_rmse"],
            "mean_anchor_distance_mae": metrics["mean_anchor_distance_mae"],
            "mean_relative_distance_error": metrics["mean_relative_distance_error"],
            "nearest_anchor_match_ratio": metrics["nearest_anchor_match_ratio"],
            "one_hop_anchor_violation_rate": metrics["one_hop_anchor_violation_rate"],
            "multi_hop_anchor_shortcut_rate": metrics["multi_hop_anchor_shortcut_rate"],
            "finite_anchor_support_ratio": metrics["finite_anchor_support_ratio"],
        }
        for algorithm, metrics in algorithm_summary.items()
    }
    analysis = {
        "application": "wsn",
        "primary_metric": primary_metric,
        "algorithms": algorithms,
        "case_metadata": case_metadata,
        "summary_by_case": summary_by_case,
        "rank_table": rank_table,
        "suite_summary": suite_summary,
        "pairwise_significance": significance,
        "lade_advantage": lade_advantage,
        "localization_ratio": localization_ratio,
        "llm_efficiency": llm_efficiency,
        "algorithm_summary": algorithm_summary,
        "geometry_consistency": geometry_consistency,
        "factor_breakdown": factor_breakdown,
    }
    save_json(analysis, output_path)
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite_results", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    analyze_wsn_suite(suite_results_path=args.suite_results, output_path=args.output)


if __name__ == "__main__":
    main()
