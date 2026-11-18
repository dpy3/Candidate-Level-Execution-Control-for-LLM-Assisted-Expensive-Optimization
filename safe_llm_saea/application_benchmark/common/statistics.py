from __future__ import annotations

from collections import defaultdict
from statistics import mean, median, pstdev
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

try:
    from scipy.stats import friedmanchisquare, rankdata, wilcoxon
except Exception:
    friedmanchisquare = None
    rankdata = None
    wilcoxon = None


def _resolve_metric_value(record: Dict[str, Any], metric_key: str) -> float:
    if metric_key in record:
        return float(record[metric_key])
    if metric_key in record.get("metrics", {}):
        return float(record["metrics"][metric_key])
    current: Any = record
    for part in metric_key.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            raise KeyError(f"Metric key not found: {metric_key}")
    return float(current)


def summarize_metric(records: List[Dict[str, Any]], metric_key: str, lower_is_better: bool = True) -> Dict[str, Any]:
    values = [_resolve_metric_value(record, metric_key) for record in records]
    if not values:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "median": None,
            "best": None,
            "worst": None,
            "lower_is_better": lower_is_better,
        }
    best = min(values) if lower_is_better else max(values)
    worst = max(values) if lower_is_better else min(values)
    return {
        "count": len(values),
        "mean": float(mean(values)),
        "std": float(pstdev(values)) if len(values) > 1 else 0.0,
        "median": float(median(values)),
        "best": float(best),
        "worst": float(worst),
        "lower_is_better": lower_is_better,
    }


def _group_by_case_and_algorithm(records: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        grouped[str(record["case_id"])][str(record["algorithm"])].append(record)
    return grouped


def summarize_by_case(records: List[Dict[str, Any]], primary_metric: str) -> Dict[str, Any]:
    grouped = _group_by_case_and_algorithm(records)
    summary: Dict[str, Any] = {}
    for case_id, algorithms in grouped.items():
        summary[case_id] = {}
        for algorithm, algo_records in algorithms.items():
            summary[case_id][algorithm] = summarize_metric(algo_records, primary_metric, lower_is_better=True)
    return summary


def build_rank_table(summary_by_case: Dict[str, Any], primary_metric: str) -> Dict[str, Any]:
    case_ranks: Dict[str, Dict[str, float]] = {}
    average_rank_map: Dict[str, List[float]] = defaultdict(list)
    for case_id, case_summary in summary_by_case.items():
        ordered: List[Tuple[str, float]] = []
        for algorithm, metrics in case_summary.items():
            value = metrics.get("mean")
            if value is not None:
                ordered.append((algorithm, float(value)))
        if not ordered:
            continue
        values = np.array([item[1] for item in ordered], dtype=float)
        if rankdata is None:
            order = np.argsort(values)
            ranks = np.empty_like(values, dtype=float)
            ranks[order] = np.arange(1, len(values) + 1, dtype=float)
        else:
            ranks = rankdata(values, method="average")
        case_ranks[case_id] = {}
        for index, (algorithm, _) in enumerate(ordered):
            case_ranks[case_id][algorithm] = float(ranks[index])
            average_rank_map[algorithm].append(float(ranks[index]))
    average_rank = {
        algorithm: float(mean(ranks))
        for algorithm, ranks in average_rank_map.items()
        if ranks
    }
    return {
        "primary_metric": primary_metric,
        "case_ranks": case_ranks,
        "average_rank": average_rank,
    }


def run_pairwise_significance(
    records: List[Dict[str, Any]],
    primary_metric: str,
    reference_algorithm: str,
) -> Dict[str, Any]:
    grouped = _group_by_case_and_algorithm(records)
    comparisons: Dict[str, Any] = {}
    reference_values: Dict[str, float] = {}
    for case_id, algorithms in grouped.items():
        if reference_algorithm in algorithms:
            reference_values[case_id] = summarize_metric(algorithms[reference_algorithm], primary_metric)["mean"]
    all_algorithms = sorted({record["algorithm"] for record in records if record["algorithm"] != reference_algorithm})
    for algorithm in all_algorithms:
        x_values: List[float] = []
        y_values: List[float] = []
        for case_id, algorithms in grouped.items():
            if reference_algorithm in algorithms and algorithm in algorithms:
                x_values.append(reference_values[case_id])
                y_values.append(summarize_metric(algorithms[algorithm], primary_metric)["mean"])
        result: Dict[str, Any] = {"n_cases": len(x_values), "reference_algorithm": reference_algorithm}
        if len(x_values) >= 2 and wilcoxon is not None:
            try:
                stat = wilcoxon(x_values, y_values, zero_method="wilcox", alternative="two-sided")
                result["pvalue"] = float(stat.pvalue)
                result["statistic"] = float(stat.statistic)
            except ValueError:
                result["pvalue"] = None
                result["statistic"] = None
        else:
            result["pvalue"] = None
            result["statistic"] = None
        comparisons[algorithm] = result
    if friedmanchisquare is not None:
        case_vectors: Dict[str, List[float]] = defaultdict(list)
        for case_id, algorithms in grouped.items():
            if len(algorithms) < 2:
                continue
            for algorithm, algo_records in sorted(algorithms.items()):
                case_vectors[algorithm].append(summarize_metric(algo_records, primary_metric)["mean"])
        comparable = [values for values in case_vectors.values() if len(values) == len(next(iter(case_vectors.values()), []))]
        if len(comparable) >= 2 and comparable and len(comparable[0]) >= 2:
            friedman = friedmanchisquare(*comparable)
            comparisons["friedman"] = {"statistic": float(friedman.statistic), "pvalue": float(friedman.pvalue)}
    return comparisons


def build_suite_summary(summary_by_case: Dict[str, Any], rank_table: Dict[str, Any]) -> Dict[str, Any]:
    wins: Dict[str, int] = defaultdict(int)
    top3: Dict[str, int] = defaultdict(int)
    for _, case_ranks in rank_table.get("case_ranks", {}).items():
        sorted_algorithms = sorted(case_ranks.items(), key=lambda item: item[1])
        for index, (algorithm, _) in enumerate(sorted_algorithms):
            if index == 0:
                wins[algorithm] += 1
            if index < 3:
                top3[algorithm] += 1
    return {
        "average_rank": rank_table.get("average_rank", {}),
        "wins": dict(wins),
        "top3": dict(top3),
        "n_cases": len(summary_by_case),
    }


def build_reference_advantage(
    summary_by_case: Dict[str, Any],
    reference_algorithm: str,
    lower_is_better: bool = True,
) -> Dict[str, Any]:
    comparisons: Dict[str, Any] = {}
    algorithms = sorted(
        {
            algorithm
            for case_summary in summary_by_case.values()
            for algorithm in case_summary.keys()
            if algorithm != reference_algorithm
        }
    )
    for algorithm in algorithms:
        reference_values: List[float] = []
        challenger_values: List[float] = []
        wins = 0
        ties = 0
        losses = 0
        relative_gains: List[float] = []
        for case_summary in summary_by_case.values():
            if reference_algorithm not in case_summary or algorithm not in case_summary:
                continue
            reference_mean = case_summary[reference_algorithm].get("mean")
            challenger_mean = case_summary[algorithm].get("mean")
            if reference_mean is None or challenger_mean is None:
                continue
            ref_value = float(reference_mean)
            challenger_value = float(challenger_mean)
            reference_values.append(ref_value)
            challenger_values.append(challenger_value)
            delta = challenger_value - ref_value if lower_is_better else ref_value - challenger_value
            scale = max(abs(challenger_value), 1e-12)
            relative_gains.append(delta / scale)
            if abs(delta) <= 1e-12:
                ties += 1
            elif delta > 0:
                wins += 1
            else:
                losses += 1
        if not reference_values:
            continue
        avg_reference = float(mean(reference_values))
        avg_challenger = float(mean(challenger_values))
        absolute_gain = avg_challenger - avg_reference if lower_is_better else avg_reference - avg_challenger
        comparisons[algorithm] = {
            "n_cases": len(reference_values),
            "reference_mean": avg_reference,
            "challenger_mean": avg_challenger,
            "absolute_gain": float(absolute_gain),
            "relative_gain_pct": float(mean(relative_gains) * 100.0) if relative_gains else 0.0,
            "wins": int(wins),
            "ties": int(ties),
            "losses": int(losses),
            "win_rate": float(wins / len(reference_values)),
        }
    return {
        "reference_algorithm": reference_algorithm,
        "lower_is_better": bool(lower_is_better),
        "comparisons": comparisons,
    }
