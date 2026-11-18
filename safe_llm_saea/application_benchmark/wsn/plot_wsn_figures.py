from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from safe_llm_saea.application_benchmark.common.io_utils import ensure_dir, load_json


def plot_wsn_localization_ratio(analysis_path: str, output_dir: str) -> None:
    analysis = load_json(analysis_path)
    localization_ratio = analysis.get("localization_ratio", {})
    algorithms = list(localization_ratio.keys())
    values = [localization_ratio[name] for name in algorithms]
    ensure_dir(output_dir)
    plt.figure(figsize=(10, 5))
    plt.bar(algorithms, values)
    plt.ylim(0, 1.05)
    plt.ylabel("Localization ratio")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "wsn_localization_ratio.png", dpi=300)
    plt.close()


def plot_wsn_average_rank(analysis_path: str, output_dir: str) -> None:
    analysis = load_json(analysis_path)
    average_rank = analysis.get("rank_table", {}).get("average_rank", {})
    algorithms = list(average_rank.keys())
    values = [average_rank[name] for name in algorithms]
    ensure_dir(output_dir)
    plt.figure(figsize=(10, 5))
    plt.bar(algorithms, values)
    plt.ylabel("Average rank")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "wsn_average_rank.png", dpi=300)
    plt.close()


def _ordered_algorithms(analysis: dict) -> list[str]:
    average_rank = analysis.get("rank_table", {}).get("average_rank", {})
    return sorted(average_rank, key=lambda name: average_rank[name])


def _plot_factor_metric(analysis: dict, factor_key: str, metric_key: str, ylabel: str, output_path: Path) -> None:
    factor_breakdown = analysis.get("factor_breakdown", {}).get(factor_key, {})
    if not factor_breakdown:
        return
    algorithms = _ordered_algorithms(analysis)
    factor_values = sorted(factor_breakdown, key=lambda item: float(item))
    plt.figure(figsize=(8.4, 5.2))
    for algorithm in algorithms:
        xs = [float(value) for value in factor_values]
        ys = [
            factor_breakdown[value].get(algorithm, {}).get(metric_key)
            for value in factor_values
        ]
        valid_points = [(x, y) for x, y in zip(xs, ys) if y is not None]
        if not valid_points:
            continue
        plt.plot(
            [item[0] for item in valid_points],
            [item[1] for item in valid_points],
            marker="o",
            linewidth=2,
            label=algorithm,
        )
    plt.xlabel(factor_key.replace("_", " ").title())
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_wsn_factor_panels(analysis_path: str, output_dir: str) -> None:
    analysis = load_json(analysis_path)
    ensure_dir(output_dir)
    output_root = Path(output_dir)
    _plot_factor_metric(
        analysis=analysis,
        factor_key="node_count",
        metric_key="mean_ale",
        ylabel="Mean ALE",
        output_path=output_root / "wsn_ale_vs_node_count.png",
    )
    _plot_factor_metric(
        analysis=analysis,
        factor_key="anchor_count",
        metric_key="mean_ale",
        ylabel="Mean ALE",
        output_path=output_root / "wsn_ale_vs_anchor_count.png",
    )
    _plot_factor_metric(
        analysis=analysis,
        factor_key="comm_radius",
        metric_key="mean_ale",
        ylabel="Mean ALE",
        output_path=output_root / "wsn_ale_vs_comm_radius.png",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    plot_wsn_localization_ratio(args.analysis, args.output_dir)
    plot_wsn_average_rank(args.analysis, args.output_dir)
    plot_wsn_factor_panels(args.analysis, args.output_dir)


if __name__ == "__main__":
    main()
