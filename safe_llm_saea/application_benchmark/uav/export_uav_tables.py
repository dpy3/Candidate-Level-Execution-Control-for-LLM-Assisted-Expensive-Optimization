from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

from safe_llm_saea.application_benchmark.common.io_utils import ensure_dir, load_json


def _format_float(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value:.4f}"


def export_uav_latex_table(analysis_path: str, output_path: str) -> str:
    analysis = load_json(analysis_path)
    summary_by_case: Dict[str, Dict[str, Dict[str, float]]] = analysis["summary_by_case"]
    average_rank = analysis["rank_table"]["average_rank"]
    algorithms = sorted(average_rank, key=lambda name: average_rank[name])
    lines = [
        "\\begin{tabular}{l" + "c" * len(algorithms) + "}",
        "\\toprule",
        "Case & " + " & ".join(algorithms) + " \\\\",
        "\\midrule",
    ]
    for case_id, case_summary in summary_by_case.items():
        row = [case_id]
        for algorithm in algorithms:
            value = case_summary.get(algorithm, {}).get("mean")
            row.append(_format_float(value))
        lines.append(" & ".join(row) + " \\\\")
    rank_row = ["Avg. rank"] + [_format_float(average_rank.get(algorithm)) for algorithm in algorithms]
    lines.extend(["\\midrule", " & ".join(rank_row) + " \\\\", "\\bottomrule", "\\end{tabular}"])
    table = "\n".join(lines)
    output = Path(output_path)
    ensure_dir(output.parent)
    output.write_text(table, encoding="utf-8")
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    export_uav_latex_table(args.analysis, args.output)


if __name__ == "__main__":
    main()
