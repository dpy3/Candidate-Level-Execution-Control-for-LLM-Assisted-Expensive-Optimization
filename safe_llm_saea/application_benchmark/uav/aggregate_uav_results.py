from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from safe_llm_saea.application_benchmark.uav.analyze_uav_results import analyze_uav_suite
from safe_llm_saea.application_benchmark.uav.plot_uav_figures import (
    plot_uav_average_ranks,
    plot_uav_benchmark_overview,
    plot_uav_case_cost_heatmap,
    plot_uav_feasibility_collision,
    plot_uav_success_rates,
)
from safe_llm_saea.application_benchmark.common.io_utils import load_json, save_json


def load_uav_run_records(raw_dir: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for json_path in sorted(Path(raw_dir).glob("uav_*.json")):
        records.append(load_json(json_path))
    return records


def export_uav_suite_results(raw_dir: str, output_path: str, manifest_path: str = "") -> Dict[str, Any]:
    records = load_uav_run_records(raw_dir)
    suite = {
        "application": "uav",
        "n_records": len(records),
        "algorithms": sorted({record["algorithm"] for record in records}),
        "levels": sorted({record["level"] for record in records}),
        "case_ids": sorted({record["case_id"] for record in records}),
        "manifest_path": manifest_path,
        "records": records,
    }
    save_json(suite, output_path)
    return suite


def aggregate_uav_pipeline(
    raw_dir: str,
    suite_output: str,
    manifest_path: str = "",
    analysis_output: str = "",
    figures_dir: str = "",
) -> Dict[str, Any]:
    suite = export_uav_suite_results(raw_dir=raw_dir, output_path=suite_output, manifest_path=manifest_path)
    result: Dict[str, Any] = {"suite": suite, "analysis": None}
    if analysis_output:
        analysis = analyze_uav_suite(suite_results_path=suite_output, output_path=analysis_output)
        result["analysis"] = analysis
        if figures_dir:
            plot_uav_success_rates(analysis_output, figures_dir)
            plot_uav_average_ranks(analysis_output, figures_dir)
            plot_uav_feasibility_collision(analysis_output, figures_dir)
            plot_uav_case_cost_heatmap(analysis_output, figures_dir)
            plot_uav_benchmark_overview(analysis_output, figures_dir)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", default="")
    parser.add_argument("--analysis_output", default="")
    parser.add_argument("--figures_dir", default="")
    args = parser.parse_args()
    aggregate_uav_pipeline(
        raw_dir=args.raw_dir,
        suite_output=args.output,
        manifest_path=args.manifest,
        analysis_output=args.analysis_output,
        figures_dir=args.figures_dir,
    )


if __name__ == "__main__":
    main()
