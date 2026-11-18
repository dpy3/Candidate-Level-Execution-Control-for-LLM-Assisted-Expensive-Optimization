from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from safe_llm_saea.application_benchmark.common.io_utils import ensure_dir, load_json, save_json
from safe_llm_saea.application_benchmark.wsn.analyze_wsn_results import analyze_wsn_suite
from safe_llm_saea.application_benchmark.wsn.plot_wsn_figures import (
    plot_wsn_average_rank,
    plot_wsn_factor_panels,
    plot_wsn_localization_ratio,
)

try:
    from scipy.io import savemat
except Exception:
    savemat = None


def load_wsn_run_records(raw_dir: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for json_path in sorted(Path(raw_dir).glob("wsn_*.json")):
        records.append(load_json(json_path))
    return records


def export_wsn_suite_results(
    raw_dir: str,
    output_json_path: str,
    output_mat_path: str | None = None,
    manifest_path: str = "",
) -> Dict[str, Any]:
    records = load_wsn_run_records(raw_dir)
    suite = {
        "application": "wsn",
        "n_records": len(records),
        "algorithms": sorted({record["algorithm"] for record in records}),
        "case_ids": sorted({record["case_id"] for record in records}),
        "manifest_path": manifest_path,
        "records": records,
    }
    save_json(suite, output_json_path)
    if output_mat_path and savemat is not None:
        ensure_dir(Path(output_mat_path).parent)
        savemat(
            output_mat_path,
            {
                "algorithms": np.array(suite["algorithms"], dtype=object),
                "ale": np.array([record["metrics"]["ale"] for record in records], dtype=float),
                "rmse": np.array([record["metrics"]["rmse"] for record in records], dtype=float),
                "case_ids": np.array([record["case_id"] for record in records], dtype=object),
            },
        )
    return suite


def aggregate_wsn_pipeline(
    raw_dir: str,
    suite_output: str,
    manifest_path: str = "",
    analysis_output: str = "",
    figures_dir: str = "",
    mat_output: str = "",
) -> Dict[str, Any]:
    suite = export_wsn_suite_results(
        raw_dir=raw_dir,
        output_json_path=suite_output,
        output_mat_path=mat_output or None,
        manifest_path=manifest_path,
    )
    result: Dict[str, Any] = {"suite": suite, "analysis": None}
    if analysis_output:
        analysis = analyze_wsn_suite(suite_results_path=suite_output, output_path=analysis_output)
        result["analysis"] = analysis
        if figures_dir:
            plot_wsn_localization_ratio(analysis_output, figures_dir)
            plot_wsn_average_rank(analysis_output, figures_dir)
            plot_wsn_factor_panels(analysis_output, figures_dir)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_mat")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--analysis_output", default="")
    parser.add_argument("--figures_dir", default="")
    args = parser.parse_args()
    aggregate_wsn_pipeline(
        raw_dir=args.raw_dir,
        suite_output=args.output_json,
        manifest_path=args.manifest,
        analysis_output=args.analysis_output,
        figures_dir=args.figures_dir,
        mat_output=args.output_mat or "",
    )


if __name__ == "__main__":
    main()
