from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List

sys.path.append(str(Path(__file__).resolve().parents[2]))

from safe_llm_saea.application_benchmark.common.io_utils import load_json, save_json
from safe_llm_saea.application_benchmark.uav.aggregate_uav_results import aggregate_uav_pipeline
from safe_llm_saea.application_benchmark.uav.build_uav_suite import build_uav_suite_manifest
from safe_llm_saea.application_benchmark.uav.run_uav_benchmark import run_single_uav_case
from safe_llm_saea.application_benchmark.wsn.aggregate_wsn_results import aggregate_wsn_pipeline
from safe_llm_saea.application_benchmark.wsn.build_wsn_suite import build_wsn_suite_manifest
from safe_llm_saea.application_benchmark.wsn.run_wsn_benchmark import run_single_wsn_case

_LLM_ALGORITHMS = {"L-ADE-Sniper", "LLM-DE", "LLM-SAEA"}
_UAV_BOUNDS = {"x": [-10.0, 190.0], "y": [-10.0, 190.0], "z": [5.0, 40.0]}


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_output_root() -> Path:
    return _repository_root() / "tmp" / "app_benchmarks" / "lade_proof"


def _default_uav_source_dir() -> Path:
    return _repository_root() / "safe_llm_saea" / "paper_results"


def _has_api_key(args: argparse.Namespace) -> bool:
    return bool(
        str(args.api_key).strip()
        or os.getenv("ds_api", "").strip()
        or os.getenv("DEEPSEEK_API_KEY", "").strip()
        or os.getenv("OPENAI_API_KEY", "").strip()
    )


def _budget_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "max_fes": int(args.max_fes),
        "population_size": int(args.population_size),
        "max_iter": int(args.max_iter),
        "llm_call_budget": int(args.llm_call_budget),
        "llm_cost_budget_usd": float(args.llm_cost_budget_usd),
    }


def _base_llm_config(args: argparse.Namespace, enabled_algorithms: Iterable[str], use_api: bool) -> Dict[str, Any]:
    return {
        "enabled_algorithms": set(str(name) for name in enabled_algorithms),
        "use_api": bool(use_api),
        "api_key": str(args.api_key),
        "base_url": str(args.base_url),
        "llm_model": str(args.llm_model),
    }


def _proof_lade_variant(application: str, profile: str = "default", **overrides: Any) -> Dict[str, Any]:
    base = {
        "force_initial_llm_call": True,
        "min_stagnation_generations": 1,
        "min_call_interval": 1,
        "min_remaining_budget": 6,
        "enable_stage2": True,
        "stage2_stagnation_generations": 1,
    }
    if application == "wsn" and profile == "engaged":
        base.update(
            {
                "min_remaining_budget": 4,
                "min_confidence": 0.55,
                "llm_stagnation_trigger": 2,
                "stage2_eval_ratio_threshold": 0.30,
                "stage2_call_interval_factor": 0.5,
                "stage1_calls": 5,
                "stage2_calls": 3,
                "llm_suggestions_k": 4,
                "precision_progress_threshold": 0.78,
                "sniper_interval": 2,
                "enable_micro_refine": True,
                "micro_refine_interval": 2,
                "micro_refine_steps": 4,
                "micro_refine_sigma": 0.025,
            }
        )
    if application == "uav" and profile == "engaged":
        base.update(
            {
                "min_remaining_budget": 10,
                "min_confidence": 0.58,
                "min_uncertainty_threshold": 0.05,
                "llm_stagnation_trigger": 3,
                "stage2_eval_ratio_threshold": 0.32,
                "stage2_call_interval_factor": 0.5,
                "llm_suggestions_k": 4,
                "precision_progress_threshold": 0.72,
                "sniper_interval": 2,
                "enable_micro_refine": True,
                "micro_refine_interval": 2,
                "micro_refine_steps": 3,
                "micro_refine_sigma": 0.02,
            }
        )
    base.update(overrides)
    return base


def _variant_specs(
    application: str,
    llm_variants_use_api: bool,
    proof_profile: str = "default",
    selected_labels: set[str] | None = None,
) -> List[Dict[str, Any]]:
    reference_algorithm = "RRT" if application == "uav" else "DV-Hop"
    variants: List[Dict[str, Any]] = [
        {
            "label": "L-ADE-Sniper",
            "algorithm_name": "L-ADE-Sniper",
            "llm_enabled": True,
            "use_api": bool(llm_variants_use_api),
            "lade_variant": _proof_lade_variant(application, proof_profile, use_api=bool(llm_variants_use_api)),
        },
        {
            "label": "L-ADE-Sniper-no-LLM",
            "algorithm_name": "L-ADE-Sniper",
            "llm_enabled": False,
            "use_api": False,
            "lade_variant": _proof_lade_variant(
                application,
                proof_profile,
                llm_enabled=False,
                use_api=False,
                force_initial_llm_call=False,
            ),
        },
        {
            "label": "L-ADE-Sniper-strict-zero-call",
            "algorithm_name": "L-ADE-Sniper",
            "llm_enabled": False,
            "use_api": False,
            "lade_variant": _proof_lade_variant(
                application,
                proof_profile,
                strict_zero_call=True,
                llm_enabled=False,
                use_api=False,
                force_initial_llm_call=False,
            ),
        },
        {
            "label": "L-ADE-Sniper-no-Surrogate",
            "algorithm_name": "L-ADE-Sniper",
            "llm_enabled": True,
            "use_api": bool(llm_variants_use_api),
            "lade_variant": _proof_lade_variant(
                application,
                proof_profile,
                use_surrogate=False,
                use_api=bool(llm_variants_use_api),
            ),
        },
        {
            "label": "L-ADE-Sniper-no-Sniper",
            "algorithm_name": "L-ADE-Sniper",
            "llm_enabled": True,
            "use_api": bool(llm_variants_use_api),
            "lade_variant": _proof_lade_variant(
                application,
                proof_profile,
                enable_sniper=False,
                use_api=bool(llm_variants_use_api),
            ),
        },
        {
            "label": "DE",
            "algorithm_name": "DE",
            "llm_enabled": False,
            "use_api": False,
            "lade_variant": {},
        },
        {
            "label": reference_algorithm,
            "algorithm_name": reference_algorithm,
            "llm_enabled": False,
            "use_api": False,
            "lade_variant": {},
        },
    ]
    if selected_labels:
        variants = [variant for variant in variants if str(variant["label"]) in selected_labels]
    return variants


def _variant_llm_config(args: argparse.Namespace, variant: Dict[str, Any]) -> Dict[str, Any]:
    algorithm_name = str(variant["algorithm_name"])
    enabled_algorithms = [algorithm_name] if bool(variant.get("llm_enabled")) and algorithm_name in _LLM_ALGORITHMS else []
    return _base_llm_config(
        args=args,
        enabled_algorithms=enabled_algorithms,
        use_api=bool(variant.get("use_api", False)),
    )


def _safe_mean(values: Iterable[float]) -> float:
    values_list = [float(value) for value in values]
    return float(mean(values_list)) if values_list else 0.0


def _collect_variant_runtime_profile(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, float | bool]]:
    labels = sorted({str(record["algorithm"]) for record in records})
    profile: Dict[str, Dict[str, float | bool]] = {}
    for label in labels:
        label_records = [record for record in records if str(record["algorithm"]) == label]
        profile[label] = {
            "n_records": float(len(label_records)),
            "avg_total_llm_calls": _safe_mean(record.get("llm_stats", {}).get("total_calls", 0.0) for record in label_records),
            "avg_success_llm_calls": _safe_mean(record.get("llm_stats", {}).get("success_calls", 0.0) for record in label_records),
            "avg_token_cost_usd": _safe_mean(record.get("llm_stats", {}).get("token_cost_usd", 0.0) for record in label_records),
            "api_used_rate": _safe_mean(1.0 if record.get("llm_stats", {}).get("api_used", False) else 0.0 for record in label_records),
            "runtime_llm_enabled_rate": _safe_mean(
                1.0 if record.get("llm_stats", {}).get("runtime_llm_enabled", False) else 0.0 for record in label_records
            ),
            "runtime_surrogate_enabled_rate": _safe_mean(
                1.0 if record.get("llm_stats", {}).get("runtime_surrogate_enabled", False) else 0.0 for record in label_records
            ),
            "runtime_sniper_enabled_rate": _safe_mean(
                1.0 if record.get("llm_stats", {}).get("runtime_sniper_enabled", False) else 0.0 for record in label_records
            ),
        }
    return profile


def _metric_delta(
    algorithm_summary: Dict[str, Dict[str, float]],
    winner_label: str,
    challenger_label: str,
    lower_is_better: List[str],
    higher_is_better: List[str],
) -> Dict[str, Dict[str, float]]:
    winner = algorithm_summary.get(winner_label)
    challenger = algorithm_summary.get(challenger_label)
    if not winner or not challenger:
        return {}
    delta: Dict[str, Dict[str, float]] = {}
    for metric_name in lower_is_better:
        winner_value = float(winner.get(metric_name, 0.0))
        challenger_value = float(challenger.get(metric_name, 0.0))
        delta[metric_name] = {
            "winner": winner_value,
            "challenger": challenger_value,
            "improvement": challenger_value - winner_value,
        }
    for metric_name in higher_is_better:
        winner_value = float(winner.get(metric_name, 0.0))
        challenger_value = float(challenger.get(metric_name, 0.0))
        delta[metric_name] = {
            "winner": winner_value,
            "challenger": challenger_value,
            "improvement": winner_value - challenger_value,
        }
    return delta


def _build_wsn_proof_summary(
    analysis: Dict[str, Any],
    records: List[Dict[str, Any]],
    llm_variants_use_api: bool,
) -> Dict[str, Any]:
    algorithm_summary = analysis.get("algorithm_summary", {})
    lower_is_better = [
        "mean_ale",
        "mean_rmse",
        "mean_anchor_distance_rmse",
        "mean_anchor_distance_mae",
        "mean_relative_distance_error",
        "one_hop_anchor_violation_rate",
        "multi_hop_anchor_shortcut_rate",
        "average_rank",
    ]
    higher_is_better = [
        "mean_localized_ratio",
        "nearest_anchor_match_ratio",
        "finite_anchor_support_ratio",
    ]
    component_proof = {
        "llm": _metric_delta(algorithm_summary, "L-ADE-Sniper", "L-ADE-Sniper-no-LLM", lower_is_better, higher_is_better),
        "surrogate": _metric_delta(
            algorithm_summary,
            "L-ADE-Sniper",
            "L-ADE-Sniper-no-Surrogate",
            lower_is_better,
            higher_is_better,
        ),
        "sniper": _metric_delta(
            algorithm_summary,
            "L-ADE-Sniper",
            "L-ADE-Sniper-no-Sniper",
            lower_is_better,
            higher_is_better,
        ),
        "vs_de": _metric_delta(algorithm_summary, "L-ADE-Sniper", "DE", lower_is_better, higher_is_better),
        "vs_reference": _metric_delta(algorithm_summary, "L-ADE-Sniper", "DV-Hop", lower_is_better, higher_is_better),
    }
    return {
        "suite_summary": analysis.get("suite_summary", {}),
        "pairwise_significance": analysis.get("pairwise_significance", {}),
        "algorithm_summary": algorithm_summary,
        "component_proof": component_proof,
        "variant_runtime_profile": _collect_variant_runtime_profile(records),
        "live_api_condition": bool(llm_variants_use_api),
    }


def _build_uav_proof_summary(
    analysis: Dict[str, Any],
    records: List[Dict[str, Any]],
    llm_variants_use_api: bool,
) -> Dict[str, Any]:
    algorithm_summary = analysis.get("algorithm_summary", {})
    lower_is_better = [
        "mean_system_cost",
        "avg_collision_count",
        "mean_clearance_deficit",
        "mean_time_to_first_feasible_eval",
        "mean_path_length",
        "mean_curvature_cost",
        "mean_altitude_variation",
        "average_rank",
    ]
    higher_is_better = [
        "success_rate",
        "collision_free_rate",
        "mean_min_clearance",
        "wins",
    ]
    component_proof = {
        "llm": _metric_delta(algorithm_summary, "L-ADE-Sniper", "L-ADE-Sniper-no-LLM", lower_is_better, higher_is_better),
        "surrogate": _metric_delta(
            algorithm_summary,
            "L-ADE-Sniper",
            "L-ADE-Sniper-no-Surrogate",
            lower_is_better,
            higher_is_better,
        ),
        "sniper": _metric_delta(
            algorithm_summary,
            "L-ADE-Sniper",
            "L-ADE-Sniper-no-Sniper",
            lower_is_better,
            higher_is_better,
        ),
        "safety_vs_de": _metric_delta(algorithm_summary, "L-ADE-Sniper", "DE", lower_is_better, higher_is_better),
        "stability_vs_rrt": _metric_delta(algorithm_summary, "L-ADE-Sniper", "RRT", lower_is_better, higher_is_better),
    }
    return {
        "suite_summary": analysis.get("suite_summary", {}),
        "pairwise_significance": analysis.get("pairwise_significance", {}),
        "algorithm_summary": algorithm_summary,
        "component_proof": component_proof,
        "variant_runtime_profile": _collect_variant_runtime_profile(records),
        "live_api_condition": bool(llm_variants_use_api),
    }


def _run_wsn_proof(
    args: argparse.Namespace,
    output_root: Path,
    budget_config: Dict[str, Any],
    seeds: List[int],
    llm_variants_use_api: bool,
) -> Dict[str, Any]:
    wsn_root = output_root / "wsn"
    manifest_path = wsn_root / "manifest.json"
    raw_dir = wsn_root / "raw"
    suite_output = wsn_root / "suite.json"
    analysis_output = wsn_root / "analysis.json"
    proof_output = wsn_root / "proof_summary.json"
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
    manifest = load_json(manifest_path)
    variants = _variant_specs(
        "wsn",
        llm_variants_use_api=llm_variants_use_api,
        proof_profile=str(args.wsn_proof_profile),
        selected_labels=set(str(label) for label in args.wsn_variant_labels) if args.wsn_variant_labels else None,
    )
    records: List[Dict[str, Any]] = []
    for case_data in manifest["cases"]:
        for variant in variants:
            llm_config = _variant_llm_config(args, variant)
            for run_id, seed in enumerate(seeds):
                records.append(
                    run_single_wsn_case(
                        case_data=case_data,
                        algorithm_name=str(variant["algorithm_name"]),
                        run_id=run_id,
                        seed=seed,
                        budget_config=budget_config,
                        llm_config=llm_config,
                        output_dir=str(raw_dir),
                        algorithm_label=str(variant["label"]),
                        algorithm_extra_config={
                            "lade_variant": dict(variant.get("lade_variant", {})),
                            "max_unknown_nodes": int(args.wsn_max_unknown_nodes),
                            "unknown_subset_seed": int(args.seed_base),
                        },
                    )
                )
    aggregated = aggregate_wsn_pipeline(
        raw_dir=str(raw_dir),
        suite_output=str(suite_output),
        manifest_path=str(manifest_path),
        analysis_output=str(analysis_output),
        figures_dir=str(figures_dir),
    )
    analysis = aggregated.get("analysis") or {}
    proof_summary = _build_wsn_proof_summary(
        analysis=analysis,
        records=records,
        llm_variants_use_api=llm_variants_use_api,
    )
    save_json(proof_summary, proof_output)
    return {
        "manifest_path": str(manifest_path),
        "suite_output": str(suite_output),
        "analysis_output": str(analysis_output),
        "proof_output": str(proof_output),
        "figures_dir": str(figures_dir),
        "variants": [str(item["label"]) for item in variants],
        "records": records,
        "proof_summary": proof_summary,
    }


def _run_uav_proof(
    args: argparse.Namespace,
    output_root: Path,
    budget_config: Dict[str, Any],
    seeds: List[int],
    llm_variants_use_api: bool,
) -> Dict[str, Any]:
    uav_root = output_root / "uav"
    manifest_path = uav_root / "manifest.json"
    raw_dir = uav_root / "raw"
    suite_output = uav_root / "suite.json"
    analysis_output = uav_root / "analysis.json"
    proof_output = uav_root / "proof_summary.json"
    figures_dir = uav_root / "figures"
    build_uav_suite_manifest(
        source_dir=str(args.uav_source_dir),
        output_path=str(manifest_path),
        levels=[str(level) for level in args.uav_levels],
        control_point_dim=int(args.uav_control_point_dim),
        bounds={axis: [float(value) for value in values] for axis, values in _UAV_BOUNDS.items()},
        synthetic_hard_cases=int(args.uav_synthetic_hard_cases),
    )
    manifest = load_json(manifest_path)
    variants = _variant_specs(
        "uav",
        llm_variants_use_api=llm_variants_use_api,
        proof_profile=str(args.uav_proof_profile),
        selected_labels=set(str(label) for label in args.uav_variant_labels) if args.uav_variant_labels else None,
    )
    records: List[Dict[str, Any]] = []
    for scenario in manifest["scenarios"]:
        for variant in variants:
            llm_config = _variant_llm_config(args, variant)
            for run_id, seed in enumerate(seeds):
                records.append(
                    run_single_uav_case(
                        scenario=scenario,
                        algorithm_name=str(variant["algorithm_name"]),
                        run_id=run_id,
                        seed=seed,
                        budget_config=budget_config,
                        llm_config=llm_config,
                        output_dir=str(raw_dir),
                        algorithm_label=str(variant["label"]),
                        algorithm_extra_config={"lade_variant": dict(variant.get("lade_variant", {}))},
                    )
                )
    aggregated = aggregate_uav_pipeline(
        raw_dir=str(raw_dir),
        suite_output=str(suite_output),
        manifest_path=str(manifest_path),
        analysis_output=str(analysis_output),
        figures_dir=str(figures_dir),
    )
    analysis = aggregated.get("analysis") or {}
    proof_summary = _build_uav_proof_summary(
        analysis=analysis,
        records=records,
        llm_variants_use_api=llm_variants_use_api,
    )
    save_json(proof_summary, proof_output)
    return {
        "manifest_path": str(manifest_path),
        "suite_output": str(suite_output),
        "analysis_output": str(analysis_output),
        "proof_output": str(proof_output),
        "figures_dir": str(figures_dir),
        "variants": [str(item["label"]) for item in variants],
        "records": records,
        "proof_summary": proof_summary,
    }


def run_lade_proof_experiments(args: argparse.Namespace) -> Dict[str, Any]:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    budget_config = _budget_config(args)
    seeds = [int(args.seed_base) + offset for offset in range(int(args.runs))]
    api_key_available = _has_api_key(args)
    llm_variants_use_api = bool(args.use_api and api_key_available)
    summary: Dict[str, Any] = {
        "applications": [str(name) for name in args.applications],
        "runs": int(args.runs),
        "seed_base": int(args.seed_base),
        "budget_config": budget_config,
        "api_requested": bool(args.use_api),
        "api_key_available": bool(api_key_available),
        "live_api_condition": bool(llm_variants_use_api),
    }
    if "wsn" in args.applications:
        wsn_result = _run_wsn_proof(
            args=args,
            output_root=output_root,
            budget_config=budget_config,
            seeds=seeds,
            llm_variants_use_api=llm_variants_use_api,
        )
        summary["wsn"] = {
            "manifest_path": wsn_result["manifest_path"],
            "suite_output": wsn_result["suite_output"],
            "analysis_output": wsn_result["analysis_output"],
            "proof_output": wsn_result["proof_output"],
            "figures_dir": wsn_result["figures_dir"],
            "variants": wsn_result["variants"],
            "proof_summary": wsn_result["proof_summary"],
        }
    if "uav" in args.applications:
        uav_result = _run_uav_proof(
            args=args,
            output_root=output_root,
            budget_config=budget_config,
            seeds=seeds,
            llm_variants_use_api=llm_variants_use_api,
        )
        summary["uav"] = {
            "manifest_path": uav_result["manifest_path"],
            "suite_output": uav_result["suite_output"],
            "analysis_output": uav_result["analysis_output"],
            "proof_output": uav_result["proof_output"],
            "figures_dir": uav_result["figures_dir"],
            "variants": uav_result["variants"],
            "proof_summary": uav_result["proof_summary"],
        }
    summary_path = output_root / "lade_proof_summary.json"
    summary["summary_path"] = str(summary_path)
    save_json(summary, summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default=str(_default_output_root()))
    parser.add_argument("--applications", nargs="+", choices=["wsn", "uav"], default=["wsn", "uav"])
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--seed_base", type=int, default=20260407)
    parser.add_argument("--max_fes", type=int, default=120)
    parser.add_argument("--population_size", type=int, default=12)
    parser.add_argument("--max_iter", type=int, default=8)
    parser.add_argument("--llm_call_budget", type=int, default=4)
    parser.add_argument("--llm_cost_budget_usd", type=float, default=0.05)
    parser.add_argument("--use_api", action="store_true")
    parser.add_argument("--api_key", default="")
    parser.add_argument("--base_url", default="https://api.deepseek.com")
    parser.add_argument("--llm_model", default="deepseek-chat")
    parser.add_argument("--wsn_node_counts", nargs="+", type=int, default=[50, 70])
    parser.add_argument("--wsn_anchor_counts", nargs="+", type=int, default=[4, 6])
    parser.add_argument("--wsn_comm_radii", nargs="+", type=float, default=[20.0, 25.0])
    parser.add_argument("--wsn_cases_per_setting", type=int, default=1)
    parser.add_argument("--wsn_area_size", type=float, default=100.0)
    parser.add_argument("--wsn_max_unknown_nodes", type=int, default=0)
    parser.add_argument("--wsn_proof_profile", choices=["default", "engaged"], default="default")
    parser.add_argument("--wsn_variant_labels", nargs="+", default=None)
    parser.add_argument("--uav_proof_profile", choices=["default", "engaged"], default="engaged")
    parser.add_argument("--uav_source_dir", default=str(_default_uav_source_dir()))
    parser.add_argument("--uav_levels", nargs="+", default=["hard"])
    parser.add_argument("--uav_control_point_dim", type=int, default=24)
    parser.add_argument("--uav_synthetic_hard_cases", type=int, default=4)
    args = parser.parse_args()
    run_lade_proof_experiments(args)


if __name__ == "__main__":
    main()
