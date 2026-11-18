from __future__ import annotations

import argparse
import os
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Tuple

from safe_llm_saea.application_benchmark.common.io_utils import load_json, make_run_filename, save_json
from safe_llm_saea.application_benchmark.run_cached_proposal_replay import CAPTURE_LABEL, _budget, _llm_config, _validate_trace
from safe_llm_saea.application_benchmark.run_lade_proof_experiments import _proof_lade_variant
from safe_llm_saea.application_benchmark.uav.run_uav_benchmark import run_single_uav_case


def _source_path(source_raw_dir: Path, case_id: str, run_id: int) -> Path:
    return source_raw_dir / make_run_filename("uav", case_id, CAPTURE_LABEL, run_id)


def _semantic_evaluations(record: Dict[str, Any]) -> int:
    events = record.get("diagnostics", {}).get("admission_events", [])
    return sum(int(event.get("evaluated_candidates", 0)) for event in events)


def _score_records(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    items = list(records)
    costs = [float(item.get("metrics", {}).get("system_cost", item.get("best_objective", float("inf")))) for item in items]
    feasible = [bool(item.get("metrics", {}).get("success_under_strict_rule", item.get("feasible", False))) for item in items]
    semantic_evaluations = [_semantic_evaluations(item) for item in items]
    return {
        "n_runs": len(items),
        "strict_feasible_runs": sum(feasible),
        "strict_feasible_rate": sum(feasible) / max(1, len(feasible)),
        "mean_system_cost": mean(costs),
        "mean_semantic_evaluations": mean(semantic_evaluations),
        "total_semantic_evaluations": sum(semantic_evaluations),
    }


def _selection_key(item: Dict[str, Any]) -> Tuple[float, float, float, int, float]:
    score = item["score"]
    return (
        -float(score["strict_feasible_rate"]),
        float(score["mean_system_cost"]),
        -float(score["mean_semantic_evaluations"]),
        -int(item["surrogate_min_history"]),
        -float(item["surrogate_acceptance_margin"]),
    )


def capture_development(args: argparse.Namespace, manifest: Dict[str, Any]) -> None:
    if not (args.api_key or os.getenv("ds_api") or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")):
        raise ValueError("An API key is required for development proposal capture")
    raw_dir = Path(args.output_root) / "development_capture" / "raw"
    seeds = [int(args.dev_seed_base) + offset for offset in range(int(args.dev_runs))]
    for scenario in manifest["scenarios"]:
        for run_id, seed in enumerate(seeds):
            output_path = raw_dir / make_run_filename("uav", str(scenario["case_id"]), CAPTURE_LABEL, run_id)
            if output_path.exists():
                continue
            run_single_uav_case(
                scenario=scenario,
                algorithm_name="L-ADE-Sniper",
                run_id=run_id,
                seed=seed,
                budget_config=_budget(args),
                llm_config=_llm_config(args, use_api=True),
                output_dir=str(raw_dir),
                algorithm_label=CAPTURE_LABEL,
                algorithm_extra_config={"lade_variant": _proof_lade_variant(
                    "uav", str(args.uav_proof_profile), proposal_trace_mode="capture",
                    enforce_full_evaluation_budget=True, surrogate_min_history=12, use_api=True,
                )},
            )


def run_development_grid(args: argparse.Namespace, manifest: Dict[str, Any], prereg: Dict[str, Any]) -> Dict[str, Any]:
    source_dir = Path(args.output_root) / "development_capture" / "raw"
    grid_root = Path(args.output_root) / "development_grid"
    seeds = [int(args.dev_seed_base) + offset for offset in range(int(args.dev_runs))]
    summaries: List[Dict[str, Any]] = []
    for readiness in prereg["sensitivity_grid"]["surrogate_min_history"]:
        for margin in prereg["sensitivity_grid"]["surrogate_acceptance_margin"]:
            label = f"CSA-dev-r{int(readiness):02d}-m{int(round(float(margin) * 1000)):03d}"
            raw_dir = grid_root / label / "raw"
            records: List[Dict[str, Any]] = []
            for scenario in manifest["scenarios"]:
                for run_id, seed in enumerate(seeds):
                    source_path = _source_path(source_dir, str(scenario["case_id"]), run_id)
                    source = _validate_trace(source_path)
                    if int(source.get("seed", -1)) != seed or dict(source.get("budget", {})) != _budget(args):
                        raise ValueError(f"Capture fairness mismatch: {source_path}")
                    output_path = raw_dir / make_run_filename("uav", str(scenario["case_id"]), label, run_id)
                    if output_path.exists():
                        records.append(load_json(output_path))
                        continue
                    variant = _proof_lade_variant(
                        "uav", str(args.uav_proof_profile), proposal_trace_mode="replay",
                        proposal_replay_path=str(source_path.resolve()), proposal_replay_strict=True,
                        enforce_full_evaluation_budget=True, surrogate_min_history=int(readiness),
                        surrogate_acceptance_margin=float(margin), use_api=False,
                    )
                    records.append(run_single_uav_case(
                        scenario=scenario, algorithm_name="L-ADE-Sniper", run_id=run_id, seed=seed,
                        budget_config=_budget(args), llm_config=_llm_config(args, use_api=False),
                        output_dir=str(raw_dir), algorithm_label=label,
                        algorithm_extra_config={"lade_variant": variant},
                    ))
            summaries.append({
                "label": label,
                "surrogate_min_history": int(readiness),
                "surrogate_acceptance_margin": float(margin),
                "score": _score_records(records),
            })
    ranked = sorted(summaries, key=_selection_key)
    selected = ranked[0]
    result = {
        "protocol_id": prereg["protocol_id"],
        "selection_rule": prereg["selection_rule"],
        "grid_results": ranked,
        "selected_configuration": selected,
    }
    save_json(result, grid_root / "development_selection.json")
    frozen = {
        "protocol_id": prereg["protocol_id"],
        "locked_before_test": True,
        "surrogate_min_history": selected["surrogate_min_history"],
        "surrogate_acceptance_margin": selected["surrogate_acceptance_margin"],
        "development_score": selected["score"],
        "source": str((grid_root / "development_selection.json").resolve()),
    }
    save_json(frozen, Path(args.output_root) / "frozen_csa_configuration.json")
    return frozen


def run_locked_test(args: argparse.Namespace, test_manifest: Dict[str, Any]) -> Dict[str, Any]:
    frozen_path = Path(args.output_root) / "frozen_csa_configuration.json"
    frozen = load_json(frozen_path)
    source_dir = Path(args.test_source_raw_dir)
    raw_dir = Path(args.output_root) / "locked_test" / "raw"
    label = "CSA-locked-validation"
    seeds = [int(args.test_seed_base) + offset for offset in range(int(args.test_runs))]
    records: List[Dict[str, Any]] = []
    for scenario in test_manifest["scenarios"]:
        for run_id, seed in enumerate(seeds):
            source_path = _source_path(source_dir, str(scenario["case_id"]), run_id)
            source = _validate_trace(source_path)
            if int(source.get("seed", -1)) != seed or dict(source.get("budget", {})) != _budget(args):
                raise ValueError(f"Locked-test fairness mismatch: {source_path}")
            output_path = raw_dir / make_run_filename("uav", str(scenario["case_id"]), label, run_id)
            if output_path.exists():
                raise RuntimeError(f"Locked validation output already exists; refusing a second pass: {output_path}")
            variant = _proof_lade_variant(
                "uav", str(args.uav_proof_profile), proposal_trace_mode="replay",
                proposal_replay_path=str(source_path.resolve()), proposal_replay_strict=True,
                enforce_full_evaluation_budget=True,
                surrogate_min_history=int(frozen["surrogate_min_history"]),
                surrogate_acceptance_margin=float(frozen["surrogate_acceptance_margin"]), use_api=False,
            )
            records.append(run_single_uav_case(
                scenario=scenario, algorithm_name="L-ADE-Sniper", run_id=run_id, seed=seed,
                budget_config=_budget(args), llm_config=_llm_config(args, use_api=False),
                output_dir=str(raw_dir), algorithm_label=label,
                algorithm_extra_config={"lade_variant": variant},
            ))
    report = {
        "protocol_id": frozen["protocol_id"],
        "configuration": frozen,
        "one_pass_lock_enforced": True,
        "case_ids": [item["case_id"] for item in test_manifest["scenarios"]],
        "score": _score_records(records),
    }
    save_json(report, Path(args.output_root) / "locked_test" / "locked_validation_report.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-registered CSA development sensitivity and one-pass locked validation.")
    parser.add_argument("--mode", choices=["capture", "development", "locked", "all"], required=True)
    parser.add_argument("--development_manifest", required=True)
    parser.add_argument("--test_manifest", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--test_source_raw_dir", required=True)
    parser.add_argument("--dev_runs", type=int, default=3)
    parser.add_argument("--dev_seed_base", type=int, default=20260711)
    parser.add_argument("--test_runs", type=int, default=5)
    parser.add_argument("--test_seed_base", type=int, default=20260407)
    parser.add_argument("--max_fes", type=int, default=120)
    parser.add_argument("--population_size", type=int, default=12)
    parser.add_argument("--max_iter", type=int, default=8)
    parser.add_argument("--llm_call_budget", type=int, default=2)
    parser.add_argument("--llm_cost_budget_usd", type=float, default=0.05)
    parser.add_argument("--uav_proof_profile", choices=["default", "engaged"], default="default")
    parser.add_argument("--api_key", default="")
    parser.add_argument("--base_url", default="https://api.deepseek.com")
    parser.add_argument("--llm_model", default="deepseek-chat")
    args = parser.parse_args()
    development_manifest = load_json(args.development_manifest)
    test_manifest = load_json(args.test_manifest)
    prereg = load_json(args.preregistration)
    if args.mode in {"capture", "all"}:
        capture_development(args, development_manifest)
    if args.mode in {"development", "all"}:
        run_development_grid(args, development_manifest, prereg)
    if args.mode in {"locked", "all"}:
        run_locked_test(args, test_manifest)


if __name__ == "__main__":
    main()
