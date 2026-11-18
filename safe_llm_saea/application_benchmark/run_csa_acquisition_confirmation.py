from __future__ import annotations

import argparse
import os
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple

from safe_llm_saea.application_benchmark.common.io_utils import load_json, make_run_filename, save_json
from safe_llm_saea.application_benchmark.run_cached_proposal_replay import CAPTURE_LABEL, _budget, _llm_config, _validate_trace
from safe_llm_saea.application_benchmark.run_lade_proof_experiments import _proof_lade_variant
from safe_llm_saea.application_benchmark.uav.run_uav_benchmark import run_single_uav_case


def _source_path(raw_dir: Path, case_id: str, run_id: int) -> Path:
    return raw_dir / make_run_filename("uav", case_id, CAPTURE_LABEL, run_id)


def _semantic_evals(record: Dict[str, Any]) -> int:
    return sum(int(event.get("evaluated_candidates", 0)) for event in record.get("diagnostics", {}).get("admission_events", []))


def _score(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    feasible = [bool(item["metrics"].get("success_under_strict_rule", item.get("feasible", False))) for item in records]
    return {
        "n_runs": len(records),
        "strict_feasible_runs": sum(feasible),
        "strict_feasible_rate": sum(feasible) / max(1, len(records)),
        "mean_system_cost": mean(float(item["metrics"]["system_cost"]) for item in records),
        "semantic_evaluations": sum(_semantic_evals(item) for item in records),
    }


def _selection_key(item: Dict[str, Any]) -> Tuple[float, float, int, int, int]:
    score = item["score"]
    return (
        -float(score["strict_feasible_rate"]), float(score["mean_system_cost"]),
        int(score["semantic_evaluations"]),
        0 if item["surrogate_admission_policy"] == "improvement_threshold" else 1,
        -int(item["surrogate_min_history"]),
    )


def _run_replay(
    args: argparse.Namespace, scenarios: List[Dict[str, Any]], seeds: List[int], source_dir: Path,
    output_dir: Path, label: str, readiness: int, policy: str, weak: bool = False,
    refuse_existing: bool = False,
) -> List[Dict[str, Any]]:
    records = []
    for scenario in scenarios:
        for run_id, seed in enumerate(seeds):
            source_path = _source_path(source_dir, str(scenario["case_id"]), run_id)
            source = _validate_trace(source_path)
            if int(source.get("seed", -1)) != seed or dict(source.get("budget", {})) != _budget(args):
                raise ValueError(f"Replay fairness mismatch: {source_path}")
            output_path = output_dir / make_run_filename("uav", str(scenario["case_id"]), label, run_id)
            if output_path.exists():
                if refuse_existing:
                    raise RuntimeError(f"Confirmation output already exists; refusing a second pass: {output_path}")
                records.append(load_json(output_path))
                continue
            variant = _proof_lade_variant(
                "uav", str(args.uav_proof_profile), weak_admission_control=weak,
                proposal_trace_mode="replay", proposal_replay_path=str(source_path.resolve()),
                proposal_replay_strict=True, enforce_full_evaluation_budget=True,
                surrogate_min_history=int(readiness), surrogate_acceptance_margin=0.01,
                surrogate_admission_policy=str(policy), use_api=False,
            )
            records.append(run_single_uav_case(
                scenario=scenario, algorithm_name="L-ADE-Sniper", run_id=run_id, seed=seed,
                budget_config=_budget(args), llm_config=_llm_config(args, use_api=False),
                output_dir=str(output_dir), algorithm_label=label,
                algorithm_extra_config={"lade_variant": variant},
            ))
    return records


def select_on_development(args: argparse.Namespace, manifest: Dict[str, Any], prereg: Dict[str, Any]) -> Dict[str, Any]:
    seeds = [int(args.dev_seed_base) + offset for offset in range(int(args.dev_runs))]
    source_dir = Path(args.development_source_raw_dir)
    root = Path(args.output_root) / "development_policy_grid"
    summaries = []
    grid = prereg["development_selection"]["candidate_grid"]
    for readiness in grid["surrogate_min_history"]:
        for policy in grid["surrogate_admission_policy"]:
            label = f"CSA-acq-dev-{policy}-r{int(readiness)}"
            records = _run_replay(
                args, manifest["scenarios"], seeds, source_dir, root / label / "raw",
                label, int(readiness), str(policy), refuse_existing=False,
            )
            summaries.append({
                "label": label, "surrogate_min_history": int(readiness),
                "surrogate_admission_policy": str(policy), "surrogate_acceptance_margin": 0.01,
                "score": _score(records),
            })
    ranked = sorted(summaries, key=_selection_key)
    result = {"protocol_id": prereg["protocol_id"], "grid_results": ranked, "selected_configuration": ranked[0]}
    save_json(result, root / "development_policy_selection.json")
    frozen = {
        "protocol_id": prereg["protocol_id"], "locked_before_confirmation": True,
        "surrogate_min_history": ranked[0]["surrogate_min_history"],
        "surrogate_admission_policy": ranked[0]["surrogate_admission_policy"],
        "surrogate_acceptance_margin": 0.01, "development_score": ranked[0]["score"],
    }
    save_json(frozen, Path(args.output_root) / "frozen_acquisition_configuration.json")
    return frozen


def capture_confirmation(args: argparse.Namespace, manifest: Dict[str, Any]) -> None:
    if not (args.api_key or os.getenv("ds_api") or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")):
        raise ValueError("An API key is required for confirmation capture")
    raw_dir = Path(args.output_root) / "confirmation_capture" / "raw"
    seeds = [int(args.confirm_seed_base) + offset for offset in range(int(args.confirm_runs))]
    for scenario in manifest["scenarios"]:
        for run_id, seed in enumerate(seeds):
            output_path = raw_dir / make_run_filename("uav", str(scenario["case_id"]), CAPTURE_LABEL, run_id)
            if output_path.exists():
                continue
            run_single_uav_case(
                scenario=scenario, algorithm_name="L-ADE-Sniper", run_id=run_id, seed=seed,
                budget_config=_budget(args), llm_config=_llm_config(args, use_api=True),
                output_dir=str(raw_dir), algorithm_label=CAPTURE_LABEL,
                algorithm_extra_config={"lade_variant": _proof_lade_variant(
                    "uav", str(args.uav_proof_profile), proposal_trace_mode="capture",
                    enforce_full_evaluation_budget=True, surrogate_min_history=24,
                    surrogate_admission_policy="improvement_threshold", use_api=True,
                )},
            )


def confirm_once(args: argparse.Namespace, manifest: Dict[str, Any]) -> Dict[str, Any]:
    frozen = load_json(Path(args.output_root) / "frozen_acquisition_configuration.json")
    seeds = [int(args.confirm_seed_base) + offset for offset in range(int(args.confirm_runs))]
    source_dir = Path(args.output_root) / "confirmation_capture" / "raw"
    raw_dir = Path(args.output_root) / "confirmation_locked" / "raw"
    locked = _run_replay(
        args, manifest["scenarios"], seeds, source_dir, raw_dir, "CSA-acquisition-locked",
        int(frozen["surrogate_min_history"]), str(frozen["surrogate_admission_policy"]),
        refuse_existing=True,
    )
    weak = _run_replay(
        args, manifest["scenarios"], seeds, source_dir, raw_dir, "CSA-confirmation-weak-control",
        int(frozen["surrogate_min_history"]), str(frozen["surrogate_admission_policy"]),
        weak=True, refuse_existing=True,
    )
    report = {"protocol_id": frozen["protocol_id"], "configuration": frozen, "one_pass_lock_enforced": True,
              "locked_score": _score(locked), "weak_score": _score(weak)}
    save_json(report, Path(args.output_root) / "confirmation_locked" / "confirmation_report.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["development", "capture", "confirm", "all"], required=True)
    parser.add_argument("--development_manifest", required=True)
    parser.add_argument("--confirmation_manifest", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--development_source_raw_dir", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--dev_runs", type=int, default=3); parser.add_argument("--dev_seed_base", type=int, default=20260711)
    parser.add_argument("--confirm_runs", type=int, default=5); parser.add_argument("--confirm_seed_base", type=int, default=20260712)
    parser.add_argument("--max_fes", type=int, default=120); parser.add_argument("--population_size", type=int, default=12)
    parser.add_argument("--max_iter", type=int, default=8); parser.add_argument("--llm_call_budget", type=int, default=2)
    parser.add_argument("--llm_cost_budget_usd", type=float, default=0.05)
    parser.add_argument("--uav_proof_profile", choices=["default", "engaged"], default="default")
    parser.add_argument("--api_key", default=""); parser.add_argument("--base_url", default="https://api.deepseek.com")
    parser.add_argument("--llm_model", default="deepseek-chat")
    args = parser.parse_args()
    development = load_json(args.development_manifest); confirmation = load_json(args.confirmation_manifest)
    prereg = load_json(args.preregistration)
    if args.mode in {"development", "all"}: select_on_development(args, development, prereg)
    if args.mode in {"capture", "all"}: capture_confirmation(args, confirmation)
    if args.mode in {"confirm", "all"}: confirm_once(args, confirmation)


if __name__ == "__main__":
    main()
