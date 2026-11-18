from __future__ import annotations

import argparse
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

from safe_llm_saea.application_benchmark.common.io_utils import load_json, make_run_filename, save_json
from safe_llm_saea.application_benchmark.run_lade_proof_experiments import _proof_lade_variant
from safe_llm_saea.application_benchmark.uav.aggregate_uav_results import aggregate_uav_pipeline
from safe_llm_saea.application_benchmark.uav.run_uav_benchmark import run_single_uav_case


CAPTURE_LABEL = "L-ADE-Sniper-proposal-capture"
FULL_REPLAY_LABEL = "L-ADE-Sniper-cached-full"
WEAK_REPLAY_LABEL = "L-ADE-Sniper-cached-weak-control"
STRICT_ZERO_LABEL = "L-ADE-Sniper-cached-strict-zero"


def _budget(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "max_fes": int(args.max_fes),
        "population_size": int(args.population_size),
        "max_iter": int(args.max_iter),
        "llm_call_budget": int(args.llm_call_budget),
        "llm_cost_budget_usd": float(args.llm_cost_budget_usd),
    }


def _llm_config(args: argparse.Namespace, use_api: bool) -> Dict[str, Any]:
    return {
        "enabled_algorithms": {"L-ADE-Sniper"},
        "use_api": bool(use_api),
        "api_key": str(args.api_key),
        "base_url": str(args.base_url),
        "llm_model": str(args.llm_model),
    }


def _source_path(source_raw_dir: Path, case_id: str, run_id: int) -> Path:
    return source_raw_dir / make_run_filename("uav", case_id, CAPTURE_LABEL, run_id)


def _validate_trace(path: Path) -> Dict[str, Any]:
    record = load_json(path)
    diagnostics = record.get("diagnostics", {})
    events = diagnostics.get("proposal_trace") or diagnostics.get("admission_events") or []
    if not events:
        raise ValueError(f"No proposal trace found in {path}")
    missing = [index for index, event in enumerate(events) if "returned_candidate_vectors" not in event]
    if missing:
        raise ValueError(f"Trace {path} predates candidate capture; missing vectors in events {missing}")
    return record


def _run_capture(
    args: argparse.Namespace,
    scenarios: List[Dict[str, Any]],
    seeds: List[int],
    capture_raw_dir: Path,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for scenario in scenarios:
        for run_id, seed in enumerate(seeds):
            output_path = capture_raw_dir / make_run_filename("uav", str(scenario["case_id"]), CAPTURE_LABEL, run_id)
            if output_path.exists():
                records.append(load_json(output_path))
                continue
            records.append(
                run_single_uav_case(
                    scenario=scenario,
                    algorithm_name="L-ADE-Sniper",
                    run_id=run_id,
                    seed=seed,
                    budget_config=_budget(args),
                    llm_config=_llm_config(args, use_api=True),
                    output_dir=str(capture_raw_dir),
                    algorithm_label=CAPTURE_LABEL,
                    algorithm_extra_config={
                        "lade_variant": _proof_lade_variant(
                            "uav",
                            str(args.uav_proof_profile),
                            proposal_trace_mode="capture",
                            enforce_full_evaluation_budget=True,
                            surrogate_min_history=16,
                            use_api=True,
                        )
                    },
                )
            )
    return records


def _run_replay(
    args: argparse.Namespace,
    scenarios: List[Dict[str, Any]],
    seeds: List[int],
    source_raw_dir: Path,
    replay_raw_dir: Path,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for scenario in scenarios:
        for run_id, seed in enumerate(seeds):
            source_path = _source_path(source_raw_dir, str(scenario["case_id"]), run_id)
            source_record = _validate_trace(source_path)
            if int(source_record.get("seed", -1)) != int(seed):
                raise ValueError(f"Seed mismatch for {source_path}: {source_record.get('seed')} != {seed}")
            if dict(source_record.get("budget", {})) != _budget(args):
                raise ValueError(f"Budget mismatch for {source_path}")

            for label, weak_control in (
                (FULL_REPLAY_LABEL, False),
                (WEAK_REPLAY_LABEL, True),
            ):
                output_path = replay_raw_dir / make_run_filename("uav", str(scenario["case_id"]), label, run_id)
                if output_path.exists():
                    records.append(load_json(output_path))
                    continue
                variant = _proof_lade_variant(
                    "uav",
                    str(args.uav_proof_profile),
                    weak_admission_control=weak_control,
                    proposal_trace_mode="replay",
                    proposal_replay_path=str(source_path.resolve()),
                    proposal_replay_strict=True,
                    enforce_full_evaluation_budget=True,
                    surrogate_min_history=16,
                    surrogate_unavailable_policy="response_order",
                    use_api=False,
                )
                records.append(
                    run_single_uav_case(
                        scenario=scenario,
                        algorithm_name="L-ADE-Sniper",
                        run_id=run_id,
                        seed=seed,
                        budget_config=_budget(args),
                        llm_config=_llm_config(args, use_api=False),
                        output_dir=str(replay_raw_dir),
                        algorithm_label=label,
                        algorithm_extra_config={"lade_variant": variant},
                    )
                )
            strict_output = replay_raw_dir / make_run_filename("uav", str(scenario["case_id"]), STRICT_ZERO_LABEL, run_id)
            if strict_output.exists():
                records.append(load_json(strict_output))
            else:
                strict_variant = _proof_lade_variant(
                    "uav",
                    str(args.uav_proof_profile),
                    strict_zero_call=True,
                    llm_enabled=False,
                    force_initial_llm_call=False,
                    enforce_full_evaluation_budget=True,
                    use_api=False,
                )
                records.append(
                    run_single_uav_case(
                        scenario=scenario,
                        algorithm_name="L-ADE-Sniper",
                        run_id=run_id,
                        seed=seed,
                        budget_config=_budget(args),
                        llm_config={**_llm_config(args, use_api=False), "enabled_algorithms": set()},
                        output_dir=str(replay_raw_dir),
                        algorithm_label=STRICT_ZERO_LABEL,
                        algorithm_extra_config={"lade_variant": strict_variant},
                    )
                )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture and replay identical LLM proposal streams on UAV cases.")
    parser.add_argument("--mode", choices=["capture", "replay", "all"], default="replay")
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--source_raw_dir", default="")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed_base", type=int, default=20260407)
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

    manifest = load_json(args.manifest_path)
    scenarios = list(manifest.get("scenarios", []))
    seeds = [int(args.seed_base) + offset for offset in range(int(args.runs))]
    output_root = Path(args.output_root)
    capture_root = output_root / "capture"
    replay_root = output_root / "replay"
    capture_raw_dir = capture_root / "raw"
    replay_raw_dir = replay_root / "raw"

    capture_records: List[Dict[str, Any]] = []
    replay_records: List[Dict[str, Any]] = []
    if args.mode in {"capture", "all"}:
        if not (
            args.api_key
            or os.getenv("ds_api")
            or os.getenv("DEEPSEEK_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        ):
            raise ValueError("An API key is required for capture mode")
        capture_records = _run_capture(args, scenarios, seeds, capture_raw_dir)
    if args.mode in {"replay", "all"}:
        source_raw_dir = Path(args.source_raw_dir) if args.source_raw_dir else capture_raw_dir
        replay_records = _run_replay(args, scenarios, seeds, source_raw_dir, replay_raw_dir)
        aggregate_uav_pipeline(
            raw_dir=str(replay_raw_dir),
            suite_output=str(replay_root / "suite.json"),
            manifest_path=str(args.manifest_path),
            analysis_output=str(replay_root / "analysis.json"),
            figures_dir=str(replay_root / "figures"),
        )

    save_json(
        {
            "mode": args.mode,
            "manifest_path": str(Path(args.manifest_path).resolve()),
            "budget": _budget(args),
            "seeds": seeds,
            "capture_records": len(capture_records),
            "replay_records": len(replay_records),
            "capture_label": CAPTURE_LABEL,
            "full_replay_label": FULL_REPLAY_LABEL,
            "weak_replay_label": WEAK_REPLAY_LABEL,
            "strict_zero_label": STRICT_ZERO_LABEL,
        },
        output_root / "cached_replay_summary.json",
    )


if __name__ == "__main__":
    main()
