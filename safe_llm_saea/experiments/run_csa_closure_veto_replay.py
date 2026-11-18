from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean, median

import numpy as np

from safe_llm_saea.cec2022 import get_bounds, get_function
from safe_llm_saea.experiments.run_csa_representative_closure import _cec_config
from safe_llm_saea.paper_ready_safe_llm_saea import LLMAssistedDE


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _run_replay(
    trace_path: Path,
    fid: int,
    run_id: int,
    seed: int,
    policy: str,
    max_fes: int,
) -> dict:
    config = _cec_config(fid, enabled=True, max_fes=max_fes, calls=2)
    config.algorithm_name = f"L-ADE-CSA-{policy}-replay"
    config.proposal_trace_mode = "replay"
    config.proposal_replay_path = str(trace_path)
    config.proposal_replay_strict = True
    config.surrogate_admission_policy = policy
    config.surrogate_acceptance_margin = 0.0

    function = get_function(fid, 10)
    lo, hi = get_bounds(10)
    _seed(seed)
    optimizer = LLMAssistedDE(config, use_api=False, llm_model="gpt-5.4")
    result = optimizer.optimize(function, [(lo, hi)] * 10)
    events = result["diagnostics"].get("admission_events", [])
    return {
        "case": f"F{fid:02d}-10D",
        "run_id": run_id,
        "seed": seed,
        "policy": policy,
        "objective": float(result["best_fitness"]),
        "evaluations": int(result["evaluations_used"]),
        "semantic_evaluations": sum(int(event.get("evaluated_candidates", 0)) for event in events),
        "population_accepted": sum(int(event.get("population_accepted_candidates", 0)) for event in events),
        "best_updates": sum(int(event.get("best_update_candidates", 0)) for event in events),
        "events": len(events),
        "trace_source": str(trace_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen-proposal CEC quality-veto counterfactual replay.")
    parser.add_argument(
        "--closure_root",
        default="paper/q1/experiment_outputs/csa_representative_closure_v2_llmtoken_20260716",
    )
    parser.add_argument(
        "--output",
        default="paper/q1/experiment_outputs/csa_closure_veto_replay_20260716/report.json",
    )
    parser.add_argument("--max_fes", type=int, default=240)
    args = parser.parse_args()

    closure_root = Path(args.closure_root)
    closure_report = json.loads((closure_root / "representative_closure_report.json").read_text(encoding="utf-8"))
    original = {
        (row["case"], int(row["run_id"]), row["policy"]): row
        for row in closure_report["records"]
        if row["domain"] == "CEC2022"
    }

    records = []
    for fid in (1, 6, 9):
        for run_id in range(3):
            case = f"F{fid:02d}-10D"
            seed = int(original[(case, run_id, "fail_closed")]["seed"])
            trace_path = closure_root / "raw" / f"cec_f{fid:02d}_fail_closed_r{run_id:02d}.json"
            for policy in ("rank_top", "improvement_threshold"):
                records.append(_run_replay(trace_path, fid, run_id, seed, policy, args.max_fes))

    paired = []
    for fid in (1, 6, 9):
        for run_id in range(3):
            case = f"F{fid:02d}-10D"
            zero = float(original[(case, run_id, "strict_zero")]["objective"])
            captured = float(original[(case, run_id, "fail_closed")]["objective"])
            rank = next(r for r in records if r["case"] == case and r["run_id"] == run_id and r["policy"] == "rank_top")
            veto = next(r for r in records if r["case"] == case and r["run_id"] == run_id and r["policy"] == "improvement_threshold")
            paired.append({
                "case": case,
                "run_id": run_id,
                "strict_zero": zero,
                "captured_rank_top": captured,
                "replayed_rank_top": rank["objective"],
                "rank_top_reproduction_error": rank["objective"] - captured,
                "quality_veto": veto["objective"],
                "veto_vs_zero_pct": 100.0 * (veto["objective"] - zero) / max(abs(zero), 1e-12),
                "veto_semantic_evaluations": veto["semantic_evaluations"],
            })

    effects = [row["veto_vs_zero_pct"] for row in paired]
    summary = {
        "paired_runs": len(paired),
        "rank_top_exact_reproductions": sum(abs(row["rank_top_reproduction_error"]) <= 1e-8 for row in paired),
        "quality_veto_wins": sum(row["quality_veto"] < row["strict_zero"] - 1e-9 for row in paired),
        "quality_veto_ties": sum(abs(row["quality_veto"] - row["strict_zero"]) <= 1e-9 for row in paired),
        "quality_veto_losses": sum(row["quality_veto"] > row["strict_zero"] + 1e-9 for row in paired),
        "median_veto_vs_zero_pct": median(effects),
        "mean_veto_vs_zero_pct": mean(effects),
        "quality_veto_semantic_evaluations": sum(row["veto_semantic_evaluations"] for row in paired),
    }
    report = {
        "purpose": "Counterfactual diagnosis only; this does not replace the submitted rank-top policy.",
        "quality_veto": "improvement_threshold with zero margin: require LCB < current incumbent",
        "summary": summary,
        "paired": paired,
        "records": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
