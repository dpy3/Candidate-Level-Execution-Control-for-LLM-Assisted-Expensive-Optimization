from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from safe_llm_saea.experiments import run_csa_pso_risk_v2_confirmation as base
from safe_llm_saea.experiments.run_csa_calibrated_regret_study import _save_json, _sha256_json


PROTOCOL_ID = "CSA-PSO-LATE-COMMIT-F04-F07-CONFIRMATION-2026-07-18"
OUTPUT_ROOT = "paper/q1/experiment_outputs/csa_pso_late_commit_f04_f07_confirmation_20260718"


def _configure() -> None:
    base.PROTOCOL_ID = PROTOCOL_ID
    base.TEST_FIDS = (4, 7)
    base.RUNS = 5
    base.SEED_BASE = 20260718
    base.PSO_BUDGET = 200
    base.CAPTURE_BUDGET = 80
    base.CALLS = 2
    base.PROJECTION_ALPHA = 0.10
    base.PROPOSAL_GENERATION = 18


def _protocol() -> dict[str, Any]:
    payload = {
        "protocol_id": PROTOCOL_ID,
        "status": "frozen_before_final_confirmation",
        "scope": "PSO-specific late semantic risk controller; no DE effectiveness claim",
        "development_functions": [1, 2, 3, 6, 8, 9, 10],
        "confirmation_functions": [4, 7],
        "runs_per_function": 5,
        "seed_rule": "20260718 + fid * 100 + run_id",
        "proposal_capture": {
            "model_id": "gpt-5.4",
            "calls_per_run": 2,
            "capture_budget": 80,
            "fresh_calls": True,
            "labels_persisted": False,
        },
        "pso": {
            "evaluation_budget": 200,
            "population_size": 10,
            "proposal_generation": 18,
            "projection_alpha": 0.10,
            "commit_policy": "incumbent_improvement",
        },
        "paired_policies": ["strict-zero", "early_raw_rank_top", "late_projected_commit"],
        "development_evidence": {
            "runs": 35,
            "late_raw_wtl": [0, 35, 0],
            "late_projection_010_wtl": [12, 23, 0],
            "late_projection_010_mean_relative_change": -0.02399076888969104,
            "late_projection_025_wtl": [8, 27, 0],
            "late_projection_025_mean_relative_change": -0.025106499897648613,
            "selection_rule": "prefer more wins subject to zero development losses",
        },
        "success_gate": {
            "projected_wins_exceed_losses": True,
            "projected_mean_bootstrap_upper_below_zero": True,
            "projected_negative_transfer_not_above_raw": True,
            "nonzero_semantic_evaluations": True,
            "all_wrapped_runs_csa_compliant": True,
        },
        "stopping_rule": (
            "execute all ten paired runs; this is the final confirmation and no "
            "additional function substitution is permitted"
        ),
        "claim_rule": (
            "A bounded PSO-specific late-commit effectiveness claim is eligible only "
            "if every frozen success gate passes; otherwise retain the negative result."
        ),
    }
    payload["protocol_sha256"] = _sha256_json(payload)
    return payload


def _load_or_freeze(path: Path) -> dict[str, Any]:
    expected = _protocol()
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        actual = _sha256_json({
            key: value for key, value in current.items() if key != "protocol_sha256"
        })
        if current.get("protocol_sha256") != actual or current != expected:
            raise RuntimeError("late-commit protocol differs from frozen implementation")
        return current
    _save_json(expected, path)
    return expected


def main() -> None:
    _configure()
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("freeze", "validate", "capture", "execute", "all"), default="validate")
    parser.add_argument("--output_root", default=OUTPUT_ROOT)
    args = parser.parse_args()
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    protocol = _load_or_freeze(root / "frozen_protocol.json")
    if args.mode == "freeze":
        print(json.dumps({"protocol_sha256": protocol["protocol_sha256"]}, indent=2))
        return
    if args.mode == "validate":
        print(json.dumps(base.validate(root, protocol), indent=2))
        return
    if args.mode in {"capture", "all"}:
        base.capture(root, protocol)
    if args.mode in {"execute", "all"}:
        report = base.execute(root, protocol)
        print(json.dumps({
            "raw_rank_top": report["raw_rank_top"],
            "late_projected_commit": report["projected_commit"],
            "success_gates": report["success_gates"],
            "bounded_pso_effectiveness_claim_eligible": report["bounded_pso_effectiveness_claim_eligible"],
        }, indent=2))


if __name__ == "__main__":
    main()
