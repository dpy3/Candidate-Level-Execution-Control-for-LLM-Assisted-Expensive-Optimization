from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np

from safe_llm_saea.cec2022 import get_function
from safe_llm_saea.experiments.run_csa_calibrated_regret_study import (
    BOUNDS,
    DIMENSION,
    _performance_row,
    _save_json,
    _sha256_json,
    run_pso,
)
from safe_llm_saea.experiments.run_csa_prospective_f02_f10 import (
    DEDICATED_BASE_URL_ENV,
    DEDICATED_KEY_ENV,
    EXPOSED_KEY_ACK_ENV,
    ROTATION_ACK_ENV,
    _generic_de_config,
    _sanitize_capture_event,
)
from safe_llm_saea.experiments.run_csa_regret_crossfit import _summarize
from safe_llm_saea.paper_ready_safe_llm_saea import LLMAssistedDE


PROTOCOL_ID = "CSA-PSO-RISK-V2-F03-F08-CONFIRMATION-2026-07-18"
TEST_FIDS = (3, 8)
RUNS = 5
SEED_BASE = 20260718
PSO_BUDGET = 200
CAPTURE_BUDGET = 80
CALLS = 2
PROJECTION_ALPHA = 0.50
PROPOSAL_GENERATION = 1


def _protocol() -> dict[str, Any]:
    payload = {
        "protocol_id": PROTOCOL_ID,
        "status": "frozen_before_confirmation",
        "scope": "PSO-specific semantic risk controller; no DE effectiveness claim",
        "development_functions": [1, 2, 6, 9, 10],
        "confirmation_functions": [3, 8],
        "runs_per_function": RUNS,
        "seed_rule": "20260718 + fid * 100 + run_id",
        "proposal_capture": {
            "model_id": "gpt-5.4",
            "calls_per_run": CALLS,
            "capture_budget": CAPTURE_BUDGET,
            "fresh_calls": True,
            "labels_persisted": False,
        },
        "pso": {
            "evaluation_budget": PSO_BUDGET,
            "population_size": 10,
            "projection_alpha": PROJECTION_ALPHA,
            "commit_policy": "incumbent_improvement",
        },
        "paired_policies": ["strict-zero", "raw_rank_top", "projected_commit"],
        "development_evidence": {
            "runs": 25,
            "raw_rank_top_wtl": [6, 5, 14],
            "raw_mean_relative_change": 0.6791442632368181,
            "projected_commit_wtl": [14, 4, 7],
            "projected_commit_mean_relative_change": -0.065592573774642,
        },
        "success_gate": {
            "projected_wins_exceed_losses": True,
            "projected_mean_bootstrap_upper_below_zero": True,
            "projected_negative_transfer_not_above_raw": True,
            "nonzero_semantic_evaluations": True,
            "all_wrapped_runs_csa_compliant": True,
        },
        "stopping_rule": "execute all ten paired runs; no early stopping or test-time retuning",
        "claim_rule": (
            "A bounded PSO-specific effectiveness claim is eligible only if every "
            "success-gate item passes; otherwise retain the negative result."
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
        if current.get("protocol_sha256") != actual:
            raise RuntimeError("confirmation protocol checksum mismatch")
        if current != expected:
            raise RuntimeError("confirmation protocol differs from frozen implementation")
        return current
    _save_json(expected, path)
    return expected


def _seed(fid: int, run_id: int) -> int:
    return SEED_BASE + int(fid) * 100 + int(run_id)


def _cache_path(root: Path, fid: int, run_id: int) -> Path:
    return root / "proposal_cache" / f"cec_f{fid:02d}_10d_r{run_id:02d}.json"


def _verify_cache(path: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    actual = _sha256_json({
        key: value for key, value in payload.items() if key != "artifact_sha256"
    })
    if payload.get("artifact_sha256") != actual:
        raise RuntimeError(f"proposal cache checksum mismatch: {path}")
    if payload.get("protocol_sha256") != protocol["protocol_sha256"]:
        raise RuntimeError(f"proposal cache protocol mismatch: {path}")
    serialized = json.dumps(payload)
    for forbidden in (
        "raw_response", "evaluated_candidate_objectives",
        "evaluated_candidate_regrets", "incumbent_at_admission",
    ):
        if forbidden in serialized:
            raise RuntimeError(f"proposal cache contains forbidden field {forbidden}: {path}")
    if payload.get("labels_persisted") is not False:
        raise RuntimeError(f"proposal cache label boundary missing: {path}")
    return payload


def _credentials() -> tuple[str, str]:
    key = os.getenv(DEDICATED_KEY_ENV, "").strip()
    base_url = os.getenv(DEDICATED_BASE_URL_ENV, "").strip().rstrip("/")
    acknowledged = (
        os.getenv(ROTATION_ACK_ENV, "").strip().upper() == "YES"
        or os.getenv(EXPOSED_KEY_ACK_ENV, "").strip().upper() == "YES"
    )
    if not key or not base_url or not acknowledged:
        raise RuntimeError("capture requires dedicated credentials and an explicit risk acknowledgement")
    return key, base_url


def capture(root: Path, protocol: dict[str, Any]) -> None:
    api_key, base_url = _credentials()
    for fid in TEST_FIDS:
        objective = get_function(fid, DIMENSION)
        for run_id in range(RUNS):
            path = _cache_path(root, fid, run_id)
            if path.exists():
                _verify_cache(path, protocol)
                continue
            seed = _seed(fid, run_id)
            random.seed(seed)
            np.random.seed(seed)
            config = _generic_de_config(fid, True, CAPTURE_BUDGET, CALLS, "rank_top")
            silent = io.StringIO()
            with contextlib.redirect_stdout(silent), contextlib.redirect_stderr(silent):
                result = LLMAssistedDE(
                    config, api_key=api_key, base_url=base_url,
                    use_api=True, llm_model="gpt-5.4",
                ).optimize(objective, BOUNDS)
            events = result.get("diagnostics", {}).get("proposal_trace", [])
            if int(result.get("api_success_calls", 0)) != CALLS or len(events) != CALLS:
                raise RuntimeError(f"capture did not produce exactly {CALLS} calls: F{fid:02d} r{run_id}")
            sanitized = [_sanitize_capture_event(event) for event in events]
            vectors = [
                vector for event in sanitized
                for vector in event.get("returned_candidate_vectors", [])
            ]
            if not vectors:
                raise RuntimeError(f"capture returned no candidate: F{fid:02d} r{run_id}")
            payload = {
                "protocol_sha256": protocol["protocol_sha256"],
                "fid": fid, "run_id": run_id, "seed": seed,
                "model_id": "gpt-5.4", "base_url": base_url,
                "successful_calls": CALLS,
                "capture_cost_usd": float(result.get("total_cost_usd", 0.0)),
                "capture_total_tokens": int(result.get("total_tokens", 0)),
                "labels_persisted": False,
                "proposal_vectors": vectors,
                "diagnostics": {"proposal_trace": sanitized},
            }
            payload["artifact_sha256"] = _sha256_json(payload)
            _save_json(payload, path)


def _run_path(root: Path, fid: int, run_id: int, policy: str) -> Path:
    return root / "runs" / f"f{fid:02d}_r{run_id:02d}" / f"pso_{policy}.json"


def _load_or_run(path: Path, producer: Any, protocol: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        actual = _sha256_json({
            key: value for key, value in payload.items() if key != "artifact_sha256"
        })
        if payload.get("artifact_sha256") != actual:
            raise RuntimeError(f"run artifact checksum mismatch: {path}")
        if payload.get("protocol_sha256") != protocol["protocol_sha256"]:
            raise RuntimeError(f"run artifact protocol mismatch: {path}")
        return payload
    payload = producer()
    payload["protocol_sha256"] = protocol["protocol_sha256"]
    payload["artifact_sha256"] = _sha256_json(payload)
    _save_json(payload, path)
    return payload


def execute(root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    raw_rows = []
    projected_rows = []
    all_compliant = True
    total_cost = 0.0
    total_tokens = 0
    for fid in TEST_FIDS:
        objective = get_function(fid, DIMENSION)
        for run_id in range(RUNS):
            cache = _verify_cache(_cache_path(root, fid, run_id), protocol)
            proposals = cache["proposal_vectors"]
            total_cost += float(cache.get("capture_cost_usd", 0.0))
            total_tokens += int(cache.get("capture_total_tokens", 0))
            seed = _seed(fid, run_id)
            runs = {}
            for policy in ("strict-zero", "raw_rank_top", "projected_commit"):
                def produce(policy: str = policy) -> dict[str, Any]:
                    if policy == "strict-zero":
                        return run_pso(objective, seed, PSO_BUDGET, proposals, "strict-zero", 1, 1, 0)
                    if policy == "raw_rank_top":
                        return run_pso(objective, seed, PSO_BUDGET, proposals, "rank_top", 1, 1, 0)
                    return run_pso(
                        objective, seed, PSO_BUDGET, proposals, "rank_top", 1, 1, 0,
                        projection_alphas=[PROJECTION_ALPHA],
                        commit_policy="incumbent_improvement",
                        proposal_generation=PROPOSAL_GENERATION,
                    )
                runs[policy] = _load_or_run(
                    _run_path(root, fid, run_id, policy), produce, protocol
                )
            strict_value = float(runs["strict-zero"]["best_fitness"])
            for policy in ("raw_rank_top", "projected_commit"):
                compliant = bool(runs[policy]["compliance"]["csa_compliant"])
                all_compliant = all_compliant and compliant
                row = _performance_row("PSO", policy, run_id, runs[policy], strict_value)
                row["fid"] = fid
                (raw_rows if policy == "raw_rank_top" else projected_rows).append(row)

    raw = _summarize(raw_rows, 2026071801)
    projected = _summarize(projected_rows, 2026071802)
    gates = {
        "projected_wins_exceed_losses": projected["wins"] > projected["losses"],
        "projected_mean_bootstrap_upper_below_zero": (
            projected["mean_relative_change_bootstrap_95"][1] < 0.0
        ),
        "projected_negative_transfer_not_above_raw": (
            projected["negative_transfer_rate"] <= raw["negative_transfer_rate"]
        ),
        "nonzero_semantic_evaluations": projected["semantic_evaluations"] > 0,
        "all_wrapped_runs_csa_compliant": all_compliant,
    }
    report = {
        "status": "complete",
        "protocol_sha256": protocol["protocol_sha256"],
        "capture": {"calls": RUNS * len(TEST_FIDS) * CALLS, "cost_usd": total_cost, "tokens": total_tokens},
        "raw_rank_top": raw,
        "projected_commit": projected,
        "success_gates": gates,
        "bounded_pso_effectiveness_claim_eligible": all(gates.values()),
        "claim_rule": protocol["claim_rule"],
    }
    report["artifact_sha256"] = _sha256_json(report)
    _save_json(report, root / "confirmation_report.json")
    return report


def validate(root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    valid = 0
    for fid in TEST_FIDS:
        probe = float(get_function(fid, DIMENSION)(np.zeros(DIMENSION)))
        if not np.isfinite(probe):
            raise RuntimeError(f"non-finite F{fid:02d} probe")
        for run_id in range(RUNS):
            path = _cache_path(root, fid, run_id)
            if path.exists():
                _verify_cache(path, protocol)
                valid += 1
    return {"protocol_sha256": protocol["protocol_sha256"], "valid_caches": valid, "expected_caches": 10, "live_api_called": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("freeze", "validate", "capture", "execute", "all"), default="validate")
    parser.add_argument("--output_root", default="paper/q1/experiment_outputs/csa_pso_risk_v2_f03_f08_confirmation_20260718")
    args = parser.parse_args()
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    protocol = _load_or_freeze(root / "frozen_protocol.json")
    if args.mode == "freeze":
        print(json.dumps({"protocol_sha256": protocol["protocol_sha256"]}, indent=2))
    elif args.mode == "validate":
        print(json.dumps(validate(root, protocol), indent=2))
    else:
        if args.mode in {"capture", "all"}:
            capture(root, protocol)
        if args.mode in {"execute", "all"}:
            report = execute(root, protocol)
            print(json.dumps({
                "raw_rank_top": report["raw_rank_top"],
                "projected_commit": report["projected_commit"],
                "success_gates": report["success_gates"],
                "bounded_pso_effectiveness_claim_eligible": report["bounded_pso_effectiveness_claim_eligible"],
            }, indent=2))


if __name__ == "__main__":
    main()
