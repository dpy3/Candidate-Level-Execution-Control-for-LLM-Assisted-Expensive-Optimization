from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import requests

from safe_llm_saea.cec2022 import get_function
from safe_llm_saea.csa_protocol import candidate_id
from safe_llm_saea.directional_semantic import (
    build_directional_prompt,
    construct_directional_candidates,
    parse_directional_response,
    summarize_de_state,
    summarize_pso_state,
)
from safe_llm_saea.experiments.run_csa_calibrated_regret_study import (
    BOUNDS,
    DIMENSION,
    _save_json,
    _sha256_json,
)
from safe_llm_saea.experiments.run_csa_prospective_f02_f10 import (
    DEDICATED_BASE_URL_ENV,
    DEDICATED_KEY_ENV,
    EXPOSED_KEY_ACK_ENV,
    ROTATION_ACK_ENV,
)
from safe_llm_saea.paper_ready_safe_llm_saea import ExperimentConfig, LLMAssistedDE


DEVELOPMENT_FIDS = (1, 2, 3, 6, 8, 9, 10)
RUNS = 3
STATE_BUDGET = 60
MODEL_ID = "gpt-5.4"
PROPOSALS = 3
SEED_BASE = 20260718
PROMPT_VERSION = "direction-step-v1"


def _seed(backbone: str, fid: int, run_id: int) -> int:
    return SEED_BASE + fid * 100 + run_id + (0 if backbone == "DE" else 50000)


def _de_snapshot(fid: int, seed: int) -> dict[str, Any]:
    config = ExperimentConfig(
        algorithm_name="directional-development-DE",
        problem_name=f"CEC2022_F{fid:02d}",
        dimension=DIMENSION,
        max_evaluations=STATE_BUDGET,
        population_size=12,
        llm_enhancement=False,
        gating_enabled=False,
        enable_sniper=False,
        use_surrogate=False,
        enforce_full_evaluation_budget=True,
    )
    random.seed(seed)
    np.random.seed(seed)
    silent = io.StringIO()
    optimizer = LLMAssistedDE(config, use_api=False)
    with contextlib.redirect_stdout(silent), contextlib.redirect_stderr(silent):
        result = optimizer.optimize(get_function(fid, DIMENSION), BOUNDS)
    incumbent = np.asarray(result["best_solution"], dtype=float)
    state = summarize_de_state(
        optimizer.population, optimizer.fitness, BOUNDS,
        float(result["evaluations_used"] / STATE_BUDGET),
    )
    return {
        "state": state,
        "state_sha256": _sha256_json(state),
        "incumbent": incumbent,
        "incumbent_id": candidate_id(incumbent),
        "incumbent_objective": float(result["best_fitness"]),
    }


def _pso_snapshot(fid: int, seed: int) -> dict[str, Any]:
    objective = get_function(fid, DIMENSION)
    rng = np.random.default_rng(seed)
    lower = np.asarray([item[0] for item in BOUNDS], dtype=float)
    upper = np.asarray([item[1] for item in BOUNDS], dtype=float)
    span = upper - lower
    positions = rng.uniform(lower, upper, size=(10, DIMENSION))
    velocities = rng.uniform(-0.1 * span, 0.1 * span, size=(10, DIMENSION))
    values = np.asarray([float(objective(position)) for position in positions])
    personal_best = positions.copy()
    personal_values = values.copy()
    evaluations = 10
    while evaluations + 10 <= STATE_BUDGET:
        best_index = int(np.argmin(personal_values))
        global_best = personal_best[best_index].copy()
        r1 = rng.random(size=positions.shape)
        r2 = rng.random(size=positions.shape)
        velocities = (
            0.72 * velocities
            + 1.49 * r1 * (personal_best - positions)
            + 1.49 * r2 * (global_best - positions)
        )
        positions = np.clip(positions + velocities, lower, upper)
        values = np.asarray([float(objective(position)) for position in positions])
        evaluations += 10
        improved = values < personal_values
        personal_best[improved] = positions[improved]
        personal_values[improved] = values[improved]
    best_index = int(np.argmin(personal_values))
    incumbent = personal_best[best_index].copy()
    state = summarize_pso_state(
        positions, values, velocities, BOUNDS, evaluations / STATE_BUDGET
    )
    return {
        "state": state,
        "state_sha256": _sha256_json(state),
        "incumbent": incumbent,
        "incumbent_id": candidate_id(incumbent),
        "incumbent_objective": float(personal_values[best_index]),
    }


def _snapshot(backbone: str, fid: int, seed: int) -> dict[str, Any]:
    return _de_snapshot(fid, seed) if backbone == "DE" else _pso_snapshot(fid, seed)


def _credentials() -> tuple[str, str]:
    key = os.getenv(DEDICATED_KEY_ENV, "").strip()
    base_url = os.getenv(DEDICATED_BASE_URL_ENV, "").strip().rstrip("/")
    acknowledged = (
        os.getenv(ROTATION_ACK_ENV, "").strip().upper() == "YES"
        or os.getenv(EXPOSED_KEY_ACK_ENV, "").strip().upper() == "YES"
    )
    if not key or not base_url or not acknowledged:
        raise RuntimeError("directional capture requires credentials and risk acknowledgement")
    return key, base_url


def _chat_url(base_url: str) -> str:
    return base_url if base_url.endswith("/chat/completions") else base_url + "/chat/completions"


def _call_directional(
    api_key: str, base_url: str, state: dict[str, Any]
) -> tuple[list[Any], dict[str, int], str]:
    system, user, prompt_sha256 = build_directional_prompt(state, PROPOSALS)
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
        "max_tokens": 2048,
    }
    last_error = None
    for _ in range(3):
        try:
            response = requests.post(
                _chat_url(base_url),
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=90,
            )
            if response.status_code >= 400:
                raise RuntimeError(f"directional API HTTP {response.status_code}")
            body = response.json()
            content = (((body.get("choices") or [{}])[0]).get("message") or {}).get("content", "")
            instructions = parse_directional_response(content, int(state["dimension"]), PROPOSALS)
            usage = body.get("usage") or {}
            return instructions, {
                "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                "total_tokens": int(usage.get("total_tokens", 0) or 0),
            }, prompt_sha256
        except Exception as error:
            last_error = error
    raise RuntimeError(f"directional response failed validation after retries: {last_error}")


def _cache_path(root: Path, backbone: str, fid: int, run_id: int) -> Path:
    return root / "direction_cache" / f"{backbone.lower()}_f{fid:02d}_r{run_id:02d}.json"


def capture(root: Path) -> None:
    api_key, base_url = _credentials()
    for backbone in ("DE", "PSO"):
        for fid in DEVELOPMENT_FIDS:
            for run_id in range(RUNS):
                path = _cache_path(root, backbone, fid, run_id)
                if path.exists():
                    _verify_cache(path, backbone, fid, run_id)
                    continue
                seed = _seed(backbone, fid, run_id)
                snapshot = _snapshot(backbone, fid, seed)
                instructions, usage, prompt_sha256 = _call_directional(
                    api_key, base_url, snapshot["state"]
                )
                candidates, lineage, records = construct_directional_candidates(
                    snapshot["incumbent"], instructions, BOUNDS
                )
                payload = {
                    "prompt_version": PROMPT_VERSION,
                    "backbone": backbone,
                    "fid": fid,
                    "run_id": run_id,
                    "seed": seed,
                    "state_budget": STATE_BUDGET,
                    "state_sha256": snapshot["state_sha256"],
                    "incumbent_id": snapshot["incumbent_id"],
                    "prompt_sha256": prompt_sha256,
                    "model_id": MODEL_ID,
                    "base_url": base_url,
                    "usage": usage,
                    "candidate_vectors": [candidate.tolist() for candidate in candidates],
                    "candidate_lineage": lineage,
                    "instructions": records,
                    "objective_labels_persisted": False,
                    "raw_response_persisted": False,
                }
                payload["artifact_sha256"] = _sha256_json(payload)
                _save_json(payload, path)


def _verify_cache(path: Path, backbone: str, fid: int, run_id: int) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    actual = _sha256_json({
        key: value for key, value in payload.items() if key != "artifact_sha256"
    })
    if payload.get("artifact_sha256") != actual:
        raise RuntimeError(f"direction cache checksum mismatch: {path}")
    if (payload.get("backbone"), payload.get("fid"), payload.get("run_id")) != (
        backbone, fid, run_id,
    ):
        raise RuntimeError(f"direction cache identity mismatch: {path}")
    if payload.get("objective_labels_persisted") is not False:
        raise RuntimeError(f"direction cache contains objective labels: {path}")
    return payload


def _legacy_cache(fid: int, run_id: int) -> Path:
    base = Path("paper/q1/experiment_outputs")
    if fid in (1, 6, 9):
        return base / "csa_cmaes_portability/proposal_cache" / f"cec_f{fid:02d}_10d_r{run_id:02d}.json"
    if fid in (2, 10):
        return base / "csa_regret_prospective_f02_f10_20260717_complete/proposal_cache" / f"cec_f{fid:02d}_10d_r{run_id:02d}.json"
    return base / "csa_pso_risk_v2_f03_f08_confirmation_20260718/proposal_cache" / f"cec_f{fid:02d}_10d_r{run_id:02d}.json"


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_regrets = [value for row in rows for value in row["candidate_regrets"]]
    best_regrets = [min(row["candidate_regrets"]) for row in rows]
    return {
        "events": len(rows),
        "candidates": len(candidate_regrets),
        "candidate_improvement_rate": float(np.mean(np.asarray(candidate_regrets) < 0.0)),
        "event_hit_rate": float(np.mean(np.asarray(best_regrets) < 0.0)),
        "mean_best_normalized_regret": float(np.mean(best_regrets)),
        "median_best_normalized_regret": float(np.median(best_regrets)),
    }


def execute(root: Path) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {
        "DE_directional": [], "DE_legacy_absolute": [],
        "PSO_directional": [], "PSO_legacy_absolute": [],
    }
    detailed = []
    for backbone in ("DE", "PSO"):
        for fid in DEVELOPMENT_FIDS:
            objective = get_function(fid, DIMENSION)
            for run_id in range(RUNS):
                seed = _seed(backbone, fid, run_id)
                snapshot = _snapshot(backbone, fid, seed)
                cache = _verify_cache(
                    _cache_path(root, backbone, fid, run_id), backbone, fid, run_id
                )
                if cache["state_sha256"] != snapshot["state_sha256"]:
                    raise RuntimeError("direction cache state does not reproduce")
                if cache["incumbent_id"] != snapshot["incumbent_id"]:
                    raise RuntimeError("direction cache incumbent does not reproduce")
                incumbent_value = float(snapshot["incumbent_objective"])
                scale = max(1.0, abs(incumbent_value))
                directional = [
                    (float(objective(np.asarray(vector))) - incumbent_value) / scale
                    for vector in cache["candidate_vectors"]
                ]
                legacy_payload = json.loads(_legacy_cache(fid, run_id).read_text(encoding="utf-8"))
                legacy_vectors = legacy_payload.get("proposal_vectors", [])[:PROPOSALS]
                legacy = [
                    (float(objective(np.asarray(vector))) - incumbent_value) / scale
                    for vector in legacy_vectors
                    if len(vector) == DIMENSION
                ]
                if len(legacy) != PROPOSALS:
                    raise RuntimeError(f"legacy cache lacks three candidates F{fid:02d} r{run_id}")
                for source, regrets in (("directional", directional), ("legacy_absolute", legacy)):
                    row = {
                        "backbone": backbone, "fid": fid, "run_id": run_id,
                        "source": source, "candidate_regrets": regrets,
                    }
                    grouped[f"{backbone}_{source}"].append(row)
                    detailed.append(row)
    metrics = {name: _metrics(rows) for name, rows in grouped.items()}
    gates = {}
    for backbone in ("DE", "PSO"):
        new = metrics[f"{backbone}_directional"]
        old = metrics[f"{backbone}_legacy_absolute"]
        gates[backbone] = {
            "candidate_rate_gain_at_least_0_10": (
                new["candidate_improvement_rate"] >= old["candidate_improvement_rate"] + 0.10
            ),
            "event_hit_gain_at_least_0_10": (
                new["event_hit_rate"] >= old["event_hit_rate"] + 0.10
            ),
            "directional_candidate_rate_at_least_0_20": (
                new["candidate_improvement_rate"] >= 0.20
            ),
        }
    report = {
        "status": "complete",
        "prompt_version": PROMPT_VERSION,
        "development_fids": list(DEVELOPMENT_FIDS),
        "runs_per_function": RUNS,
        "metrics": metrics,
        "quality_gates": gates,
        "both_backbones_ready": all(all(value.values()) for value in gates.values()),
        "records": detailed,
    }
    report["artifact_sha256"] = _sha256_json(report)
    _save_json(report, root / "directional_development_report.json")
    return report


def validate(root: Path) -> dict[str, Any]:
    valid = 0
    for backbone in ("DE", "PSO"):
        for fid in DEVELOPMENT_FIDS:
            for run_id in range(RUNS):
                path = _cache_path(root, backbone, fid, run_id)
                if path.exists():
                    _verify_cache(path, backbone, fid, run_id)
                    valid += 1
    return {"valid_caches": valid, "expected_caches": 42, "live_api_called": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("validate", "capture", "execute", "all"), default="validate")
    parser.add_argument("--output_root", default="paper/q1/experiment_outputs/directional_candidate_development_20260718")
    args = parser.parse_args()
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    if args.mode == "validate":
        print(json.dumps(validate(root), indent=2))
        return
    if args.mode in {"capture", "all"}:
        capture(root)
    if args.mode in {"execute", "all"}:
        report = execute(root)
        print(json.dumps({
            "metrics": report["metrics"],
            "quality_gates": report["quality_gates"],
            "both_backbones_ready": report["both_backbones_ready"],
        }, indent=2))


if __name__ == "__main__":
    main()
