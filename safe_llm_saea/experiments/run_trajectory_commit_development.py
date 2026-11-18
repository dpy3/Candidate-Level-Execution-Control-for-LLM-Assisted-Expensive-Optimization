from __future__ import annotations

import argparse
import contextlib
import io
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from safe_llm_saea.experiments import run_directional_bbob_confirmation as study


PROTOCOL_ID = "CSA-TRAJECTORY-COMMIT-DEVELOPMENT-2026-07-19"
SOURCE_ROOT = "paper/q1/experiment_outputs/directional_bbob_confirmation_20260718"
OUTPUT_ROOT = "paper/q1/experiment_outputs/trajectory_commit_development_20260719"
CONTINUATIONS = 10
CONTINUATION_SEED_BASE = 20261900


def _protocol(source_protocol_sha256: str) -> dict[str, Any]:
    payload = {
        "protocol_id": PROTOCOL_ID,
        "status": "frozen_before_trajectory_capture",
        "analysis_role": "development_only",
        "source_protocol_sha256": source_protocol_sha256,
        "benchmark": {
            "family": "BBOB", "functions": list(study.FUNCTIONS),
            "instances": list(study.INSTANCES), "dimension": study.DIMENSION,
        },
        "candidate_source": "all three cached direction-step candidates per state",
        "continuations_per_candidate": CONTINUATIONS,
        "paired_rule": (
            "Reconstruct the same optimizer state and use the same continuation seed for "
            "native and immediate-commit branches."
        ),
        "budget_rule": (
            "Both branches reserve exactly one objective evaluation before continuation; "
            "the native branch re-evaluates its incumbent."
        ),
        "commit_rule": "Commit only after the candidate truly improves the state incumbent.",
        "claim_boundary": "Development labels cannot support a final-performance claim.",
    }
    payload["protocol_sha256"] = study._sha256_json(payload)
    return payload


def _load_or_freeze(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        if current != expected:
            raise RuntimeError("trajectory development protocol differs from frozen protocol")
        return current
    study._save_json(expected, path)
    return expected


def _de_continuation(
    fid: int, instance: int, candidate: np.ndarray | None, continuation_seed: int,
) -> tuple[float, float, bool]:
    seed = study._seed("DE", fid, instance)
    optimizer, objective, _ = study._de_runtime(fid, instance, seed)
    incumbent = float(optimizer.best_fit)
    point = optimizer.population[optimizer.best_idx].copy() if candidate is None else candidate.copy()
    value = float(objective(point))
    optimizer.evaluations += 1
    optimizer.X_history.append(point.copy())
    optimizer.y_history.append(value)
    committed = candidate is not None and value < incumbent
    if committed:
        worst = int(np.argmax(optimizer.fitness))
        optimizer.population[worst] = point.copy()
        optimizer.fitness[worst] = value
        optimizer.best_idx = worst
        optimizer.best_fit = value
        optimizer.last_generation_best = value
    random.seed(continuation_seed)
    np.random.seed(continuation_seed)
    optimizer.config.max_evaluations = study.DE_FINAL_BUDGET
    silent = io.StringIO()
    with contextlib.redirect_stdout(silent), contextlib.redirect_stderr(silent):
        while optimizer.evaluations < study.DE_FINAL_BUDGET:
            before = optimizer.evaluations
            optimizer._evolve(study.BOUNDS, objective)
            if optimizer.evaluations <= before:
                break
    return float(optimizer.best_fit), value, bool(committed)


def _pso_continuation(
    fid: int, instance: int, candidate: np.ndarray | None, continuation_seed: int,
) -> tuple[float, float, bool]:
    seed = study._seed("PSO", fid, instance)
    state = study._pso_runtime(fid, instance, seed)
    incumbent = float(state["incumbent_objective"])
    point = state["incumbent"].copy() if candidate is None else candidate.copy()
    value = float(state["objective"](point))
    state["evaluations"] += 1
    state["history_x"].append(point.copy())
    state["history_y"].append(value)
    committed = candidate is not None and value < incumbent
    if committed:
        worst = int(np.argmax(state["personal_values"]))
        state["positions"][worst] = point.copy()
        state["personal_best"][worst] = point.copy()
        state["personal_values"][worst] = value
    state["rng"] = np.random.default_rng(continuation_seed)
    while state["evaluations"] + len(state["positions"]) <= study.PSO_FINAL_BUDGET:
        global_best = state["personal_best"][int(np.argmin(state["personal_values"]))].copy()
        r1 = state["rng"].random(size=state["positions"].shape)
        r2 = state["rng"].random(size=state["positions"].shape)
        state["velocities"] = (
            0.72 * state["velocities"]
            + 1.49 * r1 * (state["personal_best"] - state["positions"])
            + 1.49 * r2 * (global_best - state["positions"])
        )
        state["positions"] = np.clip(
            state["positions"] + state["velocities"], state["lower"], state["upper"],
        )
        values = np.asarray([float(state["objective"](x)) for x in state["positions"]])
        state["evaluations"] += len(state["positions"])
        improved = values < state["personal_values"]
        state["personal_best"][improved] = state["positions"][improved]
        state["personal_values"][improved] = values[improved]
    return float(np.min(state["personal_values"])), value, bool(committed)


def execute(source_root: Path, output_root: Path) -> dict[str, Any]:
    source_protocol = study._load_or_freeze(source_root / "frozen_protocol.json")
    protocol = _load_or_freeze(
        output_root / "frozen_protocol.json", _protocol(source_protocol["protocol_sha256"]),
    )
    records: list[dict[str, Any]] = []
    for backbone, runner in (("DE", _de_continuation), ("PSO", _pso_continuation)):
        for fid in study.FUNCTIONS:
            for instance in study.INSTANCES:
                cache = study._verify_cache(
                    study._cache_path(source_root, backbone, fid, instance),
                    source_protocol, backbone, fid, instance,
                )
                candidates = [np.asarray(vector, dtype=float) for vector in cache["candidate_vectors"]]
                if backbone == "DE":
                    optimizer, _, _ = study._de_runtime(fid, instance, study._seed(backbone, fid, instance))
                    incumbent = float(optimizer.best_fit)
                    history_x, history_y = optimizer.X_history, optimizer.y_history
                else:
                    state = study._pso_runtime(fid, instance, study._seed(backbone, fid, instance))
                    incumbent = float(state["incumbent_objective"])
                    history_x, history_y = state["history_x"], state["history_y"]
                mu, sigma = study._gp_scores(
                    history_x, history_y, candidates, study._seed(backbone, fid, instance),
                )
                scale = max(1.0, abs(incumbent))
                for candidate_index, candidate in enumerate(candidates):
                    effects: list[float] = []
                    candidate_value = None
                    committed = False
                    for continuation in range(CONTINUATIONS):
                        continuation_seed = (
                            CONTINUATION_SEED_BASE + fid * 10000 + instance * 100
                            + candidate_index * 10 + continuation
                            + (0 if backbone == "DE" else 1_000_000)
                        )
                        strict_final, _, _ = runner(fid, instance, None, continuation_seed)
                        semantic_final, value, did_commit = runner(
                            fid, instance, candidate, continuation_seed,
                        )
                        candidate_value = value
                        committed = did_commit
                        effects.append(
                            float((semantic_final - strict_final) / max(1.0, abs(strict_final)))
                        )
                    instruction = cache["instructions"][candidate_index]
                    records.append({
                        "backbone": backbone, "fid": fid, "instance": instance,
                        "candidate_index": candidate_index,
                        "incumbent_value": incumbent, "candidate_value": candidate_value,
                        "immediate_regret": float((candidate_value - incumbent) / scale),
                        "surrogate_mean_regret": float((mu[candidate_index] - incumbent) / scale),
                        "surrogate_sigma": float(sigma[candidate_index] / scale),
                        "step_scale": float(instruction["step_scale"]),
                        "committed": committed,
                        "continuation_effects": effects,
                        "mean_trajectory_regret": float(np.mean(effects)),
                        "trajectory_loss_rate": float(np.mean(np.asarray(effects) > 1e-12)),
                        "trajectory_win_rate": float(np.mean(np.asarray(effects) < -1e-12)),
                    })
    report = {
        "status": "complete", "protocol_sha256": protocol["protocol_sha256"],
        "candidate_events": len(records),
        "paired_continuations": len(records) * CONTINUATIONS,
        "records": records,
    }
    report["artifact_sha256"] = study._sha256_json(report)
    study._save_json(report, output_root / "trajectory_commit_development_report.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_root", default=SOURCE_ROOT)
    parser.add_argument("--output_root", default=OUTPUT_ROOT)
    args = parser.parse_args()
    report = execute(Path(args.source_root), Path(args.output_root))
    print(json.dumps({
        "protocol_sha256": report["protocol_sha256"],
        "candidate_events": report["candidate_events"],
        "paired_continuations": report["paired_continuations"],
    }, indent=2))


if __name__ == "__main__":
    main()
