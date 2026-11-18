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


PROTOCOL_ID = "CSA-PARTIAL-DIRECTION-INTERVENTION-DEVELOPMENT-2026-07-19"
ORIGINAL_ROOT = "paper/q1/experiment_outputs/directional_bbob_confirmation_20260718"
EXTENSION_ROOT = "paper/q1/experiment_outputs/directional_bbob_power_extension_20260718"
OUTPUT_ROOT = "paper/q1/experiment_outputs/partial_direction_intervention_development_20260719"
ALPHAS = (0.25, 0.50, 0.75)
DIVERSITY_FLOOR = 0.95
CONTINUATIONS = 10
ORIGINAL_SEED_BASE = 20260718
EXTENSION_SEED_BASE = 20261718
CONTINUATION_SEED_BASE = 20262000


def _seed(seed_base: int, backbone: str, fid: int, instance: int) -> int:
    return seed_base + fid * 100 + instance + (0 if backbone == "DE" else 50000)


def _protocol(source_hashes: list[str]) -> dict[str, Any]:
    payload = {
        "protocol_id": PROTOCOL_ID,
        "status": "frozen_before_execution",
        "analysis_role": "development_only",
        "source_protocol_sha256": source_hashes,
        "benchmark": {
            "family": "BBOB", "functions": list(study.FUNCTIONS),
            "instances": list(range(1, 14)), "dimension": study.DIMENSION,
        },
        "intervention": {
            "alphas": list(ALPHAS), "diversity_floor": DIVERSITY_FLOOR,
            "continuations_per_state": CONTINUATIONS,
            "DE": "add alpha times the selected semantic delta to one native best-target trial vector",
            "PSO": "add alpha times the selected semantic delta to one native global-best velocity",
            "no_worst_individual_replacement": True,
            "native_selection_controls_acceptance": True,
        },
        "pairing": "same reconstructed state, continuation seed, and numerical budget",
        "gate": {
            "accepted_interventions_nonzero": True,
            "wins_exceed_losses": True,
            "negative_transfer_rate_at_most_0_25": True,
        },
        "selection_rule": (
            "Among passing alphas choose lowest negative-transfer rate, then lowest mean "
            "relative change, then the smallest alpha."
        ),
        "stopping_rule": "Execute all instances 1--13 and all three alphas without early stopping.",
        "future_boundary": (
            "Do not capture instances 14--21 unless at least one backbone-alpha pair passes every gate."
        ),
    }
    payload["protocol_sha256"] = study._sha256_json(payload)
    return payload


def _load_or_freeze(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        if current != expected:
            raise RuntimeError("partial-intervention protocol differs from frozen protocol")
        return current
    study._save_json(expected, path)
    return expected


def _source(instance: int, original_root: Path, extension_root: Path) -> tuple[Path, int]:
    return (
        (original_root, ORIGINAL_SEED_BASE)
        if instance <= 5 else (extension_root, EXTENSION_SEED_BASE)
    )


def _selected_delta(
    backbone: str, fid: int, instance: int, source_root: Path,
    source_protocol: dict[str, Any], seed: int,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    cache = study._verify_cache(
        study._cache_path(source_root, backbone, fid, instance),
        source_protocol, backbone, fid, instance,
    )
    candidates = [np.asarray(vector, dtype=float) for vector in cache["candidate_vectors"]]
    if backbone == "DE":
        optimizer, _, _ = study._de_runtime(fid, instance, seed)
        incumbent = optimizer.population[optimizer.best_idx].copy()
        history_x, history_y = optimizer.X_history, optimizer.y_history
        incumbent_value = float(optimizer.best_fit)
    else:
        state = study._pso_runtime(fid, instance, seed)
        incumbent = state["incumbent"].copy()
        history_x, history_y = state["history_x"], state["history_y"]
        incumbent_value = float(state["incumbent_objective"])
    selected, diagnostics = study._select(
        history_x, history_y, candidates, incumbent_value, seed,
        study.RISK_MULTIPLIERS[backbone],
    )
    return (
        None if selected is None else np.asarray(selected, dtype=float) - incumbent,
        diagnostics,
    )


def _run_de(
    fid: int, instance: int, state_seed: int, continuation_seed: int,
    delta: np.ndarray | None, alpha: float | None,
) -> tuple[float, dict[str, Any]]:
    optimizer, objective, _ = study._de_runtime(fid, instance, state_seed)
    random.seed(continuation_seed)
    np.random.seed(continuation_seed)
    optimizer.config.max_evaluations = study.DE_FINAL_BUDGET
    if delta is not None and alpha is not None:
        optimizer.configure_one_shot_directional_injection(delta, alpha, DIVERSITY_FLOOR)
    silent = io.StringIO()
    with contextlib.redirect_stdout(silent), contextlib.redirect_stderr(silent):
        while optimizer.evaluations < study.DE_FINAL_BUDGET:
            before = optimizer.evaluations
            optimizer._evolve(study.BOUNDS, objective)
            if optimizer.evaluations <= before:
                break
    return float(optimizer.best_fit), dict(optimizer.last_directional_injection)


def _position_diversity(positions: np.ndarray) -> float:
    return float(np.mean(np.std(positions, axis=0) / 10.0))


def _run_pso(
    fid: int, instance: int, state_seed: int, continuation_seed: int,
    delta: np.ndarray | None, alpha: float | None,
) -> tuple[float, dict[str, Any]]:
    state = study._pso_runtime(fid, instance, state_seed)
    state["rng"] = np.random.default_rng(continuation_seed)
    injection = {"configured": delta is not None and alpha is not None, "applied": False, "accepted": False}
    first_iteration = True
    while state["evaluations"] + len(state["positions"]) <= study.PSO_FINAL_BUDGET:
        best_index = int(np.argmin(state["personal_values"]))
        global_best = state["personal_best"][best_index].copy()
        r1 = state["rng"].random(size=state["positions"].shape)
        r2 = state["rng"].random(size=state["positions"].shape)
        velocities = (
            0.72 * state["velocities"]
            + 1.49 * r1 * (state["personal_best"] - state["positions"])
            + 1.49 * r2 * (global_best - state["positions"])
        )
        native_positions = np.clip(state["positions"] + velocities, state["lower"], state["upper"])
        positions = native_positions
        applied = False
        if first_iteration and delta is not None and alpha is not None:
            proposed_velocities = velocities.copy()
            proposed_velocities[best_index] += alpha * delta
            proposed_positions = native_positions.copy()
            proposed_positions[best_index] = np.clip(
                state["positions"][best_index] + proposed_velocities[best_index],
                state["lower"], state["upper"],
            )
            diversity_before = _position_diversity(state["positions"])
            diversity_after = _position_diversity(proposed_positions)
            ratio = diversity_after / max(diversity_before, 1e-12)
            applied = ratio >= DIVERSITY_FLOOR
            if applied:
                velocities = proposed_velocities
                positions = proposed_positions
            injection.update({
                "applied": bool(applied), "alpha": float(alpha),
                "diversity_before": diversity_before, "diversity_after": diversity_after,
                "diversity_ratio": float(ratio), "target_index": best_index,
            })
        old_personal_value = float(state["personal_values"][best_index])
        values = np.asarray([float(state["objective"](position)) for position in positions])
        state["evaluations"] += len(positions)
        improved = values < state["personal_values"]
        state["positions"] = positions
        state["velocities"] = velocities
        state["personal_best"][improved] = positions[improved]
        state["personal_values"][improved] = values[improved]
        if first_iteration and applied:
            injection["accepted"] = bool(values[best_index] < old_personal_value)
        first_iteration = False
    return float(np.min(state["personal_values"])), injection


def _summary(state_rows: list[dict[str, Any]], alpha: float, backbone: str) -> dict[str, Any]:
    rows = [row for row in state_rows if row["alpha"] == alpha and row["backbone"] == backbone]
    effects = [float(row["mean_relative_change"]) for row in rows]
    wins = sum(value < -1e-12 for value in effects)
    losses = sum(value > 1e-12 for value in effects)
    ties = len(effects) - wins - losses
    accepted = sum(int(row["accepted_continuations"]) for row in rows)
    summary = {
        "states": len(rows), "wins": wins, "ties": ties, "losses": losses,
        "negative_transfer_rate": losses / len(rows),
        "mean_relative_change": float(np.mean(effects)),
        "mean_bootstrap_95": study._bootstrap(effects, CONTINUATION_SEED_BASE + int(alpha * 100) + (0 if backbone == "DE" else 1)),
        "selected_states": sum(bool(row["direction_selected"]) for row in rows),
        "applied_continuations": sum(int(row["applied_continuations"]) for row in rows),
        "accepted_continuations": accepted,
        "diversity_rejections": sum(int(row["diversity_rejections"]) for row in rows),
    }
    gates = {
        "accepted_interventions_nonzero": accepted > 0,
        "wins_exceed_losses": wins > losses,
        "negative_transfer_rate_at_most_0_25": summary["negative_transfer_rate"] <= 0.25,
    }
    return {"summary": summary, "gates": gates, "passed": all(gates.values())}


def execute(original_root: Path, extension_root: Path, output_root: Path) -> dict[str, Any]:
    original_protocol = study._load_or_freeze(original_root / "frozen_protocol.json")
    extension_protocol = json.loads((extension_root / "frozen_protocol.json").read_text(encoding="utf-8"))
    protocol = _load_or_freeze(
        output_root / "frozen_protocol.json",
        _protocol([original_protocol["protocol_sha256"], extension_protocol["protocol_sha256"]]),
    )
    state_rows: list[dict[str, Any]] = []
    runner = {"DE": _run_de, "PSO": _run_pso}
    for backbone in ("DE", "PSO"):
        for fid in study.FUNCTIONS:
            for instance in range(1, 14):
                source_root, seed_base = _source(instance, original_root, extension_root)
                source_protocol = original_protocol if instance <= 5 else extension_protocol
                state_seed = _seed(seed_base, backbone, fid, instance)
                delta, selection = _selected_delta(
                    backbone, fid, instance, source_root, source_protocol, state_seed,
                )
                strict_by_continuation: dict[int, float] = {}
                for continuation in range(CONTINUATIONS):
                    continuation_seed = (
                        CONTINUATION_SEED_BASE + fid * 10000 + instance * 100 + continuation
                        + (0 if backbone == "DE" else 1_000_000)
                    )
                    strict_by_continuation[continuation], _ = runner[backbone](
                        fid, instance, state_seed, continuation_seed, None, None,
                    )
                for alpha in ALPHAS:
                    effects: list[float] = []
                    applied = accepted = rejected = 0
                    for continuation in range(CONTINUATIONS):
                        continuation_seed = (
                            CONTINUATION_SEED_BASE + fid * 10000 + instance * 100 + continuation
                            + (0 if backbone == "DE" else 1_000_000)
                        )
                        semantic, injection = runner[backbone](
                            fid, instance, state_seed, continuation_seed, delta, alpha,
                        )
                        strict = strict_by_continuation[continuation]
                        effects.append(float((semantic - strict) / max(1.0, abs(strict))))
                        applied += int(bool(injection.get("applied", False)))
                        accepted += int(bool(injection.get("accepted", False)))
                        rejected += int(bool(injection.get("configured", False) and not injection.get("applied", False)))
                    state_rows.append({
                        "backbone": backbone, "fid": fid, "instance": instance,
                        "alpha": alpha, "direction_selected": delta is not None,
                        "selection": selection, "continuation_effects": effects,
                        "mean_relative_change": float(np.mean(effects)),
                        "applied_continuations": applied,
                        "accepted_continuations": accepted,
                        "diversity_rejections": rejected,
                    })
    results = {
        backbone: {str(alpha): _summary(state_rows, alpha, backbone) for alpha in ALPHAS}
        for backbone in ("DE", "PSO")
    }
    selected_policy: dict[str, Any] = {}
    for backbone in ("DE", "PSO"):
        passing = [
            (alpha, results[backbone][str(alpha)]) for alpha in ALPHAS
            if results[backbone][str(alpha)]["passed"]
        ]
        passing.sort(key=lambda item: (
            item[1]["summary"]["negative_transfer_rate"],
            item[1]["summary"]["mean_relative_change"], item[0],
        ))
        selected_policy[backbone] = None if not passing else {"alpha": passing[0][0], **passing[0][1]}
    report = {
        "status": "complete", "protocol_sha256": protocol["protocol_sha256"],
        "results": results, "selected_policy": selected_policy,
        "future_confirmation_eligible": any(value is not None for value in selected_policy.values()),
        "state_records": state_rows,
    }
    report["artifact_sha256"] = study._sha256_json(report)
    study._save_json(report, output_root / "partial_direction_intervention_report.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("freeze", "execute"), default="execute")
    parser.add_argument("--original_root", default=ORIGINAL_ROOT)
    parser.add_argument("--extension_root", default=EXTENSION_ROOT)
    parser.add_argument("--output_root", default=OUTPUT_ROOT)
    args = parser.parse_args()
    original_root, extension_root, output_root = Path(args.original_root), Path(args.extension_root), Path(args.output_root)
    original_protocol = json.loads((original_root / "frozen_protocol.json").read_text(encoding="utf-8"))
    extension_protocol = json.loads((extension_root / "frozen_protocol.json").read_text(encoding="utf-8"))
    protocol = _load_or_freeze(
        output_root / "frozen_protocol.json",
        _protocol([original_protocol["protocol_sha256"], extension_protocol["protocol_sha256"]]),
    )
    if args.mode == "freeze":
        print(json.dumps({"protocol_sha256": protocol["protocol_sha256"]}, indent=2))
        return
    report = execute(original_root, extension_root, output_root)
    print(json.dumps({
        "results": report["results"], "selected_policy": report["selected_policy"],
        "future_confirmation_eligible": report["future_confirmation_eligible"],
    }, indent=2))


if __name__ == "__main__":
    main()
