from __future__ import annotations

import argparse
import contextlib
import io
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from safe_llm_saea.cec2022 import get_function
from safe_llm_saea.csa_protocol import RuntimeAdmissionGate
from safe_llm_saea.experiments.run_csa_calibrated_regret_study import (
    BOUNDS,
    DIMENSION,
    _gp_scores,
    _save_json,
    _sha256_json,
)
from safe_llm_saea.experiments.run_csa_regret_crossfit import _summarize
from safe_llm_saea.experiments.run_directional_candidate_development import (
    DEVELOPMENT_FIDS,
    RUNS,
    STATE_BUDGET,
    _cache_path,
    _seed,
    _verify_cache,
)
from safe_llm_saea.paper_ready_safe_llm_saea import ExperimentConfig, LLMAssistedDE


DE_FINAL_BUDGET = 241
PSO_FINAL_BUDGET = 191
RISK_MULTIPLIERS = (0.0, 0.5, 1.0)


def _de_runtime(fid: int, seed: int) -> tuple[Any, Any]:
    objective = get_function(fid, DIMENSION)
    config = ExperimentConfig(
        algorithm_name="directional-integration-DE",
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
        optimizer.optimize(objective, BOUNDS)
    return optimizer, objective


def _select_candidate(
    history_x: list[np.ndarray],
    history_y: list[float],
    candidates: list[np.ndarray],
    incumbent: float,
    seed: int,
    policy: str,
    risk_multiplier: float,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    mu, sigma = _gp_scores(history_x, history_y, candidates, seed)
    scale = max(1.0, abs(float(incumbent)))
    lcb = mu - 0.5 * sigma
    upper = (mu + float(risk_multiplier) * sigma - incumbent) / scale
    if policy == "rank_top":
        index = int(np.argmin(lcb))
        selected = candidates[index]
    elif policy == "risk_upper":
        index = int(np.argmin(upper))
        selected = candidates[index] if upper[index] <= 0.0 else None
    else:
        raise ValueError("unsupported directional policy")
    return selected, {
        "mu": mu.tolist(), "sigma": sigma.tolist(),
        "risk_upper": upper.tolist(), "selected_index": index,
        "abstained": selected is None,
    }


def _run_de(
    fid: int, run_id: int, root: Path, policy: str, risk_multiplier: float
) -> dict[str, Any]:
    seed = _seed("DE", fid, run_id)
    optimizer, objective = _de_runtime(fid, seed)
    cache = _verify_cache(_cache_path(root, "DE", fid, run_id), "DE", fid, run_id)
    candidates = [np.asarray(vector, dtype=float) for vector in cache["candidate_vectors"]]
    incumbent = float(optimizer.best_fit)
    selected = None
    diagnostics = {"abstained": True}
    if policy != "strict-zero":
        selected, diagnostics = _select_candidate(
            optimizer.X_history, optimizer.y_history, candidates,
            incumbent, seed, policy, risk_multiplier,
        )
    evaluated = 0
    committed = 0
    if selected is not None:
        gate = RuntimeAdmissionGate(
            candidates, [selected], remaining_budget=1,
            event_id=f"directional-DE-F{fid:02d}-r{run_id}", policy=policy,
        )
        authorization = gate.issue(selected)
        value = float(gate.evaluate(authorization, selected, objective))
        evaluated = 1
        optimizer.evaluations += 1
        optimizer.X_history.append(selected.copy())
        optimizer.y_history.append(value)
        if value < incumbent:
            worst = int(np.argmax(optimizer.fitness))
            optimizer.population[worst] = selected.copy()
            optimizer.fitness[worst] = value
            optimizer.best_idx = worst
            optimizer.best_fit = value
            optimizer.last_generation_best = value
            committed = 1
    else:
        value = float(objective(optimizer.population[optimizer.best_idx]))
        optimizer.evaluations += 1
        optimizer.X_history.append(optimizer.population[optimizer.best_idx].copy())
        optimizer.y_history.append(value)
    optimizer.config.max_evaluations = DE_FINAL_BUDGET
    while optimizer.evaluations < DE_FINAL_BUDGET:
        before = optimizer.evaluations
        optimizer._evolve(BOUNDS, objective)
        if optimizer.evaluations <= before:
            break
    return {
        "objective": float(optimizer.best_fit),
        "evaluations": int(optimizer.evaluations),
        "semantic_evaluations": evaluated,
        "candidate_events": 1,
        "abstained_events": int(selected is None and policy != "strict-zero"),
        "committed_candidates": committed,
        "selection": diagnostics,
    }


def _pso_state(fid: int, seed: int) -> dict[str, Any]:
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
    history_x = [position.copy() for position in positions]
    history_y = [float(value) for value in values]
    evaluations = 10
    while evaluations + 10 <= STATE_BUDGET:
        global_best = personal_best[int(np.argmin(personal_values))].copy()
        r1 = rng.random(size=positions.shape)
        r2 = rng.random(size=positions.shape)
        velocities = 0.72 * velocities + 1.49 * r1 * (personal_best - positions) + 1.49 * r2 * (global_best - positions)
        positions = np.clip(positions + velocities, lower, upper)
        values = np.asarray([float(objective(position)) for position in positions])
        evaluations += 10
        history_x.extend(position.copy() for position in positions)
        history_y.extend(float(value) for value in values)
        improved = values < personal_values
        personal_best[improved] = positions[improved]
        personal_values[improved] = values[improved]
    return locals()


def _run_pso(
    fid: int, run_id: int, root: Path, policy: str, risk_multiplier: float
) -> dict[str, Any]:
    seed = _seed("PSO", fid, run_id)
    state = _pso_state(fid, seed)
    objective = state["objective"]
    cache = _verify_cache(_cache_path(root, "PSO", fid, run_id), "PSO", fid, run_id)
    candidates = [np.asarray(vector, dtype=float) for vector in cache["candidate_vectors"]]
    best_index = int(np.argmin(state["personal_values"]))
    incumbent = float(state["personal_values"][best_index])
    selected = None
    diagnostics = {"abstained": True}
    if policy != "strict-zero":
        selected, diagnostics = _select_candidate(
            state["history_x"], state["history_y"], candidates,
            incumbent, seed, policy, risk_multiplier,
        )
    evaluated = 0
    committed = 0
    if selected is not None:
        gate = RuntimeAdmissionGate(
            candidates, [selected], remaining_budget=1,
            event_id=f"directional-PSO-F{fid:02d}-r{run_id}", policy=policy,
        )
        authorization = gate.issue(selected)
        value = float(gate.evaluate(authorization, selected, objective))
        evaluated = 1
        state["evaluations"] += 1
        state["history_x"].append(selected.copy())
        state["history_y"].append(value)
        if value < incumbent:
            worst = int(np.argmax(state["personal_values"]))
            state["positions"][worst] = selected.copy()
            state["personal_best"][worst] = selected.copy()
            state["personal_values"][worst] = value
            committed = 1
    else:
        point = state["personal_best"][best_index]
        value = float(objective(point))
        state["evaluations"] += 1
        state["history_x"].append(point.copy())
        state["history_y"].append(value)
    while state["evaluations"] + 10 <= PSO_FINAL_BUDGET:
        global_best = state["personal_best"][int(np.argmin(state["personal_values"]))].copy()
        r1 = state["rng"].random(size=state["positions"].shape)
        r2 = state["rng"].random(size=state["positions"].shape)
        state["velocities"] = 0.72 * state["velocities"] + 1.49 * r1 * (state["personal_best"] - state["positions"]) + 1.49 * r2 * (global_best - state["positions"])
        state["positions"] = np.clip(
            state["positions"] + state["velocities"], state["lower"], state["upper"]
        )
        values = np.asarray([float(objective(position)) for position in state["positions"]])
        state["evaluations"] += 10
        improved = values < state["personal_values"]
        state["personal_best"][improved] = state["positions"][improved]
        state["personal_values"][improved] = values[improved]
    return {
        "objective": float(np.min(state["personal_values"])),
        "evaluations": int(state["evaluations"]),
        "semantic_evaluations": evaluated,
        "candidate_events": 1,
        "abstained_events": int(selected is None and policy != "strict-zero"),
        "committed_candidates": committed,
        "selection": diagnostics,
    }


def _performance(backbone: str, policy: str, run: dict[str, Any], strict: float, fid: int, run_id: int) -> dict[str, Any]:
    relative = (float(run["objective"]) - strict) / max(abs(strict), 1e-12)
    return {
        "backbone": backbone, "policy": policy, "fid": fid, "run_id": run_id,
        "objective": float(run["objective"]), "strict_zero_objective": strict,
        "relative_change": float(relative), "negative_transfer": relative > 1e-12,
        "semantic_evaluations": int(run["semantic_evaluations"]),
        "candidate_events": 1, "abstained_events": int(run["abstained_events"]),
        "committed_candidates": int(run["committed_candidates"]),
    }


def execute(root: Path) -> dict[str, Any]:
    summaries = {}
    records = []
    for backbone in ("DE", "PSO"):
        runner = _run_de if backbone == "DE" else _run_pso
        strict_values = {}
        for fid in DEVELOPMENT_FIDS:
            for run_id in range(RUNS):
                strict = runner(fid, run_id, root, "strict-zero", 0.0)
                strict_values[(fid, run_id)] = float(strict["objective"])
        for policy in ("rank_top", "risk_upper"):
            for multiplier in (RISK_MULTIPLIERS if policy == "risk_upper" else (0.0,)):
                rows = []
                for fid in DEVELOPMENT_FIDS:
                    for run_id in range(RUNS):
                        run = runner(fid, run_id, root, policy, multiplier)
                        row = _performance(
                            backbone, f"{policy}_k{multiplier}", run,
                            strict_values[(fid, run_id)], fid, run_id,
                        )
                        rows.append(row)
                        records.append(row)
                key = f"{backbone}_{policy}_k{multiplier}"
                summary = _summarize(rows, 20260718 + len(summaries))
                summary["committed_candidates"] = sum(row["committed_candidates"] for row in rows)
                summaries[key] = summary
    report = {"status": "complete", "summaries": summaries, "records": records}
    report["artifact_sha256"] = _sha256_json(report)
    _save_json(report, root / "directional_integration_report.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default="paper/q1/experiment_outputs/directional_candidate_development_20260718")
    args = parser.parse_args()
    report = execute(Path(args.output_root))
    print(json.dumps(report["summaries"], indent=2))


if __name__ == "__main__":
    main()
