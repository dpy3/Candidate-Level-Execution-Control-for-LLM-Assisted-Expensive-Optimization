from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import random
from importlib.metadata import version
from pathlib import Path
from typing import Any

import ioh
import numpy as np
import requests
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from safe_llm_saea.csa_protocol import (
    RuntimeAdmissionGate,
    audit_csa_compliance,
    candidate_id,
    make_trace_event,
)
from safe_llm_saea.directional_semantic import (
    build_directional_prompt,
    construct_directional_candidates,
    parse_directional_response,
    summarize_de_state,
    summarize_pso_state,
)
from safe_llm_saea.paper_ready_safe_llm_saea import ExperimentConfig, LLMAssistedDE


PROTOCOL_ID = "CSA-DIRECTIONAL-BBOB-CONFIRMATION-2026-07-18"
FUNCTIONS = (1, 8, 15, 21)
INSTANCES = (1, 2, 3, 4, 5)
DIMENSION = 10
BOUNDS = [(-5.0, 5.0)] * DIMENSION
STATE_BUDGET = 60
DE_FINAL_BUDGET = 241
PSO_FINAL_BUDGET = 191
MODEL_ID = "gpt-5.4"
PROPOSALS = 3
SEED_BASE = 20260718
RISK_MULTIPLIERS = {"DE": 1.0, "PSO": 0.5}
OUTPUT_ROOT = "paper/q1/experiment_outputs/directional_bbob_confirmation_20260718"
KEY_ENV = "CSA_PROSPECTIVE_API_KEY"
BASE_URL_ENV = "CSA_PROSPECTIVE_BASE_URL"
ROTATED_ENV = "CSA_PROSPECTIVE_KEY_ROTATED"
EXPOSED_ACK_ENV = "CSA_PROSPECTIVE_EXPOSED_KEY_ACCEPTED"


def _sha256_json(payload: dict[str, Any]) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def _protocol() -> dict[str, Any]:
    payload = {
        "protocol_id": PROTOCOL_ID,
        "status": "frozen_before_capture",
        "benchmark": {
            "family": "BBOB",
            "implementation": "IOHprofiler ioh",
            "ioh_version": version("ioh"),
            "dimension": DIMENSION,
            "function_ids": list(FUNCTIONS),
            "function_roles": {
                "1": "Sphere", "8": "Rosenbrock",
                "15": "rotated Rastrigin", "21": "Gallagher 101 peaks",
            },
            "instances": list(INSTANCES),
        },
        "directional_source": {
            "model_id": MODEL_ID,
            "prompt_version": "direction-step-v1",
            "one_backbone_specific_call_per_case": True,
            "proposals_per_call": PROPOSALS,
            "output_contract": "direction plus local step scale; no absolute coordinates",
            "host_candidate_construction": True,
        },
        "backbones": {
            "DE": {"state_budget": STATE_BUDGET, "final_budget": DE_FINAL_BUDGET, "risk_multiplier": 1.0},
            "PSO": {"state_budget": STATE_BUDGET, "final_budget": PSO_FINAL_BUDGET, "risk_multiplier": 0.5},
        },
        "admission": {
            "rule": "minimum GP upper regret must be <= 0",
            "reserved_evaluation": 1,
            "strict_zero_reserved_action": "incumbent re-evaluation",
            "commit_rule": "population commit only after true incumbent improvement",
            "runtime_authorization": "one-use certificate",
        },
        "development_boundary": {
            "family": "CEC 2022 only",
            "candidate_quality": {
                "DE_improvement_rate": 0.47619047619047616,
                "PSO_improvement_rate": 0.3492063492063492,
            },
            "risk_policy": {
                "DE": "k=1.0", "PSO": "k=0.5",
            },
        },
        "success_gate": {
            "evaluated_semantic_candidates_nonzero": True,
            "wins_exceed_losses": True,
            "negative_transfer_rate_at_most_0_25": True,
            "mean_effect_bootstrap_upper_below_zero": True,
            "all_traces_csa_compliant": True,
        },
        "stopping_rule": "execute all 40 backbone-case pairs; no BBOB retuning or function substitution",
        "claim_rule": "Backbone-specific effectiveness is eligible only for a backbone passing every frozen gate.",
    }
    payload["protocol_sha256"] = _sha256_json(payload)
    return payload


def _load_or_freeze(path: Path) -> dict[str, Any]:
    expected = _protocol()
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        actual = _sha256_json({key: value for key, value in current.items() if key != "protocol_sha256"})
        if current.get("protocol_sha256") != actual or current != expected:
            raise RuntimeError("BBOB protocol differs from frozen implementation")
        return current
    _save_json(expected, path)
    return expected


def _seed(backbone: str, fid: int, instance: int) -> int:
    return SEED_BASE + fid * 100 + instance + (0 if backbone == "DE" else 50000)


def _problem(fid: int, instance: int) -> Any:
    return ioh.get_problem(
        fid, instance=instance, dimension=DIMENSION,
        problem_class=ioh.ProblemClass.BBOB,
    )


def _de_runtime(fid: int, instance: int, seed: int) -> tuple[Any, Any, dict[str, Any]]:
    objective = _problem(fid, instance)
    config = ExperimentConfig(
        algorithm_name="directional-BBOB-DE",
        problem_name=f"BBOB_F{fid:02d}_I{instance:02d}",
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
    with contextlib.redirect_stdout(silent), contextlib.redirect_stderr(silent):
        optimizer = LLMAssistedDE(config, api_key="disabled", use_api=False)
        optimizer.optimize(objective, BOUNDS)
    incumbent = optimizer.population[optimizer.best_idx].copy()
    state = summarize_de_state(
        optimizer.population, optimizer.fitness, BOUNDS,
        optimizer.evaluations / STATE_BUDGET,
    )
    return optimizer, objective, {
        "state": state, "state_sha256": _sha256_json(state),
        "incumbent": incumbent, "incumbent_id": candidate_id(incumbent),
        "incumbent_objective": float(optimizer.best_fit),
    }


def _pso_runtime(fid: int, instance: int, seed: int) -> dict[str, Any]:
    objective = _problem(fid, instance)
    rng = np.random.default_rng(seed)
    lower = np.full(DIMENSION, -5.0)
    upper = np.full(DIMENSION, 5.0)
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
    best_index = int(np.argmin(personal_values))
    incumbent = personal_best[best_index].copy()
    state = summarize_pso_state(positions, values, velocities, BOUNDS, evaluations / STATE_BUDGET)
    return {
        "objective": objective, "rng": rng, "lower": lower, "upper": upper,
        "positions": positions, "velocities": velocities, "values": values,
        "personal_best": personal_best, "personal_values": personal_values,
        "history_x": history_x, "history_y": history_y, "evaluations": evaluations,
        "state": state, "state_sha256": _sha256_json(state),
        "incumbent": incumbent, "incumbent_id": candidate_id(incumbent),
        "incumbent_objective": float(personal_values[best_index]),
    }


def _snapshot(backbone: str, fid: int, instance: int, seed: int) -> dict[str, Any]:
    if backbone == "DE":
        _, _, snapshot = _de_runtime(fid, instance, seed)
        return snapshot
    return _pso_runtime(fid, instance, seed)


def _credentials() -> tuple[str, str]:
    key = os.getenv(KEY_ENV, "").strip()
    base_url = os.getenv(BASE_URL_ENV, "").strip().rstrip("/")
    ack = os.getenv(ROTATED_ENV, "").strip().upper() == "YES" or os.getenv(EXPOSED_ACK_ENV, "").strip().upper() == "YES"
    if not key or not base_url or not ack:
        raise RuntimeError("BBOB capture requires credentials and explicit acknowledgement")
    return key, base_url


def _call(api_key: str, base_url: str, state: dict[str, Any]) -> tuple[list[Any], dict[str, int], str]:
    system, user, prompt_hash = build_directional_prompt(state, PROPOSALS)
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0.2, "max_tokens": 2048,
    }
    url = base_url if base_url.endswith("/chat/completions") else base_url + "/chat/completions"
    last_error = None
    for _ in range(3):
        try:
            response = requests.post(
                url, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload, timeout=90,
            )
            if response.status_code >= 400:
                raise RuntimeError(f"BBOB directional API HTTP {response.status_code}")
            body = response.json()
            content = (((body.get("choices") or [{}])[0]).get("message") or {}).get("content", "")
            instructions = parse_directional_response(content, DIMENSION, PROPOSALS)
            usage = body.get("usage") or {}
            return instructions, {
                "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                "total_tokens": int(usage.get("total_tokens", 0) or 0),
            }, prompt_hash
        except Exception as error:
            last_error = error
    raise RuntimeError(f"BBOB directional response failed validation: {last_error}")


def _cache_path(root: Path, backbone: str, fid: int, instance: int) -> Path:
    return root / "direction_cache" / f"{backbone.lower()}_f{fid:02d}_i{instance:02d}.json"


def _verify_cache(path: Path, protocol: dict[str, Any], backbone: str, fid: int, instance: int) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    actual = _sha256_json({key: value for key, value in payload.items() if key != "artifact_sha256"})
    if payload.get("artifact_sha256") != actual:
        raise RuntimeError(f"BBOB cache checksum mismatch: {path}")
    if payload.get("protocol_sha256") != protocol["protocol_sha256"]:
        raise RuntimeError(f"BBOB cache protocol mismatch: {path}")
    if (payload.get("backbone"), payload.get("fid"), payload.get("instance")) != (backbone, fid, instance):
        raise RuntimeError(f"BBOB cache identity mismatch: {path}")
    if payload.get("objective_labels_persisted") is not False:
        raise RuntimeError(f"BBOB cache label boundary missing: {path}")
    return payload


def capture(root: Path, protocol: dict[str, Any]) -> None:
    api_key, base_url = _credentials()
    for backbone in ("DE", "PSO"):
        for fid in FUNCTIONS:
            for instance in INSTANCES:
                path = _cache_path(root, backbone, fid, instance)
                if path.exists():
                    _verify_cache(path, protocol, backbone, fid, instance)
                    continue
                seed = _seed(backbone, fid, instance)
                snapshot = _snapshot(backbone, fid, instance, seed)
                instructions, usage, prompt_hash = _call(api_key, base_url, snapshot["state"])
                candidates, lineage, records = construct_directional_candidates(snapshot["incumbent"], instructions, BOUNDS)
                payload = {
                    "protocol_sha256": protocol["protocol_sha256"],
                    "backbone": backbone, "fid": fid, "instance": instance, "seed": seed,
                    "state_sha256": snapshot["state_sha256"], "incumbent_id": snapshot["incumbent_id"],
                    "prompt_sha256": prompt_hash, "model_id": MODEL_ID, "base_url": base_url,
                    "usage": usage,
                    "candidate_vectors": [candidate.tolist() for candidate in candidates],
                    "candidate_lineage": lineage, "instructions": records,
                    "objective_labels_persisted": False, "raw_response_persisted": False,
                }
                payload["artifact_sha256"] = _sha256_json(payload)
                _save_json(payload, path)


def _gp_scores(history_x: list[np.ndarray], history_y: list[float], candidates: list[np.ndarray], seed: int) -> tuple[np.ndarray, np.ndarray]:
    model = GaussianProcessRegressor(
        kernel=ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-6, noise_level_bounds="fixed"),
        normalize_y=True, optimizer=None, random_state=seed,
    )
    lower = np.full(DIMENSION, -5.0)
    span = np.full(DIMENSION, 10.0)
    model.fit((np.asarray(history_x) - lower) / span, np.asarray(history_y))
    return model.predict((np.asarray(candidates) - lower) / span, return_std=True)


def _select(history_x: list[np.ndarray], history_y: list[float], candidates: list[np.ndarray], incumbent: float, seed: int, k: float) -> tuple[np.ndarray | None, dict[str, Any]]:
    mu, sigma = _gp_scores(history_x, history_y, candidates, seed)
    scale = max(1.0, abs(incumbent))
    upper = (mu + k * sigma - incumbent) / scale
    index = int(np.argmin(upper))
    return (candidates[index] if upper[index] <= 0.0 else None), {
        "mu": mu.tolist(), "sigma": sigma.tolist(), "risk_upper": upper.tolist(),
        "selected_index": index, "abstained": bool(upper[index] > 0.0),
    }


def _trace(candidates: list[np.ndarray], selected: np.ndarray | None, evaluated: bool, committed: bool, authorization: Any | None) -> dict[str, Any]:
    selected_vectors = [selected] if selected is not None else []
    evaluated_vectors = selected_vectors if evaluated else []
    accepted = evaluated_vectors if committed else []
    event = make_trace_event(
        candidates, candidates, selected_vectors, selected_vectors,
        evaluated_vectors, population_accepted_vectors=accepted,
        best_update_vectors=accepted,
        runtime_authorization_enforced=evaluated,
        evaluation_authorizations=[authorization.as_dict()] if authorization is not None else [],
        selection_policy="directional_risk_upper",
    )
    return {"event": event, "audit": audit_csa_compliance({"strict_zero_call": False, "csa_trace": [event]})}


def _run_de(fid: int, instance: int, root: Path, protocol: dict[str, Any], semantic: bool) -> dict[str, Any]:
    seed = _seed("DE", fid, instance)
    optimizer, objective, snapshot = _de_runtime(fid, instance, seed)
    cache = _verify_cache(_cache_path(root, "DE", fid, instance), protocol, "DE", fid, instance)
    if cache["state_sha256"] != snapshot["state_sha256"] or cache["incumbent_id"] != snapshot["incumbent_id"]:
        raise RuntimeError("DE BBOB state does not reproduce")
    candidates = [np.asarray(vector) for vector in cache["candidate_vectors"]]
    incumbent = float(optimizer.best_fit)
    selected = None
    diagnostics = {"abstained": True}
    if semantic:
        selected, diagnostics = _select(optimizer.X_history, optimizer.y_history, candidates, incumbent, seed, 1.0)
    authorization = None
    evaluated = False
    committed = False
    if selected is not None:
        gate = RuntimeAdmissionGate(candidates, [selected], remaining_budget=1, event_id=f"BBOB-DE-F{fid}-I{instance}", policy="directional_risk_upper")
        authorization = gate.issue(selected)
        value = float(gate.evaluate(authorization, selected, objective))
        evaluated = True
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
            committed = True
    else:
        point = optimizer.population[optimizer.best_idx].copy()
        value = float(objective(point))
        optimizer.evaluations += 1
        optimizer.X_history.append(point)
        optimizer.y_history.append(value)
    optimizer.config.max_evaluations = DE_FINAL_BUDGET
    while optimizer.evaluations < DE_FINAL_BUDGET:
        before = optimizer.evaluations
        optimizer._evolve(BOUNDS, objective)
        if optimizer.evaluations <= before:
            break
    trace = _trace(candidates, selected, evaluated, committed, authorization) if semantic else None
    return {
        "objective": float(optimizer.best_fit), "evaluations": int(optimizer.evaluations),
        "semantic_evaluations": int(evaluated), "abstained": bool(semantic and selected is None),
        "committed": int(committed), "selection": diagnostics, "trace": trace,
    }


def _run_pso(fid: int, instance: int, root: Path, protocol: dict[str, Any], semantic: bool) -> dict[str, Any]:
    seed = _seed("PSO", fid, instance)
    state = _pso_runtime(fid, instance, seed)
    cache = _verify_cache(_cache_path(root, "PSO", fid, instance), protocol, "PSO", fid, instance)
    if cache["state_sha256"] != state["state_sha256"] or cache["incumbent_id"] != state["incumbent_id"]:
        raise RuntimeError("PSO BBOB state does not reproduce")
    candidates = [np.asarray(vector) for vector in cache["candidate_vectors"]]
    incumbent = float(state["incumbent_objective"])
    selected = None
    diagnostics = {"abstained": True}
    if semantic:
        selected, diagnostics = _select(state["history_x"], state["history_y"], candidates, incumbent, seed, 0.5)
    authorization = None
    evaluated = False
    committed = False
    if selected is not None:
        gate = RuntimeAdmissionGate(candidates, [selected], remaining_budget=1, event_id=f"BBOB-PSO-F{fid}-I{instance}", policy="directional_risk_upper")
        authorization = gate.issue(selected)
        value = float(gate.evaluate(authorization, selected, state["objective"]))
        evaluated = True
        state["evaluations"] += 1
        state["history_x"].append(selected.copy())
        state["history_y"].append(value)
        if value < incumbent:
            worst = int(np.argmax(state["personal_values"]))
            state["positions"][worst] = selected.copy()
            state["personal_best"][worst] = selected.copy()
            state["personal_values"][worst] = value
            committed = True
    else:
        point = state["incumbent"].copy()
        value = float(state["objective"](point))
        state["evaluations"] += 1
        state["history_x"].append(point)
        state["history_y"].append(value)
    while state["evaluations"] + 10 <= PSO_FINAL_BUDGET:
        global_best = state["personal_best"][int(np.argmin(state["personal_values"]))].copy()
        r1 = state["rng"].random(size=state["positions"].shape)
        r2 = state["rng"].random(size=state["positions"].shape)
        state["velocities"] = 0.72 * state["velocities"] + 1.49 * r1 * (state["personal_best"] - state["positions"]) + 1.49 * r2 * (global_best - state["positions"])
        state["positions"] = np.clip(state["positions"] + state["velocities"], state["lower"], state["upper"])
        values = np.asarray([float(state["objective"](position)) for position in state["positions"]])
        state["evaluations"] += 10
        improved = values < state["personal_values"]
        state["personal_best"][improved] = state["positions"][improved]
        state["personal_values"][improved] = values[improved]
    trace = _trace(candidates, selected, evaluated, committed, authorization) if semantic else None
    return {
        "objective": float(np.min(state["personal_values"])), "evaluations": int(state["evaluations"]),
        "semantic_evaluations": int(evaluated), "abstained": bool(semantic and selected is None),
        "committed": int(committed), "selection": diagnostics, "trace": trace,
    }


def _bootstrap(values: list[float], seed: int, samples: int = 10000) -> list[float]:
    array = np.asarray(values)
    rng = np.random.default_rng(seed)
    means = np.mean(array[rng.integers(0, len(array), size=(samples, len(array)))], axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def _wilson(successes: int, trials: int) -> list[float]:
    z = float(norm.ppf(0.975)); p = successes / trials; d = 1 + z*z/trials
    center = (p + z*z/(2*trials))/d
    radius = z*np.sqrt(p*(1-p)/trials + z*z/(4*trials*trials))/d
    return [float(max(0, center-radius)), float(min(1, center+radius))]


def execute(root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    reports = {}
    all_records = []
    for backbone, runner in (("DE", _run_de), ("PSO", _run_pso)):
        rows = []
        candidate_regrets = []
        all_compliant = True
        for fid in FUNCTIONS:
            for instance in INSTANCES:
                strict = runner(fid, instance, root, protocol, False)
                semantic = runner(fid, instance, root, protocol, True)
                relative = (semantic["objective"] - strict["objective"]) / max(1.0, abs(strict["objective"]))
                cache = _verify_cache(_cache_path(root, backbone, fid, instance), protocol, backbone, fid, instance)
                snapshot = _snapshot(backbone, fid, instance, _seed(backbone, fid, instance))
                scale = max(1.0, abs(snapshot["incumbent_objective"]))
                objective = _problem(fid, instance)
                regrets = [(float(objective(np.asarray(vector))) - snapshot["incumbent_objective"]) / scale for vector in cache["candidate_vectors"]]
                candidate_regrets.extend(regrets)
                compliant = semantic["trace"] is not None and semantic["trace"]["audit"]["csa_compliant"]
                all_compliant = all_compliant and compliant
                row = {
                    "backbone": backbone, "fid": fid, "instance": instance,
                    "strict_objective": strict["objective"], "semantic_objective": semantic["objective"],
                    "relative_change": float(relative), "semantic_evaluations": semantic["semantic_evaluations"],
                    "abstained": semantic["abstained"], "committed": semantic["committed"],
                    "candidate_regrets": regrets, "csa_compliant": compliant,
                }
                rows.append(row); all_records.append(row)
        effects = [row["relative_change"] for row in rows]
        losses = sum(value > 1e-12 for value in effects)
        wins = sum(value < -1e-12 for value in effects)
        ties = len(effects) - wins - losses
        ci = _bootstrap(effects, SEED_BASE + len(reports))
        summary = {
            "runs": len(rows), "wins": wins, "ties": ties, "losses": losses,
            "negative_transfer_rate": losses / len(rows), "negative_transfer_wilson_95": _wilson(losses, len(rows)),
            "abstention_rate": sum(row["abstained"] for row in rows) / len(rows),
            "semantic_evaluations": sum(row["semantic_evaluations"] for row in rows),
            "committed_candidates": sum(row["committed"] for row in rows),
            "mean_relative_change": float(np.mean(effects)), "mean_bootstrap_95": ci,
            "candidate_improvement_rate": float(np.mean(np.asarray(candidate_regrets) < 0.0)),
            "event_hit_rate": float(np.mean([min(row["candidate_regrets"]) < 0.0 for row in rows])),
            "all_traces_csa_compliant": all_compliant,
        }
        gates = {
            "evaluated_semantic_candidates_nonzero": summary["semantic_evaluations"] > 0,
            "wins_exceed_losses": wins > losses,
            "negative_transfer_rate_at_most_0_25": summary["negative_transfer_rate"] <= 0.25,
            "mean_effect_bootstrap_upper_below_zero": ci[1] < 0.0,
            "all_traces_csa_compliant": all_compliant,
        }
        reports[backbone] = {"summary": summary, "success_gates": gates, "claim_eligible": all(gates.values())}
    report = {
        "status": "complete", "protocol_sha256": protocol["protocol_sha256"],
        "backbones": reports, "records": all_records,
        "any_backbone_claim_eligible": any(value["claim_eligible"] for value in reports.values()),
    }
    report["artifact_sha256"] = _sha256_json(report)
    _save_json(report, root / "bbob_confirmation_report.json")
    return report


def validate(root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    valid = 0
    for backbone in ("DE", "PSO"):
        for fid in FUNCTIONS:
            for instance in INSTANCES:
                path = _cache_path(root, backbone, fid, instance)
                if path.exists():
                    _verify_cache(path, protocol, backbone, fid, instance); valid += 1
    return {"protocol_sha256": protocol["protocol_sha256"], "valid_caches": valid, "expected_caches": 40, "live_api_called": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("freeze", "validate", "capture", "execute", "all"), default="validate")
    parser.add_argument("--output_root", default=OUTPUT_ROOT)
    args = parser.parse_args()
    root = Path(args.output_root); root.mkdir(parents=True, exist_ok=True)
    protocol = _load_or_freeze(root / "frozen_protocol.json")
    if args.mode == "freeze":
        print(json.dumps({"protocol_sha256": protocol["protocol_sha256"]}, indent=2)); return
    if args.mode == "validate":
        print(json.dumps(validate(root, protocol), indent=2)); return
    if args.mode in {"capture", "all"}: capture(root, protocol)
    if args.mode in {"execute", "all"}:
        report = execute(root, protocol)
        print(json.dumps({"backbones": report["backbones"], "any_backbone_claim_eligible": report["any_backbone_claim_eligible"]}, indent=2))


if __name__ == "__main__":
    main()
