from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from safe_llm_saea.csa_protocol import RuntimeAdmissionGate, audit_csa_compliance, make_trace_event
from safe_llm_saea.experiments import run_directional_bbob_confirmation as study
from safe_llm_saea.experiments import run_partial_direction_intervention_development as development


PROTOCOL_ID = "CSA-PARTIAL-DIRECTION-BBOB-CONFIRMATION-2026-07-19"
INSTANCES = tuple(range(14, 22))
SEED_BASE = 20262119
OUTPUT_ROOT = "paper/q1/experiment_outputs/partial_direction_bbob_confirmation_20260719"
DEVELOPMENT_REPORT = "paper/q1/experiment_outputs/partial_direction_intervention_development_20260719/partial_direction_intervention_report.json"
ALPHA = 0.50
DIVERSITY_FLOOR = 0.95
CONTINUATIONS = 10
CONTINUATION_SEED_BASE = 20262200
_base_protocol = study._protocol


def _protocol() -> dict[str, Any]:
    development_sha256 = hashlib.sha256(Path(DEVELOPMENT_REPORT).read_bytes()).hexdigest()
    payload = _base_protocol()
    payload.pop("protocol_sha256", None)
    payload["protocol_id"] = PROTOCOL_ID
    payload["status"] = "frozen_before_capture"
    payload["benchmark"]["instances"] = list(INSTANCES)
    payload["benchmark"]["exclusion"] = "Instances 1--13 are development or preceding confirmation states."
    payload["intervention"] = {
        "alpha": ALPHA, "diversity_floor": DIVERSITY_FLOOR,
        "continuations_per_state": CONTINUATIONS,
        "DE": "one-shot direction addition to one native best-target trial vector",
        "PSO": "one-shot direction addition to one native global-best velocity",
        "no_worst_individual_replacement": True,
        "native_selection_controls_acceptance": True,
    }
    payload["development_boundary"] = {
        "instances": list(range(1, 14)),
        "development_report_json_sha256": development_sha256,
        "frozen_alpha": ALPHA, "frozen_diversity_floor": DIVERSITY_FLOOR,
    }
    payload["admission"] = {
        "direction_selection": "minimum GP upper regret must be <= 0",
        "diversity_rule": "prospective diversity ratio must be >= 0.95",
        "runtime_authorization": "one-use certificate for each evaluated semantic-derived point",
        "commit_rule": "native DE selection or PSO personal-best update only",
    }
    payload["success_gate"] = {
        "accepted_interventions_nonzero": True,
        "wins_exceed_losses": True,
        "negative_transfer_rate_at_most_0_25": True,
        "all_semantic_evaluations_authorized": True,
        "all_traces_csa_compliant": True,
    }
    payload["stopping_rule"] = (
        "Capture all 64 backbone-state calls and execute all 640 paired state-continuations "
        "without threshold changes, early stopping, or instance substitution."
    )
    payload["claim_rule"] = "A backbone is eligible only when every frozen gate passes."
    payload["protocol_sha256"] = study._sha256_json(payload)
    return payload


def _configure() -> None:
    study.PROTOCOL_ID = PROTOCOL_ID
    study.INSTANCES = INSTANCES
    study.SEED_BASE = SEED_BASE
    study.OUTPUT_ROOT = OUTPUT_ROOT
    study._protocol = _protocol


def _selected_delta(
    backbone: str, fid: int, instance: int, root: Path, protocol: dict[str, Any],
) -> tuple[np.ndarray | None, list[np.ndarray], dict[str, Any]]:
    seed = study._seed(backbone, fid, instance)
    cache = study._verify_cache(study._cache_path(root, backbone, fid, instance), protocol, backbone, fid, instance)
    candidates = [np.asarray(vector, dtype=float) for vector in cache["candidate_vectors"]]
    if backbone == "DE":
        optimizer, _, _ = study._de_runtime(fid, instance, seed)
        incumbent = optimizer.population[optimizer.best_idx].copy()
        history_x, history_y, incumbent_value = optimizer.X_history, optimizer.y_history, float(optimizer.best_fit)
    else:
        state = study._pso_runtime(fid, instance, seed)
        incumbent = state["incumbent"].copy()
        history_x, history_y = state["history_x"], state["history_y"]
        incumbent_value = float(state["incumbent_objective"])
    selected, diagnostics = study._select(
        history_x, history_y, candidates, incumbent_value, seed, study.RISK_MULTIPLIERS[backbone],
    )
    return (None if selected is None else np.asarray(selected) - incumbent), candidates, diagnostics


def _trace(
    proposed: np.ndarray, applied: bool, accepted: bool, best_update: bool,
    authorization: Any | None,
) -> dict[str, Any]:
    admitted = [proposed] if applied else []
    evaluated = admitted
    population_accepted = admitted if accepted else []
    best_updated = admitted if best_update else []
    event = make_trace_event(
        [proposed], [proposed], [proposed], admitted, evaluated,
        population_accepted_vectors=population_accepted,
        best_update_vectors=best_updated,
        runtime_authorization_enforced=applied,
        evaluation_authorizations=[] if authorization is None else [authorization.as_dict()],
        selection_policy="partial_direction_diversity_gate",
    )
    audit = audit_csa_compliance({"strict_zero_call": False, "csa_trace": [event]})
    return {"event": event, "audit": audit}


def _run_de(
    fid: int, instance: int, continuation_seed: int, delta: np.ndarray | None,
) -> tuple[float, dict[str, Any]]:
    seed = study._seed("DE", fid, instance)
    optimizer, objective, _ = study._de_runtime(fid, instance, seed)
    pre_best = float(optimizer.best_fit)
    authorization_holder: dict[str, Any] = {}

    def authorized_evaluator(vector: np.ndarray, wrapped_objective: Any) -> float:
        gate = RuntimeAdmissionGate(
            [vector], [vector], remaining_budget=1,
            event_id=f"PARTIAL-DE-F{fid}-I{instance}-C{continuation_seed}",
            policy="partial_direction_diversity_gate",
        )
        authorization = gate.issue(vector)
        value = float(gate.evaluate(authorization, vector, wrapped_objective))
        authorization_holder.update({"authorization": authorization, "value": value})
        return value

    if delta is not None:
        optimizer.configure_one_shot_directional_injection(
            delta, ALPHA, DIVERSITY_FLOOR, authorized_evaluator=authorized_evaluator,
        )
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
    injection = dict(optimizer.last_directional_injection)
    proposed = np.asarray(injection.get("evaluated_vector", optimizer.population[optimizer.best_idx]), dtype=float)
    applied = bool(injection.get("applied", False))
    accepted = bool(injection.get("accepted", False))
    value = authorization_holder.get("value")
    best_update = bool(applied and value is not None and value < pre_best)
    trace = _trace(proposed, applied, accepted, best_update, authorization_holder.get("authorization")) if delta is not None else None
    injection["trace"] = trace
    injection["authorized"] = bool(not applied or authorization_holder.get("authorization") is not None)
    return float(optimizer.best_fit), injection


def _run_pso(
    fid: int, instance: int, continuation_seed: int, delta: np.ndarray | None,
) -> tuple[float, dict[str, Any]]:
    seed = study._seed("PSO", fid, instance)
    state = study._pso_runtime(fid, instance, seed)
    state["rng"] = np.random.default_rng(continuation_seed)
    injection: dict[str, Any] = {"configured": delta is not None, "applied": False, "accepted": False, "authorized": True, "trace": None}
    first_iteration = True
    while state["evaluations"] + len(state["positions"]) <= study.PSO_FINAL_BUDGET:
        best_index = int(np.argmin(state["personal_values"]))
        pre_best = float(np.min(state["personal_values"]))
        global_best = state["personal_best"][best_index].copy()
        r1 = state["rng"].random(size=state["positions"].shape)
        r2 = state["rng"].random(size=state["positions"].shape)
        velocities = 0.72 * state["velocities"] + 1.49 * r1 * (state["personal_best"] - state["positions"]) + 1.49 * r2 * (global_best - state["positions"])
        positions = np.clip(state["positions"] + velocities, state["lower"], state["upper"])
        proposed = None
        applied = False
        if first_iteration and delta is not None:
            proposed_velocities = velocities.copy()
            proposed_velocities[best_index] += ALPHA * delta
            proposed_positions = positions.copy()
            proposed_positions[best_index] = np.clip(state["positions"][best_index] + proposed_velocities[best_index], state["lower"], state["upper"])
            diversity_before = development._position_diversity(state["positions"])
            diversity_after = development._position_diversity(proposed_positions)
            ratio = diversity_after / max(diversity_before, 1e-12)
            applied = ratio >= DIVERSITY_FLOOR
            proposed = proposed_positions[best_index].copy()
            if applied:
                velocities, positions = proposed_velocities, proposed_positions
            injection.update({"applied": applied, "diversity_ratio": float(ratio), "evaluated_vector": proposed.tolist()})
        values = np.empty(len(positions), dtype=float)
        authorization = None
        for index, position in enumerate(positions):
            if first_iteration and applied and index == best_index:
                gate = RuntimeAdmissionGate(
                    [position], [position], remaining_budget=1,
                    event_id=f"PARTIAL-PSO-F{fid}-I{instance}-C{continuation_seed}",
                    policy="partial_direction_diversity_gate",
                )
                authorization = gate.issue(position)
                values[index] = float(gate.evaluate(authorization, position, state["objective"]))
            else:
                values[index] = float(state["objective"](position))
        state["evaluations"] += len(positions)
        old_target_value = float(state["personal_values"][best_index])
        improved = values < state["personal_values"]
        state["positions"], state["velocities"] = positions, velocities
        state["personal_best"][improved] = positions[improved]
        state["personal_values"][improved] = values[improved]
        if first_iteration and delta is not None:
            accepted = bool(applied and values[best_index] < old_target_value)
            best_update = bool(applied and values[best_index] < pre_best)
            injection.update({"accepted": accepted, "authorized": bool(not applied or authorization is not None)})
            injection["trace"] = _trace(proposed, applied, accepted, best_update, authorization)
        first_iteration = False
    return float(np.min(state["personal_values"])), injection


def _summary(records: list[dict[str, Any]], backbone: str) -> dict[str, Any]:
    rows = [row for row in records if row["backbone"] == backbone]
    effects = [float(row["mean_relative_change"]) for row in rows]
    wins = sum(value < -1e-12 for value in effects)
    losses = sum(value > 1e-12 for value in effects)
    ties = len(effects) - wins - losses
    summary = {
        "states": len(rows), "wins": wins, "ties": ties, "losses": losses,
        "negative_transfer_rate": losses / len(rows),
        "mean_relative_change": float(np.mean(effects)),
        "mean_bootstrap_95": study._bootstrap(effects, SEED_BASE + (0 if backbone == "DE" else 1)),
        "selected_states": sum(bool(row["direction_selected"]) for row in rows),
        "applied_interventions": sum(int(row["applied"]) for row in rows),
        "accepted_interventions": sum(int(row["accepted"]) for row in rows),
        "diversity_rejections": sum(int(row["diversity_rejections"]) for row in rows),
        "all_semantic_evaluations_authorized": all(bool(row["authorized"]) for row in rows),
        "all_traces_csa_compliant": all(bool(row["csa_compliant"]) for row in rows),
    }
    gates = {
        "accepted_interventions_nonzero": summary["accepted_interventions"] > 0,
        "wins_exceed_losses": wins > losses,
        "negative_transfer_rate_at_most_0_25": summary["negative_transfer_rate"] <= 0.25,
        "all_semantic_evaluations_authorized": summary["all_semantic_evaluations_authorized"],
        "all_traces_csa_compliant": summary["all_traces_csa_compliant"],
    }
    scientific_effect_supported = bool(summary["mean_bootstrap_95"][1] < 0.0)
    return {
        "summary": summary,
        "gates": gates,
        "claim_eligible": all(gates.values()),
        "scientific_effect_supported": scientific_effect_supported,
        "interpretation": (
            "Protocol gate passage is not sufficient for a final-performance claim; "
            "the bootstrap upper bound must also be below zero."
        ),
    }


def execute(root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for backbone, runner in (("DE", _run_de), ("PSO", _run_pso)):
        for fid in study.FUNCTIONS:
            for instance in INSTANCES:
                delta, _, selection = _selected_delta(backbone, fid, instance, root, protocol)
                effects = []
                applied = accepted = rejected = 0
                authorized = compliant = True
                for continuation in range(CONTINUATIONS):
                    continuation_seed = CONTINUATION_SEED_BASE + fid * 10000 + instance * 100 + continuation + (0 if backbone == "DE" else 1_000_000)
                    strict, _ = runner(fid, instance, continuation_seed, None)
                    semantic, injection = runner(fid, instance, continuation_seed, delta)
                    effects.append(float((semantic - strict) / max(1.0, abs(strict))))
                    applied += int(bool(injection.get("applied", False)))
                    accepted += int(bool(injection.get("accepted", False)))
                    rejected += int(bool(injection.get("configured", False) and not injection.get("applied", False)))
                    authorized = authorized and bool(injection.get("authorized", True))
                    trace = injection.get("trace")
                    compliant = compliant and bool(trace is None or trace["audit"]["csa_compliant"])
                records.append({
                    "backbone": backbone, "fid": fid, "instance": instance,
                    "direction_selected": delta is not None, "selection": selection,
                    "continuation_effects": effects, "mean_relative_change": float(np.mean(effects)),
                    "applied": applied, "accepted": accepted, "diversity_rejections": rejected,
                    "authorized": authorized, "csa_compliant": compliant,
                })
    backbones = {backbone: _summary(records, backbone) for backbone in ("DE", "PSO")}
    report = {
        "status": "complete", "protocol_sha256": protocol["protocol_sha256"],
        "backbones": backbones, "records": records,
        "any_backbone_claim_eligible": any(value["claim_eligible"] for value in backbones.values()),
        "any_backbone_scientific_effect_supported": any(
            value["claim_eligible"] and value["scientific_effect_supported"]
            for value in backbones.values()
        ),
    }
    report["artifact_sha256"] = study._sha256_json(report)
    study._save_json(report, root / "partial_direction_confirmation_report.json")
    return report


def main() -> None:
    _configure()
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("freeze", "validate", "capture", "execute", "all"), default="validate")
    parser.add_argument("--output_root", default=OUTPUT_ROOT)
    args = parser.parse_args()
    root = Path(args.output_root); root.mkdir(parents=True, exist_ok=True)
    protocol = study._load_or_freeze(root / "frozen_protocol.json")
    if args.mode == "freeze":
        print(json.dumps({"protocol_sha256": protocol["protocol_sha256"]}, indent=2)); return
    if args.mode == "validate":
        valid = sum(study._cache_path(root, b, f, i).exists() for b in ("DE", "PSO") for f in study.FUNCTIONS for i in INSTANCES)
        print(json.dumps({"valid_caches": valid, "expected_caches": 64}, indent=2)); return
    if args.mode in {"capture", "all"}: study.capture(root, protocol)
    if args.mode in {"execute", "all"}:
        report = execute(root, protocol)
        print(json.dumps({"backbones": report["backbones"], "any_backbone_claim_eligible": report["any_backbone_claim_eligible"]}, indent=2))


if __name__ == "__main__":
    main()
