from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
from statistics import mean, median
from typing import Any, Sequence

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from safe_llm_saea.calibrated_regret import (
    RegretProtocol,
    apply_regret_linear_model,
    admission_metrics,
    calibration_metrics,
    conformal_residual_radius,
    conformal_uncertainty_scale,
    fit_regret_linear_model,
    normalized_regret,
    predicted_regret,
    preregister_abstention_threshold,
    regret_feature_matrix,
)
from safe_llm_saea.backbone_risk import (
    apply_backbone_risk_model,
    backbone_risk_feature_matrix,
)
from safe_llm_saea.cec2022 import get_bounds, get_function
from safe_llm_saea.csa_protocol import (
    CSAContract,
    RuntimeAdmissionGate,
    audit_csa_compliance,
    candidate_id,
    make_trace_event,
)
from safe_llm_saea.experiments.run_csa_representative_closure import _cec_config
from safe_llm_saea.paper_ready_safe_llm_saea import LLMAssistedDE


PROTOCOL_ID = "CSA-CALIBRATED-REGRET-HOLDOUT-2026-07-16"
TRAIN_FID = 1
CALIBRATION_FID = 9
TEST_FID = 6
DIMENSION = 10
BOUNDS = [(-100.0, 100.0)] * DIMENSION
TARGET_COVERAGE = 0.90
RISK_MULTIPLIER = 1.0
MAX_CALIBRATION_NEGATIVE_TRANSFER = 0.25
PSO_CONTRACT = CSAContract(
    proposal_source="frozen LLM proposal stream",
    validator="finite dimension and bound validation",
    numerical_ranker="GP proposal-to-incumbent calibrated-regret risk",
    admission_gate="risk-bounded optional replacement of one PSO particle",
    evaluation_authority="one-use runtime-gated CEC objective evaluator",
    update_authority="numerically evaluated PSO personal/global best updates",
)


def _save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _sha256_json(payload: dict[str, Any]) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _closure_index(root: Path) -> dict[tuple[str, int, str], dict[str, Any]]:
    report = json.loads((root / "representative_closure_report.json").read_text(encoding="utf-8"))
    return {
        (str(row["case"]), int(row["run_id"]), str(row["policy"])): row
        for row in report["records"]
        if row["domain"] == "CEC2022"
    }


def _run_de_replay(
    closure_root: Path,
    fid: int,
    run_id: int,
    seed: int,
    policy: str,
    max_fes: int,
    uncertainty_scale: float = 1.0,
    risk_multiplier: float = 1.0,
    abstention_threshold: float = 0.0,
    regret_model: dict[str, Any] | None = None,
    residual_radius: float = 0.0,
    commit_policy: str = "population_improvement",
    risk_model: dict[str, Any] | None = None,
    risk_probability_threshold: float = 0.5,
    calls: int = 2,
    allow_control_updates: bool = True,
    projection_alpha: float = 0.0,
) -> dict[str, Any]:
    trace_path = closure_root / "raw" / f"cec_f{fid:02d}_fail_closed_r{run_id:02d}.json"
    config = _cec_config(fid, enabled=True, max_fes=max_fes, calls=int(calls))
    config.algorithm_name = f"L-ADE-CSA-{policy}-holdout-replay"
    config.proposal_trace_mode = "replay"
    config.proposal_replay_path = str(trace_path)
    config.proposal_replay_strict = True
    config.surrogate_admission_policy = policy
    config.surrogate_acceptance_margin = 0.0
    config.regret_uncertainty_scale = float(uncertainty_scale)
    config.regret_risk_multiplier = float(risk_multiplier)
    config.regret_abstention_threshold = float(abstention_threshold)
    config.semantic_commit_policy = commit_policy
    config.allow_llm_control_updates = bool(allow_control_updates)
    config.semantic_projection_alpha = float(projection_alpha)
    if risk_model is not None:
        config.risk_model_feature_mean = tuple(risk_model["feature_mean"])
        config.risk_model_feature_scale = tuple(risk_model["feature_scale"])
        config.risk_model_coefficients = tuple(risk_model["coefficients"])
        config.risk_model_intercept = float(risk_model["intercept"])
        config.risk_probability_threshold = float(risk_probability_threshold)
    if regret_model is not None:
        config.regret_model_feature_mean = tuple(regret_model["feature_mean"])
        config.regret_model_feature_scale = tuple(regret_model["feature_scale"])
        config.regret_model_coefficients = tuple(regret_model["coefficients"])
        config.regret_model_residual_radius = float(residual_radius)
    function = get_function(fid, DIMENSION)
    lo, hi = get_bounds(DIMENSION)
    _seed(seed)
    optimizer = LLMAssistedDE(config, use_api=False, llm_model="gpt-5.4")
    result = optimizer.optimize(function, [(lo, hi)] * DIMENSION)
    diagnostics = result["diagnostics"]
    events = diagnostics.get("admission_events", [])
    return {
        "fid": fid,
        "case": f"F{fid:02d}-10D",
        "run_id": run_id,
        "seed": seed,
        "policy": policy,
        "objective": float(result["best_fitness"]),
        "evaluations": int(result["evaluations_used"]),
        "semantic_evaluations": sum(int(event.get("evaluated_candidates", 0)) for event in events),
        "candidate_events": sum(int(event.get("validated_candidates", 0)) > 0 for event in events),
        "abstained_events": sum(bool(event.get("regret_abstained", False)) for event in events),
        "events": events,
        "trace_compliance": audit_csa_compliance({
            "strict_zero_call": False,
            "csa_trace": diagnostics.get("csa_trace", []),
        }),
    }


def _candidate_records(run: dict[str, Any]) -> list[dict[str, Any]]:
    objective = get_function(int(run["fid"]), DIMENSION)
    rows: list[dict[str, Any]] = []
    for event_number, event in enumerate(run["events"]):
        vectors = event.get("validated_candidate_vectors") or []
        mus = event.get("surrogate_mu") or []
        sigmas = event.get("surrogate_sigma") or []
        incumbent = event.get("incumbent_at_admission")
        incumbent_vector = event.get("incumbent_vector_at_admission") or []
        if (
            incumbent is None
            or len(incumbent_vector) != DIMENSION
            or not (len(vectors) == len(mus) == len(sigmas))
        ):
            continue
        features = regret_feature_matrix(
            mus,
            sigmas,
            vectors,
            incumbent_vector,
            BOUNDS,
            float(event.get("progress", 0.0)),
        )
        for candidate_number, (vector, mu, sigma, feature_row) in enumerate(
            zip(vectors, mus, sigmas, features)
        ):
            value = float(objective(np.asarray(vector, dtype=float)))
            rows.append({
                "case": run["case"],
                "fid": int(run["fid"]),
                "run_id": int(run["run_id"]),
                "event_id": f"{run['case']}-r{run['run_id']:02d}-e{event_number:02d}",
                "candidate_number": candidate_number,
                "incumbent": float(incumbent),
                "mu": float(mu),
                "sigma": float(sigma),
                "objective": value,
                "realized_regret": normalized_regret(value, float(incumbent)),
                "features": [float(item) for item in feature_row],
                "candidate_vector": [float(item) for item in vector],
            })
    return rows


def _write_candidate_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case", "fid", "run_id", "event_id", "candidate_number", "incumbent",
        "mu", "sigma", "objective", "realized_regret",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})


def _valid_proposals(vectors: Sequence[Sequence[float]]) -> list[np.ndarray]:
    valid = []
    for vector in vectors:
        candidate = np.asarray(vector, dtype=float)
        if candidate.shape != (DIMENSION,) or not np.all(np.isfinite(candidate)):
            continue
        if all(low <= candidate[index] <= high for index, (low, high) in enumerate(BOUNDS)):
            valid.append(candidate)
    return valid


def _gp_scores(
    history_x: Sequence[np.ndarray],
    history_y: Sequence[float],
    candidates: Sequence[np.ndarray],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    kernel = (
        ConstantKernel(1.0, constant_value_bounds="fixed")
        * Matern(length_scale=1.0, nu=2.5)
        + WhiteKernel(noise_level=1e-6, noise_level_bounds="fixed")
    )
    model = GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, optimizer=None, random_state=seed
    )
    lower = np.asarray([low for low, _ in BOUNDS], dtype=float)
    scale = np.asarray([high - low for low, high in BOUNDS], dtype=float)
    model.fit((np.asarray(history_x) - lower) / scale, np.asarray(history_y, dtype=float))
    return model.predict((np.asarray(candidates) - lower) / scale, return_std=True)


def _trajectory_hash(curve: Sequence[float], solution: Sequence[float]) -> str:
    payload = {
        "curve": [round(float(value), 12) for value in curve],
        "solution": [round(float(value), 12) for value in solution],
    }
    return _sha256_json(payload)


def run_pso(
    objective: Any,
    seed: int,
    max_fes: int,
    proposal_vectors: Sequence[Sequence[float]],
    policy: str,
    uncertainty_scale: float,
    risk_multiplier: float,
    abstention_threshold: float,
    regret_model: dict[str, Any] | None = None,
    residual_radius: float = 0.0,
    projection_alphas: Sequence[float] | None = None,
    commit_policy: str = "population_improvement",
    risk_model: dict[str, Any] | None = None,
    risk_probability_threshold: float = 0.5,
    proposal_generation: int = 1,
) -> dict[str, Any]:
    if commit_policy not in {"population_improvement", "incumbent_improvement"}:
        raise ValueError("unsupported PSO semantic commit policy")
    rng = np.random.default_rng(seed)
    lower = np.asarray([low for low, _ in BOUNDS], dtype=float)
    upper = np.asarray([high for _, high in BOUNDS], dtype=float)
    span = upper - lower
    population_size = 10
    positions = rng.uniform(lower, upper, size=(population_size, DIMENSION))
    velocities = rng.uniform(-0.1 * span, 0.1 * span, size=(population_size, DIMENSION))
    values = np.asarray([float(objective(position)) for position in positions])
    evaluations = population_size
    history_x = [position.copy() for position in positions]
    history_y = [float(value) for value in values]
    personal_best = positions.copy()
    personal_values = values.copy()
    global_index = int(np.argmin(personal_values))
    global_best = personal_best[global_index].copy()
    global_value = float(personal_values[global_index])
    curve = [global_value]
    trace = []
    generation = 0
    proposal_consumed = False

    while evaluations + population_size <= max_fes:
        r1 = rng.random(size=positions.shape)
        r2 = rng.random(size=positions.shape)
        velocities = (
            0.72 * velocities
            + 1.49 * r1 * (personal_best - positions)
            + 1.49 * r2 * (global_best - positions)
        )
        positions = np.clip(positions + velocities, lower, upper)
        selected = None
        selected_index = None
        mu = np.asarray([], dtype=float)
        sigma = np.asarray([], dtype=float)
        risk_upper = np.asarray([], dtype=float)
        valid = []
        candidate_lineage = {}
        abstained = False
        incumbent = float(global_value)
        incumbent_vector = global_best.copy()
        if (
            policy not in {"native", "strict-zero"}
            and not proposal_consumed
            and generation == int(proposal_generation)
        ):
            valid = _valid_proposals(proposal_vectors)
            if valid:
                if projection_alphas:
                    projected = []
                    for candidate in valid:
                        for alpha in projection_alphas:
                            value = float(alpha)
                            if not 0.0 < value <= 1.0:
                                raise ValueError("projection alphas must be in (0, 1]")
                            projected_candidate = np.clip(
                                global_best + value * (candidate - global_best),
                                lower,
                                upper,
                            )
                            projected.append(projected_candidate)
                            candidate_lineage[candidate_id(projected_candidate)] = [
                                candidate_id(candidate)
                            ]
                    valid = _valid_proposals(projected)
                mu, sigma = _gp_scores(history_x, history_y, valid, seed)
                if policy == "rank_top":
                    scores = mu - 0.5 * sigma
                    selected_index = int(np.argmin(scores))
                elif policy == "calibrated_regret":
                    if regret_model is not None:
                        feature_matrix = regret_feature_matrix(
                            mu, sigma, valid, global_best, BOUNDS,
                            evaluations / max(1, max_fes),
                        )
                        predicted = apply_regret_linear_model(feature_matrix, regret_model)
                        risk_upper = predicted + risk_multiplier * float(residual_radius)
                    else:
                        scored = [
                            predicted_regret(m, s, incumbent, uncertainty_scale)
                            for m, s in zip(mu, sigma)
                        ]
                        risk_upper = np.asarray([
                            m + risk_multiplier * s for m, s in scored
                        ], dtype=float)
                    best_index = int(np.argmin(risk_upper))
                    if risk_upper[best_index] <= abstention_threshold:
                        selected_index = best_index
                    else:
                        abstained = True
                elif policy == "backbone_risk":
                    if risk_model is None:
                        raise ValueError("PSO backbone risk model is required")
                    features = backbone_risk_feature_matrix(
                        mu, sigma, valid, global_best, incumbent, BOUNDS,
                        evaluations / max(1, max_fes),
                    )
                    probabilities = apply_backbone_risk_model(features, risk_model)
                    best_index = int(np.argmax(probabilities))
                    if probabilities[best_index] >= risk_probability_threshold:
                        selected_index = best_index
                    else:
                        abstained = True
                else:
                    raise ValueError(f"unsupported PSO policy: {policy}")
                if selected_index is not None:
                    selected = valid[selected_index].copy()
                    positions[-1] = selected.copy()
                proposal_consumed = True

        gate = None
        authorization = None
        if selected is not None:
            gate = RuntimeAdmissionGate(
                [selected], [selected], remaining_budget=max_fes - evaluations,
                event_id=f"pso-generation-{generation}", policy=policy,
            )
            authorization = gate.issue(selected)

        current_values = []
        selected_value = None
        for position in positions:
            if selected is not None and candidate_id(position) == candidate_id(selected):
                value = float(gate.evaluate(authorization, position, objective))
                selected_value = value
            else:
                value = float(objective(position))
            current_values.append(value)
            history_x.append(position.copy())
            history_y.append(value)
        values = np.asarray(current_values, dtype=float)
        evaluations += population_size

        semantic_committed = selected is not None
        if (
            selected is not None
            and commit_policy == "incumbent_improvement"
            and (selected_value is None or selected_value >= incumbent)
        ):
            semantic_committed = False
            positions[-1] = personal_best[-1].copy()
            values[-1] = float(personal_values[-1])

        improved = values < personal_values
        personal_best[improved] = positions[improved]
        personal_values[improved] = values[improved]
        global_index = int(np.argmin(personal_values))
        if float(personal_values[global_index]) < global_value:
            global_value = float(personal_values[global_index])
            global_best = personal_best[global_index].copy()

        if valid and policy not in {"native", "strict-zero"}:
            selected_vectors = [selected] if selected is not None else []
            evaluated_vectors = selected_vectors
            accepted_vectors = selected_vectors if semantic_committed else []
            improved_incumbent = bool(selected_value is not None and selected_value < incumbent)
            event = make_trace_event(
                proposal_vectors,
                valid,
                selected_vectors,
                selected_vectors,
                evaluated_vectors,
                population_accepted_vectors=accepted_vectors,
                best_update_vectors=accepted_vectors if improved_incumbent else [],
                generation=generation,
                selection_policy=policy,
                runtime_authorization_enforced=selected is not None,
                evaluation_authorizations=(
                    [authorization.as_dict()] if authorization is not None else []
                ),
                candidate_lineage=candidate_lineage,
            )
            event.update({
                "incumbent_at_admission": incumbent,
                "incumbent_vector_at_admission": incumbent_vector.tolist(),
                "validated_candidate_vectors": [value.tolist() for value in valid],
                "progress": float(evaluations / max(1, max_fes)),
                "surrogate_mu": [float(value) for value in mu],
                "surrogate_sigma": [float(value) for value in sigma],
                "regret_risk_upper": [float(value) for value in risk_upper],
                "regret_abstained": bool(abstained),
                "projection_alphas": (
                    [float(value) for value in projection_alphas]
                    if projection_alphas else []
                ),
                "semantic_commit_policy": commit_policy,
                "evaluated_candidate_objectives": (
                    [float(selected_value)] if selected_value is not None else []
                ),
                "evaluated_candidate_regrets": (
                    [normalized_regret(float(selected_value), incumbent)]
                    if selected_value is not None else []
                ),
            })
            trace.append(event)
        curve.append(global_value)
        generation += 1

    strict_zero = policy == "strict-zero"
    payload = {
        "algorithm": "native PSO" if policy == "native" else "CSA-wrapped PSO",
        "policy": policy,
        "seed": seed,
        "max_fes": max_fes,
        "strict_zero_call": strict_zero,
        "csa_contract": None if policy == "native" else PSO_CONTRACT.as_dict(),
        "evaluations_used": evaluations,
        "best_fitness": global_value,
        "best_solution": global_best.tolist(),
        "convergence_curve": curve,
        "trajectory_sha256": _trajectory_hash(curve, global_best),
        "csa_trace": trace,
        "candidate_events": len(trace),
        "abstained_events": sum(bool(event.get("regret_abstained")) for event in trace),
        "semantic_evaluations": sum(len(event.get("evaluated", [])) for event in trace),
    }
    if policy != "native":
        payload["resource_audit"] = {
            "expected": {"seed": seed, "max_fes": max_fes, "evaluations_used": max_fes},
            "observed": {"seed": seed, "max_fes": max_fes, "evaluations_used": evaluations},
        }
        payload["compliance"] = audit_csa_compliance(payload)
    return payload


def _performance_row(
    algorithm: str,
    policy: str,
    run_id: int,
    positive: dict[str, Any],
    strict_value: float,
) -> dict[str, Any]:
    relative_change = (float(positive["objective"] if "objective" in positive else positive["best_fitness"]) - strict_value) / max(abs(strict_value), 1e-12)
    return {
        "algorithm": algorithm,
        "policy": policy,
        "run_id": run_id,
        "objective": float(positive["objective"] if "objective" in positive else positive["best_fitness"]),
        "strict_zero_objective": float(strict_value),
        "relative_change": float(relative_change),
        "negative_transfer": bool(relative_change > 1e-12),
        "semantic_evaluations": int(positive.get("semantic_evaluations", 0)),
        "candidate_events": int(positive.get("candidate_events", 0)),
        "abstained_events": int(positive.get("abstained_events", 0)),
    }


def _regret_model_metrics(
    rows: Sequence[dict[str, Any]],
    model: dict[str, Any],
    residual_radius: float,
    nominal_coverage: float,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("regret model metrics require non-empty rows")
    predictions = apply_regret_linear_model(
        [row["features"] for row in rows], model
    )
    realized = np.asarray([row["realized_regret"] for row in rows], dtype=float)
    covered = np.abs(realized - predictions) <= float(residual_radius) + 1e-12
    return {
        "n": len(rows),
        "empirical_coverage": float(np.mean(covered)),
        "coverage_error": float(abs(nominal_coverage - np.mean(covered))),
        "mean_interval_half_width": float(residual_radius),
        "normalized_rmse": float(np.sqrt(np.mean((realized - predictions) ** 2))),
        "beneficial_class_rate": float(np.mean(realized < 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--closure_root",
        default="paper/q1/experiment_outputs/csa_representative_closure_v2_llmtoken_20260716",
    )
    parser.add_argument(
        "--proposal_cache",
        default="paper/q1/experiment_outputs/csa_cmaes_portability/proposal_cache",
    )
    parser.add_argument(
        "--output_root",
        default="paper/q1/experiment_outputs/csa_calibrated_regret_holdout_20260716",
    )
    parser.add_argument("--de_runs", type=int, default=3)
    parser.add_argument("--pso_runs", type=int, default=5)
    parser.add_argument("--max_fes", type=int, default=240)
    parser.add_argument("--pso_max_fes", type=int, default=200)
    parser.add_argument("--seed_base", type=int, default=20260712)
    args = parser.parse_args()

    closure_root = Path(args.closure_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    closure = _closure_index(closure_root)

    baseline_replays = []
    candidate_rows = []
    for fid in (TRAIN_FID, CALIBRATION_FID, TEST_FID):
        for run_id in range(args.de_runs):
            case = f"F{fid:02d}-10D"
            seed = int(closure[(case, run_id, "fail_closed")]["seed"])
            replay = _run_de_replay(
                closure_root, fid, run_id, seed, "rank_top", args.max_fes
            )
            baseline_replays.append(replay)
            candidate_rows.extend(_candidate_records(replay))

    _save_json({"records": candidate_rows}, output_root / "candidate_regret_dataset.json")
    _write_candidate_csv(candidate_rows, output_root / "candidate_regret_dataset.csv")
    training_rows = [row for row in candidate_rows if row["fid"] == TRAIN_FID]
    calibration_rows = [row for row in candidate_rows if row["fid"] == CALIBRATION_FID]
    test_rows = [row for row in candidate_rows if row["fid"] == TEST_FID]
    baseline_lookup = {
        (int(run["fid"]), int(run["run_id"])): run for run in baseline_replays
    }
    regret_model = fit_regret_linear_model(
        [row["features"] for row in training_rows],
        [row["realized_regret"] for row in training_rows],
        ridge=1.0,
    )
    calibration_predictions = apply_regret_linear_model(
        [row["features"] for row in calibration_rows], regret_model
    )
    residual_radius = conformal_residual_radius(
        [row["realized_regret"] for row in calibration_rows],
        calibration_predictions.tolist(),
        TARGET_COVERAGE,
    )
    threshold_records = []
    for row, prediction in zip(calibration_rows, calibration_predictions):
        threshold_records.append({
            "event_id": row["event_id"],
            "incumbent": 0.0,
            "objective": float(row["realized_regret"]),
            "mu": float(prediction),
            "sigma": float(residual_radius),
        })
    threshold, threshold_audit = preregister_abstention_threshold(
        threshold_records,
        1.0,
        RISK_MULTIPLIER,
        max_negative_transfer_rate=MAX_CALIBRATION_NEGATIVE_TRANSFER,
    )
    protocol = RegretProtocol(
        protocol_id=PROTOCOL_ID,
        training_cases=("CEC2022 F01",),
        calibration_cases=("CEC2022 F09",),
        test_cases=("CEC2022 F06",),
        target_coverage=TARGET_COVERAGE,
        uncertainty_scale=1.0,
        risk_multiplier=RISK_MULTIPLIER,
        abstention_threshold=threshold,
        threshold_objective=(
            "maximize admitted F09 calibration events subject to candidate-level "
            f"negative-transfer rate <= {MAX_CALIBRATION_NEGATIVE_TRANSFER:.2f}"
        ),
    )
    protocol_payload = protocol.as_dict()
    protocol_payload["regret_model"] = regret_model
    protocol_payload["conformal_residual_radius"] = float(residual_radius)
    protocol_payload["threshold_audit"] = threshold_audit
    protocol_payload["protocol_sha256"] = _sha256_json(protocol_payload)
    protocol_path = output_root / "preregistered_protocol.json"
    _save_json(protocol_payload, protocol_path)

    # Test execution starts only after the frozen protocol has been serialized.
    frozen = json.loads(protocol_path.read_text(encoding="utf-8"))
    if frozen["protocol_sha256"] != _sha256_json({
        key: value for key, value in frozen.items() if key != "protocol_sha256"
    }):
        raise RuntimeError("preregistered protocol hash mismatch")

    de_rows = []
    de_rank_top_rows = []
    de_payloads = []
    for run_id in range(args.de_runs):
        case = f"F{TEST_FID:02d}-10D"
        seed = int(closure[(case, run_id, "fail_closed")]["seed"])
        calibrated = _run_de_replay(
            closure_root,
            TEST_FID,
            run_id,
            seed,
            "calibrated_regret",
            args.max_fes,
            uncertainty_scale=float(frozen["uncertainty_scale"]),
            risk_multiplier=float(frozen["risk_multiplier"]),
            abstention_threshold=float(frozen["abstention_threshold"]),
            regret_model=frozen["regret_model"],
            residual_radius=float(frozen["conformal_residual_radius"]),
        )
        strict_value = float(closure[(case, run_id, "strict_zero")]["objective"])
        captured_rank_top = closure[(case, run_id, "fail_closed")]
        de_rank_top_rows.append({
            "algorithm": "DE",
            "policy": "rank_top",
            "run_id": run_id,
            "objective": float(captured_rank_top["objective"]),
            "strict_zero_objective": strict_value,
            "relative_change": (
                float(captured_rank_top["objective"]) - strict_value
            ) / max(abs(strict_value), 1e-12),
            "negative_transfer": bool(float(captured_rank_top["objective"]) > strict_value + 1e-12),
            "semantic_evaluations": int(
                baseline_lookup[(TEST_FID, run_id)]["semantic_evaluations"]
            ),
            "candidate_events": int(
                baseline_lookup[(TEST_FID, run_id)]["candidate_events"]
            ),
            "abstained_events": 0,
        })
        de_rows.append(_performance_row("DE", "calibrated_regret", run_id, calibrated, strict_value))
        de_payloads.append(calibrated)

    proposal_cache = Path(args.proposal_cache)
    pso_rows = []
    pso_rank_top_rows = []
    pso_payloads = []
    objective = get_function(TEST_FID, DIMENSION)
    for run_id in range(args.pso_runs):
        cache = json.loads(
            (proposal_cache / f"cec_f{TEST_FID:02d}_10d_r{run_id:02d}.json").read_text(encoding="utf-8")
        )
        proposals = cache["proposal_vectors"]
        seed = args.seed_base + TEST_FID * 100 + run_id
        native = run_pso(objective, seed, args.pso_max_fes, [], "native", 1.0, 1.0, 0.0)
        strict = run_pso(objective, seed, args.pso_max_fes, proposals, "strict-zero", 1.0, 1.0, 0.0)
        rank_top = run_pso(objective, seed, args.pso_max_fes, proposals, "rank_top", 1.0, 1.0, 0.0)
        calibrated = run_pso(
            objective,
            seed,
            args.pso_max_fes,
            proposals,
            "calibrated_regret",
            float(frozen["uncertainty_scale"]),
            float(frozen["risk_multiplier"]),
            float(frozen["abstention_threshold"]),
            regret_model=frozen["regret_model"],
            residual_radius=float(frozen["conformal_residual_radius"]),
        )
        if native["trajectory_sha256"] != strict["trajectory_sha256"]:
            raise RuntimeError(f"PSO strict-zero trajectory mismatch for run {run_id}")
        for policy_payload in (rank_top, calibrated):
            if not policy_payload["compliance"]["csa_compliant"]:
                raise RuntimeError(f"PSO CSA compliance failure for {policy_payload['policy']} run {run_id}")
        pso_rows.append(_performance_row(
            "PSO", "calibrated_regret", run_id, calibrated, strict["best_fitness"]
        ))
        pso_rank_top_rows.append(_performance_row(
            "PSO", "rank_top", run_id, rank_top, strict["best_fitness"]
        ))
        pso_payloads.append({
            "run_id": run_id,
            "native": native,
            "strict_zero": strict,
            "rank_top": rank_top,
            "calibrated_regret": calibrated,
        })

    calibration_report = {
        "training": _regret_model_metrics(
            training_rows, regret_model, residual_radius, TARGET_COVERAGE
        ),
        "calibration": _regret_model_metrics(
            calibration_rows, regret_model, residual_radius, TARGET_COVERAGE
        ),
        "held_out_test": _regret_model_metrics(
            test_rows, regret_model, residual_radius, TARGET_COVERAGE
        ),
    }
    compared_rows = {
        "DE_rank_top": de_rank_top_rows,
        "DE_calibrated_regret": de_rows,
        "PSO_rank_top": pso_rank_top_rows,
        "PSO_calibrated_regret": pso_rows,
    }
    performance = {
        name: admission_metrics(rows) for name, rows in compared_rows.items()
    }
    for name, rows in compared_rows.items():
        effects = [float(row["relative_change"]) for row in rows]
        performance[name].update({
            "wins": sum(effect < -1e-12 for effect in effects),
            "ties": sum(abs(effect) <= 1e-12 for effect in effects),
            "losses": sum(effect > 1e-12 for effect in effects),
            "median_relative_change": float(median(effects)),
        })

    report = {
        "protocol": frozen,
        "dataset": {
            "candidate_records": len(candidate_rows),
            "training_records": len(training_rows),
            "calibration_records": len(calibration_rows),
            "test_records": len(test_rows),
            "label": "normalized proposal-to-incumbent realized regret",
        },
        "calibration_metrics": calibration_report,
        "performance_metrics": performance,
        "de_test_rows": de_rows,
        "de_rank_top_rows": de_rank_top_rows,
        "pso_test_rows": pso_rows,
        "pso_rank_top_rows": pso_rank_top_rows,
        "claims_boundary": (
            "Exploratory independent-function holdout. F06 is never used to fit the "
            "uncertainty multiplier or abstention threshold. Results do not establish "
            "task-independent acquisition superiority."
        ),
    }
    _save_json(report, output_root / "calibrated_regret_report.json")
    _save_json({"runs": de_payloads}, output_root / "de_test_runs.json")
    _save_json({"runs": pso_payloads}, output_root / "pso_test_runs.json")
    print(json.dumps({
        "protocol_sha256": frozen["protocol_sha256"],
        "uncertainty_scale": frozen["uncertainty_scale"],
        "abstention_threshold": frozen["abstention_threshold"],
        "calibration_metrics": calibration_report,
        "performance_metrics": performance,
    }, indent=2))


if __name__ == "__main__":
    main()
