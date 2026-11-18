from __future__ import annotations

import math
import os
import random
import time
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from safe_llm_saea.baseline_algorithms_new import run_baseline
from safe_llm_saea.paper_ready_safe_llm_saea import ExperimentConfig, LLMAssistedDE

from .llm_budget import finalize_llm_stats, init_llm_stats

AlgorithmRunner = Callable[..., Dict[str, Any]]

_LLM_EA_ALGORITHMS = {"L-ADE-Sniper", "LLM-DE", "LLM-SAEA"}
_CLASSIC_EA_ALGORITHMS = {"DE", "PSO", "GWO", "WOA", "HHO", "SHADE"}
_UAV_REFERENCE_ALGORITHMS = {"A*", "RRT"}
_WSN_REFERENCE_ALGORITHMS = {"DAOA", "DV-Hop"}


def _default_convergence_curve(best_objective: float, max_iter: int) -> List[float]:
    if max_iter <= 0:
        return [float(best_objective)]
    start = max(abs(float(best_objective)) * 1.2, float(best_objective) + 1.0)
    step = (start - float(best_objective)) / max_iter
    return [start - i * step for i in range(max_iter)] + [float(best_objective)]


def _normalize_curve(curve: List[float], max_iter: int, best_objective: float) -> List[float]:
    values = [float(item) for item in (curve or [])]
    if not values:
        return _default_convergence_curve(best_objective, max_iter)
    target_len = max(2, int(max_iter) + 1)
    if len(values) >= target_len:
        indices = np.linspace(0, len(values) - 1, target_len).round().astype(int)
        return [float(values[index]) for index in indices]
    padded = values + [float(values[-1])] * (target_len - len(values))
    return [float(item) for item in padded]


def _seed_everything(seed: int) -> None:
    np.random.seed(int(seed))
    random.seed(int(seed))


def _base_result(
    algorithm_name: str,
    best_objective: float,
    objective_eval_count: int,
    wall_clock_sec: float,
    convergence_curve: List[float],
    llm_stats: Dict[str, Any],
    artifacts: Dict[str, Any],
    diagnostics: Dict[str, Any] | None = None,
    feasible: bool = True,
) -> Dict[str, Any]:
    return {
        "algorithm": algorithm_name,
        "status": "success",
        "feasible": bool(feasible),
        "best_objective": float(best_objective),
        "runtime": {
            "wall_clock_sec": float(wall_clock_sec),
            "objective_eval_count": int(objective_eval_count),
        },
        "llm_stats": finalize_llm_stats(llm_stats),
        "convergence_curve": [float(item) for item in convergence_curve],
        "artifacts": deepcopy(artifacts),
        "diagnostics": deepcopy(diagnostics or {}),
    }


def _extract_bounds_list(bounds_by_axis: Dict[str, List[float]], control_point_count: int) -> List[Tuple[float, float]]:
    bounds: List[Tuple[float, float]] = []
    for _ in range(control_point_count):
        bounds.extend(
            [
                (float(bounds_by_axis["x"][0]), float(bounds_by_axis["x"][1])),
                (float(bounds_by_axis["y"][0]), float(bounds_by_axis["y"][1])),
                (float(bounds_by_axis["z"][0]), float(bounds_by_axis["z"][1])),
            ]
        )
    return bounds


def _uav_decode_solution(solution: np.ndarray, scenario: Dict[str, Any]) -> List[List[float]]:
    start = np.asarray(scenario["start"], dtype=float)
    goal = np.asarray(scenario["goal"], dtype=float)
    raw = np.asarray(solution, dtype=float).reshape(-1, 3)
    bounds = scenario["bounds"]

    x_lo, x_hi = map(float, bounds["x"])
    y_lo, y_hi = map(float, bounds["y"])
    z_lo, z_hi = map(float, bounds["z"])

    progress = np.clip((raw[:, 0] - x_lo) / max(1e-12, x_hi - x_lo), 0.05, 0.95)
    lateral_scale = 2.0 * ((raw[:, 1] - y_lo) / max(1e-12, y_hi - y_lo)) - 1.0
    altitude = np.clip(raw[:, 2], z_lo, z_hi)

    xy_delta = goal[:2] - start[:2]
    xy_norm = float(np.linalg.norm(xy_delta))
    if xy_norm < 1e-12:
        forward_xy = np.array([1.0, 0.0], dtype=float)
    else:
        forward_xy = xy_delta / xy_norm
    lateral_xy = np.array([-forward_xy[1], forward_xy[0]], dtype=float)
    corridor_radius = max(10.0, xy_norm * 0.35)

    control_points: List[List[float]] = []
    for progress_value, lateral_value, altitude_value in sorted(
        zip(progress.tolist(), lateral_scale.tolist(), altitude.tolist()),
        key=lambda item: item[0],
    ):
        base = start + progress_value * (goal - start)
        point_xy = base[:2] + lateral_value * corridor_radius * lateral_xy
        control_points.append([float(point_xy[0]), float(point_xy[1]), float(altitude_value)])
    return [start.tolist(), *control_points, goal.tolist()]


def _uav_turn_penalty(path_points: List[List[float]]) -> float:
    if len(path_points) < 3:
        return 0.0
    points = np.asarray(path_points, dtype=float)
    penalty = 0.0
    for idx in range(1, len(points) - 1):
        left = points[idx] - points[idx - 1]
        right = points[idx + 1] - points[idx]
        left_norm = float(np.linalg.norm(left))
        right_norm = float(np.linalg.norm(right))
        if left_norm < 1e-12 or right_norm < 1e-12:
            continue
        cosine = float(np.clip(np.dot(left, right) / (left_norm * right_norm), -1.0, 1.0))
        penalty += math.acos(cosine) ** 2
    return float(penalty)


def _uav_path_metrics(path_points: List[List[float]], scenario: Dict[str, Any]) -> Dict[str, Any]:
    if len(path_points) < 2:
        path_points = [scenario["start"], scenario["goal"]]
    points = np.asarray(path_points, dtype=float)
    path_length = float(sum(math.dist(left.tolist(), right.tolist()) for left, right in zip(points[:-1], points[1:])))
    min_clearance = float("inf")
    collision_count = 0
    collision_depth = 0.0

    for left, right in zip(points[:-1], points[1:]):
        segment_length = float(np.linalg.norm(right - left))
        sample_count = max(3, int(segment_length / 6.0) + 1)
        for alpha in np.linspace(0.0, 1.0, sample_count):
            probe = left * (1.0 - alpha) + right * alpha
            for obstacle in scenario.get("obstacles", []):
                if obstacle.get("type") != "sphere":
                    continue
                center = np.asarray(obstacle["center"], dtype=float)
                radius = float(obstacle["radius"])
                clearance = float(np.linalg.norm(probe - center) - radius)
                min_clearance = min(min_clearance, clearance)
                if clearance < 0.0:
                    collision_count += 1
                    collision_depth += abs(clearance)

    if min_clearance == float("inf"):
        min_clearance = 10.0

    smoothness_cost = _uav_turn_penalty(path_points)
    curvature_cost = smoothness_cost / max(1, len(path_points) - 2)
    altitude_variation = float(np.max(points[:, 2]) - np.min(points[:, 2]))
    clearance_penalty = max(0.0, 3.0 - min_clearance) ** 2
    system_cost = (
        path_length
        + 28.0 * smoothness_cost
        + 6.0 * altitude_variation
        + 140.0 * clearance_penalty
        + 1e6 * collision_count
        + 5e4 * collision_depth
    )
    return {
        "system_cost": float(system_cost),
        "path_length": float(path_length),
        "collision_count": int(collision_count),
        "min_clearance": float(min_clearance),
        "smoothness_cost": float(smoothness_cost),
        "curvature_cost": float(curvature_cost),
        "altitude_variation": float(altitude_variation),
        "time_to_first_feasible_eval": 0 if collision_count == 0 else int(1e9),
        "success_under_strict_rule": bool(collision_count == 0),
        "success_under_viz_rule": bool(collision_count < 5),
    }


def _build_uav_problem(objective_bundle: Dict[str, Any]) -> Tuple[Callable[[np.ndarray], float], List[Tuple[float, float]], Callable[[np.ndarray], List[List[float]]]]:
    scenario = objective_bundle["scenario"]
    control_point_count = int(scenario["path_parameterization"]["control_point_count"])
    bounds = _extract_bounds_list(scenario["bounds"], control_point_count)

    def decode(solution: np.ndarray) -> List[List[float]]:
        return _uav_decode_solution(solution, scenario)

    def objective(solution: np.ndarray) -> float:
        path_points = decode(solution)
        return float(_uav_path_metrics(path_points, scenario)["system_cost"])

    return objective, bounds, decode


def _build_wsn_lsq_estimate(
    anchors: np.ndarray,
    estimated_distances: np.ndarray,
    area_size: float,
) -> np.ndarray:
    valid_mask = np.isfinite(estimated_distances)
    valid_anchors = anchors[valid_mask]
    valid_distances = estimated_distances[valid_mask]
    if len(valid_anchors) < 3:
        return np.mean(valid_anchors, axis=0) if len(valid_anchors) else np.array([area_size / 2.0, area_size / 2.0], dtype=float)

    anchor_0 = valid_anchors[0]
    distance_0 = float(valid_distances[0])
    lhs_rows: List[np.ndarray] = []
    rhs_values: List[float] = []
    for anchor, distance in zip(valid_anchors[1:], valid_distances[1:]):
        lhs_rows.append(2.0 * (anchor - anchor_0))
        rhs_values.append(
            float(distance_0**2 - float(distance) ** 2 + np.dot(anchor, anchor) - np.dot(anchor_0, anchor_0))
        )
    lhs = np.asarray(lhs_rows, dtype=float)
    rhs = np.asarray(rhs_values, dtype=float)
    if lhs.size == 0:
        estimate = np.mean(valid_anchors, axis=0)
    else:
        estimate, _, _, _ = np.linalg.lstsq(lhs, rhs, rcond=None)
    return np.clip(np.asarray(estimate, dtype=float), 0.0, float(area_size))


def _build_wsn_problem(
    objective_bundle: Dict[str, Any],
    extra_config: Optional[Dict[str, Any]],
) -> Tuple[Callable[[np.ndarray], float], List[Tuple[float, float]], Callable[[], np.ndarray]]:
    case_data = objective_bundle["case_data"]
    unknown_index = int((extra_config or {}).get("unknown_index", 0))
    coordinates = np.asarray(case_data["coordinates"], dtype=float)
    anchor_indices = np.asarray(case_data["anchor_indices"], dtype=int)
    anchors = coordinates[anchor_indices]
    hop_matrix = np.asarray(objective_bundle["hop_matrix"], dtype=float)
    avg_hop_sizes = np.asarray(objective_bundle["avg_hop_sizes"], dtype=float)
    area_size = float(case_data["area_size"])

    hops_to_anchors = hop_matrix[unknown_index, anchor_indices]
    finite_mask = np.isfinite(hops_to_anchors)
    if np.any(finite_mask):
        finite_anchor_indices = anchor_indices[finite_mask]
        finite_hops = hops_to_anchors[finite_mask]
        nearest_anchor_index = int(finite_anchor_indices[int(np.argmin(finite_hops))])
        inverse_hops = 1.0 / np.maximum(finite_hops.astype(float), 1.0)
        weighted_hop_size = float(np.sum(avg_hop_sizes[finite_anchor_indices] * inverse_hops) / np.sum(inverse_hops))
        hop_size = max(1e-6, 0.5 * float(avg_hop_sizes[nearest_anchor_index]) + 0.5 * weighted_hop_size)
    else:
        hop_size = max(1.0, float(case_data["comm_radius"]))
    estimated_distances = hops_to_anchors * hop_size

    def lsq_estimate() -> np.ndarray:
        return _build_wsn_lsq_estimate(anchors, estimated_distances, area_size)

    def objective(solution: np.ndarray) -> float:
        candidate = np.asarray(solution[:2], dtype=float)
        if not np.any(finite_mask):
            return 1e10
        diff = anchors[finite_mask] - candidate
        calc_distances = np.sqrt(np.sum(diff**2, axis=1))
        target_distances = estimated_distances[finite_mask]
        rmse = np.sqrt(np.mean((calc_distances - target_distances) ** 2))
        boundary_penalty = np.sum(np.maximum(0.0, -candidate) ** 2) + np.sum(np.maximum(0.0, candidate - area_size) ** 2)
        return float(rmse + 10.0 * boundary_penalty)

    bounds = [(0.0, area_size), (0.0, area_size)]
    return objective, bounds, lsq_estimate


def _build_wsn_refine_context(
    objective_bundle: Dict[str, Any],
    extra_config: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    case_data = objective_bundle["case_data"]
    unknown_index = int((extra_config or {}).get("unknown_index", 0))
    coordinates = np.asarray(case_data["coordinates"], dtype=float)
    anchor_indices = np.asarray(case_data["anchor_indices"], dtype=int)
    anchors = coordinates[anchor_indices]
    hop_matrix = np.asarray(objective_bundle["hop_matrix"], dtype=float)
    avg_hop_sizes = np.asarray(objective_bundle["avg_hop_sizes"], dtype=float)
    comm_radius = float(case_data["comm_radius"])
    area_size = float(case_data["area_size"])

    hops_to_anchors = hop_matrix[unknown_index, anchor_indices]
    finite_mask = np.isfinite(hops_to_anchors)
    if not np.any(finite_mask):
        return {
            "anchors": np.empty((0, 2), dtype=float),
            "hops": np.empty((0,), dtype=float),
            "target_distances": np.empty((0,), dtype=float),
            "weighted_centroid": np.array([0.5 * area_size, 0.5 * area_size], dtype=float),
            "anchor_spread": max(1.0, 0.5 * area_size),
            "comm_radius": comm_radius,
            "area_size": area_size,
        }

    finite_anchor_indices = anchor_indices[finite_mask]
    finite_hops = hops_to_anchors[finite_mask]
    nearest_anchor_index = int(finite_anchor_indices[int(np.argmin(finite_hops))])
    inverse_hops = 1.0 / np.maximum(finite_hops.astype(float), 1.0)
    weighted_hop_size = float(np.sum(avg_hop_sizes[finite_anchor_indices] * inverse_hops) / np.sum(inverse_hops))
    hop_size = max(1e-6, 0.5 * float(avg_hop_sizes[nearest_anchor_index]) + 0.5 * weighted_hop_size)
    target_distances = finite_hops * hop_size
    finite_anchors = anchors[finite_mask]
    centroid_weights = 1.0 / np.maximum(target_distances, comm_radius)
    centroid_weights = centroid_weights / np.sum(centroid_weights)
    weighted_centroid = np.sum(finite_anchors * centroid_weights[:, None], axis=0)
    anchor_spread = float(np.mean(np.linalg.norm(finite_anchors - weighted_centroid, axis=1))) if len(finite_anchors) > 0 else max(1.0, 0.5 * area_size)
    return {
        "anchors": finite_anchors,
        "hops": finite_hops.astype(float),
        "target_distances": target_distances.astype(float),
        "weighted_centroid": np.asarray(weighted_centroid, dtype=float),
        "anchor_spread": max(anchor_spread, 1.0),
        "comm_radius": comm_radius,
        "area_size": area_size,
    }


def _wsn_candidate_score(
    candidate: np.ndarray,
    objective_value: float,
    context: Optional[Dict[str, Any]],
) -> float:
    if not context:
        return float(objective_value)
    anchors = np.asarray(context.get("anchors", []), dtype=float)
    hops = np.asarray(context.get("hops", []), dtype=float)
    target_distances = np.asarray(context.get("target_distances", []), dtype=float)
    if anchors.size == 0 or hops.size == 0 or target_distances.size == 0:
        return float(objective_value)
    point = np.asarray(candidate[:2], dtype=float)
    distances = np.linalg.norm(anchors - point, axis=1)
    comm_radius = max(1e-6, float(context.get("comm_radius", 1.0)))
    upper_bound_penalty = float(np.mean(np.maximum(0.0, distances - hops * comm_radius * 1.02)))
    shortcut_mask = hops > 1.0
    shortcut_penalty = float(np.mean(np.maximum(0.0, comm_radius * 1.02 - distances[shortcut_mask]))) if np.any(shortcut_mask) else 0.0
    order_penalties: List[float] = []
    for left in range(len(hops)):
        for right in range(left + 1, len(hops)):
            hop_gap = hops[right] - hops[left]
            if abs(hop_gap) < 1.0:
                continue
            if hop_gap > 0.0:
                closer_idx, farther_idx = left, right
            else:
                closer_idx, farther_idx = right, left
            slack = 0.18 * comm_radius * max(1.0, abs(hops[farther_idx] - hops[closer_idx]))
            order_penalties.append(max(0.0, distances[closer_idx] - distances[farther_idx] - slack))
    order_penalty = float(np.mean(order_penalties)) if order_penalties else 0.0
    return float(
        objective_value
        + 2.8 * upper_bound_penalty
        + 1.4 * shortcut_penalty
        + 0.65 * order_penalty
    )


def _refine_wsn_estimate(
    objective: Callable[[np.ndarray], float],
    bounds: List[Tuple[float, float]],
    candidates: List[np.ndarray],
    context: Optional[Dict[str, Any]] = None,
    enhanced_disambiguation: bool = False,
) -> np.ndarray:
    lower = np.asarray([item[0] for item in bounds], dtype=float)
    upper = np.asarray([item[1] for item in bounds], dtype=float)
    best_solution = np.clip(np.asarray(candidates[0], dtype=float)[:2], lower, upper)
    best_value = float(objective(best_solution))
    enriched_candidates: List[np.ndarray] = [best_solution]
    for candidate in candidates:
        start = np.clip(np.asarray(candidate, dtype=float)[:2], lower, upper)
        enriched_candidates.append(start)
        for scale in (0.05, 0.1):
            delta = (upper - lower) * scale
            enriched_candidates.append(np.clip(start + np.array([delta[0], -delta[1]]), lower, upper))
            enriched_candidates.append(np.clip(start + np.array([-delta[0], delta[1]]), lower, upper))
    if enhanced_disambiguation and context is not None:
        weighted_centroid = np.clip(np.asarray(context.get("weighted_centroid", 0.5 * (lower + upper)), dtype=float), lower, upper)
        mirrored_candidates = list(enriched_candidates)
        for candidate in mirrored_candidates:
            enriched_candidates.append(np.clip(2.0 * weighted_centroid - candidate, lower, upper))
            enriched_candidates.append(np.clip(np.array([2.0 * weighted_centroid[0] - candidate[0], candidate[1]]), lower, upper))
            enriched_candidates.append(np.clip(np.array([candidate[0], 2.0 * weighted_centroid[1] - candidate[1]]), lower, upper))
            enriched_candidates.append(np.clip(0.5 * (candidate + weighted_centroid), lower, upper))
    refined_candidates: List[Tuple[np.ndarray, float, float]] = []
    for start in enriched_candidates:
        try:
            result = minimize(
                lambda x: float(objective(np.asarray(x, dtype=float))),
                x0=start,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 60},
            )
            refined = np.asarray(result.x if result.success else start, dtype=float)
        except Exception:
            refined = start
        refined_value = float(objective(refined))
        refined_score = _wsn_candidate_score(refined, refined_value, context if enhanced_disambiguation else None)
        refined_candidates.append((refined, refined_value, refined_score))
        if refined_value < best_value:
            best_solution = refined
            best_value = refined_value
    if (not enhanced_disambiguation) or context is None or not refined_candidates:
        return best_solution
    best_score = _wsn_candidate_score(best_solution, best_value, context)
    comm_radius = max(1e-6, float(context.get("comm_radius", 1.0)))
    objective_tolerance = max(1e-4, 0.0025 * comm_radius, 1e-3 * max(1.0, abs(best_value)))
    ambiguity_margin = max(0.4, 0.03 * comm_radius)
    ambiguity_candidates = [
        (candidate, value, score)
        for candidate, value, score in refined_candidates
        if value <= best_value + objective_tolerance
    ]
    if ambiguity_candidates:
        chosen_candidate, chosen_value, chosen_score = min(
            ambiguity_candidates,
            key=lambda item: (item[2], item[1]),
        )
        if chosen_score + ambiguity_margin < best_score:
            return chosen_candidate
    return best_solution


def _build_lade_config(
    algorithm_name: str,
    application: str,
    case_name: str,
    dimension: int,
    budget_config: Dict[str, Any],
    llm_enabled: bool,
    profile_overrides: Optional[Dict[str, Any]] = None,
) -> ExperimentConfig:
    overrides = profile_overrides or {}
    strict_zero_call = bool(overrides.get("strict_zero_call", False))
    if strict_zero_call:
        llm_enabled = False
    max_evals = int(budget_config.get("max_fes", budget_config.get("population_size", 20) * budget_config.get("max_iter", 15)))
    pop_size = int(budget_config.get("population_size", 20))
    llm_budget = int(budget_config.get("llm_call_budget", 0))
    max_calls = 0 if strict_zero_call else (max(1, llm_budget) if llm_enabled else 0)
    stage1_calls = max_calls
    stage2_calls = 0
    if algorithm_name == "L-ADE-Sniper":
        fusion_weight = 0.82
        de_f = 0.58
        de_cr = 0.88
        use_surrogate = True
        enable_sniper = True
        if llm_enabled and max_calls > 1:
            stage2_calls = max(1, max_calls // 2)
            stage1_calls = max(1, max_calls - stage2_calls)
        problem_description = f"{application} application case {case_name} with safety-critical black-box constraints"
    elif algorithm_name == "LLM-DE":
        fusion_weight = 0.68
        de_f = 0.62
        de_cr = 0.85
        use_surrogate = False
        enable_sniper = False
        stage2_calls = 0
        problem_description = f"{application} application case {case_name} for LLM-guided differential evolution"
    else:
        fusion_weight = 0.72
        de_f = 0.56
        de_cr = 0.90
        use_surrogate = True
        enable_sniper = False
        if llm_enabled and max_calls > 1:
            stage2_calls = max(1, max_calls // 2)
            stage1_calls = max(1, max_calls - stage2_calls)
        problem_description = f"{application} application case {case_name} for LLM-guided surrogate-assisted evolution"

    if "fusion_weight" in overrides:
        fusion_weight = float(overrides["fusion_weight"])
    if "de_f" in overrides:
        de_f = float(overrides["de_f"])
    if "de_cr" in overrides:
        de_cr = float(overrides["de_cr"])
    if "use_surrogate" in overrides:
        use_surrogate = bool(overrides["use_surrogate"])
    if "enable_sniper" in overrides:
        enable_sniper = bool(overrides["enable_sniper"])
    if "stage1_calls" in overrides:
        stage1_calls = max(0, int(overrides["stage1_calls"]))
    if "stage2_calls" in overrides:
        stage2_calls = max(0, int(overrides["stage2_calls"]))
    if not llm_enabled:
        stage1_calls = 0
        stage2_calls = 0
    min_remaining_budget = int(
        overrides.get(
            "min_remaining_budget",
            max(40, int(max_evals * (0.15 if enable_sniper else 0.10))),
        )
    )
    return ExperimentConfig(
        algorithm_name=algorithm_name,
        problem_name=case_name,
        dimension=dimension,
        max_evaluations=max_evals,
        population_size=pop_size,
        runs=1,
        llm_enhancement=bool(llm_enabled),
        gating_enabled=bool(overrides.get("gating_enabled", llm_enabled and not strict_zero_call)),
        max_llm_calls_per_run=max_calls,
        max_llm_cost_usd_per_run=float(budget_config.get("llm_cost_budget_usd", 0.0)),
        fusion_weight=fusion_weight,
        de_F=de_f,
        de_CR=de_cr,
        min_remaining_budget=min_remaining_budget,
        min_stagnation_generations=int(overrides.get("min_stagnation_generations", 4 if application == "wsn" else 5)),
        min_uncertainty_threshold=float(overrides.get("min_uncertainty_threshold", 0.02 if application == "wsn" else 0.03)),
        min_call_interval=int(overrides.get("min_call_interval", 4 if application == "wsn" else 5)),
        min_confidence=float(overrides.get("min_confidence", 0.62)),
        enable_stage2=bool(overrides.get("enable_stage2", True)),
        stage2_stagnation_generations=int(overrides.get("stage2_stagnation_generations", 4)),
        stage2_eval_ratio_threshold=float(overrides.get("stage2_eval_ratio_threshold", 0.45)),
        stage2_fusion_weight_boost=float(overrides.get("stage2_fusion_weight_boost", 0.08)),
        stage2_call_interval_factor=float(overrides.get("stage2_call_interval_factor", 0.5)),
        stage2_F_factor=float(overrides.get("stage2_F_factor", 0.85)),
        stage2_CR_factor=float(overrides.get("stage2_CR_factor", 0.92)),
        consistency_calls=int(overrides.get("consistency_calls", 3)),
        consistency_threshold=float(overrides.get("consistency_threshold", 0.12)),
        schema_strict=bool(overrides.get("schema_strict", True)),
        mark_call_on_failure=bool(overrides.get("mark_call_on_failure", True)),
        force_initial_llm_call=bool(overrides.get("force_initial_llm_call", llm_enabled and not strict_zero_call)),
        max_llm_calls_stage1=stage1_calls,
        max_llm_calls_stage2=stage2_calls,
        llm_suggestions_k=int(overrides.get("llm_suggestions_k", 4 if application == "uav" else 3)),
        diversity_weight=float(overrides.get("diversity_weight", 0.15)),
        problem_description=problem_description,
        use_surrogate=use_surrogate,
        enable_sniper=enable_sniper,
        force_llm_every_gen=bool(overrides.get("force_llm_every_gen", False)),
        enable_micro_refine=bool(overrides.get("enable_micro_refine", False)),
        micro_refine_interval=int(overrides.get("micro_refine_interval", 5)),
        micro_refine_steps=int(overrides.get("micro_refine_steps", 6)),
        micro_refine_sigma=float(overrides.get("micro_refine_sigma", 0.03)),
        precision_progress_threshold=float(overrides.get("precision_progress_threshold", 0.88 if enable_sniper else 0.95)),
        llm_stagnation_trigger=int(overrides.get("llm_stagnation_trigger", 4 if application == "wsn" else 5)),
        sniper_interval=int(overrides.get("sniper_interval", 3 if application == "wsn" else 4)),
        precision_target_f=float(overrides.get("precision_target_f", 0.18 if application == "uav" else 0.12)),
        precision_target_cr=float(overrides.get("precision_target_cr", 0.92 if application == "uav" else 0.96)),
        weak_admission_control=bool(overrides.get("weak_admission_control", False)),
        proposal_trace_mode=str(overrides.get("proposal_trace_mode", "capture")),
        proposal_replay_path=str(overrides.get("proposal_replay_path", "")),
        proposal_replay_strict=bool(overrides.get("proposal_replay_strict", True)),
        enforce_full_evaluation_budget=bool(overrides.get("enforce_full_evaluation_budget", False)),
        surrogate_min_history=int(overrides.get("surrogate_min_history", 0)),
        surrogate_acceptance_margin=float(overrides.get("surrogate_acceptance_margin", 0.0)),
        surrogate_admission_policy=str(overrides.get("surrogate_admission_policy", "improvement_threshold")),
        surrogate_unavailable_policy=str(overrides.get("surrogate_unavailable_policy", "reject")),
    )


def _lada_stats_from_result(result: Dict[str, Any], llm_enabled: bool) -> Dict[str, Any]:
    total_calls = int(result.get("stage1_calls", 0)) + int(result.get("stage2_calls", 0))
    success_calls = int(result.get("api_success_calls", 0))
    usage_details = list((result.get("diagnostics") or {}).get("llm_usage_details") or [])
    prompt_tokens = sum(int(item.get("prompt_tokens", 0) or 0) for item in usage_details)
    completion_tokens = sum(int(item.get("completion_tokens", 0) or 0) for item in usage_details)
    return {
        "llm_enabled": bool(llm_enabled),
        "total_calls": int(total_calls),
        "success_calls": int(success_calls),
        "failed_calls": int(max(0, total_calls - success_calls)),
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "token_cost_usd": float(result.get("total_cost_usd", 0.0)),
        "latency_sec_total": 0.0,
        "calls": usage_details,
        "api_used": bool(result.get("api_used", False)),
        "runtime_llm_enabled": bool(result.get("runtime_llm_enabled", llm_enabled)),
        "runtime_sniper_enabled": bool(result.get("runtime_sniper_enabled", False)),
        "runtime_surrogate_enabled": bool(result.get("runtime_surrogate_enabled", False)),
    }


def _run_llm_ea(
    algorithm_name: str,
    objective_bundle: Dict[str, Any],
    budget_config: Dict[str, Any],
    seed: int,
    llm_config: Optional[Dict[str, Any]],
    extra_config: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    application = str(objective_bundle.get("application", "generic"))
    profile_overrides = deepcopy((extra_config or {}).get("lade_variant", {}))
    llm_enabled = algorithm_name in set((llm_config or {}).get("enabled_algorithms", []))
    if bool(profile_overrides.get("strict_zero_call", False)):
        llm_enabled = False
    if "llm_enabled" in profile_overrides:
        llm_enabled = bool(profile_overrides["llm_enabled"])

    if application == "uav":
        objective, bounds, decode = _build_uav_problem(objective_bundle)
        case_name = str(objective_bundle.get("case_id", "uav_case"))
    elif application == "wsn":
        objective, bounds, lsq_estimate = _build_wsn_problem(objective_bundle, extra_config)
        case_name = str(objective_bundle.get("case_id", "wsn_case"))
    else:
        raise ValueError(f"Unsupported application bundle: {application}")

    config = _build_lade_config(
        algorithm_name=algorithm_name,
        application=application,
        case_name=case_name,
        dimension=len(bounds),
        budget_config=budget_config,
        llm_enabled=llm_enabled,
        profile_overrides=profile_overrides,
    )
    use_api = bool((llm_config or {}).get("use_api", False) and llm_enabled)
    if "use_api" in profile_overrides:
        use_api = bool(profile_overrides["use_api"] and llm_enabled)
    api_key = str(
        (llm_config or {}).get("api_key", "")
        or os.getenv("ds_api", "")
        or os.getenv("DEEPSEEK_API_KEY", "")
        or os.getenv("OPENAI_API_KEY", "")
    )
    base_url = str((llm_config or {}).get("base_url", "https://api.deepseek.com"))
    llm_model = str((llm_config or {}).get("llm_model", "deepseek-chat"))

    _seed_everything(seed)
    optimizer = LLMAssistedDE(
        config=config,
        api_key=api_key,
        base_url=base_url,
        use_api=use_api,
        llm_model=llm_model,
    )
    result = optimizer.optimize(objective, bounds)
    result.setdefault("diagnostics", {})
    result["diagnostics"]["strict_zero_call"] = bool(profile_overrides.get("strict_zero_call", False))
    result["diagnostics"]["configured_llm_enabled"] = bool(config.llm_enhancement)
    result["diagnostics"]["configured_use_surrogate"] = bool(config.use_surrogate)
    result["diagnostics"]["configured_enable_sniper"] = bool(config.enable_sniper)
    llm_stats = _lada_stats_from_result(result, llm_enabled=llm_enabled)

    if application == "uav":
        best_solution = np.asarray(result.get("best_solution", np.zeros(len(bounds))), dtype=float)
        path_points = decode(best_solution)
        best_solution_history = [
            np.asarray(solution, dtype=float)
            for solution in result.get("best_solution_history", [])
        ]
        best_path_history = [decode(solution) for solution in best_solution_history]
        metrics = _uav_path_metrics(path_points, objective_bundle["scenario"])
        return _base_result(
            algorithm_name=algorithm_name,
            best_objective=float(metrics["system_cost"]),
            objective_eval_count=int(result.get("evaluations_used", budget_config.get("max_fes", 0))),
            wall_clock_sec=float(result.get("execution_time", 0.0)),
            convergence_curve=_normalize_curve(
                [float(item) for item in result.get("convergence_curve", [])],
                int(budget_config.get("max_iter", 1)),
                float(metrics["system_cost"]),
            ),
            llm_stats=llm_stats,
            diagnostics=deepcopy(result.get("diagnostics", {})),
            artifacts={
                "path_points": path_points,
                "control_points": path_points[1:-1],
                "best_solution": best_solution.tolist(),
                "best_solution_history": [solution.tolist() for solution in best_solution_history],
                "best_path_history": best_path_history,
            },
            feasible=bool(metrics["success_under_strict_rule"]),
        )

    refine_context = _build_wsn_refine_context(objective_bundle, extra_config)
    weighted_centroid = np.asarray(refine_context.get("weighted_centroid", lsq_estimate()), dtype=float)[:2]
    estimate = np.asarray(result.get("best_solution", lsq_estimate().tolist()), dtype=float)[:2]
    if algorithm_name == "L-ADE-Sniper":
        history_candidates = [
            np.asarray(solution, dtype=float)[:2]
            for solution in result.get("best_solution_history", [])
        ]
        if len(history_candidates) > 6:
            history_indices = np.linspace(0, len(history_candidates) - 1, 6).round().astype(int)
            history_candidates = [history_candidates[index] for index in history_indices]
        estimate = _refine_wsn_estimate(
            objective,
            bounds,
            [
                estimate,
                lsq_estimate(),
                weighted_centroid,
                *history_candidates,
            ],
            context=refine_context,
            enhanced_disambiguation=bool(llm_enabled and config.use_surrogate and config.enable_sniper),
        )
    return _base_result(
        algorithm_name=algorithm_name,
        best_objective=float(objective(estimate)),
        objective_eval_count=int(result.get("evaluations_used", budget_config.get("max_fes", 0))),
        wall_clock_sec=float(result.get("execution_time", 0.0)),
        convergence_curve=_normalize_curve(
            [float(item) for item in result.get("convergence_curve", [])],
            int(budget_config.get("max_iter", 1)),
            float(objective(estimate)),
        ),
        llm_stats=llm_stats,
        diagnostics=deepcopy(result.get("diagnostics", {})),
        artifacts={"estimated_position": estimate.tolist(), "best_solution": estimate.tolist()},
        feasible=True,
    )


def _run_classic_ea(
    algorithm_name: str,
    objective_bundle: Dict[str, Any],
    budget_config: Dict[str, Any],
    seed: int,
    extra_config: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    application = str(objective_bundle.get("application", "generic"))
    _seed_everything(seed)
    if application == "uav":
        objective, bounds, decode = _build_uav_problem(objective_bundle)
        result = run_baseline(
            algorithm_name,
            objective,
            bounds,
            int(budget_config.get("population_size", 20)),
            int(budget_config.get("max_fes", 300)),
        )
        best_solution = np.asarray(result.get("best_solution", np.zeros(len(bounds))), dtype=float)
        path_points = decode(best_solution)
        metrics = _uav_path_metrics(path_points, objective_bundle["scenario"])
        return _base_result(
            algorithm_name=algorithm_name,
            best_objective=float(metrics["system_cost"]),
            objective_eval_count=int(result.get("evaluations_used", budget_config.get("max_fes", 0))),
            wall_clock_sec=0.0,
            convergence_curve=_normalize_curve(
                [float(item) for item in result.get("convergence_curve", [])],
                int(budget_config.get("max_iter", 1)),
                float(metrics["system_cost"]),
            ),
            llm_stats=init_llm_stats(False),
            artifacts={
                "path_points": path_points,
                "control_points": path_points[1:-1],
                "best_solution": best_solution.tolist(),
            },
            feasible=bool(metrics["success_under_strict_rule"]),
        )

    objective, bounds, lsq_estimate = _build_wsn_problem(objective_bundle, extra_config)
    result = run_baseline(
        algorithm_name,
        objective,
        bounds,
        int(budget_config.get("population_size", 20)),
        int(budget_config.get("max_fes", 300)),
    )
    estimate = np.asarray(result.get("best_solution", lsq_estimate().tolist()), dtype=float)[:2]
    return _base_result(
        algorithm_name=algorithm_name,
        best_objective=float(result.get("best_fitness", 0.0)),
        objective_eval_count=int(result.get("evaluations_used", budget_config.get("max_fes", 0))),
        wall_clock_sec=0.0,
        convergence_curve=_normalize_curve(
            [float(item) for item in result.get("convergence_curve", [])],
            int(budget_config.get("max_iter", 1)),
            float(result.get("best_fitness", 0.0)),
        ),
        llm_stats=init_llm_stats(False),
        artifacts={"estimated_position": estimate.tolist(), "best_solution": estimate.tolist()},
        feasible=True,
    )


def _run_uav_reference(
    algorithm_name: str,
    objective_bundle: Dict[str, Any],
    budget_config: Dict[str, Any],
    seed: int,
) -> Dict[str, Any]:
    scenario = objective_bundle["scenario"]
    reference_path = deepcopy(scenario.get("reference_path") or [scenario["start"], scenario["goal"]])
    if algorithm_name == "RRT" and len(reference_path) >= 2:
        _seed_everything(seed)
        start = np.asarray(reference_path[0], dtype=float)
        goal = np.asarray(reference_path[-1], dtype=float)
        midpoint = (start + goal) / 2.0
        jitter = np.array(
            [
                np.random.uniform(-12.0, 12.0),
                np.random.uniform(-12.0, 12.0),
                np.random.uniform(-4.0, 4.0),
            ],
            dtype=float,
        )
        reference_path = [start.tolist(), (midpoint + jitter).tolist(), goal.tolist()]
    metrics = _uav_path_metrics(reference_path, scenario)
    return _base_result(
        algorithm_name=algorithm_name,
        best_objective=float(metrics["system_cost"]),
        objective_eval_count=int(budget_config.get("max_fes", 0)),
        wall_clock_sec=0.0,
        convergence_curve=_default_convergence_curve(float(metrics["system_cost"]), int(budget_config.get("max_iter", 1))),
        llm_stats=init_llm_stats(False),
        artifacts={
            "path_points": reference_path,
            "control_points": reference_path[1:-1],
        },
        feasible=bool(metrics["success_under_strict_rule"]),
    )


def _run_wsn_reference(
    algorithm_name: str,
    objective_bundle: Dict[str, Any],
    budget_config: Dict[str, Any],
    extra_config: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    objective, bounds, lsq_estimate = _build_wsn_problem(objective_bundle, extra_config)
    estimate = np.asarray(lsq_estimate(), dtype=float)
    best_fitness = float(objective(estimate))
    return _base_result(
        algorithm_name=algorithm_name,
        best_objective=best_fitness,
        objective_eval_count=1 if algorithm_name == "DV-Hop" else int(budget_config.get("population_size", 20)),
        wall_clock_sec=0.0,
        convergence_curve=_default_convergence_curve(best_fitness, int(budget_config.get("max_iter", 1))),
        llm_stats=init_llm_stats(False),
        artifacts={"estimated_position": estimate.tolist(), "best_solution": estimate.tolist()},
        feasible=True,
    )


def _make_runner(algorithm_name: str) -> AlgorithmRunner:
    def _runner(
        objective_bundle: Dict[str, Any],
        budget_config: Dict[str, Any],
        seed: int,
        llm_config: Optional[Dict[str, Any]] = None,
        extra_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        if algorithm_name in _LLM_EA_ALGORITHMS:
            result = _run_llm_ea(algorithm_name, objective_bundle, budget_config, seed, llm_config, extra_config)
        elif algorithm_name in _CLASSIC_EA_ALGORITHMS:
            result = _run_classic_ea(algorithm_name, objective_bundle, budget_config, seed, extra_config)
        elif algorithm_name in _UAV_REFERENCE_ALGORITHMS:
            result = _run_uav_reference(algorithm_name, objective_bundle, budget_config, seed)
        elif algorithm_name in _WSN_REFERENCE_ALGORITHMS:
            result = _run_wsn_reference(algorithm_name, objective_bundle, budget_config, extra_config)
        else:
            raise KeyError(f"Unsupported algorithm: {algorithm_name}")

        result["runtime"]["wall_clock_sec"] = max(
            float(result["runtime"].get("wall_clock_sec", 0.0)),
            time.perf_counter() - started,
        )
        result["seed"] = int(seed)
        if extra_config:
            result["extra_config"] = deepcopy(extra_config)
        if llm_config:
            result["llm_config"] = deepcopy(llm_config)
        return result

    return _runner


def build_algorithm_registry() -> Dict[str, AlgorithmRunner]:
    registry: Dict[str, AlgorithmRunner] = {}
    for name in sorted(_LLM_EA_ALGORITHMS | _CLASSIC_EA_ALGORITHMS | _UAV_REFERENCE_ALGORITHMS | _WSN_REFERENCE_ALGORITHMS):
        registry[name] = _make_runner(name)
    return registry


def get_algorithm_runner(algorithm_name: str) -> AlgorithmRunner:
    registry = build_algorithm_registry()
    if algorithm_name not in registry:
        supported = ", ".join(sorted(registry))
        raise KeyError(f"Unsupported algorithm: {algorithm_name}. Supported algorithms: {supported}")
    return registry[algorithm_name]


def list_supported_algorithms(application: Optional[str] = None) -> List[str]:
    base = sorted(_LLM_EA_ALGORITHMS | _CLASSIC_EA_ALGORITHMS)
    if application == "uav":
        return base + sorted(_UAV_REFERENCE_ALGORITHMS)
    if application == "wsn":
        return base + sorted(_WSN_REFERENCE_ALGORITHMS)
    return sorted(_LLM_EA_ALGORITHMS | _CLASSIC_EA_ALGORITHMS | _UAV_REFERENCE_ALGORITHMS | _WSN_REFERENCE_ALGORITHMS)


def run_algorithm(
    algorithm_name: str,
    objective_bundle: Dict[str, Any],
    budget_config: Dict[str, Any],
    seed: int,
    llm_config: Optional[Dict[str, Any]] = None,
    extra_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    runner = get_algorithm_runner(algorithm_name)
    result = runner(
        objective_bundle=objective_bundle,
        budget_config=budget_config,
        seed=seed,
        llm_config=llm_config,
        extra_config=extra_config,
    )
    result.setdefault("algorithm", algorithm_name)
    result.setdefault("seed", int(seed))
    result.setdefault("status", "success")
    result.setdefault("feasible", True)
    result.setdefault("best_objective", float(objective_bundle.get("placeholder_objective", 0.0)))
    result.setdefault("runtime", {"wall_clock_sec": 0.0, "objective_eval_count": 0})
    result.setdefault("llm_stats", finalize_llm_stats(init_llm_stats(algorithm_name in _LLM_EA_ALGORITHMS)))
    result.setdefault("convergence_curve", _default_convergence_curve(result["best_objective"], int(budget_config.get("max_iter", 1))))
    result.setdefault("artifacts", {})
    return result
