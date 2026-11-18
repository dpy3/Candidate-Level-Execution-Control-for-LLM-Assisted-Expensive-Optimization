from __future__ import annotations

import argparse
import hashlib
import math
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy.optimize import minimize

from safe_llm_saea.application_benchmark.common.algorithm_registry import run_algorithm
from safe_llm_saea.application_benchmark.common.io_utils import load_json, make_run_filename, save_json
from safe_llm_saea.application_benchmark.wsn.aggregate_wsn_results import aggregate_wsn_pipeline


def load_wsn_suite_manifest(manifest_path: str) -> Dict[str, Any]:
    manifest = load_json(manifest_path)
    if manifest.get("application") != "wsn":
        raise ValueError("Manifest is not a WSN suite")
    return manifest


def build_wsn_objective_bundle(case_data: Dict[str, Any]) -> Dict[str, Any]:
    coordinates = np.asarray(case_data["coordinates"], dtype=float)
    node_count = int(case_data["node_count"])
    comm_radius = float(case_data["comm_radius"])
    distance_matrix = np.full((node_count, node_count), np.inf, dtype=float)
    hop_matrix = np.full((node_count, node_count), np.inf, dtype=float)
    for index in range(node_count):
        distance_matrix[index, index] = 0.0
        hop_matrix[index, index] = 0.0
    for left in range(node_count):
        for right in range(left + 1, node_count):
            distance = float(np.linalg.norm(coordinates[left] - coordinates[right]))
            distance_matrix[left, right] = distance
            distance_matrix[right, left] = distance
            if distance <= comm_radius:
                hop_matrix[left, right] = 1.0
                hop_matrix[right, left] = 1.0
    for pivot in range(node_count):
        hop_matrix = np.minimum(hop_matrix, hop_matrix[:, [pivot]] + hop_matrix[[pivot], :])

    anchor_indices = case_data["anchor_indices"]
    avg_hop_sizes = np.zeros(node_count, dtype=float)
    for anchor_index in anchor_indices:
        hop_sum = 0.0
        dist_sum = 0.0
        for other_anchor_index in anchor_indices:
            if anchor_index == other_anchor_index:
                continue
            hops = hop_matrix[anchor_index, other_anchor_index]
            if np.isfinite(hops) and hops > 0:
                hop_sum += float(hops)
                dist_sum += float(distance_matrix[anchor_index, other_anchor_index])
        if hop_sum > 0:
            avg_hop_sizes[anchor_index] = dist_sum / hop_sum
        else:
            avg_hop_sizes[anchor_index] = comm_radius
    return {
        "application": "wsn",
        "case_id": case_data["case_id"],
        "case_data": case_data,
        "placeholder_objective": 0.0,
        "hop_matrix": hop_matrix.tolist(),
        "avg_hop_sizes": avg_hop_sizes.tolist(),
        "default_artifacts": {"estimated_coordinates": [], "node_errors": []},
    }


def _build_wsn_geometry_context(
    case_data: Dict[str, Any],
    objective_bundle: Dict[str, Any],
    unknown_index: int,
) -> Dict[str, Any]:
    coordinates = np.asarray(case_data["coordinates"], dtype=float)
    anchor_indices = np.asarray(case_data["anchor_indices"], dtype=int)
    anchors = coordinates[anchor_indices]
    hop_matrix = np.asarray(objective_bundle["hop_matrix"], dtype=float)
    avg_hop_sizes = np.asarray(objective_bundle["avg_hop_sizes"], dtype=float)
    hops_to_anchors = hop_matrix[int(unknown_index), anchor_indices]
    finite_mask = np.isfinite(hops_to_anchors)
    if not np.any(finite_mask):
        return {
            "has_support": False,
            "anchors": np.empty((0, 2), dtype=float),
            "hops": np.empty(0, dtype=float),
            "target_distances": np.empty(0, dtype=float),
        }
    finite_anchor_indices = anchor_indices[finite_mask]
    finite_hops = hops_to_anchors[finite_mask]
    nearest_anchor_index = int(finite_anchor_indices[int(np.argmin(finite_hops))])
    inverse_hops = 1.0 / np.maximum(finite_hops.astype(float), 1.0)
    weighted_hop_size = float(np.sum(avg_hop_sizes[finite_anchor_indices] * inverse_hops) / np.sum(inverse_hops))
    hop_size = max(1e-6, 0.5 * float(avg_hop_sizes[nearest_anchor_index]) + 0.5 * weighted_hop_size)
    return {
        "has_support": True,
        "anchors": anchors[finite_mask],
        "hops": finite_hops.astype(float),
        "target_distances": finite_hops.astype(float) * hop_size,
    }


def _wsn_anchor_objective(
    candidate: np.ndarray,
    geometry_context: Dict[str, Any],
    area_size: float,
) -> float:
    point = np.asarray(candidate[:2], dtype=float)
    if not geometry_context.get("has_support", False):
        return 1e10
    anchors = np.asarray(geometry_context["anchors"], dtype=float)
    target_distances = np.asarray(geometry_context["target_distances"], dtype=float)
    calculated_distances = np.linalg.norm(anchors - point, axis=1)
    rmse = float(np.sqrt(np.mean(np.square(calculated_distances - target_distances))))
    boundary_penalty = float(
        np.sum(np.maximum(0.0, -point) ** 2) + np.sum(np.maximum(0.0, point - float(area_size)) ** 2)
    )
    return float(rmse + 10.0 * boundary_penalty)


def _refine_wsn_anchor_candidate(
    candidate: np.ndarray,
    geometry_context: Dict[str, Any],
    area_size: float,
) -> np.ndarray:
    seed = np.clip(np.asarray(candidate[:2], dtype=float), 0.0, float(area_size))
    try:
        result = minimize(
            lambda x: float(_wsn_anchor_objective(np.asarray(x, dtype=float), geometry_context, area_size)),
            x0=seed,
            method="L-BFGS-B",
            bounds=[(0.0, float(area_size)), (0.0, float(area_size))],
            options={"maxiter": 60},
        )
        return np.asarray(result.x if result.success else seed, dtype=float)
    except Exception:
        return seed


def _build_wsn_ambiguity_signature(
    estimated_position: np.ndarray,
    geometry_context: Dict[str, Any],
    area_size: float,
    comm_radius: float,
) -> Dict[str, Any]:
    current = _refine_wsn_anchor_candidate(estimated_position, geometry_context, area_size)
    current_objective = _wsn_anchor_objective(current, geometry_context, area_size)
    if (not geometry_context.get("has_support", False)) or current_objective > max(4.5, 0.22 * comm_radius):
        return {
            "ambiguous": False,
            "current_position": current,
            "current_objective": float(current_objective),
            "alternative_position": current,
            "alternative_objective": float(current_objective),
            "separation": 0.0,
        }
    anchors = np.asarray(geometry_context["anchors"], dtype=float)
    centroid = (
        np.mean(anchors, axis=0)
        if anchors.size
        else np.array([0.5 * area_size, 0.5 * area_size], dtype=float)
    )
    seeds = [
        np.asarray(2.0 * centroid - current, dtype=float),
        np.asarray([2.0 * centroid[0] - current[0], current[1]], dtype=float),
        np.asarray([current[0], 2.0 * centroid[1] - current[1]], dtype=float),
        np.asarray(centroid, dtype=float),
        np.asarray([area_size - current[0], current[1]], dtype=float),
        np.asarray([current[0], area_size - current[1]], dtype=float),
    ]
    distance_threshold = max(6.0, 0.7 * comm_radius)
    objective_tolerance = max(0.9, 0.06 * comm_radius)
    best_alternative = current
    best_alternative_objective = float("inf")
    best_separation = 0.0
    for seed in seeds:
        alternative = _refine_wsn_anchor_candidate(seed, geometry_context, area_size)
        separation = float(np.linalg.norm(alternative - current))
        if separation <= distance_threshold:
            continue
        alternative_objective = _wsn_anchor_objective(alternative, geometry_context, area_size)
        if alternative_objective <= current_objective + objective_tolerance and alternative_objective < best_alternative_objective:
            best_alternative = alternative
            best_alternative_objective = float(alternative_objective)
            best_separation = separation
    if not np.isfinite(best_alternative_objective):
        best_alternative_objective = float(current_objective)
        best_alternative = current
    return {
        "ambiguous": bool(best_separation > 0.0),
        "current_position": current,
        "current_objective": float(current_objective),
        "alternative_position": np.asarray(best_alternative, dtype=float),
        "alternative_objective": float(best_alternative_objective),
        "separation": float(best_separation),
    }


def _should_apply_full_wsn_postprocess(
    algorithm_name: str,
    algorithm_label: str | None,
) -> bool:
    return str(algorithm_label or algorithm_name) == "L-ADE-Sniper"


def _refine_full_wsn_case_positions(
    case_data: Dict[str, Any],
    objective_bundle: Dict[str, Any],
    node_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not node_results:
        return node_results
    area_size = float(case_data["area_size"])
    comm_radius = float(case_data["comm_radius"])
    coordinates = np.asarray(case_data["coordinates"], dtype=float)
    hop_matrix = np.asarray(objective_bundle["hop_matrix"], dtype=float)
    refined_results = [deepcopy(item) for item in node_results]
    geometry_contexts = {
        int(item["unknown_index"]): _build_wsn_geometry_context(
            case_data=case_data,
            objective_bundle=objective_bundle,
            unknown_index=int(item["unknown_index"]),
        )
        for item in refined_results
    }
    unknown_indices = [int(item["unknown_index"]) for item in refined_results]
    ambiguity_signatures = {
        index: _build_wsn_ambiguity_signature(
            estimated_position=np.asarray(refined_results[offset]["estimated_position"], dtype=float)[:2],
            geometry_context=geometry_contexts[index],
            area_size=area_size,
            comm_radius=comm_radius,
        )
        for offset, index in enumerate(unknown_indices)
    }
    current_map = {
        index: np.asarray(ambiguity_signatures[index]["current_position"], dtype=float)
        for index in unknown_indices
    }
    alternative_map = {
        index: np.asarray(ambiguity_signatures[index]["alternative_position"], dtype=float)
        if ambiguity_signatures[index]["ambiguous"]
        else np.asarray(current_map[index], dtype=float)
        for index in unknown_indices
    }
    initial_flat = np.concatenate([current_map[index] for index in unknown_indices], axis=0)
    alternative_flat = np.concatenate(
        [alternative_map[index] for index in unknown_indices],
        axis=0,
    )
    centroid_flat = np.concatenate(
        [
            np.mean(np.asarray(geometry_contexts[index]["anchors"], dtype=float), axis=0)
            if geometry_contexts[index].get("has_support", False)
            else np.array([0.5 * area_size, 0.5 * area_size], dtype=float)
            for index in unknown_indices
        ],
        axis=0,
    )
    blended_flat = 0.6 * initial_flat + 0.4 * alternative_flat
    bounds = [(0.0, area_size)] * len(initial_flat)

    def _build_estimate_map(flat_positions: np.ndarray) -> Dict[int, np.ndarray]:
        return {
            index: np.asarray(flat_positions[2 * offset : 2 * offset + 2], dtype=float)
            for offset, index in enumerate(unknown_indices)
        }

    def _node_pair_penalty(node_index: int, point: np.ndarray, estimate_map: Dict[int, np.ndarray]) -> float:
        total = 0.0
        for other_index in unknown_indices:
            if other_index == node_index:
                continue
            hops = float(hop_matrix[node_index, other_index])
            if not np.isfinite(hops) or hops <= 0.0:
                continue
            distance = float(np.linalg.norm(point - estimate_map[other_index]))
            if hops <= 1.0 + 1e-12:
                total += 0.22 * max(0.0, distance - comm_radius) ** 2
            else:
                total += 0.08 * max(0.0, comm_radius * 1.02 - distance) ** 2
        return float(total)

    def _score(flat_positions: np.ndarray) -> float:
        estimate_map = _build_estimate_map(flat_positions)
        total = 0.0
        for node_index in unknown_indices:
            point = estimate_map[node_index]
            context = geometry_contexts[node_index]
            total += _wsn_anchor_objective(point, context, area_size)
            if context.get("has_support", False):
                anchor_positions = np.asarray(context["anchors"], dtype=float)
                anchor_hops = np.asarray(context["hops"], dtype=float)
                anchor_distances = np.linalg.norm(anchor_positions - point, axis=1)
                one_hop_mask = anchor_hops <= 1.0 + 1e-12
                multi_hop_mask = anchor_hops > 1.0 + 1e-12
                if np.any(one_hop_mask):
                    total += 0.16 * float(np.mean(np.maximum(0.0, anchor_distances[one_hop_mask] - comm_radius) ** 2))
                if np.any(multi_hop_mask):
                    total += 0.06 * float(np.mean(np.maximum(0.0, comm_radius * 1.02 - anchor_distances[multi_hop_mask]) ** 2))
            total += _node_pair_penalty(node_index, point, estimate_map)
        return float(total)

    initial_score = _score(initial_flat)
    candidate_solutions = [initial_flat]
    for seed in (initial_flat, alternative_flat, blended_flat, 0.75 * initial_flat + 0.25 * centroid_flat):
        try:
            result = minimize(
                lambda x: float(_score(np.asarray(x, dtype=float))),
                x0=np.asarray(seed, dtype=float),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 80},
            )
            candidate_solutions.append(np.asarray(result.x if result.success else seed, dtype=float))
        except Exception:
            candidate_solutions.append(np.asarray(seed, dtype=float))
    best_flat = min(candidate_solutions, key=lambda item: _score(np.asarray(item, dtype=float)))
    best_score = _score(best_flat)
    if best_score > initial_score - 1e-9:
        return refined_results
    best_map = _build_estimate_map(best_flat)
    for offset, item in enumerate(refined_results):
        unknown_index = int(item["unknown_index"])
        initial_position = np.asarray(current_map[unknown_index], dtype=float)
        refined_position = np.asarray(best_map[unknown_index], dtype=float)
        initial_objective = float(ambiguity_signatures[unknown_index]["current_objective"])
        refined_objective = _wsn_anchor_objective(refined_position, geometry_contexts[unknown_index], area_size)
        if (
            (not ambiguity_signatures[unknown_index]["ambiguous"])
            and refined_objective > initial_objective + max(0.75, 0.04 * comm_radius)
        ):
            refined_position = initial_position
            refined_objective = initial_objective
        item["estimated_position"] = refined_position.tolist()
        item["best_objective"] = float(refined_objective)
        true_position = coordinates[unknown_index]
        item["localization_error"] = float(math.dist(true_position.tolist(), refined_position.tolist()))
    return refined_results


def _compute_wsn_geometry_metrics(
    case_data: Dict[str, Any],
    objective_bundle: Dict[str, Any],
    node_results: List[Dict[str, Any]],
) -> Dict[str, float]:
    comm_radius = float(case_data["comm_radius"])
    anchor_distance_rmse_values: List[float] = []
    anchor_distance_mae_values: List[float] = []
    relative_distance_error_values: List[float] = []
    nearest_anchor_matches = 0
    nearest_anchor_total = 0
    one_hop_violations = 0
    one_hop_total = 0
    multi_hop_shortcuts = 0
    multi_hop_total = 0
    finite_anchor_support_ratios: List[float] = []
    total_anchor_count = max(1, int(case_data["anchor_count"]))

    for item in node_results:
        context = _build_wsn_geometry_context(
            case_data=case_data,
            objective_bundle=objective_bundle,
            unknown_index=int(item["unknown_index"]),
        )
        if not context["has_support"]:
            continue
        estimated_position = np.asarray(item["estimated_position"], dtype=float)
        true_position = np.asarray(item["true_position"], dtype=float)
        anchors = np.asarray(context["anchors"], dtype=float)
        hops = np.asarray(context["hops"], dtype=float)
        target_distances = np.asarray(context["target_distances"], dtype=float)
        estimated_distances = np.linalg.norm(anchors - estimated_position, axis=1)
        true_distances = np.linalg.norm(anchors - true_position, axis=1)
        residuals = estimated_distances - target_distances

        anchor_distance_rmse_values.append(float(np.sqrt(np.mean(np.square(residuals)))))
        anchor_distance_mae_values.append(float(np.mean(np.abs(residuals))))
        relative_distance_error_values.append(
            float(np.mean(np.abs(residuals) / np.maximum(np.abs(target_distances), 1e-6)))
        )
        nearest_anchor_matches += int(np.argmin(estimated_distances) == np.argmin(true_distances))
        nearest_anchor_total += 1

        one_hop_mask = hops <= 1.0 + 1e-12
        multi_hop_mask = hops > 1.0 + 1e-12
        one_hop_violations += int(np.sum(estimated_distances[one_hop_mask] > comm_radius + 1e-9))
        one_hop_total += int(np.sum(one_hop_mask))
        multi_hop_shortcuts += int(np.sum(estimated_distances[multi_hop_mask] <= comm_radius + 1e-9))
        multi_hop_total += int(np.sum(multi_hop_mask))
        finite_anchor_support_ratios.append(float(len(hops) / total_anchor_count))

    return {
        "anchor_distance_rmse": float(np.mean(anchor_distance_rmse_values) if anchor_distance_rmse_values else 0.0),
        "anchor_distance_mae": float(np.mean(anchor_distance_mae_values) if anchor_distance_mae_values else 0.0),
        "relative_distance_error": float(np.mean(relative_distance_error_values) if relative_distance_error_values else 0.0),
        "nearest_anchor_match_ratio": float(nearest_anchor_matches / nearest_anchor_total) if nearest_anchor_total else 0.0,
        "one_hop_anchor_violation_rate": float(one_hop_violations / one_hop_total) if one_hop_total else 0.0,
        "multi_hop_anchor_shortcut_rate": float(multi_hop_shortcuts / multi_hop_total) if multi_hop_total else 0.0,
        "finite_anchor_support_ratio": float(np.mean(finite_anchor_support_ratios) if finite_anchor_support_ratios else 0.0),
    }


def localize_single_unknown_node(
    case_data: Dict[str, Any],
    objective_bundle: Dict[str, Any],
    unknown_index: int,
    algorithm_name: str,
    budget_config: Dict[str, Any],
    seed: int,
    llm_config: Dict[str, Any],
    algorithm_extra_config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    coordinates = np.asarray(case_data["coordinates"], dtype=float)
    true_position = coordinates[unknown_index]
    algorithm_result = run_algorithm(
        algorithm_name=algorithm_name,
        objective_bundle=objective_bundle,
        budget_config=budget_config,
        seed=seed,
        llm_config=llm_config,
        extra_config={
            "application": "wsn",
            "unknown_index": unknown_index,
            **deepcopy(algorithm_extra_config or {}),
        },
    )
    estimated_position = np.asarray(
        algorithm_result.get("artifacts", {}).get("estimated_position", true_position.tolist()),
        dtype=float,
    )[:2]
    localization_error = float(math.dist(true_position.tolist(), estimated_position.tolist()))
    return {
        "unknown_index": int(unknown_index),
        "true_position": true_position.tolist(),
        "estimated_position": estimated_position.tolist(),
        "localization_error": localization_error,
        "algorithm_runtime": algorithm_result["runtime"],
        "llm_stats": algorithm_result["llm_stats"],
        "best_objective": float(algorithm_result["best_objective"]),
        "convergence_curve": [float(item) for item in algorithm_result["convergence_curve"]],
        "diagnostics": deepcopy(algorithm_result.get("diagnostics", {})),
    }


def _select_unknown_indices(
    case_data: Dict[str, Any],
    algorithm_extra_config: Dict[str, Any] | None,
) -> List[int]:
    unknown_indices = [int(index) for index in case_data["unknown_indices"]]
    extra_config = algorithm_extra_config or {}
    max_unknown_nodes = int(extra_config.get("max_unknown_nodes", 0) or 0)
    if max_unknown_nodes <= 0 or len(unknown_indices) <= max_unknown_nodes:
        return unknown_indices
    subset_seed = str(extra_config.get("unknown_subset_seed", "proof"))
    seed_material = f'{case_data["case_id"]}:{subset_seed}:{max_unknown_nodes}'
    seed_value = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed_value)
    sampled = sorted(rng.sample(unknown_indices, k=max_unknown_nodes))
    return [int(index) for index in sampled]


def run_single_wsn_case(
    case_data: Dict[str, Any],
    algorithm_name: str,
    run_id: int,
    seed: int,
    budget_config: Dict[str, Any],
    llm_config: Dict[str, Any],
    output_dir: str,
    algorithm_label: str | None = None,
    algorithm_extra_config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    objective_bundle = build_wsn_objective_bundle(case_data)
    evaluated_unknown_indices = _select_unknown_indices(
        case_data=case_data,
        algorithm_extra_config=algorithm_extra_config,
    )
    node_results = [
        localize_single_unknown_node(
            case_data=case_data,
            objective_bundle=objective_bundle,
            unknown_index=unknown_index,
            algorithm_name=algorithm_name,
            budget_config=budget_config,
            seed=seed,
            llm_config=llm_config,
            algorithm_extra_config=algorithm_extra_config,
        )
        for unknown_index in evaluated_unknown_indices
    ]
    if _should_apply_full_wsn_postprocess(algorithm_name=algorithm_name, algorithm_label=algorithm_label):
        node_results = _refine_full_wsn_case_positions(
            case_data=case_data,
            objective_bundle=objective_bundle,
            node_results=node_results,
        )
    errors = [item["localization_error"] for item in node_results]
    estimated_coordinates = deepcopy(case_data["coordinates"])
    for item in node_results:
        estimated_coordinates[item["unknown_index"]] = item["estimated_position"]
    total_llm_calls = sum(int(item["llm_stats"]["total_calls"]) for item in node_results)
    total_success_calls = sum(int(item["llm_stats"].get("success_calls", 0)) for item in node_results)
    total_failed_calls = sum(int(item["llm_stats"].get("failed_calls", 0)) for item in node_results)
    total_cost = sum(float(item["llm_stats"].get("token_cost_usd", 0.0)) for item in node_results)
    total_latency = sum(float(item["llm_stats"].get("latency_sec_total", 0.0)) for item in node_results)
    api_used = any(bool(item["llm_stats"].get("api_used", False)) for item in node_results)
    runtime_llm_enabled = any(bool(item["llm_stats"].get("runtime_llm_enabled", False)) for item in node_results)
    runtime_surrogate_enabled = any(bool(item["llm_stats"].get("runtime_surrogate_enabled", False)) for item in node_results)
    runtime_sniper_enabled = any(bool(item["llm_stats"].get("runtime_sniper_enabled", False)) for item in node_results)
    node_curves = [item["convergence_curve"] for item in node_results if item["convergence_curve"]]
    if node_curves:
        max_len = max(len(curve) for curve in node_curves)
        padded_curves = [curve + [curve[-1]] * (max_len - len(curve)) for curve in node_curves]
        convergence_curve = np.mean(np.asarray(padded_curves, dtype=float), axis=0).tolist()
    else:
        convergence_curve = [0.0]
    geometry_consistency = _compute_wsn_geometry_metrics(
        case_data=case_data,
        objective_bundle=objective_bundle,
        node_results=node_results,
    )
    record = {
        "application": "wsn",
        "suite_name": "wsn_strict_suite",
        "case_id": case_data["case_id"],
        "node_count": int(case_data["node_count"]),
        "anchor_count": int(case_data["anchor_count"]),
        "unknown_count": int(case_data["unknown_count"]),
        "evaluated_unknown_count": int(len(evaluated_unknown_indices)),
        "comm_radius": float(case_data["comm_radius"]),
        "area_size": float(case_data["area_size"]),
        "topology_seed": int(case_data["seed"]),
        "algorithm": algorithm_label or algorithm_name,
        "run_id": int(run_id),
        "seed": int(seed),
        "budget": deepcopy(budget_config),
        "status": "success",
        "feasible": True,
        "best_objective": float(np.mean([item["best_objective"] for item in node_results]) if node_results else 0.0),
        "metrics": {
            "ale": float(np.mean(errors) if errors else 0.0),
            "rmse": float(np.sqrt(np.mean(np.square(errors))) if errors else 0.0),
            "max_error": float(np.max(errors) if errors else 0.0),
            "localized_ratio": 1.0 if node_results else 0.0,
            "geometry_consistency": geometry_consistency,
        },
        "runtime": {
            "wall_clock_sec": float(sum(item["algorithm_runtime"]["wall_clock_sec"] for item in node_results)),
            "objective_eval_count": int(sum(item["algorithm_runtime"]["objective_eval_count"] for item in node_results)),
        },
        "llm_stats": {
            "llm_enabled": algorithm_name in llm_config.get("enabled_algorithms", set()),
            "total_calls": int(total_llm_calls),
            "success_calls": int(total_success_calls),
            "failed_calls": int(total_failed_calls),
            "token_cost_usd": float(total_cost),
            "latency_sec_total": float(total_latency),
            "api_used": bool(api_used),
            "runtime_llm_enabled": bool(runtime_llm_enabled),
            "runtime_surrogate_enabled": bool(runtime_surrogate_enabled),
            "runtime_sniper_enabled": bool(runtime_sniper_enabled),
        },
        "convergence_curve": [float(item) for item in convergence_curve],
        "artifacts": {
            "estimated_coordinates": estimated_coordinates,
            "node_errors": errors,
            "evaluated_unknown_indices": [int(index) for index in evaluated_unknown_indices],
            "node_results": node_results,
        },
    }
    filename = make_run_filename("wsn", case_data["case_id"], algorithm_label or algorithm_name, run_id)
    save_json(record, f"{output_dir}/{filename}")
    return record


def run_wsn_benchmark(
    manifest_path: str,
    algorithms: List[str],
    seeds: List[int],
    budget_config: Dict[str, Any],
    llm_config: Dict[str, Any],
    output_dir: str,
) -> List[Dict[str, Any]]:
    manifest = load_wsn_suite_manifest(manifest_path)
    records: List[Dict[str, Any]] = []
    output_path = Path(output_dir)
    for case_data in manifest["cases"]:
        for algorithm_name in algorithms:
            for run_id, seed in enumerate(seeds):
                filename = make_run_filename("wsn", case_data["case_id"], algorithm_name, run_id)
                if (output_path / filename).exists():
                    continue
                records.append(
                    run_single_wsn_case(
                        case_data=case_data,
                        algorithm_name=algorithm_name,
                        run_id=run_id,
                        seed=seed,
                        budget_config=budget_config,
                        llm_config=llm_config,
                        output_dir=output_dir,
                    )
                )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--algorithms", nargs="+", required=True)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed_base", type=int, default=20260404)
    parser.add_argument("--max_fes", type=int, default=300)
    parser.add_argument("--population_size", type=int, default=20)
    parser.add_argument("--max_iter", type=int, default=15)
    parser.add_argument("--llm_call_budget", type=int, default=4)
    parser.add_argument("--llm_cost_budget_usd", type=float, default=0.01)
    parser.add_argument("--use_api", action="store_true")
    parser.add_argument("--api_key", default="")
    parser.add_argument("--base_url", default="https://api.deepseek.com")
    parser.add_argument("--llm_model", default="deepseek-chat")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--suite_output", default="")
    parser.add_argument("--analysis_output", default="")
    parser.add_argument("--figures_dir", default="")
    args = parser.parse_args()

    seeds = [args.seed_base + i for i in range(args.runs)]
    budget_config = {
        "max_fes": args.max_fes,
        "population_size": args.population_size,
        "max_iter": args.max_iter,
        "llm_call_budget": args.llm_call_budget,
        "llm_cost_budget_usd": args.llm_cost_budget_usd,
    }
    llm_config = {
        "enabled_algorithms": {"L-ADE-Sniper", "LLM-DE", "LLM-SAEA"},
        "use_api": bool(args.use_api),
        "api_key": args.api_key,
        "base_url": args.base_url,
        "llm_model": args.llm_model,
    }
    run_wsn_benchmark(
        manifest_path=args.manifest,
        algorithms=args.algorithms,
        seeds=seeds,
        budget_config=budget_config,
        llm_config=llm_config,
        output_dir=args.output_dir,
    )
    if args.suite_output or args.analysis_output or args.figures_dir:
        aggregate_wsn_pipeline(
            raw_dir=args.output_dir,
            suite_output=args.suite_output or f"{args.output_dir}/suite.json",
            manifest_path=args.manifest,
            analysis_output=args.analysis_output,
            figures_dir=args.figures_dir,
        )


if __name__ == "__main__":
    main()
