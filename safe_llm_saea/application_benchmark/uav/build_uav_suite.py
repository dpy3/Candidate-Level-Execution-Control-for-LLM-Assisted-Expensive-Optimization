from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List

from safe_llm_saea.application_benchmark.common.io_utils import save_json

_LEVEL_ALIASES = {
    "single_peak": "easy",
    "easy": "easy",
    "multi_peak": "medium",
    "medium": "medium",
    "cluttered": "hard",
    "hard": "hard",
}
_DEFAULT_BOUNDS = {"x": [-10.0, 190.0], "y": [-10.0, 190.0], "z": [5.0, 40.0]}
_DEFAULT_SYNTHETIC_HARD_CASES = 8


def _point_in_bounds(point: List[float], bounds: Dict[str, List[float]], margin: float = 0.0) -> bool:
    x_value, y_value, z_value = map(float, point)
    return (
        float(bounds["x"][0]) - margin <= x_value <= float(bounds["x"][1]) + margin
        and float(bounds["y"][0]) - margin <= y_value <= float(bounds["y"][1]) + margin
        and float(bounds["z"][0]) - margin <= z_value <= float(bounds["z"][1]) + margin
    )


def _path_in_bounds(path: List[List[float]], bounds: Dict[str, List[float]], margin: float = 0.0) -> bool:
    return all(_point_in_bounds(point, bounds, margin=margin) for point in path)


def _path_collision_count(path: List[List[float]], obstacles: List[Dict[str, Any]]) -> int:
    collision_count = 0
    for left, right in zip(path[:-1], path[1:]):
        segment_length = math.dist(left, right)
        sample_count = max(3, int(segment_length / 4.0) + 1)
        for step in range(sample_count):
            alpha = step / max(1, sample_count - 1)
            point = [
                left[0] * (1.0 - alpha) + right[0] * alpha,
                left[1] * (1.0 - alpha) + right[1] * alpha,
                left[2] * (1.0 - alpha) + right[2] * alpha,
            ]
            for obstacle in obstacles:
                if obstacle.get("type") != "sphere":
                    continue
                center = obstacle["center"]
                radius = float(obstacle["radius"])
                clearance = math.dist(point, center) - radius
                if clearance < 0.0:
                    collision_count += 1
    return collision_count


def _path_min_distance_to_point(path: List[List[float]], point: List[float]) -> float:
    min_distance = float("inf")
    for left, right in zip(path[:-1], path[1:]):
        segment_length = math.dist(left, right)
        sample_count = max(5, int(segment_length / 3.0) + 1)
        for step in range(sample_count):
            alpha = step / max(1, sample_count - 1)
            probe = [
                left[0] * (1.0 - alpha) + right[0] * alpha,
                left[1] * (1.0 - alpha) + right[1] * alpha,
                left[2] * (1.0 - alpha) + right[2] * alpha,
            ]
            min_distance = min(min_distance, math.dist(probe, point))
    return min_distance


def _interpolate_point(start: List[float], goal: List[float], progress: float) -> List[float]:
    return [
        float(start[index] * (1.0 - progress) + goal[index] * progress)
        for index in range(3)
    ]


def _lateral_unit(start: List[float], goal: List[float]) -> List[float]:
    forward_x = float(goal[0] - start[0])
    forward_y = float(goal[1] - start[1])
    norm = math.hypot(forward_x, forward_y)
    if norm < 1e-12:
        return [0.0, 1.0]
    return [-forward_y / norm, forward_x / norm]


def _build_offset_point(
    start: List[float],
    goal: List[float],
    progress: float,
    lateral_offset: float,
    altitude: float,
) -> List[float]:
    base = _interpolate_point(start, goal, progress)
    lateral = _lateral_unit(start, goal)
    return [
        float(base[0] + lateral_offset * lateral[0]),
        float(base[1] + lateral_offset * lateral[1]),
        float(altitude),
    ]


def _make_sphere(center: List[float], radius: float) -> Dict[str, Any]:
    return {"type": "sphere", "center": [float(value) for value in center], "radius": float(radius)}


def _make_synthetic_uav_scenario(
    scenario_id: int,
    bounds: Dict[str, List[float]],
    control_point_dim: int,
    tags: List[str],
    start: List[float],
    goal: List[float],
    waypoint_specs: List[List[float]],
    obstacle_specs: List[List[float]],
) -> Dict[str, Any]:
    reference_path = [start]
    for progress, lateral_offset, altitude in waypoint_specs:
        reference_path.append(_build_offset_point(start, goal, float(progress), float(lateral_offset), float(altitude)))
    reference_path.append(goal)
    obstacles = [
        _make_sphere(
            _build_offset_point(start, goal, float(progress), float(lateral_offset), float(altitude)),
            float(radius),
        )
        for progress, lateral_offset, altitude, radius in obstacle_specs
    ]
    for obstacle in obstacles:
        min_distance = _path_min_distance_to_point(reference_path, list(obstacle["center"]))
        obstacle["radius"] = min(float(obstacle["radius"]), max(3.0, min_distance - 2.0))
    if not _path_in_bounds(reference_path, bounds):
        raise ValueError(f"Synthetic hard scenario hard_s{scenario_id:02d} exceeds UAV bounds")
    collision_count = _path_collision_count(reference_path, obstacles)
    shrink_rounds = 0
    lateral = _lateral_unit(start, goal)
    while collision_count > 0 and shrink_rounds < 8:
        for obstacle_index, obstacle in enumerate(obstacles):
            direction = 1.0 if obstacle_index % 2 == 0 else -1.0
            obstacle["radius"] = max(3.0, float(obstacle["radius"]) - 1.0)
            obstacle["center"][0] = float(obstacle["center"][0] + direction * 2.0 * lateral[0])
            obstacle["center"][1] = float(obstacle["center"][1] + direction * 2.0 * lateral[1])
        collision_count = _path_collision_count(reference_path, obstacles)
        shrink_rounds += 1
    if collision_count > 0:
        raise ValueError(f"Synthetic hard scenario hard_s{scenario_id:02d} has colliding reference path")
    return {
        "application": "uav",
        "case_id": f"hard_s{scenario_id:02d}",
        "level": "hard",
        "scenario_id": int(scenario_id),
        "start": [float(value) for value in start],
        "goal": [float(value) for value in goal],
        "bounds": bounds,
        "path_parameterization": {
            "control_point_count": int(control_point_dim // 3),
            "dimension": int(control_point_dim),
        },
        "obstacles": obstacles,
        "difficulty_tags": ["hard", *tags],
        "reference_path": reference_path,
    }


def generate_synthetic_hard_scenarios(
    bounds: Dict[str, List[float]],
    control_point_dim: int,
    synthetic_hard_cases: int,
) -> List[Dict[str, Any]]:
    templates = [
        {
            "start": [12.0, 18.0, 12.0],
            "goal": [168.0, 172.0, 26.0],
            "waypoints": [[0.16, -32.0, 12.5], [0.30, -28.0, 14.0], [0.45, 8.0, 17.0], [0.62, 32.0, 20.5], [0.80, 22.0, 24.0]],
            "obstacles": [[0.22, 4.0, 16.0, 17.0], [0.38, -2.0, 18.0, 18.0], [0.56, -6.0, 20.0, 18.0], [0.72, 2.0, 22.0, 17.0]],
            "tags": ["slalom", "diagonal"],
        },
        {
            "start": [18.0, 164.0, 13.0],
            "goal": [172.0, 24.0, 28.0],
            "waypoints": [[0.18, 28.0, 14.5], [0.34, 24.0, 17.0], [0.50, -14.0, 20.0], [0.68, -32.0, 23.0], [0.84, -18.0, 26.0]],
            "obstacles": [[0.24, 2.0, 16.0, 16.0], [0.40, 6.0, 18.0, 17.0], [0.58, -4.0, 21.0, 18.0], [0.74, -2.0, 24.0, 16.0]],
            "tags": ["reverse-diagonal", "s-bend"],
        },
        {
            "start": [14.0, 28.0, 10.0],
            "goal": [176.0, 150.0, 30.0],
            "waypoints": [[0.18, -34.0, 12.0], [0.34, -10.0, 14.5], [0.50, 26.0, 18.0], [0.68, 34.0, 22.5], [0.84, 10.0, 27.0]],
            "obstacles": [[0.28, -12.0, 16.0, 15.0], [0.28, 12.0, 16.0, 15.0], [0.58, -10.0, 21.0, 16.0], [0.58, 12.0, 21.0, 16.0]],
            "tags": ["double-gate", "corridor"],
        },
        {
            "start": [16.0, 34.0, 11.0],
            "goal": [166.0, 160.0, 34.0],
            "waypoints": [[0.15, -18.0, 14.0], [0.32, -26.0, 18.0], [0.50, -8.0, 24.0], [0.68, 18.0, 30.0], [0.84, 12.0, 34.5]],
            "obstacles": [[0.22, 6.0, 15.0, 16.0], [0.40, 8.0, 18.0, 17.0], [0.58, 4.0, 22.0, 17.0], [0.74, -2.0, 26.0, 16.0], [0.88, 4.0, 31.0, 14.0]],
            "tags": ["altitude-climb", "late-turn"],
        },
        {
            "start": [20.0, 22.0, 12.0],
            "goal": [170.0, 170.0, 24.0],
            "waypoints": [[0.16, -24.0, 12.5], [0.34, -18.0, 13.5], [0.52, 18.0, 16.0], [0.72, 34.0, 19.5], [0.88, 20.0, 22.0]],
            "obstacles": [[0.24, 0.0, 14.0, 18.0], [0.44, -4.0, 16.0, 18.0], [0.66, 0.0, 18.0, 18.0], [0.82, -18.0, 21.0, 15.0], [0.88, 10.0, 22.5, 14.0]],
            "tags": ["terminal-cage", "goal-pressure"],
        },
        {
            "start": [24.0, 150.0, 12.0],
            "goal": [164.0, 26.0, 27.0],
            "waypoints": [[0.14, 20.0, 13.0], [0.30, 32.0, 15.0], [0.48, 10.0, 18.0], [0.66, -18.0, 22.0], [0.84, -26.0, 25.0]],
            "obstacles": [[0.18, -8.0, 14.0, 15.0], [0.38, 0.0, 16.5, 18.0], [0.56, 6.0, 19.0, 18.0], [0.72, 10.0, 22.0, 16.0], [0.78, -18.0, 23.0, 14.0]],
            "tags": ["narrow-corridor", "descending"],
        },
        {
            "start": [18.0, 44.0, 10.0],
            "goal": [174.0, 140.0, 32.0],
            "waypoints": [[0.18, -28.0, 12.0], [0.36, -24.0, 16.0], [0.54, 18.0, 21.0], [0.70, 30.0, 26.0], [0.86, 8.0, 30.0]],
            "obstacles": [[0.26, 8.0, 15.0, 17.0], [0.42, 12.0, 18.0, 17.0], [0.58, -8.0, 21.0, 17.0], [0.74, -10.0, 25.0, 16.0], [0.62, 26.0, 24.0, 13.0]],
            "tags": ["cross-fire", "alternating"],
        },
        {
            "start": [22.0, 132.0, 11.0],
            "goal": [168.0, 32.0, 29.0],
            "waypoints": [[0.16, 12.0, 13.0], [0.34, 30.0, 16.0], [0.52, 22.0, 20.0], [0.70, -12.0, 24.0], [0.86, -26.0, 27.0]],
            "obstacles": [[0.20, -14.0, 14.0, 14.0], [0.40, 2.0, 17.0, 17.0], [0.58, 4.0, 21.0, 17.0], [0.76, 8.0, 24.0, 16.0], [0.86, -4.0, 26.0, 15.0]],
            "tags": ["offset-clutter", "late-descent"],
        },
    ]
    case_count = min(len(templates), max(0, int(synthetic_hard_cases)))
    return [
        _make_synthetic_uav_scenario(
            scenario_id=index,
            bounds=bounds,
            control_point_dim=control_point_dim,
            tags=list(templates[index]["tags"]),
            start=list(templates[index]["start"]),
            goal=list(templates[index]["goal"]),
            waypoint_specs=list(templates[index]["waypoints"]),
            obstacle_specs=list(templates[index]["obstacles"]),
        )
        for index in range(case_count)
    ]


def _scenario_geometry_is_usable(scenario: Dict[str, Any], bounds: Dict[str, List[float]]) -> bool:
    reference_path = scenario.get("reference_path") or [scenario["start"], scenario["goal"]]
    return (
        _point_in_bounds(list(scenario["start"]), bounds)
        and _point_in_bounds(list(scenario["goal"]), bounds)
        and _path_in_bounds(reference_path, bounds, margin=3.0)
    )


def load_base_scenarios(source_dir: str) -> List[Dict[str, Any]]:
    source_path = Path(source_dir)
    records: List[Dict[str, Any]] = []
    for json_path in sorted(source_path.glob("final_benchmark_results*.json")):
        import json

        with json_path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
            if isinstance(payload, list):
                records.extend(payload)
    return records


def canonicalize_uav_levels(levels: List[str]) -> List[str]:
    canonical_levels: List[str] = []
    for level in levels:
        normalized_level = _LEVEL_ALIASES.get(level, level)
        if normalized_level not in canonical_levels:
            canonical_levels.append(normalized_level)
    return canonical_levels


def normalize_uav_scenario(raw_scenario: Dict[str, Any], level: str, scenario_id: int) -> Dict[str, Any]:
    first_path = raw_scenario.get("paths", [[]])[0] or []
    start = first_path[0] if first_path else raw_scenario.get("start", [0.0, 0.0, 0.0])
    goal = first_path[-1] if first_path else raw_scenario.get("goal", [1.0, 1.0, 1.0])
    return {
        "application": "uav",
        "case_id": f"{level}_s{int(scenario_id):02d}",
        "level": level,
        "scenario_id": int(scenario_id),
        "start": start,
        "goal": goal,
        "bounds": {"x": [-10.0, 190.0], "y": [-10.0, 190.0], "z": [5.0, 40.0]},
        "path_parameterization": {
            "control_point_count": 8,
            "dimension": 24,
        },
        "obstacles": raw_scenario.get("obstacles", []),
        "difficulty_tags": [level],
        "reference_path": first_path,
    }


def build_uav_suite_manifest(
    source_dir: str,
    output_path: str,
    levels: List[str],
    control_point_dim: int,
    bounds: Dict[str, Any],
    synthetic_hard_cases: int = _DEFAULT_SYNTHETIC_HARD_CASES,
) -> Dict[str, Any]:
    records = load_base_scenarios(source_dir)
    canonical_levels = canonicalize_uav_levels(levels)
    scenarios: Dict[str, Dict[str, Any]] = {}
    for record in records:
        level = _LEVEL_ALIASES.get(record.get("level"), record.get("level"))
        if level not in canonical_levels:
            continue
        scenario_id = int(record.get("scenario_id", -1))
        case_id = f"{level}_s{scenario_id:02d}"
        if case_id not in scenarios:
            normalized = normalize_uav_scenario(record, level=level, scenario_id=scenario_id)
            normalized["path_parameterization"]["dimension"] = int(control_point_dim)
            normalized["bounds"] = bounds
            if not _scenario_geometry_is_usable(normalized, bounds):
                continue
            scenarios[case_id] = normalized
    if "hard" in canonical_levels:
        for scenario in generate_synthetic_hard_scenarios(
            bounds=bounds,
            control_point_dim=control_point_dim,
            synthetic_hard_cases=synthetic_hard_cases,
        ):
            scenarios[scenario["case_id"]] = scenario
    ordered_scenarios = sorted(
        scenarios.values(),
        key=lambda scenario: (canonical_levels.index(str(scenario["level"])), int(scenario["scenario_id"])),
    )
    level_counts = {
        level: sum(1 for scenario in ordered_scenarios if scenario["level"] == level)
        for level in canonical_levels
    }
    manifest = {
        "application": "uav",
        "suite_name": f"uav_{'_'.join(canonical_levels)}_suite",
        "levels": canonical_levels,
        "synthetic_hard_cases": int(synthetic_hard_cases if "hard" in canonical_levels else 0),
        "level_counts": level_counts,
        "scenarios": ordered_scenarios,
    }
    validate_uav_suite_manifest(manifest)
    save_json(manifest, output_path)
    return manifest


def validate_uav_suite_manifest(manifest: Dict[str, Any]) -> None:
    if manifest.get("application") != "uav":
        raise ValueError("Manifest application must be 'uav'")
    if not manifest.get("scenarios"):
        raise ValueError("UAV manifest contains no scenarios")
    for scenario in manifest["scenarios"]:
        for key in ("case_id", "start", "goal", "bounds", "path_parameterization"):
            if key not in scenario:
                raise ValueError(f"UAV scenario missing key: {key}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", required=True)
    parser.add_argument("--levels", nargs="+", required=True)
    parser.add_argument("--control_point_count", type=int, default=8)
    parser.add_argument("--synthetic_hard_cases", type=int, default=_DEFAULT_SYNTHETIC_HARD_CASES)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    bounds = dict(_DEFAULT_BOUNDS)
    build_uav_suite_manifest(
        source_dir=args.source_dir,
        output_path=args.output,
        levels=args.levels,
        control_point_dim=args.control_point_count * 3,
        bounds=bounds,
        synthetic_hard_cases=args.synthetic_hard_cases,
    )


if __name__ == "__main__":
    main()
