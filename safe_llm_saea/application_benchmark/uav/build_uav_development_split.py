from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

from safe_llm_saea.application_benchmark.common.io_utils import load_json, save_json
from safe_llm_saea.application_benchmark.uav.build_uav_suite import (
    _DEFAULT_BOUNDS,
    _make_synthetic_uav_scenario,
    validate_uav_suite_manifest,
)


_DEVELOPMENT_TEMPLATES = [
    {
        "start": [10.0, 52.0, 11.0],
        "goal": [180.0, 126.0, 31.0],
        "waypoints": [[0.15, -30.0, 13.0], [0.31, -12.0, 16.0], [0.49, 30.0, 20.0], [0.68, 26.0, 25.0], [0.85, -8.0, 29.0]],
        "obstacles": [[0.21, 0.0, 14.0, 16.0], [0.39, 8.0, 18.0, 17.0], [0.58, -6.0, 22.0, 18.0], [0.77, -4.0, 27.0, 16.0]],
        "tags": ["development", "wide-s-turn", "climb"],
    },
    {
        "start": [30.0, 176.0, 14.0],
        "goal": [154.0, 10.0, 26.0],
        "waypoints": [[0.17, 24.0, 15.0], [0.35, 34.0, 17.0], [0.52, -6.0, 20.0], [0.70, -30.0, 23.0], [0.87, -12.0, 25.0]],
        "obstacles": [[0.24, -2.0, 16.0, 15.0], [0.42, 4.0, 18.5, 18.0], [0.61, 2.0, 21.5, 17.0], [0.80, 6.0, 24.0, 15.0]],
        "tags": ["development", "reverse-sweep", "late-corridor"],
    },
    {
        "start": [8.0, 92.0, 9.0],
        "goal": [182.0, 88.0, 35.0],
        "waypoints": [[0.14, -25.0, 12.0], [0.30, -34.0, 17.0], [0.48, 4.0, 23.0], [0.66, 32.0, 29.0], [0.84, 18.0, 34.0]],
        "obstacles": [[0.20, 2.0, 14.0, 16.0], [0.38, -2.0, 19.0, 17.0], [0.57, 0.0, 25.0, 18.0], [0.75, -4.0, 30.0, 16.0]],
        "tags": ["development", "cross-map", "steep-climb"],
    },
    {
        "start": [36.0, 12.0, 15.0],
        "goal": [146.0, 182.0, 22.0],
        "waypoints": [[0.16, 28.0, 15.0], [0.33, 18.0, 16.0], [0.51, -30.0, 18.0], [0.70, -24.0, 20.0], [0.86, 12.0, 21.0]],
        "obstacles": [[0.23, 0.0, 16.0, 17.0], [0.41, -8.0, 17.0, 18.0], [0.60, 6.0, 19.0, 17.0], [0.79, 2.0, 21.0, 16.0]],
        "tags": ["development", "low-altitude", "alternating-gates"],
    },
]


def geometry_payload(scenario: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "start": scenario["start"],
        "goal": scenario["goal"],
        "bounds": scenario["bounds"],
        "path_parameterization": scenario["path_parameterization"],
        "obstacles": scenario.get("obstacles", []),
        "reference_path": scenario.get("reference_path", []),
    }


def geometry_sha256(scenario: Dict[str, Any]) -> str:
    encoded = json.dumps(geometry_payload(scenario), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_development_manifest(test_manifest_path: str, output_path: str) -> Dict[str, Any]:
    test_manifest = load_json(test_manifest_path)
    test_hashes = {geometry_sha256(item) for item in test_manifest.get("scenarios", [])}
    scenarios: List[Dict[str, Any]] = []
    for index, template in enumerate(_DEVELOPMENT_TEMPLATES):
        scenario = _make_synthetic_uav_scenario(
            scenario_id=100 + index,
            bounds=dict(_DEFAULT_BOUNDS),
            control_point_dim=24,
            tags=list(template["tags"]),
            start=list(template["start"]),
            goal=list(template["goal"]),
            waypoint_specs=list(template["waypoints"]),
            obstacle_specs=list(template["obstacles"]),
        )
        scenario["case_id"] = f"dev_hard_d{index:02d}"
        scenario["split"] = "development"
        scenario["geometry_sha256"] = geometry_sha256(scenario)
        if scenario["geometry_sha256"] in test_hashes:
            raise ValueError(f"Development geometry overlaps the locked test set: {scenario['case_id']}")
        scenarios.append(scenario)
    manifest = {
        "application": "uav",
        "suite_name": "uav_hard_development_split",
        "split": "development",
        "source_test_manifest": str(Path(test_manifest_path).resolve()),
        "levels": ["hard"],
        "synthetic_hard_cases": len(scenarios),
        "level_counts": {"hard": len(scenarios)},
        "test_geometry_hashes": sorted(test_hashes),
        "scenarios": scenarios,
        "overlap_audit": {
            "case_id_overlap": False,
            "geometry_hash_overlap": False,
            "development_cases": len(scenarios),
            "locked_test_cases": len(test_hashes),
        },
    }
    validate_uav_suite_manifest(manifest)
    save_json(manifest, output_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a non-overlapping UAV development split.")
    parser.add_argument("--test_manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    build_development_manifest(args.test_manifest, args.output)


if __name__ == "__main__":
    main()
