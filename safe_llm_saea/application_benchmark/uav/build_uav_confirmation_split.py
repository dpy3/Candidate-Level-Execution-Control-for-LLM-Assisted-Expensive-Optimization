from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from safe_llm_saea.application_benchmark.common.io_utils import load_json, save_json
from safe_llm_saea.application_benchmark.uav.build_uav_development_split import geometry_sha256
from safe_llm_saea.application_benchmark.uav.build_uav_suite import (
    _DEFAULT_BOUNDS,
    _make_synthetic_uav_scenario,
    validate_uav_suite_manifest,
)


_CONFIRMATION_TEMPLATES = [
    {
        "start": [14.0, 8.0, 10.0], "goal": [176.0, 184.0, 33.0],
        "waypoints": [[0.14, 30.0, 12.0], [0.32, 20.0, 16.0], [0.50, -28.0, 21.0], [0.69, -34.0, 27.0], [0.86, 6.0, 31.0]],
        "obstacles": [[0.22, 0.0, 14.0, 16.0], [0.40, -5.0, 18.0, 17.0], [0.59, 5.0, 23.0, 18.0], [0.78, 2.0, 29.0, 15.0]],
        "tags": ["confirmation", "long-diagonal", "altitude-turn"],
    },
    {
        "start": [6.0, 148.0, 13.0], "goal": [184.0, 42.0, 28.0],
        "waypoints": [[0.16, -26.0, 14.0], [0.34, -32.0, 17.0], [0.52, 10.0, 21.0], [0.70, 30.0, 24.0], [0.87, 14.0, 27.0]],
        "obstacles": [[0.23, 3.0, 15.0, 16.0], [0.42, 7.0, 18.0, 18.0], [0.61, -4.0, 22.0, 17.0], [0.80, -2.0, 25.0, 16.0]],
        "tags": ["confirmation", "reverse-diagonal", "double-bend"],
    },
    {
        "start": [42.0, 186.0, 12.0], "goal": [138.0, 4.0, 34.0],
        "waypoints": [[0.15, 22.0, 14.0], [0.33, 34.0, 18.0], [0.51, -2.0, 23.0], [0.69, -30.0, 29.0], [0.86, -18.0, 33.0]],
        "obstacles": [[0.21, -4.0, 15.0, 15.0], [0.40, 2.0, 20.0, 18.0], [0.59, 4.0, 25.0, 17.0], [0.78, 6.0, 30.0, 15.0]],
        "tags": ["confirmation", "steep-reverse", "narrow-midpoint"],
    },
    {
        "start": [4.0, 34.0, 16.0], "goal": [186.0, 158.0, 20.0],
        "waypoints": [[0.16, -32.0, 16.0], [0.35, -18.0, 17.0], [0.53, 32.0, 18.0], [0.71, 26.0, 19.0], [0.87, -10.0, 20.0]],
        "obstacles": [[0.24, 0.0, 16.5, 17.0], [0.43, 5.0, 17.5, 18.0], [0.62, -6.0, 18.5, 18.0], [0.80, -2.0, 19.5, 16.0]],
        "tags": ["confirmation", "flat-slalom", "terminal-turn"],
    },
]


def build_confirmation_manifest(test_manifest_path: str, development_manifest_path: str, output_path: str) -> Dict[str, Any]:
    source_manifests = [load_json(test_manifest_path), load_json(development_manifest_path)]
    source_hashes = {
        geometry_sha256(scenario)
        for manifest in source_manifests
        for scenario in manifest.get("scenarios", [])
    }
    source_ids = {
        str(scenario["case_id"])
        for manifest in source_manifests
        for scenario in manifest.get("scenarios", [])
    }
    scenarios: List[Dict[str, Any]] = []
    for index, template in enumerate(_CONFIRMATION_TEMPLATES):
        scenario = _make_synthetic_uav_scenario(
            scenario_id=200 + index, bounds=dict(_DEFAULT_BOUNDS), control_point_dim=24,
            tags=list(template["tags"]), start=list(template["start"]), goal=list(template["goal"]),
            waypoint_specs=list(template["waypoints"]), obstacle_specs=list(template["obstacles"]),
        )
        scenario["case_id"] = f"confirm_hard_c{index:02d}"
        scenario["split"] = "confirmation"
        scenario["geometry_sha256"] = geometry_sha256(scenario)
        if scenario["case_id"] in source_ids or scenario["geometry_sha256"] in source_hashes:
            raise ValueError(f"Confirmation overlap detected: {scenario['case_id']}")
        scenarios.append(scenario)
    manifest = {
        "application": "uav", "suite_name": "uav_hard_confirmation_split", "split": "confirmation",
        "source_test_manifest": str(Path(test_manifest_path).resolve()),
        "source_development_manifest": str(Path(development_manifest_path).resolve()),
        "levels": ["hard"], "synthetic_hard_cases": len(scenarios), "level_counts": {"hard": len(scenarios)},
        "scenarios": scenarios,
        "overlap_audit": {"case_id_overlap": False, "geometry_hash_overlap": False, "source_cases": len(source_ids)},
    }
    validate_uav_suite_manifest(manifest)
    save_json(manifest, output_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_manifest", required=True)
    parser.add_argument("--development_manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    build_confirmation_manifest(args.test_manifest, args.development_manifest, args.output)


if __name__ == "__main__":
    main()
