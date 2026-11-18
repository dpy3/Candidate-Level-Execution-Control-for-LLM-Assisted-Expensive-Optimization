from __future__ import annotations

import argparse
from typing import Any, Dict, List

import numpy as np

from safe_llm_saea.application_benchmark.common.io_utils import save_json


def generate_wsn_case(
    node_count: int,
    anchor_count: int,
    comm_radius: float,
    area_size: float,
    seed: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    coordinates = rng.uniform(0.0, area_size, size=(node_count, 2))
    anchor_indices = list(range(anchor_count))
    unknown_indices = list(range(anchor_count, node_count))
    return {
        "application": "wsn",
        "case_id": f"N{node_count}_K{anchor_count}_R{int(comm_radius)}_S{seed}",
        "node_count": int(node_count),
        "anchor_count": int(anchor_count),
        "unknown_count": int(node_count - anchor_count),
        "comm_radius": float(comm_radius),
        "area_size": float(area_size),
        "seed": int(seed),
        "anchor_indices": anchor_indices,
        "unknown_indices": unknown_indices,
        "coordinates": coordinates.tolist(),
    }


def build_wsn_suite_manifest(
    node_counts: List[int],
    anchor_counts: List[int],
    comm_radii: List[float],
    cases_per_setting: int,
    area_size: float,
    seed_base: int,
    output_path: str,
) -> Dict[str, Any]:
    cases: List[Dict[str, Any]] = []
    current_seed = seed_base
    for node_count in node_counts:
        for anchor_count in anchor_counts:
            for comm_radius in comm_radii:
                for _ in range(cases_per_setting):
                    cases.append(
                        generate_wsn_case(
                            node_count=node_count,
                            anchor_count=anchor_count,
                            comm_radius=comm_radius,
                            area_size=area_size,
                            seed=current_seed,
                        )
                    )
                    current_seed += 1
    manifest = {"application": "wsn", "suite_name": "wsn_strict_suite", "cases": cases}
    validate_wsn_suite_manifest(manifest)
    save_json(manifest, output_path)
    return manifest


def validate_wsn_suite_manifest(manifest: Dict[str, Any]) -> None:
    if manifest.get("application") != "wsn":
        raise ValueError("Manifest application must be 'wsn'")
    if not manifest.get("cases"):
        raise ValueError("WSN manifest contains no cases")
    for case in manifest["cases"]:
        for key in ("case_id", "anchor_indices", "unknown_indices", "coordinates"):
            if key not in case:
                raise ValueError(f"WSN case missing key: {key}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node_counts", nargs="+", type=int, required=True)
    parser.add_argument("--anchor_counts", nargs="+", type=int, required=True)
    parser.add_argument("--comm_radii", nargs="+", type=float, required=True)
    parser.add_argument("--cases_per_setting", type=int, default=3)
    parser.add_argument("--area_size", type=float, default=100.0)
    parser.add_argument("--seed_base", type=int, default=20260404)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    build_wsn_suite_manifest(
        node_counts=args.node_counts,
        anchor_counts=args.anchor_counts,
        comm_radii=args.comm_radii,
        cases_per_setting=args.cases_per_setting,
        area_size=args.area_size,
        seed_base=args.seed_base,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
