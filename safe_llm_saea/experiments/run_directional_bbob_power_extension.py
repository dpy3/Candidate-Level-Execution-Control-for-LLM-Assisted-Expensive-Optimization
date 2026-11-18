from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from safe_llm_saea.experiments import run_directional_bbob_confirmation as study


PROTOCOL_ID = "CSA-DIRECTIONAL-BBOB-POWER-EXTENSION-2026-07-18"
INSTANCES = tuple(range(6, 14))
SEED_BASE = 20261718
OUTPUT_ROOT = "paper/q1/experiment_outputs/directional_bbob_power_extension_20260718"


_base_protocol = study._protocol


def _extension_protocol() -> dict[str, Any]:
    payload = _base_protocol()
    payload.pop("protocol_sha256", None)
    payload["protocol_id"] = PROTOCOL_ID
    payload["status"] = "frozen_before_capture"
    payload["benchmark"]["instances"] = list(INSTANCES)
    payload["benchmark"]["exclusion"] = (
        "Instances 1--5 belong to the preceding confirmation and are not reused."
    )
    payload["success_gate"]["minimum_paired_cases_per_backbone"] = 30
    payload["stopping_rule"] = (
        "Execute all 64 fresh backbone-case pairs (4 functions x 8 unseen instances "
        "x 2 backbones); no threshold refitting, early stopping, or function substitution."
    )
    payload["claim_rule"] = (
        "Final-performance effectiveness is eligible only for a backbone passing every "
        "frozen gate on this power-extension batch."
    )
    payload["protocol_sha256"] = study._sha256_json(payload)
    return payload


def _configure() -> None:
    study.PROTOCOL_ID = PROTOCOL_ID
    study.INSTANCES = INSTANCES
    study.SEED_BASE = SEED_BASE
    study.OUTPUT_ROOT = OUTPUT_ROOT
    study._protocol = _extension_protocol


def _validate(root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    valid = 0
    for backbone in ("DE", "PSO"):
        for fid in study.FUNCTIONS:
            for instance in INSTANCES:
                path = study._cache_path(root, backbone, fid, instance)
                if path.exists():
                    study._verify_cache(path, protocol, backbone, fid, instance)
                    valid += 1
    return {
        "protocol_sha256": protocol["protocol_sha256"],
        "valid_caches": valid,
        "expected_caches": 2 * len(study.FUNCTIONS) * len(INSTANCES),
        "live_api_called": False,
    }


def main() -> None:
    _configure()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("freeze", "validate", "capture", "execute", "all"),
        default="validate",
    )
    parser.add_argument("--output_root", default=OUTPUT_ROOT)
    args = parser.parse_args()
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    protocol = study._load_or_freeze(root / "frozen_protocol.json")
    if args.mode == "freeze":
        print(json.dumps({"protocol_sha256": protocol["protocol_sha256"]}, indent=2))
        return
    if args.mode == "validate":
        print(json.dumps(_validate(root, protocol), indent=2))
        return
    if args.mode in {"capture", "all"}:
        study.capture(root, protocol)
    if args.mode in {"execute", "all"}:
        report = study.execute(root, protocol)
        print(json.dumps({
            "backbones": report["backbones"],
            "any_backbone_claim_eligible": report["any_backbone_claim_eligible"],
        }, indent=2))


if __name__ == "__main__":
    main()
