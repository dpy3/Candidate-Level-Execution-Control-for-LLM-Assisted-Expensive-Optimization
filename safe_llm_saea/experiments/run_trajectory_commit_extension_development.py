from __future__ import annotations

import argparse
import json
from pathlib import Path

from safe_llm_saea.experiments import run_directional_bbob_power_extension as extension
from safe_llm_saea.experiments import run_trajectory_commit_development as development


SOURCE_ROOT = "paper/q1/experiment_outputs/directional_bbob_power_extension_20260718"
OUTPUT_ROOT = "paper/q1/experiment_outputs/trajectory_commit_extension_development_20260719"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_root", default=SOURCE_ROOT)
    parser.add_argument("--output_root", default=OUTPUT_ROOT)
    args = parser.parse_args()
    extension._configure()
    report = development.execute(Path(args.source_root), Path(args.output_root))
    print(json.dumps({
        "protocol_sha256": report["protocol_sha256"],
        "candidate_events": report["candidate_events"],
        "paired_continuations": report["paired_continuations"],
    }, indent=2))


if __name__ == "__main__":
    main()
