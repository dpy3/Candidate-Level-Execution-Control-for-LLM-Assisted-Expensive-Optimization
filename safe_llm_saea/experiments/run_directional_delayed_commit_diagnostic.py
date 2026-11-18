from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from safe_llm_saea.csa_protocol import RuntimeAdmissionGate
from safe_llm_saea.experiments import run_directional_bbob_confirmation as study
from safe_llm_saea.experiments import run_directional_bbob_power_extension as extension


OUTPUT_ROOT = "paper/q1/experiment_outputs/directional_delayed_commit_diagnostic_20260719"
SOURCE_ROOT = "paper/q1/experiment_outputs/directional_bbob_power_extension_20260718"


def _source_rows(source_root: Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    report = json.loads((source_root / "bbob_confirmation_report.json").read_text(encoding="utf-8"))
    return {
        (row["backbone"], int(row["fid"]), int(row["instance"])): row
        for row in report["records"]
    }


def _runtime(backbone: str, fid: int, instance: int) -> tuple[Any, list[np.ndarray], list[float], float]:
    seed = study._seed(backbone, fid, instance)
    if backbone == "DE":
        optimizer, objective, _ = study._de_runtime(fid, instance, seed)
        return objective, optimizer.X_history, optimizer.y_history, float(optimizer.best_fit)
    state = study._pso_runtime(fid, instance, seed)
    return (
        state["objective"], state["history_x"], state["history_y"],
        float(state["incumbent_objective"]),
    )


def execute(source_root: Path, output_root: Path, source_kind: str = "extension") -> dict[str, Any]:
    if source_kind == "extension":
        extension._configure()
        instances = extension.INSTANCES
    elif source_kind == "original":
        instances = study.INSTANCES
    else:
        raise ValueError(f"unsupported source kind: {source_kind}")
    protocol = study._load_or_freeze(source_root / "frozen_protocol.json")
    source_rows = _source_rows(source_root)
    records: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}

    for backbone in ("DE", "PSO"):
        effects: list[float] = []
        compliant = True
        evaluated = 0
        terminal_commits = 0
        for fid in study.FUNCTIONS:
            for instance in instances:
                source = source_rows[(backbone, fid, instance)]
                objective, history_x, history_y, incumbent = _runtime(backbone, fid, instance)
                cache = study._verify_cache(
                    study._cache_path(source_root, backbone, fid, instance),
                    protocol, backbone, fid, instance,
                )
                candidates = [np.asarray(vector, dtype=float) for vector in cache["candidate_vectors"]]
                selected, selection = study._select(
                    history_x, history_y, candidates, incumbent,
                    study._seed(backbone, fid, instance),
                    study.RISK_MULTIPLIERS[backbone],
                )
                candidate_value = None
                authorization = None
                if selected is not None:
                    gate = RuntimeAdmissionGate(
                        candidates, [selected], remaining_budget=1,
                        event_id=f"BBOB-{backbone}-F{fid}-I{instance}-DELAYED",
                        policy="directional_risk_upper_delayed_commit",
                    )
                    authorization = gate.issue(selected)
                    candidate_value = float(gate.evaluate(authorization, selected, objective))
                    evaluated += 1

                strict_final = float(source["strict_objective"])
                delayed_final = strict_final
                terminal_commit = candidate_value is not None and candidate_value < strict_final
                if terminal_commit:
                    delayed_final = float(candidate_value)
                    terminal_commits += 1
                relative = (delayed_final - strict_final) / max(1.0, abs(strict_final))
                effects.append(float(relative))
                trace = study._trace(
                    candidates, selected, selected is not None,
                    terminal_commit, authorization,
                )
                compliant = compliant and bool(trace["audit"]["csa_compliant"])
                records.append({
                    "backbone": backbone, "fid": fid, "instance": instance,
                    "strict_final": strict_final, "delayed_final": delayed_final,
                    "relative_change": float(relative),
                    "evaluated": selected is not None,
                    "terminal_commit": bool(terminal_commit),
                    "candidate_value": candidate_value,
                    "incumbent_value": incumbent,
                    "immediate_regret": (
                        None if candidate_value is None else
                        float((candidate_value - incumbent) / max(1.0, abs(incumbent)))
                    ),
                    "selection": selection,
                    "csa_compliant": bool(trace["audit"]["csa_compliant"]),
                })

        wins = sum(value < -1e-12 for value in effects)
        losses = sum(value > 1e-12 for value in effects)
        ties = len(effects) - wins - losses
        summaries[backbone] = {
            "runs": len(effects), "wins": wins, "ties": ties, "losses": losses,
            "semantic_evaluations": evaluated,
            "terminal_commits": terminal_commits,
            "negative_transfer_rate": losses / len(effects),
            "mean_relative_change": float(np.mean(effects)),
            "mean_bootstrap_95": study._bootstrap(effects, study.SEED_BASE + (0 if backbone == "DE" else 1)),
            "all_traces_csa_compliant": compliant,
        }

    report = {
        "status": "complete",
        "analysis_role": "post_hoc_commit-policy_diagnostic",
        "source_kind": source_kind,
        "source_protocol_sha256": protocol["protocol_sha256"],
        "source_report": str(source_root / "bbob_confirmation_report.json"),
        "policy": (
            "Evaluate an admitted candidate under one-use authorization, retain it outside "
            "the active population, preserve the native trajectory, and commit only at the "
            "terminal comparison when it beats the native final incumbent."
        ),
        "claim_boundary": (
            "This diagnostic reuses observed BBOB cases and is not confirmatory evidence."
        ),
        "backbones": summaries,
        "records": records,
    }
    report["artifact_sha256"] = study._sha256_json(report)
    study._save_json(report, output_root / "delayed_commit_report.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_root", default=SOURCE_ROOT)
    parser.add_argument("--output_root", default=OUTPUT_ROOT)
    parser.add_argument("--source_kind", choices=("original", "extension"), default="extension")
    args = parser.parse_args()
    report = execute(Path(args.source_root), Path(args.output_root), args.source_kind)
    print(json.dumps(report["backbones"], indent=2))


if __name__ == "__main__":
    main()
