from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from safe_llm_saea.experiments import run_directional_bbob_confirmation as study
from safe_llm_saea.experiments import run_directional_bbob_power_extension as extension


PROTOCOL_ID = "CSA-GEOMETRY-TRAJECTORY-RISK-DEVELOPMENT-2026-07-19"
ORIGINAL_TRAJECTORY_ROOT = "paper/q1/experiment_outputs/trajectory_commit_development_20260719"
EXTENSION_TRAJECTORY_ROOT = "paper/q1/experiment_outputs/trajectory_commit_extension_development_20260719"
ORIGINAL_SOURCE_ROOT = "paper/q1/experiment_outputs/directional_bbob_confirmation_20260718"
EXTENSION_SOURCE_ROOT = "paper/q1/experiment_outputs/directional_bbob_power_extension_20260718"
OUTPUT_ROOT = "paper/q1/experiment_outputs/geometry_trajectory_risk_development_20260719"
TRAIN_INSTANCES = tuple(range(1, 10))
CALIBRATION_INSTANCES = (10, 11)
HOLDOUT_INSTANCES = (12, 13)
RIDGE_ALPHA = 10.0
CALIBRATION_QUANTILE = 0.8
ORIGINAL_SEED_BASE = study.SEED_BASE
EXTENSION_SEED_BASE = extension.SEED_BASE
FEATURES = (
    "immediate_regret", "surrogate_mean_regret", "surrogate_sigma", "step_scale",
    "candidate_to_centroid", "candidate_to_nearest", "candidate_to_replaced",
    "diversity_ratio_after_commit", "centroid_alignment", "velocity_alignment",
)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 0.0 if denominator <= 1e-15 else float(np.dot(left, right) / denominator)


def _diversity(points: np.ndarray) -> float:
    centroid = np.mean(points, axis=0)
    return float(np.sqrt(np.mean(np.sum((points - centroid) ** 2, axis=1))))


def _geometry(
    backbone: str, fid: int, instance: int, candidates: list[np.ndarray], seed_base: int,
) -> list[dict[str, float]]:
    seed = seed_base + fid * 100 + instance + (0 if backbone == "DE" else 50000)
    if backbone == "DE":
        optimizer, _, _ = study._de_runtime(fid, instance, seed)
        points = np.asarray(optimizer.population, dtype=float)
        incumbent = points[int(optimizer.best_idx)]
        replaced_index = int(np.argmax(optimizer.fitness))
        mean_velocity = np.zeros(study.DIMENSION)
    else:
        state = study._pso_runtime(fid, instance, seed)
        points = np.asarray(state["personal_best"], dtype=float)
        incumbent = np.asarray(state["incumbent"], dtype=float)
        replaced_index = int(np.argmax(state["personal_values"]))
        mean_velocity = np.mean(np.asarray(state["velocities"], dtype=float), axis=0)
    normalized = (points + 5.0) / 10.0
    incumbent_normalized = (incumbent + 5.0) / 10.0
    centroid = np.mean(normalized, axis=0)
    before = max(_diversity(normalized), 1e-12)
    root_dimension = np.sqrt(study.DIMENSION)
    output = []
    for candidate in candidates:
        candidate_normalized = (candidate + 5.0) / 10.0
        delta = candidate_normalized - incumbent_normalized
        after_points = normalized.copy()
        after_points[replaced_index] = candidate_normalized
        output.append({
            "candidate_to_centroid": float(np.linalg.norm(candidate_normalized - centroid) / root_dimension),
            "candidate_to_nearest": float(np.min(np.linalg.norm(normalized - candidate_normalized, axis=1)) / root_dimension),
            "candidate_to_replaced": float(np.linalg.norm(candidate_normalized - normalized[replaced_index]) / root_dimension),
            "diversity_ratio_after_commit": float(_diversity(after_points) / before),
            "centroid_alignment": _cosine(delta, centroid - incumbent_normalized),
            "velocity_alignment": _cosine(delta, mean_velocity / 10.0),
        })
    return output


def _protocol(source_hashes: list[str]) -> dict[str, Any]:
    payload = {
        "protocol_id": PROTOCOL_ID, "status": "frozen_before_model_fit",
        "analysis_role": "development_only",
        "source_trajectory_protocol_sha256": source_hashes,
        "split": {
            "fit_instances": list(TRAIN_INSTANCES),
            "calibration_instances": list(CALIBRATION_INSTANCES),
            "development_holdout_instances": list(HOLDOUT_INSTANCES),
            "reserved_future_confirmation_instances": list(range(14, 22)),
        },
        "features": list(FEATURES),
        "model": {"type": "ridge", "alpha": RIDGE_ALPHA, "separate_backbones": True},
        "feature_transform": "signed log1p followed by training-only standardization",
        "one_sided_calibration_quantile": CALIBRATION_QUANTILE,
        "eligibility": "original GP admission plus true immediate incumbent improvement",
        "commit_rule": "commit only when calibrated trajectory-regret upper bound is below zero",
        "holdout_gate": {
            "nonzero_commits": True, "wins_exceed_losses": True,
            "negative_transfer_rate_at_most_0_25": True,
        },
        "claim_boundary": "No final claim is allowed before untouched instances 14--21 are executed.",
    }
    payload["protocol_sha256"] = study._sha256_json(payload)
    return payload


def _transform(rows: list[dict[str, Any]]) -> np.ndarray:
    values = np.asarray([[row[name] for name in FEATURES] for row in rows], dtype=float)
    return np.sign(values) * np.log1p(np.abs(values))


def _summary(effects: list[float], seed: int) -> dict[str, Any]:
    wins = sum(value < -1e-12 for value in effects)
    losses = sum(value > 1e-12 for value in effects)
    ties = len(effects) - wins - losses
    return {
        "runs": len(effects), "wins": wins, "ties": ties, "losses": losses,
        "negative_transfer_rate": losses / len(effects),
        "mean_relative_change": float(np.mean(effects)),
        "mean_bootstrap_95": study._bootstrap(effects, seed),
    }


def execute(
    original_trajectory_root: Path, extension_trajectory_root: Path,
    original_source_root: Path, extension_source_root: Path, output_root: Path,
) -> dict[str, Any]:
    original = json.loads((original_trajectory_root / "trajectory_commit_development_report.json").read_text(encoding="utf-8"))
    extended = json.loads((extension_trajectory_root / "trajectory_commit_development_report.json").read_text(encoding="utf-8"))
    protocol = _protocol([original["protocol_sha256"], extended["protocol_sha256"]])
    protocol_path = output_root / "frozen_protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise RuntimeError("geometry trajectory-risk protocol differs from frozen protocol")
    else:
        study._save_json(protocol, protocol_path)

    original_source_protocol = study._load_or_freeze(original_source_root / "frozen_protocol.json")
    original_records = original["records"]
    extension._configure()
    extension_source_protocol = study._load_or_freeze(extension_source_root / "frozen_protocol.json")
    all_records: list[dict[str, Any]] = []
    for records, source_root, source_protocol, seed_base in (
        (original_records, original_source_root, original_source_protocol, ORIGINAL_SEED_BASE),
        (extended["records"], extension_source_root, extension_source_protocol, EXTENSION_SEED_BASE),
    ):
        grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
        for row in records:
            grouped.setdefault((row["backbone"], int(row["fid"]), int(row["instance"])), []).append(row)
        for (backbone, fid, instance), rows in grouped.items():
            cache = study._verify_cache(
                study._cache_path(source_root, backbone, fid, instance),
                source_protocol, backbone, fid, instance,
            )
            candidates = [np.asarray(vector, dtype=float) for vector in cache["candidate_vectors"]]
            geometry = _geometry(backbone, fid, instance, candidates, seed_base)
            for row in rows:
                enriched = dict(row)
                enriched.update(geometry[int(row["candidate_index"])])
                all_records.append(enriched)

    backbones: dict[str, Any] = {}
    for backbone in ("DE", "PSO"):
        eligible = [row for row in all_records if row["backbone"] == backbone and row["immediate_regret"] < 0.0]
        train = [row for row in eligible if int(row["instance"]) in TRAIN_INSTANCES]
        calibration = [row for row in eligible if int(row["instance"]) in CALIBRATION_INSTANCES]
        model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
        model.fit(_transform(train), np.asarray([row["mean_trajectory_regret"] for row in train]))
        residual = np.asarray([row["mean_trajectory_regret"] for row in calibration]) - model.predict(_transform(calibration))
        radius = float(np.quantile(residual, CALIBRATION_QUANTILE, method="higher"))

        groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for row in all_records:
            if row["backbone"] == backbone and int(row["instance"]) in HOLDOUT_INSTANCES:
                groups.setdefault((int(row["fid"]), int(row["instance"])), []).append(row)
        baseline_effects: list[float] = []
        guarded_effects: list[float] = []
        decisions: list[dict[str, Any]] = []
        for (fid, instance), rows in sorted(groups.items()):
            k = study.RISK_MULTIPLIERS[backbone]
            gp_upper = np.asarray([
                row["surrogate_mean_regret"] + k * row["surrogate_sigma"] for row in rows
            ])
            selected = rows[int(np.argmin(gp_upper))] if float(np.min(gp_upper)) <= 0.0 else None
            baseline = 0.0
            guarded = 0.0
            trajectory_upper = None
            admitted = False
            if selected is not None and selected["immediate_regret"] < 0.0:
                baseline = float(selected["mean_trajectory_regret"])
                trajectory_upper = float(model.predict(_transform([selected]))[0] + radius)
                admitted = trajectory_upper < 0.0
                guarded = baseline if admitted else 0.0
            baseline_effects.append(baseline)
            guarded_effects.append(guarded)
            decisions.append({
                "fid": fid, "instance": instance, "selected": selected is not None,
                "trajectory_upper": trajectory_upper, "admitted": admitted,
                "baseline_effect": baseline, "guarded_effect": guarded,
            })
        baseline_summary = _summary(baseline_effects, 20260719)
        guarded_summary = _summary(guarded_effects, 20260720)
        gates = {
            "nonzero_commits": any(row["admitted"] for row in decisions),
            "wins_exceed_losses": guarded_summary["wins"] > guarded_summary["losses"],
            "negative_transfer_rate_at_most_0_25": guarded_summary["negative_transfer_rate"] <= 0.25,
        }
        backbones[backbone] = {
            "training_candidates": len(train), "calibration_candidates": len(calibration),
            "calibration_radius": radius, "holdout_baseline": baseline_summary,
            "holdout_guarded": guarded_summary, "holdout_gates": gates,
            "holdout_gate_passed": all(gates.values()), "decisions": decisions,
        }
    report = {
        "status": "complete", "protocol_sha256": protocol["protocol_sha256"],
        "backbones": backbones,
        "future_confirmation_eligible": any(value["holdout_gate_passed"] for value in backbones.values()),
    }
    report["artifact_sha256"] = study._sha256_json(report)
    study._save_json(report, output_root / "geometry_trajectory_risk_report.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_trajectory_root", default=ORIGINAL_TRAJECTORY_ROOT)
    parser.add_argument("--extension_trajectory_root", default=EXTENSION_TRAJECTORY_ROOT)
    parser.add_argument("--original_source_root", default=ORIGINAL_SOURCE_ROOT)
    parser.add_argument("--extension_source_root", default=EXTENSION_SOURCE_ROOT)
    parser.add_argument("--output_root", default=OUTPUT_ROOT)
    args = parser.parse_args()
    report = execute(
        Path(args.original_trajectory_root), Path(args.extension_trajectory_root),
        Path(args.original_source_root), Path(args.extension_source_root), Path(args.output_root),
    )
    print(json.dumps({
        "backbones": report["backbones"],
        "future_confirmation_eligible": report["future_confirmation_eligible"],
    }, indent=2))


if __name__ == "__main__":
    main()
