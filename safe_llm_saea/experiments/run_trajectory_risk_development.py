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


PROTOCOL_ID = "CSA-TRAJECTORY-RISK-DEVELOPMENT-2026-07-19"
TRAJECTORY_ROOT = "paper/q1/experiment_outputs/trajectory_commit_development_20260719"
EXTENSION_ROOT = "paper/q1/experiment_outputs/directional_bbob_power_extension_20260718"
EXTENSION_DIAGNOSTIC_ROOT = "paper/q1/experiment_outputs/directional_delayed_commit_diagnostic_20260719"
OUTPUT_ROOT = "paper/q1/experiment_outputs/trajectory_risk_development_20260719"
TRAIN_INSTANCES = (1, 2, 3)
CALIBRATION_INSTANCE = 4
HOLDOUT_INSTANCE = 5
RIDGE_ALPHA = 10.0
CALIBRATION_QUANTILE = 0.8


def _transform(rows: list[dict[str, Any]]) -> np.ndarray:
    values = np.asarray([
        [row["immediate_regret"], row["surrogate_mean_regret"], row["surrogate_sigma"]]
        for row in rows
    ], dtype=float)
    return np.sign(values) * np.log1p(np.abs(values))


def _protocol(source_sha256: str) -> dict[str, Any]:
    payload = {
        "protocol_id": PROTOCOL_ID,
        "status": "frozen_before_model_fit",
        "analysis_role": "development_only",
        "source_trajectory_protocol_sha256": source_sha256,
        "split": {
            "fit_instances": list(TRAIN_INSTANCES),
            "calibration_instance": CALIBRATION_INSTANCE,
            "development_holdout_instance": HOLDOUT_INSTANCE,
        },
        "features": ["immediate_regret", "surrogate_mean_regret", "surrogate_sigma"],
        "feature_transform": "signed log1p followed by training-only standardization",
        "model": {"type": "ridge", "alpha": RIDGE_ALPHA, "separate_backbones": True},
        "one_sided_calibration_quantile": CALIBRATION_QUANTILE,
        "eligibility": (
            "The original GP upper-regret selector must admit the candidate and its true "
            "immediate regret must be negative."
        ),
        "commit_rule": "Commit only when the calibrated predicted trajectory-regret upper bound is below zero.",
        "claim_boundary": "The instance-5 holdout is developmental; instances 6--13 are retrospective only.",
    }
    payload["protocol_sha256"] = study._sha256_json(payload)
    return payload


def _selected(rows: list[dict[str, Any]], backbone: str) -> dict[str, Any] | None:
    k = study.RISK_MULTIPLIERS[backbone]
    upper = np.asarray([
        row["surrogate_mean_regret"] + k * row["surrogate_sigma"] for row in rows
    ])
    index = int(np.argmin(upper))
    return rows[index] if upper[index] <= 0.0 else None


def _effects_summary(effects: list[float]) -> dict[str, Any]:
    wins = sum(value < -1e-12 for value in effects)
    losses = sum(value > 1e-12 for value in effects)
    ties = len(effects) - wins - losses
    return {
        "runs": len(effects), "wins": wins, "ties": ties, "losses": losses,
        "negative_transfer_rate": losses / len(effects) if effects else 0.0,
        "mean_relative_change": float(np.mean(effects)) if effects else 0.0,
        "mean_bootstrap_95": study._bootstrap(effects, 20260719) if effects else [0.0, 0.0],
    }


def execute(
    trajectory_root: Path, extension_root: Path,
    extension_diagnostic_root: Path, output_root: Path,
) -> dict[str, Any]:
    trajectory = json.loads(
        (trajectory_root / "trajectory_commit_development_report.json").read_text(encoding="utf-8")
    )
    protocol = _protocol(trajectory["protocol_sha256"])
    protocol_path = output_root / "frozen_protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise RuntimeError("trajectory-risk protocol differs from frozen protocol")
    else:
        study._save_json(protocol, protocol_path)

    extension_source = json.loads(
        (extension_root / "bbob_confirmation_report.json").read_text(encoding="utf-8")
    )
    extension_diagnostic = json.loads(
        (extension_diagnostic_root / "delayed_commit_report.json").read_text(encoding="utf-8")
    )
    extension_effect = {
        (row["backbone"], int(row["fid"]), int(row["instance"])): float(row["relative_change"])
        for row in extension_source["records"]
    }

    report_backbones: dict[str, Any] = {}
    for backbone in ("DE", "PSO"):
        eligible = [
            row for row in trajectory["records"]
            if row["backbone"] == backbone and row["immediate_regret"] < 0.0
        ]
        train = [row for row in eligible if int(row["instance"]) in TRAIN_INSTANCES]
        calibration = [row for row in eligible if int(row["instance"]) == CALIBRATION_INSTANCE]
        model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
        model.fit(_transform(train), np.asarray([row["mean_trajectory_regret"] for row in train]))
        calibration_prediction = model.predict(_transform(calibration))
        calibration_residual = np.asarray([
            row["mean_trajectory_regret"] for row in calibration
        ]) - calibration_prediction
        radius = float(np.quantile(
            calibration_residual, CALIBRATION_QUANTILE, method="higher",
        ))

        holdout_groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for row in trajectory["records"]:
            if row["backbone"] == backbone and int(row["instance"]) == HOLDOUT_INSTANCE:
                holdout_groups.setdefault((int(row["fid"]), int(row["instance"])), []).append(row)
        legacy_effects: list[float] = []
        guarded_effects: list[float] = []
        holdout_records: list[dict[str, Any]] = []
        for (fid, instance), rows in sorted(holdout_groups.items()):
            selected = _selected(rows, backbone)
            legacy = 0.0
            guarded = 0.0
            admitted = False
            upper = None
            if selected is not None and selected["immediate_regret"] < 0.0:
                legacy = float(selected["mean_trajectory_regret"])
                prediction = float(model.predict(_transform([selected]))[0])
                upper = prediction + radius
                admitted = upper < 0.0
                guarded = legacy if admitted else 0.0
            legacy_effects.append(legacy)
            guarded_effects.append(guarded)
            holdout_records.append({
                "fid": fid, "instance": instance, "selected": selected is not None,
                "trajectory_upper": upper, "guard_admitted": admitted,
                "legacy_effect": legacy, "guarded_effect": guarded,
            })

        retrospective_effects: list[float] = []
        retrospective_records: list[dict[str, Any]] = []
        for row in extension_diagnostic["records"]:
            if row["backbone"] != backbone:
                continue
            effect = extension_effect[(backbone, int(row["fid"]), int(row["instance"]))]
            admitted = False
            upper = None
            if row["immediate_regret"] is not None and row["immediate_regret"] < 0.0:
                index = int(row["selection"]["selected_index"])
                scale = max(1.0, abs(float(row["incumbent_value"])))
                feature_row = {
                    "immediate_regret": float(row["immediate_regret"]),
                    "surrogate_mean_regret": float(
                        (row["selection"]["mu"][index] - row["incumbent_value"]) / scale
                    ),
                    "surrogate_sigma": float(row["selection"]["sigma"][index] / scale),
                }
                prediction = float(model.predict(_transform([feature_row]))[0])
                upper = prediction + radius
                admitted = upper < 0.0
            guarded = effect if admitted else 0.0
            retrospective_effects.append(guarded)
            retrospective_records.append({
                "fid": int(row["fid"]), "instance": int(row["instance"]),
                "trajectory_upper": upper, "guard_admitted": admitted,
                "legacy_effect": effect, "guarded_effect": guarded,
            })

        report_backbones[backbone] = {
            "training_candidates": len(train),
            "calibration_candidates": len(calibration),
            "calibration_radius": radius,
            "holdout_legacy": _effects_summary(legacy_effects),
            "holdout_guarded": _effects_summary(guarded_effects),
            "holdout_records": holdout_records,
            "retrospective_extension_guarded": _effects_summary(retrospective_effects),
            "retrospective_extension_records": retrospective_records,
        }

    report = {
        "status": "complete", "protocol_sha256": protocol["protocol_sha256"],
        "backbones": report_backbones,
        "claim_boundary": protocol["claim_boundary"],
    }
    report["artifact_sha256"] = study._sha256_json(report)
    study._save_json(report, output_root / "trajectory_risk_development_report.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory_root", default=TRAJECTORY_ROOT)
    parser.add_argument("--extension_root", default=EXTENSION_ROOT)
    parser.add_argument("--extension_diagnostic_root", default=EXTENSION_DIAGNOSTIC_ROOT)
    parser.add_argument("--output_root", default=OUTPUT_ROOT)
    args = parser.parse_args()
    report = execute(
        Path(args.trajectory_root), Path(args.extension_root),
        Path(args.extension_diagnostic_root), Path(args.output_root),
    )
    print(json.dumps(report["backbones"], indent=2))


if __name__ == "__main__":
    main()
