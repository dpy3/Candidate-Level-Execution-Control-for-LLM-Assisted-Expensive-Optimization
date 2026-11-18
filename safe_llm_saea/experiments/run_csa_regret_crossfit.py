from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.stats import norm

from safe_llm_saea.calibrated_regret import (
    RegretProtocol,
    apply_regret_linear_model,
    conformal_residual_radius,
    fit_regret_linear_model,
    preregister_abstention_threshold,
)
from safe_llm_saea.cec2022 import get_function
from safe_llm_saea.experiments.run_csa_calibrated_regret_study import (
    BOUNDS,
    DIMENSION,
    MAX_CALIBRATION_NEGATIVE_TRANSFER,
    RISK_MULTIPLIER,
    TARGET_COVERAGE,
    _closure_index,
    _performance_row,
    _regret_model_metrics,
    _run_de_replay,
    _save_json,
    _sha256_json,
    run_pso,
)


PROTOCOL_ID = "CSA-REGRET-THREE-FUNCTION-CROSSFIT-2026-07-17"
FOLDS = (
    {"fold": "A", "train_fid": 1, "calibration_fid": 9, "test_fid": 6},
    {"fold": "B", "train_fid": 6, "calibration_fid": 1, "test_fid": 9},
    {"fold": "C", "train_fid": 9, "calibration_fid": 6, "test_fid": 1},
)


def _wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> list[float]:
    if trials <= 0:
        return [float("nan"), float("nan")]
    z = float(norm.ppf(0.5 + confidence / 2.0))
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    radius = z * np.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denominator
    return [float(max(0.0, center - radius)), float(min(1.0, center + radius))]


def _bootstrap_mean_ci(
    values: Sequence[float], seed: int, samples: int = 10000
) -> list[float]:
    array = np.asarray(values, dtype=float)
    if not len(array):
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    means = np.mean(array[indices], axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def _summarize(rows: Sequence[dict[str, Any]], seed: int) -> dict[str, Any]:
    effects = np.asarray([float(row["relative_change"]) for row in rows], dtype=float)
    negative = int(np.sum(effects > 1e-12))
    possible_events = sum(int(row.get("candidate_events", 0)) for row in rows)
    abstained = sum(int(row.get("abstained_events", 0)) for row in rows)
    semantic_evaluations = sum(int(row.get("semantic_evaluations", 0)) for row in rows)
    return {
        "runs": len(rows),
        "wins": int(np.sum(effects < -1e-12)),
        "ties": int(np.sum(np.abs(effects) <= 1e-12)),
        "losses": negative,
        "negative_transfer_rate": float(negative / len(rows)),
        "negative_transfer_rate_wilson_95": _wilson_interval(negative, len(rows)),
        "abstention_rate": (
            float(abstained / possible_events) if possible_events else 0.0
        ),
        "semantic_evaluations": semantic_evaluations,
        "mean_relative_change": float(np.mean(effects)),
        "mean_relative_change_bootstrap_95": _bootstrap_mean_ci(effects, seed),
        "median_relative_change": float(np.median(effects)),
        "budget_efficiency_relative_gain_per_semantic_evaluation": (
            float(-np.sum(effects) / semantic_evaluations)
            if semantic_evaluations else 0.0
        ),
    }


def _fit_fold(
    fold: dict[str, Any], candidate_rows: Sequence[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    training = [row for row in candidate_rows if int(row["fid"]) == fold["train_fid"]]
    calibration = [
        row for row in candidate_rows if int(row["fid"]) == fold["calibration_fid"]
    ]
    test = [row for row in candidate_rows if int(row["fid"]) == fold["test_fid"]]
    model = fit_regret_linear_model(
        [row["features"] for row in training],
        [row["realized_regret"] for row in training],
        ridge=1.0,
    )
    calibration_predictions = apply_regret_linear_model(
        [row["features"] for row in calibration], model
    )
    residual_radius = conformal_residual_radius(
        [row["realized_regret"] for row in calibration],
        calibration_predictions.tolist(),
        TARGET_COVERAGE,
    )
    threshold_records = [
        {
            "event_id": row["event_id"],
            "incumbent": 0.0,
            "objective": float(row["realized_regret"]),
            "mu": float(prediction),
            "sigma": float(residual_radius),
        }
        for row, prediction in zip(calibration, calibration_predictions)
    ]
    threshold, threshold_audit = preregister_abstention_threshold(
        threshold_records,
        uncertainty_scale=1.0,
        risk_multiplier=RISK_MULTIPLIER,
        max_negative_transfer_rate=MAX_CALIBRATION_NEGATIVE_TRANSFER,
    )
    metrics = {
        "training": _regret_model_metrics(
            training, model, residual_radius, TARGET_COVERAGE
        ),
        "calibration": _regret_model_metrics(
            calibration, model, residual_radius, TARGET_COVERAGE
        ),
        "test": _regret_model_metrics(test, model, residual_radius, TARGET_COVERAGE),
    }
    protocol = RegretProtocol(
        protocol_id=f"{PROTOCOL_ID}-FOLD-{fold['fold']}",
        training_cases=(f"CEC2022 F{fold['train_fid']:02d}",),
        calibration_cases=(f"CEC2022 F{fold['calibration_fid']:02d}",),
        test_cases=(f"CEC2022 F{fold['test_fid']:02d}",),
        target_coverage=TARGET_COVERAGE,
        uncertainty_scale=1.0,
        risk_multiplier=RISK_MULTIPLIER,
        abstention_threshold=threshold,
        threshold_objective=(
            "maximize calibration events subject to candidate-level negative-transfer "
            f"rate <= {MAX_CALIBRATION_NEGATIVE_TRANSFER:.2f}"
        ),
    ).as_dict()
    protocol.update({
        "fold": fold["fold"],
        "post_hoc_crossfit": True,
        "regret_model": model,
        "conformal_residual_radius": residual_radius,
        "threshold_audit": threshold_audit,
        "candidate_counts": {
            "training": len(training), "calibration": len(calibration), "test": len(test)
        },
    })
    protocol["protocol_sha256"] = _sha256_json(protocol)
    return protocol, metrics


def _prospective_protocol(
    candidate_rows: Sequence[dict[str, Any]], output_root: Path
) -> dict[str, Any]:
    training = [row for row in candidate_rows if int(row["fid"]) in {1, 6}]
    calibration = [row for row in candidate_rows if int(row["fid"]) == 9]
    model = fit_regret_linear_model(
        [row["features"] for row in training],
        [row["realized_regret"] for row in training],
        ridge=1.0,
    )
    predictions = apply_regret_linear_model(
        [row["features"] for row in calibration], model
    )
    residual_radius = conformal_residual_radius(
        [row["realized_regret"] for row in calibration],
        predictions.tolist(),
        TARGET_COVERAGE,
    )
    threshold_records = [
        {
            "event_id": row["event_id"], "incumbent": 0.0,
            "objective": float(row["realized_regret"]), "mu": float(prediction),
            "sigma": float(residual_radius),
        }
        for row, prediction in zip(calibration, predictions)
    ]
    threshold, threshold_audit = preregister_abstention_threshold(
        threshold_records, 1.0, RISK_MULTIPLIER,
        max_negative_transfer_rate=MAX_CALIBRATION_NEGATIVE_TRANSFER,
    )
    payload = {
        "protocol_id": "CSA-REGRET-PROSPECTIVE-UNSEEN-F02-F10-2026-07-17",
        "status": "frozen_not_executed",
        "registration_boundary": (
            "Local immutable protocol with SHA-256; not a third-party timestamped registry."
        ),
        "training_cases": ["CEC2022 F01", "CEC2022 F06"],
        "calibration_cases": ["CEC2022 F09"],
        "unseen_test_cases": ["CEC2022 F02", "CEC2022 F10"],
        "runs_per_case": 5,
        "backbones": ["DE", "PSO"],
        "budgets": {"DE": 240, "PSO": 200},
        "seed_rule": "20260717 + fid * 100 + run_id",
        "proposal_capture": {
            "required": True,
            "fresh_live_calls": True,
            "model_id": "gpt-5.4",
            "calls_per_run": 2,
            "reuse_across_compared_policies": True,
            "no_test_label_access_before_protocol_execution": True,
        },
        "regret_model": model,
        "target_coverage": TARGET_COVERAGE,
        "conformal_residual_radius": residual_radius,
        "risk_multiplier": RISK_MULTIPLIER,
        "abstention_threshold": threshold,
        "threshold_audit": threshold_audit,
        "required_metrics": [
            "coverage_error", "normalized_rmse", "negative_transfer_rate",
            "abstention_rate", "semantic_evaluations", "budget_efficiency",
            "paired_final_performance", "runtime_authorization_compliance",
        ],
        "stopping_rule": "execute all declared runs; do not stop after favorable outcomes",
        "claim_rule": (
            "No general acquisition claim unless both unseen functions and both backbones "
            "show non-degenerate admission and uncertainty coverage is reported."
        ),
    }
    payload["protocol_sha256"] = _sha256_json(payload)
    _save_json(payload, output_root / "prospective_unseen_f02_f10_protocol.json")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate_dataset",
        default=(
            "paper/q1/experiment_outputs/csa_calibrated_regret_holdout_20260716/"
            "candidate_regret_dataset.json"
        ),
    )
    parser.add_argument(
        "--closure_root",
        default="paper/q1/experiment_outputs/csa_representative_closure_v2_llmtoken_20260716",
    )
    parser.add_argument(
        "--proposal_cache",
        default="paper/q1/experiment_outputs/csa_cmaes_portability/proposal_cache",
    )
    parser.add_argument(
        "--output_root",
        default="paper/q1/experiment_outputs/csa_regret_crossfit_20260717",
    )
    parser.add_argument("--de_runs", type=int, default=3)
    parser.add_argument("--pso_runs", type=int, default=5)
    parser.add_argument("--de_max_fes", type=int, default=240)
    parser.add_argument("--pso_max_fes", type=int, default=200)
    parser.add_argument("--seed_base", type=int, default=20260717)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    candidate_rows = json.loads(
        Path(args.candidate_dataset).read_text(encoding="utf-8")
    )["records"]
    closure_root = Path(args.closure_root)
    closure = _closure_index(closure_root)
    proposal_cache = Path(args.proposal_cache)

    fold_reports = []
    all_rows: dict[str, list[dict[str, Any]]] = {
        "DE_rank_top": [], "DE_calibrated_regret": [],
        "PSO_rank_top": [], "PSO_calibrated_regret": [],
    }
    for fold in FOLDS:
        protocol, calibration_metrics = _fit_fold(fold, candidate_rows)
        protocol_path = output_root / f"fold_{fold['fold'].lower()}_protocol.json"
        _save_json(protocol, protocol_path)
        frozen = json.loads(protocol_path.read_text(encoding="utf-8"))
        expected_hash = _sha256_json({
            key: value for key, value in frozen.items() if key != "protocol_sha256"
        })
        if expected_hash != frozen["protocol_sha256"]:
            raise RuntimeError(f"protocol hash mismatch for fold {fold['fold']}")

        test_fid = int(fold["test_fid"])
        case = f"F{test_fid:02d}-10D"
        fold_rows = {name: [] for name in all_rows}
        de_runs = []
        for run_id in range(args.de_runs):
            seed = int(closure[(case, run_id, "fail_closed")]["seed"])
            rank_top = _run_de_replay(
                closure_root, test_fid, run_id, seed, "rank_top", args.de_max_fes
            )
            calibrated = _run_de_replay(
                closure_root, test_fid, run_id, seed, "calibrated_regret",
                args.de_max_fes,
                uncertainty_scale=1.0,
                risk_multiplier=float(frozen["risk_multiplier"]),
                abstention_threshold=float(frozen["abstention_threshold"]),
                regret_model=frozen["regret_model"],
                residual_radius=float(frozen["conformal_residual_radius"]),
            )
            strict_value = float(closure[(case, run_id, "strict_zero")]["objective"])
            rank_row = _performance_row("DE", "rank_top", run_id, rank_top, strict_value)
            calibrated_row = _performance_row(
                "DE", "calibrated_regret", run_id, calibrated, strict_value
            )
            fold_rows["DE_rank_top"].append(rank_row)
            fold_rows["DE_calibrated_regret"].append(calibrated_row)
            de_runs.append({"run_id": run_id, "rank_top": rank_top, "calibrated": calibrated})

        objective = get_function(test_fid, DIMENSION)
        pso_runs = []
        for run_id in range(args.pso_runs):
            cache = json.loads(
                (proposal_cache / f"cec_f{test_fid:02d}_10d_r{run_id:02d}.json").read_text(
                    encoding="utf-8"
                )
            )
            proposals = cache["proposal_vectors"]
            seed = args.seed_base + test_fid * 100 + run_id
            native = run_pso(
                objective, seed, args.pso_max_fes, [], "native", 1.0, 1.0, 0.0
            )
            strict = run_pso(
                objective, seed, args.pso_max_fes, proposals, "strict-zero", 1.0, 1.0, 0.0
            )
            rank_top = run_pso(
                objective, seed, args.pso_max_fes, proposals, "rank_top", 1.0, 1.0, 0.0
            )
            calibrated = run_pso(
                objective, seed, args.pso_max_fes, proposals, "calibrated_regret",
                1.0,
                float(frozen["risk_multiplier"]),
                float(frozen["abstention_threshold"]),
                regret_model=frozen["regret_model"],
                residual_radius=float(frozen["conformal_residual_radius"]),
            )
            if native["trajectory_sha256"] != strict["trajectory_sha256"]:
                raise RuntimeError(
                    f"PSO strict-zero mismatch fold {fold['fold']} run {run_id}"
                )
            if not rank_top["compliance"]["csa_compliant"]:
                raise RuntimeError("PSO rank-top trace is not CSA compliant")
            if not calibrated["compliance"]["csa_compliant"]:
                raise RuntimeError("PSO calibrated trace is not CSA compliant")
            fold_rows["PSO_rank_top"].append(
                _performance_row("PSO", "rank_top", run_id, rank_top, strict["best_fitness"])
            )
            fold_rows["PSO_calibrated_regret"].append(
                _performance_row(
                    "PSO", "calibrated_regret", run_id, calibrated, strict["best_fitness"]
                )
            )
            pso_runs.append({
                "run_id": run_id, "native": native, "strict_zero": strict,
                "rank_top": rank_top, "calibrated": calibrated,
            })

        for name, rows in fold_rows.items():
            for row in rows:
                row["fold"] = fold["fold"]
                row["test_fid"] = test_fid
            all_rows[name].extend(rows)
        fold_report = {
            "fold": fold,
            "protocol_sha256": frozen["protocol_sha256"],
            "calibration_metrics": calibration_metrics,
            "performance": {
                name: _summarize(rows, args.seed_base + index)
                for index, (name, rows) in enumerate(fold_rows.items())
            },
        }
        fold_reports.append(fold_report)
        _save_json({"de_runs": de_runs, "pso_runs": pso_runs}, output_root / f"fold_{fold['fold'].lower()}_runs.json")

    aggregate = {
        name: _summarize(rows, args.seed_base + 100 + index)
        for index, (name, rows) in enumerate(all_rows.items())
    }
    prospective = _prospective_protocol(candidate_rows, output_root)
    report = {
        "protocol_id": PROTOCOL_ID,
        "status": "post_hoc_crossfit_complete",
        "interpretation_boundary": (
            "Each fold excludes its test function from fitting and threshold calibration, "
            "but the three-function corpus and analysis plan are post hoc."
        ),
        "folds": fold_reports,
        "aggregate": aggregate,
        "rows": all_rows,
        "prospective_protocol": {
            "path": str(output_root / "prospective_unseen_f02_f10_protocol.json"),
            "sha256": prospective["protocol_sha256"],
            "status": prospective["status"],
        },
    }
    _save_json(report, output_root / "crossfit_report.json")
    print(json.dumps({
        "aggregate": aggregate,
        "fold_test_coverage": {
            item["fold"]["fold"]: item["calibration_metrics"]["test"]
            for item in fold_reports
        },
        "prospective_protocol": report["prospective_protocol"],
    }, indent=2))


if __name__ == "__main__":
    main()
