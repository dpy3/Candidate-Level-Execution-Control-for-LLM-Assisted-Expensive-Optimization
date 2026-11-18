from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


EPS = 1e-12


def _fractional_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(values, kind="mergesort"), kind="mergesort")
    return order.astype(float) / max(1, len(values) - 1)


def regret_feature_matrix(
    mu: Sequence[float],
    sigma: Sequence[float],
    candidates: Sequence[Sequence[float]],
    incumbent_vector: Sequence[float],
    bounds: Sequence[Sequence[float]],
    progress: float,
) -> np.ndarray:
    """Function-scale-invariant pool ranks and incumbent-relative geometry."""
    mu_array = np.asarray(mu, dtype=float)
    sigma_array = np.asarray(sigma, dtype=float)
    candidate_array = np.asarray(candidates, dtype=float)
    incumbent = np.asarray(incumbent_vector, dtype=float)
    lower = np.asarray([item[0] for item in bounds], dtype=float)
    span = np.maximum(EPS, np.asarray([item[1] - item[0] for item in bounds], dtype=float))
    center = lower + 0.5 * span
    normalized_delta = (candidate_array - incumbent) / span
    normalized_center = (candidate_array - center) / span
    lcb = mu_array - 0.5 * sigma_array
    return np.column_stack([
        _fractional_rank(mu_array),
        _fractional_rank(sigma_array),
        _fractional_rank(lcb),
        np.sqrt(np.mean(normalized_delta ** 2, axis=1)),
        np.mean(np.abs(normalized_delta), axis=1),
        np.max(np.abs(normalized_delta), axis=1),
        np.sqrt(np.mean(normalized_center ** 2, axis=1)),
        np.full(len(candidate_array), float(progress), dtype=float),
    ])


def fit_regret_linear_model(
    features: Sequence[Sequence[float]],
    targets: Sequence[float],
    ridge: float = 1.0,
) -> dict[str, Any]:
    x = np.asarray(features, dtype=float)
    y = np.asarray(targets, dtype=float)
    if x.ndim != 2 or len(x) != len(y) or not len(y):
        raise ValueError("features and targets must be non-empty aligned arrays")
    feature_mean = np.mean(x, axis=0)
    feature_scale = np.std(x, axis=0)
    feature_scale[feature_scale < EPS] = 1.0
    z = (x - feature_mean) / feature_scale
    design = np.column_stack([np.ones(len(z)), z])
    penalty = np.eye(design.shape[1], dtype=float) * float(ridge)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    predictions = design @ coefficients
    return {
        "feature_names": [
            "mu_rank", "sigma_rank", "lcb_rank", "incumbent_l2",
            "incumbent_l1", "incumbent_linf", "center_l2", "progress",
        ],
        "feature_mean": feature_mean.tolist(),
        "feature_scale": feature_scale.tolist(),
        "coefficients": coefficients.tolist(),
        "ridge": float(ridge),
        "training_rmse": float(np.sqrt(np.mean((predictions - y) ** 2))),
    }


def apply_regret_linear_model(
    features: Sequence[Sequence[float]],
    model: Mapping[str, Any],
) -> np.ndarray:
    x = np.asarray(features, dtype=float)
    mean = np.asarray(model["feature_mean"], dtype=float)
    scale = np.asarray(model["feature_scale"], dtype=float)
    coefficients = np.asarray(model["coefficients"], dtype=float)
    design = np.column_stack([np.ones(len(x)), (x - mean) / scale])
    if design.shape[1] != len(coefficients):
        raise ValueError("regret model coefficient dimension mismatch")
    return design @ coefficients


def conformal_residual_radius(
    realized: Sequence[float],
    predicted: Sequence[float],
    target_coverage: float = 0.90,
) -> float:
    if len(realized) != len(predicted) or not realized:
        raise ValueError("realized and predicted regrets must be non-empty and aligned")
    residuals = np.sort(np.abs(np.asarray(realized, dtype=float) - np.asarray(predicted, dtype=float)))
    rank = min(len(residuals), max(1, ceil((len(residuals) + 1) * target_coverage)))
    return float(residuals[rank - 1])


def normalized_regret(objective: float, incumbent: float) -> float:
    """Positive values are harmful for a minimization problem."""
    return float((float(objective) - float(incumbent)) / max(1.0, abs(float(incumbent))))


def predicted_regret(
    mu: float,
    sigma: float,
    incumbent: float,
    uncertainty_scale: float,
) -> tuple[float, float]:
    scale = max(1.0, abs(float(incumbent)))
    return (
        float((float(mu) - float(incumbent)) / scale),
        float(max(0.0, float(sigma)) * float(uncertainty_scale) / scale),
    )


@dataclass(frozen=True)
class RegretProtocol:
    protocol_id: str
    training_cases: tuple[str, ...]
    calibration_cases: tuple[str, ...]
    test_cases: tuple[str, ...]
    target_coverage: float
    uncertainty_scale: float
    risk_multiplier: float
    abstention_threshold: float
    threshold_objective: str
    frozen_before_test: bool = True

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["training_cases"] = list(self.training_cases)
        payload["calibration_cases"] = list(self.calibration_cases)
        payload["test_cases"] = list(self.test_cases)
        return payload


def conformal_uncertainty_scale(
    records: Sequence[Mapping[str, Any]],
    target_coverage: float = 0.90,
) -> float:
    """Split-conformal multiplier for absolute GP residuals.

    Each record must contain ``objective``, ``mu``, and ``sigma``. The finite-sample
    quantile uses ceil((n + 1) * coverage) / n and never shrinks uncertainty below 1.
    """
    if not 0.0 < target_coverage < 1.0:
        raise ValueError("target_coverage must be between zero and one")
    ratios = []
    for record in records:
        sigma = max(EPS, abs(float(record["sigma"])))
        ratios.append(abs(float(record["objective"]) - float(record["mu"])) / sigma)
    if not ratios:
        raise ValueError("at least one calibration record is required")
    values = np.sort(np.asarray(ratios, dtype=float))
    rank = min(len(values), max(1, ceil((len(values) + 1) * target_coverage)))
    return float(max(1.0, values[rank - 1]))


def calibration_metrics(
    records: Sequence[Mapping[str, Any]],
    uncertainty_scale: float,
    risk_multiplier: float,
    nominal_coverage: float = 0.90,
) -> dict[str, float | int]:
    if not records:
        return {
            "n": 0,
            "empirical_coverage": float("nan"),
            "coverage_error": float("nan"),
            "mean_normalized_interval_half_width": float("nan"),
            "normalized_rmse": float("nan"),
        }
    covered = []
    widths = []
    squared_errors = []
    for record in records:
        incumbent = float(record["incumbent"])
        realized = normalized_regret(float(record["objective"]), incumbent)
        mean_regret, sigma_regret = predicted_regret(
            float(record["mu"]),
            float(record["sigma"]),
            incumbent,
            uncertainty_scale,
        )
        half_width = float(risk_multiplier) * sigma_regret
        covered.append(abs(realized - mean_regret) <= half_width + EPS)
        widths.append(half_width)
        squared_errors.append((realized - mean_regret) ** 2)
    empirical = float(np.mean(covered))
    return {
        "n": len(records),
        "empirical_coverage": empirical,
        "coverage_error": float(abs(float(nominal_coverage) - empirical)),
        "mean_normalized_interval_half_width": float(np.mean(widths)),
        "normalized_rmse": float(np.sqrt(np.mean(squared_errors))),
        "nominal_reference": float(nominal_coverage),
    }


def _event_choices(
    records: Sequence[Mapping[str, Any]],
    uncertainty_scale: float,
    risk_multiplier: float,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["event_id"]), []).append(record)
    choices = []
    for event_id, candidates in grouped.items():
        scored = []
        for record in candidates:
            mean_regret, sigma_regret = predicted_regret(
                float(record["mu"]),
                float(record["sigma"]),
                float(record["incumbent"]),
                uncertainty_scale,
            )
            scored.append((mean_regret + risk_multiplier * sigma_regret, record))
        upper, selected = min(scored, key=lambda item: item[0])
        choices.append({
            "event_id": event_id,
            "risk_upper": float(upper),
            "realized_regret": normalized_regret(
                float(selected["objective"]), float(selected["incumbent"])
            ),
        })
    return choices


def preregister_abstention_threshold(
    records: Sequence[Mapping[str, Any]],
    uncertainty_scale: float,
    risk_multiplier: float,
    max_negative_transfer_rate: float = 0.10,
) -> tuple[float, dict[str, Any]]:
    """Freeze a threshold using calibration events only.

    The selected threshold maximizes admitted events subject to the requested harmful-
    admission rate. Ties prefer lower mean realized regret and then the stricter threshold.
    """
    choices = _event_choices(records, uncertainty_scale, risk_multiplier)
    if not choices:
        raise ValueError("at least one calibration event is required")
    candidates = sorted({float(item["risk_upper"]) for item in choices} | {0.0})
    feasible = []
    all_rows = []
    for threshold in candidates:
        admitted = [item for item in choices if item["risk_upper"] <= threshold + EPS]
        harmful = [item for item in admitted if item["realized_regret"] > EPS]
        rate = float(len(harmful) / len(admitted)) if admitted else 0.0
        mean_regret = (
            float(np.mean([item["realized_regret"] for item in admitted]))
            if admitted else None
        )
        row = {
            "threshold": float(threshold),
            "admitted_events": len(admitted),
            "total_events": len(choices),
            "negative_transfer_rate": rate,
            "mean_realized_regret": mean_regret,
        }
        all_rows.append(row)
        if admitted and rate <= max_negative_transfer_rate + EPS:
            feasible.append(row)
    if feasible:
        selected = min(
            feasible,
            key=lambda row: (
                -int(row["admitted_events"]),
                float(row["mean_realized_regret"]),
                float(row["threshold"]),
            ),
        )
    else:
        selected = {
            "threshold": float(min(candidates)),
            "admitted_events": 0,
            "total_events": len(choices),
            "negative_transfer_rate": 0.0,
            "mean_realized_regret": None,
        }
    return float(selected["threshold"]), {
        "selection_rule": "maximize admitted calibration events subject to negative-transfer-rate cap",
        "max_negative_transfer_rate": float(max_negative_transfer_rate),
        "selected": selected,
        "grid": all_rows,
    }


def admission_metrics(rows: Iterable[Mapping[str, Any]]) -> dict[str, float | int]:
    rows = list(rows)
    semantic_evaluations = sum(int(row.get("semantic_evaluations", 0)) for row in rows)
    possible_events = sum(int(row.get("candidate_events", 0)) for row in rows)
    abstained_events = sum(int(row.get("abstained_events", 0)) for row in rows)
    negative_transfers = sum(bool(row.get("negative_transfer", False)) for row in rows)
    relative_changes = [float(row["relative_change"]) for row in rows]
    return {
        "runs": len(rows),
        "negative_transfer_rate": float(negative_transfers / len(rows)) if rows else float("nan"),
        "abstention_rate": float(abstained_events / possible_events) if possible_events else float("nan"),
        "semantic_evaluations": semantic_evaluations,
        "budget_efficiency_mean_relative_gain_per_semantic_evaluation": (
            float(-sum(relative_changes) / semantic_evaluations)
            if semantic_evaluations else 0.0
        ),
        "mean_relative_change": float(np.mean(relative_changes)) if rows else float("nan"),
    }
