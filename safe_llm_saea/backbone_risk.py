from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression

from safe_llm_saea.calibrated_regret import EPS, regret_feature_matrix


RISK_FEATURE_NAMES = (
    "mu_rank", "sigma_rank", "lcb_rank", "incumbent_l2", "incumbent_l1",
    "incumbent_linf", "center_l2", "progress", "mu_relative",
    "sigma_relative", "lcb_relative", "ucb_relative",
)


def backbone_risk_feature_matrix(
    mu: Sequence[float],
    sigma: Sequence[float],
    candidates: Sequence[Sequence[float]],
    incumbent_vector: Sequence[float],
    incumbent_value: float,
    bounds: Sequence[Sequence[float]],
    progress: float,
) -> np.ndarray:
    base = regret_feature_matrix(
        mu, sigma, candidates, incumbent_vector, bounds, progress
    )
    mu_array = np.asarray(mu, dtype=float)
    sigma_array = np.maximum(0.0, np.asarray(sigma, dtype=float))
    scale = max(1.0, abs(float(incumbent_value)))
    relative = np.column_stack([
        (mu_array - incumbent_value) / scale,
        sigma_array / scale,
        (mu_array - 0.5 * sigma_array - incumbent_value) / scale,
        (mu_array + sigma_array - incumbent_value) / scale,
    ])
    return np.column_stack([base, np.clip(relative, -20.0, 20.0)])


def fit_backbone_risk_model(
    features: Sequence[Sequence[float]],
    beneficial_labels: Sequence[bool | int],
    regularization_c: float = 1.0,
) -> dict[str, Any]:
    x = np.asarray(features, dtype=float)
    y = np.asarray(beneficial_labels, dtype=int)
    if x.ndim != 2 or len(x) != len(y) or len(np.unique(y)) != 2:
        raise ValueError("risk model requires aligned features with both classes")
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale[scale < EPS] = 1.0
    model = LogisticRegression(
        C=float(regularization_c),
        class_weight="balanced",
        max_iter=5000,
        random_state=20260718,
    )
    model.fit((x - mean) / scale, y)
    return {
        "feature_names": list(RISK_FEATURE_NAMES),
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "coefficients": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "regularization_c": float(regularization_c),
        "training_rows": int(len(y)),
        "training_beneficial_rate": float(np.mean(y)),
    }


def apply_backbone_risk_model(
    features: Sequence[Sequence[float]], model: Mapping[str, Any]
) -> np.ndarray:
    x = np.asarray(features, dtype=float)
    mean = np.asarray(model["feature_mean"], dtype=float)
    scale = np.asarray(model["feature_scale"], dtype=float)
    coefficients = np.asarray(model["coefficients"], dtype=float)
    logits = float(model["intercept"]) + ((x - mean) / scale) @ coefficients
    logits = np.clip(logits, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-logits))
