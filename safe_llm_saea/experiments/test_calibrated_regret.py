from __future__ import annotations

import pytest

from safe_llm_saea.backbone_risk import (
    apply_backbone_risk_model,
    fit_backbone_risk_model,
)

from safe_llm_saea.calibrated_regret import (
    apply_regret_linear_model,
    conformal_uncertainty_scale,
    fit_regret_linear_model,
    normalized_regret,
    predicted_regret,
    preregister_abstention_threshold,
)


def test_normalized_regret_sign_for_minimization() -> None:
    assert normalized_regret(8.0, 10.0) == pytest.approx(-0.2)
    assert normalized_regret(12.0, 10.0) == pytest.approx(0.2)


def test_conformal_scale_never_shrinks_gp_uncertainty() -> None:
    records = [
        {"objective": 11.0, "mu": 10.0, "sigma": 0.5},
        {"objective": 9.0, "mu": 10.0, "sigma": 0.5},
    ]
    assert conformal_uncertainty_scale(records, 0.5) >= 1.0


def test_preregistered_threshold_respects_negative_transfer_cap() -> None:
    records = [
        {"event_id": "a", "incumbent": 10.0, "mu": 8.0, "sigma": 0.1, "objective": 8.5},
        {"event_id": "b", "incumbent": 10.0, "mu": 9.0, "sigma": 0.1, "objective": 9.5},
        {"event_id": "c", "incumbent": 10.0, "mu": 9.5, "sigma": 0.1, "objective": 11.0},
    ]
    threshold, audit = preregister_abstention_threshold(
        records, uncertainty_scale=1.0, risk_multiplier=1.0,
        max_negative_transfer_rate=0.0,
    )
    assert audit["selected"]["negative_transfer_rate"] == 0.0
    assert audit["selected"]["admitted_events"] == 2
    assert threshold < 0.0


def test_predicted_regret_scales_sigma() -> None:
    mean, sigma = predicted_regret(9.0, 2.0, 10.0, uncertainty_scale=3.0)
    assert mean == pytest.approx(-0.1)
    assert sigma == pytest.approx(0.6)


def test_regret_linear_model_learns_direction() -> None:
    features = [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]
    targets = [-1.0, 0.0, 1.0]
    model = fit_regret_linear_model(features, targets, ridge=0.01)
    predictions = apply_regret_linear_model(features, model)
    assert predictions[0] < predictions[1] < predictions[2]


def test_backbone_risk_model_returns_ordered_probabilities() -> None:
    features = [[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
    labels = [1, 1, 0, 0]
    model = fit_backbone_risk_model(features, labels, regularization_c=1.0)
    probabilities = apply_backbone_risk_model(features, model)
    assert probabilities[0] > probabilities[-1]
    assert all(0.0 <= value <= 1.0 for value in probabilities)
