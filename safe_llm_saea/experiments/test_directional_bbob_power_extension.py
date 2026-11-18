from safe_llm_saea.experiments import run_directional_bbob_power_extension as extension
from safe_llm_saea.experiments import run_directional_bbob_confirmation as study


def test_power_extension_protocol_is_frozen_on_unseen_instances() -> None:
    extension._configure()
    protocol = extension._extension_protocol()

    assert protocol["benchmark"]["instances"] == list(range(6, 14))
    assert protocol["success_gate"]["minimum_paired_cases_per_backbone"] == 30
    assert "all 64 fresh backbone-case pairs" in protocol["stopping_rule"]
    assert protocol["protocol_sha256"] == study._sha256_json({
        key: value for key, value in protocol.items() if key != "protocol_sha256"
    })


def test_power_extension_retains_development_risk_multipliers() -> None:
    extension._configure()
    protocol = extension._extension_protocol()

    assert protocol["backbones"]["DE"]["risk_multiplier"] == 1.0
    assert protocol["backbones"]["PSO"]["risk_multiplier"] == 0.5
    assert protocol["admission"]["commit_rule"] == (
        "population commit only after true incumbent improvement"
    )
