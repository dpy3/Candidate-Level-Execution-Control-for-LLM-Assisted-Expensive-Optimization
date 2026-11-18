from safe_llm_saea.experiments import run_geometry_trajectory_risk_development as geometry
from safe_llm_saea.experiments import run_trajectory_commit_development as trajectory
from safe_llm_saea.experiments import run_trajectory_risk_development as compact
from safe_llm_saea.experiments import run_directional_bbob_confirmation as study


def test_trajectory_capture_protocol_is_paired_and_development_only() -> None:
    protocol = trajectory._protocol("source-checksum")

    assert protocol["analysis_role"] == "development_only"
    assert protocol["continuations_per_candidate"] == 10
    assert "same continuation seed" in protocol["paired_rule"]
    assert protocol["protocol_sha256"] == study._sha256_json({
        key: value for key, value in protocol.items() if key != "protocol_sha256"
    })


def test_compact_risk_protocol_keeps_extension_retrospective() -> None:
    protocol = compact._protocol("trajectory-checksum")

    assert protocol["split"]["fit_instances"] == [1, 2, 3]
    assert protocol["split"]["calibration_instance"] == 4
    assert "retrospective only" in protocol["claim_boundary"]


def test_geometry_protocol_reserves_future_instances() -> None:
    protocol = geometry._protocol(["first", "second"])

    assert protocol["split"]["development_holdout_instances"] == [12, 13]
    assert protocol["split"]["reserved_future_confirmation_instances"] == list(range(14, 22))
    assert protocol["holdout_gate"]["nonzero_commits"] is True
    assert protocol["protocol_sha256"] == study._sha256_json({
        key: value for key, value in protocol.items() if key != "protocol_sha256"
    })
