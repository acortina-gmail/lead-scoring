"""Unit tests for the schema/routing/gate contract in leadscoring.config.

GCP-free: these exercise pure functions and constants only, so they run in CI
without credentials.
"""

from leadscoring import config


def test_model_uri_paths():
    # The candidate/live stage and segment name must land in the exact GCS layout
    # serving and the promotion gate both depend on.
    assert config.model_uri("landing", "candidate").endswith(
        "/candidate/lead_scoring_landing.joblib"
    )
    assert config.model_uri("main", "live").endswith("/live/lead_scoring_main.joblib")
    # Defaults to the live stage (what serving loads).
    assert config.model_uri("main") == config.model_uri("main", "live")


def test_route_segment_explicit_segmento_wins():
    assert config.route_segment({"segmento": "landing"}) == "landing"
    assert config.route_segment({"segmento": "  MAIN "}) == "main"


def test_route_segment_unknown_segmento_falls_back_to_main():
    assert config.route_segment({"segmento": "weird"}) == "main"


def test_route_segment_derives_from_form_name():
    assert config.route_segment({"form_name": "unbounce_spring_campaign"}) == "landing"
    assert config.route_segment({"form_name": "web_contact"}) == "main"


def test_route_segment_empty_defaults_to_main():
    assert config.route_segment({}) == "main"
    assert config.route_segment({"segmento": "  ", "form_name": ""}) == "main"


def test_promotion_gate_contract():
    # The soft promotion gate's thresholds are a non-negotiable contract; lock them in.
    assert config.PROMOTION["metric"] == "lift_top"
    assert config.PROMOTION["min_abs"] == 1.0
    assert config.PROMOTION["max_regression"] == 0.15
