"""Workflow routing tests."""

from src.workflow.graph import load_and_route_json


def test_load_and_route_json_has_20_results() -> None:
    """Routes all dummy records and keeps item count."""
    routed = load_and_route_json("data/workflow_input_dummy.json")
    assert len(routed) == 20


def test_security_route_precedence() -> None:
    """Routes security-sensitive query to security route."""
    routed = load_and_route_json("data/workflow_input_dummy.json")
    target = next(item for item in routed if item.session_id == "sess_sesac_1005")
    assert target.selected_route.value == "security"

