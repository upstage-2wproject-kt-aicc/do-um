from src.evaluation.graph import build_llm_request, build_workflow_input
from src.evaluation.schemas import EvaluationScenario


def test_build_llm_request_reuses_workflow_prompt_rules() -> None:
    scenario = EvaluationScenario(
        scenario_id="procedure_001",
        user_query="대출 철회권 신청 절차를 알려주세요.",
        intent="절차형",
        domain="대출",
        subdomain="대출철회권",
        retrieved_context="대출 철회권은 계약서 수령일 등 기준일로부터 정해진 기간 안에 신청합니다.",
        reference_answer="대출 철회권 신청 가능 기간과 신청 경로를 안내합니다.",
    )

    request = build_llm_request(scenario)

    assert request.session_id == "procedure_001"
    assert request.route is not None
    assert request.route.value == "procedure"
    assert "Route=PROCEDURE" in (request.system_prompt or "")
    assert "[USER_QUERY]\n대출 철회권 신청 절차를 알려주세요." in request.prompt
    assert "[INTERNAL_CONTEXT]" in request.prompt
    assert "대출 철회권은 계약서 수령일" in request.prompt


def test_build_workflow_input_passes_metadata_for_handoff_routing() -> None:
    scenario = EvaluationScenario(
        scenario_id="loan_limit_handoff_001",
        user_query="대출 한도가 너무 낮게 나왔는데 왜 그런가요?",
        intent="설명형",
        domain="금융상담",
        subdomain="대출/한도",
        retrieved_context="대출 한도는 소득, 신용도, 기존 부채 등에 따라 달라질 수 있습니다.",
        metadata={"risk_level": "높음", "handoff_required": "Y"},
    )

    payload = build_workflow_input(scenario)
    request = build_llm_request(scenario)

    assert payload.routing_info.metadata["risk_level"] == "높음"
    assert payload.routing_info.metadata["handoff_required"] == "Y"
    assert request.route is not None
    assert request.route.value == "handoff"
    assert "Route=HANDOFF" in (request.system_prompt or "")
