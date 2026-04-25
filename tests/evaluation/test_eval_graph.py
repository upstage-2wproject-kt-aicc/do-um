from src.evaluation.graph import build_llm_request
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
