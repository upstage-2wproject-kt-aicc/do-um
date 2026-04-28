from pathlib import Path

from src.evaluation.graph import build_llm_request
from src.evaluation.loader import load_scenarios_tsv


def test_rag_faq_v1_contains_six_diverse_single_turn_scenarios() -> None:
    scenarios = load_scenarios_tsv(Path("evaluation/scenarios/rag_faq_v1.tsv"))

    assert [scenario.scenario_id for scenario in scenarios] == [
        "rag_faq_001_rate_explain",
        "rag_faq_016_autopay_procedure",
        "rag_faq_019_loan_eligibility_no_guarantee",
        "rag_faq_015_card_dispute_no_handoff",
        "rag_faq_context_insufficient_account_freeze",
        "rag_faq_028_voice_phishing_handoff",
    ]
    assert {scenario.intent for scenario in scenarios} == {
        "설명형",
        "절차형",
        "조회형",
        "민원형",
    }
    assert {
        scenario.metadata["handoff_required"]
        for scenario in scenarios
        if "handoff_required" in scenario.metadata
    } == {
        "N",
        "Y",
    }
    assert {
        scenario.metadata["risk_level"]
        for scenario in scenarios
        if "risk_level" in scenario.metadata
    } == {
        "낮음",
        "중간",
        "높음",
    }
    assert sum(1 for scenario in scenarios if not scenario.retrieved_context) == 1
    assert (
        next(
            scenario
            for scenario in scenarios
            if scenario.scenario_id == "rag_faq_context_insufficient_account_freeze"
        ).metadata["retrieval_status"]
        == "no_match"
    )
    assert all(scenario.reference_answer for scenario in scenarios)


def test_rag_faq_v1_routes_cover_core_workflow_paths() -> None:
    scenarios = load_scenarios_tsv(Path("evaluation/scenarios/rag_faq_v1.tsv"))

    assert {
        scenario.scenario_id: build_llm_request(scenario).route.value
        for scenario in scenarios
    } == {
        "rag_faq_001_rate_explain": "faq",
        "rag_faq_016_autopay_procedure": "procedure",
        "rag_faq_019_loan_eligibility_no_guarantee": "faq",
        "rag_faq_015_card_dispute_no_handoff": "handoff",
        "rag_faq_context_insufficient_account_freeze": "handoff",
        "rag_faq_028_voice_phishing_handoff": "security",
    }
