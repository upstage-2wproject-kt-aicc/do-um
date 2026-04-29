from pathlib import Path

from src.evaluation.loader import load_scenarios_tsv


def test_load_scenarios_tsv_maps_workflow_ready_fields(tmp_path: Path) -> None:
    path = tmp_path / "scenarios.tsv"
    path.write_text(
        "\t".join(
            [
                "scenario_id",
                "question",
                "intent",
                "domain",
                "subdomain",
                "retrieved_context",
                "context_metadata",
                "keywords",
                "reference_answer",
            ]
        )
        + "\n"
        + "\t".join(
            [
                "loan_rate_basic_001",
                "고정금리와 변동금리의 차이는 무엇인가요?",
                "설명형",
                "금융상담",
                "대출/금리",
                "고정금리는 약정 기간 동안 동일합니다.",
                '{"faq_id":"1","risk_level":"중간","handoff_required":"N"}',
                "고정금리,변동금리",
                "고정금리와 변동금리의 핵심 차이를 설명합니다.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    scenarios = load_scenarios_tsv(path)

    assert len(scenarios) == 1
    scenario = scenarios[0]
    assert scenario.scenario_id == "loan_rate_basic_001"
    assert scenario.user_query == "고정금리와 변동금리의 차이는 무엇인가요?"
    assert scenario.intent == "설명형"
    assert scenario.domain == "금융상담"
    assert scenario.subdomain == "대출/금리"
    assert scenario.retrieved_context == "고정금리는 약정 기간 동안 동일합니다."
    assert scenario.reference_answer == "고정금리와 변동금리의 핵심 차이를 설명합니다."
    assert scenario.metadata["faq_id"] == "1"
    assert scenario.metadata["risk_level"] == "중간"
    assert scenario.metadata["keywords"] == ["고정금리", "변동금리"]


def test_load_scenarios_tsv_allows_empty_retrieved_context(tmp_path: Path) -> None:
    path = tmp_path / "scenarios.tsv"
    path.write_text(
        "\t".join(
            [
                "scenario_id",
                "question",
                "intent",
                "domain",
                "subdomain",
                "retrieved_context",
                "context_metadata",
                "keywords",
                "reference_answer",
            ]
        )
        + "\n"
        + "\t".join(
            [
                "account_freeze_no_context",
                "제 계좌가 왜 지급정지됐는지 알려주세요.",
                "민원형",
                "금융상담",
                "계좌/지급정지",
                "",
                "{}",
                "지급정지,본인확인",
                "검색된 문서가 없으므로 사유를 단정하지 않고 본인확인 채널을 안내합니다.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    scenario = load_scenarios_tsv(path)[0]

    assert scenario.retrieved_context == ""
    assert scenario.metadata["keywords"] == ["지급정지", "본인확인"]
