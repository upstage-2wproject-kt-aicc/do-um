import json

from src.evaluation.summarize_run import summarize_run


def test_summarize_run_writes_model_answers_and_scores(tmp_path) -> None:
    result_path = tmp_path / "scenario.json"
    result_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "candidate_model": "gpt-4o",
                        "primary_score": 9.33,
                        "answer_text": "후보 답변입니다.",
                        "report_metrics": {
                            "intent_fit": 10,
                            "accuracy": 9,
                            "groundedness": 9,
                            "safety_conservatism": 9,
                            "handoff_appropriateness": 10,
                            "guidance_quality": 9,
                        },
                        "timing_ms": {"total": 1000},
                        "token_usage": {
                            "prompt_tokens": 1000,
                            "completion_tokens": 500,
                        },
                        "llm_request": {
                            "prompt": (
                                "[USER_QUERY]\n질문입니다.\n\n"
                                "[INTERNAL_CONTEXT]\n문서입니다.\n\n"
                                "[ROUTING_METADATA]\n- risk_level: 중간"
                            )
                        },
                        "judge_evaluations": [
                            {
                                "judge_model": "gpt-5.5",
                                "metrics": {
                                    "intent_fit": {"score": 10},
                                    "accuracy": {"score": 9},
                                    "groundedness": {"score": 9},
                                    "safety_conservatism": {"score": 9},
                                    "handoff_appropriateness": {"score": 10},
                                    "guidance_quality": {"score": 9},
                                },
                                "summary": {
                                    "overall_profile": "좋은 응답입니다.",
                                    "strongest_dimension": "intent_fit",
                                    "weakest_dimension": "accuracy",
                                },
                                "token_usage": {
                                    "prompt_tokens": 2000,
                                    "completion_tokens": 1000,
                                },
                            }
                        ],
                    }
                    ,
                    {
                        "candidate_model": "solar-pro3",
                        "primary_score": 7.0,
                        "answer_text": "구버전 결과입니다.",
                        "report_metrics": {},
                        "timing_ms": {"total": 500},
                        "token_usage": {
                            "prompt_tokens": 100,
                            "completion_tokens": 50,
                        },
                        "llm_request": {"prompt": ""},
                        "judge_evaluations": [
                            {
                                "judge_model": "gpt-5.5",
                                "metrics": {},
                                "summary": {},
                            }
                        ],
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (tmp_path / "index.json").write_text(
        json.dumps(
            {
                "scenario_count": 1,
                "scenarios": [
                    {
                        "scenario_id": "scenario_001",
                        "output_path": str(result_path),
                        "record_count": 1,
                        "duration_ms": 1234,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    output_path = summarize_run(tmp_path)
    text = output_path.read_text(encoding="utf-8")

    assert "## scenario_001" in text
    assert "| gpt-4o | 9.33 | 1,000 | 0.00 | 1,000 | 500 | $0.007500 | $7.5000 | $0.040000 |" in text
    assert "질문입니다." in text
    assert "후보 답변입니다." in text
    assert "| gpt-4o | 9.33 | 10 | 9 | 9 | 9 | 10 | 9 |" in text
    assert "| gpt-4o | 9.33 | 1,000 | 500 | $0.007500 | $7.5000 | 1,000 | $0.040000 |" in text
    assert "| solar-pro3 | 7.00 | 100 | 50 | $0.000045 | $0.0450 | 0 | N/A |" in text
    assert "좋은 응답입니다." in text
