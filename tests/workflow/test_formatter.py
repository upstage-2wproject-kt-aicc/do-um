"""Workflow formatter tests."""

from src.common.schemas import LLMBatchResponse, LLMResponse
from src.workflow.formatter import format_workflow_output


def test_format_workflow_output_selects_first_success() -> None:
    """Selects first non-error answer as final output."""
    batch = LLMBatchResponse(
        session_id="sess-1",
        responses=[
            LLMResponse(
                session_id="sess-1",
                provider="solar",
                text="ok",
                ttft_ms=120,
                latency_ms=400,
                finish_reason="stop",
                grounded=True,
                citations=["doc-1"],
                error=None,
                token_usage={"prompt_tokens": 10, "completion_tokens": 20},
            ),
            LLMResponse(
                session_id="sess-1",
                provider="gpt",
                text="",
                ttft_ms=0,
                latency_ms=0,
                finish_reason=None,
                grounded=False,
                citations=[],
                error="MISSING_API_KEY",
                token_usage={},
            ),
        ],
    )
    output = format_workflow_output(batch)
    assert output.final_answer_text == "ok"
    assert output.is_handoff_decided is False

