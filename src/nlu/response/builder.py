"""Guardrail 결과 기반 최종 응답 생성기."""

from __future__ import annotations

from typing import Any


def build_direct_handoff_response(
    router: Any,
    *,
    intent: str,
    subdomain_pred: str | None,
    metadata: dict[str, Any],
    retrieved_context: str,
    retrieved_faq_ids: list[str],
    routing_signals: dict[str, Any],
    cache_max_similarity: float,
    timings: dict[str, float | None],
    reasons: list[str],
    guardrail_score: float,
    guardrail_components: dict[str, int] | None = None,
) -> dict[str, Any]:
    return {
        "status": "HANDOFF_DIRECT",
        "intent": intent,
        "subdomain_pred": subdomain_pred,
        "final_answer": router.direct_handoff_message,
        "metadata": metadata,
        "retrieved_context": retrieved_context,
        "retrieved_faq_ids": retrieved_faq_ids,
        "routing_signals": routing_signals,
        "handoff_reason": reasons,
        "handoff_confidence": 1.0 if reasons else 0.0,
        "guardrail_decision": "HANDOFF",
        "guardrail_score": guardrail_score,
        "guardrail_reasons": reasons,
        "guardrail_components": guardrail_components or {},
        "transfer_action": {
            "type": "TRANSFER_CALL",
            "required": True,
            "reason": reasons,
        },
        "action": {
            "type": "TRANSFER_CALL",
            "required": True,
            "reason": reasons,
        },
        "cache_max_similarity": cache_max_similarity,
        "timings_sec": timings,
    }


def build_reject_response(
    router: Any,
    *,
    intent: str,
    subdomain_pred: str | None,
    metadata: dict[str, Any],
    retrieved_context: str,
    retrieved_faq_ids: list[str],
    routing_signals: dict[str, Any],
    cache_max_similarity: float,
    timings: dict[str, float | None],
    reasons: list[str],
    guardrail_score: float,
    guardrail_components: dict[str, int] | None = None,
) -> dict[str, Any]:
    return {
        "status": "REJECT_DIRECT",
        "intent": intent,
        "subdomain_pred": subdomain_pred,
        "final_answer": router.guardrail_reject_message,
        "metadata": metadata,
        "retrieved_context": retrieved_context,
        "retrieved_faq_ids": retrieved_faq_ids,
        "routing_signals": routing_signals,
        "guardrail_decision": "REJECT",
        "guardrail_score": guardrail_score,
        "guardrail_reasons": reasons,
        "guardrail_components": guardrail_components or {},
        "action": {
            "type": "REJECT_QUERY",
            "required": True,
            "reason": reasons,
        },
        "cache_max_similarity": cache_max_similarity,
        "timings_sec": timings,
    }

