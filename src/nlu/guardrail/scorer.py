"""Guardrail score 계산기."""

from __future__ import annotations

from typing import Any


def compute_guardrail(
    router: Any,
    *,
    risk_level: str,
    handoff_required: str,
    sensitive_keyword_hit: str | None,
    abusive_keyword_hit: str | None,
    missing_customer_context_reason: str | None,
    rag_miss: bool,
    query_text: str,
) -> dict[str, Any]:
    reasons: list[str] = []

    meta_score_raw = 0
    if router.direct_handoff_on_high_risk and risk_level in {"high", "critical"}:
        meta_score_raw += router.guardrail_score_meta_high
        reasons.append(f"meta_risk_high(+{router.guardrail_score_meta_high})")
    if router.direct_handoff_on_required and handoff_required == "Y":
        meta_score_raw += router.guardrail_score_meta_required
        reasons.append(f"meta_handoff_required(+{router.guardrail_score_meta_required})")
    meta_score = min(meta_score_raw, router.guardrail_meta_cap)
    if meta_score_raw > router.guardrail_meta_cap:
        reasons.append(f"meta_cap_applied({meta_score_raw}->{meta_score})")

    keyword_score = 0
    if router.guardrail_enable_keyword:
        if abusive_keyword_hit:
            keyword_score = router.guardrail_score_keyword_abusive
            reasons.append(
                f"keyword_abusive:{abusive_keyword_hit}(+{router.guardrail_score_keyword_abusive})"
            )
        elif sensitive_keyword_hit:
            keyword_score = router.guardrail_score_keyword_sensitive
            reasons.append(
                f"keyword_sensitive:{sensitive_keyword_hit}(+{router.guardrail_score_keyword_sensitive})"
            )

    missing_ctx_score = 0
    if missing_customer_context_reason:
        missing_ctx_score = router.guardrail_score_missing_customer_context
        reasons.append(
            f"{missing_customer_context_reason}(+{router.guardrail_score_missing_customer_context})"
        )

    ood_score = 0
    if rag_miss:
        ood_score = router.guardrail_score_ood
        reasons.append(f"rag_miss(+{router.guardrail_score_ood})")

    total_score = meta_score + keyword_score + missing_ctx_score + ood_score
    finance_related = router._contains_finance_keyword(query_text)

    decision = "ALLOW"
    if total_score >= router.guardrail_handoff_threshold:
        decision = "HANDOFF"
    elif (
        router.guardrail_enable_ood_reject
        and rag_miss
        and not finance_related
        and total_score >= router.guardrail_reject_threshold
    ):
        decision = "REJECT"
    elif total_score >= router.guardrail_limit_threshold:
        decision = "LIMIT"

    return {
        "decision": decision,
        "score": float(total_score),
        "reasons": reasons,
        "components": {
            "meta": meta_score,
            "keyword": keyword_score,
            "missing_customer_context": missing_ctx_score,
            "ood": ood_score,
        },
        "finance_related": finance_related,
        "rag_miss": rag_miss,
    }

