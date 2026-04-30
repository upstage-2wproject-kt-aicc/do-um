"""NLU 서비스 오케스트레이션 진입점."""

from __future__ import annotations

import time
from typing import Any

from src.nlu.guardrail.policy import build_policy_rules
from src.nlu.retrieval.cache import evaluate_semantic_cache
from src.nlu.retrieval.vector_store import run_rag_search

_CACHE_SIM_THRESHOLD = 0.75


def process_nlu_query(
    router: Any,
    stt_text: str,
    *,
    disable_cache: bool = False,
    customer_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """STT 텍스트 -> 의도/RAG/가드레일/응답 오케스트레이션."""
    total_t0 = time.perf_counter()
    timings: dict[str, float | None] = {
        "intent_sec": 0.0,
        "subdomain_sec": 0.0,
        "embedding_sec": 0.0,
        "intent_embed_parallel_wall_sec": 0.0,
        "cache_check_sec": 0.0,
        "rag_vector_sec": None,
        "rag_bm25_sec": None,
        "rag_search_sec": None,
        "total_sec": 0.0,
    }

    print("\n" + "=" * 72)
    print(f"📥 [process_query] 고객 발화: {stt_text!r}")

    intent, subdomain_pred, query_vector, i_sec, s_sec, e_sec, wall_pe = router._run_intent_embed_parallel(
        stt_text
    )
    timings["intent_sec"] = i_sec
    timings["subdomain_sec"] = s_sec
    timings["embedding_sec"] = e_sec
    timings["intent_embed_parallel_wall_sec"] = wall_pe
    print(
        f"  🧠📐 [1+2] NLU 의도 + 임베딩 (병렬 wall ⏱️ {wall_pe:.3f}s, "
        f"의도 {i_sec:.3f}s, subdomain {s_sec:.3f}s, 임베딩 {e_sec:.3f}s)"
        f" → intent={intent!r}, subdomain_pred={subdomain_pred!r}"
    )

    cached_payload, max_sim = evaluate_semantic_cache(
        router,
        query_vector=query_vector,
        disable_cache=disable_cache,
        timings=timings,
        total_t0=total_t0,
        intent=intent,
        subdomain_pred=subdomain_pred,
        threshold=_CACHE_SIM_THRESHOLD,
    )
    if cached_payload is not None:
        return cached_payload

    vector_res_with_score, bm25_docs, k_vec = run_rag_search(
        router,
        query_vector=query_vector,
        stt_text=stt_text,
        timings=timings,
    )

    retrieved_context = ""
    metadata: dict[str, Any] = {}
    retrieved_faq_ids: list[str] = []
    routing_signals: dict[str, Any] = {}

    if router.rag_hybrid and bm25_docs:
        selected = router._hybrid_rrf_select(vector_res_with_score, bm25_docs)
        fusion_note = "hybrid_rrf_bm25_vector"
    else:
        selected = router._vector_only_select(vector_res_with_score)
        fusion_note = "vector_only"

    sensitive_keyword_hit = router._find_direct_handoff_keyword(stt_text)
    abusive_keyword_hit = router._find_abusive_keyword(stt_text)
    context_required_keyword = router._find_customer_context_required_keyword(stt_text)
    customer_context_present = router._has_customer_context(customer_context)
    missing_customer_context_reason: str | None = None
    if (
        router.direct_handoff_on_missing_customer_context
        and context_required_keyword is not None
        and not customer_context_present
    ):
        missing_customer_context_reason = (
            f"missing_customer_context:{context_required_keyword}"
        )
    if not selected:
        print(
            "  ⚠️ [5] RAG 채택 문서 없음 "
            "(top1 연관도 미달, 하이브리드 게이트, 또는 검색 결과 없음)"
        )
        guardrail = router._compute_guardrail(
            risk_level="low",
            handoff_required="N",
            sensitive_keyword_hit=sensitive_keyword_hit,
            abusive_keyword_hit=abusive_keyword_hit,
            missing_customer_context_reason=missing_customer_context_reason,
            rag_miss=True,
            query_text=stt_text,
        )
        guardrail_meta = {
            "risk_level": "low",
            "handoff_required": "N",
            "direct_handoff_keyword": sensitive_keyword_hit,
            "abusive_keyword": abusive_keyword_hit,
            "customer_context_present": customer_context_present,
            "customer_context_required_keyword": context_required_keyword,
            "rag_fusion": fusion_note,
        }
        if guardrail["decision"] == "HANDOFF":
            timings["total_sec"] = time.perf_counter() - total_t0
            print(f"  🚨 [H] Direct handoff 발동 (reason={guardrail['reasons']})")
            print("=" * 72)
            return router._build_direct_handoff_response(
                intent=intent,
                subdomain_pred=subdomain_pred,
                metadata=guardrail_meta,
                retrieved_context="",
                retrieved_faq_ids=[],
                routing_signals={
                    "routing_mode": "risk_first",
                    "risk_level": "low",
                    "handoff_required": "N",
                    "guardrail_decision": guardrail["decision"],
                },
                cache_max_similarity=max_sim,
                timings=timings,
                reasons=guardrail["reasons"],
                guardrail_score=guardrail["score"],
                guardrail_components=guardrail.get("components"),
            )
        if guardrail["decision"] == "REJECT":
            timings["total_sec"] = time.perf_counter() - total_t0
            print(f"  ⛔ [G] Guardrail REJECT 발동 (reason={guardrail['reasons']})")
            print("=" * 72)
            return router._build_reject_response(
                intent=intent,
                subdomain_pred=subdomain_pred,
                metadata=guardrail_meta,
                retrieved_context="",
                retrieved_faq_ids=[],
                routing_signals={
                    "routing_mode": "risk_first",
                    "risk_level": "low",
                    "handoff_required": "N",
                    "guardrail_decision": guardrail["decision"],
                },
                cache_max_similarity=max_sim,
                timings=timings,
                reasons=guardrail["reasons"],
                guardrail_score=guardrail["score"],
                guardrail_components=guardrail.get("components"),
            )
        timings["total_sec"] = time.perf_counter() - total_t0
        print(f"  ✅ 파이프라인 종료 — 총 ⏱️ {timings['total_sec']:.3f}s (LLM 단계로 전달)")
        print("=" * 72)
        policy_rules_miss = build_policy_rules(
            guardrail["decision"], router.guardrail_limit_message
        )
        return {
            "status": "REQUIRE_LLM",
            "intent": intent,
            "subdomain_pred": subdomain_pred,
            "retrieved_context": "",
            "metadata": guardrail_meta,
            "retrieved_faq_ids": [],
            "routing_signals": {
                "routing_mode": "risk_first",
                "risk_level": "low",
                "handoff_required": "N",
                "guardrail_decision": guardrail["decision"],
            },
            "guardrail_decision": guardrail["decision"],
            "guardrail_score": guardrail["score"],
            "guardrail_reasons": guardrail["reasons"],
            "guardrail_components": guardrail.get("components", {}),
            "policy_rules": policy_rules_miss,
            "cache_max_similarity": max_sim,
            "timings_sec": timings,
        }

    top_doc = selected[0][0]
    meta_top = dict(top_doc.metadata)
    agg_risk, agg_ho = router._aggregate_risk_handoff([d for d, _ in selected])
    metadata = meta_top
    metadata["risk_level"] = agg_risk
    metadata["handoff_required"] = agg_ho
    metadata["rag_fusion"] = fusion_note

    if router.subdomain_source == "rag":
        sub_topic = str(meta_top.get("subdomain", "")).strip()
        if sub_topic:
            subdomain_pred = sub_topic

    routing_signals = {
        "routing_mode": "risk_first",
        "risk_level": agg_risk,
        "handoff_required": agg_ho,
    }

    retrieved_context = "\n\n".join(doc.page_content for doc, _ in selected)
    retrieved_faq_ids = [
        str(doc.metadata.get("faq_id", "")).strip()
        for doc, _ in selected
        if str(doc.metadata.get("faq_id", "")).strip()
    ]
    all_scores = [round(float(s), 4) for _, s in vector_res_with_score]
    selected_scores = [round(float(s), 4) for _, s in selected]
    print(
        f"  📋 [5] 검색 문서 ({fusion_note}, 후보 벡터 {len(vector_res_with_score)}건"
        f"{', BM25 ' + str(len(bm25_docs)) + '건' if router.rag_hybrid and bm25_docs else ''}"
        f" → 채택 {len(selected)}건):"
    )
    print(
        "      • 2차 필터: 벡터 score 기준 "
        f"score >= top1*{router.rag_secondary_min_ratio:.2f} "
        f"and score >= {router.rag_min_relevance:.2f} "
        "(하이브리드 시 RRF 보조)"
    )
    print(f"      • scores(vector 후보): {all_scores}")
    print(f"      • scores(selected 벡터스코어): {selected_scores}")
    print(f"      • faq_ids: {retrieved_faq_ids}")
    print(
        f"      • top1 domain > subdomain: {metadata.get('domain')} > {metadata.get('subdomain')}"
    )
    print(f"      • top1 intent_type(문서 메타): {metadata.get('intent_type')}")
    print(f"      • top1 source_url: {metadata.get('source_url')}")
    print(
        "      • 집계 risk_level / handoff (채택 문서 전체): "
        f"{metadata.get('risk_level')} / {metadata.get('handoff_required')}"
    )
    preview = (retrieved_context[:120] + "…") if len(retrieved_context) > 120 else retrieved_context
    print(f"      • 본문 미리보기: {preview!r}")

    guardrail = router._compute_guardrail(
        risk_level=str(metadata.get("risk_level", "low")).strip().lower(),
        handoff_required=str(metadata.get("handoff_required", "N")).strip().upper(),
        sensitive_keyword_hit=sensitive_keyword_hit,
        abusive_keyword_hit=abusive_keyword_hit,
        missing_customer_context_reason=missing_customer_context_reason,
        rag_miss=False,
        query_text=stt_text,
    )

    if guardrail["decision"] == "HANDOFF":
        timings["total_sec"] = time.perf_counter() - total_t0
        print(f"  🚨 [H] Direct handoff 발동 (reason={guardrail['reasons']})")
        print(f"  ✅ 파이프라인 종료 — 총 ⏱️ {timings['total_sec']:.3f}s (LLM 우회)")
        print("=" * 72)
        return router._build_direct_handoff_response(
            intent=intent,
            subdomain_pred=subdomain_pred,
            metadata=metadata,
            retrieved_context=retrieved_context,
            retrieved_faq_ids=retrieved_faq_ids,
            routing_signals=routing_signals,
            cache_max_similarity=max_sim,
            timings=timings,
            reasons=guardrail["reasons"],
            guardrail_score=guardrail["score"],
            guardrail_components=guardrail.get("components"),
        )
    if guardrail["decision"] == "REJECT":
        timings["total_sec"] = time.perf_counter() - total_t0
        print(f"  ⛔ [G] Guardrail REJECT 발동 (reason={guardrail['reasons']})")
        print(f"  ✅ 파이프라인 종료 — 총 ⏱️ {timings['total_sec']:.3f}s (LLM 우회)")
        print("=" * 72)
        return router._build_reject_response(
            intent=intent,
            subdomain_pred=subdomain_pred,
            metadata=metadata,
            retrieved_context=retrieved_context,
            retrieved_faq_ids=retrieved_faq_ids,
            routing_signals=routing_signals,
            cache_max_similarity=max_sim,
            timings=timings,
            reasons=guardrail["reasons"],
            guardrail_score=guardrail["score"],
            guardrail_components=guardrail.get("components"),
        )

    timings["total_sec"] = time.perf_counter() - total_t0
    print(f"  ✅ 파이프라인 종료 — 총 ⏱️ {timings['total_sec']:.3f}s (LLM 단계로 전달)")
    print("=" * 72)
    policy_rules = build_policy_rules(guardrail["decision"], router.guardrail_limit_message)

    return {
        "status": "REQUIRE_LLM",
        "intent": intent,
        "subdomain_pred": subdomain_pred,
        "retrieved_context": retrieved_context,
        "metadata": metadata,
        "retrieved_faq_ids": retrieved_faq_ids,
        "routing_signals": routing_signals,
        "guardrail_decision": guardrail["decision"],
        "guardrail_score": guardrail["score"],
        "guardrail_reasons": guardrail["reasons"],
        "guardrail_components": guardrail.get("components", {}),
        "policy_rules": policy_rules,
        "cache_max_similarity": max_sim,
        "timings_sec": timings,
    }

