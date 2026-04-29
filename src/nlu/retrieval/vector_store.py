"""Vector store 관련 경량 유틸."""

from __future__ import annotations

import time
from typing import Any


def calc_rag_vector_k(rag_top_k: int, rag_hybrid: bool, rag_fusion_pool_mult: int) -> int:
    if rag_hybrid:
        return max(rag_top_k * rag_fusion_pool_mult, rag_top_k)
    return rag_top_k


def run_rag_search(
    router: Any,
    *,
    query_vector: Any,
    stt_text: str,
    timings: dict[str, float | None],
) -> tuple[list[tuple[Any, Any]], list[Any], int]:
    k_vec = calc_rag_vector_k(router.rag_top_k, router.rag_hybrid, router.rag_fusion_pool_mult)
    t0 = time.perf_counter()
    vector_res_with_score = router.vector_db.similarity_search_by_vector_with_score(
        query_vector, k=k_vec
    )
    vec_sec = time.perf_counter() - t0
    bm25_docs: list[Any] = []
    bm25_sec = 0.0
    if router.rag_hybrid:
        t0 = time.perf_counter()
        bm25_docs, bm25_sec = router._bm25_get_docs(stt_text, k_vec)

    timings["rag_vector_sec"] = round(vec_sec, 6)
    timings["rag_bm25_sec"] = round(bm25_sec, 6) if router.rag_hybrid else None
    timings["rag_search_sec"] = vec_sec + bm25_sec
    print(
        f"  🔎 [4] RAG: Pinecone 벡터 k={k_vec} (⏱️ {vec_sec:.3f}s)"
        + (
            f" + BM25 k={k_vec} (⏱️ {bm25_sec:.3f}s) → hybrid=RRF"
            if router.rag_hybrid
            else ""
        )
    )
    return vector_res_with_score, bm25_docs, k_vec

