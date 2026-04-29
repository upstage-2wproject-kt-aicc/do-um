"""시맨틱 캐시 조회 헬퍼."""

from __future__ import annotations

import time
from typing import Any


def find_semantic_cache_hit(
    router: Any,
    query_vector: Any,
    *,
    threshold: float,
) -> tuple[dict[str, Any] | None, float]:
    max_similarity = 0.0
    for item in router.semantic_cache:
        sim = router.cosine_similarity(query_vector, item["vector"])
        if sim > max_similarity:
            max_similarity = sim
        if sim >= threshold:
            return item, sim
    return None, max_similarity


def evaluate_semantic_cache(
    router: Any,
    *,
    query_vector: Any,
    disable_cache: bool,
    timings: dict[str, float | None],
    total_t0: float,
    intent: str,
    subdomain_pred: str | None,
    threshold: float,
) -> tuple[dict[str, Any] | None, float]:
    t0 = time.perf_counter()
    if disable_cache:
        timings["cache_check_sec"] = time.perf_counter() - t0
        print("  🚫 [3] 평가 모드: 시맨틱 캐시 비활성화")
        return None, 0.0

    max_sim = 0.0
    for i, item in enumerate(router.semantic_cache):
        sim = router.cosine_similarity(query_vector, item["vector"])
        if sim > max_sim:
            max_sim = sim
        print(f"     · 캐시[{i}] 유사도: {sim:.4f} (기준 ≥ {threshold})")
        if sim >= threshold:
            timings["cache_check_sec"] = time.perf_counter() - t0
            timings["total_sec"] = time.perf_counter() - total_t0
            print(f"  🔥 [3] 시맨틱 캐시 적중 (⏱️ 캐시 검사 {timings['cache_check_sec']:.3f}s)")
            print(f"  ✅ 파이프라인 종료 — 총 ⏱️ {timings['total_sec']:.3f}s (캐시 응답)")
            print("=" * 72)
            return {
                "status": "CACHED",
                "intent": intent,
                "subdomain_pred": subdomain_pred,
                "final_answer": item["answer"],
                "cache_similarity": sim,
                "timings_sec": timings,
            }, max_sim

    timings["cache_check_sec"] = time.perf_counter() - t0
    print(
        f"  ❄️ [3] 캐시 미적중 — 최고 유사도 {max_sim:.4f} < {threshold} "
        f"(⏱️ 캐시 검사 {timings['cache_check_sec']:.3f}s)"
    )
    return None, max_sim

