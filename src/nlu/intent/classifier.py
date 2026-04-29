"""Intent/Subdomain 분류 및 병렬 실행 헬퍼."""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch


def predict_intent(router: Any, text: str) -> str:
    if not router.use_real_nlu or router.nlu_model is None or router.nlu_tokenizer is None:
        return "절차형"

    inputs = router.nlu_tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=128
    )
    if "token_type_ids" in inputs:
        inputs["token_type_ids"] = torch.zeros_like(inputs["token_type_ids"])
    inputs = {k: v.to(router.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = router.nlu_model(**inputs)
        predicted_id = int(outputs.logits.argmax(dim=-1).item())
    return router.intent_map.get(predicted_id, "분류불가")


def predict_subdomain(router: Any, text: str) -> str | None:
    if (
        not router.use_real_subdomain_nlu
        or router.subdomain_model is None
        or router.subdomain_tokenizer is None
    ):
        return None

    inputs = router.subdomain_tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=128
    )
    if "token_type_ids" in inputs:
        inputs["token_type_ids"] = torch.zeros_like(inputs["token_type_ids"])
    inputs = {k: v.to(router.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = router.subdomain_model(**inputs)
        predicted_id = int(outputs.logits.argmax(dim=-1).item())
    return router.subdomain_label_map.get(predicted_id, str(predicted_id))


async def intent_embed_parallel_async(
    router: Any, stt_text: str
) -> tuple[str, str | None, Any, float, float, float, float]:
    async def timed_intent() -> tuple[str, float]:
        t0 = time.perf_counter()
        out = await asyncio.to_thread(predict_intent, router, stt_text)
        return out, time.perf_counter() - t0

    async def timed_embed() -> tuple[Any, float]:
        t0 = time.perf_counter()
        out = await asyncio.to_thread(router.embeddings.embed_query, stt_text)
        return out, time.perf_counter() - t0

    async def timed_subdomain() -> tuple[str | None, float]:
        t0 = time.perf_counter()
        out = await asyncio.to_thread(predict_subdomain, router, stt_text)
        return out, time.perf_counter() - t0

    wall0 = time.perf_counter()
    if router.subdomain_source == "rag":
        (intent, intent_sec), (query_vector, embed_sec) = await asyncio.gather(
            timed_intent(),
            timed_embed(),
        )
        subdomain_pred = None
        subdomain_sec = 0.0
    else:
        (intent, intent_sec), (subdomain_pred, subdomain_sec), (
            query_vector,
            embed_sec,
        ) = await asyncio.gather(
            timed_intent(),
            timed_subdomain(),
            timed_embed(),
        )
    wall_sec = time.perf_counter() - wall0
    return intent, subdomain_pred, query_vector, intent_sec, subdomain_sec, embed_sec, wall_sec


def intent_embed_parallel_threadpool(
    router: Any, stt_text: str
) -> tuple[str, str | None, Any, float, float, float, float]:
    intent_sec_local = 0.0
    subdomain_sec_local = 0.0
    embed_sec_local = 0.0

    def timed_intent() -> str:
        nonlocal intent_sec_local
        t0 = time.perf_counter()
        r = predict_intent(router, stt_text)
        intent_sec_local = time.perf_counter() - t0
        return r

    def timed_embed() -> Any:
        nonlocal embed_sec_local
        t0 = time.perf_counter()
        r = router.embeddings.embed_query(stt_text)
        embed_sec_local = time.perf_counter() - t0
        return r

    def timed_subdomain() -> str | None:
        nonlocal subdomain_sec_local
        t0 = time.perf_counter()
        r = predict_subdomain(router, stt_text)
        subdomain_sec_local = time.perf_counter() - t0
        return r

    wall0 = time.perf_counter()
    if router.subdomain_source == "rag":
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_i = pool.submit(timed_intent)
            fut_e = pool.submit(timed_embed)
            intent = fut_i.result()
            subdomain_pred = None
            subdomain_sec_local = 0.0
            query_vector = fut_e.result()
    else:
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_i = pool.submit(timed_intent)
            fut_s = pool.submit(timed_subdomain)
            fut_e = pool.submit(timed_embed)
            intent = fut_i.result()
            subdomain_pred = fut_s.result()
            query_vector = fut_e.result()
    wall_sec = time.perf_counter() - wall0
    return (
        intent,
        subdomain_pred,
        query_vector,
        intent_sec_local,
        subdomain_sec_local,
        embed_sec_local,
        wall_sec,
    )


def run_intent_embed_parallel(
    router: Any, stt_text: str
) -> tuple[str, str | None, Any, float, float, float, float]:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(intent_embed_parallel_async(router, stt_text))
    return intent_embed_parallel_threadpool(router, stt_text)

