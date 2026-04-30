"""RAG 인덱스 준비/초기화 관련 로직."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone


def bm25_get_docs(router: Any, query: str, k: int) -> tuple[list[Any], float]:
    prev_k = router.bm25.k
    router.bm25.k = max(1, k)
    t0 = time.perf_counter()
    try:
        docs = router.bm25.invoke(query)
    finally:
        router.bm25.k = prev_k
    return docs, time.perf_counter() - t0


def warm_up_cache(router: Any) -> None:
    t0 = time.perf_counter()
    q = "비대면 통장 개설"
    a = "비대면 계좌 개설은 당행 모바일 앱을 통해 24시간 언제든 가능합니다."
    router.semantic_cache.append({"vector": router.embeddings.embed_query(q), "answer": a})
    t_warm = time.perf_counter() - t0
    print(f"   ↳ 시맨틱 캐시 워밍 1건 (⏱️ {t_warm:.3f}s)")


def prepare_datasets(
    router: Any,
    *,
    faq_csv: Path,
    persist_dir: str,
    sha256_file: Callable[[Path], str],
    read_index_fingerprint: Callable[[Path], dict[str, Any] | None],
    write_index_fingerprint: Callable[[Path], None] | Callable[..., None],
    pinecone_total_vector_count: Callable[[Any], int],
) -> None:
    if not faq_csv.is_file():
        raise FileNotFoundError(f"FAQ CSV가 없습니다: {faq_csv}")

    csv_fp = sha256_file(faq_csv)
    router.rag_source_fingerprint_sha256 = csv_fp
    print(f"   ↳ FAQ 소스 지문(SHA256): {csv_fp[:16]}… (전체 {len(csv_fp)} hex)")

    t0 = time.perf_counter()
    df = pd.read_csv(faq_csv).fillna("")
    router.docs = []
    for _, row in df.iterrows():
        if str(row["embedding_text"]).strip():
            metadata = {
                "faq_id": str(row["faq_id"]),
                "domain": str(row["domain"]),
                "subdomain": str(row["subdomain"]),
                "intent_type": str(row["intent_type"]),
                "keywords": str(row["keywords"]),
                "source_url": str(row.get("source_url", "")).strip(),
                "risk_level": router._normalize_risk_level(row["risk_level"]),
                "handoff_required": router._normalize_handoff_required(row["handoff_required"]),
            }
            router.docs.append(
                Document(page_content=str(row["embedding_text"]), metadata=metadata)
            )
    t_csv = time.perf_counter() - t0
    print(f"   ↳ FAQ CSV 로드: 문서 {len(router.docs)}건 (⏱️ {t_csv:.3f}s)")

    t0 = time.perf_counter()
    router.bm25 = BM25Retriever.from_documents(router.docs)
    router.bm25.k = 2
    t_bm25 = time.perf_counter() - t0
    print(f"   ↳ BM25 인덱스 구축 (⏱️ {t_bm25:.3f}s)")

    t0 = time.perf_counter()
    persist_path = Path(persist_dir)
    stored = read_index_fingerprint(persist_path)
    index_matches_csv = bool(
        stored
        and stored.get("sha256") == csv_fp
        and stored.get("source_csv") == faq_csv.name
    )
    if not index_matches_csv:
        if stored:
            print("   ↳ 저장된 인덱스 지문과 FAQ CSV 불일치 — 재색인이 필요합니다.")
        else:
            print("   ↳ 인덱스 지문 파일 없음(구버전 포함) — 필요 시 재색인합니다.")

    index_name = os.environ["PINECONE_INDEX_NAME"]
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(index_name)
    index_stats = index.describe_index_stats()
    total_vectors = pinecone_total_vector_count(index_stats)
    needs_rebuild = (not index_matches_csv) or (total_vectors == 0)

    if needs_rebuild:
        if total_vectors > 0 and not index_matches_csv:
            print("   ↳ 기존 Pinecone 인덱스 데이터를 전체 삭제 후 재색인합니다...")
            index.delete(delete_all=True)
            time.sleep(2)
        print("   ↳ Pinecone Vector DB 적재 중(최초 구축 또는 재구축)...")
        router.vector_db = PineconeVectorStore.from_documents(
            documents=router.docs,
            embedding=router.embeddings,
            index_name=index_name,
        )
        write_index_fingerprint(
            persist_path, source_name=faq_csv.name, sha256_hex=csv_fp
        )
        print(
            f"   ↳ Pinecone 적재 및 지문 저장 완료 (⏱️ {time.perf_counter() - t0:.3f}s)"
        )
    else:
        print(f"   ↳ Pinecone 기존 인덱스 연결 (벡터 {total_vectors}건)...")
        router.vector_db = PineconeVectorStore(
            index_name=index_name,
            embedding=router.embeddings,
        )
        print(f"   ↳ Pinecone 연결 완료 (⏱️ {time.perf_counter() - t0:.3f}s)")

