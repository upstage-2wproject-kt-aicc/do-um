"""NLU + RAG 라우터: 의도·검색·캐시 결과를 dict로 반환해 워크플로우 단계에서 재사용합니다."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import numpy as np
import pandas as pd
import torch
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_upstage import UpstageEmbeddings
from pinecone import Pinecone
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from transformers.models.auto.tokenization_auto import AutoTokenizer

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
faq_csv = _SCRIPT_DIR / "RAG_FAQ.csv"
persist_dir = str(_SCRIPT_DIR / "aicc_chroma_db")

# 프로젝트 루트의 .env (이미 설정된 환경 변수는 덮어쓰지 않음)
load_dotenv(_REPO_ROOT / ".env")

# 시맨틱 캐시 적중 임계값 (코사인 유사도)
_CACHE_SIM_THRESHOLD = 0.75

_FINGERPRINT_FILENAME = "aicc_source_fingerprint.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_index_fingerprint(persist_path: Path) -> dict[str, Any] | None:
    fp = persist_path / _FINGERPRINT_FILENAME
    if not fp.is_file():
        return None
    try:
        return json.loads(fp.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _write_index_fingerprint(
    persist_path: Path, *, source_name: str, sha256_hex: str
) -> None:
    persist_path.mkdir(parents=True, exist_ok=True)
    payload = {"source_csv": source_name, "sha256": sha256_hex}
    (persist_path / _FINGERPRINT_FILENAME).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _pinecone_total_vector_count(stats: Any) -> int:
    """Pinecone describe_index_stats 응답에서 총 벡터 수를 추출."""
    if isinstance(stats, dict):
        return int(stats.get("total_vector_count", 0))
    return int(getattr(stats, "total_vector_count", 0))


class AICC_NLU_Router:
    """KLUE 기반 의도분류 + BM25/Pinecone RAG + 시맨틱 캐시."""

    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(AICC_NLU_Router, cls).__new__(cls)
        return cls._instance

    def __init__(
        self,
        *,
        intent_model_dir: str | Path | None = None,
        subdomain_model_dir: str | Path | None = None,
    ) -> None:
        if AICC_NLU_Router._initialized:
            return
            
        if not os.environ.get("LLM_SOLAR_API_KEY"):
            raise RuntimeError(
                "LLM_SOLAR_API_KEY가 없습니다. 저장소 루트에 .env를 두고 "
                "LLM_SOLAR_API_KEY=... 를 설정하거나, .env.example을 복사해 채워 넣으세요."
            )
        if not os.environ.get("PINECONE_API_KEY"):
            raise RuntimeError(
                "PINECONE_API_KEY가 없습니다. 저장소 루트 .env에 "
                "PINECONE_API_KEY=... 를 설정하세요."
            )
        if not os.environ.get("PINECONE_INDEX_NAME"):
            raise RuntimeError(
                "PINECONE_INDEX_NAME가 없습니다. 저장소 루트 .env에 "
                "PINECONE_INDEX_NAME=... 를 설정하세요."
            )
        os.environ.setdefault("UPSTAGE_API_KEY", os.environ["LLM_SOLAR_API_KEY"])

        boot_t0 = time.perf_counter()
        print("\n🚀 [NLU Router] KLUE 로컬 모델 부팅 중...")
        self.model_path = (
            Path(intent_model_dir)
            if intent_model_dir is not None
            else (_SCRIPT_DIR / "my_aicc_nlu_model_klue")
        )
        self.subdomain_model_path = (
            Path(subdomain_model_dir) if subdomain_model_dir is not None else None
        )
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"  🖥️ NLU 가속기: {self.device}")

        t0 = time.perf_counter()
        try:
            self.nlu_tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.nlu_model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.nlu_model.to(self.device)
            self.nlu_model.eval()
            self.use_real_nlu = True
            t_nlu_load = time.perf_counter() - t0
            print(f"  ✅ Intent 모델 로드 성공: {self.model_path} (⏱️ {t_nlu_load:.3f}s)")
            self.intent_map = self._resolve_label_map(
                model=self.nlu_model,
                env_key="NLU_INTENT_LABELS",
                fallback={0: "절차형", 1: "민원형", 2: "조회형"},
            )
        except Exception as e:
            t_nlu_load = time.perf_counter() - t0
            print(f"  ⚠️ Intent 모델 로드 실패, 임시 모드 (⏱️ {t_nlu_load:.3f}s): {e}")
            self.use_real_nlu = False
            self.nlu_tokenizer = None
            self.nlu_model = None
            self.intent_map = {}

        self.use_real_subdomain_nlu = False
        self.subdomain_tokenizer = None
        self.subdomain_model = None
        self.subdomain_label_map: dict[int, str] = {}
        if self.subdomain_model_path is not None:
            t0 = time.perf_counter()
            try:
                self.subdomain_tokenizer = AutoTokenizer.from_pretrained(self.subdomain_model_path)
                self.subdomain_model = AutoModelForSequenceClassification.from_pretrained(
                    self.subdomain_model_path
                )
                self.subdomain_model.to(self.device)
                self.subdomain_model.eval()
                self.use_real_subdomain_nlu = True
                self.subdomain_label_map = self._resolve_label_map(
                    model=self.subdomain_model,
                    env_key="NLU_SUBDOMAIN_LABELS",
                    fallback={},
                )
                print(
                    "  ✅ Subdomain 모델 로드 성공: "
                    f"{self.subdomain_model_path} (⏱️ {time.perf_counter() - t0:.3f}s)"
                )
            except Exception as e:
                print(
                    "  ⚠️ Subdomain 모델 로드 실패, 비활성화 "
                    f"(⏱️ {time.perf_counter() - t0:.3f}s): {e}"
                )

        t0 = time.perf_counter()
        self.embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
        t_emb_factory = time.perf_counter() - t0
        print(f"  📎 Upstage Embeddings 준비 (⏱️ {t_emb_factory:.3f}s)")

        self.semantic_cache: list[dict[str, Any]] = []
        self.rag_source_fingerprint_sha256: str = ""
        self.rag_top_k: int = max(1, int(os.getenv("NLU_RAG_TOP_K", "2")))
        self.rag_min_relevance: float = float(os.getenv("NLU_RAG_MIN_RELEVANCE", "0.0"))
        self.rag_top1_min_relevance: float = float(
            os.getenv("NLU_RAG_TOP1_MIN_RELEVANCE", "0.30")
        )
        self.rag_secondary_min_ratio: float = float(
            os.getenv("NLU_RAG_SECONDARY_MIN_RATIO", "0.9")
        )
        print(f"  🔢 RAG top-k: {self.rag_top_k}")

        self._prepare_datasets()
        self._warm_up_cache()

        print(
            f"✅ [NLU Router] 부팅 완료 — 총 소요 ⏱️ {time.perf_counter() - boot_t0:.3f}s\n"
        )
        AICC_NLU_Router._initialized = True

    @staticmethod
    def _normalize_risk_level(value: Any) -> str:
        """Normalizes risk level labels for workflow handoff rules."""
        raw = str(value).strip().lower()
        if raw in {"high", "높음", "상", "critical", "crit"}:
            return "high"
        if raw in {"medium", "보통", "중간", "mid"}:
            return "medium"
        if raw in {"low", "낮음", "하"}:
            return "low"
        return raw or "low"

    @staticmethod
    def _normalize_handoff_required(value: Any) -> str:
        """Normalizes handoff flags to Y/N."""
        raw = str(value).strip().lower()
        if raw in {"y", "yes", "true", "1", "required"}:
            return "Y"
        if raw in {"n", "no", "false", "0", "optional"}:
            return "N"
        return "N"

    @staticmethod
    def _resolve_label_map(
        *,
        model: Any,
        env_key: str,
        fallback: dict[int, str],
    ) -> dict[int, str]:
        raw_labels = os.getenv(env_key, "").strip()
        if raw_labels:
            parsed = [x.strip() for x in raw_labels.split(",") if x.strip()]
            return {i: name for i, name in enumerate(parsed)}

        id2label = getattr(getattr(model, "config", None), "id2label", None)
        if isinstance(id2label, dict) and id2label:
            out: dict[int, str] = {}
            for k, v in id2label.items():
                try:
                    out[int(k)] = str(v)
                except (TypeError, ValueError):
                    continue
            if out:
                return out
        return fallback.copy()

    def _prepare_datasets(self) -> None:
        if not faq_csv.is_file():
            raise FileNotFoundError(f"FAQ CSV가 없습니다: {faq_csv}")

        csv_fp = _sha256_file(faq_csv)
        self.rag_source_fingerprint_sha256 = csv_fp
        print(f"   ↳ FAQ 소스 지문(SHA256): {csv_fp[:16]}… (전체 {len(csv_fp)} hex)")

        t0 = time.perf_counter()
        df = pd.read_csv(faq_csv).fillna("")
        self.docs: list[Document] = []
        for _, row in df.iterrows():
            if str(row["embedding_text"]).strip():
                metadata = {
                    "faq_id": str(row["faq_id"]),
                    "domain": str(row["domain"]),
                    "subdomain": str(row["subdomain"]),
                    "intent_type": str(row["intent_type"]),
                    "keywords": str(row["keywords"]),
                    "source_url": str(row.get("source_url", "")).strip(),
                    "risk_level": self._normalize_risk_level(row["risk_level"]),
                    "handoff_required": self._normalize_handoff_required(
                        row["handoff_required"]
                    ),
                }
                self.docs.append(
                    Document(page_content=str(row["embedding_text"]), metadata=metadata)
                )
        t_csv = time.perf_counter() - t0
        print(f"   ↳ FAQ CSV 로드: 문서 {len(self.docs)}건 (⏱️ {t_csv:.3f}s)")

        t0 = time.perf_counter()
        self.bm25 = BM25Retriever.from_documents(self.docs)
        self.bm25.k = 2
        t_bm25 = time.perf_counter() - t0
        print(f"   ↳ BM25 인덱스 구축 (⏱️ {t_bm25:.3f}s)")

        t0 = time.perf_counter()
        persist_path = Path(persist_dir)
        stored = _read_index_fingerprint(persist_path)
        index_matches_csv = bool(
            stored
            and stored.get("sha256") == csv_fp
            and stored.get("source_csv") == faq_csv.name
        )
        if not index_matches_csv:
            if stored:
                print(
                    "   ↳ 저장된 인덱스 지문과 FAQ CSV 불일치 — 재색인이 필요합니다."
                )
            else:
                print(
                    "   ↳ 인덱스 지문 파일 없음(구버전 포함) — 필요 시 재색인합니다."
                )

        index_name = os.environ["PINECONE_INDEX_NAME"]
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index = pc.Index(index_name)
        index_stats = index.describe_index_stats()
        total_vectors = _pinecone_total_vector_count(index_stats)
        needs_rebuild = (not index_matches_csv) or (total_vectors == 0)

        if needs_rebuild:
            if total_vectors > 0 and not index_matches_csv:
                print("   ↳ 기존 Pinecone 인덱스 데이터를 전체 삭제 후 재색인합니다...")
                index.delete(delete_all=True)
                time.sleep(2)
            print("   ↳ Pinecone Vector DB 적재 중(최초 구축 또는 재구축)...")
            self.vector_db = PineconeVectorStore.from_documents(
                documents=self.docs,
                embedding=self.embeddings,
                index_name=index_name,
            )
            _write_index_fingerprint(
                persist_path, source_name=faq_csv.name, sha256_hex=csv_fp
            )
            print(
                f"   ↳ Pinecone 적재 및 지문 저장 완료 (⏱️ {time.perf_counter() - t0:.3f}s)"
            )
        else:
            print(f"   ↳ Pinecone 기존 인덱스 연결 (벡터 {total_vectors}건)...")
            self.vector_db = PineconeVectorStore(
                index_name=index_name,
                embedding=self.embeddings,
            )
            print(f"   ↳ Pinecone 연결 완료 (⏱️ {time.perf_counter() - t0:.3f}s)")

    def _warm_up_cache(self) -> None:
        t0 = time.perf_counter()
        q = "비대면 통장 개설"
        a = "비대면 계좌 개설은 당행 모바일 앱을 통해 24시간 언제든 가능합니다."
        self.semantic_cache.append({"vector": self.embeddings.embed_query(q), "answer": a})
        t_warm = time.perf_counter() - t0
        print(f"   ↳ 시맨틱 캐시 워밍 1건 (⏱️ {t_warm:.3f}s)")

    def cosine_similarity(self, v1: Any, v2: Any) -> float:
        a = np.asarray(v1, dtype=np.float64)
        b = np.asarray(v2, dtype=np.float64)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def predict_intent(self, text: str) -> str:
        if not self.use_real_nlu or self.nlu_model is None or self.nlu_tokenizer is None:
            return "절차형"

        inputs = self.nlu_tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=128
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.nlu_model(**inputs)
            predicted_id = int(outputs.logits.argmax(dim=-1).item())
        return self.intent_map.get(predicted_id, "분류불가")

    def predict_subdomain(self, text: str) -> str | None:
        if (
            not self.use_real_subdomain_nlu
            or self.subdomain_model is None
            or self.subdomain_tokenizer is None
        ):
            return None

        inputs = self.subdomain_tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=128
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.subdomain_model(**inputs)
            predicted_id = int(outputs.logits.argmax(dim=-1).item())
        return self.subdomain_label_map.get(predicted_id, str(predicted_id))

    async def _intent_embed_parallel_async(
        self, stt_text: str
    ) -> tuple[str, str | None, Any, float, float, float, float]:
        """의도(스레드·로컬 추론)와 임베딩(스레드·네트워크 I/O)을 동시에 수행."""

        async def timed_intent() -> tuple[str, float]:
            t0 = time.perf_counter()
            out = await asyncio.to_thread(self.predict_intent, stt_text)
            return out, time.perf_counter() - t0

        async def timed_embed() -> tuple[Any, float]:
            t0 = time.perf_counter()
            out = await asyncio.to_thread(self.embeddings.embed_query, stt_text)
            return out, time.perf_counter() - t0

        async def timed_subdomain() -> tuple[str | None, float]:
            t0 = time.perf_counter()
            out = await asyncio.to_thread(self.predict_subdomain, stt_text)
            return out, time.perf_counter() - t0

        wall0 = time.perf_counter()
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

    def _intent_embed_parallel_threadpool(
        self, stt_text: str
    ) -> tuple[str, str | None, Any, float, float, float, float]:
        """이미 실행 중인 이벤트 루프 안에서는 asyncio.run 불가 → 스레드 풀로 동일 병렬화."""

        intent_sec_local = 0.0
        subdomain_sec_local = 0.0
        embed_sec_local = 0.0

        def timed_intent() -> str:
            nonlocal intent_sec_local
            t0 = time.perf_counter()
            r = self.predict_intent(stt_text)
            intent_sec_local = time.perf_counter() - t0
            return r

        def timed_embed() -> Any:
            nonlocal embed_sec_local
            t0 = time.perf_counter()
            r = self.embeddings.embed_query(stt_text)
            embed_sec_local = time.perf_counter() - t0
            return r

        def timed_subdomain() -> str | None:
            nonlocal subdomain_sec_local
            t0 = time.perf_counter()
            r = self.predict_subdomain(stt_text)
            subdomain_sec_local = time.perf_counter() - t0
            return r

        wall0 = time.perf_counter()
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

    def _run_intent_embed_parallel(
        self, stt_text: str
    ) -> tuple[str, str | None, Any, float, float, float, float]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._intent_embed_parallel_async(stt_text))
        return self._intent_embed_parallel_threadpool(stt_text)

    def process_query(self, stt_text: str, *, disable_cache: bool = False) -> dict[str, Any]:
        """STT 텍스트 → 의도·RAG·캐시 상태를 담은 dict. 워크플로우에서 분기용."""
        total_t0 = time.perf_counter()
        timings: dict[str, float | None] = {
            "intent_sec": 0.0,
            "subdomain_sec": 0.0,
            "embedding_sec": 0.0,
            "intent_embed_parallel_wall_sec": 0.0,
            "cache_check_sec": 0.0,
            "rag_search_sec": None,
            "total_sec": 0.0,
        }

        print("\n" + "=" * 72)
        print(f"📥 [process_query] 고객 발화: {stt_text!r}")

        intent, subdomain_pred, query_vector, i_sec, s_sec, e_sec, wall_pe = self._run_intent_embed_parallel(
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

        t0 = time.perf_counter()
        max_sim = 0.0
        if disable_cache:
            timings["cache_check_sec"] = time.perf_counter() - t0
            print("  🚫 [3] 평가 모드: 시맨틱 캐시 비활성화")
        else:
            for i, item in enumerate(self.semantic_cache):
                sim = self.cosine_similarity(query_vector, item["vector"])
                if sim > max_sim:
                    max_sim = sim
                print(f"     · 캐시[{i}] 유사도: {sim:.4f} (기준 ≥ {_CACHE_SIM_THRESHOLD})")
                if sim >= _CACHE_SIM_THRESHOLD:
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
                    }

            timings["cache_check_sec"] = time.perf_counter() - t0
            print(
                f"  ❄️ [3] 캐시 미적중 — 최고 유사도 {max_sim:.4f} < {_CACHE_SIM_THRESHOLD} "
                f"(⏱️ 캐시 검사 {timings['cache_check_sec']:.3f}s)"
            )

        t0 = time.perf_counter()
        vector_res_with_score = self.vector_db.similarity_search_by_vector_with_score(
            query_vector, k=self.rag_top_k
        )
        timings["rag_search_sec"] = time.perf_counter() - t0
        print(f"  🔎 [4] Pinecone 벡터 검색 (⏱️ {timings['rag_search_sec']:.3f}s)")

        retrieved_context = ""
        metadata: dict[str, Any] = {}
        retrieved_faq_ids: list[str] = []
        if vector_res_with_score:
            top_score = float(vector_res_with_score[0][1])
            if top_score < self.rag_top1_min_relevance:
                print(
                    "  ⚠️ [5] top1 연관도 부족으로 검색 결과 제외: "
                    f"top1={top_score:.4f} < min={self.rag_top1_min_relevance:.4f}"
                )
                timings["total_sec"] = time.perf_counter() - total_t0
                print(f"  ✅ 파이프라인 종료 — 총 ⏱️ {timings['total_sec']:.3f}s (LLM 단계로 전달)")
                print("=" * 72)
                return {
                    "status": "REQUIRE_LLM",
                    "intent": intent,
                    "subdomain_pred": subdomain_pred,
                    "retrieved_context": "",
                    "metadata": {},
                    "retrieved_faq_ids": [],
                    "cache_max_similarity": max_sim,
                    "timings_sec": timings,
                }
            selected: list[tuple[Any, float]] = []
            for i, (doc, score_raw) in enumerate(vector_res_with_score):
                score = float(score_raw)
                if i == 0:
                    selected.append((doc, score))
                    continue
                keep_by_ratio = score >= (top_score * self.rag_secondary_min_ratio)
                keep_by_min = score >= self.rag_min_relevance
                if keep_by_ratio and keep_by_min:
                    selected.append((doc, score))

            # 항상 최소 1개는 유지
            if not selected:
                selected = [vector_res_with_score[0]]

            # top-1 메타데이터는 기존 워크플로우 라우팅 호환을 위해 유지하고,
            # retrieved_context/faq_ids는 점수 조건을 통과한 문서만 전달한다.
            top_doc = selected[0][0]
            metadata = dict(top_doc.metadata)
            retrieved_context = "\n\n".join(doc.page_content for doc, _ in selected)
            retrieved_faq_ids = [
                str(doc.metadata.get("faq_id", "")).strip()
                for doc, _ in selected
                if str(doc.metadata.get("faq_id", "")).strip()
            ]
            all_scores = [round(float(s), 4) for _, s in vector_res_with_score]
            selected_scores = [round(float(s), 4) for _, s in selected]
            print(
                f"  📋 [5] 검색 문서 메타데이터 "
                f"(후보 {len(vector_res_with_score)}건 → 채택 {len(selected)}건):"
            )
            print(
                "      • 필터: "
                f"score >= top1*{self.rag_secondary_min_ratio:.2f} "
                f"and score >= {self.rag_min_relevance:.2f}"
            )
            print(f"      • scores(all): {all_scores}")
            print(f"      • scores(selected): {selected_scores}")
            print(f"      • faq_ids: {retrieved_faq_ids}")
            print(
                f"      • top1 domain > subdomain: {metadata.get('domain')} > {metadata.get('subdomain')}"
            )
            print(f"      • top1 intent_type: {metadata.get('intent_type')}")
            print(f"      • top1 source_url: {metadata.get('source_url')}")
            print(
                f"      • top1 risk_level / handoff: {metadata.get('risk_level')} / {metadata.get('handoff_required')}"
            )
            preview = (retrieved_context[:120] + "…") if len(retrieved_context) > 120 else retrieved_context
            print(f"      • 본문 미리보기: {preview!r}")
        else:
            print("  📋 [5] 벡터 검색 결과 없음")

        timings["total_sec"] = time.perf_counter() - total_t0
        print(f"  ✅ 파이프라인 종료 — 총 ⏱️ {timings['total_sec']:.3f}s (LLM 단계로 전달)")
        print("=" * 72)

        return {
            "status": "REQUIRE_LLM",
            "intent": intent,
            "subdomain_pred": subdomain_pred,
            "retrieved_context": retrieved_context,
            "metadata": metadata,
            "retrieved_faq_ids": retrieved_faq_ids,
            "cache_max_similarity": max_sim,
            "timings_sec": timings,
        }


if __name__ == "__main__":
    router = AICC_NLU_Router()
    demo = router.process_query("햇살론 비대면 신청되나요?")
    print("\n📦 [반환 dict 요약]")
    for k, v in demo.items():
        print(f"  • {k}: {v!r}")
