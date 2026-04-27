"""NLU + RAG 라우터: 의도·검색·캐시 결과를 dict로 반환해 워크플로우 단계에서 재사용합니다."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import numpy as np
import pandas as pd
import torch
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_upstage import UpstageEmbeddings
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


def _chroma_sqlite_exists(persist_path: Path) -> bool:
    """로컬 Chroma persist 디렉터리에 SQLite 스토어가 있는지(이미 구축됐는지) 확인."""
    return persist_path.is_dir() and (persist_path / "chroma.sqlite3").is_file()


class AICC_NLU_Router:
    """KLUE 기반 의도분류 + BM25/Chroma RAG + 시맨틱 캐시."""

    def __init__(self) -> None:
        if not os.environ.get("LLM_SOLAR_API_KEY"):
            raise RuntimeError(
                "LLM_SOLAR_API_KEY가 없습니다. 저장소 루트에 .env를 두고 "
                "LLM_SOLAR_API_KEY=... 를 설정하거나, .env.example을 복사해 채워 넣으세요."
            )
        os.environ.setdefault("UPSTAGE_API_KEY", os.environ["LLM_SOLAR_API_KEY"])

        boot_t0 = time.perf_counter()
        print("\n🚀 [NLU Router] KLUE 로컬 모델 부팅 중...")
        self.model_path = _SCRIPT_DIR / "my_aicc_nlu_model_klue"
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
            print(f"  ✅ KLUE 모델 로드 성공 (⏱️ {t_nlu_load:.3f}s)")
            self.intent_map = {0: "절차형", 1: "민원형", 2: "조회형"}
        except Exception as e:
            t_nlu_load = time.perf_counter() - t0
            print(f"  ⚠️ 모델 로드 실패, 임시 모드 (⏱️ {t_nlu_load:.3f}s): {e}")
            self.use_real_nlu = False
            self.nlu_tokenizer = None
            self.nlu_model = None
            self.intent_map = {}

        t0 = time.perf_counter()
        self.embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
        t_emb_factory = time.perf_counter() - t0
        print(f"  📎 Upstage Embeddings 준비 (⏱️ {t_emb_factory:.3f}s)")

        self.semantic_cache: list[dict[str, Any]] = []
        self.rag_source_fingerprint_sha256: str = ""

        self._prepare_datasets()
        self._warm_up_cache()

        print(
            f"✅ [NLU Router] 부팅 완료 — 총 소요 ⏱️ {time.perf_counter() - boot_t0:.3f}s\n"
        )

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
                    "   ↳ 저장된 Chroma 지문과 FAQ CSV 불일치 — 재색인이 필요합니다."
                )
            else:
                print(
                    "   ↳ Chroma 지문 파일 없음(구버전 포함) — 필요 시 재색인합니다."
                )

        loaded = False
        if index_matches_csv and _chroma_sqlite_exists(persist_path):
            try:
                print("   ↳ Chroma: 동일 지문 확인, persist에서 로드 시도...")
                candidate = Chroma(
                    persist_directory=persist_dir,
                    embedding_function=self.embeddings,
                )
                probe = candidate.get(limit=1, include=[])
                if probe.get("ids"):
                    self.vector_db = candidate
                    loaded = True
                    print(
                        f"   ↳ Chroma 로드 완료 (재임베딩 없음, ⏱️ {time.perf_counter() - t0:.3f}s)"
                    )
                else:
                    print("   ↳ Chroma persist는 있으나 문서가 없어 재구축합니다.")
            except Exception as e:
                print(f"   ⚠️ Chroma 로드 실패, 재구축합니다: {e}")

        if not loaded:
            if persist_path.is_dir():
                print("   ↳ 기존 Chroma persist 디렉터리를 비우고 재구축합니다…")
                shutil.rmtree(persist_path)
            print("   ↳ Chroma Vector DB 적재 중(최초 구축 또는 재구축)...")
            self.vector_db = Chroma.from_documents(
                documents=self.docs,
                embedding=self.embeddings,
                persist_directory=persist_dir,
            )
            _write_index_fingerprint(
                persist_path, source_name=faq_csv.name, sha256_hex=csv_fp
            )
            print(f"   ↳ Chroma 적재 및 지문 저장 완료 (⏱️ {time.perf_counter() - t0:.3f}s)")

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

    def process_query(self, stt_text: str) -> dict[str, Any]:
        """STT 텍스트 → 의도·RAG·캐시 상태를 담은 dict. 워크플로우에서 분기용."""
        total_t0 = time.perf_counter()
        timings: dict[str, float | None] = {
            "intent_sec": 0.0,
            "embedding_sec": 0.0,
            "cache_check_sec": 0.0,
            "rag_search_sec": None,
            "total_sec": 0.0,
        }

        print("\n" + "=" * 72)
        print(f"📥 [process_query] 고객 발화: {stt_text!r}")

        t0 = time.perf_counter()
        intent = self.predict_intent(stt_text)
        timings["intent_sec"] = time.perf_counter() - t0
        print(f"  🧠 [1] NLU 의도: {intent!r} (⏱️ {timings['intent_sec']:.3f}s)")

        t0 = time.perf_counter()
        query_vector = self.embeddings.embed_query(stt_text)
        timings["embedding_sec"] = time.perf_counter() - t0
        print(f"  📐 [2] 질의 임베딩 완료 (⏱️ {timings['embedding_sec']:.3f}s)")

        t0 = time.perf_counter()
        max_sim = 0.0
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
        vector_res = self.vector_db.similarity_search_by_vector(query_vector, k=1)
        timings["rag_search_sec"] = time.perf_counter() - t0
        print(f"  🔎 [4] Chroma 벡터 검색 (⏱️ {timings['rag_search_sec']:.3f}s)")

        retrieved_context = ""
        metadata: dict[str, Any] = {}
        if vector_res:
            res_doc = vector_res[0]
            retrieved_context = res_doc.page_content
            metadata = dict(res_doc.metadata)
            print("  📋 [5] 검색 문서 메타데이터:")
            print(f"      • faq_id: {metadata.get('faq_id')}")
            print(f"      • domain > subdomain: {metadata.get('domain')} > {metadata.get('subdomain')}")
            print(f"      • intent_type: {metadata.get('intent_type')}")
            print(f"      • source_url: {metadata.get('source_url')}")
            print(f"      • risk_level / handoff: {metadata.get('risk_level')} / {metadata.get('handoff_required')}")
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
            "query_vector": list(map(float, query_vector)),
            "retrieved_context": retrieved_context,
            "metadata": metadata,
            "cache_max_similarity": max_sim,
            "timings_sec": timings,
        }


if __name__ == "__main__":
    router = AICC_NLU_Router()
    demo = router.process_query("햇살론 비대면 신청되나요?")
    print("\n📦 [반환 dict 요약]")
    for k, v in demo.items():
        if k == "query_vector":
            print(f"  • {k}: <벡터 길이 {len(v)}>")
        else:
            print(f"  • {k}: {v!r}")
