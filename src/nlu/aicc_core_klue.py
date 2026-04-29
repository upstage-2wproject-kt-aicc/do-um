"""NLU + RAG 라우터: 의도·검색·캐시 결과를 dict로 반환해 워크플로우 단계에서 재사용합니다."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
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
from transformers import BertConfig, BertForSequenceClassification, PreTrainedTokenizerFast
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from transformers.models.auto.tokenization_auto import AutoTokenizer
from src.nlu.guardrail.policy import build_policy_rules
from src.nlu.guardrail.scorer import compute_guardrail
from src.nlu.intent.classifier import (
    intent_embed_parallel_async,
    intent_embed_parallel_threadpool,
    predict_intent as intent_predict_intent,
    predict_subdomain as intent_predict_subdomain,
    run_intent_embed_parallel,
)
from src.nlu.response.builder import build_direct_handoff_response, build_reject_response
from src.nlu.retrieval.index_manager import bm25_get_docs, prepare_datasets, warm_up_cache
from src.nlu.retrieval.selector import hybrid_rrf_select, vector_only_select
from src.nlu.retrieval.vector_store import calc_rag_vector_k
from src.nlu.service import process_nlu_query

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

    def __init__(
        self,
        *,
        intent_model_dir: str | Path | None = None,
        subdomain_model_dir: str | Path | None = None,
    ) -> None:
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
            pt_loaded = self._load_pt_intent_checkpoint(self.model_path)
            if pt_loaded is not None:
                self.nlu_tokenizer, self.nlu_model, self.intent_map = pt_loaded
                self.nlu_model.to(self.device)
                self.nlu_model.eval()
                self.use_real_nlu = True
                t_nlu_load = time.perf_counter() - t0
                print(
                    "  ✅ Intent .pt 체크포인트 로드 성공: "
                    f"{self.model_path} (⏱️ {t_nlu_load:.3f}s)"
                )
            else:
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
        self.rag_hybrid: bool = os.getenv("NLU_RAG_HYBRID", "1").strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )
        self.rrf_k: int = max(1, int(os.getenv("NLU_RAG_RRF_K", "60")))
        self.rag_fusion_pool_mult: int = max(1, int(os.getenv("NLU_RAG_FUSION_POOL_MULT", "2")))
        self.subdomain_source: str = os.getenv("NLU_SUBDOMAIN_SOURCE", "rag").strip().lower()
        self.direct_handoff_on_high_risk: bool = os.getenv(
            "NLU_DIRECT_HANDOFF_ON_HIGH_RISK", "1"
        ).strip().lower() not in ("0", "false", "no", "off")
        self.direct_handoff_on_required: bool = os.getenv(
            "NLU_DIRECT_HANDOFF_ON_REQUIRED", "1"
        ).strip().lower() not in ("0", "false", "no", "off")
        raw_keywords = os.getenv(
            "NLU_GUARDRAIL_SENSITIVE_KEYWORDS",
            os.getenv(
                "NLU_DIRECT_HANDOFF_KEYWORDS",
                "보이스피싱,피싱,사기,명의도용,해킹,도난,분실,신고,긴급,112",
            ),
        ).strip()
        self.direct_handoff_keywords: tuple[str, ...] = tuple(
            item.strip().lower() for item in raw_keywords.split(",") if item.strip()
        )
        self.direct_handoff_message: str = os.getenv(
            "NLU_DIRECT_HANDOFF_MESSAGE",
            "해당 문의는 개인정보 확인 또는 안전 조치가 필요하여 상담사에게 연결해 드리겠습니다.",
        ).strip()
        self.guardrail_limit_message: str = os.getenv(
            "NLU_GUARDRAIL_LIMIT_MESSAGE",
            "개인별 조건 확인이 필요한 문의일 수 있어 확정 답변을 피하고, 약관/공식 채널 확인을 안내하세요.",
        ).strip()
        self.guardrail_reject_message: str = os.getenv(
            "NLU_GUARDRAIL_REJECT_MESSAGE",
            "금융 관련 문의에 한해 답변드릴 수 있습니다. 금융 상담 질문으로 다시 요청해 주세요.",
        ).strip()
        self.guardrail_enable_keyword: bool = os.getenv(
            "NLU_GUARDRAIL_ENABLE_KEYWORD", "1"
        ).strip().lower() not in ("0", "false", "no", "off")
        self.guardrail_enable_ood_reject: bool = os.getenv(
            "NLU_GUARDRAIL_ENABLE_OOD_REJECT", "1"
        ).strip().lower() not in ("0", "false", "no", "off")
        self.guardrail_score_meta_high: int = int(os.getenv("NLU_GUARDRAIL_SCORE_META_HIGH", "60"))
        self.guardrail_score_meta_required: int = int(
            os.getenv("NLU_GUARDRAIL_SCORE_META_REQUIRED", "40")
        )
        self.guardrail_score_keyword_sensitive: int = int(
            os.getenv("NLU_GUARDRAIL_SCORE_KEYWORD_SENSITIVE", "50")
        )
        self.guardrail_score_keyword_abusive: int = int(
            os.getenv("NLU_GUARDRAIL_SCORE_KEYWORD_ABUSIVE", "100")
        )
        self.guardrail_score_missing_customer_context: int = int(
            os.getenv("NLU_GUARDRAIL_SCORE_MISSING_CUSTOMER_CONTEXT", "40")
        )
        self.guardrail_score_ood: int = int(os.getenv("NLU_GUARDRAIL_SCORE_OOD", "30"))
        self.guardrail_meta_cap: int = int(os.getenv("NLU_GUARDRAIL_META_CAP", "70"))
        self.guardrail_handoff_threshold: int = int(
            os.getenv("NLU_GUARDRAIL_HANDOFF_THRESHOLD", "80")
        )
        self.guardrail_limit_threshold: int = int(
            os.getenv("NLU_GUARDRAIL_LIMIT_THRESHOLD", "50")
        )
        self.guardrail_reject_threshold: int = int(
            os.getenv("NLU_GUARDRAIL_REJECT_THRESHOLD", "30")
        )
        raw_abusive = os.getenv(
            "NLU_GUARDRAIL_ABUSIVE_KEYWORDS",
            "씨발,병신,개새끼,좆같,죽여버리,살인,테러",
        ).strip()
        self.guardrail_abusive_keywords: tuple[str, ...] = tuple(
            item.strip().lower() for item in raw_abusive.split(",") if item.strip()
        )
        raw_finance = os.getenv(
            "NLU_FINANCE_DOMAIN_KEYWORDS",
            "금융,대출,금리,이자,카드,계좌,예금,적금,환불,수수료,보이스피싱,명의도용,상담",
        ).strip()
        self.finance_domain_keywords: tuple[str, ...] = tuple(
            item.strip().lower() for item in raw_finance.split(",") if item.strip()
        )
        self.direct_handoff_on_missing_customer_context: bool = os.getenv(
            "NLU_DIRECT_HANDOFF_ON_MISSING_CUSTOMER_CONTEXT", "1"
        ).strip().lower() not in ("0", "false", "no", "off")
        raw_ctx_keywords = os.getenv(
            "NLU_CUSTOMER_CONTEXT_REQUIRED_KEYWORDS",
            "내 명의,내 계좌,내 카드,내 대출,거래내역,조회해,확인해,환불,승인 결과,한도 조회",
        ).strip()
        self.customer_context_required_keywords: tuple[str, ...] = tuple(
            item.strip().lower() for item in raw_ctx_keywords.split(",") if item.strip()
        )
        print(f"  🔢 RAG top-k: {self.rag_top_k}")
        print(
            f"  🔀 RAG hybrid(BM25+벡터 RRF): {'ON' if self.rag_hybrid else 'OFF'} "
            f"(RRF_K={self.rrf_k}, pool×{self.rag_fusion_pool_mult})"
        )
        print(f"  🏷️ 서브도메인(주제) 출처: {self.subdomain_source} (rag=검색 상위 메타, model=KLUE 서브모델)")
        print(
            "  🚨 Direct handoff: "
            f"high_risk={'ON' if self.direct_handoff_on_high_risk else 'OFF'}, "
            f"required={'ON' if self.direct_handoff_on_required else 'OFF'}, "
            f"risk_keywords={len(self.direct_handoff_keywords)}개, "
            f"missing_ctx={'ON' if self.direct_handoff_on_missing_customer_context else 'OFF'}"
        )
        print(
            "  🛡️ Guardrail: "
            f"handoff>={self.guardrail_handoff_threshold}, "
            f"limit>={self.guardrail_limit_threshold}, "
            f"reject>={self.guardrail_reject_threshold}, "
            f"ood_reject={'ON' if self.guardrail_enable_ood_reject else 'OFF'}"
        )

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
                    idx = int(k)
                    label = str(v).strip()
                    # HuggingFace 기본 LABEL_0/1/2 형식은 평가 라벨과 맞지 않으므로
                    # fallback(또는 env 라벨)로 한글 라벨을 우선 치환한다.
                    if label.upper().startswith("LABEL_"):
                        out[idx] = fallback.get(idx, label)
                    else:
                        out[idx] = label
                except (TypeError, ValueError):
                    continue
            if out:
                return out
        return fallback.copy()

    @staticmethod
    def _load_pt_intent_checkpoint(model_dir: Path) -> tuple[Any, Any, dict[int, str]] | None:
        """커스텀 학습(.pt) 포맷 Intent 체크포인트를 로드."""
        ckpt_path = model_dir / "intent_model.pt"
        labels_path = model_dir / "label_maps.json"
        tokenizer_path = model_dir / "tokenizer.json"
        if not (ckpt_path.is_file() and labels_path.is_file() and tokenizer_path.is_file()):
            return None

        payload = json.loads(labels_path.read_text(encoding="utf-8"))
        intent_classes_raw = payload.get("intent_classes", [])
        if not isinstance(intent_classes_raw, list) or not intent_classes_raw:
            return None
        intent_classes = [str(x).strip() for x in intent_classes_raw if str(x).strip()]
        if not intent_classes:
            return None
        intent_map = {i: label for i, label in enumerate(intent_classes)}

        state = torch.load(ckpt_path, map_location="cpu")
        if not isinstance(state, dict) or "bert.embeddings.word_embeddings.weight" not in state:
            return None

        vocab_size = int(state["bert.embeddings.word_embeddings.weight"].shape[0])
        hidden_size = int(state["bert.embeddings.word_embeddings.weight"].shape[1])
        max_position_embeddings = int(state["bert.embeddings.position_embeddings.weight"].shape[0])
        type_vocab_size = int(state["bert.embeddings.token_type_embeddings.weight"].shape[0])
        layer_indices = [
            int(k.split(".")[3])
            for k in state
            if k.startswith("bert.encoder.layer.") and k.split(".")[3].isdigit()
        ]
        if not layer_indices:
            return None
        num_hidden_layers = max(layer_indices) + 1
        intermediate_size = int(state["bert.encoder.layer.0.intermediate.dense.weight"].shape[0])
        num_attention_heads = max(1, hidden_size // 64)

        cfg = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            num_labels=len(intent_classes),
            id2label={i: label for i, label in intent_map.items()},
            label2id={label: i for i, label in intent_map.items()},
        )
        model = BertForSequenceClassification(cfg)
        model.load_state_dict(state, strict=True)
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(tokenizer_path),
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]",
        )
        return tokenizer, model, intent_map

    @staticmethod
    def _risk_rank_value(level: str) -> int:
        normalized = str(level).strip().lower()
        if normalized in {"high", "높음", "상", "critical", "crit"}:
            return 3
        if normalized in {"medium", "보통", "중간", "mid"}:
            return 2
        return 1

    def _aggregate_risk_handoff(self, docs: list[Any]) -> tuple[str, str]:
        """후보 문서들에서 최고 리스크·이관 필요 여부를 집계(방법 5)."""
        if not docs:
            return "low", "N"
        risks: list[str] = []
        hands: list[str] = []
        for d in docs:
            meta = getattr(d, "metadata", None) or {}
            risks.append(self._normalize_risk_level(meta.get("risk_level", "low")))
            hands.append(self._normalize_handoff_required(meta.get("handoff_required", "N")))
        best = max(risks, key=self._risk_rank_value)
        any_ho = "Y" if any(h == "Y" for h in hands) else "N"
        return best, any_ho

    def _find_direct_handoff_keyword(self, text: str) -> str | None:
        q = text.strip().lower()
        if not q:
            return None
        for keyword in self.direct_handoff_keywords:
            if keyword and keyword in q:
                return keyword
        return None

    def _has_customer_context(self, customer_context: dict[str, Any] | None) -> bool:
        if not isinstance(customer_context, dict):
            return False
        for v in customer_context.values():
            if isinstance(v, str) and v.strip():
                return True
            if isinstance(v, (int, float, bool)):
                return True
        return False

    def _find_customer_context_required_keyword(self, text: str) -> str | None:
        q = text.strip().lower()
        if not q:
            return None
        for keyword in self.customer_context_required_keywords:
            if keyword and keyword in q:
                return keyword
        return None

    def _contains_finance_keyword(self, text: str) -> bool:
        q = text.strip().lower()
        if not q:
            return False
        return any(kw in q for kw in self.finance_domain_keywords if kw)

    def _find_abusive_keyword(self, text: str) -> str | None:
        q = text.strip().lower()
        if not q:
            return None
        for keyword in self.guardrail_abusive_keywords:
            if keyword and keyword in q:
                return keyword
        return None

    def _compute_guardrail(
        self,
        *,
        risk_level: str,
        handoff_required: str,
        sensitive_keyword_hit: str | None,
        abusive_keyword_hit: str | None,
        missing_customer_context_reason: str | None,
        rag_miss: bool,
        query_text: str,
    ) -> dict[str, Any]:
        return compute_guardrail(
            self,
            risk_level=risk_level,
            handoff_required=handoff_required,
            sensitive_keyword_hit=sensitive_keyword_hit,
            abusive_keyword_hit=abusive_keyword_hit,
            missing_customer_context_reason=missing_customer_context_reason,
            rag_miss=rag_miss,
            query_text=query_text,
        )

    def _build_direct_handoff_response(
        self,
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
        return build_direct_handoff_response(
            self,
            intent=intent,
            subdomain_pred=subdomain_pred,
            metadata=metadata,
            retrieved_context=retrieved_context,
            retrieved_faq_ids=retrieved_faq_ids,
            routing_signals=routing_signals,
            cache_max_similarity=cache_max_similarity,
            timings=timings,
            reasons=reasons,
            guardrail_score=guardrail_score,
            guardrail_components=guardrail_components,
        )

    def _build_reject_response(
        self,
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
        return build_reject_response(
            self,
            intent=intent,
            subdomain_pred=subdomain_pred,
            metadata=metadata,
            retrieved_context=retrieved_context,
            retrieved_faq_ids=retrieved_faq_ids,
            routing_signals=routing_signals,
            cache_max_similarity=cache_max_similarity,
            timings=timings,
            reasons=reasons,
            guardrail_score=guardrail_score,
            guardrail_components=guardrail_components,
        )

    def _bm25_get_docs(self, query: str, k: int) -> tuple[list[Any], float]:
        return bm25_get_docs(self, query, k)

    def _vector_only_select(
        self, vector_res_with_score: list[tuple[Any, Any]]
    ) -> list[tuple[Any, float]]:
        return vector_only_select(self, vector_res_with_score)

    def _hybrid_rrf_select(
        self,
        vector_res_with_score: list[tuple[Any, Any]],
        bm25_docs: list[Any],
    ) -> list[tuple[Any, float]]:
        return hybrid_rrf_select(self, vector_res_with_score, bm25_docs)

    def _prepare_datasets(self) -> None:
        prepare_datasets(
            self,
            faq_csv=faq_csv,
            persist_dir=persist_dir,
            sha256_file=_sha256_file,
            read_index_fingerprint=_read_index_fingerprint,
            write_index_fingerprint=_write_index_fingerprint,
            pinecone_total_vector_count=_pinecone_total_vector_count,
        )

    def _warm_up_cache(self) -> None:
        warm_up_cache(self)

    def cosine_similarity(self, v1: Any, v2: Any) -> float:
        a = np.asarray(v1, dtype=np.float64)
        b = np.asarray(v2, dtype=np.float64)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def predict_intent(self, text: str) -> str:
        return intent_predict_intent(self, text)

    def predict_subdomain(self, text: str) -> str | None:
        return intent_predict_subdomain(self, text)

    async def _intent_embed_parallel_async(
        self, stt_text: str
    ) -> tuple[str, str | None, Any, float, float, float, float]:
        return await intent_embed_parallel_async(self, stt_text)

    def _intent_embed_parallel_threadpool(
        self, stt_text: str
    ) -> tuple[str, str | None, Any, float, float, float, float]:
        return intent_embed_parallel_threadpool(self, stt_text)

    def _run_intent_embed_parallel(
        self, stt_text: str
    ) -> tuple[str, str | None, Any, float, float, float, float]:
        return run_intent_embed_parallel(self, stt_text)

    def process_query(
        self,
        stt_text: str,
        *,
        disable_cache: bool = False,
        customer_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """STT 텍스트 → 의도·RAG·캐시 상태를 담은 dict. 워크플로우에서 분기용."""
        return process_nlu_query(
            self,
            stt_text,
            disable_cache=disable_cache,
            customer_context=customer_context,
        )


if __name__ == "__main__":
    router = AICC_NLU_Router()
    demo = router.process_query("햇살론 비대면 신청되나요?")
    print("\n📦 [반환 dict 요약]")
    for k, v in demo.items():
        print(f"  • {k}: {v!r}")
