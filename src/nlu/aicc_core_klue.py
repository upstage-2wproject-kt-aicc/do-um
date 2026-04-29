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
        reasons: list[str] = []

        # 1) Post-RAG 메타 점수 (중복가산 상한 적용)
        meta_score_raw = 0
        if self.direct_handoff_on_high_risk and risk_level in {"high", "critical"}:
            meta_score_raw += self.guardrail_score_meta_high
            reasons.append(f"meta_risk_high(+{self.guardrail_score_meta_high})")
        if self.direct_handoff_on_required and handoff_required == "Y":
            meta_score_raw += self.guardrail_score_meta_required
            reasons.append(f"meta_handoff_required(+{self.guardrail_score_meta_required})")
        meta_score = min(meta_score_raw, self.guardrail_meta_cap)
        if meta_score_raw > self.guardrail_meta_cap:
            reasons.append(f"meta_cap_applied({meta_score_raw}->{meta_score})")

        # 2) Pre-RAG 키워드 점수 (욕설 우선)
        keyword_score = 0
        if self.guardrail_enable_keyword:
            if abusive_keyword_hit:
                keyword_score = self.guardrail_score_keyword_abusive
                reasons.append(
                    f"keyword_abusive:{abusive_keyword_hit}(+{self.guardrail_score_keyword_abusive})"
                )
            elif sensitive_keyword_hit:
                keyword_score = self.guardrail_score_keyword_sensitive
                reasons.append(
                    f"keyword_sensitive:{sensitive_keyword_hit}(+{self.guardrail_score_keyword_sensitive})"
                )

        # 3) 고객정보 부재 점수
        missing_ctx_score = 0
        if missing_customer_context_reason:
            missing_ctx_score = self.guardrail_score_missing_customer_context
            reasons.append(
                f"{missing_customer_context_reason}(+{self.guardrail_score_missing_customer_context})"
            )

        # 4) OOD/RAG 미적중 점수
        ood_score = 0
        if rag_miss:
            ood_score = self.guardrail_score_ood
            reasons.append(f"rag_miss(+{self.guardrail_score_ood})")

        total_score = meta_score + keyword_score + missing_ctx_score + ood_score
        finance_related = self._contains_finance_keyword(query_text)

        decision = "ALLOW"
        if total_score >= self.guardrail_handoff_threshold:
            decision = "HANDOFF"
        elif (
            self.guardrail_enable_ood_reject
            and rag_miss
            and not finance_related
            and total_score >= self.guardrail_reject_threshold
        ):
            decision = "REJECT"
        elif total_score >= self.guardrail_limit_threshold:
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
        return {
            "status": "HANDOFF_DIRECT",
            "intent": intent,
            "subdomain_pred": subdomain_pred,
            "final_answer": self.direct_handoff_message,
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
        return {
            "status": "REJECT_DIRECT",
            "intent": intent,
            "subdomain_pred": subdomain_pred,
            "final_answer": self.guardrail_reject_message,
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

    def _vector_scores_by_faq(
        self, vector_res_with_score: list[tuple[Any, Any]]
    ) -> dict[str, float]:
        out: dict[str, float] = {}
        for doc, score_raw in vector_res_with_score:
            fid = str(doc.metadata.get("faq_id", "")).strip()
            if fid:
                out[fid] = float(score_raw)
        return out

    def _rrf_ranked_faq_ids(
        self, ranked_id_lists: list[list[str]], *, rrf_k: int
    ) -> list[tuple[str, float]]:
        scores: dict[str, float] = {}
        for id_list in ranked_id_lists:
            for rank, fid in enumerate(id_list):
                if not fid:
                    continue
                scores[fid] = scores.get(fid, 0.0) + 1.0 / (rrf_k + rank + 1)
        return sorted(scores.items(), key=lambda x: -x[1])

    def _doc_by_faq_prefer_vector(
        self,
        vector_pairs: list[tuple[Any, Any]],
        bm25_docs: list[Any],
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for doc, _ in vector_pairs:
            fid = str(doc.metadata.get("faq_id", "")).strip()
            if fid:
                out.setdefault(fid, doc)
        for doc in bm25_docs:
            fid = str(doc.metadata.get("faq_id", "")).strip()
            if fid and fid not in out:
                out[fid] = doc
        return out

    def _bm25_get_docs(self, query: str, k: int) -> tuple[list[Any], float]:
        prev_k = self.bm25.k
        self.bm25.k = max(1, k)
        t0 = time.perf_counter()
        try:
            # LangChain BaseRetriever는 Runnable API → invoke(쿼리)가 표준
            docs = self.bm25.invoke(query)
        finally:
            self.bm25.k = prev_k
        return docs, time.perf_counter() - t0

    def _vector_only_select(
        self, vector_res_with_score: list[tuple[Any, Any]]
    ) -> list[tuple[Any, float]]:
        if not vector_res_with_score:
            return []
        top_score = float(vector_res_with_score[0][1])
        if top_score < self.rag_top1_min_relevance:
            return []
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
        if not selected:
            selected = [(vector_res_with_score[0][0], top_score)]
        return selected

    def _hybrid_rrf_select(
        self,
        vector_res_with_score: list[tuple[Any, Any]],
        bm25_docs: list[Any],
    ) -> list[tuple[Any, float]]:
        if not vector_res_with_score and not bm25_docs:
            return []
        v_scores = self._vector_scores_by_faq(vector_res_with_score)
        v_ids = [
            str(d.metadata.get("faq_id", "")).strip()
            for d, _ in vector_res_with_score
            if str(d.metadata.get("faq_id", "")).strip()
        ]
        b_ids = [
            str(d.metadata.get("faq_id", "")).strip()
            for d in bm25_docs
            if str(d.metadata.get("faq_id", "")).strip()
        ]
        rrf_ranked = self._rrf_ranked_faq_ids([v_ids, b_ids], rrf_k=self.rrf_k)
        faq_to_doc = self._doc_by_faq_prefer_vector(vector_res_with_score, bm25_docs)
        fused: list[tuple[Any, float, float]] = []
        for fid, rrf_s in rrf_ranked:
            if fid not in faq_to_doc:
                continue
            vec_s = v_scores.get(fid, float("nan"))
            fused.append((faq_to_doc[fid], vec_s, rrf_s))
        if not fused:
            return []

        top_doc, top_vec, top_rrf = fused[0]
        top_fid = str(top_doc.metadata.get("faq_id", "")).strip()

        if not math.isnan(top_vec):
            if top_vec < self.rag_top1_min_relevance:
                return []
        else:
            if top_fid not in set(b_ids[:2]):
                return []

        selected: list[tuple[Any, float]] = []
        top_vec_ref = top_vec if not math.isnan(top_vec) else None
        selected.append(
            (top_doc, float(top_vec) if top_vec_ref is not None else 0.0),
        )

        for doc, vec_s, rrf_s in fused[1:]:
            if len(selected) >= self.rag_top_k:
                break
            if top_vec_ref is not None and not math.isnan(vec_s):
                if vec_s >= top_vec_ref * self.rag_secondary_min_ratio and vec_s >= self.rag_min_relevance:
                    selected.append((doc, float(vec_s)))
            elif top_vec_ref is not None and math.isnan(vec_s):
                if rrf_s >= top_rrf * self.rag_secondary_min_ratio:
                    selected.append((doc, 0.0))
            else:
                if rrf_s >= top_rrf * self.rag_secondary_min_ratio:
                    selected.append((doc, float(vec_s) if not math.isnan(vec_s) else 0.0))
        return selected

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
        if "token_type_ids" in inputs:
            # 일부 커스텀 토크나이저는 비정상 segment id를 내보내므로 0으로 고정
            inputs["token_type_ids"] = torch.zeros_like(inputs["token_type_ids"])
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
        if "token_type_ids" in inputs:
            inputs["token_type_ids"] = torch.zeros_like(inputs["token_type_ids"])
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
        if self.subdomain_source == "rag":
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
        if self.subdomain_source == "rag":
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

    def _run_intent_embed_parallel(
        self, stt_text: str
    ) -> tuple[str, str | None, Any, float, float, float, float]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._intent_embed_parallel_async(stt_text))
        return self._intent_embed_parallel_threadpool(stt_text)

    def process_query(
        self,
        stt_text: str,
        *,
        disable_cache: bool = False,
        customer_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """STT 텍스트 → 의도·RAG·캐시 상태를 담은 dict. 워크플로우에서 분기용."""
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

        k_vec = (
            max(self.rag_top_k * self.rag_fusion_pool_mult, self.rag_top_k)
            if self.rag_hybrid
            else self.rag_top_k
        )
        t0 = time.perf_counter()
        vector_res_with_score = self.vector_db.similarity_search_by_vector_with_score(
            query_vector, k=k_vec
        )
        vec_sec = time.perf_counter() - t0
        bm25_docs: list[Any] = []
        bm25_sec = 0.0
        if self.rag_hybrid:
            t0 = time.perf_counter()
            bm25_docs, bm25_sec = self._bm25_get_docs(stt_text, k_vec)

        timings["rag_vector_sec"] = round(vec_sec, 6)
        timings["rag_bm25_sec"] = round(bm25_sec, 6) if self.rag_hybrid else None
        timings["rag_search_sec"] = vec_sec + bm25_sec
        print(
            f"  🔎 [4] RAG: Pinecone 벡터 k={k_vec} (⏱️ {vec_sec:.3f}s)"
            + (
                f" + BM25 k={k_vec} (⏱️ {bm25_sec:.3f}s) → hybrid=RRF"
                if self.rag_hybrid
                else ""
            )
        )

        retrieved_context = ""
        metadata: dict[str, Any] = {}
        retrieved_faq_ids: list[str] = []
        routing_signals: dict[str, Any] = {}

        if self.rag_hybrid and bm25_docs:
            selected = self._hybrid_rrf_select(vector_res_with_score, bm25_docs)
            fusion_note = "hybrid_rrf_bm25_vector"
        else:
            selected = self._vector_only_select(vector_res_with_score)
            fusion_note = "vector_only"

        sensitive_keyword_hit = self._find_direct_handoff_keyword(stt_text)
        abusive_keyword_hit = self._find_abusive_keyword(stt_text)
        context_required_keyword = self._find_customer_context_required_keyword(stt_text)
        customer_context_present = self._has_customer_context(customer_context)
        missing_customer_context_reason: str | None = None
        if (
            self.direct_handoff_on_missing_customer_context
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
            guardrail = self._compute_guardrail(
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
                return self._build_direct_handoff_response(
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
                return self._build_reject_response(
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
            policy_rules_miss: list[dict[str, Any]] = []
            if guardrail["decision"] == "LIMIT":
                policy_rules_miss = [
                    {
                        "rule_id": "guardrail_limit",
                        "title": "안전 축소 답변",
                        "description": self.guardrail_limit_message,
                    }
                ]
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
        agg_risk, agg_ho = self._aggregate_risk_handoff([d for d, _ in selected])
        metadata = meta_top
        metadata["risk_level"] = agg_risk
        metadata["handoff_required"] = agg_ho
        metadata["rag_fusion"] = fusion_note

        if self.subdomain_source == "rag":
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
            f"{', BM25 ' + str(len(bm25_docs)) + '건' if self.rag_hybrid and bm25_docs else ''}"
            f" → 채택 {len(selected)}건):"
        )
        print(
            "      • 2차 필터: 벡터 score 기준 "
            f"score >= top1*{self.rag_secondary_min_ratio:.2f} "
            f"and score >= {self.rag_min_relevance:.2f} "
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

        # RAG hit(selected) 경로에서도 guardrail을 계산해야 REJECT/LIMIT/REQUIRE_LLM 분기에서 안전하게 사용할 수 있다.
        guardrail = self._compute_guardrail(
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
            return self._build_direct_handoff_response(
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
            return self._build_reject_response(
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
        policy_rules: list[dict[str, Any]] = []
        if guardrail["decision"] == "LIMIT":
            policy_rules = [
                {
                    "rule_id": "guardrail_limit",
                    "title": "안전 축소 답변",
                    "description": self.guardrail_limit_message,
                }
            ]

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


if __name__ == "__main__":
    router = AICC_NLU_Router()
    demo = router.process_query("햇살론 비대면 신청되나요?")
    print("\n📦 [반환 dict 요약]")
    for k, v in demo.items():
        print(f"  • {k}: {v!r}")
