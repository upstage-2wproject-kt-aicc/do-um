"""NLU 벤치: 고정 지표(ms) + 캐시 적중률 + (선택) 라벨 CSV로 정확도·A/B 비교."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

from aicc_core_klue import AICC_NLU_Router

DEFAULT_QUERIES = [
    "햇살론 비대면 신청되나요?",
    "중도상환수수료는 왜 내야 하나요?",
    "카드 분실했는데 어떻게 해야 해요?",
    "대출 이자가 지난달보다 많이 나온 이유가 뭐죠?",
    "보이스피싱 피해가 의심되면 뭘 먼저 해야 하나요?",
    "정책서민금융은 어디서 신청하나요?",
]


def _sec_to_ms(sec: float | None) -> float:
    if sec is None:
        return 0.0
    return float(sec) * 1000.0


def _parse_gold_faq_ids(raw: Any) -> set[str]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return set()
    s = str(raw).strip().strip('"').strip("'")
    if not s:
        return set()
    parts = [p.strip() for p in s.replace(" ", "").split(",") if p.strip()]
    return set(parts)


def _norm_yn(v: Any) -> str | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    t = str(v).strip().upper()
    if t in {"Y", "YES", "TRUE", "1"}:
        return "Y"
    if t in {"N", "NO", "FALSE", "0"}:
        return "N"
    return t or None


def _load_eval_rows(csv_path: Path, limit: int) -> list[dict[str, Any]]:
    df = pd.read_csv(csv_path).fillna("")
    col_q = "user_query" if "user_query" in df.columns else None
    if col_q is None:
        raise ValueError(f"CSV에 user_query 컬럼이 필요합니다: {csv_path}")
    rows: list[dict[str, Any]] = []
    for i, r in df.iterrows():
        if len(rows) >= limit:
            break
        q = str(r[col_q]).strip()
        if not q:
            continue
        row: dict[str, Any] = {
            "query_id": r.get("query_id", i),
            "user_query": q,
            "gold_faq_ids": _parse_gold_faq_ids(r.get("gold_faq_ids"))
            if "gold_faq_ids" in df.columns
            else set(),
            "gold_handoff": _norm_yn(r.get("should_handoff"))
            if "should_handoff" in df.columns
            else None,
            "gold_subdomain": str(r["expected_subdomain"]).strip()
            if "expected_subdomain" in df.columns and str(r["expected_subdomain"]).strip()
            else None,
        }
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NLU 벤치마크 (부팅 + 질의). A/B는 --run-label 과 동일 조건으로 두 번 실행해 비교하세요."
    )
    parser.add_argument(
        "--run-label",
        default="",
        help="A/B 구분용 라벨 (예: A, B, baseline, chroma_opt). 요약·JSON에 포함됩니다.",
    )
    parser.add_argument(
        "--eval-csv",
        type=Path,
        default=None,
        help="평가용 CSV (user_query 필수, 선택: gold_faq_ids, should_handoff, expected_subdomain)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="eval-csv에서 사용할 최대 행 수 (기본 100)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="process_query 상세 로그 숨김 (대량 질의 시 권장)",
    )
    parser.add_argument(
        "--json-summary",
        type=Path,
        default=None,
        help="요약 1줄 JSON을 이 경로에 추가 기록 (append)",
    )
    args = parser.parse_args()

    if args.eval_csv is not None:
        bench_rows = _load_eval_rows(args.eval_csv.resolve(), args.limit)
        queries = [r["user_query"] for r in bench_rows]
    else:
        bench_rows = [{"user_query": q} for q in DEFAULT_QUERIES]
        queries = DEFAULT_QUERIES

    print("🚀 [벤치마크] NLU 엔진 부팅…\n")
    print(
        "💡 A/B: 코드/설정을 바꾼 뒤 동일 --run-label 규칙으로 다시 실행하고, "
        "아래 고정 지표·정확도를 나란히 비교하세요.\n"
        "💡 RAG 인덱스: FAQ CSV가 바뀌면 SHA256 지문이 달라져 Chroma가 자동 재색인됩니다.\n"
    )

    boot_t0 = time.perf_counter()
    router = AICC_NLU_Router()
    boot_sec = time.perf_counter() - boot_t0

    intent_ms_list: list[float] = []
    embed_ms_list: list[float] = []
    rag_ms_list: list[float] = []
    total_ms_list: list[float] = []
    cache_hits = 0

    faq_checked = 0
    faq_correct = 0
    handoff_checked = 0
    handoff_correct = 0
    sub_checked = 0
    sub_correct = 0

    print(f"\n⏱️ [벤치] 부팅(초기화): {boot_sec * 1000:.1f} ms")
    if router.rag_source_fingerprint_sha256:
        fp = router.rag_source_fingerprint_sha256
        print(f"📌 [벤치] RAG 소스 지문 SHA256: {fp[:16]}…\n")

    print("-" * 72)
    print(
        "고정 컬럼(행별): status | intent_ms | embed_ms | rag_ms | total_ms | "
        "(평가 CSV 시) faq_hit handoff_ok sub_ok"
    )
    print("-" * 72)

    for idx, q in enumerate(queries):
        ctx = contextlib.redirect_stdout(io.StringIO()) if args.quiet else contextlib.nullcontext()
        with ctx:
            out = router.process_query(q)
        t = out.get("timings_sec", {})
        intent_ms = _sec_to_ms(t.get("intent_sec"))
        embed_ms = _sec_to_ms(t.get("embedding_sec"))
        rag_ms = _sec_to_ms(t.get("rag_search_sec"))
        total_ms = _sec_to_ms(t.get("total_sec"))

        intent_ms_list.append(intent_ms)
        embed_ms_list.append(embed_ms)
        rag_ms_list.append(rag_ms)
        total_ms_list.append(total_ms)

        st = out.get("status", "")
        if st == "CACHED":
            cache_hits += 1

        meta = out.get("metadata") or {}
        gold = bench_rows[idx] if idx < len(bench_rows) else {}

        faq_hit_s = "-"
        ho_s = "-"
        sub_s = "-"

        gold_ids = gold.get("gold_faq_ids") or set()
        if gold_ids and st == "REQUIRE_LLM":
            faq_checked += 1
            got = str(meta.get("faq_id", "")).strip()
            ok = got in gold_ids
            if ok:
                faq_correct += 1
            faq_hit_s = "1" if ok else "0"

        gh = gold.get("gold_handoff")
        if gh is not None and st == "REQUIRE_LLM":
            pred = str(meta.get("handoff_required", "N")).strip().upper()
            handoff_checked += 1
            ok = pred == gh
            if ok:
                handoff_correct += 1
            ho_s = "1" if ok else "0"

        gs = gold.get("gold_subdomain")
        if gs and st == "REQUIRE_LLM":
            pred_sub = str(meta.get("subdomain", "")).strip()
            sub_checked += 1
            ok = pred_sub == gs
            if ok:
                sub_correct += 1
            sub_s = "1" if ok else "0"

        if not args.quiet:
            pass
        print(
            f"[{st}] intent={intent_ms:.1f} embed={embed_ms:.1f} rag={rag_ms:.1f} "
            f"total={total_ms:.1f} | faq={faq_hit_s} ho={ho_s} sub={sub_s} | {q[:56]}"
            + ("…" if len(q) > 56 else "")
        )

    n = len(queries)
    cache_rate = cache_hits / n if n else 0.0
    rag_nonzero = [x for x in rag_ms_list if x > 0]
    summary = {
        "run_label": args.run_label or None,
        "n_queries": n,
        "boot_ms": round(boot_sec * 1000, 3),
        "rag_source_sha256_prefix": (router.rag_source_fingerprint_sha256 or "")[:16],
        "mean_intent_ms": round(mean(intent_ms_list), 3) if intent_ms_list else 0.0,
        "mean_embed_ms": round(mean(embed_ms_list), 3) if embed_ms_list else 0.0,
        "mean_rag_ms": round(mean(rag_ms_list), 3) if rag_ms_list else 0.0,
        "mean_rag_ms_nonzero_only": round(mean(rag_nonzero), 3) if rag_nonzero else 0.0,
        "mean_total_ms": round(mean(total_ms_list), 3) if total_ms_list else 0.0,
        "cache_hit_rate": round(cache_rate, 6),
        "faq_acc": round(faq_correct / faq_checked, 6) if faq_checked else None,
        "faq_eval_n": faq_checked,
        "handoff_acc": round(handoff_correct / handoff_checked, 6)
        if handoff_checked
        else None,
        "handoff_eval_n": handoff_checked,
        "subdomain_acc": round(sub_correct / sub_checked, 6) if sub_checked else None,
        "subdomain_eval_n": sub_checked,
    }

    print("-" * 72)
    print("\n📊 [A/B 고정 요약] run_label=", repr(args.run_label or "(없음)"))
    print(f"• 부팅_ms: {summary['boot_ms']}")
    print(
        f"• mean intent_ms / embed_ms / rag_ms / total_ms: "
        f"{summary['mean_intent_ms']} / {summary['mean_embed_ms']} / "
        f"{summary['mean_rag_ms']} / {summary['mean_total_ms']}"
    )
    print(
        f"• mean rag_ms (RAG 호출한 행만): {summary['mean_rag_ms_nonzero_only']} "
        f"(캐시 적중 행은 rag=0으로 집계)"
    )
    print(f"• cache_hit_rate: {summary['cache_hit_rate']:.4f} ({cache_hits}/{n})")
    if faq_checked:
        print(
            f"• faq_acc (gold_faq_ids, REQUIRE_LLM만): {summary['faq_acc']} "
            f"({faq_correct}/{faq_checked})"
        )
    else:
        print("• faq_acc: (평가 CSV·gold_faq_ids 없음 또는 해당 없음)")
    if handoff_checked:
        print(
            f"• handoff_acc (메타 handoff vs should_handoff): {summary['handoff_acc']} "
            f"({handoff_correct}/{handoff_checked})"
        )
    else:
        print("• handoff_acc: (should_handoff 컬럼 없음 또는 REQUIRE_LLM 없음)")
    if sub_checked:
        print(
            f"• subdomain_acc: {summary['subdomain_acc']} "
            f"({sub_correct}/{sub_checked})"
        )
    else:
        print("• subdomain_acc: (expected_subdomain 없음)")
    print("=" * 72)

    if args.json_summary:
        line = json.dumps(summary, ensure_ascii=False)
        args.json_summary.parent.mkdir(parents=True, exist_ok=True)
        with args.json_summary.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(f"\n📝 JSON 요약 append: {args.json_summary}")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
