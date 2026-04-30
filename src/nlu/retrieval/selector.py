"""RAG 검색 결과 선택 로직."""

from __future__ import annotations

import math
from typing import Any


def _vector_scores_by_faq(vector_res_with_score: list[tuple[Any, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for doc, score_raw in vector_res_with_score:
        fid = str(doc.metadata.get("faq_id", "")).strip()
        if fid:
            out[fid] = float(score_raw)
    return out


def _rrf_ranked_faq_ids(ranked_id_lists: list[list[str]], *, rrf_k: int) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for id_list in ranked_id_lists:
        for rank, fid in enumerate(id_list):
            if not fid:
                continue
            scores[fid] = scores.get(fid, 0.0) + 1.0 / (rrf_k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


def _doc_by_faq_prefer_vector(
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


def vector_only_select(router: Any, vector_res_with_score: list[tuple[Any, Any]]) -> list[tuple[Any, float]]:
    if not vector_res_with_score:
        return []
    top_score = float(vector_res_with_score[0][1])
    if top_score < router.rag_top1_min_relevance:
        return []
    selected: list[tuple[Any, float]] = []
    for i, (doc, score_raw) in enumerate(vector_res_with_score):
        score = float(score_raw)
        if i == 0:
            selected.append((doc, score))
            continue
        keep_by_ratio = score >= (top_score * router.rag_secondary_min_ratio)
        keep_by_min = score >= router.rag_min_relevance
        if keep_by_ratio and keep_by_min:
            selected.append((doc, score))
    if not selected:
        selected = [(vector_res_with_score[0][0], top_score)]
    return selected


def hybrid_rrf_select(
    router: Any,
    vector_res_with_score: list[tuple[Any, Any]],
    bm25_docs: list[Any],
) -> list[tuple[Any, float]]:
    if not vector_res_with_score and not bm25_docs:
        return []
    v_scores = _vector_scores_by_faq(vector_res_with_score)
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
    rrf_ranked = _rrf_ranked_faq_ids([v_ids, b_ids], rrf_k=router.rrf_k)
    faq_to_doc = _doc_by_faq_prefer_vector(vector_res_with_score, bm25_docs)
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
        if top_vec < router.rag_top1_min_relevance:
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
        if len(selected) >= router.rag_top_k:
            break
        if top_vec_ref is not None and not math.isnan(vec_s):
            if vec_s >= top_vec_ref * router.rag_secondary_min_ratio and vec_s >= router.rag_min_relevance:
                selected.append((doc, float(vec_s)))
        elif top_vec_ref is not None and math.isnan(vec_s):
            if rrf_s >= top_rrf * router.rag_secondary_min_ratio:
                selected.append((doc, 0.0))
        else:
            if rrf_s >= top_rrf * router.rag_secondary_min_ratio:
                selected.append((doc, float(vec_s) if not math.isnan(vec_s) else 0.0))
    return selected

