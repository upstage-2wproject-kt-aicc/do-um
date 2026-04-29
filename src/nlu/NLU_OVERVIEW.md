# NLU 모듈 상세 안내서 (최신 분리 구조)

이 문서는 `src/nlu`의 현재 구조와 처리 단계를 설명합니다.
민감/보안 정보(키 값, 내부 운영 정책의 구체 값)는 제외했습니다.

---

## 1) NLU의 역할

현재 NLU(2단계)는 다음을 수행합니다.

- 의도/서브도메인 추론
- RAG 검색(Hybrid: Vector + BM25)
- Semantic Cache 조회
- Guardrail 점수 계산 및 의사결정
- 3단계 Workflow로 전달할 payload 구성
- 필요 시 `HANDOFF_DIRECT` / `REJECT_DIRECT`로 워크플로우 우회

---

## 2) 파일 구조

```text
src/nlu/
  __init__.py
  aicc_core_klue.py
  schemas.py
  service.py
  RAG_FAQ.csv

  intent/
    __init__.py
    classifier.py

  retrieval/
    __init__.py
    index_manager.py
    vector_store.py
    selector.py
    cache.py

  guardrail/
    __init__.py
    scorer.py
    policy.py

  response/
    __init__.py
    builder.py

  archive/
    aicc_core_system.py
    classifier.py
    sllm.py
    Eval_Queries (2).csv
    test_bench.py
    test_consistency.py
```

---

## 3) 모듈별 책임

### `aicc_core_klue.py`
- NLU Router 초기화 컨테이너
- 모델/인덱스/설정값 로딩
- 외부 호출 인터페이스(`process_query`) 제공
- 실제 오케스트레이션은 `service.py`에 위임

### `schemas.py`
- NLU 단계 간 데이터 계약(dataclass)
- 코어 결과/검색 결과/가드레일 결과/최종 결과 구조 정의

### `service.py`
- NLU 전체 파이프라인 오케스트레이션
- 분기 판단과 최종 payload 조립

### `intent/classifier.py`
- intent/subdomain 분류 함수
- 병렬 실행 유틸(async + threadpool)

### `retrieval/index_manager.py`
- FAQ CSV 로딩 및 문서화
- BM25 준비
- Vector DB 연결/갱신
- warm-up/초기 cache 준비

### `retrieval/vector_store.py`
- RAG 검색 실행 함수
- vector 검색 개수 계산 유틸

### `retrieval/selector.py`
- vector-only 선택 로직
- hybrid RRF 선택 로직
- 최종 후보 문서 정렬/필터링

### `retrieval/cache.py`
- semantic cache hit 판정
- cache 응답 반환 규격 처리

### `guardrail/scorer.py`
- 점수 기반 Guardrail 계산
- decision(`ALLOW`/`LIMIT`/`HANDOFF`/`REJECT`) 및 사유 생성

### `guardrail/policy.py`
- Guardrail decision을 policy rule로 변환
- `LIMIT` 상황에서 3단계에 전달할 제한 규칙 구성

### `response/builder.py`
- `HANDOFF_DIRECT` 응답 빌드
- `REJECT_DIRECT` 응답 빌드

---

## 4) NLU 처리 순서 (실행 플로우)

`AICC_NLU_Router.process_query()` 기준:

1. 입력 텍스트 정규화
2. intent/subdomain/embedding 병렬 실행
3. semantic cache 검사
4. cache hit면 `CACHED` 반환
5. cache miss면 Hybrid RAG 검색 수행
6. 검색 결과에서 메타데이터 집계(리스크/이관 여부 등)
7. Guardrail 점수 계산
8. decision별 분기:
   - `HANDOFF` -> `HANDOFF_DIRECT`
   - `REJECT` -> `REJECT_DIRECT`
   - `LIMIT`/`ALLOW` -> `REQUIRE_LLM` (policy_rules 포함 가능)
9. 최종 payload 반환

---

## 5) Guardrail 출력 항목

Guardrail 계산 후 응답에는 다음 필드들이 포함될 수 있습니다.

- `guardrail_decision`
- `guardrail_score`
- `guardrail_reasons`
- `guardrail_components`
- `action` (예: 이관/거절 액션)
- `policy_rules` (`LIMIT`일 때 주로 사용)

이 값들은 추적성(왜 이런 분기가 났는지)과 운영 튜닝에 사용됩니다.

---

## 6) 1단계(STT) / 3단계(Workflow) 연계

### 1단계(STT) -> 2단계(NLU)
- STT 텍스트를 NLU 입력으로 전달
- 고객 컨텍스트(있다면)도 함께 전달

### 2단계(NLU) -> 3단계(Workflow)
- `status`가 `REQUIRE_LLM`이면 워크플로우 실행
- `status`가 `HANDOFF_DIRECT` 또는 `REJECT_DIRECT`이면 워크플로우 우회 가능
- NLU가 산출한 `metadata`, `policy_rules`, `guardrail_*` 정보는 downstream 제어 신호로 사용

---

## 7) 반환 계약(핵심)

주요 반환 필드 예시:

- 공통: `status`, `intent`, `timings_sec`
- 검색 사용 시: `retrieved_context`, `retrieved_faq_ids`, `metadata`
- 가드레일 사용 시: `guardrail_decision`, `guardrail_score`, `guardrail_reasons`, `policy_rules`
- 직접 처리 시: `final_answer`, `action`

`status` 기반 분기가 2-3단계 연동의 핵심 계약입니다.

---

## 8) archive 정책

`archive/`는 운영 경로에서 제외된 파일을 보관합니다.

- 현재 이동된 비핵심 테스트 스크립트:
  - `archive/test_bench.py`
  - `archive/test_consistency.py`

운영 코드와 실험/백업 코드를 분리해 유지보수 복잡도를 낮추는 목적입니다.

