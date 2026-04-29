# Workflow Handoff

## 1) 현재 범위
- 대상: `NLU 이후 -> Workflow(라우팅/프롬프트/LLM 호출/응답 정규화) -> TTS 전달 포맷`
- 입력 더미 파일: `data/workflow_input_dummy.json`
- 출력 파일: `data/workflow_output_generated.json`

## 2) 구현 완료 항목

### 2.1 라우팅
- 파일: `src/workflow/graph.py`
- 함수:
  - `parse_workflow_inputs`
  - `load_workflow_inputs_from_json`
  - `route_selector`
  - `execute_workflow_item`
  - `execute_workflow_json`
- 규칙:
  - 보안 키워드 포함 시 `security` 우선

  - `metadata.risk_level`이 `높음/high/critical`이면 `handoff` (NLU는 채택 RAG 문서들에서 리스크를 집계해 메타에 반영)
  - `metadata.handoff_required`가 `Y/YES/TRUE/1/REQUIRED`이면 `handoff` (채택 문서 중 하나라도 이관이면 `Y`)
  - `intent=절차형 -> procedure` (의도만 절차/FAQ 분기에 사용; 민원형만으로는 이관하지 않음)
  - 그 외 `faq`

### 2.2 프롬프트
- 시스템 프롬프트: `src/workflow/prompt.py`
  - 공통 Persona + 가드레일(지어내기 금지)
  - 라우트별 추가 지시(FAQ/HANDOFF/PROCEDURE/SECURITY)
- 유저 프롬프트: `src/workflow/context_builder.py`
  - 병합 순서:
    1. `USER_QUERY`
    2. `CHAT_HISTORY`
    3. `INTERNAL_CONTEXT`
    4. `POLICY_RULES`

### 2.3 LLM 호출
- 파일: `src/workflow/multi_llm.py`
- 현재 활성 provider: `solar`만 활성
  - `ACTIVE_PROVIDERS = ("solar",)`
- 비활성 provider(`claude-sonnet`, `gpt`, `grok`)는 호출하지 않고 `PROVIDER_DISABLED` 반환
- 타임아웃:
  - `connect=5s`, `read=15s`, `write=10s`, `pool=5s`
- 서킷 브레이커:
  - 연속 실패 카운트 기반 오픈/리셋
- 에러 분류:
  - `NETWORK_ERROR`
  - `RESP_DELAY`
  - `AUTH_FAILED`
  - `RATE_LIMIT`
  - `UPSTREAM_ERROR`
  - `PROVIDER_EXCEPTION`

### 2.4 Solar 엔드포인트 보강
- `.env`: `LLM_SOLAR_BASE_URL=https://api.upstage.ai/v1`
- 코드 폴백:
  - Solar base URL이 `v2`이고 `404` 발생 시 자동으로 `v1/chat/completions` 재시도

### 2.5 응답 정규화
- 파일: `src/workflow/formatter.py`
- 출력 스키마: `WorkflowOutput` (`src/common/schemas.py`)
- TTS 전달 핵심 필드:
  - `session_id`
  - `final_answer_text`
  - `is_handoff_decided`
  - `reference_links`
  - `llm_token_usage`

### 2.6 실행 엔트리
- 파일: `src/workflow/run_workflow.py`
- 기능:
  - 입력 JSON 읽기
  - 전체 워크플로우 실행
  - 결과 JSON 파일 저장
  - 표준출력에도 JSON 출력

## 3) 실행 방법
```bash
cd /Users/cocomong_98/Documents/SESAC/doum
source .venv311/bin/activate
set -a && source .env && set +a
python -m src.workflow.run_workflow --input data/workflow_input_dummy.json --output data/workflow_output_generated.json
```

## 4) 결과 해석
- `results[]`는 provider별 결과
- `solar` 성공 시:
  - `error=null`
  - `answer` 채워짐
- `solar` 실패 시:
  - 에러코드로 원인 확인 (`NETWORK_ERROR`, `AUTH_FAILED` 등)
- `final_answer_text`:
  - 첫 성공 provider 답변
  - 전부 실패 시 빈 문자열
- `is_handoff_decided`:
  - 최종 답변 없으면 `true`

## 5) 자주 발생한 이슈와 해석
- `PROVIDER_EXCEPTION` 반복:
  - 현재는 분류 개선으로 대부분 상세 코드로 분리됨
- `NETWORK_ERROR`:
  - 실행 컨텍스트 DNS/외부망 차단 가능성 높음
- `404`:
  - Solar base URL/경로 불일치 가능성 (`v1` 기준으로 통일)

## 6) 현재 제한사항
- TTFT는 실스트리밍 기반이 아니라 `latency`와 동일하게 기록
- 환각 검증(`grounded/citations`)은 본격 구현 전(기본값 중심)
- semantic cache는 경량 골격 수준

## 7) 파일 인덱스
- `src/workflow/graph.py`
- `src/workflow/context_builder.py`
- `src/workflow/prompt.py`
- `src/workflow/multi_llm.py`
- `src/workflow/formatter.py`
- `src/workflow/semantic_cache.py`
- `src/workflow/run_workflow.py`
- `src/common/schemas.py`
