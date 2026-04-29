# 비동기 음성 AICC 모듈러 모놀리스

## 아키텍처 개요
- 패턴: Modular Monolith
- 런타임: Python 3.11
- 계약 레이어: Pydantic v2 (`src/common/schemas.py`)
- 파이프라인: Audio → STT(VAD/정제) → NLU(Intent/RAG) → Workflow(4-Way) → Multi-LLM(Async) → TTS(Stream) → Evaluation

## 모듈 R&R
| 모듈 | 역할 | 주요 파일 |
| --- | --- | --- |
| common | 공통 스키마, 설정, 예외, 로깅 | `src/common/schemas.py`, `src/common/config.py` |
| stt | 스트리밍 STT, 전처리/VAD, WebSocket 입력 | `src/stt/streaming/pipeline.py`, `src/stt/streaming/websocket_stream.py` |
| nlu | 의도 분류, 임베딩, RAG 조회 | `src/nlu/aicc_core_klue.py` |
| workflow | LangGraph 라우팅, 멀티 LLM 호출, 응답 정규화 | `src/workflow/graph.py`, `src/workflow/multi_llm.py`, `src/workflow/prompt.py` |
| tts | TTS Provider 추상화 및 스트리밍 출력 | `src/tts/factory.py`, `src/tts/openai.py`, `src/tts/azure.py` |
| evaluation | 품질/성능 평가 실행 및 스코어링 | `src/evaluation/run_evaluation.py`, `src/evaluation/scoring.py` |
| pipeline | workflow→TTS 오케스트레이션 | `src/pipeline.py`, `src/main.py` |

## 팀 공통 필수 규칙 (호환성)
아래 항목을 지키지 않으면 `git pull` 이후 실행 환경이 달라져 재현이 깨진다.

1. Python 버전 고정
- 반드시 Python `3.11` 사용.
- `3.12+` 또는 `3.14` 기본 인터프리터에서 바로 `uv sync`하면 의존성 해석이 깨질 수 있음.

2. 의존성 설치 명령 고정
- 최초/갱신 설치는 반드시 `uv sync --python 3.11` 사용.
- `pyproject.toml`과 `uv.lock`은 macOS/Windows/Linux를 모두 포함하도록 잠금되어 있음.
- OS별 캐시 변수 설정은 아래 실행 절차를 따를 것.

3. 모델 디렉토리 경로 고정
- 공유받은 NLU 모델 디렉토리를 아래 경로에 배치:
  - `src/nlu/my_aicc_nlu_model_klue`

4. 환경변수 키 이름 고정
- `.env.example`을 복사해 `.env` 생성 후 값 주입.
- 키 이름을 임의로 바꾸지 말 것. 코드에서 다음 이름을 기준으로 읽음:
  - `LLM_SOLAR_API_KEY`
  - `LLM_GPT_API_KEY`
  - `LLM_CLAUDE_SONNET_API_KEY` (사용 시)
  - `LLM_GROK_API_KEY` (사용 시)
  - `STT_PROVIDER` (`openai` 기본, 필요 시 `google`)
  - `TTS_PROVIDER` (`openai` 기본)
  - `TTS_FALLBACK_PROVIDERS` (기본: `google,azure`)
  - `GOOGLE_PROJECT_ID`, `GOOGLE_APPLICATION_CREDENTIALS` (Google STT/TTS 사용 시)
  - `AZURE_SPEECH_KEY`, `AZURE_SERVICE_REGION` (Azure TTS fallback 사용 시)

5. 캐시/로컬 산출물 커밋 금지
- `.venv/`, `.uv-cache/`, `.hf-cache/`, 실행 결과 오디오/JSON은 커밋하지 않음.

## OS별 실행 방법 (pull 이후)
아래 순서를 그대로 실행하면 됨.

### macOS (zsh/bash)
1. 설치
```bash
UV_CACHE_DIR=.uv-cache uv sync --python 3.11
```
2. 서버 실행
```bash
HF_HOME=.hf-cache STT_PROVIDER=openai TTS_PROVIDER=openai UV_CACHE_DIR=.uv-cache uv run python -m src.main
```

### Windows (PowerShell)
1. 설치
```powershell
$env:UV_CACHE_DIR=".uv-cache"
uv sync --python 3.11
```
2. 서버 실행
```powershell
$env:HF_HOME=".hf-cache"
$env:STT_PROVIDER="openai"
$env:TTS_PROVIDER="openai"
$env:TTS_FALLBACK_PROVIDERS="google,azure"
$env:UV_CACHE_DIR=".uv-cache"
uv run python -m src.main
```

### Linux (bash)
1. 설치
```bash
UV_CACHE_DIR=.uv-cache uv sync --python 3.11
```
2. 서버 실행
```bash
HF_HOME=.hf-cache STT_PROVIDER=openai TTS_PROVIDER=openai UV_CACHE_DIR=.uv-cache uv run python -m src.main
```

## 현재 기본 실행 기준
- STT 기본값: `openai` (`src/stt/streaming/websocket_stream.py`)
- TTS 기본값: `openai` (`src/pipeline.py`)
- TTS fallback 순서: `google -> azure` (`TTS_FALLBACK_PROVIDERS`)
- Google 경로 사용 시 ADC 필요: `GOOGLE_APPLICATION_CREDENTIALS`

## 프론트 연동용 WebSocket 이벤트 규약
- 엔드포인트: `ws://localhost:8000/ws/stt`
- 서버는 단계별 이벤트와 최종 결과를 JSON으로 전송한다.

1. 단계 이벤트 (`event = "pipeline_stage"`)
- 공통 필드
  - `session_id`
  - `stage`: `stt | nlu | workflow | tts`
  - `status`: `started | completed | chunk | skipped | failed`
  - `started_at_ms`, `ended_at_ms`, `latency_ms`
  - `payload` (단계별 상세 데이터)
- 예시
```json
{
  "event": "pipeline_stage",
  "session_id": "sess_xxx",
  "stage": "tts",
  "status": "chunk",
  "started_at_ms": 1714360000000,
  "ended_at_ms": 1714360000200,
  "latency_ms": 200,
  "payload": {
    "chunk_id": 3,
    "is_last": false,
    "size_bytes": 4096,
    "audio_base64": "..."
  }
}
```

2. 최종 결과 (`event = "pipeline_result"`)
- 공통 필드
  - `session_id`, `transcript`, `transcript_confidence`, `transcript_timestamp_ms`
  - `nlu_analysis`
  - `workflow` (없을 수 있음)
  - `saved_file` (`data/runtime/*.json` 저장 경로)

## 런타임 산출물 저장
- WebSocket 파이프라인 결과는 `data/runtime/YYMMDD_HH:MM:SS.json` 형식으로 저장된다.
- 이 파일은 디버깅/재현용 로컬 산출물이며 Git에 커밋하지 않는다.

## 스트리밍 종료 처리
- STT 스트리밍 파이프라인은 `finalize()`를 통해 입력 종료 시점의 버퍼를 강제 flush할 수 있다.
- 프론트에서 녹음 종료 시점을 명확히 관리하면 마지막 발화 누락을 줄일 수 있다.

## 인터페이스 우선 원칙
- 모듈 간 I/O는 `src/common/schemas.py`의 Pydantic 모델을 기준으로 교환.
- 모듈 직접 의존보다 계약(스키마) 기반 연결을 우선.

## 비동기 파이프라인 순서
1. `AudioChunk` 수신
2. STT 전처리/VAD/전사
3. NLU 의도 분류 + RAG 컨텍스트 구성
4. Workflow 라우팅 (`FAQ`, `HANDOFF`, `PROCEDURE`, `SECURITY`)
5. Multi-LLM 비동기 fan-out/fan-in
6. TTS 스트리밍 (`TTSChunk`)
7. 평가/로그 수집

## 답변/근거 링크 분리
- TTS로 읽힐 텍스트에는 원문 URL을 직접 넣지 않는다.
- RAG 문서의 `source_url`은 `WorkflowOutput.reference_links`에 별도로 담아 상담사 화면이나 로그에서 참조한다.
- `final_answer_text`와 `pre_tts_text`는 음성 응답에 적합한 자연어 문장만 유지한다.

## 평가 시나리오 실행

대표 RAG FAQ 평가 시나리오는 `evaluation/scenarios/rag_faq_v1.tsv`에 있다.
TSV를 사용하는 이유는 긴 한국어 답변, 쉼표가 포함된 keywords, JSON metadata를 CSV보다 덜 헷갈리게 관리하기 위해서다.

### 시나리오 파일 전체를 하나씩 평가

아래 명령은 TSV의 각 시나리오를 하나씩 workflow prompt에 주입하고, candidate 모델 응답을 judge/RAGAS로 평가한 결과를 시나리오별 JSON으로 저장한다.
각 시나리오 내부에서는 candidate 모델, judge 평가, RAGAS 계산을 비동기로 실행한다.

```bash
UV_CACHE_DIR=.uv-cache uv run --python 3.11 python -m src.evaluation.run_scenarios \
  --scenarios evaluation/scenarios/rag_faq_v1.tsv \
  --output-dir evaluation_runs/rag_faq_v1
```

결과는 `evaluation_runs/rag_faq_v1/<scenario_id>.json`에 저장되고, 전체 실행 목록은 `evaluation_runs/rag_faq_v1/index.json`에 저장된다.
`index.json`의 각 시나리오 항목에는 candidate 생성, judge 평가, RAGAS 계산을 포함한 전체 소요 시간인 `duration_ms`가 함께 기록된다.

### 특정 시나리오만 평가

```bash
UV_CACHE_DIR=.uv-cache uv run --python 3.11 python -m src.evaluation.run_scenarios \
  --scenarios evaluation/scenarios/rag_faq_v1.tsv \
  --scenario-id rag_faq_006_loan_limit_handoff \
  --output-dir evaluation_runs/rag_faq_v1_single
```

### 모델을 명시해서 평가

모델을 명시하지 않으면 `.env`의 `LLM_*_MODEL`, `JUDGE_*_MODEL` 설정을 읽는다.
일회성으로 지정하려면 `--candidate`, `--judge`를 반복해서 넣는다.

```bash
UV_CACHE_DIR=.uv-cache uv run --python 3.11 python -m src.evaluation.run_scenarios \
  --candidate solar:solar-pro3 \
  --candidate gpt:gpt-4o \
  --candidate claude-sonnet:claude-sonnet-4-6 \
  --judge openai:gpt-5.5 \
  --judge anthropic:claude-opus-4-7 \
  --judge google:gemini-3.1-pro-preview
```

현재 운영 평가 기준은 `comparative_10`이며, 이 모드는 RAGAS를 사용하지 않는다.
`legacy_5`와 `--disable-ragas`는 이전 5점 체계와 회귀 확인용으로만 유지한다.

### v4 비교 평가 프롬프트로 테스트

`comparative_10` rubric은 `judge_v4_comparative.md`를 사용한다.
이 모드는 모델별 응답 성향 비교를 위해 6개 judge 항목을 1~10점 원점수로 평가하며, RAGAS를 제외한다.
`groundedness`가 Faithfulness 역할을, `intent_fit`과 `guidance_quality`가 Answer Relevancy 역할을 대신한다.
handoff/no-context 시나리오는 RAGAS 해석이 불안정할 수 있으므로 LLM-as-a-Judge 결과를 기준으로 본다.
한 시나리오 안에서는 candidate 4개와 judge 3개 기준 최대 12개 judge 호출이 병렬 실행되도록 provider별 judge 동시성 기본값을 4로 둔다.

```bash
UV_CACHE_DIR=.uv-cache uv run --python 3.11 python -m src.evaluation.run_scenarios \
  --scenarios evaluation/scenarios/rag_faq_v1.tsv \
  --scenario-id rag_faq_001_rate_explain \
  --judge-rubric comparative_10 \
  --output-dir evaluation_runs/rag_faq_v1_v4_single
```

결과를 사람이 읽기 좋은 Markdown으로 보려면 아래 명령을 실행한다.
리포트에는 질문, context/metadata, 모델별 답변 전문, 6개 점수, primary score, judge 요약, 토큰 사용량 기반 비용 추정치가 함께 포함된다.

```bash
UV_CACHE_DIR=.uv-cache uv run --python 3.11 python -m src.evaluation.summarize_run \
  --run-dir evaluation_runs/rag_faq_v1_v4_single
```

### 비용 통계 확인

`summary.md` 상단의 `Model Cost / Latency Overview`는 모델별 평균 점수, 평균 전체 소요 시간, 점수 표준편차, candidate 평균 입력/출력 토큰, candidate 1회/1,000회 예상 비용, judge 평균 비용을 보여준다.
시나리오별 섹션에도 candidate 비용과 judge 비용이 별도 표로 기록된다.

비용은 `src/evaluation/pricing.py`의 가격 스냅샷을 기준으로 계산한다.
기존 평가 결과처럼 judge 응답의 `token_usage`가 저장되지 않은 run은 judge 비용이 `N/A`로 표시되고, 새로 실행하는 평가부터 judge 비용까지 계산된다.
