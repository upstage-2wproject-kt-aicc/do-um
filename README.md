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
| evaluation | 품질/성능 평가 인터페이스 | `src/evaluation/service.py` |
| pipeline | workflow→TTS 오케스트레이션 | `src/pipeline.py`, `src/main.py` |

## 팀 공통 필수 규칙 (호환성)
아래 항목을 지키지 않으면 `git pull` 이후 실행 환경이 달라져 재현이 깨진다.

1. Python 버전 고정
- 반드시 Python `3.11` 사용.
- `3.12+` 또는 `3.14` 기본 인터프리터에서 바로 `uv sync`하면 의존성 해석이 깨질 수 있음.

2. 의존성 설치 명령 고정
- 최초/갱신 설치는 반드시 아래 명령 사용:
```bash
UV_CACHE_DIR=.uv-cache uv sync --python 3.11
```
- `pyproject.toml`의 `tool.uv.environments`는 현재 `darwin + py3.11` 기준으로 잠겨 있음.

3. 환경변수 키 이름 고정
- `.env.example`을 복사해 `.env` 생성 후 값 주입.
- 키 이름을 임의로 바꾸지 말 것. 코드에서 다음 이름을 기준으로 읽음:
  - `LLM_SOLAR_API_KEY`
  - `LLM_GPT_API_KEY`
  - `LLM_CLAUDE_SONNET_API_KEY` (사용 시)
  - `LLM_GROK_API_KEY` (사용 시)
  - `STT_PROVIDER` (`google` 또는 `openai`)
  - `TTS_PROVIDER` (`azure`, `openai`, `google`, `naver`)
  - `GOOGLE_PROJECT_ID` (Google STT 사용 시)
  - `AZURE_SPEECH_KEY`, `AZURE_SERVICE_REGION` (Azure TTS 사용 시)

4. 캐시/로컬 산출물 커밋 금지
- `.venv/`, `.uv-cache/`, `.hf-cache/`, 실행 결과 오디오/JSON은 커밋하지 않음.

## 실행 방법
### 1) API 서버 실행
```bash
HF_HOME=.hf-cache STT_PROVIDER=openai TTS_PROVIDER=openai UV_CACHE_DIR=.uv-cache uv run python -m src.main
```

### 2) 워크플로우 JSON 단독 실행
```bash
UV_CACHE_DIR=.uv-cache uv run python -m src.workflow.run_workflow --input data/workflow_input_dummy.json --input-type workflow
```

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
