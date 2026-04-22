# Modular Monolith Skeleton (Async Voice AI)

## Architecture Overview
- Pattern: Modular Monolith
- Runtime: Python 3.11
- Contract Layer: Pydantic v2 BaseModel
- Flow: Audio -> STT(VAD/Refine) -> NLU(Intent/RAG) -> Workflow(4-Way) -> Multi-LLM(Async) -> TTS(Stream) -> Evaluation
- Scope: Interface-only skeleton with no business logic implementation

## Module R&R
| Module     | Responsibility                                     | Main Files                                           |
| ---------- | -------------------------------------------------- | ---------------------------------------------------- |
| common     | Shared schemas and typed contracts                 | `src/common/schemas.py`                              |
| stt        | VAD, audio refinement, transcription boundaries    | `src/stt/service.py`                                 |
| nlu        | Intent classification interface (sLLM, DL)         | `src/nlu/classifier.py`                              |
| workflow   | LangGraph state routing and orchestration skeleton | `src/workflow/graph.py`, `src/workflow/multi_llm.py` |
| tts        | TTS streaming output interface                     | `src/tts/service.py`                                 |
| evaluation | RAGAS and LLM-as-a-Judge evaluation interfaces     | `src/evaluation/service.py`                          |
| pipeline   | End-to-end async orchestration contract            | `src/pipeline.py`                                    |

## Run (Placeholder)
```bash
docker compose up --build
```

## Interface-First Principle
- All module boundaries are typed with Pydantic v2 models.
- Inter-module exchanges must use contracts in `src/common/schemas.py`.
- Implementation logic is intentionally omitted; every function raises `NotImplementedError`.

## Async Pipeline Order
1. Receive `AudioChunk`.
2. STT stage runs `apply_vad`, `refine_audio`, `transcribe`.
3. NLU stage runs classifier `classify` and emits `IntentResult`.
4. Workflow stage resolves one of `FAQ`, `HANDOFF`, `PROCEDURE`, `SECURITY`.
5. Multi-LLM stage fans out to Solar, GPT-4o, Claude 3.5 concurrently.
6. TTS stage streams `TTSChunk`.
7. Evaluation stage returns `EvalResult`.

