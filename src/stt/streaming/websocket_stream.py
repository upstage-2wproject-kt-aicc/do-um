"""FastAPI WebSocket 연동 — STT(1단계) -> NLU(2단계) 브릿지.

백엔드 main.py 등록 예시:
    from src.stt.streaming.websocket_stream import router
    app.include_router(router)

프론트엔드 연동 예시 (JavaScript / React):
    const ws = new WebSocket("ws://localhost:8000/ws/stt");

    // MediaRecorder로 캡처한 PCM 청크를 30ms 단위로 전송
    ws.send(pcmChunk);  // ArrayBuffer (16kHz · 16-bit · mono)

    // 발화 완료 시 서버가 JSON 반환
    ws.onmessage = (e) => {
        const { transcript, is_final } = JSON.parse(e.data);
        console.log(transcript);
    };

프론트 오디오 캡처 조건:
    - sampleRate: 16000
    - channelCount: 1
    - format: PCM 16-bit (Int16Array)
    - chunkSize: 480 samples = 30ms
"""

import os
import asyncio
import time
import base64
from typing import Any

from dotenv import load_dotenv
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.nlu.aicc_core_klue import AICC_NLU_Router
from src.pipeline import VoiceAIPipeline
from src.workflow.graph import execute_workflow_item, workflow_input_from_nlu_dict
from src.workflow.online_evaluation import OnlineEvaluationService
from .pipeline import StreamingPipeline

load_dotenv()

router = APIRouter()
nlu_router: AICC_NLU_Router | None = None
@router.on_event("startup")
async def startup_event() -> None:
    """Loads heavy NLU resources once at service startup."""
    global nlu_router
    if nlu_router is not None:
        print("✅ [STT->NLU] NLU 라우터 이미 로드됨 (중복 초기화 생략)")
        return
    try:
        nlu_router = AICC_NLU_Router()
        print("✅ [STT->NLU] NLU 라우터 로드 완료")
    except Exception as e:
        nlu_router = None
        print(f"❌ [STT->NLU] NLU 라우터 로드 실패: {e}")


@router.websocket("/ws/stt")
async def stt_websocket(websocket: WebSocket):
    """30ms PCM 청크를 수신해 STT+NLU 결과를 JSON으로 반환."""
    await websocket.accept()
    stt_provider = os.getenv("STT_PROVIDER", "openai").strip().lower()
    pipeline = StreamingPipeline(
        google_project_id=(
            os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
            or os.getenv("GOOGLE_PROJECT_ID", "").strip()
        ),
        stt_provider="openai" if stt_provider == "openai" else "google",
    )
    voice_pipeline = VoiceAIPipeline()

    async def send_event(
        stage: str,
        status: str,
        session_id: str,
        started_at_ms: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        ended_at_ms = int(time.time() * 1000)
        print(
            f"[WS_EVENT] session={session_id} stage={stage} status={status} "
            f"latency_ms={(ended_at_ms - started_at_ms) if started_at_ms else None}"
        )
        await websocket.send_json(
            {
                "event": "pipeline_stage",
                "session_id": session_id,
                "stage": stage,
                "status": status,
                "started_at_ms": started_at_ms,
                "ended_at_ms": ended_at_ms,
                "latency_ms": (ended_at_ms - started_at_ms) if started_at_ms else None,
                "payload": payload or {},
            }
        )

    async def process_transcript(transcript, stt_started_at_ms: int | None = None) -> None:
        transcript_text = transcript.text
        await send_event(
            stage="stt",
            status="completed",
            session_id=transcript.session_id,
            started_at_ms=stt_started_at_ms,
            payload={
                "text": transcript_text,
                "is_final": True,
                "confidence": transcript.confidence,
                "timestamp_ms": transcript.timestamp_ms,
            },
        )
        if nlu_router is None:
            await send_event(
                stage="nlu",
                status="failed",
                session_id=transcript.session_id,
                payload={"reason": "nlu_not_initialized"},
            )
            return

        nlu_started_at_ms = int(time.time() * 1000)
        await send_event(stage="nlu", status="started", session_id=transcript.session_id)
        loop = asyncio.get_running_loop()
        nlu_result = await loop.run_in_executor(
            None, nlu_router.process_query, transcript_text
        )

        nlu_payload = {
            "status": nlu_result.get("status"),
            "intent": nlu_result.get("intent"),
            "final_answer": nlu_result.get("final_answer"),
            "metadata": nlu_result.get("metadata"),
            "timings_sec": nlu_result.get("timings_sec"),
        }
        await send_event(
            stage="nlu",
            status="completed",
            session_id=transcript.session_id,
            started_at_ms=nlu_started_at_ms,
            payload=nlu_payload,
        )

        workflow_output = None
        workflow_payload = None
        evaluation_task: asyncio.Task[dict[str, Any]] | None = None
        evaluation_started_at_ms: int | None = None

        # Force evaluation regardless of NLU route so front can always receive evaluation latency/result.
        force_evaluation = os.getenv("FORCE_EVALUATION", "1").strip().lower() not in {
            "0",
            "false",
            "off",
            "no",
        }
        if force_evaluation:
            try:
                workflow_payload = workflow_input_from_nlu_dict(
                    {
                        "session_id": transcript.session_id,
                        "user_query": transcript_text,
                        **nlu_result,
                    }
                )
                evaluation_started_at_ms = int(time.time() * 1000)
                await send_event(
                    stage="evaluation",
                    status="started",
                    session_id=transcript.session_id,
                    started_at_ms=evaluation_started_at_ms,
                )
                evaluation_task = asyncio.create_task(
                    OnlineEvaluationService().start(workflow_payload)
                )
            except Exception:
                evaluation_task = None
                await send_event(
                    stage="evaluation",
                    status="failed",
                    session_id=transcript.session_id,
                    started_at_ms=evaluation_started_at_ms,
                    payload={"reason": "evaluation_start_failed"},
                )

        if nlu_result.get("status") == "REQUIRE_LLM":
            workflow_started_at_ms = int(time.time() * 1000)
            await send_event(
                stage="workflow",
                status="started",
                session_id=transcript.session_id,
            )
            if workflow_payload is None:
                workflow_payload = workflow_input_from_nlu_dict(
                    {
                        "session_id": transcript.session_id,
                        "user_query": transcript_text,
                        **nlu_result,
                    }
                )
            workflow_output = await execute_workflow_item(workflow_payload)
            workflow_json = workflow_output.model_dump(mode="json")
            await send_event(
                stage="workflow",
                status="completed",
                session_id=transcript.session_id,
                started_at_ms=workflow_started_at_ms,
                payload=workflow_json,
            )
            tts_text = (workflow_output.pre_tts_text or workflow_output.final_answer_text or "").strip()
            if tts_text:
                await send_event(
                    stage="tts",
                    status="started",
                    session_id=transcript.session_id,
                    payload={"pre_tts_text": tts_text},
                )
                tts_started_at_ms = int(time.time() * 1000)
                tts_chunks = 0
                async for chunk in voice_pipeline.stream_tts_for_workflow_output(workflow_output):
                    tts_chunks += 1
                    await send_event(
                        stage="tts",
                        status="chunk",
                        session_id=transcript.session_id,
                        started_at_ms=tts_started_at_ms,
                        payload={
                            "chunk_id": chunk.chunk_id,
                            "is_last": chunk.is_last,
                            "size_bytes": len(chunk.audio_bytes),
                            "audio_base64": base64.b64encode(chunk.audio_bytes).decode("ascii"),
                        },
                    )
                await send_event(
                    stage="tts",
                    status="completed",
                    session_id=transcript.session_id,
                    started_at_ms=tts_started_at_ms,
                    payload={"chunks": tts_chunks},
                )
            else:
                await send_event(
                    stage="tts",
                    status="skipped",
                    session_id=transcript.session_id,
                    payload={"reason": "missing_tts_text"},
                )
        else:
            await send_event(
                stage="workflow",
                status="skipped",
                session_id=transcript.session_id,
                payload={"reason": "nlu_status_not_require_llm"},
            )
            nlu_status = str(nlu_result.get("status") or "").strip().upper()
            if nlu_status == "REJECT_DIRECT":
                direct_answer = str(
                    nlu_result.get("final_answer")
                    or "금융 관련 문의에 한해 답변드릴 수 있습니다. 금융 상담 질문으로 다시 요청해 주세요."
                ).strip()
            else:
                direct_answer = str(nlu_result.get("final_answer") or "").strip()
            if direct_answer:
                await send_event(
                    stage="tts",
                    status="started",
                    session_id=transcript.session_id,
                    payload={"pre_tts_text": direct_answer},
                )
                tts_started_at_ms = int(time.time() * 1000)
                tts_chunks = 0
                async for chunk in voice_pipeline.stream_tts_for_text(
                    session_id=transcript.session_id,
                    text=direct_answer,
                ):
                    tts_chunks += 1
                    await send_event(
                        stage="tts",
                        status="chunk",
                        session_id=transcript.session_id,
                        started_at_ms=tts_started_at_ms,
                        payload={
                            "chunk_id": chunk.chunk_id,
                            "is_last": chunk.is_last,
                            "size_bytes": len(chunk.audio_bytes),
                            "audio_base64": base64.b64encode(chunk.audio_bytes).decode("ascii"),
                        },
                    )
                await send_event(
                    stage="tts",
                    status="completed",
                    session_id=transcript.session_id,
                    started_at_ms=tts_started_at_ms,
                    payload={"chunks": tts_chunks},
                )
            else:
                await send_event(
                    stage="tts",
                    status="skipped",
                    session_id=transcript.session_id,
                    payload={"reason": "missing_final_answer"},
                )

        result_payload = {
            "event": "pipeline_result",
            "session_id": transcript.session_id,
            "transcript": transcript_text,
            "transcript_confidence": transcript.confidence,
            "transcript_timestamp_ms": transcript.timestamp_ms,
            "is_final": True,
            "nlu_analysis": nlu_payload,
            "workflow": (
                workflow_output.model_dump(mode="json")
                if workflow_output
                else None
            ),
        }

        await websocket.send_json(result_payload)
        if evaluation_task is not None:
            try:
                online_run = await evaluation_task
                if online_run.evaluation_task is not None:
                    evaluation_payload = await online_run.evaluation_task
                    await websocket.send_json(
                        {
                            "event": "evaluation_result",
                            **evaluation_payload,
                        }
                    )
                await send_event(
                    stage="evaluation",
                    status="completed",
                    session_id=transcript.session_id,
                    started_at_ms=evaluation_started_at_ms,
                    payload={"status": "done"},
                )
            except Exception:
                await send_event(
                    stage="evaluation",
                    status="failed",
                    session_id=transcript.session_id,
                    started_at_ms=evaluation_started_at_ms,
                    payload={"reason": "evaluation_failed"},
                )

    try:
        while True:
            frame = await websocket.receive_bytes()
            stt_started_at_ms = int(time.time() * 1000)
            await send_event(
                stage="stt",
                status="started",
                session_id=pipeline.session_id,
                started_at_ms=stt_started_at_ms,
            )
            transcript = pipeline.feed(frame)
            if transcript:
                await process_transcript(transcript, stt_started_at_ms=stt_started_at_ms)
                break
    except WebSocketDisconnect:
        pass
