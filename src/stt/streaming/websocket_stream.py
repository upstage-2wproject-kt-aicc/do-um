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

from dotenv import load_dotenv
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.nlu.aicc_core_klue import AICC_NLU_Router
from src.workflow.graph import execute_workflow_item, workflow_input_from_nlu_dict
from .pipeline import StreamingPipeline

load_dotenv()

router = APIRouter()
nlu_router: AICC_NLU_Router | None = None


@router.on_event("startup")
async def startup_event() -> None:
    """Loads heavy NLU resources once at service startup."""
    global nlu_router
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
        google_project_id=os.getenv("GOOGLE_PROJECT_ID", ""),
        stt_provider="openai" if stt_provider == "openai" else "google",
    )

    try:
        while True:
            frame = await websocket.receive_bytes()
            transcript = pipeline.feed(frame)
            if transcript:
                transcript_text = transcript.text
                if nlu_router is None:
                    # NLU 초기화 실패 시 STT 결과만 전송
                    await websocket.send_json(
                        {
                            "session_id": transcript.session_id,
                            "transcript": transcript_text,
                            "is_final": True,
                        }
                    )
                    continue

                # NLU는 동기/무거운 작업이므로 executor에서 실행
                loop = asyncio.get_running_loop()
                nlu_result = await loop.run_in_executor(
                    None, nlu_router.process_query, transcript_text
                )

                # 클라이언트 전송은 경량 필드 중심으로 제한
                nlu_payload = {
                    "status": nlu_result.get("status"),
                    "intent": nlu_result.get("intent"),
                    "final_answer": nlu_result.get("final_answer"),
                    "metadata": nlu_result.get("metadata"),
                    "handoff_reason": nlu_result.get("handoff_reason"),
                    "handoff_confidence": nlu_result.get("handoff_confidence"),
                    "transfer_action": nlu_result.get("transfer_action"),
                    "timings_sec": nlu_result.get("timings_sec"),
                }

                workflow_payload = None
                workflow_output = None
                if nlu_result.get("status") == "HANDOFF_DIRECT":
                    workflow_output = None
                elif nlu_result.get("status") == "REQUIRE_LLM":
                    workflow_payload = workflow_input_from_nlu_dict(
                        {
                            "session_id": transcript.session_id,
                            "user_query": transcript_text,
                            **nlu_result,
                        }
                    )
                    workflow_output = await execute_workflow_item(workflow_payload)

                await websocket.send_json(
                    {
                        "session_id": transcript.session_id,
                        "transcript": transcript_text,
                        "is_final": True,
                        "nlu_analysis": nlu_payload,
                        "workflow": (
                            workflow_output.model_dump(mode="json")
                            if workflow_output
                            else None
                        ),
                        "action": nlu_result.get("transfer_action"),
                    }
                )
    except WebSocketDisconnect:
        pass
