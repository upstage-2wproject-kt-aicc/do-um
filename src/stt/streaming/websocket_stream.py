"""FastAPI WebSocket 연동 — 백엔드 담당 참고용.

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

from dotenv import load_dotenv
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .pipeline import StreamingPipeline

load_dotenv()

router = APIRouter()


@router.websocket("/ws/stt")
async def stt_websocket(websocket: WebSocket):
    """30ms PCM 청크를 수신해 발화 완료 시 전사 결과를 JSON으로 반환."""
    await websocket.accept()
    pipeline = StreamingPipeline(
        google_project_id=os.getenv("GOOGLE_PROJECT_ID", "")
    )

    try:
        while True:
            frame = await websocket.receive_bytes()
            transcript = pipeline.feed(frame)
            if transcript:
                await websocket.send_json({"transcript": transcript, "is_final": True})
    except WebSocketDisconnect:
        pass
