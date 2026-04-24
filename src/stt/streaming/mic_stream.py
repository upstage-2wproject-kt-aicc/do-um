"""로컬 마이크 입력 → StreamingPipeline 테스트 진입점.

실행:
    uv run python -m src.stt.streaming.mic_stream

사전 요건:
    - .env에 GOOGLE_PROJECT_ID 설정
    - Google Cloud 인증: gcloud auth application-default login
"""

import os
import queue
import threading

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv

from .pipeline import FRAME_SAMPLES, SAMPLE_RATE, StreamingPipeline

load_dotenv()


def run():
    project_id = os.getenv("GOOGLE_PROJECT_ID", "")
    if not project_id:
        print("❌ GOOGLE_PROJECT_ID 환경변수가 없습니다.")
        return

    pipeline = StreamingPipeline(google_project_id=project_id)

    # callback은 빠르게 리턴해야 하므로 Queue로 분리
    # VAD·NR·STT 처리는 별도 스레드에서 수행
    audio_queue: queue.Queue[bytes | None] = queue.Queue()

    def callback(indata: np.ndarray, frames: int, time, status):
        if status:
            print(f"  ⚠️ {status}")
        pcm_frame = (indata[:, 0] * 32767).astype(np.int16).tobytes()
        audio_queue.put_nowait(pcm_frame)

    was_speaking = False

    def processing_worker():
        nonlocal was_speaking
        while True:
            frame = audio_queue.get()
            if frame is None:   # 종료 신호
                break
            result = pipeline.feed(frame)

            # VAD 상태 변화 출력
            if pipeline._is_speaking and not was_speaking:
                print("🔴 말 감지됨...")
                was_speaking = True
            elif not pipeline._is_speaking and was_speaking:
                print("⏸️  침묵 — STT 처리 중...")
                was_speaking = False

            if result:
                print(f"📝 전사: {result}")

    worker = threading.Thread(target=processing_worker, daemon=True)
    worker.start()

    print("🎤 마이크 스트리밍 시작 (Ctrl+C로 종료)\n")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=FRAME_SAMPLES,
        callback=callback,
    ):
        try:
            while True:
                sd.sleep(100)
        except KeyboardInterrupt:
            print("\n🛑 스트리밍 종료 중...")
            audio_queue.put(None)
            worker.join()
            print("종료 완료")


if __name__ == "__main__":
    run()
