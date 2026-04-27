import asyncio
import websockets
import json
import time
import requests
import sys
import os
from pydub import AudioSegment
import io
import numpy as np

# 서버 주소
WS_URL = "ws://localhost:8000/ws/stt"
HTTP_URL = "http://localhost:8000/pipeline/stream"

def get_audio_from_mic(duration=5, sample_rate=16000):
    """
    마이크로부터 실시간으로 오디오를 캡처하는 제너레이터 (동기 방식 예시)
    실제 서비스에서는 콜백 방식을 쓰지만, 여기서는 테스트를 위해 단순화합니다.
    """
    import sounddevice as sd
    print(f"🎤 마이크 입력 시작... ({duration}초 동안 말씀해주세요)")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()
    return audio_data.tobytes()

async def send_mic_streaming(websocket, duration=15, sample_rate=16000):
    """
    마이크 입력을 실시간으로 서버로 스트리밍하고, 서버가 발화 완료를 알리면 즉시 종료합니다.
    """
    import sounddevice as sd
    print(f"🎤 [실시간 마이크] 말씀을 시작하세요! (최대 {duration}초 대기, 말을 멈추면 자동 분석)")
    
    chunk_samples = 480  # 30ms @ 16kHz
    loop = asyncio.get_event_loop()
    queue = asyncio.Queue()
    stop_event = asyncio.Event()

    def callback(indata, frames, time, status):
        if status:
            pass # 불필요한 경고 숨김
        if not stop_event.is_set():
            loop.call_soon_threadsafe(queue.put_nowait, bytes(indata))

    async def receive_from_server():
        """서버 응답을 기다리다가 최종 결과가 오면 스트리밍을 중단시킴"""
        try:
            while not stop_event.is_set():
                response = await websocket.recv()
                result = json.loads(response)
                
                # 서버가 발화 종료(is_final)를 판단하고 NLU 결과를 보냈을 때
                if result.get("is_final") and "nlu_analysis" in result:
                    print(f"\n  ✅ 서버 감지 완료! 텍스트: {result.get('transcript')}")
                    stop_event.set()  # 마이크 녹음 중지 신호
                    return result
        except websockets.exceptions.ConnectionClosed:
            print("  ❌ 서버 연결 종료")
        except Exception as e:
            print(f"  ❌ 수신 오류: {e}")
        return None

    # 스트림 시작
    try:
        # 서버 응답 수신 태스크를 백그라운드로 실행
        recv_task = asyncio.create_task(receive_from_server())
        
        with sd.InputStream(samplerate=sample_rate, channels=1, dtype='int16', 
                            blocksize=chunk_samples, callback=callback):
            start_time = time.time()
            while time.time() - start_time < duration and not stop_event.is_set():
                try:
                    # 큐에서 데이터를 가져와서 전송
                    data = await asyncio.wait_for(queue.get(), timeout=0.1)
                    if len(data) == 960 and not stop_event.is_set():
                        await websocket.send(data)
                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed:
                    break
                    
        # 루프를 빠져나왔다면(사용자가 말을 멈췄다면), 서버 응답을 마저 기다림
        if not stop_event.is_set():
             print("\n  ⏳ 최대 입력 시간 초과. 서버 분석을 기다립니다...")
             
        final_result = await recv_task
        print("  [마이크 입력 종료]")
        return final_result
        
    except Exception as e:
        print(f"  ❌ 마이크 스트리밍 중 오류: {e}")
        return None

async def send_audio_to_stt_nlu(mode="file", audio_file_path="0424.MP3"):
    """
    1단계: 오디오(파일 또는 마이크)를 WebSocket으로 전송하여 STT & NLU 결과를 받아옴.
    """
    print(f"\n🎙️ 1단계: STT & NLU 처리 시작 (모드: {mode})")
    
    audio_data = b""
    if mode == "file":
        if not os.path.exists(audio_file_path):
            print(f"❌ 오디오 파일을 찾을 수 없습니다: {audio_file_path}")
            return None
        try:
            audio = AudioSegment.from_file(audio_file_path)
            audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            audio_data = audio.raw_data
        except Exception as e:
            print(f"  ❌ 오디오 파일 로드 실패: {e}")
            return None

    # 2. WebSocket 연결 및 전송
    try:
        async with websockets.connect(WS_URL) as websocket:
            print("  [WebSocket 연결 성공] 데이터 전송 중...")
            
            if mode == "file":
                chunk_size = 960
                for i in range(0, len(audio_data), chunk_size):
                    chunk = audio_data[i:i + chunk_size]
                    if len(chunk) < chunk_size:
                        chunk = chunk + b'\x00' * (chunk_size - len(chunk))
                    await websocket.send(chunk)
                    await asyncio.sleep(0.005)

                print("  [전송 완료] 서버 VAD 종료 트리거를 위해 무음(Silence) 데이터를 보냅니다...")
                silence_chunk = b'\x00' * 960
                for _ in range(25):
                    await websocket.send(silence_chunk)
                    await asyncio.sleep(0.03)
                
                print("  [대기 중] 서버에서 분석 결과가 나올 때까지 기다립니다...")
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                    result = json.loads(response)
                except asyncio.TimeoutError:
                    print("  ❌ 타임아웃: 서버가 응답을 보내지 않았습니다.")
                    return None
            else:
                # 마이크 스트리밍 모드는 내부에서 이미 결과를 받아서 반환함
                result = await send_mic_streaming(websocket, duration=15)
                if not result:
                    return None
            
            if "error" in result:
                print(f"  ❌ 서버 에러: {result['error']}")
                return None
                
            print(f"  ✅ STT 텍스트: {result.get('transcript', 'N/A')}")
            print(f"  🧠 NLU 인텐트: {result.get('nlu_analysis', {}).get('intent', 'N/A')}")
            
            return result
            
    except Exception as e:
        print(f"  ❌ WebSocket 통신 실패: {e}")
        return None

def trigger_workflow_and_tts(nlu_result, output_audio_path="output_response.mp3"):
    """
    2단계: NLU 결과를 바탕으로 Workflow 실행 및 TTS 스트리밍 수신.
    """
    print("\n⚙️ 2단계: Workflow & TTS 스트리밍 시작")
    
    nlu_data = nlu_result.get("nlu_analysis", {})
    metadata = nlu_data.get("metadata", {})
    
    payload = {
        "session_id": nlu_result.get("session_id", "test_session"),
        "original_query": nlu_result.get("transcript", ""),
        "routing_info": {
            "intent": nlu_data.get("intent", "FAQ"),
            "subdomain": metadata.get("subdomain", "일반"),
            "domain": metadata.get("domain", "공통"),
            "router_confidence": 0.9,
            "metadata": metadata
        },
        "chat_history": [],
        "internal_context": []
    }
    
    try:
        start_time = time.time()
        print("  [POST 전송] 답변 생성 및 음성 합성 대기 중...")
        
        with requests.post(HTTP_URL, json=payload, stream=True) as response:
            if response.status_code != 200:
                print(f"  ❌ HTTP 통신 실패 (Status: {response.status_code})")
                return
                
            with open(output_audio_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
                            
            duration = time.time() - start_time
            print(f"\n  ✅ TTS 스트리밍 완료! (소요 시간: {duration:.2f}초)")
            print(f"  💾 저장 위치: {output_audio_path}")
            
    except Exception as e:
        print(f"  ❌ HTTP 통신 실패: {e}")

async def main():
    print("=" * 60)
    print("🚀 AICC E2E 통합 테스트 (파일 & 마이크 지원)")
    print("=" * 60)
    print("원하시는 테스트 모드를 선택하세요:")
    print("1. 파일 테스트 (0424.MP3)")
    print("2. 마이크 실시간 테스트")
    
    choice = input("\n번호 입력 (1 또는 2): ").strip()
    
    mode = "file" if choice == "1" else "mic"
    
    # 1단계 실행
    nlu_result = await send_audio_to_stt_nlu(mode=mode)
    
    if nlu_result:
        # 2단계 실행
        trigger_workflow_and_tts(nlu_result, "tests/test_response.mp3")
    else:
        print("\n❌ 1단계 실패로 인해 테스트를 중단합니다.")
        
    print("\n✨ 모든 테스트가 종료되었습니다.")

if __name__ == "__main__":
    # Windows 환경에서 asyncio 관련 에러 방지
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # 임시 테스트 폴더 생성
    os.makedirs("tests", exist_ok=True)
    asyncio.run(main())
