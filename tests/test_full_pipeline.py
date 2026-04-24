import asyncio
import websockets
import json
import time
import requests
import wave
import sys
import os

# 서버 주소
WS_URL = "ws://localhost:8000/ws/stt"
HTTP_URL = "http://localhost:8000/pipeline/stream"

async def send_audio_to_stt_nlu(audio_file_path):
    """
    1단계: 오디오를 WebSocket으로 전송하여 STT & NLU 결과를 받아옴.
    """
    print(f"\n🎙️ 1단계: STT & NLU 처리 시작 ({audio_file_path})")
    
    # 1. 오디오 파일 로드
    if not os.path.exists(audio_file_path):
        print(f"❌ 오디오 파일을 찾을 수 없습니다: {audio_file_path}")
        return None
        
    with wave.open(audio_file_path, "rb") as wf:
        audio_data = wf.readframes(wf.getnframes())

    # 2. WebSocket 연결 및 전송
    try:
        async with websockets.connect(WS_URL) as websocket:
            print("  [WebSocket 연결 성공] 오디오 전송 중...")
            
            # 오디오 바이트 전송
            await websocket.send(audio_data)
            
            # STT 종료 신호 전송
            await websocket.send(json.dumps({"type": "end_of_stream"}))
            
            # 결과 대기
            response = await websocket.recv()
            result = json.loads(response)
            
            if "error" in result:
                print(f"  ❌ 서버 에러: {result['error']}")
                return None
                
            print(f"  ✅ STT 텍스트: {result.get('stt_text', 'N/A')}")
            print(f"  🧠 NLU 인텐트: {result.get('nlu_analysis', {}).get('predicted_intent', 'N/A')}")
            
            return result
            
    except Exception as e:
        print(f"  ❌ WebSocket 통신 실패: {e}")
        return None

def trigger_workflow_and_tts(nlu_result, output_audio_path="output_response.mp3"):
    """
    2단계: NLU 결과를 바탕으로 Workflow 실행 및 TTS 스트리밍 수신.
    """
    print("\n⚙️ 2단계: Workflow & TTS 스트리밍 시작")
    
    # 서버의 /pipeline/stream 엔드포인트는 WorkflowRoutingInput 스키마를 요구함.
    # NLU 결과에서 필요한 데이터 추출하여 Payload 구성
    nlu_data = nlu_result.get("nlu_analysis", {})
    payload = {
        "text": nlu_result.get("stt_text", ""),
        "routing_info": {
            "intent": nlu_data.get("predicted_intent", "Unknown"),
            "confidence": nlu_data.get("confidence_score", 0.0),
            "suggested_action": nlu_data.get("suggested_action", "chitchat")
        },
        "rag_context": nlu_data.get("rag_context", [])
    }
    
    try:
        # 스트리밍 요청
        start_time = time.time()
        print("  [POST 전송] 답변 생성 및 음성 합성 대기 중...")
        
        with requests.post(HTTP_URL, json=payload, stream=True) as response:
            if response.status_code != 200:
                print(f"  ❌ HTTP 통신 실패 (Status: {response.status_code})")
                print(f"  Detail: {response.text}")
                return
                
            # TTS 오디오 스트림 수신 및 저장
            with open(output_audio_path, "wb") as f:
                chunk_count = 0
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
                        chunk_count += 1
                        if chunk_count % 10 == 0:
                            print(".", end="", flush=True) # 진행 상태 표시
                            
            duration = time.time() - start_time
            print(f"\n  ✅ TTS 스트리밍 완료! (소요 시간: {duration:.2f}초)")
            print(f"  💾 저장 위치: {output_audio_path}")
            
    except Exception as e:
        print(f"  ❌ HTTP 통신 실패: {e}")

async def main():
    print("=" * 60)
    print("🚀 AICC E2E (STT -> NLU -> Workflow -> TTS) 통합 테스트")
    print("=" * 60)
    
    # 테스트에 사용할 로컬 오디오 파일 (16kHz Mono WAV 권장)
    # 임의로 src/stt/data/evaluation/subway/06_07_014041_210927_SN.wav 를 사용
    test_audio = "src/stt/data/evaluation/subway/06_07_014041_210927_SN.wav"
    output_audio = "tests/test_response.mp3"
    
    # 1단계 실행
    nlu_result = await send_audio_to_stt_nlu(test_audio)
    
    if nlu_result:
        # 2단계 실행
        trigger_workflow_and_tts(nlu_result, output_audio)
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
