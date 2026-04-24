import requests
import time
import os

def test_tts_streaming(text: str = None):
    url = "http://localhost:8000/tts/stream"
    params = {"text": text} if text else {}
    
    print(f"[{time.strftime('%H:%M:%S')}] 서버에 스트리밍 요청을 보냅니다...")
    
    try:
        start_time = time.time()
        # stream=True로 설정하여 응답을 조각(Chunk) 단위로 받음
        response = requests.get(url, params=params, stream=True)
        response.raise_for_status()
        
        chunk_count = 0
        total_bytes = 0
        first_byte_time = None
        
        # 다운로드받은 오디오를 저장할 임시 파일
        output_file = "streamed_output.mp3"
        with open(output_file, 'wb') as f:
            # 1024 바이트(1KB) 단위로 스트림 읽기
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    if first_byte_time is None:
                        first_byte_time = time.time()
                        ttfb = first_byte_time - start_time
                        print(f"[{time.strftime('%H:%M:%S')}] ⚡ 첫 번째 오디오 청크 수신 완료! (TTFB: {ttfb:.3f}초)")
                        
                    chunk_count += 1
                    total_bytes += len(chunk)
                    f.write(chunk)
                    
                    # 수신 진행 상황 출력 (간략히)
                    if chunk_count % 50 == 0:
                        print(f"[{time.strftime('%H:%M:%S')}] ... 수신 중 (현재 {total_bytes / 1024:.1f} KB)")
                        
        total_time = time.time() - start_time
        print(f"\n✅ 스트리밍 수신 완료!")
        print(f"   - 총 소요 시간: {total_time:.2f}초")
        print(f"   - 다운로드 파일: {os.path.abspath(output_file)}")
        print(f"   - 총 파일 크기: {total_bytes / 1024:.2f} KB")

    except requests.exceptions.RequestException as e:
        print(f"❌ 요청 중 오류 발생: {e}")

if __name__ == "__main__":
    print("==============================================")
    print("   LLM + TTS HTTP 스트리밍 클라이언트 테스트   ")
    print("==============================================")
    print("서버가 실행 중인지 확인하세요 (uvicorn src.main:app --reload)\n")
    test_tts_streaming()
