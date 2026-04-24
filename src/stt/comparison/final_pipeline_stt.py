import os
import time
import asyncio
import json
import glob
import subprocess
from datetime import datetime
from dotenv import load_dotenv
from google.cloud import speech_v2
from pydub import AudioSegment
import webrtcvad

# .env 로드
load_dotenv()

# --- 1. 최적화된 전처리 파이프라인 (FFmpeg -> webrtcvad) ---
def apply_optimal_preprocessing(input_path, temp_dir):
    """FFmpeg NR 후 webrtcvad로 구간을 잘라 병합한 오디오 반환"""
    base_name = os.path.basename(input_path)
    nr_output = os.path.join(temp_dir, f"nr_ffmpeg_{base_name}.wav")
    final_output = os.path.join(temp_dir, f"final_vad_{base_name}.wav")
    
    # 1-1. FFmpeg NR 적용 (16kHz Mono 변환 포함)
    try:
        audio = AudioSegment.from_file(input_path).set_frame_rate(16000).set_channels(1)
        temp_input = os.path.join(temp_dir, "temp_in.wav")
        audio.export(temp_input, format="wav")
        cmd = ["ffmpeg", "-y", "-i", temp_input, "-af", "afftdn=nf=-25", "-loglevel", "error", nr_output]
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"  ⚠️ FFmpeg 처리 실패, 원본 사용: {e}")
        nr_output = temp_input # 실패 시 NR 건너뜀

    # 1-2. webrtcvad 적용 및 병합
    try:
        audio_nr = AudioSegment.from_file(nr_output)
        vad = webrtcvad.Vad(3)
        raw_data = audio_nr.raw_data
        frame_duration = 30
        frame_size = int(16000 * frame_duration / 1000)
        
        speech_chunks = []
        current_chunk = b""
        
        for i in range(0, len(raw_data) - (frame_size * 2), frame_size * 2):
            frame = raw_data[i:i + (frame_size * 2)]
            if vad.is_speech(frame, 16000):
                current_chunk += frame
            else:
                if current_chunk:
                    speech_chunks.append(current_chunk)
                    current_chunk = b""
        if current_chunk:
            speech_chunks.append(current_chunk)
            
        # 음성 구간만 모아서 새 오디오 생성
        if speech_chunks:
            final_raw = b"".join(speech_chunks)
            final_audio = AudioSegment(data=final_raw, sample_width=2, frame_rate=16000, channels=1)
            final_audio.export(final_output, format="wav")
            return final_output
        else:
            return nr_output # 전부 무음으로 판정되면 NR 결과물 그대로 리턴
    except Exception as e:
        print(f"  ⚠️ VAD 처리 실패, NR 결과물 사용: {e}")
        return nr_output

# --- 2. Google STT API 호출 ---
async def transcribe_google_optimal(file_path):
    project_id = os.getenv("GOOGLE_PROJECT_ID")
    if not project_id: return "Google Project ID missing", 0
        
    client = speech_v2.SpeechClient()
    start_t = time.time()
    
    try:
        import io
        audio = AudioSegment.from_file(file_path)
        if len(audio) > 60000:
            audio = audio[:50000] # API 제약 60초
            buffer = io.BytesIO()
            audio.export(buffer, format="wav")
            content = buffer.getvalue()
        else:
            with open(file_path, "rb") as f:
                content = f.read()

        config = speech_v2.types.RecognitionConfig(
            auto_decoding_config=speech_v2.types.AutoDetectDecodingConfig(),
            language_codes=["ko-KR"],
            model="telephony",
        )
        request = speech_v2.types.RecognizeRequest(
            recognizer=f"projects/{project_id}/locations/global/recognizers/_",
            config=config,
            content=content,
        )
        response = client.recognize(request=request)
        transcript = " ".join([result.alternatives[0].transcript for result in response.results])
        return transcript, time.time() - start_t
    except Exception as e:
        return f"Google Error: {e}", 0

# --- 3. 메인 실행 로직 ---
async def process_file(input_file, temp_dir):
    print(f"\n🚀 처리 중: {os.path.basename(input_file)}")
    
    # 1. 전처리 파이프라인
    start_pre = time.time()
    processed_audio_path = apply_optimal_preprocessing(input_file, temp_dir)
    pre_latency = time.time() - start_pre
    
    # 2. STT 호출
    text_gg, stt_latency = await transcribe_google_optimal(processed_audio_path)
    print(f"  [STT 결과] {text_gg[:50]}...")
    
    total_latency = pre_latency + stt_latency
    return {
        "source_file": os.path.basename(input_file),
        "pre_latency": round(pre_latency, 2),
        "stt_latency": round(stt_latency, 2),
        "total_latency": round(total_latency, 2),
        "text": text_gg
    }

async def main():
    data_dir = "src/stt/data/evaluation"
    temp_dir = "src/stt/temp/final_pipeline"
    os.makedirs(temp_dir, exist_ok=True)

    audio_files = glob.glob(os.path.join(data_dir, "**", "*.*"), recursive=True)
    audio_files = [f for f in audio_files if f.lower().endswith(('.wav', '.mp3', '.m4a'))]
    
    if not audio_files:
        print("❌ 테스트할 오디오 파일이 없습니다.")
        return

    print(f"🌟 [최종 파이프라인 검증] FFmpeg + webrtcvad -> Google STT 시작 ({len(audio_files)}개 파일)")
    print("-" * 60)

    all_results = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "pipeline": "FFmpeg -> webrtcvad -> Google Telephony",
        "files_processed": len(audio_files),
        "results": []
    }

    # API 동시 호출 제한을 위해 청크(chunk) 단위로 나누어 실행
    chunk_size = 5 
    for i in range(0, len(audio_files), chunk_size):
        chunk_files = audio_files[i:i + chunk_size]
        tasks = [process_file(f, temp_dir) for f in chunk_files]
        chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
        all_results["results"].extend(
            [r for r in chunk_results if r and not isinstance(r, Exception)]
        )

    # 결과 저장
    result_dir = "src/stt/comparison/result"
    os.makedirs(result_dir, exist_ok=True)
    metric_file = os.path.join(result_dir, "final_pipeline_stt_result.json")
    
    with open(metric_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # 임시 파일 정리
    for f in os.listdir(temp_dir):
        try: os.remove(os.path.join(temp_dir, f))
        except: pass

    print(f"\n✅ 모든 파이프라인 처리가 완료되었습니다! 결과: {metric_file}")

if __name__ == "__main__":
    asyncio.run(main())
