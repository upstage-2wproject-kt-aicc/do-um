import os
import time
import asyncio
import json
import glob
import subprocess
import uuid
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from pydub import AudioSegment
import webrtcvad

# .env 로드
load_dotenv()

# --- 1. 최적화된 전처리 파이프라인 (FFmpeg -> webrtcvad) ---
def apply_optimal_preprocessing(input_path, temp_dir):
    """FFmpeg NR 후 webrtcvad로 구간을 잘라 병합한 오디오 반환"""
    base_name = os.path.basename(input_path)
    uid = str(uuid.uuid4())[:8]
    
    nr_output = os.path.join(temp_dir, f"nr_ffmpeg_{uid}_{base_name}.wav")
    final_output = os.path.join(temp_dir, f"final_vad_{uid}_{base_name}.wav")
    temp_input = os.path.join(temp_dir, f"temp_in_{uid}.wav")
    
    # 1-1. FFmpeg NR 적용
    try:
        audio = AudioSegment.from_file(input_path).set_frame_rate(16000).set_channels(1)
        audio.export(temp_input, format="wav")
        cmd = ["ffmpeg", "-y", "-i", temp_input, "-af", "afftdn=nf=-25", "-loglevel", "error", nr_output]
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"  ⚠️ FFmpeg 처리 실패, 원본 사용: {e}")
        nr_output = temp_input
    finally:
        if os.path.exists(temp_input) and temp_input != nr_output:
            try: os.remove(temp_input)
            except: pass

    # 1-2. webrtcvad 적용
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
            
        if speech_chunks:
            final_raw = b"".join(speech_chunks)
            final_audio = AudioSegment(data=final_raw, sample_width=2, frame_rate=16000, channels=1)
            final_audio.export(final_output, format="wav")
            return final_output, nr_output
        else:
            return nr_output, None
    except Exception as e:
        print(f"  ⚠️ VAD 처리 실패, NR 결과물 사용: {e}")
        return nr_output, None

# --- 2. OpenAI STT API 호출 ---
async def transcribe_openai_optimal(file_path):
    api_key = os.getenv("LLM_GPT_API_KEY")
    if not api_key: return "LLM_GPT_API_KEY missing", 0
        
    client = OpenAI(api_key=api_key)
    start_t = time.time()
    
    try:
        import io
        audio = AudioSegment.from_file(file_path)
        
        # [중요] 구글 STT와 동일한 절단 로직 적용
        if len(audio) > 60000:
            audio = audio[:50000] # 50초로 절단
            buffer = io.BytesIO()
            audio.export(buffer, format="wav")
            buffer.seek(0)
            audio_file_tuple = ("temp.wav", buffer, "audio/wav")
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file_tuple,
                response_format="text"
            )
        else:
            # [버그 수정] with 문을 사용하여 파일이 자동으로 닫히도록 보장 (윈도우 잠금 해결)
            with open(file_path, "rb") as f:
                response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    response_format="text"
                )
        return response.strip(), time.time() - start_t
    except Exception as e:
        return f"OpenAI Error: {e}", 0

# --- 3. 메인 실행 로직 ---
async def process_file(input_file, temp_dir):
    print(f"🚀 처리 중: {os.path.basename(input_file)}")
    
    start_pre = time.time()
    processed_audio_path, mid_audio_path = apply_optimal_preprocessing(input_file, temp_dir)
    pre_latency = time.time() - start_pre
    
    text_oa, stt_latency = await transcribe_openai_optimal(processed_audio_path)
    
    # [중요] STT 완료 후 해당 파일의 임시 파일 즉시 삭제
    for p in [processed_audio_path, mid_audio_path]:
        if p and os.path.exists(p):
            try: os.remove(p)
            except: pass
            
    total_latency = pre_latency + stt_latency
    return {
        "source_file": os.path.basename(input_file),
        "pre_latency": round(pre_latency, 2),
        "stt_latency": round(stt_latency, 2),
        "total_latency": round(total_latency, 2),
        "text": text_oa
    }

async def main():
    data_dir = "src/stt/data/evaluation"
    temp_dir = "src/stt/temp/final_pipeline_openai"
    os.makedirs(temp_dir, exist_ok=True)

    audio_files = glob.glob(os.path.join(data_dir, "**", "*.*"), recursive=True)
    audio_files = [f for f in audio_files if f.lower().endswith(('.wav', '.mp3', '.m4a'))]
    
    if not audio_files:
        print("❌ 테스트할 오디오 파일이 없습니다.")
        return

    print(f"🌟 [최종 파이프라인 검증] FFmpeg + webrtcvad -> OpenAI Whisper 시작 ({len(audio_files)}개 파일)")
    print("-" * 70)

    all_results = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "pipeline": "FFmpeg -> webrtcvad -> OpenAI Whisper",
        "files_processed": len(audio_files),
        "results": []
    }

    # API 동시 호출 제한을 위해 청크 단위 실행
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
    metric_file = os.path.join(result_dir, "final_pipeline_openai_stt_result.json")
    
    with open(metric_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # 잔여 임시 파일 정리 (혹시 남은 것들)
    if os.path.exists(temp_dir):
        for f in os.listdir(temp_dir):
            try: os.remove(os.path.join(temp_dir, f))
            except: pass

    print(f"\n✅ OpenAI 파이프라인 전수 조사 완료! 결과: {metric_file}")

if __name__ == "__main__":
    asyncio.run(main())
