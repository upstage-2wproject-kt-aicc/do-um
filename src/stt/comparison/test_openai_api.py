import os
import time
import asyncio
import glob
import subprocess
from dotenv import load_dotenv
from openai import OpenAI
from pydub import AudioSegment
import webrtcvad

# .env 로드 (LLM_GPT_API_KEY 확인)
load_dotenv()

# --- 1. 최적화된 전처리 파이프라인 (FFmpeg -> webrtcvad) ---
def apply_optimal_preprocessing(input_path, temp_dir):
    """FFmpeg NR 후 webrtcvad로 구간을 잘라 병합한 오디오 반환"""
    base_name = os.path.basename(input_path)
    nr_output = os.path.join(temp_dir, f"nr_ffmpeg_{base_name}.wav")
    final_output = os.path.join(temp_dir, f"final_vad_{base_name}.wav")
    
    # 1-1. FFmpeg NR 적용
    try:
        audio = AudioSegment.from_file(input_path).set_frame_rate(16000).set_channels(1)
        temp_input = os.path.join(temp_dir, "temp_in.wav")
        audio.export(temp_input, format="wav")
        cmd = ["ffmpeg", "-y", "-i", temp_input, "-af", "afftdn=nf=-25", "-loglevel", "error", nr_output]
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"  ⚠️ FFmpeg 처리 실패, 원본 사용: {e}")
        nr_output = temp_input

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
            return final_output
        else:
            return nr_output
    except Exception as e:
        print(f"  ⚠️ VAD 처리 실패, NR 결과물 사용: {e}")
        return nr_output

# --- 2. OpenAI STT API 호출 ---
async def transcribe_openai_optimal(file_path):
    api_key = os.getenv("LLM_GPT_API_KEY")
    if not api_key:
        return "❌ LLM_GPT_API_KEY가 설정되지 않았습니다.", 0
        
    client = OpenAI(api_key=api_key)
    start_t = time.time()
    
    try:
        with open(file_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1", # 현재 최신 Whisper V3 모델 자동 라우팅됨
                file=audio_file,
                response_format="text"
            )
        return response.strip(), time.time() - start_t
    except Exception as e:
        return f"OpenAI Error: {e}", 0

# --- 3. 메인 실행 로직 ---
async def process_file(input_file, temp_dir):
    print(f"\n🚀 처리 중: {os.path.basename(input_file)}")
    
    # 1. 전처리 파이프라인
    start_pre = time.time()
    processed_audio_path = apply_optimal_preprocessing(input_file, temp_dir)
    pre_latency = time.time() - start_pre
    
    # 2. STT 호출
    text_oa, stt_latency = await transcribe_openai_optimal(processed_audio_path)
    
    print(f"  [STT 결과] {text_oa[:100]}..." if len(text_oa) > 100 else f"  [STT 결과] {text_oa}")
    print(f"  ⏱️ 소요 시간: 전처리({pre_latency:.2f}s) + API호출({stt_latency:.2f}s) = {pre_latency+stt_latency:.2f}초")
    
    return True

async def main():
    data_dir = "src/stt/data/evaluation"
    temp_dir = "src/stt/temp/openai_test"
    os.makedirs(temp_dir, exist_ok=True)

    # 모든 오디오 파일 찾아서 상위 5개만 슬라이싱
    audio_files = glob.glob(os.path.join(data_dir, "**", "*.*"), recursive=True)
    audio_files = [f for f in audio_files if f.lower().endswith(('.wav', '.mp3', '.m4a'))][:5]
    
    if not audio_files:
        print("❌ 테스트할 오디오 파일이 없습니다.")
        return

    print(f"🌟 [OpenAI API 연동 테스트] FFmpeg + webrtcvad -> OpenAI Whisper 시작 (5개 샘플)")
    print("-" * 65)

    for f in audio_files:
        await process_file(f, temp_dir)

    # 임시 파일 정리
    for f in os.listdir(temp_dir):
        try: os.remove(os.path.join(temp_dir, f))
        except: pass

    print(f"\n✅ OpenAI 연동 테스트가 완료되었습니다!")

if __name__ == "__main__":
    asyncio.run(main())
