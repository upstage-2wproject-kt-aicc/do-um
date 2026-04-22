import time
import os
import torch
import numpy as np
from pydub import AudioSegment, silence
import webrtcvad

def compare_vad(audio_path):
    print(f"\n{'='*60}")
    print(f" 🔍 VAD 3종 성능 비교 분석 시작")
    print(f" 📂 대상 파일: {audio_path}")
    print(f"{'='*60}")
    
    # 1. 오디오 로드 및 전처리 (16kHz, Mono)
    try:
        audio = AudioSegment.from_file(audio_path)
        audio_processed = audio.set_frame_rate(16000).set_channels(1)
        temp_wav = "temp_16k_processed.wav"
        audio_processed.export(temp_wav, format="wav")
    except Exception as e:
        print(f"❌ 오디오 로드 실패: {e}")
        return

    # --- Test 1: Pydub (Energy-based) ---
    print("\n[1] Pydub (볼륨 에너지 기반)")
    start_time = time.time()
    chunks = silence.detect_nonsilent(audio_processed, min_silence_len=500, silence_thresh=-40)
    pydub_duration = time.time() - start_time
    for i, (start, end) in enumerate(chunks):
        print(f"  ✅ 구간 {i+1}: {start/1000:.2f}s ~ {end/1000:.2f}s ({(end-start)/1000:.2f}s)")
    print(f"  ⏱️ 소요시간: {pydub_duration:.4f}초")

    # --- Test 2: webrtcvad (Statistical 기반) ---
    print("\n[2] webrtcvad (Google 개발 알고리즘)")
    vad = webrtcvad.Vad(3) # 민감도 3 (0~3)
    sample_rate = 16000
    frame_duration = 30 # ms
    frame_size = int(sample_rate * frame_duration / 1000) # 480 samples
    
    raw_data = audio_processed.raw_data
    start_time = time.time()
    
    # 30ms 단위의 True/False 결과 수집
    speech_frames = []
    for i in range(0, len(raw_data) - (frame_size * 2), frame_size * 2):
        frame = raw_data[i:i + (frame_size * 2)]
        is_speech = vad.is_speech(frame, sample_rate)
        speech_frames.append(is_speech)
    
    # --- 프레임 묶기 (Smoothing) ---
    chunks = []
    current_start = None
    silence_count = 0
    # 최소 500ms의 묵음을 허용 (30ms 프레임 * 17개 ≒ 510ms)
    max_silence_frames = 17 
    
    for i, is_speech in enumerate(speech_frames):
        current_time_s = i * frame_duration / 1000.0
        
        if is_speech:
            if current_start is None:
                current_start = current_time_s # 말소리 시작
            silence_count = 0 # 묵음 카운트 리셋
        else:
            if current_start is not None:
                silence_count += 1
                # 묵음이 허용치 이상 지속되면 구간 닫기
                if silence_count >= max_silence_frames:
                    end_time_s = (i - silence_count) * frame_duration / 1000.0
                    chunks.append((current_start, end_time_s))
                    current_start = None
                    silence_count = 0
                    
    # 마지막으로 닫히지 않은 구간이 있다면 닫기
    if current_start is not None:
        end_time_s = len(speech_frames) * frame_duration / 1000.0
        chunks.append((current_start, end_time_s))

    webrtc_duration = time.time() - start_time
    
    if not chunks:
        print("  ⚠️ 검출된 구간 없음")
    for i, (start, end) in enumerate(chunks):
        print(f"  ✅ 구간 {i+1}: {start:.2f}s ~ {end:.2f}s (길이: {end-start:.2f}s)")
    print(f"  ⏱️ 소요시간: {webrtc_duration:.4f}초")

    # --- Test 3: Silero VAD (AI 딥러닝 기반) ---
    print("\n[3] Silero VAD (AI 딥러닝 기반)")
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', trust_repo=True, verbose=False)
    (get_speech_timestamps, _, _, _, _) = utils
    
    start_time = time.time()
    samples = np.array(audio_processed.get_array_of_samples(), dtype=np.float32) / 32768.0
    wav_tensor = torch.from_numpy(samples)
    speech_timestamps = get_speech_timestamps(wav_tensor, model, sampling_rate=16000, threshold=0.5)
    silero_duration = time.time() - start_time
    
    for i, ts in enumerate(speech_timestamps):
        s, e = ts['start']/16000, ts['end']/16000
        print(f"  ✅ 구간 {i+1}: {s:.2f}s ~ {e:.2f}s ({e-s:.2f}s)")
    print(f"  ⏱️ 소요시간: {silero_duration:.4f}초")

    print(f"\n{'='*60}")
    print(f" ✨ 분석 완료")
    print(f"{'='*60}")
    if os.path.exists(temp_wav): os.remove(temp_wav)

if __name__ == "__main__":
    import sys
    target_file = sys.argv[1] if len(sys.argv) > 1 else None
    if not target_file:
        for f in os.listdir('.'):
            if f.lower().endswith(('.wav', '.mp3', '.m4a')) and not f.startswith('temp_'):
                target_file = f; break
    if target_file: compare_vad(target_file)
    else: print("❌ 파일을 찾을 수 없습니다.")
