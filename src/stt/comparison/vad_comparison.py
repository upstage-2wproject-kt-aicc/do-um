import time
import os
import torch
import numpy as np
from pydub import AudioSegment, silence
import webrtcvad

def compare_vad(audio_path):
    print(f"\n{'='*60}")
    print(f" 🔍 VAD 대상 파일: {audio_path}")
    print(f"{'='*60}")
    
    try:
        audio = AudioSegment.from_file(audio_path)
        audio_processed = audio.set_frame_rate(16000).set_channels(1)
        temp_wav = "temp_16k_processed.wav"
        audio_processed.export(temp_wav, format="wav")
    except Exception as e:
        print(f"❌ 오디오 로드 실패: {e}")
        return None

    total_len = len(audio_processed)

    # --- Test 1: Pydub ---
    print("[1] Pydub 처리 중...")
    start_t = time.time()
    chunks_pydub = silence.detect_nonsilent(audio_processed, min_silence_len=500, silence_thresh=-40)
    pydub_duration = time.time() - start_t
    pydub_speech_ms = sum([end - start for start, end in chunks_pydub])
    pydub_trim = (1 - (pydub_speech_ms / total_len)) * 100 if total_len > 0 else 0

    # --- Test 2: webrtcvad ---
    print("[2] webrtcvad 처리 중...")
    vad = webrtcvad.Vad(3)
    sample_rate, frame_duration = 16000, 30
    frame_size = int(sample_rate * frame_duration / 1000)
    raw_data = audio_processed.raw_data
    
    start_t = time.time()
    speech_frames = []
    for i in range(0, len(raw_data) - (frame_size * 2), frame_size * 2):
        frame = raw_data[i:i + (frame_size * 2)]
        speech_frames.append(vad.is_speech(frame, sample_rate))
    
    webrtc_speech_ms = sum([frame_duration for f in speech_frames if f])
    webrtc_duration = time.time() - start_t
    webrtc_trim = (1 - (webrtc_speech_ms / total_len)) * 100 if total_len > 0 else 0
    # 간단히 frame 변화 횟수를 세그먼트 개수로 추정 (또는 smoothing 적용 결과)
    segments_webrtc = len([i for i in range(1, len(speech_frames)) if speech_frames[i] != speech_frames[i-1]]) // 2

    # --- Test 3: Silero VAD ---
    print("[3] Silero VAD 처리 중...")
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', trust_repo=True, verbose=False)
    (get_speech_timestamps, _, _, _, _) = utils
    
    start_t = time.time()
    samples = np.array(audio_processed.get_array_of_samples(), dtype=np.float32) / 32768.0
    wav_tensor = torch.from_numpy(samples)
    speech_timestamps = get_speech_timestamps(wav_tensor, model, sampling_rate=16000, threshold=0.5)
    silero_duration = time.time() - start_t
    
    silero_speech_ms = sum([(ts['end'] - ts['start']) / 16 for ts in speech_timestamps])
    silero_trim = (1 - (silero_speech_ms / total_len)) * 100 if total_len > 0 else 0

    if os.path.exists(temp_wav): os.remove(temp_wav)

    return {
        "source_file": os.path.basename(audio_path),
        "metrics": [
            {"engine": "Pydub", "latency": round(pydub_duration, 4), "trim_rate": round(pydub_trim, 2), "segment_count": len(chunks_pydub)},
            {"engine": "webrtcvad", "latency": round(webrtc_duration, 4), "trim_rate": round(webrtc_trim, 2), "segment_count": segments_webrtc},
            {"engine": "Silero VAD", "latency": round(silero_duration, 4), "trim_rate": round(silero_trim, 2), "segment_count": len(speech_timestamps)}
        ]
    }


if __name__ == "__main__":
    import sys
    import json
    import glob
    from datetime import datetime

    data_dir = os.path.join("src", "stt", "data", "evaluation")
    
    # 하위 폴더를 포함하여 모든 오디오 파일 검색 (recursive)
    search_path = os.path.join(data_dir, "**", "*.*")
    all_files = glob.glob(search_path, recursive=True)
    audio_files = [f for f in all_files if f.lower().endswith(('.wav', '.mp3', '.m4a'))]

    if not audio_files:
        print(f"❌ 분석할 오디오 파일을 찾을 수 없습니다. 경로: {data_dir}")
        sys.exit(0)

    print(f"🚀 총 {len(audio_files)}개의 파일에 대해 VAD 비교 분석을 시작합니다. (10개 폴더 전수 조사)")
    print("-" * 60)
    
    all_results = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "total_files": len(audio_files),
        "results": []
    }

    for i, file_path in enumerate(audio_files):
        print(f"[{i+1}/{len(audio_files)}] 처리 중: {os.path.basename(file_path)}")
        res = compare_vad(file_path)
        if res:
            all_results["results"].append(res)

    result_dir = os.path.join("src", "stt", "comparison", "result")
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, "vad_metrics_latest.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 분석 완료! 100개 파일 메트릭 저장됨: vad_metrics_latest.json")


