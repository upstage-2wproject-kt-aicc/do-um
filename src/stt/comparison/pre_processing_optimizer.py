import os
import time
import json
import glob
import torch
import numpy as np
import subprocess
from datetime import datetime
from pydub import AudioSegment, silence
import webrtcvad

# --- 전처리 엔진 정의 ---
def apply_nr(input_path, nr_type, temp_dir):
    """노이즈 제거 라이브러리 적용"""
    output_path = os.path.join(temp_dir, f"temp_nr_{nr_type}.wav")
    try:
        if nr_type == "ffmpeg":
            subprocess.run(["ffmpeg", "-y", "-i", input_path, "-af", "afftdn=nf=-25", "-loglevel", "error", output_path], check=True)
        elif nr_type == "sox":
            prof = os.path.join(temp_dir, "noise.prof")
            subprocess.run(["sox", input_path, "-n", "trim", "0", "0.5", "noiseprof", prof], check=True, stderr=subprocess.DEVNULL)
            subprocess.run(["sox", input_path, output_path, "noisered", prof, "0.21"], check=True, stderr=subprocess.DEVNULL)
        elif nr_type == "noisereduce":
            import noisereduce as nr_lib
            import soundfile as sf
            data, rate = sf.read(input_path)
            reduced = nr_lib.reduce_noise(y=data, sr=rate)
            sf.write(output_path, reduced, rate)
        return output_path
    except Exception as e:
        print(f"  ⚠️ NR({nr_type}) 처리 실패, 원본 사용: {e}")
        return input_path

def apply_vad(audio_path, vad_type, silero_model=None, silero_utils=None):
    """VAD 라이브러리 적용 및 결과 지표 반환"""
    audio = AudioSegment.from_file(audio_path).set_frame_rate(16000).set_channels(1)
    total_ms = len(audio)
    speech_ms = 0
    segments = 0

    if vad_type == "pydub":
        chunks = silence.detect_nonsilent(audio, min_silence_len=500, silence_thresh=-40)
        speech_ms = sum([end - start for start, end in chunks])
        segments = len(chunks)
    elif vad_type == "webrtcvad":
        vad = webrtcvad.Vad(3)
        raw_data = audio.raw_data
        frame_duration = 30
        frame_size = int(16000 * frame_duration / 1000)
        speech_frames = 0
        for i in range(0, len(raw_data) - (frame_size * 2), frame_size * 2):
            frame = raw_data[i:i + (frame_size * 2)]
            if vad.is_speech(frame, 16000):
                speech_frames += 1
        speech_ms = speech_frames * frame_duration
        segments = 0 # webrtcvad는 단순 프레임 기반이므로 0 처리
    elif vad_type == "silero":
        get_speech_timestamps = silero_utils[0]
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
        wav_tensor = torch.from_numpy(samples)
        stamps = get_speech_timestamps(wav_tensor, silero_model, sampling_rate=16000, threshold=0.5)
        speech_ms = sum([(ts['end'] - ts['start']) / 16 for ts in stamps])
        segments = len(stamps)

    trim_rate = (1 - (speech_ms / total_ms)) * 100 if total_ms > 0 else 0
    return {"trim_rate": round(trim_rate, 2), "segment_count": segments}

# --- 메인 최적화 로직 ---
def run_optimization():
    print("🚀 [전처리 조합 최적화 시뮬레이션] 시작 (STT 비용 0원)")
    
    # 1. 환경 설정
    data_dir = "src/stt/data/evaluation"
    temp_dir = "src/stt/temp/optimizer"
    os.makedirs(temp_dir, exist_ok=True)
    
    # 2. 모델 로드 (Silero VAD)
    print("📦 Silero VAD 모델 로딩 중...")
    silero_model, silero_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', trust_repo=True, verbose=False)

    # 3. 실험군 설정
    nr_engines = ["ffmpeg", "sox", "noisereduce"]
    vad_engines = ["pydub", "webrtcvad", "silero"]
    
    audio_files = glob.glob(os.path.join(data_dir, "**", "*.*"), recursive=True)
    audio_files = [f for f in audio_files if f.lower().endswith(('.wav', '.mp3', '.m4a'))]
    
    print(f"📂 총 대상 파일: {len(audio_files)}개 (100개 전수 조사 시작)")



    
    all_metrics = []

    for i, file_path in enumerate(audio_files):
        print(f"[{i+1}/{len(audio_files)}] 분석 중: {os.path.basename(file_path)}")
        
        for nr in nr_engines:
            start_nr = time.time()
            nr_audio = apply_nr(file_path, nr, temp_dir)
            nr_latency = time.time() - start_nr
            
            for vad in vad_engines:
                start_vad = time.time()
                vad_result = apply_vad(nr_audio, vad, silero_model, silero_utils)
                vad_latency = time.time() - start_vad
                
                all_metrics.append({
                    "file": os.path.basename(file_path),
                    "combination": f"{nr} + {vad}",
                    "nr_engine": nr,
                    "vad_engine": vad,
                    "trim_rate": vad_result["trim_rate"],
                    "total_latency": round(nr_latency + vad_latency, 4)
                })

    # 4. 결과 요약 및 점수 산출
    # Trim Rate(40%) + Latency(60%) 비중으로 간단 평가
    summary = []
    for combo in [f"{n} + {v}" for n in nr_engines for v in vad_engines]:
        combo_data = [m for m in all_metrics if m["combination"] == combo]
        avg_trim = sum([d["trim_rate"] for d in combo_data]) / len(combo_data)
        avg_lat = sum([d["total_latency"] for d in combo_data]) / len(combo_data)
        
        # 점수화 (단순화: Trim은 높을수록, Latency는 낮을수록)
        score = (avg_trim * 0.5) + (max(0, 10 - avg_lat) * 5)
        
        summary.append({
            "combination": combo,
            "avg_trim_rate": round(avg_trim, 2),
            "avg_latency": round(avg_lat, 4),
            "optimization_score": round(score, 2)
        })

    summary.sort(key=lambda x: x["optimization_score"], reverse=True)

    # 5. 저장
    result_dir = "src/stt/comparison/result"
    os.makedirs(result_dir, exist_ok=True)
    report_path = os.path.join(result_dir, "preprocessing_optimization_report.json")
    
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "raw_data": all_metrics}, f, ensure_ascii=False, indent=2)

    print("\n" + "="*50)
    print("🏆 [최종 추천 전처리 조합 순위]")
    for i, s in enumerate(summary[:3]):
        print(f"{i+1}위: {s['combination']} (Score: {s['optimization_score']})")
    print("="*50)
    print(f"✅ 상세 리포트 저장됨: {report_path}")

if __name__ == "__main__":
    run_optimization()
