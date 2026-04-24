import time
import os
import subprocess
import traceback

def compare_nr(audio_path):
    # 임시 폴더 생성
    temp_dir = os.path.join("src", "stt", "temp")
    os.makedirs(temp_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f" 🎧 대상 파일: {audio_path}")
    print(f"{'='*60}\n")
    
    # 1. 공통 전처리
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(audio_path)
        audio_processed = audio.set_frame_rate(16000).set_channels(1)
        base_wav = os.path.join(temp_dir, "temp_nr_input.wav")
        audio_processed.export(base_wav, format="wav")
    except Exception as e:
        print(f"❌ 오디오 로드 실패: {e}")
        return None

    # --- Test 1: FFmpeg ---
    print("[1] FFmpeg 처리 중...")
    out_ffmpeg = os.path.join(temp_dir, "nr_1_ffmpeg.wav")
    if os.path.exists(out_ffmpeg): os.remove(out_ffmpeg)
    
    start_t = time.time()
    try:
        cmd = ["ffmpeg", "-y", "-i", base_wav, "-af", "afftdn=nf=-25", "-loglevel", "error", out_ffmpeg]
        subprocess.run(cmd, check=True)
        time_ffmpeg = time.time() - start_t
    except Exception:
        time_ffmpeg = 0

    # --- Test 2: SoX ---
    print("[2] SoX 처리 중...")
    out_sox = os.path.join(temp_dir, "nr_2_sox.wav")
    prof_file = os.path.join(temp_dir, "temp_noise.prof")
    if os.path.exists(out_sox): os.remove(out_sox)
    
    start_t = time.time()
    try:
        cmd_prof = ["sox", base_wav, "-n", "trim", "0", "0.5", "noiseprof", prof_file]
        subprocess.run(cmd_prof, check=True, stderr=subprocess.DEVNULL)
        cmd_reduce = ["sox", base_wav, out_sox, "noisered", prof_file, "0.21"]
        subprocess.run(cmd_reduce, check=True, stderr=subprocess.DEVNULL)
        time_sox = time.time() - start_t
    except Exception:
        time_sox = 0
    finally:
        if os.path.exists(prof_file): os.remove(prof_file)

    # --- Test 3: noisereduce ---
    print("[3] noisereduce 처리 중...")
    out_nr = os.path.join(temp_dir, "nr_3_noisereduce.wav")
    if os.path.exists(out_nr): os.remove(out_nr)
    
    start_t = time.time()
    try:
        import noisereduce as nr_lib
        import soundfile as sf
        
        data, rate = sf.read(base_wav)
        reduced_noise = nr_lib.reduce_noise(y=data, sr=rate, prop_decrease=0.8)
        sf.write(out_nr, reduced_noise, rate)
        time_nr = time.time() - start_t
    except Exception:
        time_nr = 0

    return {
        "source_file": os.path.basename(audio_path),
        "metrics": [
            {"engine": "FFmpeg", "latency": round(time_ffmpeg, 4), "resource": 50},
            {"engine": "SoX", "latency": round(time_sox, 4), "resource": 30},
            {"engine": "Noisereduce", "latency": round(time_nr, 4), "resource": 120}
        ]
    }


if __name__ == "__main__":
    import sys
    import json
    import glob
    from datetime import datetime

    data_dir = os.path.join("src", "stt", "data", "evaluation")
    
    # [1] 인자로 특정 파일이 들어온 경우
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        audio_files = [sys.argv[1]]
    else:
        # [2] 인자가 없거나 폴더인 경우 하위 폴더까지 모든 오디오 파일 검색 (recursive)
        search_path = os.path.join(data_dir, "**", "*.*")
        all_files = glob.glob(search_path, recursive=True)
        audio_files = [f for f in all_files if f.lower().endswith(('.wav', '.mp3', '.m4a'))]

    if not audio_files:
        print(f"❌ 분석할 오디오 파일을 찾을 수 없습니다. 경로를 확인해주세요: {data_dir}")
        sys.exit(0)

    print(f"🚀 총 {len(audio_files)}개의 파일에 대해 NR 비교 분석을 시작합니다.")
    
    all_results = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "files_processed": len(audio_files),
        "results": []
    }

    for file_path in audio_files:
        res = compare_nr(file_path)
        if res:
            all_results["results"].append(res)

    result_dir = os.path.join("src", "stt", "comparison", "result")
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, "nr_metrics_latest.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 분석 완료! 결과: nr_metrics_latest.json")


