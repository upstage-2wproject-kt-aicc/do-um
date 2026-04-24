import os
import time
import asyncio
import glob
from dotenv import load_dotenv
from openai import OpenAI
from google.cloud import speech_v2

# .env 파일에서 API 키 로드
load_dotenv()

async def transcribe_openai(file_path):
    """OpenAI Whisper API를 이용한 전사"""
    client = OpenAI(api_key=os.getenv("LLM_GPT_API_KEY"))
    start_time = time.time()
    try:
        with open(file_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        return response.strip(), time.time() - start_time
    except Exception as e:
        return f"OpenAI Error: {e}", 0

async def transcribe_xai(file_path):
    """xAI (Grok) API를 이용한 전사 (OpenAI 호환 규격 기반)"""
    # xAI는 OpenAI SDK와 호환되는 base_url을 제공합니다.
    client = OpenAI(
        api_key=os.getenv("XAI_API_KEY"),
        base_url="https://api.x.ai/v1",
    )
    start_time = time.time()
    try:
        with open(file_path, "rb") as audio_file:
            # xAI의 STT 모델명은 공식 문서에 따라 'grok-speech' 등으로 변경될 수 있습니다.
            response = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file,
                response_format="text"
            )
        return response.strip(), time.time() - start_time
    except Exception as e:
        return f"xAI Error (STT endpoint might be limited): {e}", 0

async def transcribe_google(file_path):
    """Google Cloud STT (Telephony)를 이용한 전사 (60초 초과 시 앞 50초만 사용)"""
    project_id = os.getenv("GOOGLE_PROJECT_ID")
    if not project_id:
        return "Google Project ID missing", 0
        
    client = speech_v2.SpeechClient()
    start_time = time.time()
    
    try:
        from pydub import AudioSegment
        import io
        
        # 오디오 로드 및 길이 확인
        audio = AudioSegment.from_file(file_path)
        if len(audio) > 60000:  # 60초 초과 확인
            print(f"  ⚠️ 오디오가 60초를 넘어 앞 50초만 변환합니다. ({len(audio)/1000:.1f}s -> 50.0s)")
            audio = audio[:50000]
            buffer = io.BytesIO()
            audio.export(buffer, format="wav")
            content = buffer.getvalue()
        else:
            with open(file_path, "rb") as audio_file:
                content = audio_file.read()

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
        return transcript, time.time() - start_time
    except Exception as e:
        return f"Google Error: {e}", 0

async def process_file(input_file):
    print(f"\n🚀 대상: {os.path.basename(input_file)}")
    
    # 태스크 리스트 생성
    tasks = [
        # transcribe_openai(input_file),  # API 키가 생기면 주석 해제하세요
        transcribe_google(input_file),
        # transcribe_xai(input_file)      # API 키가 생기면 주석 해제하세요
    ]
    
    # 병렬 실행
    results = await asyncio.gather(*tasks)
    text_gg, time_gg = results[0] 

    engines = [
        ("Google Telephony", text_gg, time_gg),
    ]

    metrics = []
    for name, text, duration in engines:
        print(f"  [{name}] 시간: {duration:.2f}초")
        # 더미 WER: 향후 jiwer 등을 활용한 실제 WER 계산 로직이 들어갈 자리
        dummy_wer = 0.15 if duration > 0 else 1.0 
        cost_per_min = 0.016 if "Google" in name else 0.0
        
        metrics.append({
            "engine": name,
            "latency": round(duration, 4),
            "text": text, # 제한 없이 전체 텍스트 저장
            "wer": dummy_wer,
            "cost_per_min": cost_per_min
        })


    return {
        "source_file": os.path.basename(input_file),
        "metrics": metrics
    }

async def main(sample: bool = False):
    import json
    from datetime import datetime

    data_dir = os.path.join("src", "stt", "data", "evaluation")
    if not os.path.exists(data_dir):
        print(f"❌ {data_dir} 폴더가 없습니다.")
        return

    if sample:
        target_subfolders = ["transaction_history", "subway", "grandma", "teen_sns"]
        audio_files = []
        print("📋 샘플 추출 중...")
        for folder in target_subfolders:
            folder_path = os.path.join(data_dir, folder)
            if os.path.isdir(folder_path):
                files = sorted(f for f in os.listdir(folder_path)
                               if f.lower().endswith(('.wav', '.mp3', '.m4a')))
                if files:
                    audio_files.append(os.path.join(folder_path, files[0]))
                    print(f"  ✅ [{folder}] -> {files[0]}")
                else:
                    print(f"  ⚠️ [{folder}] 오디오 파일 없음")
            else:
                print(f"  ⚠️ [{folder}] 폴더를 찾을 수 없음")
    else:
        all_files = glob.glob(os.path.join(data_dir, "**", "*.*"), recursive=True)
        audio_files = [f for f in all_files if f.lower().endswith(('.wav', '.mp3', '.m4a'))]

    if not audio_files:
        print("❌ 분석할 오디오 파일이 없습니다.")
        return

    test_type = "sample_test" if sample else "full_test"
    print(f"\n🚀 총 {len(audio_files)}개의 파일에 대해 STT 비교 분석을 시작합니다.")
    print("-" * 60)

    all_results = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "test_type": test_type,
        "files_processed": len(audio_files),
        "results": []
    }

    for file_path in audio_files:
        res = await process_file(file_path)
        if res:
            all_results["results"].append(res)

    result_dir = os.path.join("src", "stt", "comparison", "result")
    os.makedirs(result_dir, exist_ok=True)

    metric_file = os.path.join(result_dir, "stt_metrics_latest.json")
    with open(metric_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    temp_dir = os.path.join("src", "stt", "temp")
    if os.path.exists(temp_dir):
        for f in os.listdir(temp_dir):
            if f.endswith(('.wav', '.prof')):
                try: os.remove(os.path.join(temp_dir, f))
                except: pass

    print(f"\n✅ STT 분석 완료! 결과: stt_metrics_latest.json")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="STT 엔진 비교 분석")
    parser.add_argument("--sample", action="store_true",
                        help="4개 대표 도메인에서 1개씩 샘플링 후 분석")
    args = parser.parse_args()
    asyncio.run(main(sample=args.sample))
