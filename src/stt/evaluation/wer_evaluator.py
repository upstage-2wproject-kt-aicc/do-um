import json
import pandas as pd
import os
import jiwer
import re
from datetime import datetime

def normalize_text(text):
    """평가를 위한 텍스트 정규화: 특수문자 제거 및 공백 1개로 통일"""
    if not text:
        return ""
    # 한글, 숫자, 영문, 공백만 남기고 특수기호 모두 제거
    text = re.sub(r'[^가-힣0-9a-zA-Z\s]', '', text)
    # 연속된 공백을 1개로 압축
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def run_wer_evaluation(model_type="google"):
    print(f"📊 [{model_type.upper()} STT 최종 벤치마크 전수 조사]를 시작합니다")
    print("-" * 110)

    # 1. 경로 설정
    gt_path = "src/stt/data/evaluation/web_test.csv"
    result_filename = f"final_pipeline_{model_type}_stt_result.json"
    stt_result_path = os.path.join("src", "stt", "comparison", "result", result_filename)
    
    if not os.path.exists(gt_path) or not os.path.exists(stt_result_path):
        print(f"❌ 필요한 파일({gt_path} 또는 {stt_result_path})이 없습니다.")
        return

    # 2. 데이터 로드
    df_gt = pd.read_csv(gt_path)
    with open(stt_result_path, "r", encoding="utf-8") as f:
        stt_data = json.load(f)
    
    # JSON 결과를 {파일명: 상세데이터} 딕셔너리로 변환
    stt_map = {res['source_file']: res for res in stt_data['results']}

    # 3. 데이터 분석 및 계산
    results = []
    domain_stats = {} 
    
    total_wer, total_cer, total_lat, valid_count = 0, 0, 0, 0

    print(f"{'Folder':<20} | {'Filename':<30} | {'WER (%)':<8} | {'CER (%)':<8} | {'Lat (s)':<8}")
    print("-" * 110)

    for _, row in df_gt.iterrows():
        folder = row['foldername']
        fname = row['filename']
        raw_truth = str(row['ground_truth']).strip()
        
        if not raw_truth or raw_truth == "nan": continue
            
        stt_entry = stt_map.get(fname)
        if not stt_entry: continue

        raw_prediction = stt_entry['text'].strip()
        latency = stt_entry.get('total_latency', 0)

        # 정규화 및 에러율 계산
        truth = normalize_text(raw_truth)
        prediction = normalize_text(raw_prediction)
        
        if not truth or not prediction: continue
             
        error_rate_wer = jiwer.wer(truth, prediction)
        error_rate_cer = jiwer.cer(truth, prediction)
        
        res_entry = {
            "folder": folder,
            "filename": fname,
            "wer": round(error_rate_wer * 100, 2),
            "cer": round(error_rate_cer * 100, 2),
            "latency": latency
        }
        results.append(res_entry)
        
        # 도메인별 합산
        if folder not in domain_stats:
            domain_stats[folder] = {"wer_sum": 0, "cer_sum": 0, "lat_sum": 0, "count": 0}
        domain_stats[folder]["wer_sum"] += error_rate_wer
        domain_stats[folder]["cer_sum"] += error_rate_cer
        domain_stats[folder]["lat_sum"] += latency
        domain_stats[folder]["count"] += 1

        total_wer += error_rate_wer
        total_cer += error_rate_cer
        total_lat += latency
        valid_count += 1
        
        print(f"{folder:<20} | {fname[:30]:<30} | {error_rate_wer*100:<8.2f}% | {error_rate_cer*100:<8.2f}% | {latency:<8.2f}s")

    # 4. 결과 요약 출력
    if valid_count > 0:
        print("\n" + "="*110)
        print(f"📈 [종합 {model_type.upper()} 도메인별 평균 성적표]")
        print("-" * 85)
        print(f"{'Domain Folder':<25} | {'Avg WER (%)':<15} | {'Avg CER (%)':<15} | {'Avg Lat (s)':<15}")
        print("-" * 85)
        
        for folder, stats in domain_stats.items():
            avg_w = (stats["wer_sum"] / stats["count"]) * 100
            avg_c = (stats["cer_sum"] / stats["count"]) * 100
            avg_l = (stats["lat_sum"] / stats["count"])
            print(f"{folder:<25} | {avg_w:<15.2f}% | {avg_c:<15.2f}% | {avg_l:<15.2f}s")
            
        print("-" * 85)
        print(f"🌟 전체 평균 (총 {valid_count}개 파일):")
        print(f"   - Average WER: {(total_wer/valid_count)*100:.2f}%")
        print(f"   - Average CER: {(total_cer/valid_count)*100:.2f}%")
        print(f"   - Average Latency: {(total_lat/valid_count):.2f}s")
        print("="*110)
        
        # 최종 결과 저장
        report_dir = "src/stt/comparison/result"
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, f"final_{model_type}_stt_accuracy_report.json")
        
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump({
                "model": model_type,
                "overall": {
                    "avg_wer": (total_wer/valid_count)*100,
                    "avg_cer": (total_cer/valid_count)*100,
                    "avg_latency": (total_lat/valid_count),
                    "total_files": valid_count
                },
                "domain_summary": domain_stats,
                "details": results
            }, f, ensure_ascii=False, indent=2)
        print(f"✅ {model_type.upper()} 상세 리포트 저장 완료: {report_path}")

if __name__ == "__main__":
    import sys
    # 인자로 google 또는 openai를 받음 (기본값 google)
    m_type = sys.argv[1] if len(sys.argv) > 1 else "google"
    run_wer_evaluation(m_type)
