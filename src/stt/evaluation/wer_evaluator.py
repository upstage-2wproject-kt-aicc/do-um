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

def run_wer_evaluation():
    print("📊 [구글 STT 최종 정확도(WER/CER) 전수 조사]를 시작합니다 (대상: 100개 파일)")
    print("-" * 90)

    # 1. 경로 설정
    gt_path = "src/stt/data/evaluation/web_test.csv"
    stt_result_path = "src/stt/comparison/result/final_pipeline_google_stt_result.json"
    
    if not os.path.exists(gt_path) or not os.path.exists(stt_result_path):
        print("❌ 필요한 파일(web_test.csv 또는 결과 JSON)이 없습니다.")
        return

    # 2. 데이터 로드
    df_gt = pd.read_csv(gt_path)
    with open(stt_result_path, "r", encoding="utf-8") as f:
        stt_data = json.load(f)
    
    stt_map = {res['source_file']: res['text'] for res in stt_data['results']}

    # 3. 데이터 분석 및 계산
    results = []
    domain_stats = {} # 폴더별 통계용
    
    total_wer, total_cer, valid_count = 0, 0, 0

    print(f"{'Folder':<20} | {'Filename':<30} | {'WER (%)':<10} | {'CER (%)':<10}")
    print("-" * 90)

    for _, row in df_gt.iterrows():
        folder = row['foldername']
        fname = row['filename']
        raw_truth = str(row['ground_truth']).strip()
        
        if not raw_truth or raw_truth == "nan":
            continue
            
        raw_prediction = stt_map.get(fname, "").strip()
        if not raw_prediction:
            continue

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
            "cer": round(error_rate_cer * 100, 2)
        }
        results.append(res_entry)
        
        # 도메인별 합산
        if folder not in domain_stats:
            domain_stats[folder] = {"wer_sum": 0, "cer_sum": 0, "count": 0}
        domain_stats[folder]["wer_sum"] += error_rate_wer
        domain_stats[folder]["cer_sum"] += error_rate_cer
        domain_stats[folder]["count"] += 1

        total_wer += error_rate_wer
        total_cer += error_rate_cer
        valid_count += 1
        
        print(f"{folder:<20} | {fname[:30]:<30} | {error_rate_wer*100:<10.2f}% | {error_rate_cer*100:<10.2f}%")

    # 4. 결과 요약 출력
    if valid_count > 0:
        print("\n" + "="*90)
        print("📈 [종합 도메인별 평균 성적표]")
        print("-" * 60)
        print(f"{'Domain Folder':<25} | {'Avg WER (%)':<15} | {'Avg CER (%)':<15}")
        print("-" * 60)
        
        for folder, stats in domain_stats.items():
            avg_w = (stats["wer_sum"] / stats["count"]) * 100
            avg_c = (stats["cer_sum"] / stats["count"]) * 100
            print(f"{folder:<25} | {avg_w:<15.2f}% | {avg_c:<15.2f}%")
            
        print("-" * 60)
        print(f"🌟 전체 평균 (총 {valid_count}개 파일):")
        print(f"   - Average WER: {(total_wer/valid_count)*100:.2f}%")
        print(f"   - Average CER: {(total_cer/valid_count)*100:.2f}%")
        print("="*90)
        
        # 최종 결과 저장
        report_dir = "src/stt/comparison/result"
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, f"final_stt_accuracy_report.json")
        
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump({
                "overall": {
                    "avg_wer": (total_wer/valid_count)*100,
                    "avg_cer": (total_cer/valid_count)*100,
                    "total_files": valid_count
                },
                "domain_summary": domain_stats,
                "details": results
            }, f, ensure_ascii=False, indent=2)
        print(f"✅ 전수 조사 리포트가 저장되었습니다: {report_path}")
    else:
        print("❌ 테스트할 수 있는 정답 데이터가 없습니다.")


if __name__ == "__main__":
    run_wer_evaluation()

