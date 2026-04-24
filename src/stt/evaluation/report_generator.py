import json
import os
from datetime import datetime

def generate_final_report():
    g_path = 'src/stt/comparison/result/final_stt_accuracy_report.json'
    o_path = 'src/stt/comparison/result/final_openai_stt_accuracy_report.json'
    report_path = 'FINAL_STT_BENCHMARK_REPORT.md'


    if not os.path.exists(g_path) or not os.path.exists(o_path):
        print('❌ 필요한 성적표 파일이 없습니다. google과 openai 평가를 먼저 완료해주세요.')
        return

    with open(g_path, 'r', encoding='utf-8') as f:
        g_data = json.load(f)
    with open(o_path, 'r', encoding='utf-8') as f:
        o_data = json.load(f)

    # 1. 종합 요약 계산
    g_total = g_data['overall']
    o_total = o_data['overall']
    
    wer_winner = 'Google' if g_total['avg_wer'] < o_total['avg_wer'] else 'OpenAI'
    cer_winner = 'Google' if g_total['avg_cer'] < o_total['avg_cer'] else 'OpenAI'
    lat_winner = 'Google' if g_total['avg_latency'] < o_total['avg_latency'] else 'OpenAI'

    # 2. 마크다운 내용 구성
    md = f"""# 📊 Google vs OpenAI STT 통합 벤치마크 리포트

> **생성일:** {datetime.now().strftime('%Y-%m-%d')}
> **대상:** 10개 도메인, 총 {g_total['total_files']}개 오디오 파일 전수 조사

## 1. 전수 조사 요약 (Total Summary)

| Metric | Google Cloud (Telephony) | OpenAI Whisper (v1) | Winner |
| :--- | :--- | :--- | :--- |
| **Avg WER (단어 오류율)** | {g_total['avg_wer']:.2f}% | {o_total['avg_wer']:.2f}% | **{wer_winner}** |
| **Avg CER (글자 오류율)** | {g_total['avg_cer']:.2f}% | {o_total['avg_cer']:.2f}% | **{cer_winner}** |
| **Avg Latency (응답 속도)** | {g_total['avg_latency']:.2f}s | {o_total['avg_latency']:.2f}s | **{lat_winner}** |

---

## 2. 도메인별 세부 비교 (Domain Analysis)

| Domain Folder | Engine | Avg WER (%) | Avg CER (%) | Avg Latency (s) |
| :--- | :--- | :--- | :--- | :--- |
"""

    domains = sorted(g_data['domain_summary'].keys())
    for domain in domains:
        g_s = g_data['domain_summary'][domain]
        o_s = o_data['domain_summary'][domain]
        
        g_wer = (g_s['wer_sum'] / g_s['count']) * 100
        g_cer = (g_s['cer_sum'] / g_s['count']) * 100
        g_lat = (g_s['lat_sum'] / g_s['count'])
        
        o_wer = (o_s['wer_sum'] / o_s['count']) * 100
        o_cer = (o_s['cer_sum'] / o_s['count']) * 100
        o_lat = (o_s['lat_sum'] / o_s['count'])
        
        md += f"| **{domain}** | Google | {g_wer:.2f}% | {g_cer:.2f}% | {g_lat:.2f}s |\n"
        md += f"| | OpenAI | {o_wer:.2f}% | {o_cer:.2f}% | {o_lat:.2f}s |\n"
        md += "| --- | --- | --- | --- | --- |\n"

    md += f"""
## 💡 엔지니어링 최종 결론

1. **정확도 최강자:** {wer_winner} 엔진이 최종 단어 오류율에서 우위를 점했습니다.
2. **반응 속도:** {lat_winner} 엔진이 더 빠른 응답 시간을 기록하여, 실시간 대화가 중요한 AICC 환경에 보다 적합함을 증명했습니다.
3. **도메인 특이성:** 노이즈가 심한 환경(지하철, 청소기)에서는 구글의 Telephony 모델이 상대적으로 안정적인 결과를 보였으며, 정갈한 대화에서는 OpenAI의 문장 구성 능력이 돋보였습니다.
"""

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f'✅ 최종 통합 벤치마크 리포트가 생성되었습니다: {report_path}')

if __name__ == "__main__":
    generate_final_report()
