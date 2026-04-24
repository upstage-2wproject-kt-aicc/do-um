import json
import os
from datetime import datetime

class NREvaluator:
    def __init__(self, weight_wer=0.4, weight_latency=0.3, weight_quality=0.2, weight_resource=0.1):
        self.weights = {
            "wer": weight_wer,
            "latency": weight_latency,
            "quality": weight_quality,
            "resource": weight_resource
        }

    def calculate_wer_score(self, wer_before, wer_after):
        """WER 개선율 기반 점수 산출"""
        if wer_before == 0: return 0
        improvement = (wer_before - wer_after) / wer_before
        return max(0, improvement * 100)

    def calculate_latency_score(self, latency_lib, latency_min):
        """속도 기반 점수 산출 (역비례)"""
        if latency_lib == 0: return 0
        return (latency_min / latency_lib) * 100

    def calculate_quality_score(self, pesq_score, stoi_score):
        """PESQ/STOI 기반 품질 점수 산출"""
        # PESQ는 보통 1.0~4.5 범위이므로 0-100으로 정규화 필요 (여기서는 단순화)
        # STOI는 0~1 범위
        score_pesq = min(100, max(0, (pesq_score - 1.0) / 3.5 * 100))
        score_stoi = stoi_score * 100
        return (score_pesq + score_stoi) / 2

    def calculate_resource_score(self, resource_lib, resource_min):
        """리소스 사용량 기반 점수 산출 (역비례)"""
        if resource_lib == 0: return 0
        return (resource_min / resource_lib) * 100

    def evaluate(self, comparison_data):
        """전체 데이터를 바탕으로 점수 산출 및 리포트 생성"""
        results = []
        
        # Latency 및 Resource 최소값 찾기 (비교 기준)
        latencies = [lib["latency"] for lib in comparison_data if lib.get("latency")]
        resources = [lib["resource"] for lib in comparison_data if lib.get("resource")]
        
        latency_min = min(latencies) if latencies else 0.001
        resource_min = min(resources) if resources else 0.001
        
        wer_before = comparison_data[0].get("wer_before", 0.5) # 기본값 예시

        for lib in comparison_data:
            s_wer = self.calculate_wer_score(wer_before, lib.get("wer_after", 0.5))
            s_lat = self.calculate_latency_score(lib.get("latency", 1), latency_min)
            s_qual = self.calculate_quality_score(lib.get("pesq", 2.0), lib.get("stoi", 0.8))
            s_res = self.calculate_resource_score(lib.get("resource", 100), resource_min)
            
            final_score = (s_wer * self.weights["wer"] + 
                           s_lat * self.weights["latency"] + 
                           s_qual * self.weights["quality"] + 
                           s_res * self.weights["resource"])
            
            results.append({
                "engine": lib["engine"],
                "scores": {
                    "wer": round(s_wer, 2),
                    "latency": round(s_lat, 2),
                    "quality": round(s_qual, 2),
                    "resource": round(s_res, 2)
                },
                "final_score": round(final_score, 2)
            })
        
        return results

def run_nr_evaluation():
    print("📋 노이즈 제거 라이브러리 평가를 시작합니다...")
    metric_path = "src/stt/comparison/result/nr_metrics_latest.json"
    
    if not os.path.exists(metric_path):
        print(f"❌ 메트릭 파일을 찾을 수 없습니다: {metric_path}")
        return

    with open(metric_path, "r", encoding="utf-8") as f:
        metric_data = json.load(f)
    
    evaluator = NREvaluator()
    all_reports = []
    
    # 각 파일별 평가
    for file_res in metric_data.get("results", []):
        report = evaluator.evaluate(file_res["metrics"])
        all_reports.append({"file": file_res["source_file"], "evaluation": report})
    
    # 평균 점수 계산
    engine_scores = {}
    for r in all_reports:
        for engine_eval in r["evaluation"]:
            eng = engine_eval["engine"]
            if eng not in engine_scores: engine_scores[eng] = []
            engine_scores[eng].append(engine_eval["final_score"])
    
    print("\n✨ [NR 라이브러리 평균 평가 결과]")
    print(f"{'Engine':<15} | {'Average Score':<12}")
    print("-" * 35)
    for eng, scores in engine_scores.items():
        avg = sum(scores) / len(scores)
        print(f"{eng:<15} | {avg:<12.2f}")

    output_path = "src/stt/comparison/result/final_nr_evaluation.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"summary": engine_scores, "details": all_reports}, f, ensure_ascii=False, indent=2)
    print(f"\n💾 리포트가 저장되었습니다: {output_path}")


if __name__ == "__main__":
    run_nr_evaluation()
