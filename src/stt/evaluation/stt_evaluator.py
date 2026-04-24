import json
import os
from datetime import datetime

class STTEvaluator:
    def __init__(self, weight_wer=0.5, weight_latency=0.3, weight_cost=0.2):
        self.weights = {
            "wer": weight_wer,
            "latency": weight_latency,
            "cost": weight_cost
        }

    def calculate_wer_score(self, wer):
        """WER 수치를 점수화 (WER 0% = 100점, 50% 이상 = 0점)"""
        return max(0, 100 - (wer * 200))

    def calculate_latency_score(self, latency_lib, latency_min):
        if latency_lib == 0: return 0
        return (latency_min / latency_lib) * 100

    def calculate_cost_score(self, cost_per_min, cost_min):
        """가장 저렴한 엔진 대비 점수 산출"""
        if cost_per_min == 0: return 100 # 무료 엔진(Local)
        return (cost_min / cost_per_min) * 100

    def evaluate(self, comparison_data):
        results = []
        latencies = [lib["latency"] for lib in comparison_data if lib.get("latency")]
        costs = [lib["cost_per_min"] for lib in comparison_data if lib.get("cost_per_min") > 0]
        
        latency_min = min(latencies) if latencies else 0.001
        cost_min = min(costs) if costs else 0.01
        
        for lib in comparison_data:
            s_wer = self.calculate_wer_score(lib.get("wer", 0.15))
            s_lat = self.calculate_latency_score(lib.get("latency", 1.0), latency_min)
            s_cost = self.calculate_cost_score(lib.get("cost_per_min", 0), cost_min)
            
            final_score = (s_wer * self.weights["wer"] + 
                           s_lat * self.weights["latency"] + 
                           s_cost * self.weights["cost"])
            
            results.append({
                "engine": lib["engine"],
                "scores": {
                    "accuracy_wer": round(s_wer, 2),
                    "latency": round(s_lat, 2),
                    "cost_efficiency": round(s_cost, 2)
                },
                "final_score": round(final_score, 2)
            })
        return results

def run_stt_evaluation():
    print("📋 STT API 모델 평가를 시작합니다...")
    metric_path = "src/stt/comparison/result/stt_metrics_latest.json"
    
    if not os.path.exists(metric_path):
        print(f"❌ 메트릭 파일을 찾을 수 없습니다: {metric_path}")
        return

    with open(metric_path, "r", encoding="utf-8") as f:
        metric_data = json.load(f)
    
    evaluator = STTEvaluator()
    all_reports = []
    
    for file_res in metric_data.get("results", []):
        report = evaluator.evaluate(file_res["metrics"])
        all_reports.append({"file": file_res["source_file"], "evaluation": report})
    
    engine_scores = {}
    for r in all_reports:
        for engine_eval in r["evaluation"]:
            eng = engine_eval["engine"]
            if eng not in engine_scores: engine_scores[eng] = []
            engine_scores[eng].append(engine_eval["final_score"])
    
    print("\n✨ [STT 평균 평가 결과]")
    print(f"{'Engine':<20} | {'Average Score':<12}")
    print("-" * 35)
    for eng, scores in engine_scores.items():
        avg = sum(scores) / len(scores)
        print(f"{eng:<20} | {avg:<12.2f}")

    output_path = "src/stt/comparison/result/final_stt_evaluation.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"summary": engine_scores, "details": all_reports}, f, ensure_ascii=False, indent=2)
    print(f"\n💾 리포트가 저장되었습니다: {output_path}")


if __name__ == "__main__":
    run_stt_evaluation()
