import json
import os
from datetime import datetime

class VADEvaluator:
    def __init__(self, weight_trim=0.4, weight_latency=0.4, weight_stability=0.2):
        self.weights = {
            "trim": weight_trim,
            "latency": weight_latency,
            "stability": weight_stability
        }

    def calculate_trim_score(self, trim_rate):
        """무음 제거율 기반 점수 (30~50%를 이상적으로 가정)"""
        if trim_rate > 70: # 너무 많이 깎이면 말이 잘렸을 위험이 큼
            return 100 - (trim_rate - 70) * 2
        return min(100, trim_rate * 2)

    def calculate_latency_score(self, latency_lib, latency_min):
        if latency_lib == 0: return 0
        return (latency_min / latency_lib) * 100

    def calculate_stability_score(self, segment_count):
        """세그먼트 개수가 너무 많으면(잘게 쪼개지면) 감점"""
        # 1분 오디오 기준 10~20개 세그먼트가 적절하다고 가정
        if segment_count > 30:
            return max(0, 100 - (segment_count - 30) * 5)
        return 100

    def evaluate(self, comparison_data):
        results = []
        latencies = [lib["latency"] for lib in comparison_data if lib.get("latency")]
        latency_min = min(latencies) if latencies else 0.001
        
        for lib in comparison_data:
            s_trim = self.calculate_trim_score(lib.get("trim_rate", 30))
            s_lat = self.calculate_latency_score(lib.get("latency", 0.1), latency_min)
            s_stab = self.calculate_stability_score(lib.get("segment_count", 15))
            
            final_score = (s_trim * self.weights["trim"] + 
                           s_lat * self.weights["latency"] + 
                           s_stab * self.weights["stability"])
            
            results.append({
                "engine": lib["engine"],
                "scores": {
                    "trim_efficiency": round(s_trim, 2),
                    "latency": round(s_lat, 2),
                    "stability": round(s_stab, 2)
                },
                "final_score": round(final_score, 2)
            })
        return results

def run_vad_evaluation():
    print("📋 VAD 라이브러리 전수 평가(100개)를 시작합니다...")
    
    metric_path = "src/stt/comparison/result/vad_metrics_latest.json"
    if not os.path.exists(metric_path):
        print(f"❌ 메트릭 파일을 찾을 수 없습니다: {metric_path}")
        return

    with open(metric_path, "r", encoding="utf-8") as f:
        metric_data = json.load(f)
    
    evaluator = VADEvaluator()
    all_file_results = []
    engine_scores = {} # 엔진별 점수 리스트

    # 1. 파일별 평가 수행
    for file_res in metric_data.get("results", []):
        report = evaluator.evaluate(file_res["metrics"])
        all_file_results.append({
            "file": file_res["source_file"],
            "evaluation": report
        })
        
        # 엔진별 평균을 내기 위한 수집
        for eng_eval in report:
            eng_name = eng_eval["engine"]
            if eng_name not in engine_scores:
                engine_scores[eng_name] = {"total_score": 0, "count": 0, "scores": []}
            engine_scores[eng_name]["total_score"] += eng_eval["final_score"]
            engine_scores[eng_name]["count"] += 1
            engine_scores[eng_name]["scores"].append(eng_eval["final_score"])

    # 2. 최종 평균 점수 산출
    final_summary = []
    for name, data in engine_scores.items():
        avg_score = data["total_score"] / data["count"] if data["count"] > 0 else 0
        final_summary.append({
            "engine": name,
            "average_score": round(avg_score, 2),
            "processed_files": data["count"]
        })

    # 정렬 (점수 높은 순)
    final_summary.sort(key=lambda x: x["average_score"], reverse=True)

    print(f"\n📊 [VAD 100개 파일 평균 평가 결과]")
    print(f"{'Engine':<15} | {'Avg Score':<12} | {'Files'}")
    print("-" * 40)
    for s in final_summary:
        print(f"{s['engine']:<15} | {s['average_score']:<12} | {s['processed_files']}")

    # 3. 결과 저장
    output_path = "src/stt/comparison/result/final_vad_evaluation.json"
    save_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary": final_summary,
        "details": all_file_results
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 최종 종합 리포트 저장 완료: {output_path}")
    if final_summary:
        print(f"🏆 Best VAD Engine: {final_summary[0]['engine']}")



if __name__ == "__main__":
    run_vad_evaluation()
