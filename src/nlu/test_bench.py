import time
from statistics import mean

from aicc_core_klue import AICC_NLU_Router

queries = [
    "햇살론 비대면 신청되나요?",
    "중도상환수수료는 왜 내야 하나요?",
    "카드 분실했는데 어떻게 해야 해요?",
    "대출 이자가 지난달보다 많이 나온 이유가 뭐죠?",
    "보이스피싱 피해가 의심되면 뭘 먼저 해야 하나요?",
    "정책서민금융은 어디서 신청하나요?",
]

print("🚀 [벤치마크 테스트] NLU 엔진 부팅 중...\n")
print(
    "💡 비교 팁: (1) 이 폴더에서 연속 두 번 실행 → 2번째는 Chroma persist 로드로 부팅이 짧아집니다.\n"
    "          (2) `aicc_chroma_db` 폴더를 지운 뒤 실행 → 최초 구축(임베딩 API 다수 호출)과 동일 조건.\n"
)
boot_t0 = time.perf_counter()
router = AICC_NLU_Router()
boot_sec = time.perf_counter() - boot_t0
print(f"\n⏱️ [벤치] 부팅(라우터 초기화) 총 시간: {boot_sec:.3f}s\n")

results = []
cache_hits = 0

print("\n" + "-"*50)
for q in queries:
    out = router.process_query(q)
    t = out.get("timings_sec", {})
    
    # 시간 데이터 추출
    tot = float(t.get("total_sec", 0.0))
    intent_ms = float(t.get("intent_sec", 0.0))
    embed_ms = float(t.get("embedding_sec", 0.0))
    
    if out["status"] == "CACHED":
        cache_hits += 1
        
    print(f"[{out['status']}] ⏱️ 총 {tot:.3f}초 (의도:{intent_ms:.3f}초, 임베딩:{embed_ms:.3f}초) | Q: {q}")
    results.append(tot)

print("-"*50)
print("\n📊 [테스트 결과 요약 — 질의당 process_query 시간]")
print(f"• 부팅(초기화) 시간 : {boot_sec:.3f}초 (로그의 Chroma 적재/로드 구간과 함께 보세요)")
print(f"• 테스트 질문 수 : {len(queries)}개")
print(f"• 캐시 적중 횟수 : {cache_hits}번")
print(f"• 평균 소요 시간 : {mean(results):.3f}초")
print(f"• 최대 소요 시간 : {max(results):.3f}초")
print(f"• 최소 소요 시간 : {min(results):.3f}초")
print("="*50)
