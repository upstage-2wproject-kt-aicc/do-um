from src.nlu.aicc_core_klue import AICC_NLU_Router

def run_consistency_test():
    print("🚀 NLU 라우터 부팅 중...")
    router = AICC_NLU_Router()
    
    # 테스트할 질문
    test_query = "고정금리와 변동금리의 차이는 무엇인가요?"
    print(f"\n🎯 [일관성 테스트] 동일한 질문 5회 연속 호출")
    print(f"질문: '{test_query}'\n")
    print("-" * 50)

    first_result = None

    for i in range(1, 6):
        # 1. 쿼리 실행
        out = router.process_query(test_query)
        
        # 2. 비교할 핵심 데이터 추출
        intent = out.get("intent")
        faq_ids = out.get("retrieved_faq_ids", [])
        status = out.get("status")
        
        # 3. 결과 비교
        if i == 1:
            first_result = {"intent": intent, "faq_ids": faq_ids}
            print(f"[{i}회차 (기준)] 의도: {intent} | FAQ_ID: {faq_ids} | 상태: {status}")
        else:
            current_result = {"intent": intent, "faq_ids": faq_ids}
            is_match = (first_result == current_result)
            match_icon = "✅ 일치" if is_match else "❌ 불일치"
            
            print(f"[{i}회차] {match_icon} -> 의도: {intent} | FAQ_ID: {faq_ids} | 상태: {status}")

    print("-" * 50)

if __name__ == "__main__":
    run_consistency_test()