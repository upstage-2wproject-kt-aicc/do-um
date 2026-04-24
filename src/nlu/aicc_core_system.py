import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from langchain_core.documents import Document
from langchain_upstage import UpstageEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever

# ---------------------------------------------------------
# ⚙️ 0. 환경 설정 및 시스템 초기화
# ---------------------------------------------------------
# 파일이 위치한 현재 폴더 경로를 절대 경로로 잡습니다.
_SCRIPT_DIR = Path(__file__).resolve().parent
faq_csv = _SCRIPT_DIR / "RAG_FAQ.csv"
persist_dir = str(_SCRIPT_DIR / "aicc_chroma_db")

# 🚨 본인의 Upstage API 키로 반드시 변경하세요!
os.environ["UPSTAGE_API_KEY"] = "up_o80maPpk95UkGrqxxd2ldfORTQVWB"

class AICCPipeline:
    def __init__(self):
        print("\n🚀 [System Init] AICC 통합 엔진 부팅 중...")
        init_start = time.perf_counter()
        
        # 1. 임베딩 모델 및 캐시 저장소 초기화
        self.embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
        self.semantic_cache = [] # 실제 상용에서는 Redis를 사용하게 될 공간
        
        # 2. 데이터 적재 및 엔진 예열
        self._prepare_datasets()
        self._warm_up_cache()
        
        print(f"✅ [System Init] 서버 구동 완료 (총 초기화 시간: {time.perf_counter() - init_start:.3f}초)\n")

    def _prepare_datasets(self):
        """CSV 파일을 읽어 BM25와 Vector DB(Chroma)에 하이브리드 지식 창고를 구축합니다."""
        if not faq_csv.is_file():
            raise FileNotFoundError(f"🚨 에러: 파이썬 파일과 같은 폴더에 '{faq_csv.name}' 파일이 없습니다.")
        
        df = pd.read_csv(faq_csv).fillna("")
        self.docs = []
        for _, row in df.iterrows():
            if str(row['embedding_text']).strip():
                # 🌟 [요구사항 반영] 3단계로 넘길 완벽한 메타데이터 구성
                metadata = {
                    "faq_id": str(row['faq_id']),
                    "domain": str(row['domain']),
                    "subdomain": str(row['subdomain']),
                    "intent_type": str(row['intent_type']),
                    "risk_level": str(row['risk_level']),
                    "handoff_required": str(row['handoff_required'])
                }
                self.docs.append(Document(
                    page_content=str(row['embedding_text']),
                    metadata=metadata
                ))
        
        # 키워드 검색기 (BM25)
        self.bm25 = BM25Retriever.from_documents(self.docs)
        self.bm25.k = 2
        
        # 의미 검색기 (Vector DB)
        self.vector_db = Chroma(persist_directory=persist_dir, embedding_function=self.embeddings)

    def _warm_up_cache(self):
        """서버가 켜질 때 단골 질문들을 캐시에 미리 장전(Warming)합니다."""
        hot_queries = [("비대면 통장 개설", "비대면 계좌 개설은 당행 모바일 앱을 통해 24시간 언제든 가능합니다.")]
        for q, a in hot_queries:
            vec = self.embeddings.embed_query(q)
            self.semantic_cache.append({"vector": vec, "answer": a})

    def cosine_similarity(self, v1, v2):
        """두 벡터 간의 의미 유사도를 계산합니다. (1에 가까울수록 같은 의미)"""
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    # ---------------------------------------------------------
    # 🔄 코어 파이프라인 (실제 고객 응답 로직)
    # ---------------------------------------------------------
    def run_pipeline(self, stt_text):
        """1단계에서 텍스트를 받아 4단계로 텍스트를 내보내는 메인 엔진"""
        total_start = time.perf_counter()
        print("\n" + "="*80)
        print(f"🎙️ [1단계: STT 수신] 고객 발화: '{stt_text}'")

        # ==========================================
        # [2단계] NLU 의도 분류 및 임베딩
        # ==========================================
        step2_start = time.perf_counter()
        
        # 1. 의도 분류 (가짜 로직: 실무에선 KoBERT 등이 수행)
        time.sleep(0.05) 
        intent = "절차형" 
        
        # 2. 질의 임베딩 (핵심: 캐시와 RAG 검색에 모두 쓰일 숫자 생성)
        query_vector = self.embeddings.embed_query(stt_text)
        
        print(f"  🧠 [2단계: NLU/임베딩] 완료 (소요시간: {time.perf_counter()-step2_start:.3f}초)")
        
        # 3. 🌟 시맨틱 캐시 검사
        cache_start = time.perf_counter()
        for item in self.semantic_cache:
            sim = self.cosine_similarity(query_vector, item["vector"])
            if sim >= 0.95:
                # [캐시 적중 시나리오]
                print(f"  🔥 [2단계: 캐시 적중] 의미 유사도 {sim:.2f} (소요시간: {time.perf_counter()-cache_start:.3f}초)")
                print(f"  ⏭️ [3단계 스킵] 과거 답변을 즉시 반환합니다.")
                
                final_answer = item["answer"]
                total_latency = time.perf_counter() - total_start
                print(f"\n  🔊 [4단계: TTS 송출] >> {final_answer}")
                print(f"  ⏱️  [총 응답 Latency] {total_latency:.3f}초 (비용 절감!)")
                print("="*80)
                return final_answer
        
        # [캐시 실패 시나리오]
        print(f"  ❄️ [2단계: 캐시 실패] 유사 질문 없음. 3단계로 데이터를 전송합니다.")

        # ==========================================
        # [3단계] 워크플로우 (LangGraph RAG + LLM)
        # ==========================================
        step3_start = time.perf_counter()
        print(f"\n  ⚙️ [3단계: 워크플로우 진입] (수신 데이터: '{intent}' 의도 및 4096차원 벡터)")

        # 1. 하이브리드 검색 (2단계 벡터 재활용!)
        rag_start = time.perf_counter()
        bm25_res = self.bm25.invoke(stt_text) # 원문 글자로 키워드 검색
        vector_res = self.vector_db.similarity_search_by_vector(query_vector, k=1) # 벡터 숫자로 의미 검색
        print(f"  🔎 [RAG 검색 완료] (소요시간: {time.perf_counter() - rag_start:.3f}초)")
        
        # 2. 🌟 메타데이터 추출 및 로깅
        if vector_res:
            res_doc = vector_res[0]
            m = res_doc.metadata
            print(f"\n      [참고 문서 메타데이터]")
            print(f"      • 도메인/의도: {m.get('domain')} > {m.get('subdomain')} ({m.get('intent_type')})")
            print(f"      • 위험도 수준: {m.get('risk_level')}")
            print(f"      • 이관 필요성: {m.get('handoff_required')}")
            
            # 위험/이관 로직 트리거
            if m.get('risk_level') == '높음' or m.get('handoff_required') == 'Y':
                print(f"      ⚠️ [시스템 경고] 이관 필수 문서가 감지되었습니다. 상담사 연결 멘트로 전환해야 합니다.")
        
        # 3. LLM 응답 생성 (가짜 LLM 모듈: 1.5초 대기)
        llm_start = time.perf_counter()
        time.sleep(1.5) 
        final_answer = "(GPT-4o 생성 답변) 조회된 데이터를 기반으로 안내해 드립니다."
        print(f"\n  🤖 [LLM 생성 완료] (소요시간: {time.perf_counter() - llm_start:.3f}초)")

        # 4. 다음 질문을 위해 캐시에 저장 (이게 바로 머신러닝!)
        self.semantic_cache.append({"vector": query_vector, "answer": final_answer})

        # ==========================================
        # [4단계] TTS 송출 준비
        # ==========================================
        total_latency = time.perf_counter() - total_start
        print(f"\n  🔊 [4단계: TTS 송출] >> {final_answer}")
        print(f"  ⏱️  [총 응답 Latency] {total_latency:.3f}초")
        print("="*80)
        
        return final_answer

# =========================================================
# 🎮 대화형(Interactive) 테스트 모드
# =========================================================
if __name__ == "__main__":
    app = AICCPipeline()
    
    print("\n" + "★"*80)
    print("🤖 AICC 실시간 파이프라인 테스트를 시작합니다.")
    print("   (테스트를 종료하시려면 'q'를 입력하세요)")
    print("★"*80)

    while True:
        try:
            user_input = input("\n🧑‍🦰 고객 질문 입력: ")
            
            if user_input.strip().lower() in ['q', 'quit', 'exit']:
                print("👋 테스트를 종료합니다. 수고하셨습니다!")
                break
                
            if not user_input.strip():
                continue

            app.run_pipeline(user_input)
            
        except KeyboardInterrupt:
            print("\n👋 테스트를 종료합니다.")
            break
        except Exception as e:
            print(f"🚨 시스템 오류 발생: {e}")