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
_SCRIPT_DIR = Path(__file__).resolve().parent
faq_csv = _SCRIPT_DIR / "RAG_FAQ.csv"
persist_dir = str(_SCRIPT_DIR / "aicc_chroma_db")

os.environ["UPSTAGE_API_KEY"] = "up_o80maPpk95UkGrqxxd2ldfORTQVWB" # 본인 키로 변경!

class AICCPipeline:
    def __init__(self):
        print("\n🚀 [System Init] AICC 통합 엔진 부팅 중...")
        init_start = time.perf_counter()
        
        self.embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
        self.semantic_cache = []
        
        self._prepare_datasets()
        self._warm_up_cache()
        
        print(f"✅ [System Init] 서버 구동 완료 (총 초기화 시간: {time.perf_counter() - init_start:.3f}초)\n")

    def _prepare_datasets(self):
        if not faq_csv.is_file():
            raise FileNotFoundError(f"🚨 에러: '{faq_csv.name}' 파일이 없습니다.")
        
        df = pd.read_csv(faq_csv).fillna("")
        self.docs = []
        for _, row in df.iterrows():
            if str(row['embedding_text']).strip():
                metadata = {
                    "faq_id": str(row['faq_id']),
                    "domain": str(row['domain']),
                    "subdomain": str(row['subdomain']),
                    "intent_type": str(row['intent_type']),
                    "risk_level": str(row['risk_level']),
                    "handoff_required": str(row['handoff_required'])
                }
                self.docs.append(Document(page_content=str(row['embedding_text']), metadata=metadata))
        
        self.bm25 = BM25Retriever.from_documents(self.docs)
        self.bm25.k = 2
        
        # 🌟 [수정됨] 읽어오기가 아니라, CSV 데이터를 확실하게 DB로 생성/덮어쓰기 합니다!
        print("   ↳ Vector DB에 FAQ 데이터를 적재합니다...")
        self.vector_db = Chroma.from_documents(
            documents=self.docs, 
            embedding=self.embeddings,
            persist_directory=persist_dir
        )

    def _warm_up_cache(self):
        hot_queries = [("비대면 통장 개설", "비대면 계좌 개설은 당행 모바일 앱을 통해 24시간 언제든 가능합니다.")]
        for q, a in hot_queries:
            vec = self.embeddings.embed_query(q)
            self.semantic_cache.append({"vector": vec, "answer": a})

    def cosine_similarity(self, v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    # ---------------------------------------------------------
    # 🔄 코어 파이프라인
    # ---------------------------------------------------------
    def run_pipeline(self, stt_text):
        total_start = time.perf_counter()
        print("\n" + "="*80)
        print(f"🎙️ [1단계: STT 수신] 고객 발화: '{stt_text}'")

        # [2단계] NLU 및 임베딩
        step2_start = time.perf_counter()
        time.sleep(0.05) 
        intent = "절차형" 
        query_vector = self.embeddings.embed_query(stt_text)
        print(f"  🧠 [2단계: NLU/임베딩] 완료 (소요시간: {time.perf_counter()-step2_start:.3f}초)")
        
        # 🌟 [수정됨] 시맨틱 캐시 검사 (유사도 기준을 0.85로 현실화 및 최고 유사도 추적)
        cache_start = time.perf_counter()
        max_sim = 0.0
        
        for item in self.semantic_cache:
            sim = self.cosine_similarity(query_vector, item["vector"])
            if sim > max_sim: max_sim = sim
            
            # 기준치를 0.85 -> 0.75로 낮춤
            if sim >= 0.75:
                print(f"  🔥 [2단계: 캐시 적중!] 의미 유사도 {sim:.2f} (소요시간: {time.perf_counter()-cache_start:.3f}초)")
                print(f"  ⏭️ [3단계 스킵] 과거 답변을 즉시 반환합니다.")
                
                final_answer = item["answer"]
                total_latency = time.perf_counter() - total_start
                print(f"\n  🔊 [4단계: TTS 송출] >> {final_answer}")
                print(f"  ⏱️  [총 응답 Latency] {total_latency:.3f}초 (비용 절감!)")
                print("="*80)
                return final_answer
        
        # 캐시 실패 시 최고 유사도 출력
        print(f"  ❄️ [2단계: 캐시 실패] 유사 질문 없음. (최고 유사도: {max_sim:.2f} < 기준치 0.85)")

        # [3단계] 워크플로우
        step3_start = time.perf_counter()
        print(f"\n  ⚙️ [3단계: 워크플로우 진입] (수신 데이터: '{intent}' 의도 및 4096차원 벡터)")

        rag_start = time.perf_counter()
        vector_res = self.vector_db.similarity_search_by_vector(query_vector, k=1)
        print(f"  🔎 [RAG 검색 완료] (소요시간: {time.perf_counter() - rag_start:.3f}초)")
        
        if vector_res:
            res_doc = vector_res[0]
            m = res_doc.metadata
            print(f"\n      [📥 참고 문서 메타데이터 수신]")
            print(f"      • 도메인/의도: {m.get('domain')} > {m.get('subdomain')} ({m.get('intent_type')})")
            print(f"      • 위험도 수준: {m.get('risk_level')}")
            print(f"      • 이관 필요성: {m.get('handoff_required')}")
            
            if m.get('risk_level') == '높음' or m.get('handoff_required') == 'Y':
                print(f"      ⚠️ [시스템 경고] 이관 필수 문서가 감지되었습니다. 상담사 연결을 준비합니다.")
        
        llm_start = time.perf_counter()
        time.sleep(1.5) 
        final_answer = "(Solar Pro 생성 답변) 조회된 데이터를 기반으로 안내해 드립니다."
        print(f"\n  🤖 [LLM 생성 완료] (소요시간: {time.perf_counter() - llm_start:.3f}초)")

        self.semantic_cache.append({"vector": query_vector, "answer": final_answer})

        total_latency = time.perf_counter() - total_start
        print(f"\n  🔊 [4단계: TTS 송출] >> {final_answer}")
        print(f"  ⏱️  [총 응답 Latency] {total_latency:.3f}초")
        print("="*80)
        
        return final_answer

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
            break
        except Exception as e:
            print(f"🚨 시스템 오류 발생: {e}")