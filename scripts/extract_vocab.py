import pandas as pd
import re

def extract_vocabulary():
    try:
        # 엑셀 로드
        df = pd.read_excel('새싹_데이터.xlsx', sheet_name='RAG_FAQ')
        all_terms = set()
        
        # 1. keywords 컬럼만 신뢰하여 추출 (여기에 핵심 명사가 다 들어있음)
        if 'keywords' in df.columns:
            for kw in df['keywords'].dropna():
                # 콤마, 슬래시, 파이프 등으로 분리
                parts = re.split(r'[,|/]', str(kw))
                for p in parts:
                    term = p.strip()
                    # 2글자 이상인 경우만 명사로 간주하여 추가
                    if len(term) >= 2:
                        all_terms.add(term)
        
        # 2. 추가적으로 꼭 필요한 금융 공통 용어 (엑셀에 없을 수도 있는 것들)
        COMMON_FINANCE = [
            "중도상환수수료", "대출계약철회", "금리인하요구권", "원리금균등", "원금균등", 
            "마이너스통장", "한도상향", "비대면계좌", "착오송금", "지급정지"
        ]
        all_terms.update(COMMON_FINANCE)

        # 3. vocabulary.py 생성
        with open('src/stt/streaming/vocabulary.py', 'w', encoding='utf-8') as f:
            f.write('"""금융권 AICC STT 인식을 위한 핵심 명사 사전 (정제됨)."""\n\n')
            f.write('FINANCE_VOCABULARY = [\n')
            for term in sorted(list(all_terms)):
                f.write(f'    "{term}",\n')
            f.write(']\n\n')
            f.write('def get_financial_vocabulary() -> list[str]:\n')
            f.write('    return FINANCE_VOCABULARY\n')
            
        print(f"✅ 정제 성공: {len(all_terms)}개의 금융 명사를 저장했습니다.")
        
    except Exception as e:
        print(f"❌ 오류: {e}")

if __name__ == "__main__":
    extract_vocabulary()
