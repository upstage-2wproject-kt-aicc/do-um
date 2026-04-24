import json
import os
import sys

# 프로젝트 루트 경로를 파이썬 모듈 검색 경로에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.stt.post_processor import STTPostProcessor


def test_real_data_masking():
    processor = STTPostProcessor()

    test_cases = [
        {
            "name": "전화번호+생년월일(공백혼용)",
            "text": "제 번호는 0 1 0 1 2 3 4 5 6 7 8 이고 생년월일은 9 공공 하나0 1 입니다",
            "expected_mask": "[개인정보 마스킹]"
        },
        {
            "name": "카드비밀번호(문맥기반)",
            "text": "고객님 카드 비밀번호 사 이 구 칠 입력해 주세요",
            "expected_mask": "[비밀번호 마스킹]"
        },
        {
            "name": "일반 숫자(마스킹 제외)",
            "text": "현재 시간은 2 0 2 6 년 4 월 입니다",
            "expected_mask": None # 마스킹되면 안 됨
        }
    ]

    print("🔍 [PII 마스킹 종합 테스트]")
    print("-" * 60)

    for tc in test_cases:
        print(f"▶ Test: {tc['name']}")
        cleaned = processor.process(tc['text'])
        print(f"  Input:  {tc['text']}")
        print(f"  Output: {cleaned}")

        if tc['expected_mask']:
            if tc['expected_mask'] in cleaned:
                print("  ✅ 성공")
            else:
                print("  ❌ 실패 (마스킹 누락)")
        else:
            if "[개인정보 마스킹]" not in cleaned and "[비밀번호 마스킹]" not in cleaned:
                print("  ✅ 성공 (일반 숫자 보존)")
            else:
                print("  ❌ 실패 (과도한 마스킹)")
        print()




if __name__ == "__main__":
    test_real_data_masking()
