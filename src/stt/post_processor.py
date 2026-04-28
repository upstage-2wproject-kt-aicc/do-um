import re
from stt.streaming.vocabulary import get_financial_vocabulary

_ONSET = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
_VOWEL = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
_CODA  = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

def _decompose(text: str) -> str:
    out = []
    for ch in text:
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3:
            offset = code - 0xAC00
            out.append(_ONSET[offset // 588])
            out.append(_VOWEL[(offset % 588) // 28])
            coda = _CODA[offset % 28]
            if coda:
                out.append(coda)
        else:
            out.append(ch)
    return ''.join(out)

def _jamo_distance(a: str, b: str) -> int:
    ja, jb = _decompose(a), _decompose(b)
    m, n = len(ja), len(jb)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            prev, dp[j] = dp[j], prev if ja[i-1] == jb[j-1] else 1 + min(prev, dp[j], dp[j-1])
    return dp[n]

class STTPostProcessor:
    def __init__(self):
        self._vocab = get_financial_vocabulary()

        # 한글 숫자 매핑 (가상 지도 생성용)
        self.num_map = {
            "공": "0", "영": "0", "일": "1", "하나": "1", "이": "2", "둘": "2",
            "삼": "3", "셋": "3", "사": "4", "넷": "4", "오": "5", "육": "6",
            "칠": "7", "팔": "8", "구": "9"
        }
        
        # 금융 도메인 오인식 교정
        self.domain_dict = {
            "해살론": "햇살론", "해살론2": "햇살론2", "안심전환": "안심전환대출",
            "중도상환": "중도상환수수료", "금니": "금리", "한도조해": "한도조회"
        }
        
        self.filler_words = ["음", "어", "아니", "그게", "그", "저기", "어머", "에", "있잖아요", "막"]

    def remove_fillers(self, text):
        pattern = r"\b(" + "|".join(self.filler_words) + r")\b"
        text = re.sub(pattern, "", text)
        return re.sub(r"\s+", " ", text).strip()

    def correct_domain_terms(self, text):
        for wrong, right in self.domain_dict.items():
            text = text.replace(wrong, right)
        return text

    def correct_with_vocabulary(self, text: str, max_jamo_dist: int = 2) -> str:
        tokens = text.split()
        result = []
        for token in tokens:
            m = re.search(r'[^가-힣a-zA-Z0-9]+$', token)
            suffix = m.group() if m else ''
            core = token[:m.start()] if m else token

            best, best_dist = None, max_jamo_dist + 1
            for word in self._vocab:
                if abs(len(core) - len(word)) > 1:
                    continue
                dist = _jamo_distance(core, word)
                if dist < best_dist:
                    best_dist, best = dist, word

            result.append((best if best else core) + suffix)
        return ' '.join(result)

    def mask_pii(self, text):
        """정교한 PII 마스킹: 4자리 비밀번호 및 파편화된 번호 대응"""
        
        # 1. 가상 지도 생성 (원본 길이 유지)
        # 2자 고유어 수사는 char 단위 루프에서 매칭 불가 → digit+space(2자)로 선치환
        _MULTI_CHAR_NUM = [
            ("하나", "1 "), ("다섯", "5 "), ("여섯", "6 "),
            ("일곱", "7 "), ("여덟", "8 "), ("아홉", "9 "),
        ]
        mapping_input = text
        for kor, digit in _MULTI_CHAR_NUM:
            mapping_input = mapping_input.replace(kor, digit)
        mapped_chars = []
        for char in mapping_input:
            mapped_chars.append(self.num_map.get(char, char))
        mapped_text = "".join(mapped_chars)
        
        # 2. 감지할 패턴들
        # [주민번호/전화번호] 기존 패턴 강화
        jumin_full = r"[0-9]{6}\s*[-]?\s*[0-9]{7}"
        phone_full = r"0\s*[16789]\s*[0-9](\s*[0-9]){7,8}"
        birth_front = r"\b[0-9](\s*[0-9]){5}\b"
        phone_partial = r"\b[0-9](\s*[0-9]){6,7}\b"

        # [4자리 비밀번호/카드번호] 키워드 기반 마스킹
        # "비밀번호는 일 이 삼 사 입니다" 등의 케이스 대응
        pw_keywords = r"(비밀번호|비번|인증번호|카드번호|뒷자리)"
        # 키워드 뒤에 공백/조사가 오고 그 뒤에 숫자 4개가 오는 패턴
        password_pattern = rf"{pw_keywords}[\s가-힣]*[0-9](\s*[0-9]){{3}}"

        # 3. 마스킹 대상 위치 찾기
        mask_ranges = []
        
        # 패스워드 패턴은 키워드 뒷부분 숫자만 마스킹하기 위해 별도 처리
        for match in re.finditer(password_pattern, mapped_text):
            # 전체 매치 중 숫자 부분만 찾아서 마스킹 범위 지정
            full_match_text = match.group()
            num_match = re.search(r"[0-9](\s*[0-9]){3}", full_match_text)
            if num_match:
                start = match.start() + num_match.start()
                end = match.start() + num_match.end()
                mask_ranges.append((start, end, "[비밀번호 마스킹]"))

        # 나머지 패턴들 (긴 것부터)
        for pattern in [jumin_full, phone_full, phone_partial, birth_front]:
            for match in re.finditer(pattern, mapped_text):
                start, end = match.span()
                # 이미 비밀번호로 마스킹된 범위와 겹치면 패스
                if any(max(start, r[0]) < min(end, r[1]) for r in mask_ranges):
                    continue
                mask_ranges.append((start, end, "[개인정보 마스킹]"))
                
        # 4. 원본 텍스트에 적용
        mask_ranges.sort(key=lambda x: x[0], reverse=True)
        result = list(text)
        for start, end, label in mask_ranges:
            result[start:end] = list(label)
            
        return "".join(result)


    def process(self, raw_text):
        if not raw_text: return ""

        text = self.remove_fillers(raw_text)
        text = self.correct_domain_terms(text)
        text = self.correct_with_vocabulary(text)
        text = self.mask_pii(text)

        return text

if __name__ == "__main__":
    processor = STTPostProcessor()
    test_text = "상담사 오공팔입니다 감4합니다 0 1 0 1 2 3 4 5 6 7 8 이요 생년월일은 9 0 0 1 0 1 입니다"
    print(f"Test Result: {processor.process(test_text)}")
