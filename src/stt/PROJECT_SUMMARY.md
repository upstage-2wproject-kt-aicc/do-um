# 🎙️ AICC STT 파이프라인 구축 및 최적화 작업 요약

본 작업은 금융권 AI 컨택센터(AICC) 환경에 최적화된 **"저비용·고성능 음성 인식 파이프라인"**을 설계하고, 데이터 기반의 벤치마크를 통해 최적의 라이브러리 조합을 도출하는 것을 목표로 하였습니다.

## 📁 1. 폴더 구조 및 모듈 역할
전체 프로젝트 구조를 관심사 분리(SoC) 원칙에 따라 체계화하였습니다.

*   `src/stt/comparison/`: NR, VAD, STT 등 기능별 비교 분석 및 대량 테스트 스크립트
*   `src/stt/evaluation/`: 수집된 데이터를 바탕으로 점수를 산출하는 평가 엔진 (WER/CER 등)
*   `src/stt/data/evaluation/`: 10개 도메인, 총 100개의 실제 테스트 오디오 및 정답지(`web_test.csv`)
*   `src/stt/post_processor.py`: LLM 전달 전 텍스트 정제 및 개인정보 마스킹 핵심 로직
*   `src/stt/temp/`: 공정 과정에서 발생하는 임시 오디오 파일 관리

---

## ⚙️ 2. STT 파이프라인 단계별 작업 내용

### 1단계: 노이즈 제거 (Noise Reduction)
*   **비교 엔진:** FFmpeg(afftdn), SoX, noisereduce
*   **성과:** 실시간 처리를 위해 CPU 점유율이 가장 낮고 속도가 빠른 **FFmpeg**을 기본 엔진으로 채택.

### 2단계: 음성 구간 검출 (VAD)
*   **비교 엔진:** Pydub, webrtcvad, Silero VAD
*   **전수 조사:** 100개 파일을 대상으로 속도, 무음 제거율, 안정성(Stability) 측정.
*   **결과:** **webrtcvad**가 압도적인 속도로 1위를 차지했으나, 정확도가 필요한 상황을 위해 Silero VAD 하이브리드 운영안 도출.

### 3단계: 최적의 조합(Best Combo) 도출
*   **그리드 서치:** 9개 조합(3 NR x 3 VAD)을 100개 파일에 대해 모두 시뮬레이션.
*   **최종 Winner:** **[FFmpeg + webrtcvad]** 조합이 비용 및 성능 밸런스에서 최고점 획득.

### 4단계: 텍스트 후처리 및 보안 (Post-Processing)
*   **간투어 제거:** "음..", "어.." 등 불필요한 추임새 제거.
*   **PII 마스킹:** 정규식과 한글 수사 변환 로직을 결합하여 **전화번호, 주민번호, 4자리 비밀번호**를 정교하게 가림. (원본 훼손 없는 위치 기반 치환 방식 적용)

---

## 📊 3. 최종 성능 지표 (Benchmark)

100개 파일(유효 89개) 전수 조사 결과, 구글 STT(Telephony 모델) 파이프라인의 성적은 다음과 같습니다.

*   **Average WER (단어 오류율):** **20.89%**
*   **Average CER (글자 오류율):** **8.32%**
*   **인사이트:** 띄어쓰기 문제를 제외한 **순수 글자 인식률은 91% 이상**으로, 금융 상담 의도 파악에 충분한 고성능을 확보함.

---

## 🚀 4. 주요 실행 명령어 가이드

작업한 내용들을 다시 돌려보거나 검증할 때 사용하는 명령어입니다.

```powershell
# 1. [최적화] 9개 조합 전수 조사 (비용 0원)
uv run python src/stt/comparison/pre_processing_optimizer.py

# 2. [실행] 최종 확정 파이프라인으로 100개 STT 돌리기
uv run python src/stt/comparison/final_pipeline_stt.py

# 3. [평가] 정답지와 비교하여 WER/CER 성적표 뽑기
uv run python src/stt/evaluation/wer_evaluator.py

# 4. [검증] 후처리 및 마스킹 엔진 성능 테스트
uv run python src/stt/post_process_test.py
```

---

## ✨ 5. 핵심 성과 요약
1.  **비용 절감:** VAD 사전 필터링 및 시뮬레이션을 통해 불필요한 API 호출 비용을 최소화함.
2.  **데이터 기반 결정:** "그냥 쓰는 것"이 아니라 100개 데이터 전수 조사를 통해 최적의 조합임을 증명함.
3.  **금융권 보안 준수:** 강력한 PII 마스킹 로직을 구현하여 실제 상담 환경의 개인정보 유출 위험을 방지함.
