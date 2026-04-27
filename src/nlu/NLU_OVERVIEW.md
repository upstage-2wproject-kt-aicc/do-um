# NLU 모듈 상세 안내서

## 1) 배경: 스켈레톤에서 현재 구조로

- **실행 가능한 KLUE 기반 라우터** 추가 (`aicc_core_klue.py`)
- `.env` 기반 키 관리 추가 (`load_dotenv`)
- FAQ CSV 기반 **BM25 + Chroma 인덱싱** 및 시맨틱 캐시
- `process_query()` 반환형 인터페이스로 워크플로우 연동 준비
- 운영 확인용 터미널 로그/단계별 소요 시간(`timings_sec`) 제공
- 과거 실험 코드는 `archive/`로 이동해 레포를 정리

---

## 2) 현재 `src/nlu` 파일 구조와 역할

### A. 현재 운영(Active) 파일

#### `src/nlu/aicc_core_klue.py`
- 현재 NLU 모듈의 **메인 실행/연동 파일**
- 주요 기능:
  - `.env` 로드 및 `UPSTAGE_API_KEY` 검증
  - 로컬 KLUE 분류 모델 로드 (`my_aicc_nlu_model_klue:roberta-base`)
  - FAQ CSV(`RAG_FAQ.csv`)를 `Document`로 변환
  - BM25 인덱스 생성 + Chroma 벡터 저장소 적재
  - 시맨틱 캐시 검사 후 결과 반환
  - `process_query()`에서 워크플로우가 사용할 dict 반환
- 반환 형태:
  - 캐시 적중: `status="CACHED"` + `final_answer`
  - 캐시 미적중: `status="REQUIRE_LLM"` + `retrieved_context`/`metadata` (질의 임베딩 벡터는 응답에 포함하지 않음; 내부에서만 사용)
  - 공통: `intent`, `timings_sec`

#### `src/nlu/RAG_FAQ.csv`
- RAG 인덱싱용 FAQ 원천 데이터
- `aicc_core_klue.py`에서 로드하여 문서/메타데이터 생성
- 주요 컬럼(코드 기준): `embedding_text`, `faq_id`, `domain`, `subdomain`, `intent_type`, `risk_level`, `handoff_required`

#### `src/nlu/__init__.py`
- NLU 패키지 마커 파일
- 현재 로직은 없고 모듈 인식 용도

---

### B. 로컬 산출물(버전관리 제외 대상)

#### `src/nlu/my_aicc_nlu_model_klue:roberta-base/`
- 로컬 파인튜닝 모델 아티팩트
- 예: `model.safetensors`, `tokenizer.json`, `config.json`
- `.gitignore`에서 제외되어 Git에 올라가지 않음

#### `src/nlu/aicc_chroma_db/`
- Chroma 로컬 퍼시스트 DB 파일
- 실행 시 FAQ 임베딩/인덱스 결과 저장
- 재생성 가능 산출물이라 `.gitignore` 제외 대상

---

### C. 백업/참고 코드(`archive/`)

#### `src/nlu/archive/aicc_core_system.py`
- 이전 AICC 코어 실험 버전
- 현재 메인 경로에서 제거된 과거 코드

#### `src/nlu/archive/sllm.py`
- sLLM(Zero-shot 계열) 평가 실험 스크립트
- 데이터셋 기반 분류 리포트 실험용

#### `src/nlu/archive/classifier.py`
- 초기 스켈레톤 NLU 인터페이스 코드 백업
- `BaseClassifier`, `sLLMClassifier`, `DLClassifier` 계약 정의 중심

#### `src/nlu/archive/Eval_Queries (2).csv`
- 과거 실험용 질의 평가 데이터셋
- 운영 경로에서는 사용하지 않고 백업 보관

---

## 3) 실행 흐름(현재 기준)

`AICC_NLU_Router.process_query(text)` 기준:

1. 입력 텍스트의 의도 분류 (`predict_intent`)
2. 입력 텍스트 임베딩 생성
3. 시맨틱 캐시 유사도 검사
4. 캐시 적중 시 즉시 반환 (`CACHED`)
5. 미적중 시 Chroma 벡터 검색
6. 검색 컨텍스트/메타데이터와 함께 `REQUIRE_LLM` 반환

즉, 현재 NLU는 워크플로우 입장에서:
- **분기 기준(intent)**
- **검색 컨텍스트**
- **성능 계측 정보(timings)**
를 한번에 제공하는 전처리 라우터 역할입니다.

---

## 4) 초기 스켈레톤 대비 “추가된 핵심 포인트”

초기 스켈레톤(인터페이스-only) 대비 현재 추가된 실질 로직:

- 모델 추론 로직(Transformers 기반)
- 임베딩 및 벡터 저장소 구축
- 캐시 레이어(의미 유사도 기반)
- 환경변수 로딩/검증
- 운영 로그 및 단계별 시간 측정
- 워크플로우 전달용 구조화 dict 반환

---

## 5) 팀원 온보딩 체크리스트

신규 팀원이 NLU를 로컬에서 실행하려면:

1. 저장소 루트에서 `.env` 준비  
   - `cp .env.example .env`
   - `UPSTAGE_API_KEY=...` 입력
2. 의존성 설치  
   - `pip install -r requirements.txt`
3. 로컬 모델 폴더 준비  
   - `src/nlu/my_aicc_nlu_model_klue:roberta-base/` 존재 확인
4. FAQ 데이터 확인  
   - `src/nlu/RAG_FAQ.csv` 존재 확인
5. 실행  
   - `python src/nlu/aicc_core_klue.py`

---

## 6) 유지보수 원칙(권장)

- `archive/`는 백업 전용으로 유지하고 운영 경로와 분리
- 로컬 산출물(모델/DB)은 계속 `.gitignore` 유지
- NLU 출력 스키마(`status`, `intent`, `metadata`, `timings_sec`)는 워크플로우와 계약이므로 변경 시 공유 필수
- 의도 라벨 순서(`intent_map`)는 학습 체크포인트 라벨 인덱스와 항상 동기화

