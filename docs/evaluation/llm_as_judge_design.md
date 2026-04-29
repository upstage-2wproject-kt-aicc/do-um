# LLM-as-a-Judge Evaluation Design Notes

이 문서는 AICC 모델 평가에서 LLM-as-a-Judge를 사용할 때의 설계 원칙을 기록한다.
주요 참고 자료는 카카오테크의 `LLM as a Judge를 활용한 CodeBuddy 성능 평가` 글이다.

- Reference: https://tech.kakao.com/posts/690

## 우리 평가의 목적

평가 목적은 AICC workflow에 투입할 후보 LLM 모델이 금융 상담 응답 생성에 적합한지 비교하는 것이다.
모델별 응답을 같은 시나리오, 같은 retrieved context, 같은 workflow prompt 조건에서 생성한 뒤 LLM-as-a-Judge로 평가한다.

현재 운영 기준은 `comparative_10` rubric이다.
primary score는 LLM-as-a-Judge 6개 항목의 1~10점 원점수 중앙값 기반 종합 점수다.
RAGAS는 초기 검토 단계에서 보조 지표로 고려했지만, v4 운영 평가에서는 제외한다.

## 카카오 글에서 가져온 적용 포인트

### 1. System/User 영역을 명확히 분리한다

카카오 글에서는 평가 프롬프트에서 task의 system message와 user message를 명확히 구분했을 때 평가 기준이 더 잘 전달되고 변별력이 좋아졌다고 설명한다.
반대로 둘을 구분하지 않으면 무승부나 애매한 판단이 늘어날 수 있다.

우리 평가에서는 candidate 모델이 실제로 받은 workflow prompt를 judge 입력에 다음처럼 분리해서 넣는다.

- `workflow_prompt_given_to_candidate.system_message`
- `workflow_prompt_given_to_candidate.user_message`
- `workflow_prompt_given_to_candidate.route`

이 구조를 통해 judge는 후보 답변이 단순히 질문에 맞는지만 보지 않고, route별 system instruction, 길이 제한, 안전성 지시, context 사용 지시를 따랐는지 함께 판단할 수 있다.

### 2. 여러 judge 모델의 median을 사용한다

LLM judge는 자기 편향, 장황성 편향, 위치 편향을 가질 수 있다.
단일 judge만 사용하면 특정 모델 계열이나 특정 문체에 유리한 결과가 나올 수 있으므로, 최종 리포트용 평가는 여러 최상위 judge 모델의 중앙값을 사용한다.

현재 운영 구조는 다음과 같다.

- candidate 모델: 서비스에 투입할 후보 응답 생성 모델
- judge 모델: OpenAI, Anthropic, Google 계열의 상위 모델
- aggregation: 항목별 1~10 raw score median
- report: 6개 점수 벡터, primary score, latency, token usage, estimated cost

## v4에서 RAGAS를 제외하는 이유

`comparative_10`에서는 RAGAS를 사용하지 않는다.
LLM-as-a-Judge 프롬프트가 context, metadata, handoff rule, source_url 분리, workflow prompt 준수 여부를 함께 보도록 설계되어 있어, 기존 RAGAS 보조 지표의 주요 역할을 대부분 흡수한다.

- `groundedness`: Faithfulness 역할을 대체한다.
- `intent_fit`과 `guidance_quality`: Answer Relevancy 역할을 대체한다.
- `safety_conservatism`: 금융 상담의 위험한 단정 표현을 별도 평가한다.
- `handoff_appropriateness`: `risk_level`, `handoff_required`, no-context 상황의 이관 판단을 별도 평가한다.

특히 handoff/no-context 시나리오는 RAGAS 점수 해석이 불안정할 수 있다.
문서가 없는 상황에서 상담사 이관을 잘한 답변은 서비스 관점에서는 좋은 응답이지만, 자동 relevancy/faithfulness 계열 지표에서는 낮게 보일 수 있다.
따라서 최종 운영 평가는 LLM-as-a-Judge only로 유지한다.

### 3. Pairwise 평가는 tie-breaker로만 고려한다

카카오 글은 pairwise 비교와 position bias 검증을 다룬다.
우리의 기본 평가는 single answer grading이므로 position bias 영향이 상대적으로 작다.
다만 상위 모델들의 점수가 비슷해 변별력이 부족한 경우에는 pairwise 비교를 tie-breaker로 추가할 수 있다.

pairwise를 도입할 경우 반드시 순서를 뒤집어 평가한다.

- A vs B
- B vs A

두 결과가 크게 다르면 position bias 가능성을 별도 리스크로 기록한다.

### 4. Judge disagreement를 리포트에 노출한다

judge 모델 간 점수 차이가 큰 항목은 평가 기준이 애매하거나 답변 품질 차이가 미묘한 케이스일 수 있다.
현재 `AggregatedMetricScore.disagreement`가 있으므로, 리포트에서는 평균 점수뿐 아니라 disagreement가 큰 시나리오를 따로 표시하는 것이 좋다.

## judge_v2 변경 사항

`judge_v2.md`는 `judge_v1.md` 대비 다음을 강화한다.

- candidate가 받은 workflow system/user message를 분리해서 읽도록 지시
- route 지시, context, reference answer, metadata를 기준으로 평가하도록 명시
- 장황한 답변을 선호하지 말라는 anti-verbosity 지시 추가
- 같은 모델 계열 candidate를 선호하지 말라는 anti-self-bias 지시 추가
- `risk_level`, `handoff_required` metadata를 handoff 판단에 반영하도록 명시
- `source_url`이 있으면 근거 링크 또는 공식 확인 경로 안내 여부를 보도록 명시

## 운영 해석 원칙

- `primary_score`: v4 judge 6개 항목 기반 종합 점수
- `intent_fit`, `accuracy`, `groundedness`, `safety_conservatism`, `handoff_appropriateness`, `guidance_quality`: 모델별 응답 성향 비교용 점수 벡터
- `review_required`: 모델 성능 비교 목적에서는 실패 라벨이 아니라 상세 확인 대상 표시로만 사용한다
- `latency`, `token_usage`, `estimated_cost`: 점수가 접전일 때 모델 선택을 보조하는 운영 지표

## 다음 개선 후보

- pairwise tie-breaker mode 추가
- disagreement가 큰 항목을 자동 요약하는 report script 추가
- high risk, handoff, no-context, intent, subdomain subset별 평균/표준편차 report 추가
- judge prompt v4의 점수 차이를 같은 시나리오로 비교하는 prompt regression run
