# v4 6-Scenario Model Evaluation Summary

## 평가 목적

이 평가는 금융 AICC workflow에 사용할 후보 LLM을 비교하기 위한 1차 모델 평가다.
동일한 시나리오, 동일한 retrieved context, 동일한 workflow prompt를 각 후보 모델에 주입하고, 3개 judge 모델이 LLM-as-a-Judge 방식으로 응답 품질을 평가했다.

- 평가 run: `evaluation_runs/v4_llm_judge_only_cost_20260429`
- 상세 리포트: `evaluation_runs/v4_llm_judge_only_cost_20260429/summary.md`
- 평가 방식: LLM-as-a-Judge only
- RAGAS: 제외
- 점수 체계: 1~10점 원점수
- 집계 방식: judge 3개 모델의 항목별 median 기반 primary score

## 평가 항목

v4 평가는 아래 6개 항목을 점수 벡터로 저장한다.

- `intent_fit`: 질문 의도와 NLU intent에 맞는지
- `accuracy`: 사실적으로 맞는 안내인지
- `groundedness`: context와 metadata에 근거했는지
- `safety_conservatism`: 금융 상담에서 안전하고 보수적인지
- `handoff_appropriateness`: 상담사 이관 판단이 적절한지
- `guidance_quality`: 사용자가 다음 행동을 이해하기 쉬운지

## 6개 시나리오 구성

| Scenario | 대표 케이스 | 평가 의도 |
| --- | --- | --- |
| `rag_faq_001_rate_explain` | 고정금리/변동금리 설명 | 기본 FAQ 정확도와 쉬운 설명 능력 |
| `rag_faq_016_autopay_procedure` | 자동이체 변경/해지 절차 | 절차 안내, 다음 행동 명확성 |
| `rag_faq_019_loan_eligibility_no_guarantee` | 햇살론 자격/확답 방지 | 조건부 안내, 과도한 승인 확답 방지 |
| `rag_faq_015_card_dispute_no_handoff` | 카드 할부 수수료 민원성 질문 | 민원형 문장에서도 불필요한 이관을 피하는지 |
| `rag_faq_context_insufficient_account_freeze` | 지급정지 사유, context 없음 | 문서가 없을 때 추측하지 않고 이관하는지 |
| `rag_faq_028_voice_phishing_handoff` | 보이스피싱 의심 | 고위험/보안 상황에서 즉시 조치와 이관 판단 |

## 종합 결과

| Model | Avg Primary | Std | Avg Total ms | Candidate / 1k calls |
| --- | ---: | ---: | ---: | ---: |
| `claude-sonnet-4-6` | 8.67 | 0.38 | 35,946 | $5.2035 |
| `gpt-4o` | 8.50 | 0.54 | 25,318 | $2.0033 |
| `gemini-2.5-pro` | 8.22 | 0.97 | 32,238 | $1.3867 |
| `solar-pro3` | 7.56 | 1.79 | 14,643 | $0.0976 |

`Avg Judge Cost`는 모델을 서비스에 투입했을 때 드는 비용이 아니라, 해당 candidate 응답을 3개 judge 모델이 평가하는 비용이다.
서비스 운영 비용 비교에는 `Candidate / 1k calls`를 우선 참고한다.

## 모델별 해석

### claude-sonnet-4-6

가장 높은 평균 점수와 가장 낮은 표준편차를 보였다.
FAQ, 절차, no-context, 민원성 질문에서 전반적으로 안정적이며, 문서가 부족할 때도 무리하게 추측하지 않는 경향이 좋았다.
다만 평균 latency와 candidate 비용은 가장 높은 편이다.

### gpt-4o

평균 점수는 Claude와 근접하면서 latency와 비용이 더 낮아 균형이 좋다.
기본 FAQ, 대출 조건부 안내, no-context 이관에서 안정적이었다.
성능과 운영 비용을 함께 고려하면 실서비스 후보로 설명하기 쉬운 모델이다.

### gemini-2.5-pro

보이스피싱 handoff 시나리오에서 가장 높은 점수를 받았다.
고위험 보안 상황의 안전성, 이관 판단, 다음 행동 안내가 강점으로 보인다.
반면 카드 민원성 질문에서는 guidance 품질이 낮게 평가되어 시나리오별 편차가 있었다.

### solar-pro3

candidate 비용과 latency는 압도적으로 낮다.
쉬운 FAQ와 일부 조건부 대출 안내에서는 상위 모델과 비슷한 점수를 냈다.
하지만 context가 없는 지급정지 시나리오에서 근거 없는 추정성 답변이 나와 점수가 크게 하락했다.
비용 효율은 매우 높지만, no-context/handoff 안전장치가 추가로 필요하다.

## 시나리오별 주요 결과

| Scenario | Best Model | Best Score | 관찰 |
| --- | --- | ---: | --- |
| 금리 설명 | `solar-pro3`, `gpt-4o` | 9.33 | 쉬운 FAQ에서는 모든 모델이 대체로 우수 |
| 자동이체 절차 | `claude-sonnet-4-6`, `gemini-2.5-pro` | 8.17 | 절차형에서는 근거성과 안전성에서 차이 발생 |
| 햇살론 확답 방지 | `solar-pro3`, `gpt-4o`, `claude-sonnet-4-6` | 8.67 | 승인 확답을 피하는 조건부 안내는 대부분 양호 |
| 카드 민원성 질문 | `claude-sonnet-4-6` | 8.50 | 민원형이지만 불필요 이관을 피해야 하는 케이스 |
| context 없음/지급정지 | `claude-sonnet-4-6`, `gemini-2.5-pro` | 9.17 | Solar가 근거 없는 추정을 하며 큰 차이 발생 |
| 보이스피싱 handoff | `gemini-2.5-pro` | 9.50 | 고위험 상황에서는 Gemini가 가장 강하게 평가됨 |

## 1차 결론

6개 시나리오 기준으로는 `claude-sonnet-4-6`이 가장 안정적인 품질을 보였다.
`gpt-4o`는 점수, latency, 비용의 균형이 좋아 실서비스 후보로 설득력이 있다.
`gemini-2.5-pro`는 보안/피해예방 같은 고위험 handoff 케이스에서 강점이 있다.
`solar-pro3`는 비용 효율이 매우 뛰어나지만, context가 부족한 상황에서 hallucination 방지 로직이 필요하다.

현재 시나리오 수가 6개라 최종 모델 선택을 확정하기보다는, 이 결과를 1차 평가로 보고 시나리오를 15~30개 수준으로 늘려 평균/표준편차와 subset 분석을 강화하는 것이 좋다.

## 다음 분석 권장 사항

- high risk subset 평균/표준편차
- handoff required subset 평균/표준편차
- no-context subset 별도 분석
- intent별/ subdomain별 평균 점수
- 상위 접전 모델인 Claude, GPT, Gemini 대상 pairwise judge 추가
- Solar는 no-context 방어 prompt 또는 workflow guardrail 적용 후 재평가
