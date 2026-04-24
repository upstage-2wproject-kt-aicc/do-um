import os
import time
import pandas as pd
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from sklearn.metrics import classification_report

# =====================================================================
# 1. API 키 설정 (테스트할 모델의 키를 입력하세요. 빈칸이어도 에러는 안 납니다)
# =====================================================================
os.environ["UPSTAGE_API_KEY"] = "up_o80maPpk95UkGrqxxd2ldfORTQVWB"
# os.environ["OPENAI_API_KEY"] = "sk-여기에_OPENAI_키_입력"
# os.environ["ANTHROPIC_API_KEY"] = "sk-ant-여기에_ANTHROPIC_키_입력"
# os.environ["GOOGLE_API_KEY"] = "AIzaSy_여기에_GOOGLE_키_입력"

# =====================================================================
# 2. Pydantic 스키마 정의 (CoT 추론 + 의도 분류)
# =====================================================================
class NLUResponse(BaseModel):
    reasoning: str = Field(description="고객의 질문을 분석하여, 이것이 왜 특정 의도에 해당하는지 콜센터 정책에 따라 논리적으로 분석")
    intent: str = Field(description="'설명형', '절차형', '민원형' 중 하나의 값")
    subdomain: str = Field(description="대출/금리, 대출/이자, 카드, 예금/계좌, 보안/피해예방, 정책서민금융 등")

# =====================================================================
# 3. 모델 선택 (테스트하고 싶은 모델 딱 1개만 주석을 해제하세요!)
# =====================================================================

# 🟢 Upstage Solar Pro
from langchain_upstage import ChatUpstage
model_name = "Upstage Solar Pro"
llm = ChatUpstage(model="solar-pro", temperature=0.0)

# 🔵 OpenAI GPT-4o 
# from langchain_openai import ChatOpenAI
# model_name = "OpenAI GPT-4o"
# llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

# 🟣 Anthropic Claude 3.5 Sonnet 
# from langchain_anthropic import ChatAnthropic
# model_name = "Claude 3.5 Sonnet"
# llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0.0)

# 🟡 Google Gemini 1.5 Pro
# from langchain_google_genai import ChatGoogleGenerativeAI
# model_name = "Gemini 1.5 Pro"
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.0)

# =====================================================================
# 구조화된 출력 래핑 (LangChain이 선택된 llm에 맞춰서 동작합니다)
structured_llm = llm.with_structured_output(NLUResponse)

test_df = pd.read_csv("Eval_Queries (2).csv")

predictions = []
actuals = []
latencies = []

print(f"🚀 {model_name} (Ultimate Business Logic 퓨샷) 평가 시작...\n")

# 🌟 [궁극의 고도화] 비즈니스 룰 기반의 Few-Shot + CoT 프롬프트
system_prompt = """당신은 금융권 AICC의 수석 NLU 라우터입니다.
단순한 언어적 의미를 넘어, **실제 콜센터의 업무 처리 정책(Business Logic)**에 따라 의도를 3가지로 정확히 분류하세요.

[콜센터 업무 처리 정책 및 절대 기준]

1. 설명형: 상품의 개념, 원리, 이유, 차이점 등에 대한 지식 설명.
   - 🚨(주의) "중도상환수수료 왜 내야 해요?" -> (설명형) 수수료 부과의 '원리/이유'를 묻는 질문이므로 설명형입니다. 단순 불만이 아닙니다.
   - 예시: "예금을 담보로 대출 가능한가요?" -> (설명형)

2. 절차형: 가입, 해지, 조회, 발급 등 특정 업무의 진행 방식이나 '그에 따른 후속 결과'를 묻는 질문.
   - 🚨(주의) "카드 해지하면 자동납부도 같이 사라지나요?" -> (절차형) 해지라는 '업무 절차'에 수반되는 전산 상의 후속 결과를 묻는 것이므로 절차형입니다.
   - 예시: "내 이름으로 누가 대출받았는지 확인하고 싶어요" -> (절차형)

3. 민원형: 불만, 과다 청구 의심, 보이스피싱, 그리고 **금전적 환불 요청** 등 상담사 개입이 필수적인 질문.
   - 🚨(주의) "연회비 환불되나요?" -> (민원형) 언어적으로는 절차를 묻는 것 같지만, 금융사 정책상 금전적 '환불' 요청은 무조건 민원/이관 부서에서 처리하므로 민원형입니다.
   - 예시: "이번 달 이자가 왜 저번 달보다 많이 나왔죠?" -> (민원형) 과다 청구에 대한 의문과 불만이므로 민원형입니다.

위의 [🚨주의] 케이스들을 완벽하게 숙지하고, reasoning 필드에 해당 정책을 근거로 추론을 작성한 뒤 intent를 출력하세요.
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "고객 질문: {question}")
])

for index, row in test_df.iterrows():
    question = row['user_query']
    actual_intent = row['expected_intent']
    
    start_time = time.time()
    
    try:
        messages = prompt_template.format_messages(question=question)
        result = structured_llm.invoke(messages)
        predicted_intent = result.intent
        
        print(f"질문: {question}")
        print(f"🤔 {model_name}의 생각: {result.reasoning}")
        print(f"✅ 분류 결과: {predicted_intent} (정답: {actual_intent})\n")
        
    except Exception as e:
        print(f"분류 에러 발생: {e}")
        predicted_intent = "분류실패"
        
    end_time = time.time()
    
    predictions.append(predicted_intent)
    actuals.append(actual_intent)
    latencies.append(end_time - start_time)

avg_latency = sum(latencies) / len(latencies)
print(f"⏱️ 평균 추론 속도(Latency): {avg_latency:.4f}초 / 질문당")

print(f"\n🏆 Ultimate 프롬프트 적용 [{model_name}] 최종 성적표")
print(classification_report(actuals, predictions, zero_division=0))