# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 12. Multi-Agent Orchestration (코드 기반 멀티에이전트 오케스트레이션)
#
# **04_Handoffs**에서는 LLM이 자율적으로 핸드오프를 결정하는 **LLM 기반 오케스트레이션**을 배웠습니다.
# 이번에는 **코드로 직접** 에이전트 흐름을 제어하는 패턴을 학습합니다.
#
# ### LLM 기반 vs 코드 기반 오케스트레이션
#
# | 구분 | LLM 기반 (Handoffs) | 코드 기반 |
# |------|-------------------|----------|
# | 제어 주체 | LLM이 자율 판단 | 개발자가 코드로 제어 |
# | 예측 가능성 | 낮음 (LLM 판단에 의존) | 높음 (결정적 흐름) |
# | 유연성 | 높음 | 보통 |
# | 비용/속도 | 변동적 | 예측 가능 |
# | 적합한 경우 | 복잡한 대화, 분류가 어려운 경우 | 파이프라인, 배치 처리, 정해진 워크플로우 |
#
# ### 주요 패턴
# 1. **순차 체이닝 (Sequential Chaining)**: A → B → C
# 2. **병렬 실행 (Parallelization)**: A, B, C 동시 실행
# 3. **분류 → 라우팅 (Classification + Routing)**: 분류 후 전문 에이전트에 전달

# %%
from dotenv import load_dotenv
load_dotenv()

# %%
import openai
import asyncio

Model = "gpt-5-mini"

# %% [markdown]
# ## 1. 순차 체이닝 (Sequential Chaining)
#
# 에이전트A의 출력을 에이전트B의 입력으로 전달하는 파이프라인입니다.
# 각 단계가 순서대로 실행됩니다.
#
# ```
# [작성 에이전트] → 초안 → [편집 에이전트] → 수정본 → [번역 에이전트] → 최종 결과
# ```

# %%
from agents import Agent, Runner, trace

# Step 1: 글 작성 에이전트
writer_agent = Agent(
    name="작성기",
    instructions="""당신은 기술 블로그 작성자입니다.
주어진 주제에 대해 3~4문장의 짧은 기술 소개글을 작성하세요.
전문 용어를 적절히 사용하되 이해하기 쉽게 작성하세요.""",
    model=Model,
)

# Step 2: 편집 에이전트
editor_agent = Agent(
    name="편집기",
    instructions="""당신은 전문 편집자입니다.
주어진 글을 다음 기준으로 수정하세요:
- 문법 오류 수정
- 문장 가독성 향상
- 불필요한 표현 제거
수정된 글만 출력하세요.""",
    model=Model,
)

# Step 3: 번역 에이전트
translator_agent = Agent(
    name="번역기",
    instructions="""당신은 한영 번역가입니다.
주어진 한국어 글을 자연스러운 영어로 번역하세요.
번역된 영어만 출력하세요.""",
    model=Model,
)

# 순차 체이닝 실행
with trace("순차 체이닝: 작성 → 편집 → 번역"):
    # Step 1: 글 작성
    write_result = await Runner.run(writer_agent, "에이전트 AI에 대해 소개글을 작성해주세요.")
    print(f"[Step 1] 초안:\n{write_result.final_output}\n")

    # Step 2: 편집 (이전 단계의 출력을 입력으로 전달)
    edit_result = await Runner.run(editor_agent, write_result.final_output)
    print(f"[Step 2] 편집본:\n{edit_result.final_output}\n")

    # Step 3: 번역 (이전 단계의 출력을 입력으로 전달)
    translate_result = await Runner.run(translator_agent, edit_result.final_output)
    print(f"[Step 3] 번역본:\n{translate_result.final_output}")

# %% [markdown]
# ## 2. 병렬 실행 (Parallelization)
#
# 서로 독립적인 에이전트를 `asyncio.gather()`로 **동시에** 실행하여 시간을 절약합니다.
#
# ```
#         ┌→ [감성 분석 에이전트] → 결과1   ─┐
# [입력] ──┼→ [키워드 추출 에이전트] → 결과2 ─┼→ [종합]
#         └→ [요약 에이전트] → 결과3   ─────┘
# ```

# %%
# 3개의 독립적인 분석 에이전트
sentiment_agent = Agent(
    name="감성 분석기",
    instructions="""주어진 텍스트의 감성을 분석하세요.
결과 형식: "감성: [긍정/부정/중립], 신뢰도: [높음/보통/낮음], 이유: [간단한 설명]" """,
    model=Model,
)

keyword_agent = Agent(
    name="키워드 추출기",
    instructions="""주어진 텍스트에서 핵심 키워드 3~5개를 추출하세요.
결과 형식: "키워드: [키워드1], [키워드2], [키워드3], ..." """,
    model=Model,
)

summary_agent = Agent(
    name="요약기",
    instructions="""주어진 텍스트를 한 문장으로 요약하세요.
결과 형식: "요약: [한 문장 요약]" """,
    model=Model,
)

text_to_analyze = """
OpenAI의 Agent SDK는 개발자들이 AI 에이전트를 쉽게 만들 수 있도록 설계되었습니다.
핸드오프, 도구 사용, 가드레일 등의 기능을 제공하여 복잡한 AI 워크플로우를
간단한 코드로 구현할 수 있게 해줍니다. 특히 Python 개발자에게 친숙한 인터페이스를
제공하며, 프로덕션 환경에서의 안정성도 고려되어 있습니다.
"""

# asyncio.gather로 3개 에이전트를 동시에 실행
with trace("병렬 분석"):
    sentiment_result, keyword_result, summary_result = await asyncio.gather(
        Runner.run(sentiment_agent, text_to_analyze),
        Runner.run(keyword_agent, text_to_analyze),
        Runner.run(summary_agent, text_to_analyze),
    )

print(f"{sentiment_result.final_output}")
print(f"{keyword_result.final_output}")
print(f"{summary_result.final_output}")

# %% [markdown]
# ## 3. 분류 → 라우팅 (Classification + Routing)
#
# **Structured Output**으로 입력을 분류한 후, 결과에 따라 적절한 전문 에이전트를 실행합니다.
#
# ```
# [입력] → [분류 에이전트] → 카테고리 → [전문 에이전트 선택] → [전문 에이전트 실행] → [응답]
# ```

# %%
from pydantic import BaseModel

# 분류 결과를 위한 Pydantic 모델
class RequestClassification(BaseModel):
    category: str  # "기술지원", "결제문의", "일반문의"
    confidence: float  # 0.0 ~ 1.0
    reason: str

# 분류 에이전트
classifier_agent = Agent(
    name="요청 분류기",
    instructions="""고객의 요청을 다음 카테고리 중 하나로 분류하세요:
- "기술지원": 소프트웨어 오류, 설치 문제, 기술적 질문
- "결제문의": 요금, 환불, 결제 수단, 구독 관련
- "일반문의": 영업시간, 위치, 서비스 소개 등 일반 질문

category에 정확한 카테고리명을 넣고, confidence에 확신도(0~1)를 넣으세요.""",
    model=Model,
    output_type=RequestClassification,
)

# 전문 에이전트들
tech_support_agent = Agent(
    name="기술지원 전문가",
    instructions="당신은 기술지원 전문가입니다. 소프트웨어 문제를 해결하고 기술적 도움을 제공합니다.",
    model=Model,
)

billing_agent = Agent(
    name="결제 전문가",
    instructions="당신은 결제 전문가입니다. 요금, 환불, 구독 관련 질문에 답변합니다.",
    model=Model,
)

general_agent = Agent(
    name="일반 상담사",
    instructions="당신은 일반 상담사입니다. 서비스에 대한 일반적인 질문에 답변합니다.",
    model=Model,
)

# 카테고리 → 에이전트 매핑
agent_map = {
    "기술지원": tech_support_agent,
    "결제문의": billing_agent,
    "일반문의": general_agent,
}

# %%
# 분류 → 라우팅 실행
async def handle_customer_request(request: str):
    with trace("분류 → 라우팅"):
        # Step 1: 분류
        classification_result = await Runner.run(classifier_agent, request)
        classification = classification_result.final_output
        print(f"분류 결과: {classification.category} (확신도: {classification.confidence:.0%})")
        print(f"   이유: {classification.reason}")

        # Step 2: 라우팅 - 분류 결과에 따라 적절한 에이전트 선택
        selected_agent = agent_map.get(classification.category, general_agent)
        print(f"선택된 에이전트: {selected_agent.name}\n")

        # Step 3: 전문 에이전트 실행
        response = await Runner.run(selected_agent, request)
        print(f"응답: {response.final_output}")

    return response.final_output

# %%
# 테스트 1: 기술지원 질문
print("=" * 60)
print("테스트 1: 기술지원")
print("=" * 60)
await handle_customer_request("프로그램이 갑자기 멈추고 에러 메시지가 떠요. 어떻게 해야 하나요?")

# %%
# 테스트 2: 결제 질문
print("=" * 60)
print("테스트 2: 결제문의")
print("=" * 60)
await handle_customer_request("지난달 결제가 두 번 된 것 같아요. 환불 가능한가요?")

# %%
# 테스트 3: 일반 질문
print("=" * 60)
print("테스트 3: 일반문의")
print("=" * 60)
await handle_customer_request("주말에도 고객센터가 운영되나요?")

# %% [markdown]
# ## 4. 종합 패턴: 병렬 + 순차 + 체이닝 조합
#
# 실전에서는 여러 패턴을 **조합**하여 사용합니다.
#
# ```
# [고객 리뷰] ──→ [병렬 분석] ──→ [종합 에이전트] ──→ [최종 보고서]
#                  ├ 감성 분석
#                  ├ 키워드 추출
#                  └ 요약
# ```

# %%
# 종합 보고서를 작성하는 에이전트
report_agent = Agent(
    name="보고서 작성기",
    instructions="""당신은 분석 보고서 작성자입니다.
주어진 분석 결과들을 종합하여 간결한 보고서를 작성하세요.
보고서 형식:
---
[분석 보고서]
- 요약: ...
- 감성: ...
- 핵심 키워드: ...
- 종합 의견: (위 분석을 바탕으로 한 줄 의견)
---""",
    model=Model,
)

customer_review = """
이 제품 정말 좋아요! 배송도 빠르고 포장도 꼼꼼했습니다.
다만 설명서가 영어로만 되어 있어서 처음에 좀 헷갈렸어요.
전체적으로는 가격 대비 성능이 훌륭하고, 다음에도 이 브랜드를 구매할 의향이 있습니다.
"""

with trace("종합 분석 파이프라인"):
    # Step 1: 병렬 분석 (동시 실행)
    print("Step 1: 병렬 분석 실행 중...\n")
    s_result, k_result, sm_result = await asyncio.gather(
        Runner.run(sentiment_agent, customer_review),
        Runner.run(keyword_agent, customer_review),
        Runner.run(summary_agent, customer_review),
    )

    print(f"  감성: {s_result.final_output}")
    print(f"  키워드: {k_result.final_output}")
    print(f"  요약: {sm_result.final_output}")

    # Step 2: 순차 체이닝 - 분석 결과를 종합
    print(f"\nStep 2: 종합 보고서 작성 중...\n")
    combined_input = f"""다음 분석 결과를 종합해주세요:

원본 리뷰: {customer_review}

분석 결과:
1. {s_result.final_output}
2. {k_result.final_output}
3. {sm_result.final_output}
"""

report_result = await Runner.run(report_agent, combined_input)
print(f"📊 최종 보고서:\n{report_result.final_output}")

# %% [markdown]
# ---------------------
# ### 정리
#
# | 패턴 | 코드 | 적합한 경우 |
# |------|------|-----------|
# | 순차 체이닝 | `result1 = run(A, input)` → `result2 = run(B, result1.final_output)` | 파이프라인, 단계별 처리 |
# | 병렬 실행 | `await asyncio.gather(run(A, x), run(B, x), run(C, x))` | 독립적 분석, 시간 절약 |
# | 분류 → 라우팅 | `Structured Output`으로 분류 → `agent_map[category]` | 고객 문의 분류, 의도 분석 |
# | 조합 패턴 | 병렬 + 순차를 결합 | 실전 워크플로우 |
#
# **핵심 포인트:**
# - `trace()`로 전체 워크플로우를 하나의 트레이스로 묶어 추적
# - `asyncio.gather()`로 독립적인 작업을 병렬 실행하여 성능 향상
# - `Structured Output (output_type)`으로 결정적(deterministic) 라우팅 구현
# - 코드 기반 오케스트레이션은 **예측 가능성**과 **비용 제어**에 유리

# %%
