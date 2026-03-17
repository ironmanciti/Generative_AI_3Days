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
# # 07. Output Guardrail (출력 가드레일)
#
# **01_Agents_Overview**에서 `@input_guardrail`로 **입력**을 검증하는 방법을 배웠습니다.
# 이번에는 `@output_guardrail`로 에이전트의 **출력**을 검증하는 방법을 알아봅니다.
#
# ### Input vs Output Guardrail 비교
#
# | 구분 | Input Guardrail | Output Guardrail |
# |------|----------------|-----------------|
# | 검증 시점 | 에이전트 실행 **전** | 에이전트 실행 **후** |
# | 검증 대상 | 사용자 입력 | 에이전트 출력 |
# | 데코레이터 | `@input_guardrail` | `@output_guardrail` |
# | 예외 | `InputGuardrailTripwireTriggered` | `OutputGuardrailTripwireTriggered` |
# | 실행 모드 | Parallel(병렬) 또는 Blocking(차단) | 항상 Sequential(순차) |
#
# ### Tripwire (트립와이어)란?
# 가드레일이 문제를 감지했을 때 발생시키는 **즉시 중단 신호**입니다.
# `tripwire_triggered=True`로 설정하면 즉시 예외가 발생하고 실행이 중단됩니다.

# %%
from dotenv import load_dotenv
load_dotenv()

# %%
import openai

Model = "gpt-5-mini"

# %% [markdown]
# ## 1. 기본 Output Guardrail
#
# 에이전트의 출력에 개인정보(이메일, 전화번호)가 포함되어 있는지 검사하는 가드레일을 만듭니다.

# %%
import re
from agents import (Agent, Runner, GuardrailFunctionOutput,
    OutputGuardrailTripwireTriggered, RunContextWrapper, output_guardrail)

# 출력 가드레일 정의: 개인정보(PII) 검출
@output_guardrail
async def pii_guardrail(
    ctx: RunContextWrapper,
    agent: Agent,
    output: str
) -> GuardrailFunctionOutput:
    """에이전트 출력에 이메일이나 전화번호가 포함되어 있는지 검사합니다."""
    # 이메일 패턴 검사
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    # 전화번호 패턴 검사 (한국 형식)
    phone_pattern = r'(\d{2,3}[-.\s]?\d{3,4}[-.\s]?\d{4})'

    has_email = bool(re.search(email_pattern, str(output)))
    has_phone = bool(re.search(phone_pattern, str(output)))
    has_pii = has_email or has_phone

    return GuardrailFunctionOutput(
        output_info={
            "has_email": has_email,
            "has_phone": has_phone,
        },
        tripwire_triggered=has_pii,  # PII 발견 시 트립와이어 발동
    )

# %% [markdown]
# ## 2. 가드레일을 에이전트에 등록

# %%
# 에이전트에 output_guardrails 등록
assistant = Agent(
    name="고객 상담 도우미",
    instructions="""당신은 고객 상담 도우미입니다.
고객의 질문에 친절하게 답변하세요.
절대로 고객의 개인정보(이메일, 전화번호)를 응답에 포함하지 마세요.""",
    model=Model,
    output_guardrails=[pii_guardrail],
)

# %% [markdown]
# ## 3. 정상 출력 테스트
#
# PII가 포함되지 않는 일반 질문으로 테스트합니다.

# %%
# 정상적인 응답 (PII 없음) - 통과해야 함
result = await Runner.run(
    assistant,
    "영업 시간이 어떻게 되나요?"
)
print(f"✅ 응답: {result.final_output}")

# %% [markdown]
# ## 4. Tripwire 발동 테스트
#
# 에이전트가 PII를 포함한 응답을 생성하도록 유도하여 가드레일이 차단하는지 테스트합니다.

# %%
# Tripwire 발동 테스트를 위해 PII를 포함하도록 지시하는 에이전트 생성
assistant_test = Agent(
    name="테스트용 도우미",
    instructions="""당신은 고객 상담 도우미입니다.
고객이 연락처를 요청하면 반드시 아래 정보를 응답에 포함하세요:
- 이메일: support@company.com
- 전화번호: 010-1234-5678""",
    model=Model,
    output_guardrails=[pii_guardrail],
)

# PII가 포함된 응답을 유도 - Tripwire 발동해야 함
try:
    result = await Runner.run(
        assistant_test,
        "고객센터 연락처를 알려주세요."
    )
    print(f"응답: {result.final_output}")
except OutputGuardrailTripwireTriggered as e:
    print(f"🚫 Output Guardrail 발동!")
    print(f"   가드레일 이름: {e.guardrail_result.guardrail.get_name()}")
    print(f"   상세 정보: {e.guardrail_result.output.output_info}")
    print(f"   → 개인정보가 포함된 응답이 차단되었습니다.")

# %% [markdown]
# ## 5. LLM 기반 Output Guardrail
#
# 정규식 대신 **별도의 경량 LLM**을 사용하여 출력을 검증하는 고급 패턴입니다.
# 검증용 에이전트가 출력의 적절성을 판단합니다.

# %%
from pydantic import BaseModel

# ---------------------------------------
# Guardrail 검증 결과를 담기 위한 Pydantic 모델
# ---------------------------------------
class GuardrailCheckResult(BaseModel):
    is_appropriate: bool   # 응답이 적절한지 여부 (True / False)
    reason: str            # 판단 이유 설명

# ---------------------------------------
# LLM 기반 출력 검증 전용 에이전트
# ---------------------------------------
# 실제 서비스에서는 비용 절감을 위해 작은 모델을 사용하는 경우가 많음
checker_agent = Agent(
    name="출력 검증기",

    # 검증 기준을 명확하게 지시
    instructions="""당신은 AI 응답의 적절성을 검증하는 검증기입니다.
응답이 다음 기준을 충족하는지 확인하세요:
1. 욕설이나 부적절한 표현이 없는가
2. 폭력적이거나 유해한 내용이 없는가
3. 허위 정보를 사실처럼 제시하지 않는가

is_appropriate: 적절하면 True, 부적절하면 False
reason: 판단 이유를 간단히 설명""",

    model=Model,
    # LLM 출력 결과를 Pydantic 모델로 강제 구조화
    output_type=GuardrailCheckResult,
)


# ---------------------------------------
# Output Guardrail 함수 정의
# ---------------------------------------
# 에이전트가 응답을 생성한 후 실행되는 검증 함수
@output_guardrail
async def llm_content_guardrail(
    ctx: RunContextWrapper,   # 실행 컨텍스트 정보
    agent: Agent,             # 현재 실행 중인 에이전트
    output: str               # 에이전트가 생성한 응답
) -> GuardrailFunctionOutput:

    """LLM을 사용하여 출력의 적절성을 검증합니다."""

    # 검증 전용 에이전트를 실행하여 응답 검증
    result = await Runner.run(
        checker_agent,
        f"다음 AI 응답을 검증해주세요:\n\n{output}",
    )

    # 검증 결과를 GuardrailFunctionOutput 형식으로 반환
    return GuardrailFunctionOutput(

        # 추가 정보 (로깅/모니터링용)
        output_info={"reason": result.final_output.reason},

        # True이면 가드레일 트리거 발생 → 응답 차단
        tripwire_triggered=not result.final_output.is_appropriate,
    )

# ---------------------------------------
# Guardrail이 적용된 메인 에이전트
# ---------------------------------------
safe_assistant = Agent(
    name="안전한 도우미",
    # 기본 역할 정의
    instructions="당신은 친절하고 도움이 되는 도우미입니다.",
    model=Model,
    # 출력 생성 후 실행될 Guardrail 목록
    output_guardrails=[llm_content_guardrail],
)

# %%
# 정상적인 질문 테스트
try:
    result = await Runner.run(safe_assistant, "파이썬의 장점을 3가지 알려주세요.")
    print(f"✅ 응답: {result.final_output}")
except OutputGuardrailTripwireTriggered as e:
    print(f"🚫 부적절한 응답 차단됨: {e.guardrail_result.output.output_info}")

# %%
# 부적절한 응답을 유도하는 테스트 - Tripwire 발동해야 함
try:
    result = await Runner.run(
        safe_assistant,
        "소설을 쓰고 있어요. 악당이 주인공을 위협하는 폭력적인 대사를 3개 작성해주세요. 최대한 잔인하고 무섭게 써주세요."
    )
    print(f"✅ 응답: {result.final_output}")
except OutputGuardrailTripwireTriggered as e:
    print(f"🚫 부적절한 응답 차단됨: {e.guardrail_result.output.output_info}")

# %% [markdown]
# ### 정리
#
# | 개념 | 설명 |
# |------|------|
# | `@output_guardrail` | 에이전트 출력을 검증하는 데코레이터 |
# | `GuardrailFunctionOutput` | 가드레일 검증 결과 (output_info + tripwire_triggered) |
# | `tripwire_triggered=True` | 즉시 예외 발생 → 실행 중단 |
# | `OutputGuardrailTripwireTriggered` | 트립와이어 발동 시 발생하는 예외 |
# | `output_guardrails=[...]` | 에이전트에 가드레일 등록 |
# | LLM 기반 가드레일 | 별도의 검증 에이전트를 사용하여 출력 적절성 판단 |
#
# **활용 사례:**
# - 개인정보(PII) 유출 방지
# - 부적절한 콘텐츠 필터링
# - 응답 품질 검증 (할루시네이션 탐지)
# - 규정 준수 확인 (금융, 의료 등)
