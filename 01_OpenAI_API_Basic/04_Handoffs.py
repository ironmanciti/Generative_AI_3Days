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
# # 04. Handoffs (핸드오프 심화)
#
# **핸드오프(Handoff)** 는 에이전트가 특정 작업을 다른 에이전트에게 위임하는 기능입니다.  
# 핸드오프는 LLM에게 **도구(tool)** 로 표현됩니다.  
# 예) `Refund Agent`로의 핸드오프 → LLM 도구 이름: `transfer_to_refund_agent`
#
# | 방식 | 예시 | 특징 |
# |------|------|------|
# | Agent 직접 전달 | `handoffs=[billing_agent]` | 간단, 기본 동작 |
# | `handoff()` 함수 | `handoffs=[handoff(billing_agent, on_handoff=...)]` | 콜백, 이름 재정의, 입력 필터 등 고급 옵션 |
#
# **`handoff()` 함수의 주요 파라미터:**
#
# | 파라미터 | 설명 |
# |---------|------|
# | `agent` | 위임할 대상 에이전트 |
# | `tool_name_override` | LLM에게 노출되는 도구 이름 재정의 (기본: `transfer_to_<agent_name>`) |
# | `tool_description_override` | 도구 설명 재정의 |
# | `on_handoff` | 핸드오프 호출 시 실행되는 콜백 함수 |
# | `input_type` | LLM이 핸드오프 시 함께 전달할 데이터 타입 (Pydantic 모델) |
# | `input_filter` | 다음 에이전트에게 전달되는 대화 기록 필터링 함수 |
# | `is_enabled` | 핸드오프 활성화 여부 (bool 또는 런타임 함수) |

# %%
from dotenv import load_dotenv
load_dotenv()

# %%
Model = "gpt-5-nano"

# %% [markdown]
# ## 1. handoff() 함수 기본 사용법
#
# Agent 인스턴스를 직접 전달하는 방식과 `handoff()` 함수를 사용하는 방식은 **동일하게 동작**합니다.  
# `handoff()` 함수 형태는 추가 옵션이 필요할 때 사용합니다.

# %%
from agents import Agent, Runner, handoff

# --- 1. 하위 에이전트(Sub-Agent) 정의 ---

# 청구(Billing) 전담 에이전트
billing_agent = Agent(
    name="Billing agent",
    instructions="당신은 청구 관련 질문을 전문으로 처리합니다.",
    model=Model
)

# 환불(Refund) 전담 에이전트
refund_agent = Agent(
    name="Refund agent",
    instructions="당신은 환불 요청을 전문으로 처리합니다.",
    model=Model
)

# --- 2. 트리아지(분류) 에이전트 정의 ---
# 사용자 요청을 분석한 뒤, 적절한 하위 에이전트로 핸드오프(handoff)하는 역할
triage_agent = Agent(
    name="Triage agent",
    instructions="사용자 요청을 분류하여 적합한 에이전트에게 넘겨주세요.",
    model=Model,
    handoffs=[
        # 방법 1 : Agent 인스턴스를 직접 전달 → 자동으로 tool_name/description 생성
        billing_agent,

        # 방법 2 : handoff() 함수로 감싸서 전달 → tool 이름·설명을 직접 커스터마이징
        handoff(
            refund_agent,
            tool_name_override="request_refund",           # LLM이 호출할 tool 이름 재정의
            tool_description_override="환불 요청이 있을 때 사용",  # tool 설명 재정의
        ),
    ],
)

# --- 3. 실행 ---
# triage_agent가 입력을 받아 내부적으로 billing_agent 또는 refund_agent로 핸드오프
result = await Runner.run(triage_agent, input="제 청구서에 오류가 있는 것 같습니다.")

# 최종 출력 — 핸드오프된 에이전트(billing_agent)가 생성한 응답
print(result.final_output)

# %%
result.last_agent.name

# %% [markdown]
# ## 2. on_handoff 콜백
#
# `on_handoff` 콜백은 핸드오프가 호출되는 순간 실행됩니다.  
# 데이터 준비, 로깅, 알림 등 사이드 이펙트 처리에 유용합니다.
#
# - `input_type` 없이 사용 시: `def on_handoff(ctx: RunContextWrapper[None])`  
# - `input_type` 함께 사용 시: `async def on_handoff(ctx, input_data: MyModel)` (섹션 3 참고)

# %%
from agents import Agent, Runner, handoff, RunContextWrapper

# --- 1. 핸드오프 콜백(callback) 함수 정의 ---

# 핸드오프가 실행되는 시점에 자동 호출되는 함수
def on_billing_handoff(ctx: RunContextWrapper[None]):
    """Billing Agent로 핸드오프될 때 실행되는 콜백"""
    print("[로그] Billing Agent로 핸드오프 발생")

def on_refund_handoff(ctx: RunContextWrapper[None]):
    """Refund Agent로 핸드오프될 때 실행되는 콜백"""
    print("[로그] Refund Agent로 핸드오프 발생")

# --- 2. 하위 에이전트 정의 ---

# 청구 전담 에이전트
billing_agent = Agent(
    name="Billing agent",
    instructions="당신은 청구 관련 질문을 전문으로 처리합니다.",
    model=Model
)

# 환불 전담 에이전트
refund_agent = Agent(
    name="Refund agent",
    instructions="당신은 환불 요청을 전문으로 처리합니다.",
    model=Model
)

# --- 3. 트리아지(분류) 에이전트 정의 ---
triage_agent = Agent(
    name="Triage agent",
    instructions="사용자 요청을 분류하여 적합한 에이전트에게 넘겨주세요.",
    model=Model,
    handoffs=[
        # on_handoff 파라미터로 콜백 등록
        # 핸드오프 실행 직전에 해당 콜백이 호출됨
        handoff(billing_agent, on_handoff=on_billing_handoff),
        handoff(refund_agent, on_handoff=on_refund_handoff),
    ],
)

# --- 4. 실행 ---
# 입력: "환불을 받고 싶습니다." → triage_agent가 환불 요청으로 분류
# → on_refund_handoff 콜백 실행 → refund_agent로 핸드오프 → 최종 응답 생성
result = await Runner.run(triage_agent, input="환불을 받고 싶습니다.")

print(result.final_output)

# %% [markdown]
# ## 3. Handoff 입력 데이터 (input_type)
#
# `input_type`을 사용하면 LLM이 핸드오프 시 **구조화된 데이터를 함께 전달**하도록 할 수 있습니다.  
# 예를 들어, 에스컬레이션 이유를 함께 전달받아 로깅하거나 처리에 활용할 수 있습니다.
#
# - `input_type`이 있을 때 `on_handoff`는 `async def`로 정의하고 두 번째 인자로 `input_data`를 받습니다.

# %%
from pydantic import BaseModel
from agents import Agent, Runner, handoff, RunContextWrapper
from datetime import datetime

# 현재 시간을 문자열로 생성 (에이전트 프롬프트에 사용)
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 에스컬레이션 시 전달할 데이터 구조 정의
# Pydantic 모델을 사용하면 에이전트가 전달해야 하는 입력 형식을 강제할 수 있음
class EscalationData(BaseModel):
    reason: str          # 에스컬레이션 사유
    datetime: str        # 접수 시간


# 에스컬레이션이 발생했을 때 실행되는 콜백 함수
# ctx : 실행 컨텍스트
# input_data : 에이전트가 전달한 EscalationData 객체
async def on_escalation(ctx: RunContextWrapper[None], input_data: EscalationData):
    print(f"[에스컬레이션] 이유: {input_data.reason}, 접수시간: {input_data.datetime}")


# 복잡한 문제를 처리하는 전문 상담 에이전트
# Support agent가 처리하지 못하는 문제를 넘겨받아 해결
escalation_agent = Agent(
    name="Escalation agent",
    instructions="당신은 복잡한 문제를 처리하는 전문 상담사입니다.",
    model=Model
)

# 일반 고객 지원을 처리하는 1차 상담 에이전트
support_agent = Agent(
    name="Support agent",
    instructions=(
        f"현재 시간은 {now} 입니다."
        "일반 고객 지원 요청을 처리하세요. "
        "복잡하거나 민감한 문제는 에스컬레이션 에이전트에게 이유와 접수시간을 함께 넘겨주세요."
    ),
    model=Model,

    # 다른 에이전트로 작업을 넘기는 handoff 설정
    handoffs=[
        handoff(
            escalation_agent,      # 넘겨받을 대상 에이전트
            on_handoff=on_escalation,  # handoff 발생 시 실행할 콜백
            input_type=EscalationData, # 전달해야 하는 데이터 형식
        ),
    ],
)

# Support agent 실행
# 사용자의 입력을 전달하면 필요 시 escalation_agent로 handoff 발생
result = await Runner.run(
    support_agent,
    input="제 계정이 해킹당한 것 같습니다. 즉시 도움이 필요합니다!"
)

# 최종 에이전트 응답 출력
print(result.final_output)

# %% [markdown]
# ## 4. Input Filter (입력 필터)
#
# 핸드오프가 발생하면 새로운 에이전트는 기존 대화 기록 전체를 받습니다.  
# `input_filter`를 사용하면 다음 에이전트에게 전달되는 **대화 기록을 가공**할 수 있습니다.
#
# `agents.extensions.handoff_filters`에는 자주 사용되는 내장 필터가 제공됩니다:
#
# | 필터 | 설명 |
# |------|------|
# | `remove_all_tools` | 대화 기록에서 모든 도구 호출/결과 제거 |
# |`nest_handoff_history` | 이전 대화 내용을 요약해서 단일 assistant 메시지로 압축 후 전달 |
#
# ### 흐름 설명
# ```
# 사용자: "주문번호 ORD-1234 배송 조회해주고, 반품 정책도 알려줘"
#       ↓
# main_agent
#   → check_shipping_status("ORD-1234") 도구 호출  ← 도구 메시지 생성
#   → "반품 정책은 FAQ agent한테 넘겨야겠다"
#   → remove_all_tools 필터 실행
#       ┌─────────────────────────────────────┐
#       │ 도구 호출 기록 제거 전                 │
#       │ [user] 배송 조회해주고 반품도 알려줘    │
#       │ [tool_call] check_shipping_status   │  ← 제거됨
#       │ [tool_result] 배송 중...             │  ← 제거됨
#       └─────────────────────────────────────┘
#       ↓
# faq_agent (도구 기록 없이 깔끔한 대화만 전달)
#   → 반품 정책 답변
# ```

# %%
from agents import Agent, Runner, handoff, function_tool
from agents.extensions import handoff_filters

# -------------------------------
# 도구(tool) 정의
# -------------------------------

@function_tool
def check_shipping_status(order_id: str) -> str:
    """
    주문 ID로 배송 상태를 조회하는 도구 함수
    function_tool 데코레이터를 사용하면
    LLM이 호출 가능한 Tool로 자동 등록됨
    """
    print(f"** check_shipping_status 실행 ** order_id: {order_id}")

    # 실제 시스템에서는 DB 또는 API 조회
    return f"주문 {order_id}의 배송 상태: 배송 중 (예상 도착: 내일)"


# -------------------------------
# FAQ 에이전트 정의
# -------------------------------
faq_agent = Agent(
    name="FAQ agent",
    # FAQ 관련 질문을 담당하는 에이전트
    instructions="자주 묻는 질문에 답변합니다. 대화 기록에 도구 호출 내역이 있는지 확인하세요.",

    model=Model
)


# -------------------------------
# 메인 에이전트 정의
# -------------------------------
main_agent = Agent(
    name="Main agent",
    # 역할 정의
    instructions=(
        "당신은 배송 조회만 담당합니다. "
        "배송 조회는 check_shipping_status 도구를 사용하세요. "
        "반품, 환불, 정책 등 FAQ 성격의 질문에는 직접 답변하지 말고 "
        "반드시 FAQ agent에게 핸드오프하세요."
    ),

    model=Model,
    # 이 에이전트가 사용할 수 있는 Tool 목록
    tools=[check_shipping_status],

    # 다른 에이전트에게 작업을 넘기는 handoff 설정
    handoffs=[
        handoff(
            faq_agent,  # 작업을 넘길 대상 에이전트

            # handoff 시 tool 호출 기록 등을 제거하는 필터
            # (주석 해제하면 tool call 기록을 제거하고 전달)
            # input_filter=handoff_filters.remove_all_tools,
        ),
    ],
)


# -------------------------------
# 에이전트 실행
# -------------------------------
result = await Runner.run(
    main_agent,

    # 사용자 입력
    # 배송조회 + FAQ 질문이 동시에 포함된 요청
    input="주문번호 ORD-1234 배송 조회해주고, 반품 정책도 알려줘"
)

# 최종 에이전트 응답 출력
print(result.final_output)

# 마지막으로 응답한 에이전트 확인
# handoff가 발생했다면 FAQ agent가 출력됨
print(f"\n최종 응답 에이전트: {result.last_agent.name}")

# %%
# 지금까지의 대화 기록을 다음 Runner.run 호출에서
# 그대로 이어서 사용할 수 있는 형태로 만들어 줌
result.to_input_list()

# %% [markdown]
# ## 5. 권장 프롬프트 (RECOMMENDED_PROMPT_PREFIX)
#
# LLM이 핸드오프를 올바르게 이해하고 활용하려면 관련 정보를 프롬프트에 포함시키는 것이 좋습니다.  
# ```
# LLM은 핸드오프가 뭔지 기본적으로 모릅니다. 그래서 "너는 다른 에이전트에게 작업을 넘길 수 있어" 라고 미리 알려줘야 합니다.
#
# 핸드오프 설명 없이:
#   LLM: "나는 그냥 질문에 답하는 AI야"
#   → 핸드오프 도구가 있어도 잘 안 씀
#
# 핸드오프 설명 포함:
#   LLM: "나는 다른 전문 에이전트에게 작업을 위임할 수 있어"
#   → 핸드오프를 적극적으로 활용
# ```
# `agents.extensions.handoff_prompt`에서 권장 프롬프트를 제공합니다:
#
# | 방법 | 설명 |
# |------|------|
# | `RECOMMENDED_PROMPT_PREFIX` | 프롬프트 앞에 직접 붙여 사용하는 문자열 상수 |
# | `prompt_with_handoff_instructions(prompt)` | 기존 프롬프트에 권장 내용을 자동으로 추가하는 함수 |
#
# ```
# # 시스템 컨텍스트
# 당신은 Agents SDK라는 멀티 에이전트 시스템의 일부입니다.
# 이 시스템은 에이전트 간의 조율과 실행을 쉽게 만들기 위해 설계되었습니다.
#
# Agents SDK는 두 가지 핵심 추상화를 사용합니다: **에이전트(Agents)** 와 **핸드오프(Handoffs)**.
#
# 에이전트는 지시사항(instructions)과 도구(tools)를 포함하며,
# 필요한 경우 대화를 다른 에이전트에게 넘길 수 있습니다.
#
# 핸드오프는 일반적으로 `transfer_to_<에이전트명>` 형태로 명명된
# 핸드오프 함수를 호출함으로써 이루어집니다.
#
# 에이전트 간의 전환은 백그라운드에서 원활하게 처리되므로,
# 사용자와의 대화에서 이러한 전환을 언급하거나 드러내지 마세요.
# ```

# %%
from agents import Agent, Runner, handoff
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX, prompt_with_handoff_instructions

RECOMMENDED_PROMPT_PREFIX

# %%
prompt_with_handoff_instructions("당신은 환불 처리 전문가입니다.")

# %%
# 청구/결제 전문 에이전트
billing_agent = Agent(
    name="Billing agent",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
당신은 청구 및 결제 관련 질문을 처리하는 전문가입니다.""",
    model=Model
)

# 환불 전문 에이전트
refund_agent = Agent(
    name="Refund agent",
    instructions=prompt_with_handoff_instructions("당신은 환불 처리 전문가입니다."),
    model=Model
)

# 사용자 요청을 분석하여 적절한 전문 에이전트로 라우팅하는 triage 에이전트
triage_agent = Agent(
    name="Triage agent",
    instructions=prompt_with_handoff_instructions(
        "사용자 요청을 분석하여 청구 질문은 Billing agent에게, 환불 요청은 Refund agent에게 넘겨주세요."
    ),
    model=Model,
    handoffs=[billing_agent, refund_agent],  # 핸드오프 가능한 에이전트 목록
)

# triage_agent 실행 → 청구 관련 질문이므로 billing_agent로 핸드오프 예상
result = await Runner.run(triage_agent, input="지난달 청구 금액이 잘못된 것 같습니다.")
print(result.final_output)

# %% [markdown]
# ## 6. 종합 예제 - 고객 지원 시스템
#
# 여러 핸드오프 기능을 조합한 고객 지원 시스템입니다.
#
# ```
# 사용자
#   └─→ Triage Agent (분류)
#         ├─→ Order Agent    (주문 조회)
#         ├─→ Refund Agent   (환불 처리, on_handoff + input_type)
#         └─→ FAQ Agent      (일반 문의, input_filter 적용)
# ```

# %%
from pydantic import BaseModel
from agents import Agent, Runner, handoff, RunContextWrapper  #에이전트 실행 중 공유 상태(Context)에 접근하는 래퍼(wrapper)
from agents.extensions import handoff_filters
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions

# 환불 요청 데이터 구조 정의 (LLM이 핸드오프 시 채워서 전달)
class RefundRequest(BaseModel):
    order_id: str  # 주문번호
    reason: str    # 환불 사유

# 환불 핸드오프 발생 시 실행되는 콜백 함수 (로깅 용도)
async def on_refund_handoff(ctx: RunContextWrapper[None], input_data: RefundRequest):
    print(f"[환불 요청 접수] 주문번호: {input_data.order_id}, 사유: {input_data.reason}")

# 주문/배송 전문 에이전트
order_agent = Agent(
    name="Order Agent",
    instructions=prompt_with_handoff_instructions(
        "주문 상태 및 배송 관련 문의를 처리합니다. 주문번호를 확인하고 현황을 안내하세요."
    ),
    model=Model
)

# 환불 전문 에이전트
refund_agent = Agent(
    name="Refund Agent",
    instructions=prompt_with_handoff_instructions(
        "환불 요청을 처리합니다. 고객에게 환불 절차와 소요 시간을 안내하세요."
    ),
    model=Model
)

# FAQ 전문 에이전트
faq_agent = Agent(
    name="FAQ Agent",
    instructions=prompt_with_handoff_instructions(
        "자주 묻는 질문에 답변합니다. 배송, 반품 정책, 결제 방법 등 일반적인 문의를 처리하세요."
    ),
    model=Model
)

# 고객 문의를 분류하여 적절한 전문 에이전트로 라우팅하는 triage 에이전트
triage_agent = Agent(
    name="Triage Agent",
    instructions=prompt_with_handoff_instructions(
        "고객 문의를 분류하세요:\n"
        "- 주문/배송 조회 → Order Agent\n"
        "- 환불 요청 → Refund Agent (주문번호와 사유 필요)\n"
        "- 일반 문의/FAQ → FAQ Agent"
    ),
    model=Model,
    handoffs=[
        order_agent,   # 주문 에이전트: 기본 핸드오프 (옵션 없음)
        handoff(
            refund_agent,   # 환불 에이전트: 핸드오프 시 on_refund_handoff 콜백 실행
            on_handoff=on_refund_handoff,  # 핸드오프 발생 시 로깅
            input_type=RefundRequest,       # LLM이 채워야 할 구조화된 입력
        ),
        handoff(
            faq_agent,   # FAQ 에이전트: 도구 호출 기록을 제거하고 깔끔한 대화만 전달
            input_filter=handoff_filters.remove_all_tools,  # 도구 메시지 제거
        ),
    ],
)

# %%
print("=== 테스트 1: 주문 조회 ===")
result = await Runner.run(triage_agent, "주문번호 ORD-1234의 배송 현황을 알고 싶습니다.")
print(result.final_output)

# %%
print("=== 테스트 2: 환불 요청 ===")
result = await Runner.run(triage_agent, "주문번호 ORD-5678 상품을 환불하고 싶습니다. 사이즈가 맞지 않아서요.")
print(result.final_output)

# %%
print("=== 테스트 3: 일반 문의 ===")
result = await Runner.run(triage_agent, "반품 정책이 어떻게 되나요?")
print(result.final_output)

# %% [markdown]
# ### 실습 문제
#
# 아래 요구사항에 맞는 **여행 예약 지원 시스템**을 구현하세요.
#
# **에이전트 구성:**
#
# 1. **Triage Agent**: 사용자 요청을 분류하여 적절한 에이전트에 핸드오프
# 2. **Flight Agent**: 항공편 예약 및 조회 처리
# 3. **Hotel Agent**: 호텔 예약 및 조회 처리
# 4. **Cancellation Agent**: 예약 취소 처리 (`on_handoff` 콜백으로 취소 정보 로깅)
#
# **요구사항:**
# - `Cancellation Agent`로 핸드오프 시 `input_type`으로 예약 번호(`booking_id: str`)와 취소 사유(`reason: str`)를 전달받아 출력
# - 모든 에이전트에 `prompt_with_handoff_instructions` 적용
# - `Flight Agent`, `Hotel Agent`는 `handoff()` 함수 형태로 지정
#
# **테스트 입력:**
# - `"서울-제주 항공편을 예약하고 싶습니다."` → Flight Agent 응답
# - `"제주도 호텔을 3박 예약하려고 합니다."` → Hotel Agent 응답
# - `"예약번호 BK-999 항공편을 취소하고 싶어요. 일정이 바뀌어서요."` → Cancellation Agent 핸드오프 + 콜백 출력

# %%
