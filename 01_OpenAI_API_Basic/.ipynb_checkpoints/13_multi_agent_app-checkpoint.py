"""
멀티에이전트 오케스트레이션 채팅 앱 (OpenAI Agents SDK + Streamlit)
- 12_multi_agent_orchestration 노트북의 4가지 패턴을 Streamlit UI로 제공합니다.
  1) 순차 체이닝: 작성 → 편집 → 번역
  2) 병렬 실행: 감성 분석 + 키워드 추출 + 요약 동시 실행
  3) 분류 → 라우팅: 고객 문의를 분류 후 전문 에이전트에 전달
  4) 종합 패턴: 병렬 분석 → 종합 보고서
"""

import os
import asyncio
import streamlit as st
from dotenv import load_dotenv

import openai
from agents import Agent, Runner, trace
from pydantic import BaseModel
from streamlit_chat import message

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-5-mini"
client = openai.OpenAI()

# =============================================================================
# 에이전트 정의
# =============================================================================

# ── 1. 순차 체이닝 에이전트 ──
writer_agent = Agent(
    name="작성기",
    instructions="""당신은 기술 블로그 작성자입니다.
주어진 주제에 대해 3~4문장의 짧은 기술 소개글을 작성하세요.
전문 용어를 적절히 사용하되 이해하기 쉽게 작성하세요.""",
    model=OPENAI_MODEL,
)

editor_agent = Agent(
    name="편집기",
    instructions="""당신은 전문 편집자입니다.
주어진 글을 다음 기준으로 수정하세요:
- 문법 오류 수정
- 문장 가독성 향상
- 불필요한 표현 제거
수정된 글만 출력하세요.""",
    model=OPENAI_MODEL,
)

translator_agent = Agent(
    name="번역기",
    instructions="""당신은 한영 번역가입니다.
주어진 한국어 글을 자연스러운 영어로 번역하세요.
번역된 영어만 출력하세요.""",
    model=OPENAI_MODEL,
)

# ── 2. 병렬 실행 에이전트 ──
sentiment_agent = Agent(
    name="감성 분석기",
    instructions="""주어진 텍스트의 감성을 분석하세요.
결과 형식: "감성: [긍정/부정/중립], 신뢰도: [높음/보통/낮음], 이유: [간단한 설명]" """,
    model=OPENAI_MODEL,
)

keyword_agent = Agent(
    name="키워드 추출기",
    instructions="""주어진 텍스트에서 핵심 키워드 3~5개를 추출하세요.
결과 형식: "키워드: [키워드1], [키워드2], [키워드3], ..." """,
    model=OPENAI_MODEL,
)

summary_agent = Agent(
    name="요약기",
    instructions="""주어진 텍스트를 한 문장으로 요약하세요.
결과 형식: "요약: [한 문장 요약]" """,
    model=OPENAI_MODEL,
)

# ── 3. 분류 → 라우팅 에이전트 ──
class RequestClassification(BaseModel):
    category: str   # "기술지원", "결제문의", "일반문의"
    confidence: float  # 0.0 ~ 1.0
    reason: str

classifier_agent = Agent(
    name="요청 분류기",
    instructions="""고객의 요청을 다음 카테고리 중 하나로 분류하세요:
- "기술지원": 소프트웨어 오류, 설치 문제, 기술적 질문
- "결제문의": 요금, 환불, 결제 수단, 구독 관련
- "일반문의": 영업시간, 위치, 서비스 소개 등 일반 질문

category에 정확한 카테고리명을 넣고, confidence에 확신도(0~1)를 넣으세요.""",
    model=OPENAI_MODEL,
    output_type=RequestClassification,
)

tech_support_agent = Agent(
    name="기술지원 전문가",
    instructions="당신은 기술지원 전문가입니다. 소프트웨어 문제를 해결하고 기술적 도움을 제공합니다.",
    model=OPENAI_MODEL,
)

billing_agent = Agent(
    name="결제 전문가",
    instructions="당신은 결제 전문가입니다. 요금, 환불, 구독 관련 질문에 답변합니다.",
    model=OPENAI_MODEL,
)

general_agent = Agent(
    name="일반 상담사",
    instructions="당신은 일반 상담사입니다. 서비스에 대한 일반적인 질문에 답변합니다.",
    model=OPENAI_MODEL,
)

agent_map = {
    "기술지원": tech_support_agent,
    "결제문의": billing_agent,
    "일반문의": general_agent,
}

# ── 4. 종합 패턴: 보고서 작성 에이전트 ──
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
    model=OPENAI_MODEL,
)


# =============================================================================
# 실행 함수들
# =============================================================================

def run_sequential(user_input: str) -> str:
    """순차 체이닝: 작성 → 편집 → 번역"""
    steps = []

    with trace("순차 체이닝: 작성 → 편집 → 번역"):
        write_result = Runner.run_sync(writer_agent, user_input)
        steps.append(f"**[Step 1] 초안 (작성기):**\n{write_result.final_output}")

        edit_result = Runner.run_sync(editor_agent, write_result.final_output)
        steps.append(f"**[Step 2] 편집본 (편집기):**\n{edit_result.final_output}")

        translate_result = Runner.run_sync(translator_agent, edit_result.final_output)
        steps.append(f"**[Step 3] 번역본 (번역기):**\n{translate_result.final_output}")

    return "\n\n---\n\n".join(steps)


def run_parallel(user_input: str) -> str:
    """병렬 실행: 감성 분석 + 키워드 추출 + 요약 동시 실행"""

    async def _gather():
        with trace("병렬 분석"):
            return await asyncio.gather(
                Runner.run(sentiment_agent, user_input),
                Runner.run(keyword_agent, user_input),
                Runner.run(summary_agent, user_input),
            )

    s_result, k_result, sm_result = asyncio.run(_gather())

    lines = [
        f"**감성 분석:**\n{s_result.final_output}",
        f"**키워드 추출:**\n{k_result.final_output}",
        f"**요약:**\n{sm_result.final_output}",
    ]
    return "\n\n---\n\n".join(lines)


def run_classification(user_input: str) -> str:
    """분류 → 라우팅: 분류 후 전문 에이전트에 전달"""

    async def _classify_and_route():
        with trace("분류 → 라우팅"):
            classification_result = await Runner.run(classifier_agent, user_input)
            classification = classification_result.final_output

            selected_agent = agent_map.get(classification.category, general_agent)

            response = await Runner.run(selected_agent, user_input)
            return classification, selected_agent.name, response.final_output

    classification, agent_name, answer = asyncio.run(_classify_and_route())

    lines = [
        f"**분류 결과:**\n- 카테고리: {classification.category}\n- 확신도: {classification.confidence:.0%}\n- 이유: {classification.reason}",
        f"**선택된 에이전트:** {agent_name}",
        f"**응답:**\n{answer}",
    ]
    return "\n\n---\n\n".join(lines)


def run_combined(user_input: str) -> str:
    """종합 패턴: 병렬 분석 → 종합 보고서"""

    async def _combined():
        with trace("종합 분석 파이프라인"):
            # Step 1: 병렬 분석
            s_result, k_result, sm_result = await asyncio.gather(
                Runner.run(sentiment_agent, user_input),
                Runner.run(keyword_agent, user_input),
                Runner.run(summary_agent, user_input),
            )

            # Step 2: 종합 보고서 작성
            combined_input = f"""다음 분석 결과를 종합해주세요:

원본 텍스트: {user_input}

분석 결과:
1. {s_result.final_output}
2. {k_result.final_output}
3. {sm_result.final_output}
"""
            report_result = await Runner.run(report_agent, combined_input)
            return s_result, k_result, sm_result, report_result

    s_result, k_result, sm_result, report_result = asyncio.run(_combined())

    lines = [
        "**[Step 1] 병렬 분석 결과:**",
        f"- 감성: {s_result.final_output}",
        f"- 키워드: {k_result.final_output}",
        f"- 요약: {sm_result.final_output}",
        "---",
        f"**[Step 2] 종합 보고서:**\n{report_result.final_output}",
    ]
    return "\n\n".join(lines)


# 패턴 → 실행 함수 매핑
PATTERN_MAP = {
    "순차 체이닝 (작성→편집→번역)": run_sequential,
    "병렬 실행 (감성+키워드+요약)": run_parallel,
    "분류 → 라우팅 (고객 문의)": run_classification,
    "종합 패턴 (병렬 분석→보고서)": run_combined,
}

PATTERN_PLACEHOLDERS = {
    "순차 체이닝 (작성→편집→번역)": "기술 소개글을 작성할 주제를 입력하세요. 예: 에이전트 AI에 대해 소개글을 작성해주세요.",
    "병렬 실행 (감성+키워드+요약)": "분석할 텍스트를 입력하세요.",
    "분류 → 라우팅 (고객 문의)": "고객 문의를 입력하세요. 예: 프로그램이 갑자기 멈추고 에러 메시지가 떠요.",
    "종합 패턴 (병렬 분석→보고서)": "분석할 리뷰/텍스트를 입력하세요.",
}


# =============================================================================
# Streamlit UI
# =============================================================================

st.set_page_config(page_title="멀티에이전트 오케스트레이션", page_icon=":robot_face:")
st.markdown(
    "<h1 style='text-align: center;'>멀티에이전트 오케스트레이션 챗봇</h1>",
    unsafe_allow_html=True,
)

# ── 사이드바 ──
st.sidebar.title("설정")

selected_pattern = st.sidebar.selectbox(
    "오케스트레이션 패턴 선택",
    list(PATTERN_MAP.keys()),
)

st.sidebar.markdown("---")
refresh_button = st.sidebar.button("대화 내용 초기화")
summaries_button = st.sidebar.button("대화 내용 요약")

# 패턴 설명
st.sidebar.markdown("---")
st.sidebar.markdown("### 패턴 설명")
pattern_descriptions = {
    "순차 체이닝 (작성→편집→번역)": "에이전트A의 출력을 에이전트B의 입력으로 전달하는 파이프라인.\n\n`작성기 → 편집기 → 번역기`",
    "병렬 실행 (감성+키워드+요약)": "독립적인 에이전트 3개를 동시에 실행하여 시간 절약.\n\n`감성 분석 + 키워드 추출 + 요약`",
    "분류 → 라우팅 (고객 문의)": "Structured Output으로 입력을 분류한 후, 전문 에이전트에 전달.\n\n`분류기 → 기술지원/결제/일반`",
    "종합 패턴 (병렬 분석→보고서)": "병렬 분석 후 결과를 종합하여 보고서 생성.\n\n`병렬 분석 → 종합 보고서`",
}
st.sidebar.info(pattern_descriptions[selected_pattern])

# ── 세션 초기화 ──
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── 사이드바 버튼 동작 ──
if refresh_button:
    st.session_state.messages = []

if summaries_button:
    if st.session_state.messages:
        conversation_text = []
        for msg in st.session_state.messages:
            role = "User" if msg["role"] == "user" else "AI"
            conversation_text.append(f"{role}: {msg['content']}")
        joined_conversation = "\n".join(conversation_text)
        prompt_content = f"다음 대화를 요약해주세요:\n{joined_conversation}\n---\n요약:\n"
        summary_response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt_content}],
        )
        summary_text = summary_response.choices[0].message.content
        st.sidebar.write("**대화 요약:**")
        st.sidebar.write(summary_text)
    else:
        st.sidebar.write("대화 내용이 없습니다.")

# ── 메인 영역: 입력 폼 및 에이전트 호출 ──
with st.form(key="my_form", clear_on_submit=True):
    user_input = st.text_area(
        "질문을 입력하세요:",
        key="input",
        height=100,
        placeholder=PATTERN_PLACEHOLDERS[selected_pattern],
    )
    submit_button = st.form_submit_button(label="Send")

    if submit_button and user_input:
        st.session_state.messages.append({
            "role": "user",
            "content": f"[{selected_pattern}] {user_input}",
        })

        try:
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")

            run_fn = PATTERN_MAP[selected_pattern]
            reply = run_fn(user_input)
            st.session_state.messages.append({"role": "assistant", "content": reply})
        except Exception as e:
            error_msg = f"에러가 발생했습니다: {str(e)}"
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

# ── 마지막 AIMessage 폼 바로 아래에 표시 ──
if st.session_state.messages:
    last_msg = st.session_state.messages[-1]
    if last_msg["role"] == "assistant":
        st.markdown(last_msg["content"])

# ── 이전 대화 이력 표시 ──
st.subheader("이전 대화 이력")
for idx, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        message(msg["content"], is_user=True, key=str(idx) + "_user")
    elif msg["role"] == "assistant":
        message(msg["content"], is_user=False, key=str(idx) + "_ai")
