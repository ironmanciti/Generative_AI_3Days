"""
여행·지식 베이스 채팅 앱 (OpenAI Agents SDK + Streamlit)
- 09_Chatbot_RAG_Agent 노트북 4단계와 동일한 에이전트: 날씨(get_weather), 웹 검색(WebSearchTool), RAG(FileSearchTool + Vector Store).
- 답변 생성은 OpenAI만 사용합니다. Streamlit으로 대화 화면을 제공하며, 대화 기록은 세션에 유지됩니다.
"""

import os
import requests
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path

import openai
from agents import Agent, Runner, function_tool, WebSearchTool, FileSearchTool
from streamlit_chat import message

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-5-mini"

# ── Vector Store 초기 설정 (앱 시작 시 1회) ──
# 10_Chatbot_RAG_Agent.py와 동일: OpenAI 관리형 Vector Store에 FAQ/KB 업로드
VS_NAME = "notebook-faq-kb"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

client = openai.OpenAI()
existing = [vs for vs in client.vector_stores.list() if vs.name == VS_NAME]

if existing:
    vs_id = existing[0].id
else:
    vs = client.vector_stores.create(name=VS_NAME)
    vs_id = vs.id
    files_to_upload = [
        open(DATA_DIR / "faqs.json", "rb"),
        open(DATA_DIR / "knowledgeBase.json", "rb"),
    ]
    try:
        client.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vs_id, files=files_to_upload,
        )
    finally:
        for f in files_to_upload:
            f.close()


# 노트북과 동일: 날씨 도구 (open-meteo API)
@function_tool
def get_weather(위도: float, 경도: float) -> str:
    """주어진 위도와 경도에서의 현재 기온(섭씨)을 반환합니다."""
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={위도}&longitude={경도}&current=temperature_2m"
    )
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    temp = data.get("current", {}).get("temperature_2m")
    return f"{temp}°C" if temp is not None else "Unknown"


# 에이전트 생성 후 대화 메시지로 실행하고 최종 답변 문자열 반환 
def _run_agent(messages: list) -> str:
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 또는 환경 변수를 설정하세요.")

    # 노트북 4단계와 동일: 여행 + 날씨 + 웹검색 + RAG (FileSearchTool)
    agent = Agent(
        name="여행·지식 베이스 에이전트",
        instructions=(
            "당신은 여행 계획과 회사 FAQ를 도와주는 에이전트입니다. "
            "날씨가 필요하면 get_weather, 최신 정보는 WebSearchTool을 사용하세요. "
            "사용자의 질문이 환불, 계정, 비밀번호, 이메일, 보안, 전화 상담, 주문 등과 "
            "조금이라도 관련되면 반드시 file_search 도구를 먼저 호출하여 지식 베이스를 검색하세요. "
            "file_search 결과를 기반으로만 답하고, 결과가 없으면 '정보를 찾을 수 없습니다'라고 답하세요. "
            "절대로 지식 베이스를 검색하지 않고 추측으로 답하지 마세요. 답은 간결하게 하세요."
        ),
        model=OPENAI_MODEL,
        tools=[get_weather, WebSearchTool(), FileSearchTool(vector_store_ids=[vs_id])],
    )
    result = Runner.run_sync(agent, input=messages)
    return str(result.final_output or "응답이 생성되지 않았습니다.")


# ---------------------------------------------------------------------------------
# 페이지 설정
# ---------------------------------------------------------------------------------
st.set_page_config(page_title="OpenAI Agent AI 채팅", page_icon=":robot_face:")
st.markdown("<h1 style='text-align: center;'>여행·지식 베이스 에이전트 챗봇</h1>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------------
# 사이드바 버튼 설정
# ---------------------------------------------------------------------------------
st.sidebar.title("😎")
refresh_button = st.sidebar.button("대화 내용 초기화")
summaries_button = st.sidebar.button("대화 내용 요약")

# ---------------------------------------------------------------------------------
# 세션 초기화
# ---------------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------------------------------
# 사이드바 버튼 동작 정의
# ---------------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------------
# 메인 영역: 입력 폼 및 에이전트 호출
# ---------------------------------------------------------------------------------
with st.form(key='my_form', clear_on_submit=True):
    user_input = st.text_area("질문을 입력하세요:", key='input', height=100)
    submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        try:
            messages = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ]
            reply = _run_agent(messages)
            st.session_state.messages.append({"role": "assistant", "content": reply})
        except Exception as e:
            error_msg = f"에러가 발생했습니다: {str(e)}"
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

# ---------------------------------------------------------------------------------
# 마지막 AIMessage 폼 바로 아래에 표시
# ---------------------------------------------------------------------------------
if st.session_state.messages:
    last_msg = st.session_state.messages[-1]
    if last_msg["role"] == "assistant":
        st.text(last_msg["content"])

# ---------------------------------------------------------------------------------
# 이전 대화 이력 표시
# ---------------------------------------------------------------------------------
st.subheader("이전 대화 이력")
for idx, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        message(msg["content"], is_user=True, key=str(idx) + "_user")
    elif msg["role"] == "assistant":
        message(msg["content"], is_user=False, key=str(idx) + "_ai")
