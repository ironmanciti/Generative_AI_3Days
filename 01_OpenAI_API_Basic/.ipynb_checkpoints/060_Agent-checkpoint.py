#---------------------------------------------------------
# LangGraph ReAct Agent 기반 Chatbot 구현 (Streamlit 표준 방식)
#---------------------------------------------------------
from dotenv import load_dotenv
_ = load_dotenv()

import streamlit as st
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from streamlit_chat import message
from langchain_tavily import TavilySearch

# LLM 및 도구 초기화 (init_chat_model 방식)
llm = init_chat_model("gpt-5-nano", model_provider="openai")
search_tool = TavilySearch(max_results=2)
tools = [search_tool]

# ---------------------------------------------------------------------------------
# 페이지 설정
# ---------------------------------------------------------------------------------
st.set_page_config(page_title="ReAct Agent Chatbot", page_icon=":robot_face:")
st.markdown("<h1 style='text-align: center;'>ReAct Agent 챗봇</h1>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------------
# 사이드바 버튼 설정
# ---------------------------------------------------------------------------------
st.sidebar.title("😎")
refresh_button = st.sidebar.button("대화 내용 초기화")
summaries_button = st.sidebar.button("대화 내용 요약")

# ---------------------------------------------------------------------------------
# LangGraph ReAct Agent 초기화 (Streamlit 표준 방식)
# ---------------------------------------------------------------------------------
if "agent" not in st.session_state:
    # checkpointer 없이 agent 생성 (Streamlit session_state 사용)
    agent = create_agent(llm, tools)
    st.session_state.agent = agent
    st.session_state.messages = [SystemMessage(content="당신은 유용한 도우미입니다.")]

# ---------------------------------------------------------------------------------
# 사이드바 버튼 동작 정의
# ---------------------------------------------------------------------------------
if refresh_button:
    st.session_state.messages = [SystemMessage(content="당신은 유용한 도우미입니다.")]

if summaries_button:
    conversation_text = []
    # 메시지 역할별로 텍스트로 변환
    for msg in st.session_state.messages:
        if isinstance(msg, SystemMessage):
            role = "System"
        elif isinstance(msg, HumanMessage):
            role = "User"
        elif isinstance(msg, AIMessage):
            role = "AI"
        else:
            role = "Unknown"
        conversation_text.append(f"{role}: {msg.content}")
    joined_conversation = "\n".join(conversation_text)
    # 요약 프롬프트 생성
    prompt_content = f"""다음 대화를 요약해주세요:\n{joined_conversation}\n--- \n요약:\n"""
    # LLM에게 요약 요청
    summary_response = llm.invoke([HumanMessage(content=prompt_content)])
    summary_text = summary_response.content
    # 사이드바에 요약 결과 표시
    st.sidebar.write("**대화 요약:**")
    st.sidebar.write(summary_text)

# ---------------------------------------------------------------------------------
# 메인 영역: 입력 폼 및 에이전트 호출
# ---------------------------------------------------------------------------------
with st.form(key='my_form', clear_on_submit=True):
    user_input = st.text_area("질문을 입력하세요:", key='input', height=100)
    submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        # 사용자 메시지 추가
        st.session_state.messages.append(HumanMessage(content=user_input))
        
        # ReAct Agent 실행 (Streamlit 표준 방식)
        try:
            # agent 실행 (stream 모드 없이)
            response = st.session_state.agent.invoke({"messages": st.session_state.messages})
            ai_msg = response["messages"][-1]  # 마지막 메시지 추출
            st.session_state.messages.append(AIMessage(content=ai_msg.content))
        except Exception as e:
            error_msg = f"에러가 발생했습니다: {str(e)}"
            st.session_state.messages.append(AIMessage(content=error_msg))

# ---------------------------------------------------------------------------------
# 마지막 AIMessage 폼 바로 아래에 표시
# ---------------------------------------------------------------------------------
if st.session_state.messages:
    last_msg = st.session_state.messages[-1]
    if isinstance(last_msg, AIMessage):
        st.text(last_msg.content)

# ---------------------------------------------------------------------------------
# 이전 대화 이력 표시
# ---------------------------------------------------------------------------------
st.subheader("이전 대화 이력")
# 대화 이력을 순서대로 표시
for idx, msg in enumerate(st.session_state.messages):
    if isinstance(msg, HumanMessage):
        message(msg.content, is_user=True, key=str(idx) + "_user")
    elif isinstance(msg, AIMessage):
        message(msg.content, is_user=False, key=str(idx) + "_ai") 