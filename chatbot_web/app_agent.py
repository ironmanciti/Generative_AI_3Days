"""
AI 에이전트 챗봇 - 웹 UI 백엔드 (Flask + OpenAI Agents SDK)

701_Agent_Chatbot.py(콘솔 버전)를 웹 서비스로 옮긴 버전입니다.
app.py(단순 API 호출 버전)와 비교하며 학습하세요.

실행 방법:
    pip install flask openai openai-agents python-dotenv requests
    python app_agent.py
    브라우저에서 http://127.0.0.1:5001 접속

핵심:
    - 콘솔 버전에서는 while 루프의 지역 변수로 들고 있던 두 가지 상태
      (last_response_id, current_agent)를, 웹에서는 서버가 기억하지 않고
      매 요청마다 JSON 으로 클라이언트와 주고받습니다. (무상태 서버)
    - Agent 객체 자체는 JSON 으로 직렬화할 수 없으므로 '이름'(문자열)만
      주고받고, 서버에서 이름 -> Agent 매핑(AGENTS)으로 복원합니다.
"""
import os
import asyncio
import logging
import threading
import requests
import openai
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
from agents import Agent, Runner, function_tool, ModelSettings
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

# 에이전트 이름의 공백 때문에 핸드오프 도구 이름이 자동 변환된다는
# SDK 경고(warning)가 매 호출마다 출력되므로, ERROR 이상만 표시합니다.
logging.getLogger("openai.agents").setLevel(logging.ERROR)

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

# 사용할 모델 설정
MODEL = "gpt-5-mini"

# --- 도구(Tools) 정의 영역 (701_Agent_Chatbot.py 와 동일) ---

@function_tool
def multiply(x: float, y: float) -> float:
    """두 숫자를 곱한 결과를 반환합니다."""
    print(f"\n[시스템] 곱하기 도구 실행: {x} * {y}")
    return x * y

@function_tool
def get_weather(latitude: float, longitude: float) -> str:
    """위도와 경도를 입력받아 현재 온도를 가져옵니다."""
    print(f"\n[시스템] 날씨 도구 실행: 위도 {latitude}, 경도 {longitude}")
    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m"
    response = requests.get(url)
    data = response.json()
    temp = data['current']['temperature_2m']
    return f"{temp}°C"

@function_tool
def web_search_tool(query: str) -> str:
    """인터넷에서 최신 정보나 실시간 데이터를 검색합니다."""
    print(f"\n[시스템] 웹 검색 도구 실행: {query}")
    client = openai.OpenAI()
    response = client.responses.create(
        model=MODEL,
        tools=[{"type": "web_search"}],
        input=query
    )
    return response.output_text

# --- 에이전트(Agents) 정의 영역 ---
# RECOMMENDED_PROMPT_PREFIX: SDK 가 권장하는 핸드오프 안내문.
# 이것이 없으면 모델이 가끔 "전달했습니다" 같은 안내문만 출력하고
# 실제 답변 없이 멈추는 경우가 생깁니다.

# 전문 에이전트는 첫 응답에서 반드시 도구(또는 핸드오프)를 호출하도록 강제합니다.
# 모델이 "잠시만요" 같은 멘트만 하고 도구 호출 없이 턴을 끝내는 것을 막습니다.
# (도구를 한 번 호출하면 SDK 가 자동으로 'auto' 로 되돌려 무한 반복을 방지합니다.)
SPECIALIST_SETTINGS = ModelSettings(tool_choice="required")

weather_agent = Agent(
    name="Weather Agent",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
기상 전문가입니다. 날씨 정보가 필요한 질문(날씨에 따른 옷차림, 방문지 추천 포함)은
get_weather 도구로 현재 온도를 확인한 뒤 직접 완전한 답변을 작성하세요.
날씨와 무관한 질문만 Triage Agent에게 핸드오프하세요.
핸드오프 사실을 사용자에게 언급하지 마세요.""",
    model=MODEL,
    model_settings=SPECIALIST_SETTINGS,
    tools=[get_weather]
)

math_agent = Agent(
    name="Math Agent",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
수학 선생님입니다. 계산이 필요하면 도구를 사용해 직접 완전한 답변을 작성하세요.
수학/계산과 무관한 질문만 Triage Agent에게 핸드오프하세요.
핸드오프 사실을 사용자에게 언급하지 마세요.""",
    model=MODEL,
    model_settings=SPECIALIST_SETTINGS,
    tools=[multiply]
)

search_agent = Agent(
    name="Search Agent",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
정보 검색 전문가입니다. 최신 정보가 필요하면 도구로 검색해 직접 완전한 답변을 작성하세요.
검색과 무관한 질문만 Triage Agent에게 핸드오프하세요.
핸드오프 사실을 사용자에게 언급하지 마세요.""",
    model=MODEL,
    model_settings=SPECIALIST_SETTINGS,
    tools=[web_search_tool]
)

triage_agent = Agent(
    name="Triage Agent",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
사용자의 질문에 따라 적절한 에이전트에게 핸드오프하세요.
1. 날씨 관련(옷차림, 날씨 기반 추천 포함): Weather Agent
2. 수학/계산: Math Agent
3. 최신 정보 검색: Search Agent
일상 대화나 이름 기억 등은 직접 답변하세요.
핸드오프할 때는 '전달했습니다' 같은 안내문을 출력하지 말고 조용히 넘기세요.""",
    model=MODEL,
    handoffs=[weather_agent, math_agent, search_agent]
)

# 모든 하위 에이전트들이 다시 상담원(Triage)에게 돌아올 수 있도록 경로를 추가합니다.
weather_agent.handoffs = [triage_agent]
math_agent.handoffs = [triage_agent]
search_agent.handoffs = [triage_agent]

# 이름(문자열) -> Agent 객체 매핑.
# 클라이언트가 보낸 agent_name 으로 직전 대화의 담당 에이전트를 복원할 때 사용합니다.
AGENTS = {
    agent.name: agent
    for agent in [triage_agent, weather_agent, math_agent, search_agent]
}

# --- 에이전트 실행 전용 이벤트 루프 ---
# Flask 개발 서버는 요청마다 '새 스레드'에서 핸들러를 실행합니다.
# 그런데 Agents SDK 의 비동기 클라이언트는 처음 실행된 이벤트 루프에 묶이므로,
# 요청마다 Runner.run_sync 를 부르면 두 번째 요청부터 응답이 멈춥니다.
# 해결: 백그라운드 스레드에서 이벤트 루프 하나를 상시 실행해 두고,
# 모든 에이전트 실행을 이 루프 하나로 모아서 처리합니다.

agent_loop = asyncio.new_event_loop()
threading.Thread(target=agent_loop.run_forever, daemon=True).start()


def run_agent(starting_agent, message, prev_id):
    """어느 요청 스레드에서 호출해도 항상 같은 루프에서 에이전트를 실행합니다."""
    future = asyncio.run_coroutine_threadsafe(
        Runner.run(
            starting_agent=starting_agent,
            input=message,
            previous_response_id=prev_id,
        ),
        agent_loop,
    )
    return future.result(timeout=120)  # 120초 안에 답이 없으면 오류 처리


# --- 웹 서버 영역 ---

app = Flask(__name__)


@app.route("/")
def index():
    # 같은 폴더의 index_agent.html 을 그대로 전달합니다.
    return send_from_directory(os.path.dirname(__file__), "index_agent.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = data.get("message", "")
    prev_id = data.get("response_id")    # 이전 응답 id (없으면 첫 대화)
    agent_name = data.get("agent_name")  # 직전 대화의 담당 에이전트 이름

    # 콘솔 버전의 current_agent = result.last_agent 에 해당하는 부분.
    # 이름이 없거나 모르는 이름이면 상담원(Triage)부터 시작합니다.
    current_agent = AGENTS.get(agent_name, triage_agent)

    try:
        result = run_agent(current_agent, message, prev_id)
    except Exception as e:
        return jsonify({"error": f"오류가 발생했습니다: {e}"}), 500

    # 답변과 함께 두 가지 상태(response_id, agent_name)를 돌려주면,
    # 클라이언트가 다음 요청에 그대로 실어 보내 대화가 이어집니다.
    return jsonify({
        "reply": result.final_output,
        "response_id": result.last_response_id,
        "agent_name": result.last_agent.name,
    })


if __name__ == "__main__":
    # app.py(5000번)와 동시에 띄워 비교할 수 있도록 5001번 포트를 사용합니다.
    app.run(port=5001, debug=True)
