import os
import logging
import requests
import openai
from dotenv import load_dotenv
from pydantic import BaseModel
from agents import Agent, Runner, function_tool, ModelSettings
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

# 에이전트 이름의 공백 때문에 핸드오프 도구 이름이 자동 변환된다는
# SDK 경고(warning)가 매 호출마다 출력되므로, ERROR 이상만 표시합니다.
logging.getLogger("openai.agents").setLevel(logging.ERROR)

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

# 사용할 모델 설정
MODEL = "gpt-5-mini"

# --- 도구(Tools) 정의 영역 ---

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

# 1. 먼저 하위 에이전트들을 정의하고, 아래에서 순환 참조(핸드오프 경로)를 연결합니다.
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

# 2. 모든 하위 에이전트들이 다시 상담원(Triage)에게 돌아올 수 있도록 경로를 추가합니다.
weather_agent.handoffs = [triage_agent]
math_agent.handoffs = [triage_agent]
search_agent.handoffs = [triage_agent]

# --- 챗봇 실행 영역 ---

def run_chatbot():
    print("========================================")
    print("   AI 에이전트 챗봇 (종료하려면 'exit' 입력)")
    print("========================================")

    last_response_id = None
    current_agent = triage_agent

    while True:
        user_input = input("\n나: ")

        if user_input.lower() in ['exit', 'quit', '종료', 'q']:
            print("챗봇을 종료합니다. 좋은 하루 되세요!")
            break

        try:
            result = Runner.run_sync(
                starting_agent=current_agent, 
                input=user_input, 
                previous_response_id=last_response_id
            )

            print(f"AI: {result.final_output}")

            last_response_id = result.last_response_id
            current_agent = result.last_agent

        except Exception as e:
            print(f"오류가 발생했습니다: {e}")

if __name__ == "__main__":
    run_chatbot()
