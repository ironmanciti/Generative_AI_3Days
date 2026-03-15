import os
import requests
import openai
from dotenv import load_dotenv
from pydantic import BaseModel
from agents import Agent, Runner, function_tool

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

# 1. 먼저 빈 리스트로 에이전트들을 정의하여 순환 참조가 가능하게 합니다.
weather_agent = Agent(
    name="Weather Agent",
    instructions="기상 전문가입니다. 날씨 질문이 아니면 Triage Agent에게 즉시 핸드오프하세요. 날씨 질문이라면 도구를 사용해 답변하세요.",
    model=MODEL,
    tools=[get_weather]
)

math_agent = Agent(
    name="Math Agent",
    instructions="수학 선생님입니다. 수학/계산 질문이 아니면 Triage Agent에게 즉시 핸드오프하세요. 계산이 필요하면 도구를 사용하세요.",
    model=MODEL,
    tools=[multiply]
)

search_agent = Agent(
    name="Search Agent",
    instructions="정보 검색 전문가입니다. 검색이 필요한 질문이 아니면 Triage Agent에게 즉시 핸드오프하세요.",
    model=MODEL,
    tools=[web_search_tool]
)

triage_agent = Agent(
    name="Triage Agent",
    instructions=(
        "사용자의 질문에 따라 적절한 에이전트에게 넘겨주세요.\n"
        "1. 날씨: Weather Agent\n"
        "2. 계산: Math Agent\n"
        "3. 검색: Search Agent\n"
        "일상 대화나 이름 기억 등은 직접 답변하세요."
    ),
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
