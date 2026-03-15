# chatbot_travel_agent.py
# -------------------------------------------------------------
# 콘솔 기반 Chatbot: 여행 에이전트
#  - get_weather(@function_tool) + Playwright MCP(stdio)
#  - 스트리밍 출력 (ChatGPT 스타일)
#  - 'exit' 입력 시 종료
# -------------------------------------------------------------

import asyncio
import requests
from dotenv import load_dotenv

from openai.types.responses import ResponseTextDeltaEvent
from agents import Agent, Runner, function_tool, WebSearchTool
from agents.mcp import MCPServerStdio

load_dotenv()

# 날씨 조회 툴 (Open-Meteo API 사용)
@function_tool
def get_weather(위도: float, 경도: float) -> str:
    """
    주어진 위도/경도 좌표의 현재 기온(°C)을 문자열로 반환
    """
    latitude = 위도
    longitude = 경도
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}&longitude={longitude}&current=temperature_2m"
    )
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    temp = data.get("current", {}).get("temperature_2m")
    return f"{temp}°C" if temp is not None else "알 수 없음"

async def main():
    
    async with MCPServerStdio(
        name="Playwright MCP",
        params={
            "command": "npx",
            "args": ["-y", "@playwright/mcp@latest"],  # 비대화형 설치 플래그
        },
    ) as mcp_server:

        agent = Agent(
            name="여행 에이전트",
            model="gpt-4.1-mini",
            instructions=(
                "당신은 훌륭한 여행 에이전트입니다. "
                "여행 일정을 짤 때 웹검색(WebSearch/MCP)도 활용하고, "
                "가능하면 출처(URL)도 같이 표시해 주세요. "
                "날씨가 필요하면 get_weather(위도,경도) 도구를 사용하세요."
            ),
            tools=[get_weather, WebSearchTool()],  # 로컬 툴 + 내장 웹검색 툴
            mcp_servers=[mcp_server],              # Playwright MCP (브라우징/스크린샷 등)
        )

        messages = []
        print("여행 에이전트와 대화를 시작합니다. 종료하려면 'exit' 입력.\n")

        while True:
            user_input = input("\n사용자: ").strip()
            if user_input.lower() == "exit":
                print("Bye")
                break

            # 대화 문맥 누적
            messages.append({"role": "user", "content": user_input})

            print("\n여행 에이전트: ", end="", flush=True)
            full = ""

            # 스트리밍 실행 (실시간 토큰 델타를 받아 출력)
            streamed = Runner.run_streamed(agent, input=messages)
            async for event in streamed.stream_events():
                if event.type == "raw_response_event" and isinstance(
                    event.data, ResponseTextDeltaEvent
                ):
                    delta = event.data.delta or ""
                    print(delta, end="", flush=True)
                    full += delta

            # 최종 응답을 문맥에 추가
            messages.append({"role": "assistant", "content": full})

if __name__ == "__main__":
    asyncio.run(main())