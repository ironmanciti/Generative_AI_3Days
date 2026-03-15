# tool_mcp.py
import asyncio
import requests
from dotenv import load_dotenv

from agents import Agent, Runner, function_tool
from agents.mcp import MCPServerStdio

load_dotenv()

@function_tool
def get_weather(위도: float, 경도: float) -> str:
    """주어진 위도/경도 좌표의 현재 기온(°C)을 문자열로 반환"""
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
            "args": ["-y", "@playwright/mcp@latest"],
        },
    ) as playwright_mcp_server:

        agent = Agent(
            name="여행 에이전트",
            model="gpt-5-mini",  
            instructions=(
                "당신은 훌륭한 여행 에이전트입니다. "
                "여행 일정을 짤 때 웹검색(Playwright MCP)도 활용하고, "
                "가능하면 출처(URL)도 같이 표시해 주세요. "
                "날씨가 필요하면 get_weather(위도,경도) 도구를 사용하세요. "
            ),
            tools=[get_weather],
            mcp_servers=[playwright_mcp_server],
        )

        messages = []
        print("\n 여행 에이전트와 대화를 시작합니다. 종료하려면 'exit' 입력.\n")

        while True:
            try:
                user_input = input("\n사용자: ").strip()
                if user_input.lower() == "exit":
                    print("안녕히 가세요!")
                    break

                messages.append({"role": "user", "content": user_input})
                print("\n여행 에이전트: ", end="", flush=True)

                response = await Runner.run(agent, input=messages)
                # RunResult 객체에서 텍스트 추출
                full = None
                
                # 1. final_output 속성 확인 (가장 우선)
                if hasattr(response, 'final_output') and response.final_output:
                    if isinstance(response.final_output, str):
                        full = response.final_output
                    elif hasattr(response.final_output, 'content'):
                        full = response.final_output.content
                    elif hasattr(response.final_output, 'text'):
                        full = response.final_output.text
                
                # 2. final_output_as 메서드 사용
                if not full and hasattr(response, 'final_output_as'):
                    try:
                        full = response.final_output_as(str)
                    except:
                        pass
                
                # 3. new_items에서 텍스트 추출
                if not full and hasattr(response, 'new_items') and response.new_items:
                    text_items = []
                    for item in response.new_items:
                        # item이 텍스트 타입인 경우
                        if hasattr(item, 'type') and item.type == 'text':
                            if hasattr(item, 'content'):
                                text_items.append(str(item.content))
                            elif hasattr(item, 'text'):
                                text_items.append(str(item.text))
                        # item이 문자열인 경우
                        elif isinstance(item, str):
                            text_items.append(item)
                        # item에 content 속성이 있는 경우
                        elif hasattr(item, 'content') and item.content:
                            text_items.append(str(item.content))
                        # item에 text 속성이 있는 경우
                        elif hasattr(item, 'text') and item.text:
                            text_items.append(str(item.text))
                    
                    if text_items:
                        full = '\n'.join(text_items) if len(text_items) > 1 else text_items[0]
                
                # 4. raw_responses에서 텍스트 추출
                if not full and hasattr(response, 'raw_responses') and response.raw_responses:
                    for raw_resp in response.raw_responses:
                        if hasattr(raw_resp, 'output_text') and raw_resp.output_text:
                            full = raw_resp.output_text
                            break
                        elif hasattr(raw_resp, 'text') and raw_resp.text:
                            full = raw_resp.text
                            break
                
                # 5. 최종 폴백: 디버깅 정보 출력
                if not full:
                    print("\n⚠️ 응답을 추출할 수 없습니다.")
                    # 디버깅을 위해 일부 정보 출력
                    if hasattr(response, 'new_items'):
                        print(f"new_items 수: {len(response.new_items) if response.new_items else 0}")
                    if hasattr(response, 'raw_responses'):
                        print(f"raw_responses 수: {len(response.raw_responses) if response.raw_responses else 0}")
                    if hasattr(response, 'final_output'):
                        print(f"final_output 타입: {type(response.final_output)}")
                    continue
                
                print(full)

                messages.append({"role": "assistant", "content": full})
                
            except KeyboardInterrupt:
                print("\n\n 종료합니다.")
                break
            except Exception as e:
                print(f"\n 오류 발생: {e}")
                continue

if __name__ == "__main__":
    asyncio.run(main())