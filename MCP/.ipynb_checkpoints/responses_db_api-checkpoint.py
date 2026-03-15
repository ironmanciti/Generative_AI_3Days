from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

# --------------------------------------------------------
# 1️⃣ OpenAI 클라이언트 초기화
# --------------------------------------------------------
client = OpenAI()
model = "gpt-5-mini"

# --------------------------------------------------------
# 2️⃣ MCP 서버 URL 설정
# --------------------------------------------------------
# 옵션 1: ngrok을 통한 공개 URL (클라우드에서 접근 가능)
# ngrok 실행: ngrok http 3001
ngrok_url = "https://lurchingly-unenticed-yuette.ngrok-free.dev"

# 옵션 2: 로컬 HTTP 서버 (로컬 테스트용, 클라우드에서는 작동 안 함)
local_url = "http://localhost:3001"

# MCP 서버 엔드포인트 (로컬 테스트 시 local_url, 클라우드 사용 시 ngrok_url)
mcp_server_url = f"{ngrok_url}/mcp" 

# --------------------------------------------------------
# 3️⃣ 대화 히스토리 관리
# --------------------------------------------------------
previous_response_id = None

# --------------------------------------------------------
# 4️⃣ MCP 승인 요청 처리 함수
# --------------------------------------------------------
def handle_approval_request(response):
    """
    응답에서 mcp_approval_request를 찾아 승인 처리
    """
    approval_requests = [o for o in response.output if o.type == "mcp_approval_request"]
    
    if not approval_requests:
        return None
    
    # 모든 승인 요청을 자동으로 승인 (실습용)
    # 실제 환경에서는 사용자에게 승인을 요청할 수 있음
    approval_responses = []
    for approval in approval_requests:
        approval_responses.append({
            "type": "mcp_approval_response",
            "approval_request_id": approval.id,
            "approve": True,  # 자동 승인
        })
    
    return approval_responses

# --------------------------------------------------------
# 5️⃣ 메인 대화 루프
# --------------------------------------------------------
def main():
    global previous_response_id
    
    print("===== Chinook 데이터베이스 대화형 챗봇 시작 =====")
    print("Responses API를 사용하여 MCP 서버와 통신합니다.")
    print(f"MCP 서버 URL: {mcp_server_url}")
    print("종료하려면 'quit', 'exit', 또는 '종료'를 입력하세요.\n")
    
    while True:
        try:
            # 사용자 입력 받기
            user_input = input("질문을 입력하세요: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ["quit", "exit", "종료"]:
                print("챗봇을 종료합니다.")
                break
            
            # 입력 구성
            # 이전 응답이 있으면 previous_response_id 사용
            # 없으면 일반 input 사용
            if previous_response_id:
                # 이전 응답 ID를 사용하여 대화 히스토리 유지
                response = client.responses.create(
                    model=model,
                    input=user_input,  # 단순 문자열도 가능
                    tools=[
                        {
                            "type": "mcp",
                            "server_label": "chinook_db_server",  # mcp_db_server.py의 서버 이름과 일치
                            "server_url": mcp_server_url,
                        }
                    ],
                    previous_response_id=previous_response_id,
                )
            else:
                # 첫 번째 요청
                response = client.responses.create(
                    model=model,
                    input=user_input,
                    tools=[
                        {
                            "type": "mcp",
                            "server_label": "chinook_db_server",
                            "server_url": mcp_server_url,
                        }
                    ],
                )
            
            # 승인 요청 처리
            approval_responses = handle_approval_request(response)
            
            if approval_responses:
                # 승인 응답과 함께 다시 요청
                response = client.responses.create(
                    model=model,
                    input=approval_responses,
                    tools=[
                        {
                            "type": "mcp",
                            "server_label": "chinook_db_server",
                            "server_url": mcp_server_url,
                        }
                    ],
                    previous_response_id=response.id,
                )
            
            # 응답 텍스트 추출
            response_text = response.output_text
            
            # previous_response_id 업데이트
            previous_response_id = response.id
            
            # 응답 출력
            print("\n===== RESPONSE =====")
            print(response_text)
            print("=" * 50 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n챗봇을 종료합니다.")
            break
        except Exception as e:
            print(f"\n오류 발생: {e}")
            import traceback
            traceback.print_exc()
            print("=" * 50 + "\n")

# --------------------------------------------------------
# 6️⃣ 프로그램 실행
# --------------------------------------------------------
if __name__ == "__main__":
    main()

