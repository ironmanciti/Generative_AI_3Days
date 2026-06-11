"""
카페 무브 안내 챗봇 - 간단한 웹 UI 백엔드 (Flask)

실행 방법:
    pip install flask openai python-dotenv
    python app.py
    브라우저에서 http://127.0.0.1:5000 접속

핵심:
    - instructions 로 챗봇 역할/규칙을 정한다.
    - 첫 호출에 안내 문서(dataset)를 넘겨주고, 이후에는
      previous_response_id 로 대화 맥락을 자동으로 이어간다.
"""
import os
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # 상위 폴더의 .env 에 있는 OPENAI_API_KEY 를 읽습니다.
client = OpenAI()

Model = "gpt-5-nano"

# 챗봇이 참고할 짧은 안내 문서 (FAQ)
dataset = """[카페 무브(Cafe Move) 안내]
- 영업시간: 매일 오전 9시 ~ 오후 10시 (연중무휴)
- 위치: 서울시 강남구 테헤란로 123, 1층
- 대표 메뉴: 아메리카노 4,500원 / 카페라떼 5,000원 / 바닐라라떼 5,500원
- 와이파이: 무료 제공 (비밀번호는 영수증 하단에 표시)
- 주차: 1시간 무료, 이후 10분당 1,000원
- 반려동물: 소형견에 한해 동반 가능
"""

instructions = """
당신은 '카페 무브'의 친절한 안내 도우미입니다.
사용자가 제공한 안내 문서의 내용만 근거로 답변하세요.
문서에 없는 내용은 "죄송합니다. 해당 정보는 없습니다."라고만 답하세요.
"""

app = Flask(__name__)


@app.route("/")
def index():
    # 같은 폴더의 index.html 을 그대로 전달합니다.
    return send_from_directory(os.path.dirname(__file__), "index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = data.get("message", "")
    prev_id = data.get("response_id")  # 이전 응답 id (없으면 첫 대화)

    if prev_id:
        # 두 번째 이후: previous_response_id 로 맥락을 이어갑니다.
        response = client.responses.create(
            model=Model,
            previous_response_id=prev_id,
            input=[{"role": "user", "content": message}],
        )
    else:
        # 첫 대화: 안내 문서와 첫 질문을 함께 넘겨줍니다.
        response = client.responses.create(
            model=Model,
            instructions=instructions,
            input=[
                {"role": "user", "content": dataset},
                {"role": "user", "content": message},
            ],
        )

    # output_text(답변)와 response_id(다음 호출에 사용)를 함께 반환합니다.
    return jsonify({"reply": response.output_text, "response_id": response.id})


if __name__ == "__main__":
    app.run(port=5000, debug=True)
