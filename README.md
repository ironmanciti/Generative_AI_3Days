# Generative AI 3Days

OpenAI API를 활용한 생성형 AI 실습 프로젝트 (3일 과정)

## 커리큘럼

### Day 1 - AI 기초 및 언어 모델 이해
| 노트북 | 주제 |
|--------|------|
| `010_embedding.ipynb` | 임베딩(Embedding) 개념 및 실습 |
| `150_autoregressive_language_generation.ipynb` | 자기회귀 언어 생성 |
| `151_autoregressive_language_generation_ClovaX.ipynb` | ClovaX를 활용한 언어 생성 |
| `160_Interacting_CLIP.ipynb` | CLIP 모델 활용 (텍스트-이미지 연결) |

### Day 2 - OpenAI API 활용
| 노트북 | 주제 |
|--------|------|
| `200_Text_Generation-Prompts.ipynb` | 텍스트 생성 및 프롬프트 엔지니어링 |
| `210_Structured_Output.ipynb` | 구조화된 출력 (Structured Output) |
| `220_Function_Call.ipynb` | 함수 호출 (Function Calling) |
| `225_Tools.ipynb` | 도구 활용 (Web Search, File Search, MCP) |
| `230_Conversation_state.ipynb` | 대화 상태 관리 |
| `235_Reasoning.ipynb` | 추론 (Reasoning) |
| `250_Streaming.ipynb` | 스트리밍 API 응답 |

### Day 3 - 멀티모달 및 고급 활용
| 노트북 | 주제 |
|--------|------|
| `310_Image_Generation.ipynb` | 이미지 생성 |
| `320_Vision.ipynb` | 비전 (이미지 인식 및 분석) |
| `401_Embeddings_NaverMovie_Sentiment_Analysis.ipynb` | 임베딩 기반 네이버 영화 감성 분석 |
| `700_Agents_Overview.ipynb` | AI 에이전트 개요 |
| `701_Agent_Chatbot.py` | 에이전트 챗봇 구현 |

### MCP (Model Context Protocol)
| 파일 | 주제 |
|------|------|
| `MCP/mcp_server.py` | MCP 서버 구현 |
| `MCP/mcp_db_server.py` | MCP DB 서버 구현 |
| `MCP/tool_mcp.py` | MCP 도구 클라이언트 |
| `MCP/responses_api.ipynb` | MCP + Responses API 연동 |
| `MCP/responses_db_api.py` | MCP DB + Responses API 연동 |

### Google Workspace 연동
| 파일 | 주제 |
|------|------|
| `appscript-docs.js` | Google Docs 연동 |
| `appscript-sheets.js` | Google Sheets 연동 |
| `appscript-slides.js` | Google Slides 연동 |

## 프로젝트 구조

```
├── data/                # 실습 데이터 (CSV, PDF, 이미지, 오디오)
├── db/                  # ChromaDB 벡터 데이터베이스
├── images/              # 실습용 이미지
├── output/              # 생성 결과물 (이미지, 오디오, 임베딩)
├── MCP/                 # MCP 서버 및 클라이언트
├── 재무제표/             # 재무제표 데이터 (PDF, TXT)
└── .env_sample          # 환경변수 샘플 (API 키 설정)
```

## 환경 설정

1. `.env_sample`을 `.env`로 복사 후 API 키 설정
2. 필요한 패키지 설치: `openai`, `python-dotenv`, `tiktoken`, `chromadb` 등
