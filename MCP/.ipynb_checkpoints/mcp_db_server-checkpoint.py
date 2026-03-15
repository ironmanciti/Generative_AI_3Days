# ---------------------------------------------------------
# FastMCP를 이용한 Chinook DB 조회 서버 예제
# ---------------------------------------------------------
# 기능:
# - SQL 쿼리 실행(execute_sql_query)
# - 테이블 목록 조회(list_tables)
# - 테이블 스키마 조회(get_table_schema)
# ---------------------------------------------------------

from fastmcp import FastMCP
from contextlib import asynccontextmanager
import sqlite3
import os
import sys

# 전역 데이터베이스 연결 변수
db_conn = None

@asynccontextmanager
async def lifespan(app):
    """
    서버 시작/종료 시 데이터베이스 연결 관리
    """
    global db_conn
    try:
        # 서버 시작 시 데이터베이스 연결
        base_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(base_dir, "Chinook.db")
        
        # 데이터베이스 파일 존재 여부 확인 (없으면 서버 시작 안 됨)
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Chinook.db 파일을 찾을 수 없습니다: {db_path}")
        
        db_conn = sqlite3.connect(db_path)
        db_conn.row_factory = sqlite3.Row  # 딕셔너리 형태로 결과 반환
        print("✅ Chinook 데이터베이스 연결 성공", file=sys.stderr)
        
        yield
    finally:
        # 서버 종료 시 연결 정리
        if db_conn:
            db_conn.close()
            db_conn = None
            print("✅ Chinook 데이터베이스 연결 종료", file=sys.stderr)

# MCP 서버 인스턴스 생성 (서버 이름 지정, lifespan 추가)
mcp = FastMCP("chinook_db_server", lifespan=lifespan)

# ---------------------------------------------------------
# MCP Tool 정의 - Chinook DB 조회
# ---------------------------------------------------------
@mcp.tool(description="SQL 쿼리를 실행하고 결과를 반환합니다.")
def execute_sql_query(query: str) -> str:
    """
    Chinook 데이터베이스에 SQL 쿼리를 실행하고 결과를 반환합니다.
    """
    if db_conn is None:
        raise ValueError("데이터베이스가 연결되지 않았습니다.")
    
    try:
        cursor = db_conn.cursor()
        cursor.execute(query)
        
        # SELECT 쿼리인 경우 결과 반환
        if query.strip().upper().startswith('SELECT'):
            rows = cursor.fetchall()
            if not rows:
                return "쿼리 결과가 없습니다."
            
            # 컬럼명 가져오기
            columns = [description[0] for description in cursor.description]
            
            # 결과를 읽기 쉬운 형태로 포맷팅
            result_lines = [" | ".join(columns)]
            result_lines.append("-" * (len(" | ".join(columns))))
            for row in rows:
                result_lines.append(" | ".join(str(val) for val in row))
            
            return "\n".join(result_lines)
        else:
            # INSERT, UPDATE, DELETE 등의 경우
            db_conn.commit()
            return f"쿼리가 성공적으로 실행되었습니다. 영향받은 행: {cursor.rowcount}"
            
    except Exception as e:
        return f"쿼리 실행 중 오류 발생: {str(e)}"

@mcp.tool(description="Chinook 데이터베이스의 모든 테이블 목록을 반환합니다.")
def list_tables() -> list:
    """
    현재 연결된 데이터베이스에서 사용 가능한 모든 테이블의 이름을 리스트로 반환합니다.
    """
    if db_conn is None:
        raise ValueError("데이터베이스가 연결되지 않았습니다.")
    
    try:
        cursor = db_conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cursor.fetchall()]
        return tables
    except Exception as e:
        return [f"테이블 목록 조회 중 오류 발생: {str(e)}"]

@mcp.tool(description="특정 테이블의 스키마 정보(컬럼명, 데이터 타입 등)를 조회합니다.")
def get_table_schema(table_name: str) -> str:
    """
    특정 테이블의 컬럼명, 데이터 타입, 제약조건 등의 스키마 정보를 조회합니다.
    """
    if db_conn is None:
        raise ValueError("데이터베이스가 연결되지 않았습니다.")
    
    try:
        cursor = db_conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        
        if not columns:
            return f"테이블 '{table_name}'을 찾을 수 없습니다."
        
        # 스키마 정보 포맷팅
        schema_lines = [f"테이블: {table_name}", "-" * 40]
        schema_lines.append("컬럼명 | 타입 | NULL 허용 | 기본값")
        schema_lines.append("-" * 40)
        
        for col in columns:
            col_name = col[1]
            col_type = col[2]
            not_null = "NO" if col[3] else "YES"
            default_val = col[4] if col[4] else ""
            schema_lines.append(f"{col_name} | {col_type} | {not_null} | {default_val}")
        
        return "\n".join(schema_lines)
    except Exception as e:
        return f"스키마 조회 중 오류 발생: {str(e)}"

# ---------------------------------------------------------
# MCP 서버 실행
# ---------------------------------------------------------
if __name__ == "__main__":
    # transport="streamable-http" 방식으로 HTTP 서버 실행
    # 기본 포트는 3001번 (mcp_server와 다른 포트 사용)
    mcp.run(transport="streamable-http", port=3001)

