# ---------------------------------------------------------
# FastMCP를 이용한 간단한 의류 가격 관리 서버 예제
# ---------------------------------------------------------
# 기능:
# - 가격 조회(get_price)
# - 아이템 추가/업데이트(add_item)
# - 전체 목록 보기(list_items)
# ---------------------------------------------------------

from typing import Dict, List, Tuple
from fastmcp import FastMCP

# MCP 서버 인스턴스 생성 (서버 이름 지정)
mcp = FastMCP("clothing_price_server")

# 기본 의류 재고 (아이템명: 가격)
INVENTORY: Dict[str, float] = {
    "t-shirt": 19.99,
    "jeans":   59.90,
    "hoodie":  39.95,
}

# ---------------------------------------------------------
# 헬퍼 함수 (내부용)
# ---------------------------------------------------------
def _normalize(item: str) -> str:
    """아이템 이름을 소문자 및 공백 제거 형태로 정규화"""
    return item.strip().lower()

def _item_exists(item: str) -> bool:
    """아이템 존재 여부 확인"""
    return _normalize(item) in INVENTORY

# ---------------------------------------------------------
# MCP Tool 정의 (각 함수가 외부에서 호출 가능한 도구가 됨)
# ---------------------------------------------------------
@mcp.tool(description="의류 품목의 가격을 조회합니다. 항상 (found, price)를 반환합니다.")
def get_price(item: str) -> Tuple[bool, float]:
    """
    아이템의 가격을 조회하는 함수
    반환값: (존재여부, 가격)
    """
    key = _normalize(item)
    return (_item_exists(key), INVENTORY.get(key, 0.0))

@mcp.tool(description="의류 품목을 추가하거나 가격을 업데이트합니다. 항상 (item, price)를 반환합니다.")
def add_item(item: str, price: float) -> Tuple[str, float]:
    """
    새로운 아이템 추가 또는 기존 아이템 가격 갱신
    음수 가격이 들어오면 0.0으로 처리
    """
    key = _normalize(item)
    INVENTORY[key] = max(price, 0.0)
    return key, INVENTORY[key]

@mcp.tool(description="모든 의류 품목과 가격 목록을 반환합니다.")
def list_items() -> List[Tuple[str, float]]:
    """
    현재 재고에 등록된 모든 아이템과 가격을 (정렬된) 리스트로 반환
    """
    return sorted(INVENTORY.items())

# ---------------------------------------------------------
# MCP 서버 실행
# ---------------------------------------------------------
if __name__ == "__main__":
    # transport="streamable-http" 방식으로 HTTP 서버 실행
    # 기본 포트는 3000번
    mcp.run(transport="streamable-http", port=3000)