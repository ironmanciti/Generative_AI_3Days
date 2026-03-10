# 토큰화 상세 분석
import tiktoken

def analyze_tokenization_detailed():
    """토큰화를 자세히 분석하는 함수"""
    encoding = tiktoken.get_encoding("cl100k_base")
    
    # 띄어쓰기 없는 문장
    sentence1 = "아버지가방에들어가신다"
    tokens1 = [54059, 80104, 22035, 20565, 39277, 102, 19954, 65950, 32179, 20565, 83628, 13447]
    
    # 띄어쓰기 있는 문장
    sentence2 = "아버지가 방에 들어가신다"
    tokens2 = encoding.encode(sentence2)
    
    print("=== 띄어쓰기 없는 문장 토큰 분석 ===")
    print(f"원본: {sentence1}")
    print(f"토큰 ID: {tokens1}")
    print(f"토큰 개수: {len(tokens1)}")
    print()
    
    print("각 토큰을 개별적으로 디코딩:")
    for i, token_id in enumerate(tokens1):
        decoded_token = encoding.decode([token_id])
        print(f"토큰 {i+1:2d} (ID: {token_id:5d}): '{decoded_token}'")
    
    print(f"\n전체 디코딩 결과: '{encoding.decode(tokens1)}'")
    
    print("\n" + "="*50)
    
    print("=== 띄어쓰기 있는 문장 토큰 분석 ===")
    print(f"원본: {sentence2}")
    print(f"토큰 ID: {tokens2}")
    print(f"토큰 개수: {len(tokens2)}")
    print()
    
    print("각 토큰을 개별적으로 디코딩:")
    for i, token_id in enumerate(tokens2):
        decoded_token = encoding.decode([token_id])
        print(f"토큰 {i+1:2d} (ID: {token_id:5d}): '{decoded_token}'")
    
    print(f"\n전체 디코딩 결과: '{encoding.decode(tokens2)}'")
    
    print("\n" + "="*50)
    
    print("=== 토큰화 차이점 분석 ===")
    print("1. 띄어쓰기 없는 문장:")
    print("   - 더 많은 토큰으로 분리됨 (12개)")
    print("   - 각 토큰이 더 작은 단위로 나뉨")
    print("   - 의미 단위가 아닌 문자 단위로 분리되는 경우가 많음")
    
    print("\n2. 띄어쓰기 있는 문장:")
    print("   - 더 적은 토큰으로 분리됨")
    print("   - 의미 있는 단위로 분리되는 경우가 많음")
    print("   - 공백이 토큰화에 도움을 줌")
    
    print(f"\n토큰 개수 차이: {len(tokens2) - len(tokens1)}")
    
    # 추가 분석: 부분 문자열 토큰화
    print("\n" + "="*50)
    print("=== 부분 문자열 토큰화 분석 ===")
    
    # "아버지가" 부분만 토큰화
    part1 = "아버지가"
    tokens_part1 = encoding.encode(part1)
    print(f"'{part1}' -> {tokens_part1}")
    print(f"개별 디코딩: {[encoding.decode([t]) for t in tokens_part1]}")
    
    # "방에" 부분만 토큰화
    part2 = "방에"
    tokens_part2 = encoding.encode(part2)
    print(f"'{part2}' -> {tokens_part2}")
    print(f"개별 디코딩: {[encoding.decode([t]) for t in tokens_part2]}")
    
    # "들어가신다" 부분만 토큰화
    part3 = "들어가신다"
    tokens_part3 = encoding.encode(part3)
    print(f"'{part3}' -> {tokens_part3}")
    print(f"개별 디코딩: {[encoding.decode([t]) for t in tokens_part3]}")

def compare_with_spaces():
    """공백을 추가한 다양한 버전과 비교"""
    encoding = tiktoken.get_encoding("cl100k_base")
    
    print("=== 공백 추가 버전 비교 ===")
    
    variations = [
        "아버지가방에들어가신다",      # 원본 (띄어쓰기 없음)
        "아버지가 방에들어가신다",      # 첫 번째 공백만
        "아버지가방에 들어가신다",      # 두 번째 공백만
        "아버지가 방에 들어가신다",     # 모든 공백
        "아버지 가 방에 들어가신다",    # 과도한 공백
    ]
    
    for i, text in enumerate(variations, 1):
        tokens = encoding.encode(text)
        print(f"\n{i}. '{text}'")
        print(f"   토큰 개수: {len(tokens)}")
        print(f"   토큰 ID: {tokens}")
        
        # 각 토큰 디코딩
        decoded_tokens = []
        for token_id in tokens:
            decoded = encoding.decode([token_id])
            decoded_tokens.append(f"'{decoded}'")
        print(f"   개별 토큰: {decoded_tokens}")

if __name__ == "__main__":
    analyze_tokenization_detailed()
    print("\n" + "="*60)
    compare_with_spaces() 