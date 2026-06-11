"""투자 수익률(ROI) 계산기.

투자 원금, 현재 평가액, 투자 기간을 입력받아
총수익, 총수익률, 연환산수익률(CAGR)을 계산하여 출력합니다.
SKILL.md의 지침에 따라 모델이 이 스크립트를 실행하여 정확한 수치를 얻습니다.
"""

import argparse


def calculate_roi(principal, current, years):
    """원금, 현재 평가액, 기간으로 총수익/총수익률/연환산수익률을 계산한다."""
    total_return = current - principal
    total_return_pct = (current / principal - 1) * 100
    cagr_pct = ((current / principal) ** (1 / years) - 1) * 100
    return total_return, total_return_pct, cagr_pct


def main():
    parser = argparse.ArgumentParser(description="투자 수익률(ROI) 계산기")
    parser.add_argument("--principal", type=float, required=True, help="투자 원금")
    parser.add_argument("--current", type=float, required=True, help="현재 평가액")
    parser.add_argument("--years", type=float, required=True, help="투자 기간(년)")
    args = parser.parse_args()

    total_return, total_return_pct, cagr_pct = calculate_roi(
        args.principal, args.current, args.years
    )

    print(f"총수익: {total_return:,.0f}")
    print(f"총수익률: {total_return_pct:.2f}%")
    print(f"연환산수익률(CAGR): {cagr_pct:.2f}%")


if __name__ == "__main__":
    main()
