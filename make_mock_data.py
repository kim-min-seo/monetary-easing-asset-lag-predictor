# ============================================================
#  tests/make_mock_data.py — 스모크 테스트용 합성 raw 데이터
#  FRED API 키 없이 02→07 파이프라인을 검증하기 위한 더미.
#  · 2000-01 ~ 2024-12 월별
#  · 가격 시리즈: 양수 랜덤워크
#  · Fed_Assets(WALCL): 2002-12부터 시작(2001 이벤트 NaN 재현),
#    2008·2020 구간에 급증(QE) 패턴 주입
# ============================================================

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

np.random.seed(42)


def main():
    idx = pd.date_range("2000-01-01", "2024-12-01", freq="MS")
    n = len(idx)

    def rw(start, vol):
        steps = np.random.normal(0, vol, n)
        return start * np.exp(np.cumsum(steps))

    data = {
        # Yahoo 자산
        "Gold":  rw(280, 0.04),
        "WTI":   rw(30, 0.08),
        "DXY":   rw(100, 0.02),
        "SP500": rw(1400, 0.04),
        "VIX":   np.clip(rw(20, 0.10), 9, 80),
        # FRED 거시
        "FedRate":     np.clip(rw(5, 0.05), 0.05, 8),
        "CPI":         rw(170, 0.003),
        "M2":          rw(4700, 0.006),
        "GDP":         rw(10000, 0.008),
        "CaseShiller": rw(110, 0.006),
        "T10Y":        np.clip(rw(5, 0.03), 0.5, 7),
        "T2Y":         np.clip(rw(4, 0.03), 0.1, 6),
        "TIPS_10Y":    np.clip(rw(2, 0.05), -1, 4),
        "PPI":         rw(130, 0.004),
        "PPI_Core":    rw(130, 0.004),
    }

    # Fed_Assets (WALCL): 2002-12부터, QE 급증 패턴
    fed = np.full(n, np.nan)
    base_start = pd.Timestamp("2002-12-01")
    level = 730000.0  # $백만 (~$0.73T)
    started = False
    for i, d in enumerate(idx):
        if d < base_start:
            continue
        started = True
        # 평상시 소폭 증가
        growth = 0.004
        # 2008 금융위기 QE
        if pd.Timestamp("2008-09-01") <= d <= pd.Timestamp("2010-06-01"):
            growth = 0.06
        # 2020 팬데믹 대규모 QE
        if pd.Timestamp("2020-03-01") <= d <= pd.Timestamp("2021-12-01"):
            growth = 0.09
        level *= (1 + growth)
        fed[i] = level
    data["Fed_Assets"] = fed

    df = pd.DataFrame(data, index=idx)
    df.index.name = "date"

    os.makedirs(C.DATA_RAW_DIR, exist_ok=True)
    path = os.path.join(C.DATA_RAW_DIR, "raw_data.csv")
    df.to_csv(path)
    print(f"  ✓ 합성 raw 데이터 저장: {path} {df.shape}")
    print(f"    WALCL 시작: {df['Fed_Assets'].first_valid_index()} "
          f"(2001 이벤트 NaN 재현)")
    return df


if __name__ == "__main__":
    main()
