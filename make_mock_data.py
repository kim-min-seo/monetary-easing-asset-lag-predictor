# ============================================================
#  make_mock_data.py — 스모크 테스트용 합성 raw 데이터
#  FRED API 키 없이 02→07 파이프라인을 검증하기 위한 더미.
#
#  ⚠️ 안전장치 (검수#2.5):
#  - 기본 출력은 data/raw_mock/raw_data.csv (실제 data/raw/ 는 안 건드림).
#  - 실제 raw_data.csv를 덮어쓰려면 명시적으로 --into-real 플래그가 필요.
#    이때도 기존 실데이터는 raw_data.csv.bak 으로 백업한 뒤 덮어씀.
#
#  · 2000-01 ~ 2024-12 월별 / 가격 시리즈 = 양수 랜덤워크
#  · Fed_Assets(WALCL): 2002-12부터 시작(2001 이벤트 NaN 재현),
#    2008·2020 구간에 급증(QE) 패턴 주입
# ============================================================

import os
import sys
import argparse
import shutil
import numpy as np
import pandas as pd

# config 위치를 견고하게 탐색 (루트/tests 어디서 실행되든 동작)
_HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in (_HERE, os.path.dirname(_HERE)):
    if os.path.exists(os.path.join(_cand, "config.py")):
        sys.path.insert(0, _cand)
        break
import config as C

np.random.seed(42)


def build_mock_df():
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
    for i, d in enumerate(idx):
        if d < base_start:
            continue
        growth = 0.004  # 평상시 소폭 증가
        if pd.Timestamp("2008-09-01") <= d <= pd.Timestamp("2010-06-01"):
            growth = 0.06   # 2008 금융위기 QE
        if pd.Timestamp("2020-03-01") <= d <= pd.Timestamp("2021-12-01"):
            growth = 0.09   # 2020 팬데믹 대규모 QE
        level *= (1 + growth)
        fed[i] = level
    data["Fed_Assets"] = fed

    df = pd.DataFrame(data, index=idx)
    df.index.name = "date"
    return df


def main():
    ap = argparse.ArgumentParser(
        description="스모크 테스트용 합성 raw 데이터 생성 (기본: 격리 출력)")
    ap.add_argument(
        "--into-real", action="store_true",
        help="실제 data/raw/raw_data.csv 를 합성 데이터로 덮어씀 "
             "(기존 파일은 .bak 백업). 테스트용 throwaway 클론에서만 사용 권장.")
    args = ap.parse_args()

    df = build_mock_df()

    if args.into_real:
        out_dir = C.DATA_RAW_DIR
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "raw_data.csv")
        if os.path.exists(path):
            bak = path + ".bak"
            shutil.copy2(path, bak)
            print(f"  ⚠️ 기존 실데이터 백업: {bak}")
        df.to_csv(path)
        print(f"  ⚠️ [--into-real] 실제 raw 데이터를 합성으로 덮어씀: {path} {df.shape}")
        print(f"     복구: mv '{path}.bak' '{path}'")
    else:
        out_dir = os.path.join(os.path.dirname(C.DATA_RAW_DIR), "raw_mock")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "raw_data.csv")
        df.to_csv(path)
        print(f"  ✓ 합성 raw 데이터 저장(격리): {path} {df.shape}")
        print(f"    실데이터는 안전. 파이프라인 스모크 테스트는 throwaway 클론에서")
        print(f"    `python make_mock_data.py --into-real` 로.")

    print(f"    WALCL 시작: {df['Fed_Assets'].first_valid_index()} "
          f"(2001 이벤트 NaN 재현)")
    return df


if __name__ == "__main__":
    main()