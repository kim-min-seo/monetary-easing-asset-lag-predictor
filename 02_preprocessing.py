# ============================================================
#  02_preprocessing.py — 전처리 (v6)
#  ★ v6 개선:
#  1. PerformanceWarning 해결 (pd.concat 방식)
#  2. CaseShiller 비정상 해결 (2차 차분)
#  3. TIPS 스프레드 추가 (기대 인플레이션)
# ============================================================

import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C


# ──────────────────────────────────────────────
#  기본 전처리
# ──────────────────────────────────────────────

def basic_preprocess(df):
    print("\n  [2-1] 기본 전처리")
    df = df.interpolate(method="linear",
                        limit_direction="both").ffill().bfill()

    new_cols = {}
    price_cols = ["Gold", "WTI", "SP500", "CaseShiller", "CPI"]
    for col in price_cols:
        if col in df.columns and (df[col] > 0).all():
            new_cols[f"{col}_LogReturn"] = np.log(df[col] / df[col].shift(1))
            new_cols[f"{col}_YoY"]       = df[col].pct_change(12) * 100

    # ★ v6: CaseShiller 2차 차분으로 정상성 확보
    if "CaseShiller_LogReturn" in new_cols:
        cs_lr = new_cols["CaseShiller_LogReturn"]
        new_cols["CaseShiller_LogReturn2"] = cs_lr.diff()  # 2차 차분
        print("  ✓ CaseShiller 2차 차분 적용 (정상성 확보)")

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    print(f"  ✓ 결측치: {df.isnull().sum().sum()}개")
    print(f"  ✓ 컬럼 수: {df.shape[1]}")
    return df


# ──────────────────────────────────────────────
#  통화환경 독립변수
# ──────────────────────────────────────────────

def build_monetary_vars(df):
    print("\n  [2-2] 통화환경 독립변수 구성")
    new_cols = {}

    if "FedRate" in df.columns:
        new_cols["FedRate_Change"]    = df["FedRate"].diff()
        new_cols["FedRate_Change3M"]  = df["FedRate"].diff(3)
        new_cols["FedRate_Change12M"] = df["FedRate"].diff(12)
        print("  ✓ 금리변화율 생성")

    if "FedRate" in df.columns and "CPI_YoY" in df.columns:
        new_cols["Real_Rate"]   = df["FedRate"] - df["CPI_YoY"]
        new_cols["NegRealRate"] = (new_cols["Real_Rate"] < 0).astype(int)
        print("  ✓ 실질금리 생성")

    if "Fed_Assets" in df.columns:
        new_cols["QE_Size"]   = df["Fed_Assets"].pct_change() * 100
        new_cols["QE_Active"] = (new_cols["QE_Size"] > 0).astype(int)
        print("  ✓ QE 규모 생성")

    if "DXY" in df.columns:
        new_cols["DXY_Change"] = df["DXY"].pct_change()  * 100
        new_cols["DXY_YoY"]   = df["DXY"].pct_change(12) * 100
        print("  ✓ 달러인덱스 변화율 생성")

    if "T10Y" in df.columns and "T2Y" in df.columns:
        new_cols["Yield_Spread"]   = df["T10Y"] - df["T2Y"]
        new_cols["YieldCurve_Inv"] = (new_cols["Yield_Spread"] < 0).astype(int)
        print("  ✓ 수익률 곡선 생성")

    if "M2" in df.columns and "GDP" in df.columns:
        m2g  = df["M2"].pct_change(12) * 100
        gdpg = df["GDP"].pct_change(4)  * 100
        new_cols["Excess_Liquidity"] = m2g - gdpg
        print("  ✓ 초과유동성 생성")

    if "M2" in df.columns:
        new_cols["M2_YoY"]   = df["M2"].pct_change(12) * 100
        new_cols["M2_MoM"]   = df["M2"].pct_change(1)  * 100
        new_cols["M2_Accel"] = new_cols["M2_YoY"].diff()
        new_cols["M2_High"]  = (
            new_cols["M2_YoY"] >
            new_cols["M2_YoY"].rolling(24).mean()
        ).astype(int)
        print("  ✓ M2 전년비/전월비/가속도 생성")

    # ★ v6: TIPS 스프레드 (기대 인플레이션) — Gold 유의성 확보 핵심
    if "T10Y" in df.columns and "TIPS_10Y" in df.columns:
        new_cols["TIPS_Spread"]     = df["T10Y"] - df["TIPS_10Y"]
        new_cols["Inflation_Expect"]= df["T10Y"] - df["TIPS_10Y"]
        print("  ✓ TIPS 스프레드 생성 (★ v6: 기대 인플레이션)")

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    # 통화완화 종합지수
    signals = []
    for col in ["NegRealRate","QE_Active","YieldCurve_Inv"]:
        if col in df.columns:
            signals.append(df[col])
    if "FedRate_Change" in df.columns:
        signals.append((df["FedRate_Change"] < 0).astype(int))
    if "M2_High" in df.columns:
        signals.append(df["M2_High"])
    if signals:
        df["Monetary_Ease_Index"] = sum(signals)
        print(f"  ✓ 통화완화 종합지수 (0~{len(signals)}점)")

    return df


# ──────────────────────────────────────────────
#  금리인하 더미변수
# ──────────────────────────────────────────────

def add_rate_cycle_dummies(df):
    print("\n  [2-3] 금리인하 사이클 더미변수 (3단계)")
    new_cols = {
        "Cut_Start":  pd.Series(0, index=df.index),
        "Cut_Period": pd.Series(0, index=df.index),
        "Hike_Start": pd.Series(0, index=df.index),
    }

    for i, (start, end) in enumerate(C.RATE_CUT_CYCLES):
        s = pd.Timestamp(start)
        e = pd.Timestamp(end)

        cut_end  = s + pd.DateOffset(months=3)
        hike_end = e + pd.DateOffset(months=3)

        new_cols["Cut_Start"] = new_cols["Cut_Start"].where(
            ~((df.index >= s) & (df.index < cut_end)), 1)
        new_cols["Cut_Period"] = new_cols["Cut_Period"].where(
            ~((df.index >= cut_end) & (df.index <= e)), 1)
        new_cols["Hike_Start"] = new_cols["Hike_Start"].where(
            ~((df.index >= e) & (df.index < hike_end)), 1)

        print(f"  ✓ 사이클 {i+1}: {start} ~ {end}")

    new_cols["Easing_Period"] = (
        (new_cols["Cut_Start"] == 1) | (new_cols["Cut_Period"] == 1)
    ).astype(int)

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    print(f"  ✓ 완화 구간 총 {df['Easing_Period'].sum()}개월")
    return df


# ──────────────────────────────────────────────
#  피처 엔지니어링 (★ v6: pd.concat 방식으로 PerformanceWarning 해결)
# ──────────────────────────────────────────────

def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))


def compute_macd(series, fast=12, slow=26, signal=9):
    ema_f = series.ewm(span=fast, adjust=False).mean()
    ema_s = series.ewm(span=slow, adjust=False).mean()
    macd  = ema_f - ema_s
    sig   = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig, macd - sig


def build_features(df):
    """
    ★ v6 핵심 수정:
    df[col] = ... 방식 대신
    딕셔너리에 모아서 pd.concat으로 한번에 추가
    → PerformanceWarning 완전 해결
    """
    print("\n  [2-4] 피처 엔지니어링 (v6 — PerformanceWarning 해결)")

    monetary_cols    = [c for c in C.MONETARY_VARS if c in df.columns]
    asset_price_cols = [c for c in ["Gold","WTI","SP500","CaseShiller"]
                        if c in df.columns]
    asset_lr_cols    = [c for c in df.columns if "_LogReturn" in c]

    new_cols = {}

    # 1. 시차 변수
    print("  → 시차 변수 생성")
    for col in monetary_cols:
        for lag in C.LAG_PERIODS:
            new_cols[f"{col}_lag{lag}"] = df[col].shift(lag)

    for col in asset_lr_cols:
        if col in df.columns:
            for lag in [1, 3, 6, 12]:
                new_cols[f"{col}_lag{lag}"] = df[col].shift(lag)

    # 2. 이동평균
    print("  → 이동평균 생성")
    for col in asset_price_cols:
        if col in df.columns:
            for w in C.MA_WINDOWS:
                new_cols[f"{col}_MA{w}"]  = df[col].rolling(w).mean()
                new_cols[f"{col}_EMA{w}"] = df[col].ewm(span=w).mean()
            new_cols[f"{col}_GoldenCross"] = (
                df[col].rolling(6).mean() >
                df[col].rolling(12).mean()
            ).astype(int)

    # 3. RSI + MACD
    print("  → RSI / MACD 생성")
    for col in asset_price_cols:
        if col in df.columns:
            rsi = compute_rsi(df[col], 14)
            macd, sig, hist = compute_macd(df[col])
            new_cols[f"{col}_RSI14"]     = rsi
            new_cols[f"{col}_OB"]        = (rsi > 70).astype(int)
            new_cols[f"{col}_OS"]        = (rsi < 30).astype(int)
            new_cols[f"{col}_MACD"]      = macd
            new_cols[f"{col}_MACD_Hist"] = hist
            new_cols[f"{col}_MACD_Bull"] = (macd > sig).astype(int)

    # 4. 변동성
    print("  → 변동성 생성")
    for col in asset_lr_cols:
        if col in df.columns:
            new_cols[f"{col}_Vol6"]  = df[col].rolling(6).std()
            new_cols[f"{col}_Vol12"] = df[col].rolling(12).std()

    # 5. 모멘텀
    print("  → 모멘텀 생성")
    for col in asset_price_cols:
        if col in df.columns:
            new_cols[f"{col}_Mom3_12"] = (
                df[col].rolling(3).mean() /
                (df[col].rolling(12).mean() + 1e-8)
            )
            new_cols[f"{col}_ROC6"]  = df[col].pct_change(6)  * 100
            new_cols[f"{col}_ROC12"] = df[col].pct_change(12) * 100

    # 6. 교차 자산 시차
    print("  → 교차 자산 시차 생성")
    pairs = [
        ("Gold_LogReturn",  "WTI_LogReturn"),
        ("Gold_LogReturn",  "SP500_LogReturn"),
        ("WTI_LogReturn",   "CPI_LogReturn"),
        ("SP500_LogReturn", "CaseShiller_LogReturn"),
    ]
    for src, tgt in pairs:
        if src in df.columns and tgt in df.columns:
            for lag in [1, 3, 6, 12]:
                new_cols[f"Cross_{src[:4]}_{tgt[:4]}_lag{lag}"] = \
                    df[src].shift(lag)

    # 7. M2 YoY 시차
    print("  → M2 YoY 시차 생성")
    if "M2_YoY" in df.columns:
        for lag in [1, 3, 6, 9, 12, 18, 24]:
            new_cols[f"M2_YoY_lag{lag}"] = df["M2_YoY"].shift(lag)

    # 8. TIPS 스프레드 시차 (★ v6: Gold 유의성 확보)
    print("  → TIPS 스프레드 시차 생성 (★ v6)")
    if "TIPS_Spread" in df.columns:
        for lag in [1, 3, 6, 9, 12]:
            new_cols[f"TIPS_Spread_lag{lag}"] = df["TIPS_Spread"].shift(lag)

    # ★ 한번에 concat (PerformanceWarning 해결)
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    df = df.loc[:, ~df.columns.duplicated()]  # 중복 컬럼 제거

    df = df.dropna(subset=[c for c in df.columns if "lag" in c][:3])
    print(f"  ✓ 최종 피처 수: {df.shape[1]}, 데이터: {df.shape[0]}개월")
    return df


# ──────────────────────────────────────────────
#  메인
# ──────────────────────────────────────────────

def main():
    print("\n[02] 전처리")

    raw_path = os.path.join(C.DATA_RAW_DIR, "raw_data.csv")
    if not os.path.exists(raw_path):
        print("  ⚠️  raw_data.csv 없음 → 01 먼저 실행")
        return None

    df = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    print(f"  ✓ 원본 데이터 로드: {df.shape}")

    df = basic_preprocess(df)
    df = build_monetary_vars(df)
    df = add_rate_cycle_dummies(df)
    df = build_features(df)

    path = os.path.join(C.DATA_PROC_DIR, "processed_data.csv")
    df.to_csv(path)
    print(f"\n  ✓ 전처리 완료 저장: {path}")
    return df


if __name__ == "__main__":
    main()
