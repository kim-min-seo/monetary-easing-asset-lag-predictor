# ============================================================
#  02_preprocessing.py — 전처리 (v7 Regime)
#  ★ v7 핵심 추가:
#  QVAR Spillover 결과 기반 경기 국면 변수 생성
#
#  QVAR 결과:
#  τ=0.05 침체기: SP500(+24.53), Oil(+14.28) 선도
#  τ=0.50 중립기: Oil→CPI 경로 지배적
#  τ=0.95 과열기: Oil→CPI(24.71), M2→CaseShiller(11.26)
#
#  → 국면 변수를 피처로 추가하면
#    모델이 현재 국면을 알고 예측 가능
#    국면별 별도 모델 학습 시 핵심 변수
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

    if "CaseShiller_LogReturn" in new_cols:
        new_cols["CaseShiller_LogReturn2"] = \
            new_cols["CaseShiller_LogReturn"].diff()
        print("  ✓ CaseShiller 2차 차분 적용")

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

    if "T10Y" in df.columns and "TIPS_10Y" in df.columns:
        new_cols["TIPS_Spread"]      = df["T10Y"] - df["TIPS_10Y"]
        new_cols["Inflation_Expect"] = df["T10Y"] - df["TIPS_10Y"]
        print("  ✓ TIPS 스프레드 생성")

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

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
        s = pd.Timestamp(start); e = pd.Timestamp(end)
        cut_end  = s + pd.DateOffset(months=3)
        hike_end = e + pd.DateOffset(months=3)
        new_cols["Cut_Start"]  = new_cols["Cut_Start"].where(
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
#  ★ v7 핵심: QVAR 기반 경기 국면 변수 생성
# ──────────────────────────────────────────────

def add_qvar_regime_features(df):
    """
    QVAR Spillover 결과를 피처로 변환

    QVAR 결과:
    침체기(τ=0.05): 실질금리 낮음 + QE 활발 + M2 높음
    중립기(τ=0.50): 평상시
    과열기(τ=0.95): 실질금리 높음 + QE 없음 + M2 낮음

    국면별 핵심 전이 경로:
    침체기: SP500→시장 선도, Oil→시장 선도
    중립기: Oil→CPI 지배적
    과열기: Oil→CPI, M2→CaseShiller 강함
    """
    print("\n  [2-4] QVAR 기반 경기 국면 변수 생성 (★ v7 핵심)")
    new_cols = {}

    # ────────────────────────────────────────
    # 국면 분류 (3단계)
    # Real_Rate + QE_Size + M2_YoY 기반
    # ────────────────────────────────────────
    if all(c in df.columns for c in
           ["Real_Rate","QE_Size","M2_YoY"]):

        rr   = df["Real_Rate"]
        qe   = df["QE_Size"]
        m2y  = df["M2_YoY"]

        # 각 변수 롤링 분위수 기준으로 국면 판단
        rr_low  = rr  < rr.rolling(36).quantile(0.33)
        rr_high = rr  > rr.rolling(36).quantile(0.67)
        qe_high = qe  > qe.rolling(36).quantile(0.67)
        m2_high = m2y > m2y.rolling(36).quantile(0.67)
        m2_low  = m2y < m2y.rolling(36).quantile(0.33)

        # 침체기 (τ≈0.05): 실질금리 낮 + QE 활발 + M2 높음
        recession = (rr_low & qe_high & m2_high).astype(int)

        # 과열기 (τ≈0.95): 실질금리 높 + M2 낮음
        overheating = (rr_high & m2_low).astype(int)

        # 중립기 (τ≈0.50): 나머지
        neutral = ((recession == 0) & (overheating == 0)).astype(int)

        new_cols["Regime_Recession"]   = recession
        new_cols["Regime_Neutral"]     = neutral
        new_cols["Regime_Overheating"] = overheating

        # 국면 인덱스 (0=침체, 1=중립, 2=과열)
        new_cols["Regime_Index"] = (
            recession * 0 + neutral * 1 + overheating * 2
        )

        print("  ✓ 경기 국면 분류 생성")
        print(f"    침체기: {recession.sum()}개월")
        print(f"    중립기: {neutral.sum()}개월")
        print(f"    과열기: {overheating.sum()}개월")

    # ────────────────────────────────────────
    # QVAR 침체기 핵심 경로 피처
    # SP500(NET+24.53), Oil(NET+14.28) 선도
    # ────────────────────────────────────────
    if "SP500_LogReturn" in df.columns and "Regime_Recession" in new_cols:
        for lag in [1, 3, 6]:
            # 침체기에서 SP500이 선도자 → SP500 시차 강조
            new_cols[f"Regime_Rec_SP500_lag{lag}"] = (
                df["SP500_LogReturn"].shift(lag) *
                new_cols["Regime_Recession"]
            )
        print("  ✓ 침체기 SP500 선도 피처 생성")

    if "WTI_LogReturn" in df.columns and "Regime_Recession" in new_cols:
        for lag in [1, 3, 6]:
            # 침체기에서 Oil이 선도자
            new_cols[f"Regime_Rec_Oil_lag{lag}"] = (
                df["WTI_LogReturn"].shift(lag) *
                new_cols["Regime_Recession"]
            )
        print("  ✓ 침체기 Oil 선도 피처 생성")

    # ────────────────────────────────────────
    # QVAR 중립기 핵심 경로 피처
    # Oil→CPI(27.61) 경로 지배적
    # ────────────────────────────────────────
    if "WTI_LogReturn" in df.columns and "Regime_Neutral" in new_cols:
        for lag in [1, 3, 6]:
            # 중립기에서 Oil→CPI 경로 강조
            new_cols[f"Regime_Neu_Oil_lag{lag}"] = (
                df["WTI_LogReturn"].shift(lag) *
                new_cols["Regime_Neutral"]
            )
        print("  ✓ 중립기 Oil→CPI 경로 피처 생성")

    # ────────────────────────────────────────
    # QVAR 과열기 핵심 경로 피처
    # Oil→CPI(24.71), M2→CaseShiller(11.26)
    # ────────────────────────────────────────
    if "WTI_LogReturn" in df.columns and "Regime_Overheating" in new_cols:
        for lag in [1, 3, 6]:
            # 과열기 Oil→CPI 경로
            new_cols[f"Regime_Ovr_Oil_lag{lag}"] = (
                df["WTI_LogReturn"].shift(lag) *
                new_cols["Regime_Overheating"]
            )
        print("  ✓ 과열기 Oil→CPI 경로 피처 생성")

    if "M2_YoY" in df.columns and "Regime_Overheating" in new_cols:
        for lag in [1, 3, 6]:
            # 과열기 M2→CaseShiller 경로
            new_cols[f"Regime_Ovr_M2_lag{lag}"] = (
                df["M2_YoY"].shift(lag) *
                new_cols["Regime_Overheating"]
            )
        print("  ✓ 과열기 M2→CaseShiller 경로 피처 생성")

    # ────────────────────────────────────────
    # TCI (Total Connectedness Index) 근사
    # 국면별 시장 전체 연결성 지수
    # 침체기: 28.81%, 중립기: 20.82%, 과열기: 29.82%
    # ────────────────────────────────────────
    if "Regime_Recession" in new_cols:
        new_cols["TCI_Approx"] = (
            new_cols["Regime_Recession"]   * 28.81 +
            new_cols["Regime_Neutral"]     * 20.82 +
            new_cols["Regime_Overheating"] * 29.82
        )
        print("  ✓ TCI 근사값 피처 생성")

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


# ──────────────────────────────────────────────
#  기술적 지표
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
    print("\n  [2-5] 피처 엔지니어링 (v6 + QVAR 국면 피처)")

    monetary_cols    = [c for c in C.MONETARY_VARS if c in df.columns]
    asset_price_cols = [c for c in ["Gold","WTI","SP500","CaseShiller"]
                        if c in df.columns]
    asset_lr_cols    = [c for c in df.columns if "_LogReturn" in c
                        and "lag" not in c]

    new_cols = {}

    print("  → 시차 변수 생성")
    for col in monetary_cols:
        for lag in C.LAG_PERIODS:
            new_cols[f"{col}_lag{lag}"] = df[col].shift(lag)

    for col in asset_lr_cols:
        if col in df.columns:
            for lag in [1, 3, 6, 12]:
                new_cols[f"{col}_lag{lag}"] = df[col].shift(lag)

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

    print("  → 변동성 생성")
    for col in asset_lr_cols:
        if col in df.columns:
            new_cols[f"{col}_Vol6"]  = df[col].rolling(6).std()
            new_cols[f"{col}_Vol12"] = df[col].rolling(12).std()

    print("  → 모멘텀 생성")
    for col in asset_price_cols:
        if col in df.columns:
            new_cols[f"{col}_Mom3_12"] = (
                df[col].rolling(3).mean() /
                (df[col].rolling(12).mean() + 1e-8)
            )
            new_cols[f"{col}_ROC6"]  = df[col].pct_change(6)  * 100
            new_cols[f"{col}_ROC12"] = df[col].pct_change(12) * 100

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

    print("  → M2 YoY 시차 생성")
    if "M2_YoY" in df.columns:
        for lag in [1, 3, 6, 9, 12, 18, 24]:
            new_cols[f"M2_YoY_lag{lag}"] = df["M2_YoY"].shift(lag)

    print("  → TIPS 스프레드 시차 생성")
    if "TIPS_Spread" in df.columns:
        for lag in [1, 3, 6, 9, 12]:
            new_cols[f"TIPS_Spread_lag{lag}"] = df["TIPS_Spread"].shift(lag)

    # ★ v7 추가: PPI 피처 (CPI 선행지표)
    print("  → PPI 피처 생성 (★ v7: CPI 개선)")
    if "PPI" in df.columns:
        ppi_lr = np.log(df["PPI"] / df["PPI"].shift(1)) * 100
        new_cols["PPI_LogReturn"]   = ppi_lr
        new_cols["PPI_YoY"]        = df["PPI"].pct_change(12) * 100
        new_cols["PPI_Accel"]      = ppi_lr.diff()
        new_cols["PPI_CPI_Spread"] = new_cols["PPI_YoY"] - df["CPI"].pct_change(12) * 100
        for lag in [1, 3, 6, 9, 12]:
            new_cols[f"PPI_LogReturn_lag{lag}"] = ppi_lr.shift(lag)
            new_cols[f"PPI_YoY_lag{lag}"]       = new_cols["PPI_YoY"].shift(lag)

    if "PPI_Core" in df.columns:
        ppi_core_lr = np.log(df["PPI_Core"] / df["PPI_Core"].shift(1)) * 100
        new_cols["PPI_Core_LogReturn"] = ppi_core_lr
        for lag in [1, 3, 6]:
            new_cols[f"PPI_Core_lag{lag}"] = ppi_core_lr.shift(lag)

    # ★ v7 추가: VIX 피처 (SP500 선행지표)
    print("  → VIX 피처 생성 (★ v7: SP500 개선)")
    if "VIX" in df.columns:
        new_cols["VIX_Level"]    = df["VIX"]
        new_cols["VIX_Change"]   = df["VIX"].diff()
        new_cols["VIX_YoY"]     = df["VIX"].pct_change(12) * 100
        new_cols["VIX_MA3"]     = df["VIX"].rolling(3).mean()
        new_cols["VIX_High"]    = (df["VIX"] > 30).astype(int)   # 공포 구간
        new_cols["VIX_Low"]     = (df["VIX"] < 15).astype(int)   # 탐욕 구간
        # VIX는 SP500과 역관계 → 선행 신호
        for lag in [1, 2, 3, 6]:
            new_cols[f"VIX_lag{lag}"]        = df["VIX"].shift(lag)
            new_cols[f"VIX_Change_lag{lag}"] = new_cols["VIX_Change"].shift(lag)

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.dropna(subset=[c for c in df.columns if "lag" in c][:3])

    print(f"  ✓ 최종 피처 수: {df.shape[1]}, 데이터: {df.shape[0]}개월")
    return df


# ──────────────────────────────────────────────
#  메인
# ──────────────────────────────────────────────

def main():
    print("\n[02] 전처리 (v7 Regime)")

    raw_path = os.path.join(C.DATA_RAW_DIR, "raw_data.csv")
    if not os.path.exists(raw_path):
        print("  ⚠️  raw_data.csv 없음 → 01 먼저 실행")
        return None

    df = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    print(f"  ✓ 원본 데이터 로드: {df.shape}")

    df = basic_preprocess(df)
    df = build_monetary_vars(df)
    df = add_rate_cycle_dummies(df)
    df = add_qvar_regime_features(df)  # ★ v7 핵심
    df = build_features(df)

    path = os.path.join(C.DATA_PROC_DIR, "processed_data.csv")
    df.to_csv(path)
    print(f"\n  ✓ 전처리 완료 저장: {path}")
    return df


if __name__ == "__main__":
    main()
