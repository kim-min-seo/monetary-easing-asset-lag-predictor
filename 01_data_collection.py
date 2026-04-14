# ============================================================
#  01_data_collection.py — 데이터 수집 (v6)
#  FRED API + Yahoo Finance → data/raw/ 저장
# ============================================================

import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C


def collect_fred_data():
    try:
        from fredapi import Fred
    except ImportError:
        print("  ⚠️  pip install fredapi")
        return None

    print("  📊 FRED 데이터 수집 중...")
    fred = Fred(api_key=C.FRED_API_KEY)
    data = {}

    for name, sid in C.FRED_SERIES.items():
        try:
            s = fred.get_series(
                sid,
                observation_start=C.START_DATE,
                observation_end=C.END_DATE
            )
            data[name] = s
            print(f"    ✓ {name} ({sid}): {len(s)}개")
        except Exception as e:
            print(f"    ✗ {name}: {e}")

    df = pd.DataFrame(data)
    df.index = pd.to_datetime(df.index)
    return df.resample("MS").last()


def collect_yahoo_data():
    try:
        import yfinance as yf
    except ImportError:
        print("  ⚠️  pip install yfinance")
        return None

    print("  📈 Yahoo Finance 데이터 수집 중...")
    data = {}

    for name, ticker in C.YAHOO_TICKERS.items():
        try:
            raw = yf.download(
                ticker,
                start=C.START_DATE,
                end=C.END_DATE,
                interval="1mo",
                progress=False,
                auto_adjust=True
            )
            if raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            s = raw["Close"].squeeze()
            if isinstance(s, pd.Series) and len(s) > 0:
                s.name = name
                data[name] = s
                print(f"    ✓ {name} ({ticker}): {len(s)}개")
        except Exception as e:
            print(f"    ✗ {name}: {e}")

    if not data:
        return None
    df = pd.DataFrame(data)
    df.index = pd.to_datetime(df.index)
    return df.resample("MS").last()


def generate_sample_data(n=300):
    """API 없이 테스트 가능한 샘플 데이터"""
    print("  📝 샘플 데이터 생성 (테스트 모드)...")
    dates = pd.date_range(start=C.START_DATE, periods=n, freq="MS")
    rng   = np.random.default_rng(42)

    fed_rate = np.zeros(n)
    fed_rate[:50]    = 5.5
    fed_rate[50:80]  = np.linspace(5.5, 1.0, 30)
    fed_rate[80:130] = 1.0
    fed_rate[130:160]= np.linspace(1.0, 5.25, 30)
    fed_rate[160:190]= np.linspace(5.25, 0.25, 30)
    fed_rate[190:240]= 0.25
    fed_rate[240:260]= np.linspace(0.25, 2.5, 20)
    fed_rate[260:270]= np.linspace(2.5, 1.75, 10)
    fed_rate[270:280]= np.linspace(1.75, 0.25, 10)
    fed_rate[280:]   = np.linspace(0.25, 4.5, n-280)
    fed_rate += rng.standard_normal(n) * 0.1

    fed_chg    = np.diff(fed_rate, prepend=fed_rate[0])
    cpi_base   = 100 * np.exp(np.cumsum(
        0.002 - fed_chg*0.1 + rng.standard_normal(n)*0.003))
    fed_assets = 800 + np.cumsum(
        np.maximum(-fed_chg*500, 0)) + rng.standard_normal(n)*50
    dxy     = 100 - fed_rate*2 + rng.standard_normal(n)*3
    m2      = 7000 * np.exp(np.cumsum(-fed_chg*0.05 + 0.003))
    gdp     = 10000 * np.exp(np.cumsum(0.004 + rng.standard_normal(n)*0.008))
    tips_10y= np.clip(fed_rate - 2.0 + rng.standard_normal(n)*0.3, -2, 4)

    mon    = -fed_chg
    gold_r = np.roll(mon, 4)  * 2.0 + rng.standard_normal(n)*0.03
    wti_r  = np.roll(mon, 8)  * 2.5 + rng.standard_normal(n)*0.05
    sp_r   = np.roll(mon, 6)  * 3.0 + rng.standard_normal(n)*0.04
    cs_r   = np.roll(mon, 18) * 1.5 + rng.standard_normal(n)*0.02

    df = pd.DataFrame({
        "FedRate":    fed_rate,
        "Fed_Assets": fed_assets,
        "T10Y":       fed_rate+1.5+rng.standard_normal(n)*0.3,
        "T2Y":        fed_rate+0.5+rng.standard_normal(n)*0.2,
        "CPI":        cpi_base,
        "M2":         m2,
        "GDP":        gdp,
        "CaseShiller":100*np.exp(np.cumsum(cs_r*0.01)),
        "TIPS_10Y":   tips_10y,
        "Gold":       400*np.exp(np.cumsum(gold_r*0.01)),
        "WTI":        40 *np.exp(np.cumsum(wti_r*0.01)),
        "DXY":        dxy,
        "SP500":      1000*np.exp(np.cumsum(sp_r*0.01)),
    }, index=dates)

    print(f"  ✓ 완료: {df.shape[0]}개월 × {df.shape[1]}개 변수")
    return df


def main():
    print("\n[01] 데이터 수집")

    use_sample = C.FRED_API_KEY in ["your_fred_api_key_here", ""]

    if use_sample:
        print("  ℹ️  샘플 데이터 사용 (.env 파일에 FRED_API_KEY 설정 필요)")
        df = generate_sample_data()
    else:
        df_fred  = collect_fred_data()
        df_yahoo = collect_yahoo_data()
        frames   = [f for f in [df_fred, df_yahoo] if f is not None]
        if not frames:
            df = generate_sample_data()
        else:
            df = pd.concat(frames, axis=1).sort_index()
            df = df.loc[~df.index.duplicated(keep="first")]

    print(f"\n  ✅ 완료: {df.shape[0]}개월 × {df.shape[1]}개 변수")
    print(f"     기간: {df.index[0].strftime('%Y-%m')} ~ "
          f"{df.index[-1].strftime('%Y-%m')}")

    path = os.path.join(C.DATA_RAW_DIR, "raw_data.csv")
    df.to_csv(path)
    print(f"  ✓ 저장: {path}")
    return df


if __name__ == "__main__":
    main()
