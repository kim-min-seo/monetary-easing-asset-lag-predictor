"""
==================================================
 자산 전이 경로 분석 (통합 독립 실행 버전 v2)
 - 데이터 수집 + 전처리 + 시각화 한 번에 실행
 - FRED 시리즈 ID 수정 완료
==================================================

[필수 설치]
py -m pip install pandas numpy matplotlib seaborn statsmodels scipy fredapi yfinance

[FRED API 키 발급]
https://fredaccount.stlouisfed.org/login
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import yfinance as yf
from fredapi import Fred
from datetime import datetime
import os, warnings
warnings.filterwarnings("ignore")

# ── 한글 폰트
import platform
if platform.system() == "Darwin":
    plt.rcParams["font.family"] = "AppleGothic"
elif platform.system() == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 0. 설정
# ============================================================

FRED_API_KEY = "FRED_API_KEY "   # ← 본인 키로 교체
START_DATE   = "2000-01-01"
END_DATE     = datetime.today().strftime("%Y-%m-%d")
OUTPUT_DIR   = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ASSETS = {
    "gold"    : "금",
    "wti"     : "WTI 원유",
    "sp500"   : "S&P500",
    "housing" : "부동산",
    "cpi"     : "CPI",
}

COLORS = {
    "gold"    : "#F4B942",
    "wti"     : "#3D7EAA",
    "sp500"   : "#2ECC71",
    "housing" : "#E74C3C",
    "cpi"     : "#9B59B6",
}

EASING_EVENTS = {
    "2001 닷컴버블": ("2001-01-01", "2003-06-01"),
    "2008 금융위기": ("2008-09-01", "2010-12-01"),
    "2020 팬데믹"  : ("2020-03-01", "2022-03-01"),
    "2024 피벗"    : ("2024-09-01", None),
}

# ============================================================
# 1. 데이터 수집 함수
# ============================================================

def fetch_fred(fred: Fred, series_id: str, col_name: str) -> pd.Series:
    s = fred.get_series(series_id,
                        observation_start=START_DATE,
                        observation_end=END_DATE)
    s = s.resample("ME").last()
    s.name = col_name
    print(f"  [FRED]     {col_name:<20} {len(s)}개월 ✓")
    return s

def fetch_yfinance(ticker: str, col_name: str) -> pd.Series:
    df = yf.download(ticker, start=START_DATE, end=END_DATE,
                     interval="1mo", auto_adjust=True, progress=False)
    s = df["Close"].squeeze().resample("ME").last()
    s.name = col_name
    print(f"  [yfinance] {col_name:<20} {len(s)}개월 ✓")
    return s

# ============================================================
# 2. 데이터 로드 (수정된 시리즈 ID)
# ============================================================

def load_data() -> pd.DataFrame:
    print("\n[1단계] 데이터 수집 중...")
    fred = Fred(api_key=FRED_API_KEY)

    series = [
        # ── 자산군 (금·WTI·S&P500 → yfinance로 수집)
        fetch_yfinance("GC=F",   "gold"),       # 금
        fetch_yfinance("CL=F",   "wti"),        # WTI 원유
        fetch_yfinance("^GSPC",  "sp500"),      # S&P 500

        # 부동산 → FRED (수정된 ID)
        fetch_fred(fred, "CSUSHPINSA", "housing"),   # Case-Shiller (비계절조정)

        # CPI → FRED
        fetch_fred(fred, "CPIAUCSL",   "cpi"),

        # ── 독립변수
        fetch_fred(fred, "FEDFUNDS",   "fed_rate"),      # 기준금리
        fetch_fred(fred, "WALCL",      "fed_balance"),   # Fed 자산 (QE 프록시)
        fetch_fred(fred, "T10Y2Y",     "yield_curve"),   # 수익률곡선
        fetch_fred(fred, "M2SL",       "m2"),            # M2
    ]

    df = pd.concat(series, axis=1)
    df = df[df.index >= START_DATE].sort_index()

    # 파생변수
    df["fed_rate_chg"] = df["fed_rate"].diff()   # 금리변화율

    print(f"\n  수집 완료: {df.index[0].date()} ~ {df.index[-1].date()} "
          f"| {len(df)}개월 | {df.shape[1]}개 변수")

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        print(f"  결측치:\n{missing.to_string()}")

    return df

# ============================================================
# 3. 누적 수익률 계산
# ============================================================

def calc_cumulative_returns(df: pd.DataFrame, window: int = 36) -> dict:
    results = {}
    for label, (start, end) in EASING_EVENTS.items():
        idx = df.index[df.index >= start]
        if len(idx) == 0:
            continue
        start_dt = idx[0]
        end_dt   = pd.Timestamp(end) if end else df.index[-1]
        sub      = df.loc[start_dt:end_dt].iloc[:window]

        cum_ret = {}
        for col in ASSETS:
            if col in sub.columns and sub[col].notna().any():
                base = sub[col].dropna().iloc[0]
                cum_ret[col] = ((sub[col] / base) - 1) * 100

        results[label] = {"sub": sub, "returns": cum_ret}
    return results

# ============================================================
# 4. 피크 시차 분석
# ============================================================

def calc_peak_lag(df: pd.DataFrame, window: int = 36) -> pd.DataFrame:
    rows = []
    for ev_label, (start, end) in EASING_EVENTS.items():
        idx = df.index[df.index >= start]
        if len(idx) == 0:
            continue
        start_dt = idx[0]
        end_dt   = pd.Timestamp(end) if end else df.index[-1]
        sub      = df.loc[start_dt:end_dt].iloc[:window]

        row = {"이벤트": ev_label}
        for col, label in ASSETS.items():
            if col not in sub.columns:
                continue
            s = sub[col].dropna()
            if len(s) < 2:
                continue
            base   = s.iloc[0]
            pct    = (s / base - 1) * 100
            peak_i = int(np.argmax(pct.values))
            row[f"{label}_피크월"] = peak_i
            row[f"{label}_상승폭"] = round(float(pct.iloc[peak_i]), 1)
        rows.append(row)

    return pd.DataFrame(rows).set_index("이벤트")

# ============================================================
# 5. 시각화
# ============================================================

def plot_cumulative(cum_data: dict):
    events = list(cum_data.keys())
    n = len(events)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for ax, ev in zip(axes[:n], events):
        rets = cum_data[ev]["returns"]
        for col, label in ASSETS.items():
            if col in rets:
                s = rets[col].dropna()
                ax.plot(range(len(s)), s.values,
                        label=label, color=COLORS[col], linewidth=2.2)
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.set_title(f"{ev} 이후 자산별 누적 수익률", fontsize=12, fontweight="bold")
        ax.set_xlabel("경과 월수")
        ax.set_ylabel("누적 수익률 (%)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("통화 완화 이벤트별 자산군 누적 상승폭",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/01_cumulative_returns.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  저장: {path}")


def plot_bubble(peak_df: pd.DataFrame):
    asset_labels = list(ASSETS.values())
    events       = peak_df.index.tolist()
    x_pos = {l: i for i, l in enumerate(asset_labels)}

    fig, ax = plt.subplots(figsize=(14, 6))

    for y_idx, ev in enumerate(events):
        order = []
        for col, label in ASSETS.items():
            mc = f"{label}_피크월"
            vc = f"{label}_상승폭"
            if mc in peak_df.columns:
                order.append((peak_df.loc[ev, mc],
                               label, peak_df.loc[ev, vc], col))
        order.sort(key=lambda x: x[0])

        prev_x = None
        for peak_m, label, peak_v, col in order:
            x    = x_pos[label]
            size = max(abs(peak_v) * 10, 80)
            ax.scatter(x, y_idx, s=size, color=COLORS[col],
                       alpha=0.8, edgecolors="white", lw=1.5, zorder=3)
            ax.text(x, y_idx + 0.22,
                    f"+{int(peak_m)}개월\n{peak_v:+.0f}%",
                    ha="center", va="bottom", fontsize=8,
                    color=COLORS[col], fontweight="bold")
            if prev_x is not None:
                ax.annotate("", xy=(x - 0.08, y_idx),
                            xytext=(prev_x + 0.08, y_idx),
                            arrowprops=dict(arrowstyle="->",
                                            color="#888", lw=1.8))
            prev_x = x

    ax.set_xticks(range(len(asset_labels)))
    ax.set_xticklabels(asset_labels, fontsize=12)
    ax.set_yticks(range(len(events)))
    ax.set_yticklabels(events, fontsize=11)
    ax.set_title("자산군 전이 경로 — 피크 도달 시차 & 상승폭",
                 fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.2)
    ax.set_ylim(-0.7, len(events) - 0.3)

    patches = [mpatches.Patch(color=COLORS[c], label=l)
               for c, l in ASSETS.items()]
    ax.legend(handles=patches, loc="upper right", fontsize=9)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/02_transmission_bubble.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  저장: {path}")


def plot_heatmap(peak_df: pd.DataFrame):
    cols = [f"{l}_상승폭" for l in ASSETS.values()
            if f"{l}_상승폭" in peak_df.columns]
    heat = peak_df[cols].copy().astype(float)
    heat.columns = [c.replace("_상승폭", "") for c in cols]

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(heat, annot=True, fmt=".1f", cmap="RdYlGn",
                center=0, linewidths=0.5, ax=ax,
                annot_kws={"size": 11, "weight": "bold"})
    ax.set_title("완화 이벤트별 자산군 피크 상승폭 (%) 히트맵",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/03_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  저장: {path}")


def plot_flow(peak_df: pd.DataFrame):
    avg_peak = {}
    for col, label in ASSETS.items():
        mc = f"{label}_피크월"
        vc = f"{label}_상승폭"
        if mc in peak_df.columns:
            avg_peak[label] = (peak_df[mc].mean(), peak_df[vc].mean(), col)

    sorted_assets = sorted(avg_peak.items(), key=lambda x: x[1][0])

    fig, ax = plt.subplots(figsize=(15, 3.5))
    ax.set_xlim(-0.5, len(sorted_assets) - 0.5)
    ax.set_ylim(-0.5, 1.8)
    ax.axis("off")

    for i, (label, (avg_m, avg_v, col_key)) in enumerate(sorted_assets):
        color = COLORS[col_key]
        rect  = mpatches.FancyBboxPatch(
            (i - 0.38, 0.5), 0.76, 0.9,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor="white",
            linewidth=2, alpha=0.88)
        ax.add_patch(rect)

        ax.text(i, 0.95, label, ha="center", va="center",
                fontsize=13, fontweight="bold", color="white")
        ax.text(i, 0.68, f"평균 +{avg_m:.0f}개월",
                ha="center", va="center", fontsize=9, color="white")
        ax.text(i, 0.42, f"{avg_v:+.1f}%",
                ha="center", va="center", fontsize=11,
                color=color, fontweight="bold")

        if i < len(sorted_assets) - 1:
            ax.annotate("", xy=(i + 0.43, 0.95),
                        xytext=(i + 0.40, 0.95),
                        arrowprops=dict(arrowstyle="->",
                                        color="#555", lw=2.2))

    ax.set_title("통화 완화 이후 자산 가격 전이 순서 (평균 피크 도달 시차 기준)",
                 fontsize=14, fontweight="bold", y=1.08)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/04_flow_diagram.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  저장: {path}")

# ============================================================
# 6. 콘솔 요약
# ============================================================

def print_summary(peak_df: pd.DataFrame):
    print("\n" + "=" * 55)
    print("  자산 전이 경로 요약")
    print("=" * 55)

    avg = {}
    for col, label in ASSETS.items():
        mc = f"{label}_피크월"
        vc = f"{label}_상승폭"
        if mc in peak_df.columns:
            avg[label] = (peak_df[mc].mean(), peak_df[vc].mean())

    print(f"\n{'순위':<4} {'자산군':<10} {'평균 피크 시차':>14} {'평균 상승폭':>12}")
    print("-" * 46)
    for rank, (label, (m, v)) in enumerate(
            sorted(avg.items(), key=lambda x: x[1][0]), 1):
        print(f"  {rank}위   {label:<10}   {m:>8.1f}개월   {v:>+9.1f}%")

    path_str = " → ".join(
        l for l, _ in sorted(avg.items(), key=lambda x: x[1][0]))
    print(f"\n[추정 전이 경로]\n  {path_str}")
    print("=" * 55)

# ============================================================
# 7. 메인
# ============================================================

if __name__ == "__main__":
    print("=" * 55)
    print("  자산 전이 경로 분석 (v2)")
    print("=" * 55)

    df       = load_data()

    print("\n[2단계] 피크 시차 계산...")
    cum_data = calc_cumulative_returns(df, window=36)
    peak_df  = calc_peak_lag(df, window=36)

    print("\n[3단계] 시각화 출력...")
    plot_cumulative(cum_data)
    plot_bubble(peak_df)
    plot_heatmap(peak_df)
    plot_flow(peak_df)

    print_summary(peak_df)
    print(f"\n✅ 완료 — 차트 저장 위치: {OUTPUT_DIR}/")