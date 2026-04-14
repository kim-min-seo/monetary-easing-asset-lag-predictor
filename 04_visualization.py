# ============================================================
#  04_visualization.py — 시각화 (v6)
#  ★ v6 개선:
#  1. from 03_analysis import 버그 수정 (importlib 방식)
#  2. 사이클별 개별 이벤트 스터디 차트 추가
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import platform
import importlib.util
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C


def set_font():
    system = platform.system()
    if system == "Windows":
        plt.rcParams["font.family"] = "Malgun Gothic"
    elif system == "Darwin":
        plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False


def load_analysis_module():
    """★ v6 버그 수정: importlib로 03_analysis 로드"""
    spec = importlib.util.spec_from_file_location(
        "analysis",
        os.path.join(os.path.dirname(__file__), "03_analysis.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ──────────────────────────────────────────────
#  그랜저 히트맵
# ──────────────────────────────────────────────

def plot_granger_heatmap(lag_t, pval_t):
    set_font()
    if lag_t.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(22, 7))
    sns.heatmap(lag_t.astype(float), annot=True, fmt=".0f",
                cmap="YlOrRd", ax=axes[0],
                cbar_kws={"label": "최적 시차(월)"})
    axes[0].set_title(
        "그랜저 인과관계 최적 시차\n"
        "(v6: CaseShiller 2차 차분 + TIPS 스프레드)",
        fontsize=13, fontweight="bold")

    sns.heatmap(pval_t.astype(float), annot=True, fmt=".3f",
                cmap="RdYlGn_r", ax=axes[1], vmin=0, vmax=0.1,
                cbar_kws={"label": "p-value"})
    axes[1].set_title(
        "그랜저 인과관계 p-value\n(< 0.05 = 유의)",
        fontsize=13, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(C.FIG_DIR, "granger_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ 그랜저 히트맵 저장: {path}")


# ──────────────────────────────────────────────
#  IRF 차트
# ──────────────────────────────────────────────

def plot_irf(df, irf_obj, irf_results):
    if irf_obj is None or not irf_results:
        print("  ⚠️  IRF 데이터 없음, 건너뜀")
        return

    set_font()
    var_cols = [c for c in [
        "Real_Rate", "QE_Size", "M2_YoY", "TIPS_Spread",
        "Gold_LogReturn", "WTI_LogReturn",
        "SP500_LogReturn", "CaseShiller_LogReturn2",
        "CPI_LogReturn"
    ] if c in df.columns]

    if "Real_Rate" not in var_cols:
        return

    shock_idx = var_cols.index("Real_Rate")
    asset_map = {
        "Gold_LogReturn":         "금 (Gold)",
        "WTI_LogReturn":          "WTI 원유",
        "SP500_LogReturn":        "S&P500",
        "CaseShiller_LogReturn2": "부동산",
        "CPI_LogReturn":          "CPI",
    }
    assets_in = [(c, l) for c, l in asset_map.items() if c in var_cols]

    fig, axes = plt.subplots(len(assets_in), 1,
                             figsize=(14, 4*len(assets_in)),
                             sharex=True)
    if len(assets_in) == 1:
        axes = [axes]

    for ax, (col, label) in zip(axes, assets_in):
        resp_idx = var_cols.index(col)
        irf_vals = irf_obj.irfs[:, resp_idx, shock_idx]
        peak_m   = int(np.argmax(np.abs(irf_vals)))

        ax.plot(range(len(irf_vals)), irf_vals,
                color="steelblue", lw=2.5, label=label)
        ax.axhline(0, color="black", lw=1, ls="--")
        ax.axvline(peak_m, color="red", lw=1.5, ls=":", alpha=0.7)
        ax.set_ylabel(label); ax.grid(True, alpha=0.3)
        ax.set_title(f"{label} — 최대 반응: {peak_m}개월 후",
                     fontsize=10, fontweight="bold")

    axes[-1].set_xlabel("실질금리 충격 후 경과 개월")
    fig.suptitle(
        "IRF 충격반응함수: 실질금리 하락 → 각 자산 반응 (v6)\n"
        "빨간 점선: 최대 반응 시점",
        fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(C.FIG_DIR, "irf_realrate.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ IRF 차트 저장: {path}")


# ──────────────────────────────────────────────
#  이벤트 스터디 (★ v6: 사이클별 차트 추가)
# ──────────────────────────────────────────────

def plot_event_study(df, all_rets):
    set_font()
    if not all_rets:
        return

    colors = ["#f1c40f","#e67e22","#2ecc71","#e74c3c","#9b59b6"]
    labels = list(all_rets.keys())
    pre    = 6

    # 전체 평균 차트
    fig, ax = plt.subplots(figsize=(14, 7))
    for label, color in zip(labels, colors):
        rets = all_rets[label]
        if not rets:
            continue
        min_len = min(len(r) for r in rets)
        avg_ret = np.mean([r[:min_len] for r in rets], axis=0)
        t_axis  = range(-pre, min_len - pre)
        ax.plot(list(t_axis), avg_ret,
                label=label, color=color, lw=2.5)

    ax.axvline(0, color="black", lw=2, ls="--", label="금리인하 시작")
    ax.axhline(0, color="gray",  lw=1, ls=":")
    ax.set_xlabel("금리인하 후 경과 개월")
    ax.set_ylabel("누적 수익률 (%)")
    ax.set_title("이벤트 스터디: 금리인하 시점 기준 평균 반응 (v6)",
                 fontsize=13, fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(C.FIG_DIR, "event_study_avg.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ 이벤트 스터디 (평균) 저장: {path}")

    # ★ v6: 사이클별 개별 차트
    asset_cols = {
        "금 (Gold)":  "Gold_LogReturn",
        "WTI (원유)": "WTI_LogReturn",
        "S&P500":     "SP500_LogReturn",
        "부동산":      "CaseShiller_LogReturn",
        "CPI":        "CPI_LogReturn",
    }
    event_dates = []
    for start, _ in C.RATE_CUT_CYCLES:
        ts    = pd.Timestamp(start)
        valid = df.index[df.index >= ts]
        if len(valid) > 0:
            event_dates.append((start, valid[0]))

    window = 24
    fig, axes = plt.subplots(
        len(event_dates), 1,
        figsize=(14, 5*len(event_dates)),
        sharex=True
    )
    if len(event_dates) == 1:
        axes = [axes]

    for ax, (cycle_name, event_date) in zip(axes, event_dates):
        idx = df.index.get_loc(event_date)
        for (label, col), color in zip(asset_cols.items(), colors):
            if col not in df.columns:
                continue
            s_idx  = max(0, idx - pre)
            e_idx  = min(len(df), idx + window + 1)
            series = df[col].iloc[s_idx:e_idx]
            cumret = (1 + series).cumprod() - 1
            t_axis = range(-pre, len(cumret)-pre)
            ax.plot(list(t_axis), cumret.values*100,
                    label=label, color=color, lw=2)

        ax.axvline(0, color="black", lw=2, ls="--")
        ax.axhline(0, color="gray", lw=1, ls=":")
        ax.set_title(f"금리인하 {cycle_name} 시작",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("누적 수익률 (%)")

    axes[-1].set_xlabel("금리인하 후 경과 개월")
    fig.suptitle("사이클별 이벤트 스터디 (v6)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(C.FIG_DIR, "event_study_cycles.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ 이벤트 스터디 (사이클별) 저장: {path}")


# ──────────────────────────────────────────────
#  칸티용 전이 경로 맵
# ──────────────────────────────────────────────

def plot_cantillon_path(final_order):
    set_font()
    if not final_order:
        return

    node_pos = [
        (0.05, 0.50),
        (0.30, 0.85),
        (0.30, 0.55),
        (0.30, 0.25),
        (0.62, 0.70),
        (0.90, 0.50),
    ]
    label_map = {
        "금 (Gold)":  "금\n(Gold)",
        "WTI 원유":   "WTI\n(원유)",
        "S&P500":     "S&P500\n(주식)",
        "부동산":      "부동산\n(Case-Shiller)",
        "CPI":        "CPI\n(종착점)",
    }
    color_map = {
        "통화 완화\n환경":        "#3498db",
        "금\n(Gold)":             "#f1c40f",
        "WTI\n(원유)":            "#e67e22",
        "S&P500\n(주식)":         "#2ecc71",
        "부동산\n(Case-Shiller)": "#e74c3c",
        "CPI\n(종착점)":          "#9b59b6",
    }

    nodes = {"통화 완화\n환경": node_pos[0]}
    for i, (label, _) in enumerate(final_order[:5]):
        mapped = label_map.get(label, label)
        nodes[mapped] = node_pos[i+1]

    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_xlim(-0.05, 1.0); ax.set_ylim(-0.05, 1.0)
    ax.axis("off"); ax.set_facecolor("#f8f9fa")
    ax.set_title(
        "칸티용 효과(Cantillon Effect) 자산 가격 전이 경로\n"
        "★ 데이터 기반 실증 순서 (v6)",
        fontsize=14, fontweight="bold", pad=20)

    node_list = list(nodes.keys())
    for i in range(len(node_list)-1):
        src = node_list[i]; tgt = node_list[i+1]
        x0, y0 = nodes[src]; x1, y1 = nodes[tgt]
        lag_val = (final_order[i-1][1] if i > 0
                   else final_order[0][1])
        ax.annotate("", xy=(x1,y1), xytext=(x0,y0),
                    arrowprops=dict(arrowstyle="-|>",
                                   color="steelblue", lw=2.0,
                                   connectionstyle="arc3,rad=0.08"))
        ax.text((x0+x1)/2, (y0+y1)/2+0.02,
                f"t+{lag_val:.0f}개월",
                ha="center", va="bottom", fontsize=9,
                color="steelblue", fontweight="bold")

    for name, (x, y) in nodes.items():
        color = color_map.get(name, "lightskyblue")
        ax.add_patch(plt.Circle((x,y), 0.085,
                                color=color, ec="white",
                                lw=2, zorder=3, alpha=0.9))
        ax.text(x, y, name, ha="center", va="center",
                fontsize=8, fontweight="bold", zorder=4, color="white")

    plt.tight_layout()
    path = os.path.join(C.FIG_DIR, "cantillon_path.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ 칸티용 전이경로 맵 저장: {path}")


# ──────────────────────────────────────────────
#  완화 사이클 오버레이
# ──────────────────────────────────────────────

def plot_easing_overlay(df):
    set_font()
    assets   = ["Gold","WTI","SP500","CaseShiller"]
    colors_a = ["gold","orangered","green","brown"]
    assets_in = [a for a in assets if a in df.columns]

    fig, axes = plt.subplots(len(assets_in), 1,
                             figsize=(14, 4*len(assets_in)),
                             sharex=True)
    if len(assets_in) == 1:
        axes = [axes]

    for ax, asset, color in zip(axes, assets_in, colors_a):
        ax.plot(df.index, df[asset], color=color, lw=1.5, label=asset)
        for s, e in C.RATE_CUT_CYCLES:
            ax.axvspan(pd.Timestamp(s), pd.Timestamp(e),
                       alpha=0.15, color="steelblue")
        ax.set_ylabel(asset); ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")

    axes[0].set_title("금리인하 사이클(파란 음영) + 자산 가격 반응",
                      fontsize=13, fontweight="bold")
    axes[-1].set_xlabel("날짜")
    plt.tight_layout()
    path = os.path.join(C.FIG_DIR, "easing_cycle_overlay.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ 완화사이클 오버레이 저장: {path}")


# ──────────────────────────────────────────────
#  M2 대시보드
# ──────────────────────────────────────────────

def plot_m2_dashboard(df):
    pairs = [
        ("M2_YoY","Gold_YoY",        "M2 YoY vs 금 YoY"),
        ("M2_YoY","WTI_YoY",         "M2 YoY vs WTI YoY"),
        ("M2_YoY","SP500_YoY",       "M2 YoY vs S&P500 YoY"),
        ("M2_YoY","CaseShiller_YoY", "M2 YoY vs 부동산 YoY"),
    ]
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=[p[2] for p in pairs])
    pos = [(1,1),(1,2),(2,1),(2,2)]
    pal = [("royalblue","gold"),("royalblue","orangered"),
           ("royalblue","green"),("royalblue","brown")]

    for (src,tgt,_),(r,c),(c1,c2) in zip(pairs,pos,pal):
        if src in df.columns:
            fig.add_trace(go.Scatter(x=df.index,y=df[src],
                                     name=src,line=dict(color=c1)),
                          row=r,col=c)
        if tgt in df.columns:
            fig.add_trace(go.Scatter(x=df.index,y=df[tgt],
                                     name=tgt,line=dict(color=c2)),
                          row=r,col=c)

    fig.update_layout(title="M2 전년비 증가율 vs 자산 가격 반응 (v6)",
                      height=700, template="plotly_white")
    path = os.path.join(C.FIG_DIR, "m2_dashboard.html")
    fig.write_html(path)
    print(f"  ✓ M2 대시보드 저장: {path}")


# ──────────────────────────────────────────────
#  메인
# ──────────────────────────────────────────────

def main():
    print("\n[04] 시각화")

    proc_path = os.path.join(C.DATA_PROC_DIR, "processed_data.csv")
    if not os.path.exists(proc_path):
        print("  ⚠️  processed_data.csv 없음 → 02 먼저 실행")
        return
    df = pd.read_csv(proc_path, index_col=0, parse_dates=True)

    granger_path = os.path.join(C.RESULT_DIR, "granger_results.csv")
    order_path   = os.path.join(C.RESULT_DIR, "cantillon_order.csv")
    irf_path     = os.path.join(C.RESULT_DIR, "irf_results.csv")

    # 그랜저 히트맵
    if os.path.exists(granger_path):
        gr = pd.read_csv(granger_path)
        if not gr.empty:
            lag_t  = gr.pivot(index="cause", columns="effect",
                              values="best_lag")
            pval_t = gr.pivot(index="cause", columns="effect",
                              values="p_value")
            plot_granger_heatmap(lag_t, pval_t)

    # 칸티용 경로 맵
    final_order = []
    if os.path.exists(order_path):
        od = pd.read_csv(order_path)
        final_order = list(od.itertuples(index=False, name=None))
        plot_cantillon_path(final_order)

    # ★ v6: importlib로 03_analysis 로드
    analysis_mod = load_analysis_module()
    event_peaks, all_rets = analysis_mod.run_event_study(df)
    plot_event_study(df, all_rets)

    # 기타
    plot_easing_overlay(df)
    plot_m2_dashboard(df)

    print("\n  ✅ 시각화 완료")


if __name__ == "__main__":
    main()
