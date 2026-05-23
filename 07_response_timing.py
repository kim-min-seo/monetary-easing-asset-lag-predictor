# ============================================================
#  07_response_timing.py — 반응 타이밍 + QE 규모 분석 (★ v8a)
#
#  분석 초점 전환:
#   · 자산: "얼마나 올랐나(크기)" → "언제 시작/반감/정점인가(타이밍)"
#       - onset : 최종 peak의 10% 도달 = 상승 시작 시점
#       - half  : 50% 도달 = 반감 시점 (03 D3 half-peak 재사용)
#       - peak  : 누적 반응 최대 = 최대 상승 시점
#       - 반응 크기(peak_value)는 버블 크기 등 보조 정보로만 사용
#   · QE: "시작 시점" → "시작 시점 + 푼 돈의 양($조)"
#       - 각 완화 이벤트의 연준 대차대조표(WALCL) 증가액을 정량화
#       - 자산 타이밍과 같은 표·같은 그림에 함께 표현
#
#  입력 : data/processed/processed_data.csv (02 산출물, read-only)
#  산출 : outputs/results/{response_timing, qe_magnitude, qe_response_combined}.csv
#         outputs/figures/{timing_profile_*, qe_timeline_combined, qe_vs_response_bubble}.png
#
#  데이터 한계: WALCL은 2002-12부터 → 2001 이벤트는 QE 규모 NaN (graceful)
# ============================================================

import os
import sys
import warnings
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C


# ============================================================
#  한글 폰트
# ============================================================
def set_font():
    s = platform.system()
    if s == "Windows":
        plt.rcParams["font.family"] = "Malgun Gothic"
    elif s == "Darwin":
        plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False


# ============================================================
#  공통 헬퍼
# ============================================================
def _event_dates(df):
    """RATE_CUT_CYCLES의 각 start를 데이터 인덱스에 정렬된 실제 날짜로."""
    out = []
    for start, _ in C.RATE_CUT_CYCLES:
        ts = pd.Timestamp(start)
        valid = df.index[df.index >= ts]
        if len(valid):
            out.append(valid[0])
    return out


def _crossing(post, frac, peak_v):
    """|post|가 frac*|peak_v|에 처음 도달하는 인덱스.
    부호 인지 (03 half-peak와 동일 논리). 도달 못하면 peak 위치 반환."""
    thr = frac * abs(peak_v)
    hit = np.where(np.abs(post) >= thr)[0]
    return int(hit[0]) if len(hit) else int(np.argmax(np.abs(post)))


def _post_path(df, col, idx):
    """이벤트 시점(idx) 기준 중심화 누적수익률(%)의 사후 경로(t=0..)."""
    pre, win = C.EVENT_PRE_MONTHS, C.EVENT_POST_MONTHS
    s, e = max(0, idx - pre), min(len(df), idx + win + 1)
    series = df[col].iloc[s:e]
    if series.empty:
        return None
    log_cum = series.cumsum()
    epos = idx - s
    if epos >= len(log_cum):
        return None
    # ★ 03 C4와 동일 공식: exp(중심화 누적) - 1
    cumret = (np.exp(log_cum - log_cum.iloc[epos]) - 1) * 100
    return cumret.values[epos:]


# ============================================================
#  T1 — 이벤트별 자산 반응 타이밍 추출
# ============================================================
def extract_timing(df):
    """반환: per (event, asset) DataFrame.
    onset/half/peak month + 반응크기 + 상승소요(peak-onset) + 전이속도."""
    rows = []
    for ev in _event_dates(df):
        idx = df.index.get_loc(ev)
        for label, col in C.TIMING_ASSETS.items():
            if col not in df.columns:
                continue
            post = _post_path(df, col, idx)
            if post is None or len(post) < 2:
                continue

            peak_t = int(np.argmax(np.abs(post)))
            peak_v = float(post[peak_t])
            if peak_v == 0:
                continue

            onset_t = _crossing(post, C.ONSET_FRAC, peak_v)
            half_t  = _crossing(post, C.HALF_FRAC,  peak_v)
            dur     = max(0, peak_t - onset_t)
            vel     = peak_v / dur if dur > 0 else np.nan  # 전이 속도(%/월)

            rows.append({
                "event":       ev.strftime("%Y-%m"),
                "asset":       label,
                "onset_month": onset_t,
                "half_month":  half_t,
                "peak_month":  peak_t,
                "peak_value_pct": round(peak_v, 2),
                "rise_duration":  dur,
                "transmission_velocity": round(vel, 3) if vel == vel else np.nan,
            })

    timing_df = pd.DataFrame(rows)
    if not timing_df.empty:
        path = os.path.join(C.RESULT_DIR, "response_timing.csv")
        timing_df.to_csv(path, index=False)
        print(f"  ✓ 저장: {path} ({len(timing_df)}행)")
    return timing_df


# ============================================================
#  T2 — QE 규모 정량화 (WALCL 증가액)
# ============================================================
def quantify_qe(df):
    """각 이벤트의 '푼 돈의 양'을 $조 단위로."""
    if "Fed_Assets" not in df.columns:
        print("  ⚠️  Fed_Assets(WALCL) 없음 — QE 규모 분석 건너뜀")
        return pd.DataFrame()

    fa = df["Fed_Assets"]
    win = C.EVENT_POST_MONTHS
    rows = []

    for ev in _event_dates(df):
        idx = df.index.get_loc(ev)
        e   = min(len(df) - 1, idx + win)
        v0, v1 = fa.iloc[idx], fa.iloc[e]

        if pd.isna(v0) or pd.isna(v1) or v0 == 0:   # 2001 등 WALCL 부재
            vol = pct = avg = np.nan
        else:
            vol = (v1 - v0) / C.WALCL_TO_TRILLION    # $조
            pct = (v1 / v0 - 1) * 100
            avg = vol / max(1, (e - idx))            # 월평균 주입($조)

        rows.append({
            "event": ev.strftime("%Y-%m"),
            "qe_volume_trillion":      round(vol, 3) if vol == vol else np.nan,
            "qe_growth_pct":           round(pct, 1) if pct == pct else np.nan,
            "qe_avg_monthly_trillion": round(avg, 4) if avg == avg else np.nan,
        })

    qe_df = pd.DataFrame(rows)
    if not qe_df.empty:
        path = os.path.join(C.RESULT_DIR, "qe_magnitude.csv")
        qe_df.to_csv(path, index=False)
        print(f"  ✓ 저장: {path}")
    return qe_df


# ============================================================
#  T3 — 결합 테이블 (핵심 산출물)
# ============================================================
def combine(timing_df, qe_df):
    """타이밍 × QE규모를 event 키로 merge."""
    if timing_df.empty:
        return timing_df
    if qe_df.empty:
        out = timing_df.copy()
    else:
        out = timing_df.merge(qe_df, on="event", how="left")
    path = os.path.join(C.RESULT_DIR, "qe_response_combined.csv")
    out.to_csv(path, index=False)
    print(f"  ✓ 저장: {path} (분석 핵심 테이블)")
    return out


# ============================================================
#  T4 — 시각화 1: 자산별 상승 프로파일 (onset/half/peak 마커)
# ============================================================
def plot_timing_profiles(df):
    """5사이클 평균(일관 표본) 누적반응 곡선 + onset▲/half◆/peak★ 마커."""
    set_font()
    assets = list(C.TIMING_ASSETS.items())
    n = len(assets)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.0 * n))
    if n == 1:
        axes = [axes]

    event_dates = _event_dates(df)
    drew = False

    for ax, (label, col) in zip(axes, assets):
        if col not in df.columns:
            ax.set_visible(False)
            continue

        paths = []
        for ev in event_dates:
            idx = df.index.get_loc(ev)
            post = _post_path(df, col, idx)
            if post is not None and len(post) >= 2:
                paths.append(post)
        if not paths:
            ax.set_visible(False)
            continue

        # 일관 표본 (모든 사이클 가용 horizon만) 평균
        min_len = min(len(p) for p in paths)
        avg = np.mean([p[:min_len] for p in paths], axis=0)
        months = np.arange(min_len)

        peak_t = int(np.argmax(np.abs(avg)))
        peak_v = float(avg[peak_t])
        onset_t = _crossing(avg, C.ONSET_FRAC, peak_v)
        half_t  = _crossing(avg, C.HALF_FRAC,  peak_v)

        ax.plot(months, avg, color="#3477b6", lw=1.8, label="평균 누적반응(%)")
        ax.axhline(0, color="black", lw=0.7, ls="--")
        ax.scatter(onset_t, avg[onset_t], marker="^", s=110, color="#27ae60",
                   zorder=5, label=f"상승시작 {onset_t}M")
        ax.scatter(half_t, avg[half_t], marker="D", s=70, color="#f39c12",
                   zorder=5, label=f"반감 {half_t}M")
        ax.scatter(peak_t, avg[peak_t], marker="*", s=240, color="#e74c3c",
                   zorder=5, label=f"최대상승 {peak_t}M")
        ax.set_title(f"{label} — 상승 프로파일 (이벤트 후 개월)",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("이벤트 이후 개월"); ax.set_ylabel("누적반응(%)")
        ax.legend(fontsize=8, loc="best"); ax.grid(alpha=0.3)
        drew = True

    plt.tight_layout()
    path = os.path.join(C.FIG_DIR, "timing_profile_all.png")
    if drew:
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  ✓ 저장: {path}")
    plt.close()


# ============================================================
#  T5 — 시각화 2: QE 타임라인 + 주입액 + 자산 마커 (★ 핵심)
# ============================================================
def plot_qe_timeline(df, timing_df, qe_df):
    """하나의 그림에 WALCL 면적 + 사이클별 주입액($조) + 자산 onset/peak 마커."""
    set_font()
    if "Fed_Assets" not in df.columns:
        print("  ⚠️  Fed_Assets 없음 — QE 타임라인 건너뜀")
        return

    fa_t = df["Fed_Assets"] / C.WALCL_TO_TRILLION    # $조
    fig, ax = plt.subplots(figsize=(15, 7))

    ax.fill_between(df.index, fa_t, color="#d6e4f0", alpha=0.7,
                    label="연준 대차대조표 (WALCL, $조)")
    ax.plot(df.index, fa_t, color="#3477b6", lw=1.2)

    ymax = float(np.nanmax(fa_t.values)) if fa_t.notna().any() else 1.0

    # 사이클별 QE 주입액 음영 + 주석
    qe_lookup = {r["event"]: r for _, r in qe_df.iterrows()} if not qe_df.empty else {}
    for ev in _event_dates(df):
        key = ev.strftime("%Y-%m")
        end = ev + pd.DateOffset(months=C.EVENT_POST_MONTHS)
        ax.axvspan(ev, end, color="#f0c419", alpha=0.10)
        ax.axvline(ev, color="#b8860b", lw=0.8, ls=":")
        r = qe_lookup.get(key)
        if r is not None and r["qe_volume_trillion"] == r["qe_volume_trillion"]:
            ax.annotate(f"+${r['qe_volume_trillion']:.1f}T",
                        xy=(ev, ymax * 0.96), fontsize=9, color="#9c6f00",
                        fontweight="bold")
        else:
            ax.annotate("QE 데이터 없음", xy=(ev, ymax * 0.96),
                        fontsize=8, color="#888888")

    # 자산 onset(▲)/peak(★) 마커 — 절대 날짜 위치, 색상별
    colors = plt.cm.tab10.colors
    labels_seen = set()
    for i, (label, col) in enumerate(C.TIMING_ASSETS.items()):
        sub = timing_df[timing_df["asset"] == label]
        for _, r in sub.iterrows():
            ev = pd.Timestamp(r["event"])
            d_on = ev + pd.DateOffset(months=int(r["onset_month"]))
            d_pk = ev + pd.DateOffset(months=int(r["peak_month"]))
            try:
                y = fa_t.reindex([d_on], method="nearest").iloc[0]
            except Exception:
                y = ymax * 0.5
            lab_on = f"{label} 상승시작" if label not in labels_seen else None
            ax.scatter(d_on, y, marker="^", color=colors[i % 10], s=55,
                       zorder=5, edgecolor="white", linewidth=0.5, label=lab_on)
            ax.scatter(d_pk, y, marker="*", color=colors[i % 10], s=140,
                       zorder=6, edgecolor="white", linewidth=0.5)
            labels_seen.add(label)

    ax.set_title("QE 주입 규모와 자산 반응 타이밍  (▲=상승시작, ★=최대상승, 음영=완화 윈도우)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("연도"); ax.set_ylabel("대차대조표 규모 ($조)")
    ax.legend(loc="upper left", fontsize=8, ncol=2); ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(C.FIG_DIR, "qe_timeline_combined.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ 저장: {path}")


# ============================================================
#  T6 — 시각화 3: QE 규모 vs 반응 시차 버블
# ============================================================
def plot_qe_vs_response(combined_df):
    """x=QE 주입액($조), y=상승시작 시차(개월), 버블=|반응크기|, 색=자산."""
    set_font()
    if combined_df.empty or "qe_volume_trillion" not in combined_df.columns:
        print("  ⚠️  결합 데이터 부족 — 버블 차트 건너뜀")
        return

    data = combined_df.dropna(subset=["qe_volume_trillion"]).copy()
    if data.empty:
        print("  ⚠️  QE 규모 유효값 없음 — 버블 차트 건너뜀")
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    colors = {lab: plt.cm.tab10.colors[i % 10]
              for i, lab in enumerate(C.TIMING_ASSETS.keys())}

    for ax, (ycol, ttl) in zip(axes, [
        ("onset_month", "QE 규모 vs 상승 시작 시차"),
        ("peak_month",  "QE 규모 vs 최대 상승 시차"),
    ]):
        for lab in data["asset"].unique():
            d = data[data["asset"] == lab]
            ax.scatter(d["qe_volume_trillion"], d[ycol],
                       s=(d["peak_value_pct"].abs() * 12 + 30),
                       color=colors.get(lab, "gray"), alpha=0.7,
                       edgecolor="white", label=lab)
        ax.set_xlabel("QE 주입액 ($조)"); ax.set_ylabel(f"{ycol} (개월)")
        ax.set_title(ttl, fontsize=11, fontweight="bold")
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=8, loc="best", title="자산 (버블=반응크기)")

    plt.tight_layout()
    path = os.path.join(C.FIG_DIR, "qe_vs_response_bubble.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ 저장: {path}")


# ============================================================
#  콘솔 요약
# ============================================================
def print_summary(timing_df, qe_df):
    if timing_df.empty:
        print("  (타이밍 결과 없음)")
        return

    print(f"\n{'='*72}")
    print("  자산별 평균 반응 타이밍 (모든 이벤트 평균, 단위: 개월)")
    print(f"{'='*72}")
    print(f"  {'자산':12s} {'상승시작':>8} {'반감':>8} {'최대상승':>8} "
          f"{'소요':>6} {'속도(%/월)':>10}")
    print("  " + "-" * 64)
    g = timing_df.groupby("asset")
    for asset in C.TIMING_ASSETS.keys():
        if asset not in g.groups:
            continue
        sub = g.get_group(asset)
        print(f"  {asset:12s} {sub['onset_month'].mean():>8.1f} "
              f"{sub['half_month'].mean():>8.1f} {sub['peak_month'].mean():>8.1f} "
              f"{sub['rise_duration'].mean():>6.1f} "
              f"{sub['transmission_velocity'].mean():>10.3f}")

    if not qe_df.empty:
        print(f"\n{'='*72}")
        print("  이벤트별 QE 주입 규모")
        print(f"{'='*72}")
        print(f"  {'이벤트':10s} {'주입액($조)':>12} {'증가율(%)':>10} {'월평균($조)':>12}")
        print("  " + "-" * 50)
        for _, r in qe_df.iterrows():
            vol = f"{r['qe_volume_trillion']:.2f}" if r['qe_volume_trillion'] == r['qe_volume_trillion'] else "—"
            pct = f"{r['qe_growth_pct']:.1f}" if r['qe_growth_pct'] == r['qe_growth_pct'] else "—"
            avg = f"{r['qe_avg_monthly_trillion']:.3f}" if r['qe_avg_monthly_trillion'] == r['qe_avg_monthly_trillion'] else "—"
            print(f"  {r['event']:10s} {vol:>12} {pct:>10} {avg:>12}")


# ============================================================
#  메인
# ============================================================
def main():
    print("\n[07] 반응 타이밍 + QE 규모 분석 (v8a)")
    set_font()

    proc_path = os.path.join(C.DATA_PROC_DIR, "processed_data.csv")
    if not os.path.exists(proc_path):
        print("  ⚠️  processed_data.csv 없음 — 02 먼저 실행")
        return None
    df = pd.read_csv(proc_path, index_col=0, parse_dates=True)
    print(f"  데이터 로드: {df.shape}")

    timing_df = extract_timing(df)            # T1
    qe_df     = quantify_qe(df)               # T2
    combined  = combine(timing_df, qe_df)     # T3

    plot_timing_profiles(df)                  # T4
    plot_qe_timeline(df, timing_df, qe_df)    # T5
    plot_qe_vs_response(combined)             # T6

    print_summary(timing_df, qe_df)
    print("\n  ✓ 반응 타이밍 + QE 규모 분석 완료 (v8a)")
    return {"timing": timing_df, "qe": qe_df, "combined": combined}


if __name__ == "__main__":
    main()
