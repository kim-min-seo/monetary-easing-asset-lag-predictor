"""
viz.py — 시각화 일체 (v4 통합: style + ordering/timing/inference/event_windows/qe plots)
"""
from __future__ import annotations
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import config as C
import data as D
import analysis as A


# ======================================================================
# ← viz/style.py
# ======================================================================

_ROLE_COLORS = {
    "frontrun": "#6a51a3", "financial": "#2171b5", "metal": "#cb6e17",
    "real": "#41872c", "tail": "#b2182b", "na": "#777777",
}


def _register_korean_font():
    cands = []
    for pat in ("*NanumGothic*", "*NotoSansCJK*", "*NotoSansKR*", "*Malgun*", "*AppleGothic*"):
        cands += glob.glob(f"/usr/share/fonts/**/{pat}", recursive=True)
        cands += glob.glob(f"/usr/local/share/fonts/**/{pat}", recursive=True)
    for p in cands:
        try:
            fm.fontManager.addfont(p)
            return fm.FontProperties(fname=p).get_name()
        except Exception:
            continue
    have = {f.name for f in fm.fontManager.ttflist}
    for c in C.KOR_FONT_CANDIDATES:
        if c in have:
            return c
    return None


def apply_style():
    font = _register_korean_font()
    if font:
        plt.rcParams["font.family"] = font
    plt.rcParams.update({
        "axes.unicode_minus": False, "figure.dpi": C.DPI, "savefig.dpi": C.DPI,
        "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.25,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    return font


def role_color(role: str) -> str:
    return _ROLE_COLORS.get(role, "#777777")


def save(fig, name: str) -> str | None:
    try:
        C.FIG_DIR.mkdir(parents=True, exist_ok=True)
        path = C.FIG_DIR / f"{name}.png"
        fig.savefig(path)
        plt.close(fig)
        return str(path)
    except Exception as e:  # noqa
        try:
            plt.close(fig)
        except Exception:
            pass
        C.warn(f"{name} 저장 실패: {type(e).__name__}: {e}")
        return None


# ======================================================================
# ← viz/ordering_plots.py
# ======================================================================

def rank_matrix(rank_mat, W, title_prefix="이벤트별 자산 반응 순위", name="ordering_rank_matrix"):
    apply_style()
    # y축(자산)을 평균 순위로 정렬: 작을수록(이를수록) 위, 결측 자산은 아래로
    M = rank_mat.loc[rank_mat.mean(axis=1).sort_values(na_position="last").index]
    labels = [C.ASSET_LABELS.get(k, k) for k in M.index]
    data = M.values.astype(float)
    fig, ax = plt.subplots(figsize=(min(1.1 * M.shape[1] + 4, 14), 0.55 * M.shape[0] + 2))
    im = ax.imshow(data, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(M.shape[1])); ax.set_xticklabels(M.columns, rotation=45, ha="right")
    ax.set_yticks(range(M.shape[0])); ax.set_yticklabels(labels)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}" if v == int(v) else f"{v:.1f}",
                        ha="center", va="center", fontsize=9)
    Wt = "N/A" if W is None or np.isnan(W) else f"{W:.3f}"
    ax.set_title(f"{title_prefix} (1=가장 이름 · 위에서 아래로 평균순위) · Kendall's W = {Wt}",
                 fontsize=12, pad=12)
    fig.colorbar(im, ax=ax, label="순위", shrink=0.8)
    ax.grid(False)
    return save(fig, name)


def consistency(order_df, suffix="", name="ordering_consistency", title=None):
    apply_style()
    # v6: 평균순위 오름차순 배치 + y축 반전 제거 → 큰 순위 위, 작은 순위 아래
    d = order_df.sort_values("mean_rank", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(d) + 2))
    y = np.arange(len(d))
    colors = [role_color(r) for r in d["role"]]
    ax.barh(y, d["mean_rank"], xerr=d["std_rank"], color=colors, alpha=0.85,
            error_kw=dict(ecolor="#555", capsize=3))
    ax.set_yticks(y); ax.set_yticklabels(d["label"])   # 반전 없음: y=0(작은 순위) 하단
    ax.set_xlabel("평균 순위 (작을수록 먼저 반응 · 아래에 위치)")
    if title is None:
        mark = (" — 금리 인하 이벤트" if "rate" in suffix else
                " — 급격한 QE 구간" if ("program" in suffix or "surge" in suffix or "strong" in suffix) else "")
        title = "자산별 전이 순서 일관성 (평균 ± 표준편차)" + mark
    ax.set_title(title, fontsize=12)
    roles = list(dict.fromkeys(d["role"]))
    handles = [plt.Rectangle((0, 0), 1, 1, color=role_color(r)) for r in roles]
    ax.legend(handles, roles, loc="lower right", fontsize=8, title="role")
    return save(fig, name + suffix)


def lead_lag(rank_mat, name="ordering_lead_lag"):
    """자산 쌍 평균 순위차 히트맵(양수=행이 먼저)."""
    apply_style()
    o = A.mean_ordering(rank_mat).set_index("asset")
    keys = [k for k in C.ASSET_KEYS if k in o.index and not np.isnan(o.loc[k, "mean_rank"])]
    r = o.loc[keys, "mean_rank"].values
    D = r[None, :] - r[:, None]
    fig, ax = plt.subplots(figsize=(0.5 * len(keys) + 3, 0.5 * len(keys) + 2))
    im = ax.imshow(D, cmap="RdBu", aspect="auto", vmin=-np.nanmax(np.abs(D)), vmax=np.nanmax(np.abs(D)))
    labs = [C.ASSET_LABELS.get(k, k) for k in keys]
    ax.set_xticks(range(len(keys))); ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(keys))); ax.set_yticklabels(labs, fontsize=7)
    ax.set_title("Lead-Lag 순위차 (열−행, 양수=행이 먼저)", fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.8); ax.grid(False)
    return save(fig, name)


def chain_diagram(order_df, name="transmission_chain", title=None):
    apply_style()
    d = order_df.dropna(subset=["mean_rank"]).sort_values("mean_rank").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(min(1.0 * len(d) + 2, 14), 3.2))
    x = np.arange(len(d))
    colors = [role_color(r) for r in d["role"]]
    ax.plot(x, np.zeros_like(x), color="#bbb", lw=1.5, zorder=1)
    ax.scatter(x, np.zeros_like(x), s=440, c=colors, zorder=3, alpha=0.9)
    for i, row in d.iterrows():
        ax.annotate(row["label"], (i, 0), ha="center", va="center", fontsize=8, color="white", zorder=4)
        ax.annotate(f"{row['mean_rank']:.1f}", (i, 0.13), ha="center", fontsize=7, color="#444")
    ax.annotate("", xy=(len(d) - 0.6, 0), xytext=(-0.4, 0),
                arrowprops=dict(arrowstyle="->", color="#888", lw=1.2))
    ax.set_title(title or "평균 전이 사슬 (왼쪽=먼저 → 오른쪽=나중)", fontsize=12)
    ax.set_yticks([]); ax.set_xticks([]); ax.set_ylim(-0.5, 0.45); ax.grid(False)
    return save(fig, name)


# ======================================================================
# ← viz/timing_plots.py
# ======================================================================

def timing_profile(timing_event, event_label, name="timing_profile"):
    if timing_event is None or len(timing_event) == 0:
        C.warn(f"{name} 스킵: 빈 타이밍 입력"); return None
    apply_style()
    d = timing_event.dropna(subset=["onset_m"]).copy()
    d["label"] = d["asset"].map(C.ASSET_LABELS)
    d = d.sort_values("onset_m")
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(d) + 2))
    for i, (_, r) in enumerate(d.iterrows()):
        col = role_color(C.ASSETS.get(r["asset"], {}).get("role", "na"))
        peak = r["peak_m"] if not np.isnan(r["peak_m"]) else r["onset_m"]
        ax.plot([r["onset_m"], peak], [i, i], color=col, lw=4, alpha=0.6, solid_capstyle="round")
        ax.scatter(r["onset_m"], i, marker="^", s=55, color=col, zorder=3)
        ax.scatter(peak, i, marker="*", s=90, color=col, zorder=3)
        if not np.isnan(r["half_m"]):
            ax.scatter(r["half_m"], i, marker="|", s=80, color="#333", zorder=4)
    ax.set_yticks(range(len(d))); ax.set_yticklabels(d["label"]); ax.invert_yaxis()
    ax.set_xlabel("이벤트 기준 상대월 (▲onset  |half  ★peak)")
    ax.set_title(f"타이밍 프로파일 — {event_label}", fontsize=12)
    return save(fig, name)


def walcl_timeline(walcl, events_qe, events_rate, timing_tbl=None, suffix="",
                   programs=None, name="walcl_timeline"):
    apply_style()
    fig, ax = plt.subplots(figsize=C.FIGSIZE_WIDE)
    w = walcl.dropna()
    ax.fill_between(w.index, w.values, color="#9ecae1", alpha=0.5, label="WALCL($조)")
    ax.plot(w.index, w.values, color="#2171b5", lw=1.2)
    ymax = ax.get_ylim()[1]
    # 급격한 QE 구간 음영
    if programs is not None and len(programs):
        for _, p in programs.iterrows():
            s = pd.Timestamp(p["event_date"]); e = pd.Timestamp(p["end_date"])
            ax.axvspan(s, e, color="#fdae6b", alpha=0.35, zorder=0)
            ax.text(s, ymax * 0.06, f" {p['label']} +${p['dWALCL']:.1f}T",
                    fontsize=8, color="#7a3d00", va="bottom")
    for _, e in events_qe.iterrows():
        d = pd.Timestamp(e["event_date"])
        ax.axvline(d, color="#cb6e17", ls=":", lw=1, alpha=0.7)
        tag = f"+${e['dWALCL']:.1f}T" if "dWALCL" in e and not pd.isna(e["dWALCL"]) else "QE"
        ax.text(d, ymax * 0.96, tag, fontsize=8, color="#cb6e17", ha="left")
    for _, e in events_rate.iterrows():
        d = pd.Timestamp(e["event_date"])
        ax.axvline(d, color="#6a51a3", ls="--", lw=1, alpha=0.6)
        ax.text(d, ymax * 0.86, "CUT", rotation=90, fontsize=7, color="#6a51a3", va="top")
    if timing_tbl is not None and len(timing_tbl):
        for _, r in timing_tbl.iterrows():
            ed = pd.Timestamp(r["event_date"])
            col = role_color(C.ASSETS.get(r["asset"], {}).get("role", "na"))
            for mcol, mk, sz in (("onset_m", "^", 22), ("peak_m", "*", 40)):
                if not np.isnan(r.get(mcol, np.nan)):
                    dt = ed + pd.DateOffset(months=int(r[mcol]))
                    if dt in w.index:
                        ax.scatter(dt, w.loc[dt], marker=mk, s=sz, color=col, alpha=0.7, zorder=3)
    ax.set_ylabel("대차대조표($조)"); ax.set_xlabel("연도")
    _mode = (" — 급격한 QE 구간" if "surge" in suffix else
             " — 금리 인하 이벤트" if "rate" in suffix else "")
    ax.set_title("WALCL 주입 규모 + 자산 반응 타이밍 (▲onset ★peak)" + _mode, fontsize=13)
    # WALCL·마커 범례
    leg1 = ax.legend(loc="upper left", fontsize=8)
    ax.add_artist(leg1)
    # v6: 자산 아이콘 범례 — 마커가 찍힌 자산을 role 색으로 표기
    if timing_tbl is not None and len(timing_tbl):
        seen, handles = set(), []
        for a in timing_tbl["asset"]:
            if a in seen:
                continue
            seen.add(a)
            role = C.ASSETS.get(a, {}).get("role", "na")
            handles.append(plt.Line2D([0], [0], marker="^", linestyle="",
                                      color=role_color(role), markersize=7,
                                      label=C.ASSET_LABELS.get(a, a)))
        if handles:
            ax.legend(handles=handles, loc="upper right", fontsize=7, ncol=2,
                      title="자산(▲onset/★peak)", framealpha=0.85)
    return save(fig, name + suffix)


def magnitude_bubble(qe_events, timing_qe, name="magnitude_bubble"):
    if qe_events is None or len(qe_events) == 0 or timing_qe is None or len(timing_qe) == 0:
        C.warn(f"{name} 스킵: 빈 입력"); return None
    """이벤트별 규모(ΔWALCL) 버블 + 평균 onset."""
    apply_style()
    g = timing_qe.groupby("event_date")["onset_m"].mean().reset_index()
    mag = qe_events[["event_date", "dWALCL"]] if "dWALCL" in qe_events else \
        qe_events[["event_date", "magnitude_T"]].rename(columns={"magnitude_T": "dWALCL"})
    g = g.merge(mag, on="event_date", how="left")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(g["event_date"], g["onset_m"], s=g["dWALCL"].abs() * 250 + 40,
               color="#cb6e17", alpha=0.6, edgecolor="#7a3d00")
    for _, r in g.iterrows():
        ax.annotate(f"+${r['dWALCL']:.1f}T", (r["event_date"], r["onset_m"]),
                    fontsize=8, ha="center", va="bottom")
    ax.set_ylabel("평균 onset (개월)"); ax.set_xlabel("QE 이벤트")
    ax.set_title("QE 규모(버블 크기) vs 평균 반응 시점", fontsize=12)
    return save(fig, name)


# ======================================================================
# ← viz/inference_plots.py
# ======================================================================

def lp_panels(curves, channel="qe", suffix="", ncol=4, name="lp_panels"):
    apply_style()
    sub = curves[curves["channel"] == channel]
    assets = [k for k in C.ASSET_KEYS if k in sub["asset"].unique()]
    nrow = int(np.ceil(len(assets) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.4 * nrow), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, k in zip(axes, assets):
        g = sub[sub["asset"] == k].sort_values("h")
        col = role_color(C.ASSETS.get(k, {}).get("role", "na"))
        ax.plot(g["h"], g["beta"], color=col, lw=1.4)
        ax.fill_between(g["h"], g["lo"], g["hi"], color=col, alpha=0.18)
        ax.axhline(0, color="#999", lw=0.7)
        ax.set_title(C.ASSET_LABELS.get(k, k), fontsize=9); ax.tick_params(labelsize=7)
    for ax in axes[len(assets):]:
        ax.axis("off")
    fig.suptitle(f"Local Projection 충격반응 — {channel} 채널", fontsize=12)
    fig.supxlabel("지평 h (개월)", fontsize=9)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    return save(fig, name + suffix)


def lp_ordering(scores, channel_label="QE", suffix="", name="lp_ordering"):
    apply_style()
    d = scores.dropna(subset=["onset_h"]).copy()
    d["label"] = d["asset"].map(C.ASSET_LABELS)
    d = d.sort_values("onset_h")
    fig, ax = plt.subplots(figsize=(8.5, 0.45 * len(d) + 2))
    colors = [role_color(C.ASSETS.get(k, {}).get("role", "na")) for k in d["asset"]]
    y = np.arange(len(d))
    ax.barh(y, d["onset_h"], color=colors, alpha=0.85)
    ax.set_yticks(y); ax.set_yticklabels(d["label"]); ax.invert_yaxis()
    ax.set_xlabel("onset 지평 h (개월, 작을수록 먼저)")
    ax.set_title(f"LP 기반 전이 순서 — {channel_label} 채널 (peak월·β·순위 표기)", fontsize=12)
    # v6: 수치 주석(peak월·peak β·순위)
    for yi, (_, r) in zip(y, d.iterrows()):
        pk = "" if pd.isna(r.get("peak_h")) else f"peak {int(r['peak_h'])}m"
        bt = "" if pd.isna(r.get("peak_beta")) else f", β={r['peak_beta']:.2f}"
        ax.text(r["onset_h"], yi, f"  #{yi+1} · {pk}{bt}", va="center", fontsize=7, color="#333")
    return save(fig, name + suffix)


def lp_concordance(conc: dict, channel_label="QE", suffix="", name="lp_concordance"):
    """LP 순서 vs 이벤트-스터디 순서 일치도 수치화 (Spearman ρ + 자산별 Δrank)."""
    t = conc.get("table")
    if t is None or len(t) == 0:
        C.warn(f"{name} 스킵: 빈 입력"); return None
    apply_style()
    d = t.copy()
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(d) + 2))
    y = np.arange(len(d))
    colors = ["#2171b5" if v <= 0 else "#cb6e17" for v in d["delta"]]
    ax.barh(y, d["delta"], color=colors, alpha=0.85)
    ax.axvline(0, color="#666", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(d["label"]); ax.invert_yaxis()
    ax.set_xlabel("Δrank = LP순위 − 이벤트순위 (0에 가까울수록 일치)")
    rho = conc.get("rho")
    rtxt = "n/a" if rho is None or rho != rho else f"{rho:+.3f}"
    ax.set_title(f"LP↔이벤트 순서 일치도 — {channel_label} (Spearman ρ={rtxt}, n={conc.get('n')})",
                 fontsize=12)
    return save(fig, name + suffix)


def lp_ordering_table(table_df, channel_label="QE", suffix="", name="lp_ordering_table"):
    """LP 순서 수치 표(그림): asset/onset/peak월/β/순위."""
    if table_df is None or len(table_df) == 0:
        C.warn(f"{name} 스킵: 빈 입력"); return None
    apply_style()
    d = table_df.sort_values("lp_rank").copy()
    cols = ["label", "onset_h", "peak_h", "peak_beta", "lp_rank"]
    headers = ["자산", "onset(월)", "peak(월)", "peak β", "순위"]
    fig, ax = plt.subplots(figsize=(7.5, 0.4 * len(d) + 1.2))
    ax.axis("off")
    cell = [[f"{r[c]:.2f}" if isinstance(r[c], float) else str(r[c]) for c in cols]
            for _, r in d.iterrows()]
    tbl = ax.table(cellText=cell, colLabels=headers, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.3)
    ax.set_title(f"LP 순서 수치표 — {channel_label} 채널", fontsize=12, pad=10)
    return save(fig, name + suffix)


def method_compare(compare_df, name="method_compare"):
    """LP·event-study·VAR 순위 Spearman 막대."""
    apply_style()
    d = compare_df.copy()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(d["pair"], d["spearman"], color="#4576b5", alpha=0.85)
    for i, v in enumerate(d["spearman"]):
        if not np.isnan(v):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_ylabel("Spearman ρ"); ax.set_ylim(-1.05, 1.15)
    ax.set_title("방법 간 순서 일치도(LP·이벤트스터디·VAR)", fontsize=11)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=8)
    return save(fig, name)


def channel_compare(channel_df, name="channel_comparison",
                    left_label="금리 인하 이벤트", right_label="급격한 QE 구간 +3년",
                    title="채널별 전이 순서 비교 (QE 3년 장 vs 금리 인하 이벤트)"):
    """채널 순서 슬로프그래프: 좌=rank_rate, 우=rank_qe (이벤트-스터디 순서)."""
    apply_style()
    d = channel_df.dropna(subset=["rank_rate", "rank_qe"])
    fig, ax = plt.subplots(figsize=(7, 8))
    for _, r in d.iterrows():
        col = role_color(r["role"])
        ax.plot([0, 1], [r["rank_rate"], r["rank_qe"]], "-o", color=col, alpha=0.8, lw=1.4)
        ax.text(-0.03, r["rank_rate"], r["label"], ha="right", va="center", fontsize=8)
        ax.text(1.03, r["rank_qe"], r["label"], ha="left", va="center", fontsize=8)
    ax.set_xticks([0, 1]); ax.set_xticklabels([left_label, right_label])
    ax.invert_yaxis(); ax.set_ylabel("전이 순서 순위 (낮을수록 먼저)")
    ax.set_title(title, fontsize=12); ax.set_xlim(-0.4, 1.4)
    return save(fig, name)


# ======================================================================
# ← viz/event_windows.py
# ======================================================================

def overlay_per_event(loglev, event_date, label=None, pre=C.EVENT_PRE_MONTHS,
                      post=C.EVENT_POST_MONTHS, name="event_overlay"):
    apply_style()
    wr = D.window_response(loglev, event_date, pre=pre, post=post)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for k in wr.columns:
        col = role_color(C.ASSETS.get(k, {}).get("role", "na"))
        ax.plot(wr.index, wr[k] * 100, color=col, alpha=0.75, lw=1.2, label=C.ASSET_LABELS.get(k, k))
    ax.axvline(0, color="#333", lw=1); ax.axhline(0, color="#aaa", lw=0.8)
    ax.set_xlabel("이벤트 기준 상대월"); ax.set_ylabel("누적 반응 (%)")
    ax.set_title(f"이벤트 {label or pd.Timestamp(event_date):%Y-%m} ±윈도우 누적 반응", fontsize=12)
    ax.legend(ncol=2, fontsize=7, loc="upper left")
    return save(fig, name)


def small_multiples(loglev, evlist, pre=24, post=C.EVENT_POST_MONTHS, name="event_small_multiples"):
    if evlist is None or len(evlist) == 0:
        C.warn(f"{name} 스킵: 이벤트 없음"); return None
    apply_style()
    dates = list(evlist["event_date"])
    ncol = 3
    nrow = int(np.ceil(len(dates) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.8 * nrow), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, d in zip(axes, dates):
        wr = D.window_response(loglev, d, pre=pre, post=post)
        for k in wr.columns:
            col = role_color(C.ASSETS.get(k, {}).get("role", "na"))
            ax.plot(wr.index, wr[k] * 100, color=col, alpha=0.6, lw=0.9)
        ax.axvline(0, color="#333", lw=0.8); ax.axhline(0, color="#ccc", lw=0.6)
        ax.set_title(f"{pd.Timestamp(d):%Y-%m}", fontsize=9); ax.tick_params(labelsize=7)
    for ax in axes[len(dates):]:
        ax.axis("off")
    fig.suptitle("이벤트별 ±윈도우 누적 반응 (small multiples)", fontsize=12)
    fig.supylabel("누적 반응 (%)", fontsize=9); fig.tight_layout(rect=(0, 0, 1, 0.97))
    return save(fig, name)


def event_study_average(loglev, evlist, pre=24, post=C.EVENT_POST_MONTHS, name="event_study_average"):
    if evlist is None or len(evlist) == 0:
        C.warn(f"{name} 스킵: 이벤트 없음"); return None
    """이벤트 평균 누적 반응(자산별)."""
    apply_style()
    dates = list(evlist["event_date"])
    stack = {k: [] for k in loglev.columns}
    for d in dates:
        wr = D.window_response(loglev, d, pre=pre, post=post)
        for k in wr.columns:
            stack[k].append(wr[k])
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for k, lst in stack.items():
        if not lst:
            continue
        avg = pd.concat(lst, axis=1).mean(axis=1)
        col = role_color(C.ASSETS.get(k, {}).get("role", "na"))
        ax.plot(avg.index, avg * 100, color=col, alpha=0.8, lw=1.3, label=C.ASSET_LABELS.get(k, k))
    ax.axvline(0, color="#333", lw=1); ax.axhline(0, color="#aaa", lw=0.8)
    ax.set_xlabel("이벤트 기준 상대월"); ax.set_ylabel("평균 누적 반응 (%)")
    ax.set_title(f"이벤트 평균 누적 반응 (n={len(dates)})", fontsize=12)
    ax.legend(ncol=2, fontsize=7, loc="upper left")
    return save(fig, name)


# ======================================================================
# ← viz/qe_plots.py
# ======================================================================

def w_comparison(comp_df, name="qe_W_comparison"):
    apply_style()
    d = comp_df.copy()
    colors = {"mixed(all)": "#999999", "qe_surge_3y": "#2171b5", "qe_surge_1p5y": "#6baed6", "rate_cuts": "#cb6e17"}
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(d["set"], d["kendalls_W"], color=[colors.get(s, "#777") for s in d["set"]], alpha=0.9)
    for b, wv, n in zip(bars, d["kendalls_W"], d["n_events"]):
        if not np.isnan(wv):
            ax.text(b.get_x() + b.get_width() / 2, wv + 0.008, f"W={wv:.3f}\n(n={n})", ha="center", fontsize=9)
    ax.set_ylabel("Kendall's W"); ax.set_title("이벤트셋 분리에 따른 순위 일치도 (H1)", fontsize=12)
    top = d["kendalls_W"].max(skipna=True)
    ax.set_ylim(0, max(0.5, (top * 1.3) if top == top else 0.5))
    return save(fig, name)


def qe_rank_matrix(rank_mat_qe, W_qe):
    """→ ordering_rank_matrix_qe.png"""
    return rank_matrix(rank_mat_qe, W_qe,
                          title_prefix="[QE-only] 이벤트별 자산 반응 순위",
                          name="ordering_rank_matrix_qe")


def magnitude_scatter(qe_events, timing_qe, name="qe_magnitude_scatter"):
    if qe_events is None or len(qe_events) == 0 or timing_qe is None or len(timing_qe) == 0:
        C.warn(f"{name} 스킵: 빈 입력"); return None
    apply_style()
    mag = qe_events[["event_date", "dWALCL"]] if "dWALCL" in qe_events else \
        qe_events[["event_date", "magnitude_T"]].rename(columns={"magnitude_T": "dWALCL"})
    df = timing_qe.merge(mag, on="event_date", how="left").dropna(subset=["dWALCL", "peak_resp"])
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for role in C.ROLE_ORDER:
        sub = df[df["asset"].map(lambda k: C.ASSETS.get(k, {}).get("role") == role)]
        if len(sub):
            ax.scatter(sub["dWALCL"], sub["peak_resp"].abs() * 100,
                       color=role_color(role), label=role, alpha=0.7, s=40)
    ax.set_xlabel("QE 국면 규모 ΔWALCL ($조)"); ax.set_ylabel("자산 peak 반응 |%|")
    ax.set_title("주입 규모 vs 반응 크기 (H3)", fontsize=12)
    ax.legend(fontsize=8, title="role")
    return save(fig, name)


# ======================================================================
# v6: 그림 설명 문서 (정적) — outputs/figures_guide.md
# ======================================================================
def write_figures_guide(path=None) -> str:
    """각 그림이 '무엇을(수치 속성) 어떻게(시각화 기법)' 보여주는지 고정 설명을 기록."""
    path = path or (C.OUT_DIR / "figures_guide.md")
    md = """# 그림 설명서 (figures_guide)

> 이 문서는 cantillon-sequencer 가 생성하는 그림의 **의미(수치 속성)** 와 **시각화 기법** 을 설명하는 정적 안내서입니다. 값이 아니라 *읽는 법* 을 다룹니다.
> 접미사 규칙: `_rate_cuts`(금리 인하 이벤트) · `_qe_surge_3y`/`_qe_surge_1p5y`(급격한 QE 구간 +3년/+1.5년) · `_strong_from{3y|1p5y}`(순서성 강한 자산, 출처 구간) · `_qe`/`_rate`(LP QE/금리 충격 채널).

## 시계열·이벤트
- **walcl_timeline[_*]** — 수치: 연준 대차대조표(WALCL, $조) 월별 수준 + 자산별 onset(▲)·peak(★) 시점. 기법: 면적+선 그래프 위에 이벤트 세로선, 자산 마커(role 색), 자산 범례. 읽는 법: 마커가 왼쪽일수록 빠른 반응.
- **event_small_multiples / event_study_average** — 수치: 이벤트 정렬 후 자산별 중심화 누적 로그반응(%). 기법: 소형 다중 패널 / 평균±밴드. 읽는 법: t=0 이후 곡선이 먼저 상승하는 자산이 선행.

## 순서(ordering)
- **ordering_rank_matrix[_*]** — 수치: 이벤트×자산 onset 기반 순위(1=가장 이름). 기법: 히트맵(녹→적). 읽는 법: 행이 전반적으로 녹색이면 일관되게 선행.
- **ordering_consistency[_*]** — 수치: 자산별 평균 순위 ± 표준편차. 기법: 가로 막대(평균순위 정렬, 큰 순위가 위·작은 순위가 아래), 오차막대=일관성. 읽는 법: 아래쪽일수록 먼저, 막대 짧을수록 일관.
- **ordering_lead_lag** — 수치: 자산쌍 선후 빈도. 기법: 매트릭스. 읽는 법: 한 자산이 다른 자산보다 자주 앞서는지.
- **transmission_chain[_*]** — 수치: 평균 전이 순서(급격한 QE 구간·금리 인하 이벤트·순서성 강한 자산 각각). 기법: 노드-화살표 사슬(왼쪽=먼저). 읽는 법: 칸티용 사슬(금융→금속→실물→끝단) 부합 여부.

## 규모·타이밍
- **qe_W_comparison / rate_W_comparison** — 수치: 이벤트셋별 Kendall's W(0~1, 1=완전 일치). 기법: 막대. 읽는 법: 분리/구간화가 순서를 선명하게 하는지.
- **\\*_magnitude_scatter** — 수치: 이벤트 규모(ΔWALCL 또는 인하 bp) ↔ 반응(정점·onset). 기법: 산점+추세. 읽는 법: 규모 클수록 빠르고 큰 반응인지(H3).
- **magnitude_bubble** — 수치: 규모(버블 크기)–평균 onset. 기법: 버블. 읽는 법: 큰 버블이 위(느림)/아래(빠름).
- **timing_profile[_*]** — 수치: 자산별 onset(▲)·half(|)·peak(★) 상대월. 기법: 자산별 수평 구간. 읽는 법: peak 는 onset 이후로 보장(역행 없음).

## 추론(LP/VAR)
- **lp_irf_{qe,rate}** — 수치: 충격 1단위에 대한 자산별 누적 반응 β(h) + 신뢰구간. 기법: 소형 다중 IRF 패널. 읽는 법: 먼저·크게 반응하는 자산.
- **lp_ordering_{qe,rate}** — 수치: LP onset 지평 + peak월·peak β·순위(주석). 기법: 가로 막대 + 수치 주석. 읽는 법: LP 기준 선행 순서.
- **lp_concordance_{qe,rate}** — 수치: LP 순서 vs 이벤트 순서 Spearman ρ + 자산별 Δrank. 기법: Δrank 막대(0=일치). 읽는 법: ρ 높고 막대 짧을수록 두 방법 합치.
- **lp_ordering_table_{qe,rate}** — 수치: 자산별 onset·peak월·β·순위 표. 기법: 표형 그림. 읽는 법: 수치 직접 확인.
- **channel_compare** — 수치: 금리 vs QE 채널 순서. 기법: 슬로프그래프. 읽는 법: 채널별 선후 차이.
- **method_compare** — 수치: LP·이벤트·VAR 순서 간 Spearman. 기법: 막대. 읽는 법: 방법 간 일치도.

## 순서성 강한 자산
- **\\*_strong_from{3y|1p5y}** — 수치: Kendall's W ≥ 0.80 을 만족하는 최대 N자산의 순위·일관성·사슬(출처 구간 명시). 기법: 위 순서 그림과 동일 기법을 강한 순서성 자산으로 한정. 읽는 법: 가장 견고하게 순서가 드러나는 자산군.
"""
    try:
        C.OUT_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(md, encoding="utf-8")
        return str(path)
    except Exception as e:  # noqa
        C.warn(f"figures_guide 기록 실패: {type(e).__name__}: {e}")
        return ""
