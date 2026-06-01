"""
analysis.py — 타이밍·순서·핵심N자산 (v4 통합: timing + ordering)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import config as C
import data as D


# ======================================================================
# ← timing.py
# ======================================================================

def _sustained_peak(series: np.ndarray, theta: float = C.DRAWDOWN_THETA, start: int = 0):
    """
    onset(start) 이후 구간에서 부호 인지 첫 지속 정점.
    returns (peak_idx, peak_val) — peak_idx 는 원본 post 좌표(>= start 보장).
    """
    n = len(series)
    if n == 0 or start >= n:
        return None, np.nan
    seg = np.asarray(series[start:], float)
    if np.all(np.isnan(seg)):
        return None, np.nan
    pos_ext = np.nanmax(seg) if np.any(seg > 0) else 0.0
    neg_ext = np.nanmin(seg) if np.any(seg < 0) else 0.0
    sign = 1.0 if abs(pos_ext) >= abs(neg_ext) else -1.0
    s = seg * sign
    run_max, run_arg = -np.inf, 0
    for i in range(len(seg)):
        if s[i] > run_max:
            run_max, run_arg = s[i], i
        elif run_max > 0 and (run_max - s[i]) >= theta * abs(run_max):
            return start + run_arg, seg[run_arg]
    a = int(np.nanargmax(s))
    return start + a, seg[a]


def _onset(post: np.ndarray, pre: np.ndarray,
           k: float = C.ONSET_VOL_K, floor_pct: float = C.ONSET_MIN_PCT,
           persist: int = C.ONSET_PERSIST_M):
    """
    pre 변동성 밴드 기반 onset.
      thr = max(k · std(pre 반응 변화), floor_pct/100)
      |post| 가 thr 를 persist 개월 연속 넘는 최초 시점.
    """
    pre = np.asarray(pre, float)
    pre = pre[~np.isnan(pre)]
    pre_vol = float(np.std(np.diff(pre))) if len(pre) > 2 else 0.0
    thr = max(k * pre_vol, floor_pct / 100.0)
    cnt = 0
    for i in range(len(post)):
        if not np.isnan(post[i]) and abs(post[i]) >= thr:
            cnt += 1
            if cnt >= persist:
                return i - persist + 1, thr
        else:
            cnt = 0
    return None, thr


def _half(post: np.ndarray, peak_idx, peak_val, frac: float = C.HALF_FRAC, start: int = 0):
    """onset(start)~peak 사이에서 peak 의 frac 에 처음 도달하는 월."""
    if peak_idx is None or np.isnan(peak_val) or peak_val == 0:
        return None
    target = frac * peak_val
    for i in range(max(0, start), peak_idx + 1):
        if (peak_val > 0 and post[i] >= target) or (peak_val < 0 and post[i] <= target):
            return i
    return None


def timing_for_window(window_resp: pd.DataFrame) -> pd.DataFrame:
    """
    window_resp: index=상대월(0=이벤트), columns=자산.
    returns DataFrame[asset, onset_m, half_m, peak_m, peak_resp, onset_thr]
    v6: peak 는 onset 이후 구간으로 제약 → peak_m ≥ onset_m (역행 방지).
    """
    pre_df = window_resp[window_resp.index < 0]
    post_df = window_resp[window_resp.index >= 0]
    rel = post_df.index.values
    rows = []
    for k in post_df.columns:
        post = pd.Series(post_df[k].values).interpolate(limit_direction="both").values
        pre = pre_df[k].values if len(pre_df) else np.array([0.0])
        oi, thr = _onset(post, pre)
        start = oi if oi is not None else 0          # peak 는 onset 이후에서만 탐색
        pi, pv = _sustained_peak(post, start=start)
        hi = _half(post, pi, pv, start=start)
        rows.append({
            "asset": k,
            "onset_m": float(rel[oi]) if oi is not None else np.nan,
            "half_m": float(rel[hi]) if hi is not None else np.nan,
            "peak_m": float(rel[pi]) if pi is not None else np.nan,
            "peak_resp": round(float(pv), 4) if not np.isnan(pv) else np.nan,
            "onset_thr": round(float(thr), 4),
        })
    return pd.DataFrame(rows)


def extract_timing(loglev: pd.DataFrame, event_list: pd.DataFrame,
                   tag: str = "", pre: int = C.EVENT_PRE_MONTHS,
                   post: int = C.EVENT_POST_MONTHS,
                   min_post: int = C.MIN_POST_MONTHS,
                   save: bool = True) -> pd.DataFrame:
    """
    임의 이벤트 목록에 대한 타이밍표. 사후 길이 < min_post → enough_post=False.
    event_list 에 'post_m' 열이 있으면(급격한 QE 구간 분석) 이벤트별 사후관측구간으로 사용.
    tag 가 주어지면 timing[_tag].csv 로 저장.
    """
    last = loglev.index.max()
    frames = []
    for _, ev in event_list.iterrows():
        ed = pd.Timestamp(ev["event_date"])
        avail = (last.year - ed.year) * 12 + (last.month - ed.month)
        this_post = int(ev["post_m"]) if ("post_m" in event_list.columns and
                                          not pd.isna(ev.get("post_m"))) else post
        wr = D.window_response(loglev, ed, pre=pre, post=this_post)
        tf = timing_for_window(wr)
        tf.insert(0, "event_date", ed)
        tf.insert(1, "channel", ev.get("channel", "na"))
        tf["enough_post"] = bool(avail >= min_post)
        frames.append(tf)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if save and not out.empty:
        fn = f"timing_{tag}.csv" if tag else "timing.csv"
        out.to_csv(C.RES_DIR / fn, index=False)
    return out


# ======================================================================
# ← ordering.py
# ======================================================================

# ----------------------------------------------------------------------
# 동률 허용 순위
# ----------------------------------------------------------------------
def _tie_rank_strict(g: pd.DataFrame, tol_m: int = C.TIE_TOL_MONTHS,
                     tol_pct: float = C.TIE_TOL_PCT) -> pd.Series:
    """onset 차 ≤ tol_m AND peak 상대차 ≤ tol_pct 둘 다 만족 시 동률(평균 순위)."""
    d = g.dropna(subset=["onset_m"]).copy()
    if d.empty:
        return pd.Series(dtype=float)
    d = d.sort_values(["onset_m", "peak_m"], na_position="last").reset_index(drop=True)
    onset = d["onset_m"].values
    peak = d["peak_resp"].abs().fillna(0).values if "peak_resp" in d else np.zeros(len(d))
    grp = [0]
    for i in range(1, len(d)):
        same = abs(onset[i] - onset[i - 1]) <= tol_m
        denom = max(abs(peak[i - 1]), 1e-9)
        same = same and (abs(peak[i] - peak[i - 1]) / denom <= tol_pct)
        grp.append(grp[-1] if same else grp[-1] + 1)
    d["grp"] = grp
    d["pos"] = np.arange(1, len(d) + 1)
    return pd.Series(d.groupby("grp")["pos"].transform("mean").values,
                     index=d["asset"].values)


def _rank_matrix(timing_tbl: pd.DataFrame, only_enough: bool = True) -> pd.DataFrame:
    df = timing_tbl.copy()
    if only_enough and "enough_post" in df.columns:
        df = df[df["enough_post"]]
    mats = {}
    for ed, g in df.groupby("event_date"):
        mats[pd.Timestamp(ed).strftime("%Y-%m")] = _tie_rank_strict(g)
    return pd.DataFrame(mats).reindex(C.ASSET_KEYS)


# 호환 별칭
rank_matrix = _rank_matrix


# ----------------------------------------------------------------------
# Kendall's W
# ----------------------------------------------------------------------
def kendalls_w(rank_mat: pd.DataFrame) -> float:
    M = rank_mat.dropna(how="any")
    n, m = M.shape
    if n < 2 or m < 2:
        return np.nan
    R = M.sum(axis=1).values
    S = float(np.sum((R - R.mean()) ** 2))
    T = 0.0
    for col in M.columns:
        _, counts = np.unique(M[col].values, return_counts=True)
        T += float(np.sum(counts ** 3 - counts))
    denom = m ** 2 * (n ** 3 - n) - m * T
    return round(12.0 * S / denom, 4) if denom > 0 else np.nan


# ----------------------------------------------------------------------
# 평균 순서 / 자산 선별
# ----------------------------------------------------------------------
def mean_ordering(rank_mat: pd.DataFrame) -> pd.DataFrame:
    M = rank_mat
    out = pd.DataFrame({
        "asset": M.index,
        "label": [C.ASSET_LABELS.get(k, k) for k in M.index],
        "role": [C.ASSETS.get(k, {}).get("role", "na") for k in M.index],
        "mean_rank": M.mean(axis=1).values,
        "std_rank": M.std(axis=1).values,
        "n_obs": M.notna().sum(axis=1).values,
    })
    return out.sort_values("mean_rank").reset_index(drop=True)


def select_assets(rank_mat: pd.DataFrame, top: int | None = None) -> pd.DataFrame:
    """분산(std_rank) 기반 일관성 선별: 평균순위 낮고 분산 작은 자산 우선."""
    o = mean_ordering(rank_mat).dropna(subset=["mean_rank"])
    o["consistency"] = 1.0 / (1.0 + o["std_rank"].fillna(o["std_rank"].max()))
    o = o.sort_values(["mean_rank", "std_rank"])
    return o.head(top) if top else o


def select_assets_tagged(timing_qe, timing_mixed, top: int | None = None) -> pd.DataFrame:
    """QE-only / 혼합 각각 선별 결과를 한 표로(비교용)."""
    sa_qe = select_assets(_rank_matrix(timing_qe)).set_index("asset")
    sa_mx = select_assets(_rank_matrix(timing_mixed)).set_index("asset")
    rows = []
    for k in C.ASSET_KEYS:
        rows.append({
            "asset": k, "label": C.ASSET_LABELS.get(k, k),
            "rank_qe": sa_qe["mean_rank"].get(k, np.nan),
            "std_qe": sa_qe["std_rank"].get(k, np.nan),
            "rank_mixed": sa_mx["mean_rank"].get(k, np.nan),
            "std_mixed": sa_mx["std_rank"].get(k, np.nan),
        })
    out = pd.DataFrame(rows).sort_values("rank_qe")
    return out.head(top) if top else out


# ----------------------------------------------------------------------
# LP 기반 순서
# ----------------------------------------------------------------------
def order_from_lp(lp_scores: pd.DataFrame) -> pd.DataFrame:
    """LP onset(우선)·peak 기반 순서. onset 결측은 뒤로."""
    d = lp_scores.copy()
    d["onset_sort"] = d["onset_h"].fillna(9999)
    d["peak_sort"] = d["peak_h"].fillna(9999)
    d = d.sort_values(["onset_sort", "peak_sort"]).reset_index(drop=True)
    d["lp_rank"] = np.arange(1, len(d) + 1)
    return d[["asset", "onset_h", "peak_h", "cum_resp", "lp_rank"]]


def order_from_lp_named(lp_scores: pd.DataFrame) -> pd.DataFrame:
    d = order_from_lp(lp_scores)
    d.insert(1, "label", d["asset"].map(C.ASSET_LABELS))
    d.insert(2, "role", d["asset"].map(lambda k: C.ASSETS.get(k, {}).get("role", "na")))
    return d


def lp_ordering_table(lp_scores: pd.DataFrame, channel: str = "qe") -> pd.DataFrame:
    """LP 순서 수치화 표: asset/label/role/onset_h/peak_h/peak_beta/lp_rank. CSV 저장."""
    base = order_from_lp_named(lp_scores)
    d = base.merge(lp_scores[["asset", "peak_beta"]], on="asset", how="left")
    d["peak_beta"] = d["peak_beta"].round(4)
    d = d[["asset", "label", "role", "onset_h", "peak_h", "peak_beta", "cum_resp", "lp_rank"]]
    d.to_csv(C.RES_DIR / f"lp_ordering_scores_{channel}.csv", index=False)
    return d


def lp_event_concordance(lp_order: pd.DataFrame, event_order: pd.DataFrame) -> dict:
    """
    LP 순서(lp_rank) vs 이벤트-스터디 순서(mean_rank) 일치도 수치화.
    returns {rho(Spearman), n, table[asset,label,lp_rank,event_rank,delta]}.
    """
    ev = event_order.dropna(subset=["mean_rank"]).copy()
    ev["event_rank"] = ev["mean_rank"].rank(method="average")
    t = lp_order.merge(ev[["asset", "event_rank"]], on="asset", how="inner")
    rho = _spearman(t["lp_rank"], t["event_rank"]) if len(t) >= 3 else np.nan
    t["delta"] = (t["lp_rank"] - t["event_rank"]).round(1)
    return {"rho": rho, "n": int(len(t)),
            "table": t[["asset", "label", "lp_rank", "event_rank", "delta"]]
            .sort_values("lp_rank").reset_index(drop=True)}


# ----------------------------------------------------------------------
# 방법·채널 비교
# ----------------------------------------------------------------------
def _spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 3:
        return np.nan
    from scipy.stats import spearmanr
    return round(float(spearmanr(a[mask], b[mask]).statistic), 3)


def compare_methods(lp_scores, event_order, var_scores) -> pd.DataFrame:
    """LP·event-study·VAR 순위 Spearman 상관 행렬."""
    lp = order_from_lp(lp_scores).set_index("asset")["lp_rank"]
    es = event_order.set_index("asset")["mean_rank"] if "mean_rank" in event_order else \
        event_order.set_index("asset").iloc[:, 0]
    var = var_scores.copy()
    var["v_rank"] = var["peak_h"].fillna(9999).rank(method="average")
    var = var.set_index("asset")["v_rank"]
    keys = [k for k in C.ASSET_KEYS if k in lp.index]
    L = lp.reindex(keys); E = es.reindex(keys); V = var.reindex(keys)
    return pd.DataFrame([
        {"pair": "LP vs event-study", "spearman": _spearman(L, E)},
        {"pair": "LP vs VAR-IRF", "spearman": _spearman(L, V)},
        {"pair": "event-study vs VAR-IRF", "spearman": _spearman(E, V)},
    ])


def channel_orders(scores_rate, scores_qe) -> pd.DataFrame:
    """금리 vs QE 채널 LP 순서 비교."""
    r = order_from_lp(scores_rate).set_index("asset")["lp_rank"]
    q = order_from_lp(scores_qe).set_index("asset")["lp_rank"]
    rows = []
    for k in C.ASSET_KEYS:
        if k in r.index or k in q.index:
            rows.append({"asset": k, "label": C.ASSET_LABELS.get(k, k),
                         "role": C.ASSETS.get(k, {}).get("role", "na"),
                         "rank_rate": r.get(k, np.nan), "rank_qe": q.get(k, np.nan)})
    out = pd.DataFrame(rows)
    out["shift"] = out["rank_rate"] - out["rank_qe"]
    return out.sort_values("rank_qe").reset_index(drop=True)


def compare_event_sets(tt_by_set: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """혼합 vs QE-only vs rate-only Kendall's W 비교 (H1)."""
    rows = []
    for name, df in tt_by_set.items():
        mat = _rank_matrix(df, only_enough=True)
        rows.append({"set": name, "n_events": int(mat.shape[1]),
                     "n_assets": int(mat.dropna(how="any").shape[0]),
                     "kendalls_W": kendalls_w(mat)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 핵심 K자산 선별 (계획서 §6.6 / §13.3)
# ----------------------------------------------------------------------
def _rerank(mat: pd.DataFrame) -> pd.DataFrame:
    """부분집합 순위행렬을 열(이벤트)별로 1..n 으로 재순위(평균 동률)."""
    return mat.rank(axis=0, method="average")


def _maxW_subset(M: pd.DataFrame, n: int, std_map: dict):
    """크기 n 부분집합 중 (재순위 후) Kendall's W 최대 집합. 반환 (assets, W)."""
    from itertools import combinations
    best, bestW, best_std = None, -1.0, np.inf
    for combo in combinations(list(M.index), n):
        W = kendalls_w(_rerank(M.loc[list(combo)]))
        if np.isnan(W):
            continue
        std_sum = float(np.nansum([std_map.get(a, np.inf) for a in combo]))
        if (W > bestW) or (W == bestW and std_sum < best_std):
            best, bestW, best_std = list(combo), W, std_sum
    return best, (round(float(bestW), 4) if best is not None else np.nan)


def select_top_ordered(rank_mat: pd.DataFrame,
                       w_thr: float = C.TOP_W_THRESHOLD,
                       n_min: int = C.TOP_N_MIN,
                       method: str = C.TOP_SELECT_METHOD) -> dict:
    """
    적응형 핵심 N자산 (계획서 v4 §6.6): Kendall's W ≥ w_thr 를 만족하는 '가장 큰 N' 부분집합.
    N 을 len(assets)→n_min 으로 내려가며 각 크기 최대-W 집합을 구해,
    W ≥ w_thr 을 처음 만족하는(=최대 N) 집합에서 멈춤(qualified=True).
    어떤 N 에서도 미충족이면 n_min 최대-W 집합(qualified=False).
    부분집합 W 는 N 내 1..N 재순위 후 계산(W∈[0,1]).
    returns {assets, N, W, qualified, order, sub_mat, consistency}
    """
    M = rank_mat.dropna(how="any")
    assets = list(M.index)
    o_all = mean_ordering(rank_mat)
    o_all["consistency"] = 1.0 / (1.0 + o_all["std_rank"].fillna(o_all["std_rank"].max()))
    std_map = o_all.set_index("asset")["std_rank"].to_dict()
    n_min = max(2, int(n_min))

    if len(assets) < n_min:
        sub, qualified = assets, False
    else:
        sub, qualified = None, False
        for n in range(len(assets), n_min - 1, -1):   # 최대 N → 최소 N
            cand, candW = _maxW_subset(M, n, std_map)
            if cand is not None and candW >= w_thr:
                sub, qualified = cand, True
                break
        if sub is None:                                # 미충족 → n_min 최대-W
            sub, _ = _maxW_subset(M, n_min, std_map)
            qualified = False

    sub_mat = _rerank(M.loc[sub]) if (sub and len(sub) >= 2) else M.loc[sub]
    W_sub = kendalls_w(sub_mat) if (sub and len(sub) >= 2) else np.nan
    return {"assets": sub, "N": len(sub) if sub else 0,
            "W": round(float(W_sub), 4) if W_sub == W_sub else np.nan,
            "qualified": bool(qualified), "order": mean_ordering(sub_mat),
            "sub_mat": sub_mat, "consistency": o_all}


def pick_top_across_windows(rank_mat_3y: pd.DataFrame, rank_mat_1p5y: pd.DataFrame,
                            w_thr: float = C.TOP_W_THRESHOLD,
                            n_min: int = C.TOP_N_MIN,
                            method: str = C.TOP_SELECT_METHOD) -> dict:
    """
    +3년 / +1.5년 두 창 각각 select_top_ordered → 창 선택 우선순위:
      ⓐ qualified(W≥임계) 우선 → ⓑ N 큰 창 → ⓒ W 높은 창 → ⓓ 3년.
    returns {source_window, assets, N, order, sub_mat, qualified,
             W_selected, W_3y, N_3y, W_1p5y, N_1p5y, loser_assets, threshold}
    """
    r3 = select_top_ordered(rank_mat_3y, w_thr, n_min, method)
    r1 = select_top_ordered(rank_mat_1p5y, w_thr, n_min, method)

    def _key(r):  # 클수록 우선
        return (1 if r["qualified"] else 0, r["N"],
                r["W"] if r["W"] == r["W"] else -1.0)

    if _key(r3) >= _key(r1):
        win, lose, src = r3, r1, "3y"
    else:
        win, lose, src = r1, r3, "1p5y"

    return {"source_window": src, "assets": win["assets"], "N": win["N"],
            "order": win["order"], "sub_mat": win["sub_mat"],
            "qualified": win["qualified"], "W_selected": win["W"],
            "W_3y": r3["W"], "N_3y": r3["N"], "W_1p5y": r1["W"], "N_1p5y": r1["N"],
            "loser_assets": lose["assets"], "threshold": w_thr,
            "consistency": win["consistency"]}


def save_strong_assets(pick: dict) -> None:
    """strong_assets.csv: 순서성 강한 자산·순위·출처창·양쪽 (N,W)·임계·충족여부·미선택집합."""
    o = pick["order"].copy()
    o["N"] = pick["N"]
    o["source_window"] = pick["source_window"]
    o["W_selected"] = pick["W_selected"]
    o["W_3y"] = pick["W_3y"]; o["N_3y"] = pick["N_3y"]
    o["W_1p5y"] = pick["W_1p5y"]; o["N_1p5y"] = pick["N_1p5y"]
    o["threshold"] = pick["threshold"]
    o["qualified"] = pick["qualified"]
    o["unselected_assets"] = ",".join(pick["loser_assets"])
    o.to_csv(C.RES_DIR / "strong_assets.csv", index=False)


def save_topN(pick: dict) -> None:   # 하위호환 별칭
    save_strong_assets(pick)


def channel_compare_orders(order_qe: pd.DataFrame, order_rate: pd.DataFrame) -> pd.DataFrame:
    """
    채널 비교: 급격한 QE 구간 +3년 장 순서(rank_qe) vs 금리 인하 이벤트 순서(rank_rate).
    이벤트-스터디 평균순위를 각 채널 내에서 1.. 로 재순위해 비교.
    """
    q = order_qe.dropna(subset=["mean_rank"]).copy()
    q["rank_qe"] = q["mean_rank"].rank(method="average")
    r = order_rate.dropna(subset=["mean_rank"]).copy()
    r["rank_rate"] = r["mean_rank"].rank(method="average")
    qm = q.set_index("asset")["rank_qe"]; rm = r.set_index("asset")["rank_rate"]
    rows = []
    for k in C.ASSET_KEYS:
        if k in qm.index or k in rm.index:
            rows.append({"asset": k, "label": C.ASSET_LABELS.get(k, k),
                         "role": C.ASSETS.get(k, {}).get("role", "na"),
                         "rank_rate": rm.get(k, np.nan), "rank_qe": qm.get(k, np.nan)})
    out = pd.DataFrame(rows)
    out["shift"] = out["rank_rate"] - out["rank_qe"]
    return out.sort_values("rank_qe").reset_index(drop=True)


def magnitude_response(timing_qe: pd.DataFrame, qe_events: pd.DataFrame) -> pd.DataFrame:
    """H3: 주입 규모(ΔWALCL) ↔ 반응 크기·속도 상관 (qe_magnitude_response.csv)."""
    from scipy.stats import spearmanr
    mr = timing_qe.merge(qe_events[["event_date", "magnitude_T"]], on="event_date", how="left")
    rows = []
    amsk = (~mr["magnitude_T"].isna()) & (~mr["peak_resp"].isna())
    osk = (~mr["magnitude_T"].isna()) & (~mr["onset_m"].isna())
    amp = spearmanr(mr["magnitude_T"][amsk], mr["peak_resp"].abs()[amsk]).statistic if amsk.sum() >= 3 else np.nan
    ons = spearmanr(mr["magnitude_T"][osk], mr["onset_m"][osk]).statistic if osk.sum() >= 3 else np.nan
    rows.append({"metric": "amp_rho(규모↔정점|%|)", "rho": round(float(amp), 3) if amp == amp else np.nan,
                 "pass": bool(amp > 0)})
    rows.append({"metric": "onset_rho(규모↔onset, 음수기대)", "rho": round(float(ons), 3) if ons == ons else np.nan,
                 "pass": bool(ons < 0)})
    out = pd.DataFrame(rows)
    out.to_csv(C.RES_DIR / "qe_magnitude_response.csv", index=False)
    return out


def save_ordering(mat_mixed, mat_qe, comp, order_qe,
                  ord_lp_rate=None, ord_lp_qe=None, chan=None) -> None:
    mat_mixed.to_csv(C.RES_DIR / "rank_matrix_mixed.csv")
    mat_qe.to_csv(C.RES_DIR / "rank_matrix_qe.csv")
    comp.to_csv(C.RES_DIR / "W_comparison.csv", index=False)
    order_qe.to_csv(C.RES_DIR / "ordering_qe.csv", index=False)
    if ord_lp_rate is not None:
        ord_lp_rate.to_csv(C.RES_DIR / "ordering_lp_rate.csv", index=False)
    if ord_lp_qe is not None:
        ord_lp_qe.to_csv(C.RES_DIR / "ordering_lp_qe.csv", index=False)
    if chan is not None:
        chan.to_csv(C.RES_DIR / "channel_comparison.csv", index=False)
