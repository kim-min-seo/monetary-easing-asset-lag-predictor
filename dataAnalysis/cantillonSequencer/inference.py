"""
inference.py — LP·VAR-IRF·해저드 (v4 통합: local_projection + var_irf + hazard)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import config as C


# ======================================================================
# ← local_projection.py
# ======================================================================

def _lp_curve(level: pd.Series, main_shock: pd.Series,
              ctrl_shocks: dict[str, pd.Series] | None = None,
              H: int = C.LP_HORIZON, lags: int = C.LP_CONTROL_LAGS,
              hac: int = C.LP_HAC_LAGS) -> pd.DataFrame:
    """returns DataFrame[h, beta, se, lo, hi]."""
    import statsmodels.api as sm
    ctrl_shocks = ctrl_shocks or {}

    base = pd.DataFrame(index=level.index)
    base["main"] = main_shock
    for nm, s in ctrl_shocks.items():
        base[f"ctrl_{nm}"] = s
    for l in range(1, lags + 1):
        base[f"main_l{l}"] = main_shock.shift(l)
        for nm, s in ctrl_shocks.items():
            base[f"ctrl_{nm}_l{l}"] = s.shift(l)
        base[f"y_l{l}"] = level.diff().shift(l)

    rows = []
    for h in range(0, H + 1):
        yh = level.shift(-h) - level.shift(1)
        df = base.copy(); df["yh"] = yh
        df = df.dropna()
        if len(df) < (lags * 3 + 10):
            rows.append({"h": h, "beta": np.nan, "se": np.nan, "lo": np.nan, "hi": np.nan})
            continue
        X = sm.add_constant(df.drop(columns=["yh"]))
        try:
            res = sm.OLS(df["yh"], X).fit(cov_type="HAC", cov_kwds={"maxlags": hac})
            b = res.params.get("main", np.nan); se = res.bse.get("main", np.nan)
            rows.append({"h": h, "beta": b, "se": se,
                         "lo": b - C.SIG_Z * se, "hi": b + C.SIG_Z * se})
        except Exception:
            rows.append({"h": h, "beta": np.nan, "se": np.nan, "lo": np.nan, "hi": np.nan})
    return pd.DataFrame(rows)


def _extract(curve: pd.DataFrame) -> dict:
    """onset(첫 유의 h) · peak(유의 |β| 최대 h) · 누적반응."""
    h = curve["h"].values
    b = curve["beta"].values.astype(float)
    lo = curve["lo"].values.astype(float)
    hi = curve["hi"].values.astype(float)
    sig = (lo > 0) | (hi < 0)
    onset_h = next((int(h[i]) for i in range(len(h)) if sig[i]), np.nan)
    if np.all(np.isnan(b)):
        return {"onset_h": onset_h, "peak_h": np.nan, "peak_beta": np.nan, "cum_resp": np.nan}
    # 유의 구간 우선, 없으면 전체에서 |β| 최대
    cand = np.where(sig)[0]
    ax = cand[np.nanargmax(np.abs(b[cand]))] if len(cand) else int(np.nanargmax(np.abs(b)))
    return {"onset_h": onset_h, "peak_h": int(h[ax]),
            "peak_beta": round(float(b[ax]), 5),
            "cum_resp": round(float(np.nansum(b)), 5)}


def run_lp_channels(loglev: pd.DataFrame, shocks: pd.DataFrame,
                    H: int = C.LP_HORIZON) -> dict:
    """
    금리/QE 채널 분리(서로 통제).
    returns {'curves': long DataFrame[asset,channel,h,beta,se,lo,hi],
             'scores_rate': DataFrame, 'scores_qe': DataFrame}
    """
    rate = shocks["rate_shock"]; qe = shocks["qe_shock"]
    curve_rows, sr, sq = [], [], []
    for k in loglev.columns:
        lvl = loglev[k]
        c_qe = _lp_curve(lvl, qe, {"rate": rate}, H=H)
        c_rate = _lp_curve(lvl, rate, {"qe": qe}, H=H)
        for ch, cur in (("qe", c_qe), ("rate", c_rate)):
            cc = cur.copy(); cc["asset"] = k; cc["channel"] = ch
            curve_rows.append(cc)
            sc = _extract(cur); sc["asset"] = k
            (sq if ch == "qe" else sr).append(sc)
    curves = pd.concat(curve_rows, ignore_index=True)
    return {"curves": curves,
            "scores_rate": pd.DataFrame(sr),
            "scores_qe": pd.DataFrame(sq)}


def run_lp(loglev: pd.DataFrame, shocks: pd.DataFrame, H: int = C.LP_HORIZON) -> pd.DataFrame:
    """easing 단일 충격 LP(보조)."""
    eas = shocks["easing_shock"]
    rows = []
    for k in loglev.columns:
        sc = _extract(_lp_curve(loglev[k], eas, None, H=H)); sc["asset"] = k
        rows.append(sc)
    return pd.DataFrame(rows)


def save_lp(out: dict) -> None:
    out["scores_rate"].to_csv(C.RES_DIR / "lp_scores_rate.csv", index=False)
    out["scores_qe"].to_csv(C.RES_DIR / "lp_scores_qe.csv", index=False)
    out["curves"].to_csv(C.RES_DIR / "lp_curves.csv", index=False)


# ======================================================================
# ← var_irf.py
# ======================================================================

def _irf_peak(shock: pd.Series, y_log: pd.Series,
              maxlags: int = C.VAR_MAXLAGS, H: int = C.VAR_HORIZON) -> dict:
    from statsmodels.tsa.api import VAR
    df = pd.concat([shock.rename("shock"), y_log.diff().rename("dy")], axis=1).dropna()
    if len(df) < maxlags * 4 + 10:
        return {"onset_h": np.nan, "peak_h": np.nan, "cum_resp": np.nan}
    try:
        model = VAR(df)
        sel = model.select_order(maxlags)
        p = int(getattr(sel, "aic", 0) or 0)
        if p < 1:                       # 🐞 0-시차 → 최소 1로 재적합
            p = 1
        res = model.fit(p)
        irf = res.irf(H)
        resp = irf.irfs[:, 1, 0]        # shock(0) → dy(1)
        cum = np.cumsum(resp)
        peak_h = int(np.nanargmax(np.abs(cum)))
        # onset: 표준오차 밴드 첫 이탈(근사)
        onset_h = np.nan
        try:
            se = irf.stderr(H)[:, 1, 0]
            band = np.cumsum(np.abs(se))
            for hh in range(len(cum)):
                if abs(cum[hh]) > C.SIG_Z * band[hh] / np.sqrt(max(hh + 1, 1)):
                    onset_h = hh; break
        except Exception:
            pass
        return {"onset_h": onset_h, "peak_h": peak_h, "cum_resp": round(float(cum[-1]), 5)}
    except Exception:
        return {"onset_h": np.nan, "peak_h": np.nan, "cum_resp": np.nan}


def run_var_irf(loglev: pd.DataFrame, shocks: pd.DataFrame,
                shock_col: str = "qe_shock") -> pd.DataFrame:
    sh = shocks[shock_col]
    rows = []
    for k in loglev.columns:
        sc = _irf_peak(sh, loglev[k]); sc["asset"] = k
        rows.append(sc)
    df = pd.DataFrame(rows); df["shock"] = shock_col
    return df


def save_var(df: pd.DataFrame) -> None:
    df.to_csv(C.RES_DIR / "var_irf_scores.csv", index=False)


# ======================================================================
# ← hazard.py
# ======================================================================

def _km_median(durations: np.ndarray, events: np.ndarray) -> float:
    """절단(events=0) 처리 KM 생존곡선의 중앙(survival≤0.5 최초) onset."""
    durations = np.asarray(durations, float); events = np.asarray(events, int)
    order = np.argsort(durations)
    d, e = durations[order], events[order]
    n = len(d); at_risk = n; surv = 1.0
    for t in np.unique(d):
        mask = d == t
        deaths = int(e[mask].sum()); censored = int((~e[mask].astype(bool)).sum())
        if at_risk > 0 and deaths > 0:
            surv *= (1 - deaths / at_risk)
            if surv <= 0.5:
                return float(t)
        at_risk -= (deaths + censored)
    return np.nan


def run_hazard(timing_tbl: pd.DataFrame, events_mag: pd.DataFrame | None = None,
               channel: str | None = "qe",
               post_window: int = C.EVENT_POST_MONTHS) -> pd.DataFrame:
    """
    timing_tbl: extract_timing 출력. onset 미발생 = post_window 우측 절단.
    events_mag: [event_date, magnitude_T] 있으면 규모-onset 상관(이산 해저드 근사).
    returns DataFrame[asset,label,n,mean_onset,median_onset,onset_rate,mag_corr]
    """
    df = timing_tbl.copy()
    if channel and "channel" in df.columns:
        df = df[df["channel"] == channel]
    rows = []
    for k, g in df.groupby("asset"):
        dur = g["onset_m"].values.astype(float)
        ev = ~np.isnan(dur)
        dur_c = np.where(np.isnan(dur), post_window, dur)
        mag_corr = np.nan
        if events_mag is not None:
            try:
                m = g.merge(events_mag, on="event_date", how="left")
                d2 = m["onset_m"].values.astype(float)
                mask = ~np.isnan(d2)
                if mask.sum() >= 3:
                    from scipy.stats import spearmanr
                    mag_corr = round(float(spearmanr(m["magnitude_T"].values[mask],
                                                     d2[mask]).statistic), 3)
            except Exception:
                pass
        rows.append({"asset": k, "label": C.ASSET_LABELS.get(k, k),
                     "n": int(len(dur_c)),
                     "mean_onset": round(float(np.nanmean(dur_c)), 2),
                     "median_onset": _km_median(dur_c, ev.astype(int)),
                     "onset_rate": round(float(ev.mean()), 3),
                     "mag_corr": mag_corr})
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["asset", "label", "n", "mean_onset",
                                     "median_onset", "onset_rate", "mag_corr"])
    return out.sort_values("mean_onset").reset_index(drop=True)


def save_hazard(df: pd.DataFrame) -> None:
    df.to_csv(C.RES_DIR / "hazard_onset.csv", index=False)
