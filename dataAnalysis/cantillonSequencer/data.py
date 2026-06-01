"""
data.py — 수집·변환·이벤트·충격 (v4 통합: data_loader + transform + events + shocks)
"""
from __future__ import annotations
import time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import config as C
warnings.filterwarnings('ignore')


# ======================================================================
# ← data_loader.py
# ======================================================================

# ======================================================================
# 캐시
# ======================================================================
def _cache_path(name: str) -> Path:
    return C.CACHE_DIR / f"{name}.csv"


def _load_cache(name: str) -> pd.Series | None:
    p = _cache_path(name)
    if p.exists():
        try:
            s = pd.read_csv(p, index_col=0, parse_dates=True).iloc[:, 0]
            s.name = name
            return s
        except Exception:
            return None
    return None


def _save_cache(name: str, s: pd.Series) -> None:
    try:
        s.to_frame(name).to_csv(_cache_path(name))
    except Exception:
        pass


# ======================================================================
# 재시도 (지수 백오프)
# ======================================================================
def _retry(call, label: str = "", tries: int = C.MAX_TRIES,
           base: float = C.BACKOFF_BASE):
    last = None
    for i in range(tries):
        try:
            return call()
        except Exception as e:  # noqa
            last = e
            wait = base ** (i + 1)  # 2,4,8,16,32,64
            if i < tries - 1:
                print(f"    [retry {i+1}/{tries}] {label}: {type(e).__name__} → {wait:.0f}s 대기")
                time.sleep(min(wait, 64))
    raise last  # type: ignore


# ======================================================================
# 수집기 (재시도 + 간격 + 캐시)
# ======================================================================
def _to_monthly(s: pd.Series) -> pd.Series:
    s = pd.Series(s).dropna()
    s.index = pd.to_datetime(s.index)
    return s.resample(C.FREQ).last()


def _full_index() -> pd.DatetimeIndex:
    return pd.date_range(C.START_DATE, C.END_DATE, freq=C.FREQ)


def _collect_fred(name: str, code: str, fred, use_cache: bool) -> pd.Series | None:
    if use_cache:
        c = _load_cache(name)
        if c is not None and c.notna().sum() > 12:
            return c
    if fred is None:
        return None

    def _call():
        return fred.get_series(code, observation_start=C.START_DATE,
                               observation_end=C.END_DATE)
    try:
        raw = _retry(_call, label=f"FRED {code}")
        time.sleep(C.FRED_THROTTLE)
        s = _to_monthly(pd.Series(raw)).reindex(_full_index()).interpolate(limit_area="inside")
        s.name = name
        _save_cache(name, s)
        return s
    except Exception as e:  # noqa
        print(f"  [warn] FRED {name}({code}) 실패: {type(e).__name__}")
        return None


def _collect_fred(name: str, code: str, fred, use_cache: bool):
    """returns (series|None, info{raw_rows, monthly_rows, from_cache, error, retry})."""
    info = {"raw_rows": None, "monthly_rows": None, "from_cache": False, "error": None, "retry": None}
    if use_cache:
        c = _load_cache(name)
        if c is not None and c.notna().sum() > 12:
            info.update(from_cache=True, raw_rows=int(c.notna().sum()),
                        monthly_rows=int(c.notna().sum()))
            return c, info
    if fred is None:
        info["error"] = "no_client"
        return None, info

    def _call():
        return fred.get_series(code, observation_start=C.START_DATE,
                               observation_end=C.END_DATE)
    try:
        raw = _retry(_call, label=f"FRED {code}")
        time.sleep(C.FRED_THROTTLE)
        raw_s = pd.Series(raw).dropna()
        s = _to_monthly(raw_s).reindex(_full_index()).interpolate(limit_area="inside")
        s.name = name
        _save_cache(name, s)
        info.update(raw_rows=int(len(raw_s)), monthly_rows=int(s.notna().sum()))
        return s, info
    except Exception as e:  # noqa
        info["error"] = type(e).__name__
        return None, info


def _collect_yahoo(name: str, ticker: str, use_cache: bool):
    """returns (series|None, info{...})."""
    info = {"raw_rows": None, "monthly_rows": None, "from_cache": False, "error": None, "retry": None}
    if use_cache:
        c = _load_cache(name)
        if c is not None and c.notna().sum() > 12:
            info.update(from_cache=True, raw_rows=int(c.notna().sum()),
                        monthly_rows=int(c.notna().sum()))
            return c, info

    def _call():
        import yfinance as yf
        df = yf.download(ticker, start=C.START_DATE, end=C.END_DATE,
                         progress=False, auto_adjust=True)
        if df is None or len(df) == 0:
            raise RuntimeError("empty")
        col = "Close" if "Close" in df.columns else df.columns[0]
        ser = df[col]
        if isinstance(ser, pd.DataFrame):
            ser = ser.iloc[:, 0]
        return ser
    try:
        raw = _retry(_call, label=f"Yahoo {ticker}")
        time.sleep(C.YAHOO_THROTTLE)
        raw_s = pd.Series(raw).dropna()
        s = _to_monthly(raw_s).reindex(_full_index()).interpolate(limit_area="inside")
        s.name = name
        _save_cache(name, s)
        info.update(raw_rows=int(len(raw_s)), monthly_rows=int(s.notna().sum()))
        return s, info
    except Exception as e:  # noqa
        info["error"] = type(e).__name__
        return None, info


# ======================================================================
# v5: 통합 월간 캐시 (raw_monthly.csv) — 정본 + 무결성
# ======================================================================
def _save_raw_monthly(assets: pd.DataFrame, walcl: pd.Series, ff: pd.Series) -> bool:
    """13자산 + WALCL + FedRate 를 월말 인덱스 단일 패널로 기록. 성공 여부."""
    try:
        C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        idx = _full_index()
        panel = assets.reindex(idx).copy()
        panel["WALCL"] = pd.Series(walcl).reindex(idx)
        panel["FedRate"] = pd.Series(ff).reindex(idx)
        panel.to_csv(C.RAW_MONTHLY)
        return C.RAW_MONTHLY.exists()
    except Exception as e:  # noqa
        C.warn(f"raw_monthly.csv 기록 실패: {type(e).__name__}: {e}")
        return False


def _load_raw_monthly():
    """무결성 통과 시 (assets, walcl, ff) 반환, 아니면 None."""
    p = C.RAW_MONTHLY
    if not p.exists():
        return None
    try:
        panel = pd.read_csv(p, index_col=0, parse_dates=True)
    except Exception:
        return None
    need = set(C.ASSET_KEYS) | {"WALCL", "FedRate"}
    if not need.issubset(set(panel.columns)) or len(panel) < C.CACHE_MIN_ROWS:
        return None
    assets = panel[C.ASSET_KEYS].copy()
    walcl = panel["WALCL"]; ff = panel["FedRate"]
    return assets, walcl, ff


# ======================================================================
# 합성 데이터 (WALCL 4 QE 국면 + FedRate 인하 패턴 + 칸티용 사슬)
# ======================================================================
def _sample_walcl(idx) -> pd.Series:
    anchors = {
        "2000-01-01": 0.65, "2007-12-01": 0.90, "2008-09-01": 0.95,
        "2009-01-01": 2.10, "2010-06-01": 2.30, "2011-06-01": 2.85,
        "2012-09-01": 2.85, "2014-10-01": 4.50, "2017-09-01": 4.45,
        "2019-08-01": 3.80, "2020-02-01": 4.15, "2020-06-01": 7.10,
        "2022-04-01": 8.95, "2024-12-01": 6.90,
    }
    a = pd.Series({pd.Timestamp(k): v for k, v in anchors.items()})
    a = a.reindex(idx).interpolate().bfill().ffill()
    rng = np.random.default_rng(C.SYNTH_SEED)
    noise = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx).cumsum() * 0.02
    return (a + noise).clip(lower=0.4)


def _sample_fedfunds(idx) -> pd.Series:
    anchors = {
        "2000-01-01": 5.5, "2001-12-01": 1.75, "2004-06-01": 1.0,
        "2006-06-01": 5.25, "2008-12-01": 0.15, "2015-12-01": 0.25,
        "2018-12-01": 2.40, "2019-12-01": 1.55, "2020-04-01": 0.05,
        "2022-12-01": 4.35, "2024-12-01": 4.50,
    }
    ff = pd.Series({pd.Timestamp(k): v for k, v in anchors.items()})
    return ff.reindex(idx).interpolate().bfill().ffill()


def _impulse(idx, dates, decay=18) -> pd.Series:
    base = pd.Series(0.0, index=idx)
    arr = np.asarray(idx)
    for d in dates:
        d = pd.Timestamp(d)
        if d not in idx:
            d = idx[np.argmin(np.abs(arr - np.datetime64(d)))]
        pos = idx.get_loc(d)
        kern = np.exp(-np.arange(0, len(idx)) / decay)
        seg = np.zeros(len(idx))
        seg[pos:] = kern[: len(idx) - pos]
        base += pd.Series(seg, index=idx)
    return base


def generate_sample_data(idx=None):
    """합성 (assets_df, walcl, fedfunds)."""
    idx = _full_index() if idx is None else idx
    rng = np.random.default_rng(C.SYNTH_SEED + 2)
    walcl = _sample_walcl(idx)
    ff = _sample_fedfunds(idx)

    qe_starts = ["2008-11-01", "2010-06-01", "2013-01-01", "2020-04-01"]
    cut_starts = ["2001-01-01", "2007-09-01", "2019-08-01"]
    qe_imp = _impulse(idx, qe_starts, decay=20)

    role_lag = {"frontrun": 0, "financial": 1, "metal": 6, "real": 12, "tail": 18}
    role_amp = {"frontrun": 0.06, "financial": 0.11, "metal": 0.09, "real": 0.06, "tail": 0.03}

    out = {}
    for k, spec in C.ASSETS.items():
        role = spec["role"]
        q = qe_imp.shift(role_lag[role]).fillna(0.0) * role_amp[role]
        rc = pd.Series(0.0, index=idx)  # 인하 반응은 이벤트별 무작위(순서 교란)
        for cd in cut_starts:
            lag = int(rng.integers(0, 16)); amp = float(rng.uniform(-0.05, 0.07))
            rc += _impulse(idx, [cd], decay=14).shift(lag).fillna(0.0) * amp
        drift = 0.0015 + 0.001 * (role in ("financial", "metal"))
        shocks = pd.Series(rng.normal(0, 0.012, len(idx)), index=idx)
        log_level = (drift + q.diff().fillna(q) + rc.diff().fillna(rc) + shocks).cumsum()
        out[k] = 100 * np.exp(log_level)
    return pd.DataFrame(out, index=idx), walcl, ff


# ======================================================================
# 공개 API
# ======================================================================
def load_data(use_cache: bool = True, sample: bool = False):
    """
    returns (assets_df, walcl, fedfunds, is_synthetic)
    수집 과정을 시리즈 단위로 로그하고, 성공 시 통합 캐시 raw_monthly.csv 를 기록한다.
    """
    idx = _full_index()
    if sample or not C.FRED_API_KEY:
        if not sample:
            C.status("FRED_API_KEY 없음 → 합성 데이터 폴백", "warn")
        a, w, f = generate_sample_data(idx)
        C.data_shape(a.shape[0], a.shape[1] + 2)
        return a, w, f, True

    # 1) 통합 캐시 우선 사용
    if use_cache:
        cached = _load_raw_monthly()
        if cached is not None:
            a, w, f = cached
            C.cache_confirm(C.RAW_MONTHLY, reused=True, shape=(len(a), a.shape[1] + 2))
            C.data_shape(a.shape[0], a.shape[1] + 2)
            return a.reindex(idx), w.reindex(idx), f.reindex(idx), False

    fred = None
    try:
        from fredapi import Fred
        fred = Fred(api_key=C.FRED_API_KEY)
    except Exception as e:  # noqa
        C.status(f"FRED 초기화 실패({type(e).__name__}) → 합성 폴백", "warn")
        a, w, f = generate_sample_data(idx)
        return a, w, f, True

    # 2) 시리즈별 수집 (FRED 먼저: 매크로 2 + fred 자산, 그다음 yahoo 자산)
    fred_items = [("WALCL", C.WALCL_SERIES, "macro_walcl"),
                  ("FedRate", C.MONETARY_SERIES, "macro_fedfunds")]
    fred_items += [(C.DISP_NAME[k], spec["id"], k)
                   for k, spec in C.ASSETS.items() if spec["source"] == "fred"]
    yahoo_items = [(C.DISP_NAME[k], spec["id"], k)
                   for k, spec in C.ASSETS.items() if spec["source"] == "yahoo"]
    C.collect_header(len(fred_items), len(yahoo_items))

    cols, ok = {}, 0
    walcl = ff = None
    total = len(fred_items) + len(yahoo_items)
    i = 0
    for disp, code, key in fred_items:
        i += 1
        s, info = _collect_fred(key, code, fred, use_cache)
        if s is not None:
            C.collect_line(i, len(fred_items), "FRED", disp, code,
                           info["raw_rows"], info["monthly_rows"])
            ok += 1
            if key == "macro_walcl":
                if s.median() > 1000:
                    s = s / 1e6
                walcl = s
            elif key == "macro_fedfunds":
                ff = s
            else:
                cols[key] = s
        else:
            C.collect_line(i, len(fred_items), "FRED", disp, code, None, ok=False,
                           retry=info.get("retry"))
    for disp, code, key in yahoo_items:
        i += 1
        s, info = _collect_yahoo(key, code, use_cache)
        n_y = i - len(fred_items)
        if s is not None:
            C.collect_line(n_y, len(yahoo_items), "Yahoo", disp, code,
                           info["raw_rows"], info["monthly_rows"])
            ok += 1; cols[key] = s
        else:
            C.collect_line(n_y, len(yahoo_items), "Yahoo", disp, code, None, ok=False,
                           retry=info.get("retry"))
    C.collect_summary(ok, total)

    # 3) 충분치 않으면 합성 폴백
    if walcl is None or ff is None or len(cols) < len(C.ASSETS) * 0.6:
        C.status(f"실데이터 부족(자산 {len(cols)}/{len(C.ASSETS)}) → 합성 폴백", "warn")
        a, w, f = generate_sample_data(idx)
        return a, w, f, True

    assets = pd.DataFrame(cols).reindex(idx)
    walcl = walcl.reindex(idx); ff = ff.reindex(idx)

    # 4) 통합 캐시 기록 + 확인
    if _save_raw_monthly(assets, walcl, ff):
        C.cache_confirm(C.RAW_MONTHLY, reused=False)
    C.data_shape(assets.shape[0], assets.shape[1] + 2)
    return assets, walcl, ff, False


# ======================================================================
# ← transform.py
# ======================================================================

def apply_publication_lag(assets: pd.DataFrame) -> pd.DataFrame:
    out = assets.copy()
    for k in out.columns:
        lag = C.ASSETS.get(k, {}).get("lag", 0)
        if lag:
            out[k] = out[k].shift(lag)
    return out


def deseasonalize(assets: pd.DataFrame, period: int = 12) -> pd.DataFrame:
    try:
        from statsmodels.tsa.seasonal import STL
    except Exception:
        return assets
    out = {}
    for k in assets.columns:
        spec = C.ASSETS.get(k, {})
        s = assets[k]
        if spec.get("seasonal") == "SA":      # 이미 계절조정 → 그대로
            out[k] = s
            continue
        ss = s.dropna()
        if len(ss) < period * 2 + 1:
            out[k] = s
            continue
        try:
            res = STL(np.log(ss.clip(lower=1e-9)), period=period, robust=True).fit()
            out[k] = np.exp(res.observed - res.seasonal).reindex(assets.index)
        except Exception:
            out[k] = s
    return pd.DataFrame(out, index=assets.index)


def log_levels(assets: pd.DataFrame) -> pd.DataFrame:
    return np.log(assets.clip(lower=1e-9))


def prepare(assets: pd.DataFrame) -> dict:
    aligned = apply_publication_lag(assets)
    deseason = deseasonalize(aligned)
    return {"aligned": aligned, "deseason": deseason,
            "loglev": log_levels(deseason)}


def centered_response(loglev: pd.DataFrame, event_date) -> pd.DataFrame:
    """exp(logLₜ − logL_event) − 1  (이벤트 시점 기준 누적 반응)."""
    event_date = pd.Timestamp(event_date)
    if event_date not in loglev.index:
        pos = loglev.index.get_indexer([event_date], method="nearest")[0]
        event_date = loglev.index[pos]
    return np.exp(loglev.sub(loglev.loc[event_date], axis=1)) - 1.0


def window_response(loglev: pd.DataFrame, event_date,
                    pre: int = C.EVENT_PRE_MONTHS,
                    post: int = C.EVENT_POST_MONTHS) -> pd.DataFrame:
    """이벤트 ±윈도우 누적 반응. index = 상대월(정수, 0=이벤트)."""
    event_date = pd.Timestamp(event_date)
    idx = loglev.index
    if event_date not in idx:
        pos = idx.get_indexer([event_date], method="nearest")[0]
        event_date = idx[pos]
    resp = centered_response(loglev, event_date)
    lo = event_date - pd.DateOffset(months=pre)
    hi = event_date + pd.DateOffset(months=post)
    seg = resp.loc[(resp.index >= lo) & (resp.index <= hi)].copy()
    seg.index = ((seg.index.year - event_date.year) * 12 +
                 (seg.index.month - event_date.month))
    seg.index.name = "rel_month"
    return seg


def adf_flags(assets: pd.DataFrame) -> pd.DataFrame:
    try:
        from statsmodels.tsa.stattools import adfuller
    except Exception:
        return pd.DataFrame()
    rows = []
    for k in assets.columns:
        s = assets[k].dropna()
        try:
            rows.append({"asset": k, "p_level": round(adfuller(s, autolag="AIC")[1], 4),
                         "needs_diff": bool(adfuller(s, autolag="AIC")[1] > 0.05)})
        except Exception:
            rows.append({"asset": k, "p_level": np.nan, "needs_diff": True})
    return pd.DataFrame(rows)


# ======================================================================
# ← events.py
# ======================================================================

def _snap(dates, index) -> list[pd.Timestamp]:
    out = []
    arr = np.asarray(index)
    for d in dates:
        d = pd.Timestamp(d)
        if d not in index:
            d = index[np.argmin(np.abs(arr - np.datetime64(d)))]
        out.append(d)
    return out


def event_dates(channel: str = "mixed", index=None) -> pd.DataFrame:
    """
    channel: 'rate' | 'qe_announced' | 'mixed'
    데이터 인덱스가 주어지면 가용 월에 스냅.
    """
    table = {"rate": C.RATE_CUT_EVENTS, "qe_announced": C.QE_ANNOUNCED,
             "mixed": C.MIXED_EVENTS}
    dates = table.get(channel, C.MIXED_EVENTS)
    if index is not None:
        dates = _snap(dates, index)
    else:
        dates = [pd.Timestamp(d) for d in dates]
    qe_ann = {pd.Timestamp(x) for x in C.QE_ANNOUNCED}
    rows = []
    for d in dates:
        ch = "qe" if (channel == "qe_announced" or
                      (channel == "mixed" and d in qe_ann)) else "rate"
        rows.append({"event_date": d, "channel": ch, "label": f"{d:%Y-%m}"})
    return pd.DataFrame(rows)


def detect_injection_events(walcl: pd.Series,
                            growth_thr: float = C.INJ_GROWTH_THR,
                            smooth: int = C.INJ_SMOOTH,
                            min_len: int = C.INJ_MIN_LEN) -> pd.DataFrame:
    """
    WALCL 평활 월간 로그증가율이 임계를 넘어 min_len 개월 이상 지속되는
    구간을 주입 국면으로 탐지. event_date = 국면 시작월(실제 확장 시작).
    """
    w = pd.Series(walcl).astype(float).dropna()
    gs = np.log(w).diff().rolling(smooth, min_periods=1).mean()
    hot = (gs > growth_thr).astype(int).values
    idx = w.index

    events, i, n = [], 0, len(hot)
    while i < n:
        if hot[i] == 1:
            j = i
            while j + 1 < n and hot[j + 1] == 1:
                j += 1
            if (j - i + 1) >= min_len:
                start, end = idx[i], idx[j]
                lvl0, lvl1 = float(w.loc[start]), float(w.loc[end])
                events.append({"event_date": start, "end_date": end,
                               "length_m": int(j - i + 1),
                               "dWALCL": round(lvl1 - lvl0, 3),
                               "start_level": round(lvl0, 3),
                               "end_level": round(lvl1, 3),
                               "channel": "qe"})
            i = j + 1
        else:
            i += 1
    df = pd.DataFrame(events)
    if not df.empty:
        df = df.sort_values("event_date").reset_index(drop=True)
        df["label"] = [f"QE@{d:%Y-%m}" for d in df["event_date"]]
    return df


def quantify_events(events: pd.DataFrame, walcl: pd.Series,
                    horizon: int = 12) -> pd.DataFrame:
    """
    이벤트별 WALCL 규모/강도($조).
      magnitude_T : 이벤트 후 horizon 개월 내 WALCL 증가($조)
      intensity   : magnitude / 사전 수준 (상대 강도)
    detect_injection_events 출력이면 dWALCL 을 그대로 magnitude 로 사용.
    """
    w = pd.Series(walcl).astype(float)
    rows = []
    for _, e in events.iterrows():
        d = pd.Timestamp(e["event_date"])
        if "dWALCL" in e and not pd.isna(e["dWALCL"]):
            mag = float(e["dWALCL"])
            base = float(e.get("start_level", w.reindex([d]).iloc[0]))
        else:
            seg = w.loc[d: d + pd.DateOffset(months=horizon)]
            base = float(w.reindex([d]).iloc[0]) if d in w.index else float(seg.iloc[0])
            mag = float(seg.max() - base) if len(seg) else np.nan
        rows.append({"event_date": d,
                     "magnitude_T": round(mag, 3),
                     "intensity": round(mag / base, 3) if base else np.nan})
    return pd.DataFrame(rows)


def _select_top_events(df: pd.DataFrame, mag_col: str, n: int,
                       gap_m: int = C.EVENT_MIN_GAP_M) -> pd.DataFrame:
    """규모(mag_col) 상위 n개를 간격(gap_m) 제약으로 탐욕 선택(중복 제거)."""
    if df is None or df.empty:
        return df
    cand = df.sort_values(mag_col, ascending=False)
    picked = []
    for _, r in cand.iterrows():
        d = pd.Timestamp(r["event_date"])
        if all(abs((d.year - p.year) * 12 + (d.month - p.month)) >= gap_m for p in picked):
            picked.append(d)
        if len(picked) >= n:
            break
    return df[df["event_date"].isin(picked)].sort_values("event_date").reset_index(drop=True)


def detect_rate_cut_events(fedfunds: pd.Series,
                           target_n: int = C.RATE_EVENT_TARGET_N,
                           gap_m: int = C.EVENT_MIN_GAP_M) -> pd.DataFrame:
    """
    금리 인하 이벤트(금리 인하 이벤트 분석용). 정책금리 하강 사이클을 탐지하고
    총 인하폭(bp) 상위 target_n 개만 선택. event_date = 인하 시작월,
    magnitude_T = cut_bp(=peak−trough, bp).
    """
    ff = pd.Series(fedfunds).astype(float).dropna()
    if len(ff) < 3:
        return pd.DataFrame()
    d = ff.diff()
    idx = ff.index
    n = len(ff)
    events, i = [], 0
    while i < n:
        if i > 0 and d.iloc[i] < -1e-9:        # 인하 시작
            j = i
            while j + 1 < n and d.iloc[j + 1] <= 1e-9:   # 하강·평탄 지속
                j += 1
            start = idx[i - 1]                  # 사이클 직전 고점
            seg = ff.iloc[i - 1: j + 1]
            trough = seg.idxmin()
            bp = round((float(ff.loc[start]) - float(ff.loc[trough])) * 100, 1)
            events.append({"event_date": idx[i], "end_date": trough, "cut_bp": bp,
                           "start_level": round(float(ff.loc[start]), 3),
                           "trough_level": round(float(ff.loc[trough]), 3),
                           "magnitude_T": bp, "channel": "rate"})
            i = j + 1
        else:
            i += 1
    df = pd.DataFrame(events)
    if df.empty:
        return df
    df["label"] = [f"CUT@{pd.Timestamp(x):%Y-%m}" for x in df["event_date"]]
    return _select_top_events(df, "cut_bp", target_n, gap_m)


def build_events(walcl: pd.Series, fedfunds: pd.Series | None = None) -> dict:
    """
    통화 이벤트 2종 + 베이스라인.
      qe_surge  : 급격한 QE 구간(QE1/QE3/QEinf) — QE 경로의 유일 표현
      rate_cuts : 금리 인하 이벤트(큰 인하 상위 ~5~6)
      mixed     : 통제 없이 합친 베이스라인(낮은 W 기준선, H1 비교용)
    (점 단위 'QE 증가 이벤트' 분석은 두지 않는다.)
    """
    idx = walcl.index
    rate_cuts = detect_rate_cut_events(fedfunds) if fedfunds is not None else pd.DataFrame()
    mixed = event_dates("mixed", idx)
    surge = program_events(walcl)
    return {"qe_surge": surge, "rate_cuts": rate_cuts, "mixed": mixed,
            "program": surge}        # 구 키 별칭(하위호환)


def program_events(walcl: pd.Series,
                   programs: list[dict] | None = None,
                   post_years: float = C.PROGRAM_POST_YEARS) -> pd.DataFrame:
    """
    급격한 QE 구간 이벤트.
      event_date = 구간 시작월(t=0), end_date = 구간 종료월
      span_m     = 구간 길이(개월), post_m = span_m + round(12·post_years) (분석 사후창)
      dWALCL     = 구간 누적 WALCL 증가($조), intensity = dWALCL/시작수준
    post_years 로 +3년/+1.5년 등 사후창 길이를 바꿔 동일 기법 분석을 반복 적용.
    point 이벤트와 동일 스키마(channel='program').
    """
    programs = programs or C.INJECTION_PROGRAMS
    w = pd.Series(walcl).astype(float)
    idx = w.index
    arr = np.asarray(idx)
    post_m_add = int(round(12 * post_years))
    rows = []
    for p in programs:
        s = pd.Timestamp(p["start"]).to_period("M").to_timestamp()
        e = pd.Timestamp(p["end"]).to_period("M").to_timestamp()
        if s not in idx:
            s = idx[np.argmin(np.abs(arr - np.datetime64(s)))]
        if e not in idx:
            e = idx[np.argmin(np.abs(arr - np.datetime64(e)))]
        span = (e.year - s.year) * 12 + (e.month - s.month)
        lvl0, lvl1 = float(w.loc[s]), float(w.loc[e])
        rows.append({
            "event_date": s, "end_date": e, "channel": "program",
            "label": p["label"], "span_m": int(span),
            "post_years": float(post_years),
            "post_m": int(span + post_m_add),
            "dWALCL": round(lvl1 - lvl0, 3),
            "magnitude_T": round(lvl1 - lvl0, 3),
            "intensity": round((lvl1 - lvl0) / lvl0, 3) if lvl0 else np.nan,
        })
    return pd.DataFrame(rows)


def save_events(ev: dict) -> None:
    if "qe_surge" in ev:
        ev["qe_surge"].to_csv(C.RES_DIR / "qe_surge_events.csv", index=False)
    if isinstance(ev.get("rate_cuts"), pd.DataFrame) and not ev["rate_cuts"].empty:
        ev["rate_cuts"].to_csv(C.RES_DIR / "rate_cut_events.csv", index=False)
    if isinstance(ev.get("mixed"), pd.DataFrame) and not ev["mixed"].empty:
        ev["mixed"].to_csv(C.RES_DIR / "events.csv", index=False)


# ======================================================================
# ← shocks.py
# ======================================================================

def _ar_innovation(x: pd.Series, lags: int = C.SHOCK_AR_LAGS) -> pd.Series:
    """x 를 자기시차에 회귀한 잔차(예측 못한 부분)."""
    x = pd.Series(x).astype(float).dropna()
    df = pd.DataFrame({"y": x})
    for l in range(1, lags + 1):
        df[f"l{l}"] = x.shift(l)
    df = df.dropna()
    if len(df) < lags + 5:
        return x.diff().reindex(x.index)
    try:
        import statsmodels.api as sm
        X = sm.add_constant(df[[f"l{l}" for l in range(1, lags + 1)]])
        res = sm.OLS(df["y"], X).fit()
        return (df["y"] - res.predict(X)).reindex(x.index)
    except Exception:
        return x.diff().reindex(x.index)


def build_shocks(walcl: pd.Series, fedfunds: pd.Series) -> pd.DataFrame:
    """
    returns DataFrame[index=월, columns=['rate_shock','qe_shock','easing_shock']]
    """
    rate_shock = _ar_innovation(fedfunds)                       # + = 인상
    qe_shock = _ar_innovation(np.log(walcl.clip(lower=1e-6)))   # + = 확장

    df = pd.DataFrame({"rate_shock": rate_shock,
                       "qe_shock": qe_shock}).reindex(walcl.index).fillna(0.0)

    def _z(s):
        sd = s.std()
        return (s - s.mean()) / sd if sd > 0 else s * 0.0

    df["easing_shock"] = _z(-df["rate_shock"]) + _z(df["qe_shock"])  # + = 완화
    return df


def save_shocks(shocks: pd.DataFrame) -> None:
    shocks.to_csv(C.RES_DIR / "shocks.csv")
