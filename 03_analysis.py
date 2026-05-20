# ============================================================
#  03_analysis.py — 실증 분석 (v8 — PHASE C1~C5)
#
#  v8 변경 사항:
#  ─────────────────────────────────────────────────────────
#  [C1] Granger best_lag p-해킹 제거 — 양변량 VAR AIC 시차 선택
#  [C2] VAR 적분차수 통일 — CaseShiller_LogReturn (1차 차분) 사용
#  [C3] VAR Cholesky 순서 명시 + 대안 순서 강건성 검증
#       - 기존: 변수 순서가 IRF에 영향을 주는데 그 영향이 보고되지 않음
#       - 수정: 기본 Cholesky 순서를 이론적 근거와 함께 출력하고,
#               대안 순서(자산 역순)로 재추정하여 자산별 peak month
#               차이를 표로 비교 → 결과의 순서 의존성을 투명하게 공개
#  [C4] 이벤트 스터디 누적수익률 계산 수정
#       - 기존: cumret = (1 + log_return).cumprod() - 1
#               → 로그수익률을 단순수익률처럼 잘못 누적, 끝없이 부풀어
#                 모든 자산의 peak이 윈도우 끝(15~20개월)으로 쏠림
#       - 수정: cumret = np.exp(log_return.cumsum()) - 1 (정확한 수학식)
#               + 이벤트 시점 기준(baseline)으로 재정규화하여 "이벤트
#                 이후 누적 변동량"만 평가 (사전 트렌드 영향 제거)
#  [C5] derive_cantillon_order 재설계 — 평균 lag → 순위 평균
#       - 기존: np.mean([granger_lag, irf_lag, event_lag])
#               → 방법별 스케일이 다른데 단순 평균 (event_lag가 보통
#                 가장 커서 결과를 지배). 또한 일부 방법 lag이 누락된
#                 자산은 적은 표본의 평균으로 비교 불공정.
#       - 수정: 각 방법 내에서 자산 순위(1~5)를 매긴 후 그 순위의
#               평균으로 최종 순서 결정. 스케일 차이 제거 +
#               누락 방법은 가용 방법만의 순위로 처리.
# ============================================================

import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.tsa.api import VAR
import warnings
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────
#  ADF 정상성 검정
# ──────────────────────────────────────────────

def run_adf_test(df):
    print("\n  [3-1] ADF 정상성 검정")

    target_cols = (
        [c for c in df.columns if "_LogReturn" in c] +
        [c for c in [
            "FedRate_Change", "Real_Rate", "QE_Size",
            "DXY_Change", "M2_YoY", "Yield_Spread",
            "TIPS_Spread"
        ] if c in df.columns]
    )

    results = []
    print(f"  {'변수':35s} {'ADF통계량':>10} {'p값':>8} {'정상성':>8}")
    print("  " + "-" * 65)

    for col in target_cols:
        try:
            series = df[col].dropna()
            if len(series) < 20:
                continue
            adf_stat, p_val, *_ = adfuller(series, autolag="AIC")
            is_stat = p_val < 0.05
            mark    = "✓ 정상" if is_stat else "✗ 비정상"
            print(f"  {col:35s} {adf_stat:>10.4f} {p_val:>8.4f} {mark:>8}")
            results.append({"variable": col, "adf_stat": adf_stat,
                            "p_value": p_val, "is_stationary": is_stat})
        except Exception:
            pass

    df_res = pd.DataFrame(results)
    n_stat = df_res["is_stationary"].sum() if not df_res.empty else 0
    print(f"\n  ✓ 정상 변수: {n_stat}/{len(results)}개")

    if not df_res.empty:
        path = os.path.join(C.RESULT_DIR, "adf_results.csv")
        df_res.to_csv(path, index=False)
        print(f"  ✓ 저장: {path}")

    return df_res


# ──────────────────────────────────────────────
#  그랜저 인과관계 분석  (C1 + C2 적용)
# ──────────────────────────────────────────────

def run_granger_analysis(df):
    """
    ★ C1: AIC 시차 선택 (p-해킹 제거)
    ★ C2: CaseShiller_LogReturn (1차 차분) 사용
    """
    print("\n  [3-2] 그랜저 인과관계 분석 (v8: AIC 시차 선택 + 적분차수 통일)")

    causes = [c for c in [
        "FedRate_Change", "Real_Rate", "QE_Size", "DXY_Change",
        "M2_YoY", "TIPS_Spread",
        "Monetary_Ease_Index"
    ] if c in df.columns]

    effects_map = {
        "Gold_LogReturn":        "Gold_LogReturn",
        "WTI_LogReturn":         "WTI_LogReturn",
        "SP500_LogReturn":       "SP500_LogReturn",
        "CaseShiller_LogReturn": "CaseShiller_LogReturn",
        "CPI_LogReturn":         "CPI_LogReturn",
    }
    effects = [v for v in effects_map.values() if v in df.columns]

    max_lag_map = {
        "Gold_LogReturn":        12,
        "WTI_LogReturn":         12,
        "SP500_LogReturn":       12,
        "CaseShiller_LogReturn": 24,
        "CPI_LogReturn":         24,
    }

    lag_t  = pd.DataFrame(index=causes, columns=effects, dtype=float)
    pval_t = pd.DataFrame(index=causes, columns=effects, dtype=float)
    rows   = []

    for cause in causes:
        for effect in effects:
            try:
                this_max = max_lag_map.get(effect, 24)
                data     = df[[cause, effect]].dropna()
                if len(data) < this_max + 10:
                    continue

                try:
                    bv_model = VAR(data)
                    sel      = bv_model.select_order(maxlags=this_max)
                    aic_lag  = int(max(1, min(sel.aic, this_max)))
                except Exception:
                    aic_lag = min(6, this_max)

                res  = grangercausalitytests(data, maxlag=aic_lag, verbose=False)
                pval = res[aic_lag][0]["ssr_ftest"][1]
                sig  = "✓" if pval < 0.05 else "✗"
                print(f"  {sig} {cause:22s} → {effect:25s} "
                      f"AIC시차={aic_lag:2d}개월  p={pval:.4f}")

                lag_t.loc[cause, effect]  = aic_lag
                pval_t.loc[cause, effect] = pval
                rows.append({"cause": cause, "effect": effect,
                             "lag": aic_lag, "p_value": pval,
                             "max_lag": this_max,
                             "significant": pval < 0.05})
            except Exception:
                pass

    result_df = pd.DataFrame(rows)
    if not result_df.empty:
        path = os.path.join(C.RESULT_DIR, "granger_results.csv")
        result_df.to_csv(path, index=False)
        print(f"\n  ✓ 저장: {path}")
        n_sig = result_df["significant"].sum()
        print(f"  ✓ 유의 관계 (p<0.05): {n_sig}/{len(result_df)}개")

    return result_df, lag_t, pval_t


# ──────────────────────────────────────────────
#  VAR + IRF  (★ C3: Cholesky 순서 명시 + 강건성 검증)
# ──────────────────────────────────────────────

def _fit_var_and_peaks(data, var_order, max_lag, shock_var="Real_Rate"):
    """주어진 변수 순서로 VAR/IRF 추정 후 자산별 peak month 계산.
    내부 헬퍼 — Cholesky 순서 강건성 검증에 사용."""
    data_ord = data[var_order]
    model    = VAR(data_ord)
    try:
        sel = model.select_order(maxlags=max_lag)
        opt = max(1, min(sel.aic, 12))
    except Exception:
        opt = 4
    results = model.fit(opt)
    irf     = results.irf(24)

    peaks = {}
    if shock_var in var_order:
        sidx = var_order.index(shock_var)
        for col in var_order:
            if "_LogReturn" in col:
                ridx = var_order.index(col)
                vals = irf.irfs[:, ridx, sidx]
                peaks[col] = int(np.argmax(np.abs(vals)))
    return results, irf, peaks, opt


def run_var_irf(df):
    """
    ★ C2: CaseShiller_LogReturn 사용, ADF 사전 확인
    ★ C3: Cholesky 순서 명시 + 대안 순서로 강건성 검증
    """
    print("\n  [3-3] VAR + IRF 충격반응함수 분석 "
          "(v8: 적분차수 통일 + Cholesky 순서 강건성)")

    # ★ C3: 기본 Cholesky 순서 — 이론적 근거 명시
    base_order = [c for c in [
        "Real_Rate", "QE_Size", "M2_YoY", "TIPS_Spread",   # 통화변수 (외생)
        "Gold_LogReturn", "WTI_LogReturn",                  # 빠른 시장 자산
        "SP500_LogReturn",
        "CaseShiller_LogReturn",                            # 느린 시장 (부동산)
        "CPI_LogReturn"                                     # 거시 결과 (가장 내생)
    ] if c in df.columns]

    data = df[base_order].dropna()
    print(f"  VAR 데이터: {data.shape[0]}개월 × {data.shape[1]}개 변수")

    # ★ C2: ADF 사전 확인
    print("\n  [ADF 사전 확인 — VAR 변수들의 정상성]")
    non_stat = []
    for col in base_order:
        try:
            _, p_val, *_ = adfuller(data[col].dropna(), autolag="AIC")
            mark = "✓" if p_val < 0.05 else "✗"
            print(f"    {mark} {col:28s} p={p_val:.4f}")
            if p_val >= 0.05:
                non_stat.append(col)
        except Exception:
            pass

    if non_stat:
        print(f"\n  ⚠️  비정상 변수 {len(non_stat)}개 포함됨: {non_stat}")
        print(f"      → IRF 해석 시 이 한계 고려 필요 (논문 한계 섹션 명시)")
    else:
        print(f"\n  ✓ 모든 VAR 변수 정상 (I(0))")

    # ★ C3: Cholesky 순서 명시
    print("\n  [Cholesky 분해 순서 — 변수 순서가 IRF에 영향을 줌]")
    print("  기본 순서 (이론적 근거: 통화 → 빠른 자산 → 거시 결과):")
    for i, v in enumerate(base_order, 1):
        print(f"    {i}. {v}")
    print("  ※ 앞 변수의 충격은 동시점(t=0)에 뒷변수에 영향 가능,")
    print("    뒷변수의 충격은 동시점에 앞변수에 영향 안 줌 (Cholesky 가정).")

    irf_results = {}
    try:
        results, irf, base_peaks, opt_lag = _fit_var_and_peaks(
            data, base_order, C.VAR_MAX_LAG)
        print(f"\n  ✓ VAR 최적 시차 (AIC): {opt_lag}개월")

        if "Real_Rate" in base_order:
            shock_idx = base_order.index("Real_Rate")
            asset_map = {
                "Gold_LogReturn":        "금 (Gold)",
                "WTI_LogReturn":         "WTI 원유",
                "SP500_LogReturn":       "S&P500",
                "CaseShiller_LogReturn": "부동산",
                "CPI_LogReturn":         "CPI",
            }

            print("\n  📊 IRF 기반 자산별 최대 반응 시점 (Real_Rate 충격):")
            for col, label in asset_map.items():
                if col not in base_order:
                    continue
                resp_idx = base_order.index(col)
                irf_vals = irf.irfs[:, resp_idx, shock_idx]
                peak_m   = int(np.argmax(np.abs(irf_vals)))
                peak_v   = float(irf_vals[peak_m])
                print(f"    {label:15s}: {peak_m:2d}개월 후  "
                      f"(반응크기: {peak_v:+.4f})")
                irf_results[col] = {
                    "label":      label,
                    "peak_month": peak_m,
                    "peak_value": peak_v,
                    "irf_values": irf_vals.tolist()
                }

        # ★ C3: 대안 순서 강건성 검증 — 자산 순서 역전
        print("\n  [강건성 검증 — 대안 Cholesky 순서로 재추정]")
        alt_order = [c for c in [
            "Real_Rate", "QE_Size", "M2_YoY", "TIPS_Spread",
            "CPI_LogReturn", "CaseShiller_LogReturn",
            "SP500_LogReturn", "WTI_LogReturn", "Gold_LogReturn"
        ] if c in df.columns]

        try:
            _, _, alt_peaks, _ = _fit_var_and_peaks(
                data, alt_order, C.VAR_MAX_LAG)

            print("  자산별 peak month 비교 (기본 순서 vs 자산 역순):")
            print(f"  {'자산':25s} {'기본':>8} {'대안':>8} {'차이':>8}")
            print("  " + "-" * 55)
            max_diff = 0
            for col, label in asset_map.items():
                if col not in base_peaks or col not in alt_peaks:
                    continue
                bp = base_peaks[col]
                ap = alt_peaks[col]
                diff = abs(bp - ap)
                max_diff = max(max_diff, diff)
                mark = "" if diff <= 2 else "  ⚠"
                print(f"  {label:25s} {bp:>8d} {ap:>8d} {diff:>8d}{mark}")

            if max_diff <= 2:
                print(f"\n  ✓ 강건성 양호: 최대 차이 {max_diff}개월 (≤2)")
            else:
                print(f"\n  ⚠️  순서 의존성 존재: 최대 차이 {max_diff}개월")
                print(f"      → 논문에 Cholesky 순서 정당화 필요")
        except Exception as e:
            print(f"  ⚠️  대안 순서 추정 실패: {e}")

        # ★ D2: Real_Rate 위치 변경 강건성 (자산 뒤로 이동)
        #   C3는 자산 5개만 역순으로 했으나 Cholesky 첫 변수 위치는 동일
        #   → Real_Rate 자체를 자산 뒤로 옮긴 진짜 sensitivity test
        print("\n  [강건성 검증 2 — Real_Rate를 자산 뒤로 이동]")
        alt2_order = [c for c in [
            "QE_Size", "M2_YoY", "TIPS_Spread",
            "Gold_LogReturn", "WTI_LogReturn",
            "SP500_LogReturn", "CaseShiller_LogReturn",
            "CPI_LogReturn",
            "Real_Rate"   # ← 자산 뒤로
        ] if c in df.columns]

        try:
            _, _, alt2_peaks, _ = _fit_var_and_peaks(
                data, alt2_order, C.VAR_MAX_LAG)
            print("  자산별 peak month 비교 (기본 vs Real_Rate 후순위):")
            print(f"  {'자산':25s} {'기본':>8} {'대안2':>8} {'차이':>8}")
            print("  " + "-" * 55)
            max_diff2 = 0
            for col, label in asset_map.items():
                if col not in base_peaks or col not in alt2_peaks:
                    continue
                bp = base_peaks[col]
                ap = alt2_peaks[col]
                diff = abs(bp - ap)
                max_diff2 = max(max_diff2, diff)
                mark = "" if diff <= 2 else "  ⚠"
                print(f"  {label:25s} {bp:>8d} {ap:>8d} {diff:>8d}{mark}")
            if max_diff2 <= 2:
                print(f"\n  ✓ Real_Rate 위치 강건성 양호: 최대 차이 {max_diff2}개월")
            else:
                print(f"\n  ⚠️  Real_Rate 위치 의존성 존재: 최대 차이 {max_diff2}개월")
                print(f"      → 논문에 통화→자산 식별 순서 정당화 필수")
        except Exception as e:
            print(f"  ⚠️  대안2 순서 추정 실패: {e}")

        irf_df = pd.DataFrame([
            {"asset": info["label"],
             "peak_month": info["peak_month"],
             "peak_value": info["peak_value"]}
            for info in irf_results.values()
        ])
        if not irf_df.empty:
            path = os.path.join(C.RESULT_DIR, "irf_results.csv")
            irf_df.to_csv(path, index=False)
            print(f"\n  ✓ 저장: {path}")

        return results, irf_results, irf

    except Exception as e:
        print(f"  ⚠️  VAR/IRF 실패: {e}")
        return None, {}, None


# ──────────────────────────────────────────────
#  이벤트 스터디  (★ C4: 누적수익률 + baseline 수정)
# ──────────────────────────────────────────────

def run_event_study(df):
    """
    ★ C4: 누적수익률 계산 수정 + 이벤트 시점 기준 정규화
       - (1+r).cumprod() → np.exp(r.cumsum()) (로그수익률 정확 누적)
       - 이벤트 시점을 baseline(=0)으로 재정렬하여 사전 트렌드 제거

    ★ D3: Half-Peak Time 도입 (반응 시차 metric)
       - 기존 peak month는 단조 증가 데이터에서 윈도우 끝(20mo)에 쏠림
       - 신규: 최대 누적 반응의 50%에 처음 도달하는 시점(half_month)
       - half-life 방법론 기반 (PPP/transmission 연구 표준)
       - peak month는 보조 정보로 유지
       - 6/12/24개월 누적반응도 부가 출력
    """
    print("\n  [3-4] 이벤트 스터디 (v8: 누적수익률 + baseline + half-peak)")

    asset_cols = {
        "금 (Gold)":  "Gold_LogReturn",
        "WTI 원유":   "WTI_LogReturn",
        "S&P500":     "SP500_LogReturn",
        "부동산":      "CaseShiller_LogReturn",
        "CPI":        "CPI_LogReturn",
    }

    event_dates = []
    for start, _ in C.RATE_CUT_CYCLES:
        ts    = pd.Timestamp(start)
        valid = df.index[df.index >= ts]
        if len(valid) > 0:
            event_dates.append(valid[0])

    window   = 24      # 사후 24개월
    pre      = 6       # 사전 6개월
    all_rets = {label: [] for label in asset_cols}
    peak_months    = {}
    half_months    = {}     # ★ D3: half-peak time
    horizon_resps  = {}     # ★ D3: 6/12/24개월 누적반응

    for event_date in event_dates:
        idx = df.index.get_loc(event_date)
        for label, col in asset_cols.items():
            if col not in df.columns:
                continue
            s_idx  = max(0, idx - pre)
            e_idx  = min(len(df), idx + window + 1)
            series = df[col].iloc[s_idx:e_idx]
            if series.empty:
                continue

            # ★ C4: 로그수익률을 올바르게 누적 (cumsum)
            log_cum = series.cumsum()

            # ★ C4: 이벤트 시점을 baseline(=0)으로
            event_pos_in_series = idx - s_idx
            if event_pos_in_series >= len(log_cum):
                continue
            log_cum_centered = log_cum - log_cum.iloc[event_pos_in_series]

            # 단순수익률 환산 (해석 용이, % 단위)
            cumret = (np.exp(log_cum_centered) - 1) * 100
            all_rets[label].append(cumret.values[:window+pre+1])

    print("\n  📊 이벤트 스터디 반응 시차 (이벤트 시점 = 0):")
    print(f"  {'자산':12s} {'Half월':>7} {'Peak월':>7} {'Peak값':>9} "
          f"{'6mo':>7} {'12mo':>7} {'24mo':>9}")
    print("  " + "-" * 70)

    # ★ D3 + minor fix (sample bias 방지): 두 가지 평균 분리
    #  (1) peak/half는 모든 사이클 가용한 horizon만 (min_len 일관 표본)
    #  (2) 6/12/24mo는 nanmean (시점별 가용 사이클 평균, 표본 크기 추적)
    target_len = window + pre + 1   # = 31
    n_24mo_warnings = []

    for label, rets in all_rets.items():
        if not rets:
            continue

        # ─── (1) 일관 표본 (5사이클 모두 가용) — peak/half용 ───
        min_len = min(len(r) for r in rets)
        if min_len <= pre:
            continue
        consistent_avg = np.mean([r[:min_len] for r in rets], axis=0)
        post_consistent = consistent_avg[pre:]   # 5사이클 평균

        # peak month (보조 정보)
        peak_t = int(np.argmax(np.abs(post_consistent)))
        peak_v = float(post_consistent[peak_t])
        peak_months[label] = peak_t

        # ★ D3: Half-peak time
        threshold = 0.5 * peak_v
        if peak_v > 0:
            crossed = np.where(post_consistent >= threshold)[0]
        elif peak_v < 0:
            crossed = np.where(post_consistent <= threshold)[0]
        else:
            crossed = np.array([0])
        half_t = int(crossed[0]) if len(crossed) > 0 else peak_t
        half_months[label] = half_t

        # ─── (2) nanmean 표본 — 6/12/24mo 보조 ───
        padded = []
        for r in rets:
            arr = np.full(target_len, np.nan)
            arr[:len(r)] = r[:target_len]
            padded.append(arr)
        nanmean_avg = np.nanmean(padded, axis=0)
        post_nanmean = nanmean_avg[pre:]
        n_per_horizon = np.sum(
            np.isfinite(np.array(padded)[:, pre:]), axis=0)

        def _resp_at(h):
            if h >= len(post_nanmean):
                return float("nan"), 0
            v = post_nanmean[h]
            return (float(v) if np.isfinite(v) else float("nan"),
                    int(n_per_horizon[h]) if h < len(n_per_horizon) else 0)

        h6, n6   = _resp_at(6)
        h12, n12 = _resp_at(12)
        h24, n24 = _resp_at(24)
        horizon_resps[label] = {
            "6mo": h6, "12mo": h12, "24mo": h24,
            "n_6mo": n6, "n_12mo": n12, "n_24mo": n24,
        }
        if n24 > 0 and n24 < n6:
            n_24mo_warnings.append((label, n24, n6))

        # 출력 — 24mo는 별표 (표본 다를 때)
        def _fmt_pct(v):
            return f"{v:>+6.1f}%" if np.isfinite(v) else f"  n/a "
        h24_marker = "*" if (n24 > 0 and n24 < n6) else " "
        h24_str = (f"{h24:>+6.1f}%{h24_marker}"
                   if np.isfinite(h24) else f"  n/a  ")

        print(f"  {label:12s} {half_t:>7d} {peak_t:>7d} "
              f"{peak_v:>+8.2f}% {_fmt_pct(h6)} {_fmt_pct(h12)} {h24_str}")

    # 24mo 표본 차이 footnote
    if n_24mo_warnings:
        lb, n24, n6 = n_24mo_warnings[0]
        print(f"  * 24mo는 일부 사이클이 미도달 (n={n24}, 다른 horizon n={n6}). "
              f"진행중인 2024 사이클 제외됨.")

    # CSV 저장 — 확장된 컬럼
    ev_df = pd.DataFrame([
        {"asset":      label,
         "half_month": half_months.get(label),
         "peak_month": peak_months.get(label),
         "cumret_6m":  horizon_resps.get(label, {}).get("6mo"),
         "cumret_12m": horizon_resps.get(label, {}).get("12mo"),
         "cumret_24m": horizon_resps.get(label, {}).get("24mo"),
         }
        for label in peak_months.keys()
    ])
    if not ev_df.empty:
        path = os.path.join(C.RESULT_DIR, "event_study_results.csv")
        ev_df.to_csv(path, index=False)
        print(f"\n  ✓ 저장: {path}")

    return peak_months, all_rets, half_months


# ──────────────────────────────────────────────
#  칸티용 전이 순서  (★ C5: 순위 평균)
# ──────────────────────────────────────────────

def derive_cantillon_order(granger_df, irf_results, event_peaks):
    """
    ★ C5: 평균 lag → 순위 평균
       - 각 방법(그랜저/IRF/이벤트) 내에서 자산 1~5위 순위 매김
       - 방법별 순위를 평균하여 최종 순서 결정
       - 표시: 원본 lag + 방법별 순위 + 평균 순위

    ★ D3: event_peaks 파라미터는 이제 half-peak time (peak month가 아닌
          최대 누적 반응의 50%에 도달하는 시점). PPP/transmission 연구에서
          사용되는 half-life methodology 기반의 speed-of-adjustment metric.
          → 단조 증가 데이터에서 윈도우 끝(20mo)에 쏠리는 문제 해결
          → timing 차원 보존 (Cantillon transmission ordering 핵심)
    """
    print("\n  [3-5] 칸티용 전이 순서 자동 도출 (v8: 순위 평균)")

    asset_labels = {
        "Gold_LogReturn":        "금 (Gold)",
        "WTI_LogReturn":         "WTI 원유",
        "SP500_LogReturn":       "S&P500",
        "CaseShiller_LogReturn": "부동산",
        "CPI_LogReturn":         "CPI",
    }

    # 1) 방법별 자산 → lag 수집
    method_lags = {"granger": {}, "irf": {}, "event": {}}

    if not granger_df.empty:
        sig_df = granger_df[granger_df["significant"]]
        for effect, label in asset_labels.items():
            lags = sig_df[sig_df["effect"] == effect]["lag"]
            if len(lags) > 0:
                method_lags["granger"][label] = float(lags.mean())

    for col, info in irf_results.items():
        label = asset_labels.get(col, col)
        method_lags["irf"][label] = float(info["peak_month"])

    for label, month in event_peaks.items():
        method_lags["event"][label] = float(month)

    # 2) 방법별 순위 (작은 lag = 빠른 반응 = 1위)
    method_ranks = {}
    for method, lag_dict in method_lags.items():
        if not lag_dict:
            method_ranks[method] = {}
            continue
        sorted_assets = sorted(lag_dict.items(), key=lambda x: x[1])
        method_ranks[method] = {
            label: rank + 1 for rank, (label, _) in enumerate(sorted_assets)
        }

    # 3) 모든 자산 라벨 수집
    all_labels = set()
    for ld in method_lags.values():
        all_labels.update(ld.keys())

    def _fmt(v, w, num_fmt="{:.1f}"):
        if v is None:
            return " " * w
        return f"{num_fmt.format(v):>{w}}"

    # 4) 평균 순위
    print(f"\n  {'자산':12s} "
          f"{'G_lag':>7} {'I_lag':>7} {'E_half':>7}  "
          f"{'G순위':>6} {'I순위':>6} {'E순위':>6} "
          f"{'평균순위':>8}")
    print("  " + "-" * 75)

    final_rows = []
    for label in sorted(all_labels):
        g_lag = method_lags["granger"].get(label)
        i_lag = method_lags["irf"].get(label)
        e_lag = method_lags["event"].get(label)
        g_rk  = method_ranks["granger"].get(label)
        i_rk  = method_ranks["irf"].get(label)
        e_rk  = method_ranks["event"].get(label)

        ranks = [r for r in (g_rk, i_rk, e_rk) if r is not None]
        avg_rk = float(np.mean(ranks)) if ranks else 999.0

        print(f"  {label:12s} "
              f"{_fmt(g_lag, 7)} {_fmt(i_lag, 7)} {_fmt(e_lag, 7)}  "
              f"{_fmt(g_rk, 6, '{:d}')} {_fmt(i_rk, 6, '{:d}')} "
              f"{_fmt(e_rk, 6, '{:d}')} {avg_rk:>8.2f}")

        final_rows.append({
            "asset":        label,
            "granger_lag":  g_lag,
            "irf_lag":      i_lag,
            "event_lag":    e_lag,
            "granger_rank": g_rk,
            "irf_rank":     i_rk,
            "event_rank":   e_rk,
            "avg_rank":     avg_rk,
        })

    # 평균 순위 기준 정렬
    final_rows.sort(key=lambda x: x["avg_rank"])

    print("\n  🏆 데이터 기반 칸티용 전이 순서 (순위 평균 기준):")
    print("  " + "=" * 55)
    for rank, info in enumerate(final_rows, 1):
        print(f"  {rank}위: {info['asset']:12s} "
              f"(평균 순위 {info['avg_rank']:.2f})")
    print("  " + "=" * 55)

    hypothesis = ["금 (Gold)", "WTI 원유", "S&P500", "부동산", "CPI"]
    actual     = [info["asset"] for info in final_rows]
    print("\n  📋 가설 vs 실제:")
    for rank, (hyp, act) in enumerate(
            zip(hypothesis, actual[:len(hypothesis)]), 1):
        match = "✅" if hyp == act else "❌"
        print(f"  {rank}위: 가설={hyp:10s}  실제={act:10s}  {match}")

    order_df = pd.DataFrame(final_rows)
    path = os.path.join(C.RESULT_DIR, "cantillon_order.csv")
    order_df.to_csv(path, index=False)
    print(f"\n  ✓ 저장: {path}")

    # 호환성: (label, avg_rank) 튜플 리스트
    return [(info["asset"], info["avg_rank"]) for info in final_rows]


# ──────────────────────────────────────────────
#  메인
# ──────────────────────────────────────────────

def main():
    print("\n[03] 실증 분석 (v8: PHASE C1~C5)")

    proc_path = os.path.join(C.DATA_PROC_DIR, "processed_data.csv")
    if not os.path.exists(proc_path):
        print("  ⚠️  processed_data.csv 없음 → 02 먼저 실행")
        return

    df = pd.read_csv(proc_path, index_col=0, parse_dates=True)
    print(f"  ✓ 전처리 데이터 로드: {df.shape}")

    adf_results = run_adf_test(df)
    granger_df, lag_t, pval_t = run_granger_analysis(df)
    var_results, irf_results, irf_obj = run_var_irf(df)
    event_peaks, all_rets, half_months = run_event_study(df)  # ★ D3
    # ★ D3: half-peak time을 이벤트 timing metric으로 사용
    final_order = derive_cantillon_order(
        granger_df, irf_results, half_months)

    print("\n  ✅ 실증 분석 완료 (v8 C1~C5 + D1~D3)")
    return {
        "adf":         adf_results,
        "granger":     granger_df,
        "lag_table":   lag_t,
        "pval_table":  pval_t,
        "irf":         irf_results,
        "irf_obj":     irf_obj,
        "event":       event_peaks,
        "event_half":  half_months,    # ★ D3
        "all_rets":    all_rets,
        "order":       final_order,
    }


if __name__ == "__main__":
    main()
