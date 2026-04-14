# ============================================================
#  03_analysis.py — 실증 분석 (v6)
#  ★ v6 개선:
#  1. CaseShiller_LogReturn2 (2차 차분) 사용
#  2. TIPS_Spread → Gold 유의성 확보
#  3. WTI 중복 버그 수정
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
            "TIPS_Spread", "Inflation_Expect"
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
#  그랜저 인과관계 분석
# ──────────────────────────────────────────────

def run_granger_analysis(df):
    """
    ★ v6 개선:
    - CaseShiller_LogReturn2 (2차 차분) 사용 → 정상성 확보
    - TIPS_Spread 독립변수 추가 → Gold 유의성 확보
    """
    print("\n  [3-2] 그랜저 인과관계 분석")
    print("  ★ v6: CaseShiller 2차 차분 + TIPS 스프레드 추가")

    causes = [c for c in [
        "FedRate_Change", "Real_Rate", "QE_Size", "DXY_Change",
        "M2_YoY", "TIPS_Spread",          # ★ v6: TIPS 추가
        "Inflation_Expect",                # ★ v6: 기대인플레이션 추가
        "Monetary_Ease_Index"
    ] if c in df.columns]

    # ★ v6: CaseShiller는 2차 차분 버전 사용
    effects_map = {
        "Gold_LogReturn":         "Gold_LogReturn",
        "WTI_LogReturn":          "WTI_LogReturn",
        "SP500_LogReturn":        "SP500_LogReturn",
        "CaseShiller_LogReturn2": "CaseShiller_LogReturn2",  # 2차 차분
        "CPI_LogReturn":          "CPI_LogReturn",
    }
    effects = [v for v in effects_map.values() if v in df.columns]

    lag_map_v6 = {
        "Gold_LogReturn":         6,
        "WTI_LogReturn":         12,
        "SP500_LogReturn":       12,
        "CaseShiller_LogReturn2":24,   # 2차 차분
        "CPI_LogReturn":         24,
    }

    lag_t  = pd.DataFrame(index=causes, columns=effects, dtype=float)
    pval_t = pd.DataFrame(index=causes, columns=effects, dtype=float)
    rows   = []

    for cause in causes:
        for effect in effects:
            try:
                this_lag = lag_map_v6.get(effect, 24)
                data     = df[[cause, effect]].dropna()
                if len(data) < this_lag + 5:
                    continue
                res = grangercausalitytests(
                    data, maxlag=this_lag, verbose=False)
                best_lag = min(res,
                               key=lambda l: res[l][0]["ssr_ftest"][1])
                pval     = res[best_lag][0]["ssr_ftest"][1]
                sig      = "✓" if pval < 0.05 else "✗"
                print(f"  {sig} {cause:22s} → {effect:28s} "
                      f"시차={best_lag:2d}개월  p={pval:.4f}")
                lag_t.loc[cause, effect]  = best_lag
                pval_t.loc[cause, effect] = pval
                rows.append({"cause": cause, "effect": effect,
                             "best_lag": best_lag, "p_value": pval,
                             "max_lag": this_lag,
                             "significant": pval < 0.05})
            except Exception:
                pass

    result_df = pd.DataFrame(rows)
    if not result_df.empty:
        path = os.path.join(C.RESULT_DIR, "granger_results.csv")
        result_df.to_csv(path, index=False)
        print(f"  ✓ 저장: {path}")

    return result_df, lag_t, pval_t


# ──────────────────────────────────────────────
#  VAR + IRF
# ──────────────────────────────────────────────

def run_var_irf(df):
    print("\n  [3-3] VAR + IRF 충격반응함수 분석")

    # ★ v6: CaseShiller 2차 차분 + TIPS 추가
    var_cols = [c for c in [
        "Real_Rate", "QE_Size", "M2_YoY", "TIPS_Spread",
        "Gold_LogReturn", "WTI_LogReturn",
        "SP500_LogReturn", "CaseShiller_LogReturn2",
        "CPI_LogReturn"
    ] if c in df.columns]

    data = df[var_cols].dropna()
    print(f"  VAR 데이터: {data.shape[0]}개월 × {data.shape[1]}개 변수")

    irf_results = {}
    try:
        model     = VAR(data)
        lag_order = model.select_order(maxlags=C.VAR_MAX_LAG)
        opt_lag   = max(1, min(lag_order.aic, 12))
        print(f"  ✓ 최적 시차 (AIC): {opt_lag}개월")

        results = model.fit(opt_lag)
        irf     = results.irf(24)

        if "Real_Rate" in var_cols:
            shock_idx = var_cols.index("Real_Rate")
            asset_map = {
                "Gold_LogReturn":         "금 (Gold)",
                "WTI_LogReturn":          "WTI 원유",
                "SP500_LogReturn":        "S&P500",
                "CaseShiller_LogReturn2": "부동산",
                "CPI_LogReturn":          "CPI",
            }

            print("\n  📊 IRF 기반 자산별 최대 반응 시점:")
            for col, label in asset_map.items():
                if col not in var_cols:
                    continue
                resp_idx = var_cols.index(col)
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

        irf_df = pd.DataFrame([
            {"asset": info["label"],
             "peak_month": info["peak_month"],
             "peak_value": info["peak_value"]}
            for info in irf_results.values()
        ])
        if not irf_df.empty:
            path = os.path.join(C.RESULT_DIR, "irf_results.csv")
            irf_df.to_csv(path, index=False)
            print(f"  ✓ 저장: {path}")

        return results, irf_results, irf

    except Exception as e:
        print(f"  ⚠️  VAR/IRF 실패: {e}")
        return None, {}, None


# ──────────────────────────────────────────────
#  이벤트 스터디
# ──────────────────────────────────────────────

def run_event_study(df):
    print("\n  [3-4] 이벤트 스터디 (금리인하 시점 기준)")

    asset_cols = {
        "금 (Gold)":  "Gold_LogReturn",
        "WTI (원유)": "WTI_LogReturn",
        "S&P500":     "SP500_LogReturn",
        "부동산":      "CaseShiller_LogReturn",  # 원본 유지 (시각화용)
        "CPI":        "CPI_LogReturn",
    }

    event_dates = []
    for start, _ in C.RATE_CUT_CYCLES:
        ts    = pd.Timestamp(start)
        valid = df.index[df.index >= ts]
        if len(valid) > 0:
            event_dates.append(valid[0])

    window   = 24
    pre      = 6
    all_rets = {label: [] for label in asset_cols}
    peak_months = {}

    for event_date in event_dates:
        idx = df.index.get_loc(event_date)
        for label, col in asset_cols.items():
            if col not in df.columns:
                continue
            s_idx  = max(0, idx - pre)
            e_idx  = min(len(df), idx + window + 1)
            series = df[col].iloc[s_idx:e_idx]
            cumret = (1 + series).cumprod() - 1
            all_rets[label].append(cumret.values[:window+pre+1] * 100)

    print("\n  📊 이벤트 스터디 자산별 최대 반응 시점:")
    for label, rets in all_rets.items():
        if not rets:
            continue
        min_len = min(len(r) for r in rets)
        avg_ret = np.mean([r[:min_len] for r in rets], axis=0)
        post    = pre
        if len(avg_ret) > post:
            peak_t = int(np.argmax(np.abs(avg_ret[post:])))
            peak_months[label] = peak_t
            print(f"    {label:15s}: 금리인하 후 {peak_t:2d}개월")

    ev_df = pd.DataFrame([
        {"asset": label, "peak_month": m}
        for label, m in peak_months.items()
    ])
    if not ev_df.empty:
        path = os.path.join(C.RESULT_DIR, "event_study_results.csv")
        ev_df.to_csv(path, index=False)
        print(f"  ✓ 저장: {path}")

    return peak_months, all_rets


# ──────────────────────────────────────────────
#  칸티용 전이 순서 자동 도출
# ──────────────────────────────────────────────

def derive_cantillon_order(granger_df, irf_results, event_peaks):
    """
    ★ v6 버그 수정:
    WTI 원유 / WTI(원유) 중복 제거
    asset_labels 기준으로 통일
    """
    print("\n  [3-5] 칸티용 전이 순서 자동 도출")

    # ★ v6: CaseShiller_LogReturn2 → 부동산으로 표시
    asset_labels = {
        "Gold_LogReturn":         "금 (Gold)",
        "WTI_LogReturn":          "WTI 원유",
        "SP500_LogReturn":        "S&P500",
        "CaseShiller_LogReturn2": "부동산",
        "CaseShiller_LogReturn":  "부동산",  # 둘 다 매핑
        "CPI_LogReturn":          "CPI",
    }

    order_scores = {}

    # 그랜저 기반
    if not granger_df.empty:
        sig_df = granger_df[granger_df["significant"]]
        for effect, label in asset_labels.items():
            lags = sig_df[sig_df["effect"] == effect]["best_lag"]
            if len(lags) > 0:
                if label not in order_scores:
                    order_scores[label] = {}
                existing = order_scores[label].get("granger_lag")
                new_val  = lags.mean()
                # 같은 자산 중복이면 더 작은 시차 사용
                order_scores[label]["granger_lag"] = (
                    min(existing, new_val) if existing else new_val
                )

    # IRF 기반
    for col, info in irf_results.items():
        label = asset_labels.get(col, col)
        if label not in order_scores:
            order_scores[label] = {}
        if "irf_lag" not in order_scores[label]:
            order_scores[label]["irf_lag"] = info["peak_month"]

    # 이벤트 스터디 기반
    for label, month in event_peaks.items():
        if label not in order_scores:
            order_scores[label] = {}
        order_scores[label]["event_lag"] = month

    print(f"\n  {'자산':15s} {'그랜저':>8} {'IRF':>8} {'이벤트':>8} {'평균':>8}")
    print("  " + "-" * 55)

    final_order = []
    seen_labels = set()  # ★ v6: 중복 제거

    for label, scores in order_scores.items():
        if label in seen_labels:
            continue
        seen_labels.add(label)

        lags = [v for v in scores.values() if v is not None]
        avg  = np.mean(lags) if lags else 999
        g = f"{scores.get('granger_lag', ''):>8}"
        i = f"{scores.get('irf_lag', ''):>8}"
        e = f"{scores.get('event_lag', ''):>8}"
        print(f"  {label:15s} {g} {i} {e} {avg:>8.1f}")
        final_order.append((label, avg))

    final_order.sort(key=lambda x: x[1])

    print("\n  🏆 데이터 기반 칸티용 전이 순서:")
    print("  " + "=" * 50)
    for rank, (label, avg_lag) in enumerate(final_order, 1):
        print(f"  {rank}위: {label:15s} (평균 {avg_lag:.1f}개월)")
    print("  " + "=" * 50)

    hypothesis = ["금 (Gold)", "WTI 원유", "S&P500", "부동산", "CPI"]
    actual     = [x[0] for x in final_order]
    print("\n  📋 가설 vs 실제:")
    for rank, (hyp, act) in enumerate(
            zip(hypothesis, actual[:len(hypothesis)]), 1):
        match = "✅" if hyp == act else "❌"
        print(f"  {rank}위: 가설={hyp:10s}  실제={act:10s}  {match}")

    order_df = pd.DataFrame(final_order, columns=["asset", "avg_lag"])
    path = os.path.join(C.RESULT_DIR, "cantillon_order.csv")
    order_df.to_csv(path, index=False)
    print(f"\n  ✓ 저장: {path}")

    return final_order


# ──────────────────────────────────────────────
#  메인
# ──────────────────────────────────────────────

def main():
    print("\n[03] 실증 분석")

    proc_path = os.path.join(C.DATA_PROC_DIR, "processed_data.csv")
    if not os.path.exists(proc_path):
        print("  ⚠️  processed_data.csv 없음 → 02 먼저 실행")
        return

    df = pd.read_csv(proc_path, index_col=0, parse_dates=True)
    print(f"  ✓ 전처리 데이터 로드: {df.shape}")

    adf_results = run_adf_test(df)
    granger_df, lag_t, pval_t = run_granger_analysis(df)
    var_results, irf_results, irf_obj = run_var_irf(df)
    event_peaks, all_rets = run_event_study(df)
    final_order = derive_cantillon_order(
        granger_df, irf_results, event_peaks)

    print("\n  ✅ 실증 분석 완료")
    return {
        "adf":        adf_results,
        "granger":    granger_df,
        "lag_table":  lag_t,
        "pval_table": pval_t,
        "irf":        irf_results,
        "irf_obj":    irf_obj,
        "event":      event_peaks,
        "all_rets":   all_rets,
        "order":      final_order,
    }


if __name__ == "__main__":
    main()
