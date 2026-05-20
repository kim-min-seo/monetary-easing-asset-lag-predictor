# ============================================================
#  03_analysis.py — 실증 분석 (v8 — PHASE C1 + C2)
#
#  v8 변경 사항:
#  ─────────────────────────────────────────────────────────
#  [C1] Granger 인과분석: best_lag p-해킹 제거
#       - 기존: maxlag까지 모든 시차의 p값 계산 후 가장 작은 p값의
#               시차를 "best_lag"로 보고 → 시차 선택과 검정 통계량이
#               같은 데이터에서 나와 p값이 인위적으로 부풀려짐 (p-hacking)
#       - 수정: 양변량 VAR로 AIC 최적 시차를 먼저 결정한 후, 그 단일
#               시차에서만 그랜저 검정 실시. 시차 선택과 검정 분리.
#
#  [C2] VAR 적분차수 통일: CaseShiller_LogReturn2 → CaseShiller_LogReturn
#       - 기존: CaseShiller만 2차 차분(_LogReturn2), 다른 자산은 1차 차분
#               (_LogReturn) → VAR 변수들의 경제적 의미가 불일치
#               ("월간 수익률" vs "월간 수익률의 변화량") → IRF 해석의
#               자산 간 비교가 사실상 불가능
#       - 수정: 모든 자산을 _LogReturn (1차 차분)으로 통일. ADF 사전
#               확인을 추가하여 비정상 변수가 포함되었을 때 명시적으로
#               경고하고 한계로 보고. CaseShiller의 약한 정상성은
#               논문 한계 섹션에서 다룸.
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
#  그랜저 인과관계 분석  (★ C1: p-해킹 제거)
# ──────────────────────────────────────────────

def run_granger_analysis(df):
    """
    ★ C1: best_lag 선택 방식 변경
       - 기존: maxlag까지 검정 후 가장 작은 p값의 시차를 best_lag로 선정
       - 변경: 양변량 VAR의 AIC로 시차 선택 후 그 단일 시차에서만 검정

    ★ C2: CaseShiller_LogReturn (1차 차분) 사용 → 다른 자산과 적분차수 통일
    """
    print("\n  [3-2] 그랜저 인과관계 분석 (v8: AIC 시차 선택 + 적분차수 통일)")

    causes = [c for c in [
        "FedRate_Change", "Real_Rate", "QE_Size", "DXY_Change",
        "M2_YoY", "TIPS_Spread", "Inflation_Expect",
        "Monetary_Ease_Index"
    ] if c in df.columns]

    # ★ C2: CaseShiller_LogReturn (1차 차분) 사용
    effects_map = {
        "Gold_LogReturn":        "Gold_LogReturn",
        "WTI_LogReturn":         "WTI_LogReturn",
        "SP500_LogReturn":       "SP500_LogReturn",
        "CaseShiller_LogReturn": "CaseShiller_LogReturn",
        "CPI_LogReturn":         "CPI_LogReturn",
    }
    effects = [v for v in effects_map.values() if v in df.columns]

    # 자산별 max_lag (AIC 탐색의 상한선)
    # - 빠른 시장 반응 자산은 짧게 (Gold, WTI, SP500)
    # - 느리게 반응하는 거시 자산은 길게 (CaseShiller, CPI)
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

                # ★ C1: 양변량 VAR AIC로 시차 선택 (p-해킹 방지)
                try:
                    bv_model = VAR(data)
                    sel      = bv_model.select_order(maxlags=this_max)
                    aic_lag  = int(max(1, min(sel.aic, this_max)))
                except Exception:
                    # AIC 선택 실패 시 보수적 기본값 (월간 데이터 표준)
                    aic_lag = min(6, this_max)

                # 선택된 단일 시차에서 그랜저 검정
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

        # 유의 결과 요약
        n_sig = result_df["significant"].sum()
        print(f"  ✓ 유의 관계 (p<0.05): {n_sig}/{len(result_df)}개")

    return result_df, lag_t, pval_t


# ──────────────────────────────────────────────
#  VAR + IRF  (★ C2: 적분차수 통일)
# ──────────────────────────────────────────────

def run_var_irf(df):
    """
    ★ C2: CaseShiller_LogReturn (1차 차분) 사용
       - 다른 자산(_LogReturn)과 적분차수 통일
       - ADF 사전 확인 후 비정상 변수가 있으면 경고 출력
    """
    print("\n  [3-3] VAR + IRF 충격반응함수 분석 (v8: 적분차수 통일)")

    # ★ C2: CaseShiller_LogReturn (1차 차분) — 다른 자산과 단위 일치
    var_cols = [c for c in [
        "Real_Rate", "QE_Size", "M2_YoY", "TIPS_Spread",
        "Gold_LogReturn", "WTI_LogReturn",
        "SP500_LogReturn", "CaseShiller_LogReturn",
        "CPI_LogReturn"
    ] if c in df.columns]

    data = df[var_cols].dropna()
    print(f"  VAR 데이터: {data.shape[0]}개월 × {data.shape[1]}개 변수")

    # ★ C2: ADF 사전 확인 — 비정상 변수가 있으면 명시적으로 경고
    print("\n  [ADF 사전 확인 — VAR 변수들의 정상성]")
    non_stat = []
    for col in var_cols:
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

    irf_results = {}
    try:
        model     = VAR(data)
        lag_order = model.select_order(maxlags=C.VAR_MAX_LAG)
        opt_lag   = max(1, min(lag_order.aic, 12))
        print(f"\n  ✓ VAR 최적 시차 (AIC): {opt_lag}개월")

        results = model.fit(opt_lag)
        irf     = results.irf(24)

        if "Real_Rate" in var_cols:
            shock_idx = var_cols.index("Real_Rate")
            # ★ C2: CaseShiller_LogReturn 로 매핑 통일
            asset_map = {
                "Gold_LogReturn":        "금 (Gold)",
                "WTI_LogReturn":         "WTI 원유",
                "SP500_LogReturn":       "S&P500",
                "CaseShiller_LogReturn": "부동산",
                "CPI_LogReturn":         "CPI",
            }

            print("\n  📊 IRF 기반 자산별 최대 반응 시점 (Real_Rate 충격):")
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
#  이벤트 스터디  (PHASE C에서는 미변경, C4에서 수정 예정)
# ──────────────────────────────────────────────

def run_event_study(df):
    print("\n  [3-4] 이벤트 스터디 (금리인하 시점 기준)")

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
#  칸티용 전이 순서 자동 도출  (★ C2: 매핑 단순화)
# ──────────────────────────────────────────────

def derive_cantillon_order(granger_df, irf_results, event_peaks):
    """
    ★ v8 변경:
    - C2: CaseShiller_LogReturn 단일 사용 → 이전의 _LogReturn / _LogReturn2
          이중 매핑 제거. asset_labels 깔끔하게 1:1
    - C1 후속: best_lag → lag 컬럼명 변경 반영
    """
    print("\n  [3-5] 칸티용 전이 순서 자동 도출")

    # ★ v8: CaseShiller_LogReturn 단일 사용 (이전 이중 매핑 제거)
    asset_labels = {
        "Gold_LogReturn":        "금 (Gold)",
        "WTI_LogReturn":         "WTI 원유",
        "SP500_LogReturn":       "S&P500",
        "CaseShiller_LogReturn": "부동산",
        "CPI_LogReturn":         "CPI",
    }

    order_scores = {}

    # 그랜저 기반 — 유의 결과만 사용
    if not granger_df.empty:
        sig_df = granger_df[granger_df["significant"]]
        for effect, label in asset_labels.items():
            lags = sig_df[sig_df["effect"] == effect]["lag"]
            if len(lags) > 0:
                if label not in order_scores:
                    order_scores[label] = {}
                order_scores[label]["granger_lag"] = lags.mean()

    # IRF 기반
    for col, info in irf_results.items():
        label = asset_labels.get(col, col)
        if label not in order_scores:
            order_scores[label] = {}
        order_scores[label]["irf_lag"] = info["peak_month"]

    # 이벤트 스터디 기반
    for label, month in event_peaks.items():
        if label not in order_scores:
            order_scores[label] = {}
        order_scores[label]["event_lag"] = month

    print(f"\n  {'자산':15s} {'그랜저':>8} {'IRF':>8} {'이벤트':>8} {'평균':>8}")
    print("  " + "-" * 55)

    final_order = []
    for label, scores in order_scores.items():
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
    print("\n[03] 실증 분석 (v8: PHASE C1 + C2)")

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

    print("\n  ✅ 실증 분석 완료 (v8 C1+C2)")
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
