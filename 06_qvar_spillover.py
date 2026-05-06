"""
06_qvar_spillover.py — Quantile-VAR Spillover Connectedness Analysis (v7)
==========================================================================
Variables: CPI, Oil (WTI), Gold, M2, S&P500, Case-Shiller HPI
Period: 1998-01 ~ present (US monthly data)

Based on Chatziantoniou et al. (2021) QVAR framework.
Produces:
  1) Time series plots
  2) Log-return plots
  3) Return distributions vs Normal
  4) Correlation matrix heatmap
  5) Summary statistics table (CSV)
  6) Quantile-VAR spillover connectedness at τ = 0.05, 0.50, 0.95

Author: 이찬수
v7 통합: config.py의 경로 설정 연동
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import platform
from scipy import stats
from statsmodels.tsa.stattools import acf
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C


def set_font():
    system = platform.system()
    if system == "Windows":
        plt.rcParams["font.family"] = "Malgun Gothic"
    elif system == "Darwin":
        plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False


# ★ v7: config.py 경로 연동
OUT_DIR = C.RESULT_DIR
FIG_DIR = C.FIG_DIR

START = "1998-01-01"
END   = "2026-03-31"


# ============================================================
# 1. 데이터 수집
# ============================================================
def fetch_data(api_key: str) -> pd.DataFrame:
    try:
        from fredapi import Fred
    except ImportError:
        print("  ⚠️  pip install fredapi")
        return None

    fred = Fred(api_key=api_key)

    def fred_get(series_id, retries=3, delay=5):
        """FRED 서버 에러 시 재시도"""
        import time
        for attempt in range(retries):
            try:
                return fred.get_series(series_id,
                    observation_start=START, observation_end=END)
            except Exception as e:
                if attempt < retries - 1:
                    print(f"    재시도 {attempt+1}/{retries}: {series_id}")
                    time.sleep(delay)
                else:
                    raise e

    # ★ 캐시된 데이터 먼저 확인
    import os
    cache_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data", "raw", "qvar_cache.csv")

    if os.path.exists(cache_path):
        print("  ✓ 캐시 데이터 로드 (FRED 서버 요청 생략)")
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    print("  Fetching CPI (CPIAUCSL)...")
    cpi = fred_get('CPIAUCSL')

    print("  Fetching M2 (WM2NS)...")
    try:
        m2 = fred_get('WM2NS')
    except Exception:
        print("    WM2NS failed, trying M2SL...")
        m2 = fred_get('M2SL')

    print("  Fetching Gold via yfinance (GC=F)...")
    gold_yf = yf.download("GC=F", start=START, end=END,
                           interval="1mo", progress=False)
    if isinstance(gold_yf.columns, pd.MultiIndex):
        gold = gold_yf[('Close', 'GC=F')]
    else:
        gold = gold_yf['Close']
    gold.index = gold.index.to_period('M').to_timestamp()

    print("  Fetching WTI Oil (DCOILWTICO)...")
    try:
        oil_daily = fred.get_series('DCOILWTICO',
                                    observation_start=START,
                                    observation_end=END)
        oil = oil_daily.resample('MS').mean()
    except Exception:
        print("    FRED oil failed, using yfinance CL=F...")
        oil_yf = yf.download("CL=F", start=START, end=END,
                              interval="1mo", progress=False)
        if isinstance(oil_yf.columns, pd.MultiIndex):
            oil = oil_yf[('Close', 'CL=F')]
        else:
            oil = oil_yf['Close']
        oil.index = oil.index.to_period('M').to_timestamp()

    print("  Fetching Case-Shiller HPI (CSUSHPINSA)...")
    cs = fred_get('CSUSHPINSA')

    print("  Fetching S&P 500 (^GSPC)...")
    sp = yf.download("^GSPC", start=START, end=END,
                     interval="1mo", progress=False)
    if isinstance(sp.columns, pd.MultiIndex):
        sp500 = sp[('Close', '^GSPC')]
    else:
        sp500 = sp['Close']
    sp500.index = sp500.index.to_period('M').to_timestamp()

    df = pd.DataFrame({
        'CPI': cpi, 'Oil': oil, 'Gold': gold,
        'M2': m2, 'SP500': sp500, 'CaseShiller': cs
    })
    df.index = pd.to_datetime(df.index)
    df = df.resample('MS').last().dropna()
    print(f"  Data: {df.shape}, "
          f"{df.index[0].date()} ~ {df.index[-1].date()}")

    # ★ 캐시 저장 (다음 실행 시 FRED 서버 에러 방지)
    try:
        import os
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_csv(cache_path)
        print(f"  ✓ 캐시 저장: {cache_path}")
    except Exception:
        pass

    return df


# ============================================================
# 2. 수익률 변환
# ============================================================
def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    ret = np.log(df / df.shift(1)) * 100
    return ret.dropna()


# ============================================================
# 3. Summary Statistics
# ============================================================
def summary_statistics(ret: pd.DataFrame) -> pd.DataFrame:
    from statsmodels.tsa.stattools import adfuller
    results = {}
    for col in ret.columns:
        x = ret[col].dropna().values
        n = len(x)
        jb_stat, _ = stats.jarque_bera(x)
        adf_stat, *_ = adfuller(
            x, maxlag=int(np.floor(12*(n/100)**(1/4))), autolag='AIC')
        acf_vals = acf(x, nlags=20, fft=True)
        q20 = n*(n+2)*np.sum(acf_vals[1:21]**2 / (n - np.arange(1, 21)))
        acf2 = acf(x**2, nlags=20, fft=True)
        q2_20 = n*(n+2)*np.sum(acf2[1:21]**2 / (n - np.arange(1, 21)))
        results[col] = {
            'Mean':     round(np.mean(x), 3),
            'Skewness': round(stats.skew(x), 3),
            'Kurtosis': round(stats.kurtosis(x, fisher=False), 3),
            'JB':       round(jb_stat, 2),
            'ADF':      round(adf_stat, 3),
            'Q(20)':    round(q20, 3),
            'Q²(20)':   round(q2_20, 3),
        }
    return pd.DataFrame(results)


# ============================================================
# 4. 시각화
# ============================================================
def plot_timeseries(df, save_path):
    set_font()
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    for ax, col, title in zip(
        axes.flat,
        ['CPI','Oil','Gold','M2','SP500','CaseShiller'],
        ['a) CPI','b) Oil (WTI)','c) Gold','d) M2',
         'e) S&P 500','f) Case-Shiller HPI']
    ):
        ax.plot(df.index, df[col], color='black', linewidth=0.8)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {os.path.basename(save_path)}")


def plot_returns(ret, save_path):
    set_font()
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    for ax, col, title in zip(
        axes.flat,
        ['CPI','Oil','Gold','M2','SP500','CaseShiller'],
        ['a) CPI','b) Oil (WTI)','c) Gold','d) M2',
         'e) S&P 500','f) Case-Shiller HPI']
    ):
        ax.plot(ret.index, ret[col], color='black', linewidth=0.5)
        ax.axhline(0, color='grey', linewidth=0.4, linestyle='--')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    plt.suptitle("Monthly Log Returns (%)",
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {os.path.basename(save_path)}")


def plot_distributions(ret, save_path):
    from scipy.stats import gaussian_kde
    set_font()
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    for ax, col, title in zip(
        axes.flat,
        ['CPI','Oil','Gold','M2','SP500','CaseShiller'],
        ['a) CPI','b) Oil (WTI)','c) Gold','d) M2',
         'e) S&P 500','f) Case-Shiller HPI']
    ):
        x = ret[col].dropna()
        ax.hist(x, bins=40, density=True,
                color='#cccccc', edgecolor='white')
        kde = gaussian_kde(x)
        xg  = np.linspace(x.min(), x.max(), 200)
        ax.plot(xg, kde(xg), color='black', linewidth=1.5, label='KDE')
        ax.plot(xg, stats.norm.pdf(xg, x.mean(), x.std()),
                'r--', linewidth=1, label='Normal')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    plt.suptitle("Return Distributions vs Normal",
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {os.path.basename(save_path)}")


def plot_correlation(ret, save_path):
    set_font()
    corr = ret.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, fontsize=11)
    ax.set_yticks(range(len(corr)))
    ax.set_yticklabels(corr.columns, fontsize=11)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i,j]:.2f}",
                    ha='center', va='center', fontsize=10,
                    color='white' if abs(corr.values[i,j])>0.5 else 'black')
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlation Matrix (Log Returns)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {os.path.basename(save_path)}")


def plot_summary_table(table, save_path):
    set_font()
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis('off')
    tbl = ax.table(
        cellText=table.T.values,
        rowLabels=table.columns,
        colLabels=table.index,
        cellLoc='center', loc='center'
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0 or c == -1:
            cell.set_text_props(fontweight='bold')
            cell.set_facecolor('#e6e6e6')
    ax.set_title("Summary Statistics (Monthly Log Returns, %)",
                 fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {os.path.basename(save_path)}")


# ============================================================
# 5. Quantile-VAR Core
# ============================================================
def quantile_var_estimate(Y, p, tau):
    from statsmodels.regression.quantile_regression import QuantReg
    T, k = Y.shape
    X_rows, y_rows = [], []
    for t in range(p, T):
        row = [1.0]
        for j in range(1, p+1):
            row.extend(Y[t-j])
        X_rows.append(row)
        y_rows.append(Y[t])
    X, Ymat = np.array(X_rows), np.array(y_rows)
    coefs     = []
    residuals = np.zeros((Ymat.shape[0], k))
    for i in range(k):
        res = QuantReg(Ymat[:,i], X).fit(q=tau, max_iter=1000)
        coefs.append(res.params)
        residuals[:,i] = Ymat[:,i] - X @ res.params
    coef_matrix = np.array(coefs)
    mu  = coef_matrix[:,0]
    phi = [coef_matrix[:, 1+j*k:1+(j+1)*k] for j in range(p)]
    return mu, phi, residuals


def companion_form(phi_list, k):
    p = len(phi_list); Kp = k*p
    A = np.zeros((Kp, Kp))
    for j, phi_j in enumerate(phi_list):
        A[:k, j*k:(j+1)*k] = phi_j
    if p > 1:
        A[k:, :k*(p-1)] = np.eye(k*(p-1))
    return A


def fevd_spillover(phi_list, Sigma, H=10):
    k = Sigma.shape[0]; p = len(phi_list)
    A = companion_form(phi_list, k)
    J = np.zeros((k, k*p)); J[:k,:k] = np.eye(k)
    Theta  = []; A_power = np.eye(k*p)
    for h in range(H):
        Theta.append(J @ A_power @ J.T)
        A_power = A_power @ A
    sigma_diag = np.diag(Sigma)
    Psi = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            num = 0.0; denom = 0.0
            e_j = np.zeros(k); e_j[j] = 1.0
            for h in range(H):
                num   += (Theta[h][i,:] @ Sigma @ e_j)**2
                denom += Theta[h][i,:] @ Sigma @ Theta[h][i,:]
            Psi[i,j] = ((1.0/sigma_diag[j])*num/denom
                        if denom > 0 else 0)
    return Psi / Psi.sum(axis=1, keepdims=True)


def spillover_table(Psi_tilde, var_names):
    k = len(var_names)
    from_others = Psi_tilde.sum(axis=1) - np.diag(Psi_tilde)
    to_others   = Psi_tilde.sum(axis=0) - np.diag(Psi_tilde)
    net         = to_others - from_others
    tci         = from_others.sum() / (k-1)
    df = pd.DataFrame(Psi_tilde*100, index=var_names, columns=var_names)
    df['FROM'] = from_others*100
    df.loc['TO']  = list(to_others*100) + [from_others.sum()*100]
    df.loc['NET'] = list(net*100) + [np.nan]
    return df, tci*100


# ============================================================
# 6. Main
# ============================================================
def main():
    print("\n[06] QVAR Spillover 분석 (경기국면별 전이 구조)")

    if not C.FRED_API_KEY or C.FRED_API_KEY == "your_fred_api_key_here":
        print("  ⚠️  FRED_API_KEY 없음 → .env 파일 확인")
        return

    df_raw = fetch_data(C.FRED_API_KEY)
    if df_raw is None:
        return

    ret = compute_returns(df_raw)

    # 시각화
    plot_timeseries(df_raw,
                    os.path.join(FIG_DIR, "qvar_timeseries.png"))
    plot_returns(ret,
                 os.path.join(FIG_DIR, "qvar_returns.png"))
    plot_distributions(ret,
                       os.path.join(FIG_DIR, "qvar_distributions.png"))
    plot_correlation(ret,
                     os.path.join(FIG_DIR, "qvar_correlation.png"))

    # 요약 통계
    table = summary_statistics(ret)
    plot_summary_table(table,
                       os.path.join(FIG_DIR, "qvar_summary_table.png"))
    table.T.to_csv(os.path.join(OUT_DIR, "qvar_summary_stats.csv"))
    print(f"\n  Summary Statistics:\n{table.T.to_string()}")

    # QVAR Spillover
    print("\n  Quantile-VAR Spillover Analysis")
    Y, var_names = ret.values, list(ret.columns)

    for tau in [0.05, 0.50, 0.95]:
        label = {0.05:"침체기",0.50:"중립기",0.95:"과열기"}[tau]
        print(f"\n  --- τ = {tau} ({label}) ---")
        mu, phi, resid = quantile_var_estimate(Y, p=1, tau=tau)
        Psi = fevd_spillover(phi, np.cov(resid.T), H=10)
        tbl, tci = spillover_table(Psi, var_names)
        print(tbl.round(2).to_string())
        print(f"\n  TCI: {tci:.2f}%")
        fname = f"qvar_spillover_tau_{int(tau*100):02d}.csv"
        tbl.round(2).to_csv(os.path.join(OUT_DIR, fname))
        print(f"  ✓ 저장: {fname}")

    print("\n  ✅ QVAR Spillover 분석 완료!")


if __name__ == "__main__":
    main()
