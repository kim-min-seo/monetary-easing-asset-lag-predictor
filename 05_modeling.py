# ============================================================
#  05_modeling.py — 예측 모델 (v8 Clean)
#
#  v7 → v8 핵심 변경:
#  1. StandardScaler 완전 제거 (트리 모델은 스케일 불변)
#     → 자동으로 "make_direction_labels(스케일된 y)" 레이블 버그 해결
#  2. 피처 선택을 "Granger 우선 + 공통 피처" 고정 리스트로 전환
#     → 데이터 기반 동적 선택 제거 (누수 차단)
#  3. Walk-forward는 각 폴드 내부에서만 학습 (누수 0)
#  4. Raw 베이스라인도 동일한 WF 방식으로 — 가공 모델과 공정 비교 (B4)
#  5. 깨진 앙상블 로직 제거 → 분류기 단독 사용
#  6. y는 항상 원본 수익률 유지
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

from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                              accuracy_score, f1_score,
                              balanced_accuracy_score, matthews_corrcoef)
import xgboost as xgb
import lightgbm as lgb
import shap


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
#  자산별 고정 피처 (★ v8 A안: 실제 Granger 유의 관계 기반)
#  ※ 03_analysis의 granger_results.csv에서 p<0.05인
#    (cause, AIC선택시차) 쌍만 사용 — C1(AIC 단일 시차)과 정합
#  ※ 형식: (변수명, 시차). 05 main에서 원본 변수를 해당 시차만큼
#    shift하여 피처 생성 (과거값 → 누수 없음).
#  ※ 02의 고정 LAG_PERIODS에 없는 정확한 시차도 사용 가능.
#  ※ CaseShiller는 유의한 Granger cause가 없음 → 공통 피처만
#    (= "부동산은 월간 통화변수로 설명되지 않음" v8 메시지)
# ============================================================
GRANGER_PRIORITY = {
    "Gold": [
        ("Real_Rate", 4), ("QE_Size", 3), ("TIPS_Spread", 4),
    ],
    "WTI": [
        ("Real_Rate", 12), ("M2_YoY", 12),
        ("TIPS_Spread", 2), ("Monetary_Ease_Index", 4),
    ],
    "SP500": [
        ("Real_Rate", 2), ("TIPS_Spread", 1),
        ("Monetary_Ease_Index", 7),
    ],
    "CaseShiller": [],   # 유의한 Granger cause 없음 → 공통 피처만
    "CPI": [
        ("FedRate_Change", 3), ("Real_Rate", 15),
        ("M2_YoY", 15), ("TIPS_Spread", 5),
    ],
}

# 모든 자산에 공통으로 들어가는 피처 (국면 + 사이클 더미)
COMMON_FEATURES = [
    "Regime_Recession", "Regime_Neutral", "Regime_Overheating",
    "Regime_Index",
    "Cut_Start", "Cut_Period", "Hike_Start", "Easing_Period",
    "Monetary_Ease_Index",
]

# Raw 베이스라인 변수 (피처 엔지니어링 0)
RAW_COLS = [
    "FedRate", "Fed_Assets", "T10Y", "T2Y",
    "CPI", "M2", "CaseShiller",
    "TIPS_10Y", "PPI", "PPI_Core",
    "Gold", "WTI", "DXY", "SP500", "VIX",
]

# 선행연구 벤치마크
PRIOR_BENCHMARKS = {
    "Gold":        {"paper": "Anzuini et al. (2010, ECB)",     "perf": 58.0},
    "WTI":         {"paper": "Browne & Cronin (2010)",         "perf": 55.0},
    "SP500":       {"paper": "Bernanke & Kuttner (2005)",      "perf": 62.0},
    "CaseShiller": {"paper": "Iacoviello (2005)",              "perf": 65.0},
    "CPI":         {"paper": "Aruoba & Drechsel (2024)",       "perf": 63.0},
}


# ============================================================
#  레이블 (★ 항상 원본 수익률에서)
# ============================================================
def make_direction_labels(y_raw):
    """원본 수익률 → 방향 레이블. 스케일된 y를 절대 넣지 말 것."""
    return (np.asarray(y_raw) > 0).astype(int)


# ============================================================
#  평가 지표
# ============================================================
def compute_reg_metrics(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    mae   = mean_absolute_error(y_true, y_pred)
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    smape = np.mean(2 * np.abs(y_pred - y_true) /
                    (np.abs(y_true) + np.abs(y_pred) + eps)) * 100
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / (ss_tot + eps)
    return {"MAE": mae, "RMSE": rmse, "sMAPE": smape, "R2": r2}


def compute_clf_metrics(y_true_bin, y_pred_bin):
    acc  = accuracy_score(y_true_bin, y_pred_bin) * 100
    bal  = balanced_accuracy_score(y_true_bin, y_pred_bin) * 100
    f1   = f1_score(y_true_bin, y_pred_bin, zero_division=0) * 100
    mcc  = matthews_corrcoef(y_true_bin, y_pred_bin) * 100
    base = np.mean(y_true_bin) * 100
    return {"Accuracy": acc, "Balanced_Acc": bal, "F1": f1,
            "MCC": mcc, "Base_Rate": base, "Real_Gain": acc - base}
    
# ─── 단일 클래스 대비 더미 분류기 (CaseShiller 등 상승 편중 자산용) ───
class _DummyClf:
    """train 데이터에 클래스가 하나뿐일 때 그 클래스를 그대로 예측."""
    def __init__(self, cls):
        self.cls = int(cls)
    def predict_proba(self, X):
        n = len(X)
        proba = np.zeros((n, 2))
        proba[:, self.cls] = 1.0
        return proba
    def predict(self, X):
        return np.full(len(X), self.cls)

def print_clf_metrics(m, label="분류 성능"):
    bar = "-" * 50
    print(f"\n  {bar}\n  {label}\n  {bar}")
    print(f"  Accuracy      : {m['Accuracy']:.1f}%")
    print(f"  Balanced Acc  : {m['Balanced_Acc']:.1f}%")
    print(f"  F1 Score      : {m['F1']:.1f}%")
    print(f"  MCC           : {m['MCC']:+.1f}%")
    print(f"  Base Rate     : {m['Base_Rate']:.1f}%  (무조건 상승 시 정확도)")
    print(f"  Real Gain     : {m['Real_Gain']:+.1f}%p  (vs Base Rate)")
    print(f"  {bar}")


# ============================================================
#  모델 학습 (모두 원본 y)
# ============================================================
def train_clf_xgb(X_tr, y_tr_raw, X_val=None, y_val_raw=None):
    dir_tr = make_direction_labels(y_tr_raw)
    # ★ 단일 클래스 안전 처리
    if len(np.unique(dir_tr)) < 2:
        return _DummyClf(dir_tr[0])
    base_params = dict(
        n_estimators=600, learning_rate=0.03, max_depth=5,
        min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.5, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbosity=0,
        eval_metric="logloss", base_score=0.5,
        use_label_encoder=False,
    )
    if X_val is not None and y_val_raw is not None:
        dir_val = make_direction_labels(y_val_raw)
        model = xgb.XGBClassifier(**base_params, early_stopping_rounds=50)
        model.fit(X_tr, dir_tr, eval_set=[(X_val, dir_val)], verbose=False)
    else:
        model = xgb.XGBClassifier(**base_params)
        model.fit(X_tr, dir_tr)
    return model


def train_clf_lgb(X_tr, y_tr_raw, X_val=None, y_val_raw=None):
    dir_tr = make_direction_labels(y_tr_raw)
    if len(np.unique(dir_tr)) < 2:
        return _DummyClf(dir_tr[0])
    model = lgb.LGBMClassifier(
        n_estimators=600, learning_rate=0.03, max_depth=5,
        num_leaves=31, min_child_samples=15,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.5, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbose=-1,
        is_unbalance=True,
    )
    if X_val is not None and y_val_raw is not None:
        dir_val = make_direction_labels(y_val_raw)
        model.fit(X_tr, dir_tr, eval_set=[(X_val, dir_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                             lgb.log_evaluation(period=9999)])
    else:
        model.fit(X_tr, dir_tr)
    return model


def train_reg_xgb(X_tr, y_tr_raw, X_val=None, y_val_raw=None):
    """회귀 (보조 — 수익률 크기 예측). y는 원본 단위."""
    base_params = dict(
        n_estimators=600, learning_rate=0.03, max_depth=5,
        min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbosity=0,
        eval_metric="rmse",
    )
    if X_val is not None and y_val_raw is not None:
        model = xgb.XGBRegressor(**base_params, early_stopping_rounds=50)
        model.fit(X_tr, y_tr_raw, eval_set=[(X_val, y_val_raw)], verbose=False)
    else:
        model = xgb.XGBRegressor(**base_params)
        model.fit(X_tr, y_tr_raw)
    return model


# ============================================================
#  Walk-forward (누수 제거판)
# ============================================================
def walk_forward(X, y, label="WF", n_splits=None, min_train=None, use_lgb=False):
    """각 폴드 내부에서만 학습. 스케일링 없음."""
    if n_splits  is None: n_splits  = C.WF_SPLITS
    if min_train is None: min_train = C.MIN_TRAIN

    n = len(X)
    test_size = max(1, (n - min_train) // n_splits)

    all_probs, all_dir_true, all_y_te = [], [], []
    fold_records = []

    print(f"\n  {label} ({n_splits}폴드, {'LGB' if use_lgb else 'XGB'})")
    print(f"  {'폴드':>4} | {'기간':^22} | {'Acc':>7} | {'Base':>7} | {'Gain':>7}")
    print("  " + "-" * 64)

    for fold in range(n_splits):
        tr_end = min_train + fold * test_size
        te_end = min(tr_end + test_size, n)
        if te_end > n or tr_end <= 10:
            break

        # 폴드 내부 train/val (early stopping용)
        val_size = max(10, int(tr_end * 0.10))
        X_tr = X.iloc[:tr_end - val_size]; y_tr = y.iloc[:tr_end - val_size]
        X_v  = X.iloc[tr_end - val_size:tr_end]
        y_v  = y.iloc[tr_end - val_size:tr_end]
        X_te = X.iloc[tr_end:te_end]; y_te = y.iloc[tr_end:te_end]

        if len(X_tr) < 20:
            continue

        if use_lgb:
            clf = train_clf_lgb(X_tr, y_tr.values, X_v, y_v.values)
        else:
            clf = train_clf_xgb(X_tr, y_tr.values, X_v, y_v.values)

        probs = clf.predict_proba(X_te)[:, 1]
        dir_te = make_direction_labels(y_te.values)
        pred = (probs > 0.5).astype(int)
        acc = accuracy_score(dir_te, pred) * 100
        base = np.mean(dir_te) * 100

        if hasattr(X_te.index, "strftime"):
            date_str = (f"{X_te.index[0].strftime('%Y-%m')}~"
                        f"{X_te.index[-1].strftime('%Y-%m')}")
        else:
            date_str = f"{tr_end}~{te_end}"
        print(f"  {fold+1:>4} | {date_str:^22} | "
              f"{acc:>6.1f}% | {base:>6.1f}% | {acc-base:+6.1f}%p")

        all_probs.extend(probs); all_dir_true.extend(dir_te)
        all_y_te.extend(y_te.values)
        fold_records.append({"fold": fold + 1, "acc": acc, "base": base})

    if not fold_records:
        return {"avg_acc": np.nan, "avg_base": np.nan,
                "probs": np.array([]), "dir_true": np.array([]),
                "y_te": np.array([]), "fold_records": []}

    avg_acc  = np.mean([r["acc"]  for r in fold_records])
    avg_base = np.mean([r["base"] for r in fold_records])
    print("  " + "-" * 64)
    print(f"  {'평균':>4} | {'':^22} | "
          f"{avg_acc:>6.1f}% | {avg_base:>6.1f}% | {avg_acc-avg_base:+6.1f}%p")

    return {
        "avg_acc": avg_acc, "avg_base": avg_base,
        "probs": np.array(all_probs),
        "dir_true": np.array(all_dir_true),
        "y_te": np.array(all_y_te),
        "fold_records": fold_records,
    }


# ============================================================
#  Raw 베이스라인 (★ B4 — WF 기준으로 공정 비교)
# ============================================================
def run_raw_baseline_wf(df, asset_name, target_col):
    """피처 엔지니어링 없이 원본 변수만으로 WF 정확도 측정."""
    raw_cols = [c for c in RAW_COLS if c in df.columns and c != target_col]
    data = df[[target_col] + raw_cols].dropna()
    if len(data) < C.MIN_TRAIN + 30:
        print("  Raw 베이스라인: 데이터 부족"); return None

    print(f"\n  [Raw 베이스라인 — WF 기준] {asset_name} (변수 {len(raw_cols)}개)")
    res = walk_forward(data[raw_cols], data[target_col],
                       label="Raw WF", use_lgb=False)
    return res["avg_acc"]


# ============================================================
#  시각화
# ============================================================
def plot_backtest(wf_result, asset_name):
    set_font()
    if len(wf_result["probs"]) == 0:
        return
    probs   = wf_result["probs"]
    y_te    = wf_result["y_te"]
    dir_te  = wf_result["dir_true"]
    correct = (probs > 0.5).astype(int) == dir_te

    fig, axes = plt.subplots(3, 1, figsize=(14, 11))
    axes[0].plot(y_te, color="steelblue", lw=1.3, label="실제 수익률")
    axes[0].axhline(0, color="black", lw=0.8, ls="--")
    axes[0].set_title(f"{asset_name} — 실제 수익률 (Walk-forward 통합)",
                       fontsize=12, fontweight="bold")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].fill_between(range(len(probs)), 0.5, probs,
                         where=(probs > 0.5), color="tomato",
                         alpha=0.6, label="상승 확률")
    axes[1].fill_between(range(len(probs)), probs, 0.5,
                         where=(probs <= 0.5), color="steelblue",
                         alpha=0.6, label="하락 확률")
    axes[1].axhline(0.5, color="black", lw=1, ls="--")
    axes[1].set_ylim(0, 1); axes[1].legend()
    axes[1].set_title("방향성 분류 확률", fontsize=11)
    axes[1].grid(True, alpha=0.3)

    axes[2].bar(range(len(correct)), correct,
                color=np.where(correct, "seagreen", "indianred"), alpha=0.7)
    axes[2].set_ylim(-0.1, 1.1)
    axes[2].set_title("예측 적중 여부 (초록=맞음, 빨강=틀림)", fontsize=11)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(C.FIG_DIR, f"backtest_{asset_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  백테스트 저장: {path}")


def plot_model_comparison(all_metrics):
    set_font()
    if not all_metrics:
        return

    assets = list(all_metrics.keys())
    wf      = [all_metrics[a]["wf_acc"]   for a in assets]
    base    = [all_metrics[a]["wf_base"]  for a in assets]
    raw     = [all_metrics[a]["raw_wf_acc"] if not np.isnan(all_metrics[a]["raw_wf_acc"]) else 0
                for a in assets]
    prior   = [PRIOR_BENCHMARKS.get(a, {}).get("perf", 0) for a in assets]

    x = np.arange(len(assets)); w = 0.2
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(x - 1.5*w, base,  w, label="Base Rate",      color="#95a5a6")
    ax.bar(x - 0.5*w, raw,   w, label="Raw 모델 WF",     color="#3498db")
    ax.bar(x + 0.5*w, wf,    w, label="v8 모델 WF",      color="#e74c3c")
    ax.bar(x + 1.5*w, prior, w, label="선행연구 추정치", color="#f39c12")

    ax.axhline(50, color="black", lw=0.8, ls=":")
    ax.set_xticks(x); ax.set_xticklabels(assets)
    ax.set_ylabel("방향성 정확도 (%)")
    ax.set_title("자산별 모델 성능 비교 (v8 — WF 기준 공정 비교)",
                  fontsize=13, fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(C.FIG_DIR, "model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  비교 차트 저장: {path}")


def run_shap(model, X_test, asset_name):
    set_font()
    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_test)
        if isinstance(shap_vals, list) and len(shap_vals) > 1:
            shap_vals = shap_vals[1]  # 양성 클래스(상승)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_vals, X_test, plot_type="bar", show=False)
        plt.title(f"SHAP 기여도 — {asset_name} (v8)",
                  fontsize=12, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(C.FIG_DIR, f"shap_{asset_name}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  SHAP 저장: {path}")
    except Exception as e:
        print(f"  SHAP 실패: {e}")


def print_benchmark(all_metrics):
    print(f"\n{'='*72}")
    print("  선행연구 대비 성능 벤치마크 (WF 기준)")
    print(f"{'='*72}")
    print(f"  {'자산':^10} {'선행연구':>9} {'우리 WF':>9} {'차이':>8} {'논문'}")
    print("  " + "-" * 70)
    for asset, m in all_metrics.items():
        if asset not in PRIOR_BENCHMARKS:
            continue
        prior = PRIOR_BENCHMARKS[asset]
        diff  = m["wf_acc"] - prior["perf"]
        sign  = "+" if diff >= 0 else ""
        print(f"  {asset:^10} {prior['perf']:>8.1f}% {m['wf_acc']:>8.1f}% "
              f"{sign}{diff:>6.1f}%p  {prior['paper']}")
    print(f"{'='*72}")
    print("  * 선행연구 수치는 동일 조건(월간·통화변수) 기준 추정치")


# ============================================================
#  메인
# ============================================================
def main():
    print("\n[05] 예측 모델 (v8 Clean — 누수 제거 + 레이블 통일 + WF 공정 비교)")
    set_font()

    proc_path = os.path.join(C.DATA_PROC_DIR, "processed_data.csv")
    if not os.path.exists(proc_path):
        print("  processed_data.csv 없음 — 02 먼저 실행")
        return
    df = pd.read_csv(proc_path, index_col=0, parse_dates=True)
    print(f"  데이터 로드: {df.shape}")

    target_map = {
        "Gold":        "Gold_LogReturn",
        "WTI":         "WTI_LogReturn",
        "SP500":       "SP500_LogReturn",
        "CaseShiller": "CaseShiller_LogReturn",
        "CPI":         "CPI_LogReturn",
    }

    all_metrics = {}

    for asset_name, target_col in target_map.items():
        if target_col not in df.columns:
            print(f"\n  {target_col} 없음, 건너뜀"); continue

        print(f"\n{'='*64}")
        print(f"  ▶ 자산: {asset_name}  ({target_col})")
        print("=" * 64)

        # ── 고정 피처 (★ v8 A안: 원본 변수를 AIC 시차로 즉석 shift) ──
        df_local = df.copy()
        granger_feats = []
        for var, lag in GRANGER_PRIORITY.get(asset_name, []):
            if var in df_local.columns:
                col = f"{var}_glag{lag}"          # glag = Granger AIC lag
                df_local[col] = df_local[var].shift(lag)  # 과거값 → 누수 없음
                granger_feats.append(col)
            else:
                print(f"   원본 변수 없음: {var} (건너뜀)")
        common_feats  = [c for c in COMMON_FEATURES if c in df_local.columns]
        feat_cols = list(dict.fromkeys(granger_feats + common_feats))
        feat_cols = [c for c in feat_cols if c != target_col]

        if len(feat_cols) < 3:
            print(f"  피처 부족 ({len(feat_cols)}개) → 건너뜀"); continue
        print(f"  사용 피처 {len(feat_cols)}개 "
              f"(Granger {len(granger_feats)} + 공통 {len(common_feats)})")

        data = df_local[[target_col] + feat_cols].dropna()
        if len(data) < C.MIN_TRAIN + 30:
            print(f"  데이터 부족 ({len(data)}개)"); continue

        X = data[feat_cols]; y = data[target_col]
        n = len(data)
        dev_end = int(n * 0.85)
        X_dev = X.iloc[:dev_end]; y_dev = y.iloc[:dev_end]
        X_te  = X.iloc[dev_end:];  y_te  = y.iloc[dev_end:]
        print(f"  Dev: {len(X_dev)}개월, Hold-out: {len(X_te)}개월")

        # ── WF: XGB vs LGB ──
        wf_xgb = walk_forward(X_dev, y_dev, "XGB WF", use_lgb=False)
        wf_lgb = walk_forward(X_dev, y_dev, "LGB WF", use_lgb=True)

        if (not np.isnan(wf_lgb["avg_acc"])
                and wf_lgb["avg_acc"] >= wf_xgb["avg_acc"]):
            selected = "LGB"; wf_best = wf_lgb
        else:
            selected = "XGB"; wf_best = wf_xgb
        print(f"\n  선택 모델: {selected}  "
              f"(WF 평균 {wf_best['avg_acc']:.1f}%, "
              f"Base {wf_best['avg_base']:.1f}%, "
              f"Gain {wf_best['avg_acc']-wf_best['avg_base']:+.1f}%p)")

        # ── 최종 hold-out 평가 ──
        val_size = max(10, int(dev_end * 0.10))
        X_tr2 = X_dev.iloc[:-val_size]; y_tr2 = y_dev.iloc[:-val_size]
        X_v2  = X_dev.iloc[-val_size:]; y_v2  = y_dev.iloc[-val_size:]

        if selected == "LGB":
            final_clf = train_clf_lgb(X_tr2, y_tr2.values, X_v2, y_v2.values)
        else:
            final_clf = train_clf_xgb(X_tr2, y_tr2.values, X_v2, y_v2.values)

        te_probs    = final_clf.predict_proba(X_te)[:, 1]
        te_dir_true = make_direction_labels(y_te.values)
        te_pred     = (te_probs > 0.5).astype(int)
        ho_m        = compute_clf_metrics(te_dir_true, te_pred)
        print_clf_metrics(ho_m, f"{asset_name} — Hold-out (마지막 15%) 분류 성능")

        # ── 회귀 (보조 — 수익률 크기) ──
        final_reg = train_reg_xgb(X_tr2, y_tr2.values, X_v2, y_v2.values)
        te_pred_reg = final_reg.predict(X_te)
        reg_m = compute_reg_metrics(y_te.values, te_pred_reg)
        print(f"\n  회귀 보조:  RMSE={reg_m['RMSE']:.4f}  "
              f"sMAPE={reg_m['sMAPE']:.1f}%  R²={reg_m['R2']:.3f}")

        # ── Raw 베이스라인 (B4) ──
        raw_acc = run_raw_baseline_wf(df, asset_name, target_col)
        if raw_acc is not None and not np.isnan(raw_acc):
            print(f"\n  엔지니어링 효과 (WF 기준 공정 비교):")
            print(f"     Raw 모델 WF      : {raw_acc:.1f}%")
            print(f"     v8 모델 WF       : {wf_best['avg_acc']:.1f}%")
            print(f"     Base Rate        : {wf_best['avg_base']:.1f}%")
            print(f"     ★ Raw 대비 개선  : {wf_best['avg_acc']-raw_acc:+.1f}%p")

        # ── SHAP ──
        run_shap(final_clf, X_te, asset_name)

        # ── 백테스트 차트 ──
        plot_backtest(wf_best, asset_name)

        # ── 저장 ──
        all_metrics[asset_name] = {
            "wf_acc":      wf_best["avg_acc"],
            "wf_base":     wf_best["avg_base"],
            "wf_gain":     wf_best["avg_acc"] - wf_best["avg_base"],
            "ho_acc":      ho_m["Accuracy"],
            "ho_base":     ho_m["Base_Rate"],
            "ho_gain":     ho_m["Real_Gain"],
            "ho_mcc":      ho_m["MCC"],
            "ho_f1":       ho_m["F1"],
            "ho_bal":      ho_m["Balanced_Acc"],
            "raw_wf_acc":  raw_acc if raw_acc is not None else np.nan,
            "selected":    selected,
            "rmse":        reg_m["RMSE"],
            "r2":          reg_m["R2"],
        }

    # ── 종합 요약 ──
    print(f"\n{'='*84}")
    print("   v8 자산별 성능 요약 (★ WF가 메인 지표)")
    print("=" * 84)
    print(f"  {'자산':12s} {'WF':>7} {'Base':>7} {'Gain':>8} {'Raw_WF':>8} "
          f"{'HO_Acc':>8} {'HO_MCC':>8} {'모델':>5}")
    print("  " + "-" * 82)
    for asset, m in all_metrics.items():
        raw = f"{m['raw_wf_acc']:.1f}%" if not np.isnan(m['raw_wf_acc']) else "—"
        print(f"  {asset:12s} {m['wf_acc']:6.1f}% {m['wf_base']:6.1f}% "
              f"{m['wf_gain']:+7.1f}%p {raw:>7} "
              f"{m['ho_acc']:7.1f}% {m['ho_mcc']:+7.1f}% {m['selected']:>5}")
    print("=" * 84)

    perf_df = pd.DataFrame(all_metrics).T.round(3)
    perf_path = os.path.join(C.RESULT_DIR, "model_performance.csv")
    perf_df.to_csv(perf_path)
    print(f"\n  성능 저장: {perf_path}")

    # ── 비교 차트 + 벤치마크 ──
    plot_model_comparison(all_metrics)
    print_benchmark(all_metrics)

    print("\n   예측 모델 완료 (v8 Clean)")
    return all_metrics


if __name__ == "__main__":
    main()
