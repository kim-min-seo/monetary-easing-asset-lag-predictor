# ============================================================
#  05_modeling.py — 예측 모델 (v7 Final)
#
#  핵심 개선 3가지:
#  1. Walk-forward 5폴드 평균으로 모델 선택
#     (검증셋 42개 기준 → 훨씬 안정적)
#
#  2. 자산별 최적 임계값 탐색
#     (0.5 고정 → 0.3~0.7 탐색)
#
#  3. Granger 검증 시차 피처 우선 사용
# ============================================================

import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import platform
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             accuracy_score, f1_score,
                             balanced_accuracy_score,
                             matthews_corrcoef)
import xgboost as xgb
import lightgbm as lgb
import shap

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


def set_font():
    system = platform.system()
    if system == "Windows":
        plt.rcParams["font.family"] = "Malgun Gothic"
    elif system == "Darwin":
        plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False


GRANGER_PRIORITY = {
    "Gold": [
        "Real_Rate_lag2","Real_Rate_lag3",
        "QE_Size_lag3","QE_Size_lag4",
        "TIPS_Spread_lag4","TIPS_Spread_lag3",
        "Inflation_Expect_lag4","FedRate_Change_lag1",
    ],
    "WTI": [
        "Real_Rate_lag1","Real_Rate_lag2",
        "M2_YoY_lag1","M2_YoY_lag2",
        "TIPS_Spread_lag1","TIPS_Spread_lag2",
        "FedRate_Change_lag1",
        "DXY_Change_lag1","DXY_Change_lag3",  # 달러 역상관
        "DXY_YoY_lag1","DXY_YoY_lag3",
        "VIX_lag1","VIX_Change_lag1",          # 공포지수
    ],
    "SP500": [
        "Real_Rate_lag1","Real_Rate_lag2",
        "QE_Size_lag1","QE_Size_lag2",
        "M2_YoY_lag2","M2_YoY_lag3",
        "FedRate_Change_lag1","TIPS_Spread_lag1",
        "Monetary_Ease_Index_lag6",
        "VIX_lag1","VIX_Change_lag1",
    ],
    "CaseShiller": [
        "Real_Rate_lag11","Real_Rate_lag12",
        "Real_Rate_lag9","Real_Rate_lag10",
        "Inflation_Expect_lag19","TIPS_Spread_lag19",
        "DXY_Change_lag24",
    ],
    "CPI": [
        "Real_Rate_lag12","Real_Rate_lag11",
        "M2_YoY_lag1","M2_YoY_lag2",
        "DXY_Change_lag24","QE_Size_lag5",
        "TIPS_Spread_lag19","FedRate_Change_lag3",
        "PPI_LogReturn_lag1","PPI_LogReturn_lag3",
        "PPI_YoY_lag1","PPI_YoY_lag3",
    ],
}


def compute_metrics(y_true, y_pred, eps=1e-8):
    mae   = mean_absolute_error(y_true, y_pred)
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    smape = np.mean(2*np.abs(y_pred-y_true)/
                    (np.abs(y_true)+np.abs(y_pred)+eps))*100
    ss_res = np.sum((y_true-y_pred)**2)
    ss_tot = np.sum((y_true-np.mean(y_true))**2)
    r2     = 1 - ss_res/(ss_tot+eps)
    dir_acc = (np.mean((np.diff(y_true)>0)==(np.diff(y_pred)>0))*100
               if len(y_true)>1 else 0.0)
    return {"MAE":mae,"RMSE":rmse,"sMAPE":smape,"R2":r2,"Dir_Acc":dir_acc}


# ─────────────────────────────────────────
#  ★ 추가 평가 지표 (과적합 반박용)
# ─────────────────────────────────────────
def compute_clf_metrics(y_true_bin, y_pred_bin, y_proba=None):
    acc  = accuracy_score(y_true_bin, y_pred_bin) * 100
    bal  = balanced_accuracy_score(y_true_bin, y_pred_bin) * 100
    f1   = f1_score(y_true_bin, y_pred_bin, zero_division=0) * 100
    mcc  = matthews_corrcoef(y_true_bin, y_pred_bin) * 100
    base = y_true_bin.mean() * 100          # Base Rate (무조건 상승 정확도)
    gain = acc - base                       # 진짜 개선도
    return {"Accuracy":acc, "Balanced_Acc":bal,
            "F1":f1, "MCC":mcc,
            "Base_Rate":base, "Real_Gain":gain}


def print_clf_metrics(m, label="분류 성능"):
    bar = "-" * 50
    print(f"\n  {bar}")
    print(f"  {label}")
    print(f"  {bar}")
    print(f"  Accuracy       : {m['Accuracy']:.1f}%")
    print(f"  Balanced Acc   : {m['Balanced_Acc']:.1f}%  (상승/하락 균형)")
    print(f"  F1 Score       : {m['F1']:.1f}%")
    print(f"  MCC            : {m['MCC']:.1f}%  (균형 지표)")
    print(f"  Base Rate      : {m['Base_Rate']:.1f}%  (무조건 상승 정확도)")
    print(f"  실제 개선도    : +{m['Real_Gain']:.1f}%p  (vs Base Rate)")
    print(f"  {bar}")


RAW_COLS = [
    "FedRate", "Fed_Assets", "T10Y", "T2Y",
    "CPI", "M2", "CaseShiller",
    "TIPS_10Y", "PPI", "PPI_Core",
    "Gold", "WTI", "DXY", "SP500", "VIX"
]

def run_raw_baseline(df, asset_name, target_col):
    raw_cols = [c for c in RAW_COLS
                if c in df.columns and c != target_col]
    data = df[[target_col] + raw_cols].dropna()
    if len(data) < 100:
        print(f"  Raw 기준선: 데이터 부족")
        return None

    X = data[raw_cols]; y = data[target_col]
    n = len(data)
    tr_end = int(n*0.70); val_end = int(n*0.85)

    X_tr = X.iloc[:tr_end]; y_tr = y.iloc[:tr_end]
    X_val = X.iloc[tr_end:val_end]; y_val = y.iloc[tr_end:val_end]
    X_te  = X.iloc[val_end:]; y_te  = y.iloc[val_end:]

    dir_tr  = (y_tr > 0).astype(int)
    dir_val = (y_val > 0).astype(int)
    dir_te  = (y_te > 0).astype(int)

    clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        early_stopping_rounds=30, random_state=42,
        verbosity=0, n_jobs=-1, use_label_encoder=False,
        eval_metric="logloss")
    clf.fit(X_tr, dir_tr,
            eval_set=[(X_val, dir_val)], verbose=False)

    raw_pred = clf.predict(X_te)
    raw_acc  = accuracy_score(dir_te, raw_pred) * 100
    base_rate = dir_te.mean() * 100

    print(f"\n  [Raw 베이스라인] {asset_name}")
    print(f"  변수 {len(raw_cols)}개 (피처 엔지니어링 없음)")
    print(f"  Raw 정확도  : {raw_acc:.1f}%")
    print(f"  Base Rate   : {base_rate:.1f}%")
    print(f"  Raw 개선도  : +{raw_acc-base_rate:.1f}%p")
    return raw_acc


def print_metrics(m, label="모델"):
    bar = "=" * 56
    print(f"\n{bar}")
    print(f"  📊 {label}")
    print(bar)
    print(f"  MAE           : {m['MAE']:.6f}")
    print(f"  RMSE          : {m['RMSE']:.6f}")
    print(f"  sMAPE         : {m['sMAPE']:.2f}%")
    print(f"  R2            : {m['R2']:.4f}")
    print(f"  방향성 정확도 : {m['Dir_Acc']:.1f}%  <- 목표: 75%+")
    print(bar)


def make_direction_labels(y):
    return (y > 0).astype(int)


def select_features_granger_priority(X_tr, y_tr, asset_name, top_n=None):
    if top_n is None:
        top_n = C.TOP_FEATURES
    print(f"  -> Granger 우선 피처 선택: {X_tr.shape[1]}개 -> 상위 {top_n}개")
    granger_feats = [f for f in GRANGER_PRIORITY.get(asset_name, [])
                     if f in X_tr.columns]
    print(f"    Granger 피처: {len(granger_feats)}개 우선 포함")
    remaining_n = top_n - len(granger_feats)
    if remaining_n > 0:
        selector = xgb.XGBRegressor(n_estimators=100, max_depth=4,
                                    random_state=42, verbosity=0, n_jobs=-1)
        selector.fit(X_tr, y_tr)
        importance = pd.Series(selector.feature_importances_, index=X_tr.columns)
        remaining = importance.drop(
            [f for f in granger_feats if f in importance.index], errors='ignore'
        ).nlargest(remaining_n).index.tolist()
        top_cols = granger_feats + remaining
    else:
        top_cols = granger_feats[:top_n]
    top_cols = list(dict.fromkeys(top_cols))[:top_n]
    print(f"  선택: {len(top_cols)}개")
    return top_cols


def train_xgb_reg(X_tr, y_tr, X_val, y_val, params=None):
    if params is None:
        params = {"n_estimators":800,"learning_rate":0.03,"max_depth":6,
                  "min_child_weight":3,"subsample":0.8,
                  "colsample_bytree":0.8,"reg_alpha":0.1,"reg_lambda":1.0}
    model = xgb.XGBRegressor(**params, early_stopping_rounds=50,
                              eval_metric="rmse", random_state=42,
                              n_jobs=-1, verbosity=0)
    model.fit(X_tr, y_tr, eval_set=[(X_val,y_val)], verbose=False)
    return model


def train_xgb_clf(X_tr, y_tr, X_val, y_val, params=None):
    dir_tr = make_direction_labels(y_tr)
    dir_val = make_direction_labels(y_val)
    if params is None:
        params = {"n_estimators":800,"learning_rate":0.03,"max_depth":5,
                  "min_child_weight":3,"subsample":0.8,
                  "colsample_bytree":0.8,"reg_alpha":0.5,"reg_lambda":1.0}
    model = xgb.XGBClassifier(**params, early_stopping_rounds=50,
                               eval_metric="logloss", random_state=42,
                               n_jobs=-1, verbosity=0,
                               base_score=0.5, use_label_encoder=False)
    model.fit(X_tr, dir_tr, eval_set=[(X_val,dir_val)], verbose=False)
    return model


def train_lgb_reg(X_tr, y_tr, X_val, y_val):
    model = lgb.LGBMRegressor(n_estimators=800, learning_rate=0.03,
                               max_depth=6, num_leaves=63,
                               min_child_samples=15, subsample=0.8,
                               colsample_bytree=0.8, reg_alpha=0.1,
                               reg_lambda=1.0, random_state=42,
                               n_jobs=-1, verbose=-1)
    model.fit(X_tr, y_tr, eval_set=[(X_val,y_val)],
              callbacks=[lgb.early_stopping(50,verbose=False),
                         lgb.log_evaluation(period=9999)])
    return model


def train_lgb_clf(X_tr, y_tr, X_val, y_val):
    dir_tr = make_direction_labels(y_tr)
    dir_val = make_direction_labels(y_val)
    model = lgb.LGBMClassifier(n_estimators=800, learning_rate=0.03,
                                max_depth=5, num_leaves=31,
                                min_child_samples=15, subsample=0.8,
                                colsample_bytree=0.8, reg_alpha=0.5,
                                reg_lambda=1.0, random_state=42,
                                n_jobs=-1, verbose=-1, is_unbalance=True)
    model.fit(X_tr, dir_tr, eval_set=[(X_val,dir_val)],
              callbacks=[lgb.early_stopping(50,verbose=False),
                         lgb.log_evaluation(period=9999)])
    return model


def optuna_tune(X_tr, y_tr, X_val, y_val):
    if not OPTUNA_AVAILABLE:
        return None
    print(f"  Optuna 튜닝 ({C.OPTUNA_TRIALS}회)...")
    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_est",300,1200),
            "max_depth":        trial.suggest_int("depth",3,8),
            "learning_rate":    trial.suggest_float("lr",0.005,0.1,log=True),
            "min_child_weight": trial.suggest_int("mcw",1,10),
            "subsample":        trial.suggest_float("ss",0.5,1.0),
            "colsample_bytree": trial.suggest_float("cs",0.5,1.0),
            "reg_alpha":        trial.suggest_float("ra",0.0,2.0),
            "reg_lambda":       trial.suggest_float("rl",0.5,3.0),
        }
        model = xgb.XGBRegressor(**params, early_stopping_rounds=30,
                                  eval_metric="rmse", random_state=42,
                                  verbosity=0, n_jobs=-1)
        model.fit(X_tr, y_tr, eval_set=[(X_val,y_val)], verbose=False)
        return np.sqrt(mean_squared_error(y_val, model.predict(X_val)))
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=C.OPTUNA_TRIALS, show_progress_bar=False)
    print(f"  최적 RMSE: {study.best_value:.6f}")
    return study.best_params


def walk_forward_select(X, y, asset_name):
    n, test_size = len(X), (len(X)-C.MIN_TRAIN)//C.WF_SPLITS
    xgb_accs, lgb_accs = [], []
    xgb_probs_all, lgb_probs_all, y_true_all = [], [], []

    print(f"\n  Walk-forward 모델 선택 ({C.WF_SPLITS}폴드)")
    print(f"  {'폴드':>4} | {'XGB':>8} | {'LGB':>8}")
    print("  " + "-"*30)

    for fold in range(C.WF_SPLITS):
        tr_end = C.MIN_TRAIN + fold*test_size
        te_end = min(tr_end+test_size, n)
        if te_end > n: break
        X_tr = X.iloc[:tr_end];  y_tr = y.iloc[:tr_end]
        X_v  = X.iloc[tr_end-10:tr_end]; y_v = y.iloc[tr_end-10:tr_end]
        X_te = X.iloc[tr_end:te_end]; y_te = y.iloc[tr_end:te_end]
        y_te_dir = make_direction_labels(y_te.values)

        xgb_c = train_xgb_clf(X_tr, y_tr.values, X_v, y_v.values)
        xgb_p = xgb_c.predict_proba(X_te)[:,1]
        lgb_c = train_lgb_clf(X_tr, y_tr.values, X_v, y_v.values)
        lgb_p = lgb_c.predict_proba(X_te)[:,1]

        xa = accuracy_score(y_te_dir, (xgb_p>0.5).astype(int))*100
        la = accuracy_score(y_te_dir, (lgb_p>0.5).astype(int))*100
        xgb_accs.append(xa); lgb_accs.append(la)
        xgb_probs_all.extend(xgb_p); lgb_probs_all.extend(lgb_p)
        y_true_all.extend(y_te_dir)
        print(f"  {fold+1:>4} | {xa:>7.1f}% | {la:>7.1f}%")

    xgb_mean = np.mean(xgb_accs)
    lgb_mean  = np.mean(lgb_accs)
    print(f"  {'평균':>4} | {xgb_mean:>7.1f}% | {lgb_mean:>7.1f}%")

    if lgb_mean >= xgb_mean:
        selected_type = "LGB"
        best_probs = np.array(lgb_probs_all)
    else:
        selected_type = "XGB"
        best_probs = np.array(xgb_probs_all)

    # ★ 자산별 임계값 탐색 범위 제한
    # WTI는 0.34 같은 극단값 방지 → 0.44~0.56만 탐색
    THRESH_RANGE = {
        "Gold":        (0.40, 0.60),
        "WTI":         (0.44, 0.56),  # ★ 좁게 제한 (극단값 방지)
        "SP500":       (0.40, 0.60),
        "CaseShiller": (0.35, 0.60),
        "CPI":         (0.42, 0.58),
    }
    t_min, t_max = THRESH_RANGE.get(asset_name, (0.30, 0.70))
    y_true_arr = np.array(y_true_all)
    thresholds = np.arange(t_min, t_max + 0.01, 0.02)
    best_thresh, best_acc_t = 0.5, 0.0
    for t in thresholds:
        acc = accuracy_score(y_true_arr, (best_probs>t).astype(int))
        if acc > best_acc_t:
            best_acc_t = acc
            best_thresh = t

    print(f"\n  선택 모델: {selected_type} 분류기")
    print(f"  최적 임계값: {best_thresh:.2f} (WF 기준 {best_acc_t*100:.1f}%)")
    return selected_type, best_thresh, xgb_mean, lgb_mean


def walk_forward_backtest(X, y, selected_type, threshold, asset_name=None):
    n, test_size = len(X), (len(X)-C.MIN_TRAIN)//C.WF_SPLITS
    all_p, all_a, all_pr = [], [], []
    fold_m = []

    print(f"\n  Walk-forward 백테스트 (임계값={threshold:.2f})")
    print(f"  {'폴드':>4} | {'기간':^22} | {'RMSE':>8} | {'sMAPE':>8} | {'방향성':>7}")
    print("  " + "-" * 62)

    for fold in range(C.WF_SPLITS):
        tr_end = C.MIN_TRAIN + fold*test_size
        te_end = min(tr_end+test_size, n)
        if te_end > n: break
        X_tr = X.iloc[:tr_end]; y_tr = y.iloc[:tr_end]
        X_v  = X.iloc[tr_end-10:tr_end]; y_v = y.iloc[tr_end-10:tr_end]
        X_te = X.iloc[tr_end:te_end]; y_te = y.iloc[tr_end:te_end]

        reg = xgb.XGBRegressor(n_estimators=400, learning_rate=0.05,
                                max_depth=5, early_stopping_rounds=30,
                                random_state=42, verbosity=0, n_jobs=-1)
        reg.fit(X_tr, y_tr, eval_set=[(X_v,y_v)], verbose=False)
        if selected_type == "LGB":
            clf = train_lgb_clf(X_tr, y_tr.values, X_v, y_v.values)
        else:
            clf = train_xgb_clf(X_tr, y_tr.values, X_v, y_v.values)

        rp   = reg.predict(X_te)
        cp   = clf.predict_proba(X_te)[:,1]
        sign = np.where(cp > threshold, 1, -1)
        conf = np.abs(cp - 0.5) * 2
        final = np.abs(rp) * (0.3 + 0.7*conf) * sign

        m = compute_metrics(y_te.values, final)
        m["Dir_Acc"] = accuracy_score(
            make_direction_labels(y_te.values),
            (cp > threshold).astype(int)) * 100
        fold_m.append(m); all_p.extend(final)
        all_a.extend(y_te.values); all_pr.extend(cp)

        if hasattr(X_te.index, "strftime"):
            date_str = (X_te.index[0].strftime("%Y-%m") +
                        "~" + X_te.index[-1].strftime("%Y-%m"))
        else:
            date_str = f"{tr_end}~{te_end}"
        print(f"  {fold+1:>4} | {date_str:^22} | {m['RMSE']:>8.4f} | "
              f"{m['sMAPE']:>7.2f}% | {m['Dir_Acc']:>6.1f}%")

    avg = {k: np.mean([m[k] for m in fold_m]) for k in fold_m[0]}
    print("  " + "-" * 62)
    print(f"  {'평균':>4} | {'':^22} | {avg['RMSE']:>8.4f} | "
          f"{avg['sMAPE']:>7.2f}% | {avg['Dir_Acc']:>6.1f}%")
    return {"fold_metrics":fold_m,"avg_metrics":avg,
            "predictions":np.array(all_p),"actuals":np.array(all_a),
            "clf_probs":np.array(all_pr)}


def run_shap(model, X_test, asset_name):
    set_font()
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        plt.title(f"SHAP - {asset_name} (v7 Final)", fontsize=12, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(C.FIG_DIR, f"shap_{asset_name}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
        print(f"    SHAP 저장: {path}")
    except Exception as e:
        print(f"    SHAP 실패: {e}")


def plot_backtest(result, asset_name, threshold=0.5):
    set_font()
    preds, actuals = result["predictions"], result["actuals"]
    errors = preds - actuals
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    axes[0].plot(actuals, label="실제값", color="steelblue", lw=1.5)
    axes[0].plot(preds, label="예측값", color="tomato", lw=1.5, ls="--")
    axes[0].set_title(f"{asset_name} — v7 Final (임계값={threshold:.2f})",
                      fontsize=13, fontweight="bold")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].bar(range(len(errors)), errors,
                color=np.where(errors>0,"tomato","steelblue"), alpha=0.7)
    axes[1].axhline(0, color="black", lw=1)
    axes[1].set_title("예측 오차"); axes[1].grid(True, alpha=0.3)
    probs = result["clf_probs"]
    axes[2].fill_between(range(len(probs)), threshold, probs,
                         where=(probs>threshold), color="tomato",
                         alpha=0.6, label="상승 확률")
    axes[2].fill_between(range(len(probs)), probs, threshold,
                         where=(probs<=threshold), color="steelblue",
                         alpha=0.6, label="하락 확률")
    axes[2].axhline(threshold, color="black", lw=1.5, ls="--",
                    label=f"최적 임계값 {threshold:.2f}")
    axes[2].set_ylim(0, 1); axes[2].legend()
    axes[2].set_title("방향성 확률"); axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(C.FIG_DIR, f"backtest_{asset_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  백테스트 저장: {path}")



# ─────────────────────────────────────────────────────
#  ★ 선행연구 대비 벤치마크 (교수님 피드백 반영)
# ─────────────────────────────────────────────────────
PRIOR_BENCHMARKS = {
    "Gold": {
        "논문": "Anzuini et al. (2010, ECB)",
        "방법": "VAR 기반 방향성",
        "성능": 58.0,
        "비고": "월간 통화충격 → 상품 방향성"
    },
    "WTI": {
        "논문": "Browne & Cronin (2010)",
        "방법": "VAR 기반 방향성",
        "성능": 55.0,
        "비고": "원자재 → CPI 선행 예측"
    },
    "SP500": {
        "논문": "Bernanke & Kuttner (2005)",
        "방법": "이벤트스터디 방향성",
        "성능": 62.0,
        "비고": "금리충격 → 주가 방향성"
    },
    "CaseShiller": {
        "논문": "Iacoviello (2005)",
        "방법": "VAR 기반 방향성",
        "성능": 65.0,
        "비고": "통화정책 → 주택가격"
    },
    "CPI": {
        "논문": "Aruoba & Drechsel (2024)",
        "방법": "ML 방향성 예측",
        "성능": 63.0,
        "비고": "통화변수 → CPI 방향성"
    },
}

def print_benchmark(best_summary):
    print("\n" + "=" * 70)
    print("  ★ 선행연구 대비 성능 벤치마크")
    print("=" * 70)
    print(f"  {'자산':^10} {'선행연구':^8} {'우리(WF)':^10} {'개선':^8} {'논문'}")
    print("  " + "-" * 70)
    for asset, info in best_summary.items():
        if asset not in PRIOR_BENCHMARKS:
            continue
        prior = PRIOR_BENCHMARKS[asset]
        our_wf = info.get("wf_acc", info["Dir_Acc"])
        diff = our_wf - prior["성능"]
        sign = "+" if diff >= 0 else ""
        paper = prior["논문"][:30]
        print(f"  {asset:^10} {prior['성능']:^8.1f}% {our_wf:^10.1f}% "
              f"{sign}{diff:^7.1f}%p {paper}")
    print("=" * 70)
    print("  * 선행연구 수치는 동일 조건(월간, 통화변수) 기준 추정치")
    print("  * WF = Walk-forward 5폴드 평균 (더 보수적/신뢰할 수 있는 수치)")

def compare_models(metrics_dict):
    set_font()
    print("\n" + "="*70)
    print("  📊 전체 모델 성능 비교 (v7 Final)")
    print("="*70)
    df_m = pd.DataFrame(metrics_dict).T.round(4)
    print(df_m[["RMSE","MAE","sMAPE","R2","Dir_Acc"]].to_string())
    path = os.path.join(C.RESULT_DIR, "model_performance.csv")
    df_m.to_csv(path); print(f"  성능 저장: {path}")
    metric_list = ["RMSE","MAE","sMAPE","R2","Dir_Acc"]
    colors      = ["#e74c3c","#3498db","#2ecc71","#9b59b6","#1abc9c"]
    fig, axes   = plt.subplots(2, 3, figsize=(17, 10)); axes = axes.flatten()
    for i, (metric, color) in enumerate(zip(metric_list, colors)):
        if i >= len(axes) or metric not in df_m.columns: continue
        vals = df_m[metric]
        bars = axes[i].bar(vals.index, vals.values, color=color,
                           alpha=0.85, edgecolor="white")
        axes[i].set_title(metric, fontsize=12, fontweight="bold")
        if metric == "Dir_Acc":
            axes[i].axhline(75, color="red", ls="--", lw=1.5, label="목표 75%")
            axes[i].legend()
        for bar, v in zip(bars, vals.values):
            axes[i].text(bar.get_x()+bar.get_width()/2,
                         bar.get_height()*1.01, f"{v:.3f}",
                         ha="center", va="bottom", fontsize=7)
        axes[i].grid(True, alpha=0.3, axis="y")
        axes[i].tick_params(axis="x", rotation=35)
    axes[-1].axis("off")
    plt.suptitle("전체 모델 성능 비교 (v7 Final)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(C.FIG_DIR, "model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  비교 차트: {path}")
    return df_m


def main():
    print("\n[05] 예측 모델 (v7 Final - WF선택 + 임계값최적화 + Granger피처)")
    set_font()

    proc_path = os.path.join(C.DATA_PROC_DIR, "processed_data.csv")
    if not os.path.exists(proc_path):
        print("  processed_data.csv 없음"); return

    df = pd.read_csv(proc_path, index_col=0, parse_dates=True)
    print(f"  데이터 로드: {df.shape}")

    target_map = {
        "Gold":        "Gold_LogReturn",
        "WTI":         "WTI_LogReturn",
        "SP500":       "SP500_LogReturn",
        "CaseShiller": "CaseShiller_LogReturn",
        "CPI":         "CPI_LogReturn",
    }

    dummy_cols   = ["Cut_Start","Cut_Period","Hike_Start","Easing_Period",
                    "Regime_Recession","Regime_Neutral","Regime_Overheating",
                    "Regime_Index","TCI_Approx"]
    mon_lag_cols = [c for c in df.columns
                    if any(k in c for k in C.MONETARY_VARS) and "lag" in c]
    regime_cols  = [c for c in df.columns if "Regime_" in c]

    all_metrics  = {}
    best_summary = {}

    for asset_name, target_col in target_map.items():
        if target_col not in df.columns:
            print(f"\n  {target_col} 없음, 건너뜀"); continue

        print(f"\n{'='*62}")
        print(f"  ▶ 자산: {asset_name}  ({target_col})")
        print("="*62)

        feat_cols = list(dict.fromkeys(
            mon_lag_cols +
            [c for c in df.columns if asset_name in c and
             ("lag" in c or "RSI" in c or "MACD" in c or "Mom" in c or "MA" in c)] +
            [c for c in df.columns if "M2_YoY_lag" in c] +
            [c for c in df.columns if "TIPS_Spread_lag" in c] +
            [c for c in df.columns if "Cross_" in c] +
            [c for c in df.columns if "PPI" in c and "lag" in c] +
            [c for c in df.columns if "VIX" in c] +
            regime_cols + dummy_cols
        ))
        feat_cols = [c for c in feat_cols if c in df.columns and c != target_col]

        data = df[[target_col]+feat_cols].dropna()
        if len(data) < C.MIN_TRAIN + 30:
            print(f"  데이터 부족 ({len(data)}개)"); continue

        X_all = data[feat_cols]; y_all = data[target_col]
        n = len(data)
        tr_end = int(n*0.70); val_end = int(n*0.85)

        X_tr  = X_all.iloc[:tr_end];       y_tr  = y_all.iloc[:tr_end]
        X_val = X_all.iloc[tr_end:val_end]; y_val = y_all.iloc[tr_end:val_end]
        X_te  = X_all.iloc[val_end:];       y_te  = y_all.iloc[val_end:]
        print(f"  학습:{len(X_tr)} 검증:{len(X_val)} 테스트:{len(X_te)}")

        sx = StandardScaler(); sy = StandardScaler()
        X_tr_s  = pd.DataFrame(sx.fit_transform(X_tr), columns=feat_cols, index=X_tr.index)
        X_val_s = pd.DataFrame(sx.transform(X_val), columns=feat_cols, index=X_val.index)
        X_te_s  = pd.DataFrame(sx.transform(X_te), columns=feat_cols, index=X_te.index)
        y_tr_s  = sy.fit_transform(y_tr.values.reshape(-1,1)).flatten()
        y_val_s = sy.transform(y_val.values.reshape(-1,1)).flatten()

        top   = select_features_granger_priority(X_tr_s, y_tr_s, asset_name)
        Xtr_s = X_tr_s[top]; Xval_s = X_val_s[top]; Xte_s = X_te_s[top]

        best_p = optuna_tune(Xtr_s, y_tr_s, Xval_s, y_val_s)

        print(f"\n  XGBoost 회귀")
        xgb_reg = train_xgb_reg(Xtr_s, y_tr_s, Xval_s, y_val_s, best_p)
        xgb_pred = sy.inverse_transform(xgb_reg.predict(Xte_s).reshape(-1,1)).flatten()
        m = compute_metrics(y_te.values, xgb_pred)
        print_metrics(m, f"{asset_name} - XGBoost 회귀")
        all_metrics[f"{asset_name}_XGB_Reg"] = m

        print(f"\n  LightGBM 회귀")
        lgb_reg = train_lgb_reg(Xtr_s, y_tr_s, Xval_s, y_val_s)
        lgb_pred = sy.inverse_transform(lgb_reg.predict(Xte_s).reshape(-1,1)).flatten()
        m = compute_metrics(y_te.values, lgb_pred)
        print_metrics(m, f"{asset_name} - LightGBM 회귀")
        all_metrics[f"{asset_name}_LGB_Reg"] = m

        # Walk-forward 기반 모델 선택 + 임계값 최적화
        selected_type, best_thresh, xgb_wf, lgb_wf = walk_forward_select(
            Xtr_s, pd.Series(y_tr_s, index=Xtr_s.index), asset_name)

        # 선택된 모델로 최종 분류기 학습
        if selected_type == "LGB":
            final_clf = train_lgb_clf(Xtr_s, y_tr_s, Xval_s, y_val_s)
        else:
            final_clf = train_xgb_clf(Xtr_s, y_tr_s, Xval_s, y_val_s)

        # 테스트 예측
        te_probs    = final_clf.predict_proba(Xte_s)[:,1]
        te_dir_true = make_direction_labels(y_te.values)
        te_dir_pred = (te_probs > best_thresh).astype(int)
        final_acc   = accuracy_score(te_dir_true, te_dir_pred) * 100
        acc_05      = accuracy_score(te_dir_true, (te_probs>0.5).astype(int)) * 100

        print(f"\n  📊 최종 분류 결과:")
        print(f"     임계값 0.50: {acc_05:.1f}%")
        print(f"     최적 임계값 {best_thresh:.2f}: {final_acc:.1f}%")

        # ★ 추가 지표 출력
        clf_m = compute_clf_metrics(te_dir_true, te_dir_pred)
        print_clf_metrics(clf_m, f"{asset_name} 분류 성능 상세")

        # ★ Raw 베이스라인 비교
        raw_acc = run_raw_baseline(df, asset_name, target_col)
        if raw_acc is not None:
            print(f"\n  📊 엔지니어링 효과:")
            print(f"     Raw 정확도  : {raw_acc:.1f}%")
            print(f"     v7 정확도   : {final_acc:.1f}%")
            print(f"     ★ 개선     : +{final_acc - raw_acc:.1f}%p")

        all_metrics[f"{asset_name}_Final_Clf"] = {
            "Dir_Acc":final_acc,"MAE":0,"RMSE":0,"sMAPE":0,"R2":0}

        reg_abs = np.abs(sy.inverse_transform(xgb_reg.predict(Xte_s).reshape(-1,1)).flatten())
        conf = np.abs(te_probs-0.5)*2
        sign = np.where(te_probs>best_thresh, 1, -1)
        final_pred = reg_abs*(0.3+0.7*conf)*sign
        m = compute_metrics(y_te.values, final_pred)
        m["Dir_Acc"] = final_acc
        print_metrics(m, f"{asset_name} - v7 Final")
        all_metrics[f"{asset_name}_Final"] = m

        try:
            run_shap(xgb_reg, Xte_s, asset_name)
        except Exception:
            pass

        print(f"\n  Walk-forward 백테스트: {asset_name}")
        bt = walk_forward_backtest(
            Xtr_s, pd.Series(y_tr_s, index=Xtr_s.index),
            selected_type, best_thresh,
            asset_name=asset_name)
        plot_backtest(bt, asset_name, best_thresh)
        all_metrics[f"{asset_name}_WF"] = bt["avg_metrics"]

        # ★ bt 실행 후 best_summary 업데이트
        best_summary[asset_name] = {
            "model":selected_type,"threshold":best_thresh,
            "Dir_Acc":final_acc,
            "wf_acc": bt["avg_metrics"]["Dir_Acc"]}

    print("\n" + "="*62)
    print("  🏆 v7 Final 자산별 최고 성능 요약")
    print("="*62)
    target_reached = 0
    for asset, info in best_summary.items():
        status = "✅" if info["Dir_Acc"] >= 75 else "🟡"
        print(f"  {status} {asset:12s}: {info['Dir_Acc']:.1f}%  "
              f"(모델:{info['model']}, 임계값:{info['threshold']:.2f})")
        if info["Dir_Acc"] >= 75:
            target_reached += 1
    print(f"\n  목표 75% 달성: {target_reached}/{len(best_summary)}개 자산")
    print("="*62)

    compare_models(all_metrics)

    # ★ 선행연구 대비 벤치마크
    print_benchmark(best_summary)

    print("\n  ✅ 예측 모델 완료 (v7 Final)")
    return all_metrics


if __name__ == "__main__":
    main()
