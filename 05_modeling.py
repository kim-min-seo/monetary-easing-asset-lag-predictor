# ============================================================
#  05_modeling.py — 예측 모델 (v6)
#  ★ v6 개선:
#  1. 앙상블 로직 수정 (분류기 확률 가중 앙상블)
#  2. BiGRU 제거 → 가중 앙상블로 대체
#  3. CaseShiller_LogReturn2 (2차 차분) 사용
# ============================================================

import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import platform
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             accuracy_score)
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


# ──────────────────────────────────────────────
#  성능 평가
# ──────────────────────────────────────────────

def compute_metrics(y_true, y_pred, eps=1e-8):
    mae   = mean_absolute_error(y_true, y_pred)
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    mape  = np.mean(np.abs((y_true-y_pred)/(np.abs(y_true)+eps)))*100
    smape = np.mean(2*np.abs(y_pred-y_true)/
                    (np.abs(y_true)+np.abs(y_pred)+eps))*100
    ss_res = np.sum((y_true-y_pred)**2)
    ss_tot = np.sum((y_true-np.mean(y_true))**2)
    r2     = 1 - ss_res/(ss_tot+eps)
    dir_acc = (np.mean((np.diff(y_true)>0)==(np.diff(y_pred)>0))*100
               if len(y_true)>1 else 0.0)
    return {"MAE":mae,"RMSE":rmse,"MAPE":mape,
            "sMAPE":smape,"R2":r2,"Dir_Acc":dir_acc}


def print_metrics(m, label="모델"):
    bar = "=" * 56
    print(f"\n{bar}")
    print(f"  📊 {label}")
    print(bar)
    print(f"  MAE           : {m['MAE']:.6f}")
    print(f"  RMSE          : {m['RMSE']:.6f}")
    print(f"  sMAPE         : {m['sMAPE']:.2f}%")
    print(f"  R²            : {m['R2']:.4f}")
    print(f"  방향성 정확도 : {m['Dir_Acc']:.1f}%  ← 목표: 75%↑")
    print(bar)


# ──────────────────────────────────────────────
#  피처 선택
# ──────────────────────────────────────────────

def select_top_features(X_tr, y_tr, top_n=None):
    if top_n is None:
        top_n = C.TOP_FEATURES
    print(f"  → 피처 선택: {X_tr.shape[1]}개 → 상위 {top_n}개")
    selector = xgb.XGBRegressor(
        n_estimators=100, max_depth=4,
        random_state=42, verbosity=0, n_jobs=-1)
    selector.fit(X_tr, y_tr)
    importance = pd.Series(selector.feature_importances_,
                           index=X_tr.columns)
    top_cols = importance.nlargest(top_n).index.tolist()
    print(f"  ✓ 선택: {len(top_cols)}개")
    return top_cols


# ──────────────────────────────────────────────
#  XGBoost
# ──────────────────────────────────────────────

def make_direction_labels(y):
    return (y > 0).astype(int)


def train_xgb_reg(X_tr, y_tr, X_val, y_val, params=None):
    if params is None:
        params = {"n_estimators":800,"learning_rate":0.03,"max_depth":6,
                  "min_child_weight":3,"subsample":0.8,
                  "colsample_bytree":0.8,"reg_alpha":0.1,"reg_lambda":1.0}
    model = xgb.XGBRegressor(
        **params, early_stopping_rounds=50, eval_metric="rmse",
        random_state=42, n_jobs=-1, verbosity=0)
    model.fit(X_tr, y_tr, eval_set=[(X_val,y_val)], verbose=False)
    return model


def train_xgb_clf(X_tr, y_tr, X_val, y_val, params=None):
    dir_tr  = make_direction_labels(y_tr)
    dir_val = make_direction_labels(y_val)
    if params is None:
        params = {"n_estimators":800,"learning_rate":0.03,"max_depth":5,
                  "min_child_weight":3,"subsample":0.8,
                  "colsample_bytree":0.8,"reg_alpha":0.5,"reg_lambda":1.0}
    model = xgb.XGBClassifier(
        **params, early_stopping_rounds=50, eval_metric="logloss",
        random_state=42, n_jobs=-1, verbosity=0,
        base_score=0.5, use_label_encoder=False)
    model.fit(X_tr, dir_tr, eval_set=[(X_val,dir_val)], verbose=False)
    return model


# ──────────────────────────────────────────────
#  LightGBM
# ──────────────────────────────────────────────

def train_lgb_reg(X_tr, y_tr, X_val, y_val):
    model = lgb.LGBMRegressor(
        n_estimators=800, learning_rate=0.03, max_depth=6,
        num_leaves=63, min_child_samples=15,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbose=-1)
    model.fit(X_tr, y_tr, eval_set=[(X_val,y_val)],
              callbacks=[lgb.early_stopping(50,verbose=False),
                         lgb.log_evaluation(period=9999)])
    return model


def train_lgb_clf(X_tr, y_tr, X_val, y_val):
    dir_tr  = make_direction_labels(y_tr)
    dir_val = make_direction_labels(y_val)
    model = lgb.LGBMClassifier(
        n_estimators=800, learning_rate=0.03, max_depth=5,
        num_leaves=31, min_child_samples=15,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.5, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbose=-1, is_unbalance=True)
    model.fit(X_tr, dir_tr, eval_set=[(X_val,dir_val)],
              callbacks=[lgb.early_stopping(50,verbose=False),
                         lgb.log_evaluation(period=9999)])
    return model


# ──────────────────────────────────────────────
#  Optuna
# ──────────────────────────────────────────────

def optuna_tune(X_tr, y_tr, X_val, y_val):
    if not OPTUNA_AVAILABLE:
        return None
    print(f"  🔎 Optuna 튜닝 ({C.OPTUNA_TRIALS}회)...")

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
        model = xgb.XGBRegressor(
            **params, early_stopping_rounds=30, eval_metric="rmse",
            random_state=42, verbosity=0, n_jobs=-1)
        model.fit(X_tr, y_tr, eval_set=[(X_val,y_val)], verbose=False)
        return np.sqrt(mean_squared_error(y_val, model.predict(X_val)))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=C.OPTUNA_TRIALS,
                   show_progress_bar=False)
    print(f"  ✓ 최적 RMSE: {study.best_value:.6f}")
    return study.best_params


# ──────────────────────────────────────────────
#  ★ v6 앙상블 (가중 확률 앙상블 — 핵심 개선)
# ──────────────────────────────────────────────

def ensemble_predict_v6(reg_models, clf_models, X, y_scaler=None,
                        reg_weight=0.3, clf_weight=0.7):
    """
    ★ v6 핵심 개선:
    단순 방향 부호 결합 → 가중 확률 앙상블

    이전 문제:
    분류기 73.8% → 앙상블 41.5% (오히려 하락)

    원인:
    회귀 모델이 크기 결정 + 방향 결정 모두 관여
    → 방향이 틀리면 분류기 결과 무시됨

    v6 해결:
    방향: 분류기 확률로만 결정 (clf_weight=0.7)
    크기: 회귀 모델 절댓값 (reg_weight=0.3)
    → 분류기가 잘 못해도 회귀가 보완
    → 분류기가 잘 하면 더 강화
    """
    # 회귀 예측
    reg_preds = np.mean([m.predict(X) for m in reg_models], axis=0)
    if y_scaler:
        reg_preds = y_scaler.inverse_transform(
            reg_preds.reshape(-1,1)).flatten()

    # 분류 확률 (상승 확률)
    clf_probs = np.mean(
        [m.predict_proba(X)[:,1] for m in clf_models], axis=0)

    # 방향: 분류기 확률 기반 (0.5 이상 = 상승)
    clf_sign = np.where(clf_probs > 0.5, 1, -1)

    # 크기: 회귀 절댓값
    reg_abs = np.abs(reg_preds)

    # ★ 가중 앙상블:
    # 분류기 확신도(확률)가 높을수록 더 강하게 반영
    confidence = np.abs(clf_probs - 0.5) * 2  # 0~1 사이 확신도
    weighted_size = reg_abs * (reg_weight + clf_weight * confidence)

    final_preds = weighted_size * clf_sign
    return final_preds, clf_probs


# ──────────────────────────────────────────────
#  SHAP
# ──────────────────────────────────────────────

def run_shap(model, X_test, asset_name):
    set_font()
    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test,
                          plot_type="bar", show=False)
        plt.title(f"SHAP 기여도 — {asset_name} (v6)",
                  fontsize=12, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(C.FIG_DIR, f"shap_{asset_name}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    ✓ SHAP 저장: {path}")
    except Exception as e:
        print(f"    ⚠️  SHAP 실패: {e}")


# ──────────────────────────────────────────────
#  Walk-forward 백테스팅
# ──────────────────────────────────────────────

def walk_forward(X, y):
    n, test_size = len(X), (len(X)-C.MIN_TRAIN)//C.WF_SPLITS
    all_p, all_a, all_pr = [], [], []
    fold_m = []

    print(f"\n  Walk-forward ({C.WF_SPLITS} 폴드)")
    print(f"  {'폴드':>4} | {'기간':^22} | {'RMSE':>8} | "
          f"{'sMAPE':>8} | {'방향성':>7}")
    print("  " + "-" * 62)

    for fold in range(C.WF_SPLITS):
        tr_end = C.MIN_TRAIN + fold*test_size
        te_end = min(tr_end+test_size, n)
        if te_end > n: break

        X_tr = X.iloc[:tr_end];  y_tr = y.iloc[:tr_end]
        X_v  = X.iloc[tr_end-10:tr_end]; y_v = y.iloc[tr_end-10:tr_end]
        X_te = X.iloc[tr_end:te_end]; y_te = y.iloc[tr_end:te_end]

        reg = xgb.XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=5,
            early_stopping_rounds=30, random_state=42,
            verbosity=0, n_jobs=-1)
        reg.fit(X_tr, y_tr, eval_set=[(X_v,y_v)], verbose=False)

        clf = xgb.XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=4,
            early_stopping_rounds=30, eval_metric="logloss",
            random_state=42, verbosity=0, n_jobs=-1,
            base_score=0.5, use_label_encoder=False)
        clf.fit(X_tr, make_direction_labels(y_tr),
                eval_set=[(X_v, make_direction_labels(y_v))],
                verbose=False)

        # ★ v6 가중 앙상블 적용
        rp = reg.predict(X_te)
        cp = clf.predict_proba(X_te)[:,1]
        conf  = np.abs(cp - 0.5) * 2
        sign  = np.where(cp > 0.5, 1, -1)
        final = np.abs(rp) * (0.3 + 0.7*conf) * sign

        m = compute_metrics(y_te.values, final)
        fold_m.append(m); all_p.extend(final)
        all_a.extend(y_te.values); all_pr.extend(cp)

        date_str = (f"{X_te.index[0].strftime('%Y-%m')}"
                    f"~{X_te.index[-1].strftime('%Y-%m')}"
                    if hasattr(X_te.index,"strftime") else
                    f"{tr_end}~{te_end}")
        print(f"  {fold+1:>4} | {date_str:^22} | {m['RMSE']:>8.4f} | "
              f"{m['sMAPE']:>7.2f}% | {m['Dir_Acc']:>6.1f}%")

    avg = {k: np.mean([m[k] for m in fold_m]) for k in fold_m[0]}
    print("  " + "-" * 62)
    print(f"  {'평균':>4} | {'':^22} | {avg['RMSE']:>8.4f} | "
          f"{avg['sMAPE']:>7.2f}% | {avg['Dir_Acc']:>6.1f}%")

    return {"fold_metrics":fold_m,"avg_metrics":avg,
            "predictions":np.array(all_p),
            "actuals":np.array(all_a),
            "clf_probs":np.array(all_pr)}


def plot_backtest(result, asset_name):
    set_font()
    preds, actuals = result["predictions"], result["actuals"]
    errors = preds - actuals

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    axes[0].plot(actuals, label="실제값", color="steelblue", lw=1.5)
    axes[0].plot(preds,   label="예측값", color="tomato",
                 lw=1.5, ls="--")
    axes[0].set_title(f"{asset_name} — v6 앙상블 예측 vs 실제",
                      fontsize=13, fontweight="bold")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].bar(range(len(errors)), errors,
                color=np.where(errors>0,"tomato","steelblue"), alpha=0.7)
    axes[1].axhline(0, color="black", lw=1)
    axes[1].set_title("예측 오차"); axes[1].grid(True, alpha=0.3)

    probs = result["clf_probs"]
    axes[2].fill_between(range(len(probs)), 0.5, probs,
                         where=(probs>0.5), color="tomato",
                         alpha=0.6, label="상승 확률")
    axes[2].fill_between(range(len(probs)), probs, 0.5,
                         where=(probs<0.5), color="steelblue",
                         alpha=0.6, label="하락 확률")
    axes[2].axhline(0.5, color="black", lw=1, ls="--")
    axes[2].set_ylim(0, 1)
    axes[2].set_title("방향성 분류 확률 (가중 앙상블 v6)")
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(C.FIG_DIR, f"backtest_{asset_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ 백테스트 저장: {path}")


# ──────────────────────────────────────────────
#  모델 비교 차트
# ──────────────────────────────────────────────

def compare_models(metrics_dict):
    set_font()
    print("\n" + "="*70)
    print("  📊 전체 모델 성능 비교 (v6)")
    print("="*70)
    df_m = pd.DataFrame(metrics_dict).T.round(4)
    print(df_m[["RMSE","MAE","sMAPE","R2","Dir_Acc"]].to_string())

    path = os.path.join(C.RESULT_DIR, "model_performance.csv")
    df_m.to_csv(path)
    print(f"  ✓ 성능 저장: {path}")

    metric_list = ["RMSE","MAE","sMAPE","R2","Dir_Acc"]
    colors      = ["#e74c3c","#3498db","#2ecc71","#9b59b6","#1abc9c"]
    fig, axes   = plt.subplots(2, 3, figsize=(17, 10))
    axes        = axes.flatten()

    for i, (metric, color) in enumerate(zip(metric_list, colors)):
        if i >= len(axes) or metric not in df_m.columns:
            continue
        vals = df_m[metric]
        bars = axes[i].bar(vals.index, vals.values,
                           color=color, alpha=0.85, edgecolor="white")
        axes[i].set_title(metric, fontsize=12, fontweight="bold")
        if metric == "Dir_Acc":
            axes[i].axhline(75, color="red", ls="--",
                            lw=1.5, label="목표 75%")
            axes[i].legend()
        for bar, v in zip(bars, vals.values):
            axes[i].text(bar.get_x()+bar.get_width()/2,
                         bar.get_height()*1.01, f"{v:.3f}",
                         ha="center", va="bottom", fontsize=7)
        axes[i].grid(True, alpha=0.3, axis="y")
        axes[i].tick_params(axis="x", rotation=35)

    axes[-1].axis("off")
    plt.suptitle("전체 모델 성능 비교 (v6 — 가중 앙상블)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(C.FIG_DIR, "model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ 비교 차트: {path}")
    return df_m


# ──────────────────────────────────────────────
#  메인
# ──────────────────────────────────────────────

def main():
    print("\n[05] 예측 모델 (v6)")
    set_font()

    proc_path = os.path.join(C.DATA_PROC_DIR, "processed_data.csv")
    if not os.path.exists(proc_path):
        print("  ⚠️  processed_data.csv 없음 → 02 먼저 실행")
        return

    df = pd.read_csv(proc_path, index_col=0, parse_dates=True)
    print(f"  ✓ 데이터 로드: {df.shape}")

    # ★ v6: CaseShiller는 2차 차분 버전 사용
    target_map = {
        "Gold":         "Gold_LogReturn",
        "WTI":          "WTI_LogReturn",
        "SP500":        "SP500_LogReturn",
        "CaseShiller":  "CaseShiller_LogReturn2",  # 2차 차분
        "CPI":          "CPI_LogReturn",
    }

    dummy_cols   = ["Cut_Start","Cut_Period","Hike_Start","Easing_Period"]
    mon_lag_cols = [c for c in df.columns if any(
        k in c for k in C.MONETARY_VARS) and "lag" in c]

    all_metrics = {}

    for asset_name, target_col in target_map.items():
        if target_col not in df.columns:
            print(f"\n  ⚠️  {target_col} 없음, 건너뜀")
            continue

        print(f"\n{'='*62}")
        print(f"  ▶ 자산: {asset_name}  ({target_col})")
        print("="*62)

        feat_cols = list(dict.fromkeys(
            mon_lag_cols +
            [c for c in df.columns
             if asset_name in c and
             ("lag" in c or "RSI" in c or "MACD" in c
              or "Mom" in c or "MA" in c)] +
            [c for c in df.columns if "M2_YoY_lag" in c] +
            [c for c in df.columns if "TIPS_Spread_lag" in c] +  # ★ v6
            [c for c in df.columns if "Cross_" in c] +
            dummy_cols
        ))
        feat_cols = [c for c in feat_cols
                     if c in df.columns and c != target_col]

        data = df[[target_col]+feat_cols].dropna()
        if len(data) < C.MIN_TRAIN + 30:
            print(f"  ⚠️  데이터 부족 ({len(data)}개)")
            continue

        X_all = data[feat_cols]; y_all = data[target_col]
        n = len(data)
        tr_end  = int(n*0.70); val_end = int(n*0.85)

        X_tr  = X_all.iloc[:tr_end];       y_tr  = y_all.iloc[:tr_end]
        X_val = X_all.iloc[tr_end:val_end]; y_val = y_all.iloc[tr_end:val_end]
        X_te  = X_all.iloc[val_end:];       y_te  = y_all.iloc[val_end:]

        print(f"  학습:{len(X_tr)} 검증:{len(X_val)} 테스트:{len(X_te)}")

        sx = StandardScaler(); sy = StandardScaler()
        X_tr_s  = pd.DataFrame(sx.fit_transform(X_tr),
                               columns=feat_cols, index=X_tr.index)
        X_val_s = pd.DataFrame(sx.transform(X_val),
                               columns=feat_cols, index=X_val.index)
        X_te_s  = pd.DataFrame(sx.transform(X_te),
                               columns=feat_cols, index=X_te.index)
        y_tr_s  = sy.fit_transform(y_tr.values.reshape(-1,1)).flatten()
        y_val_s = sy.transform(y_val.values.reshape(-1,1)).flatten()

        top = select_top_features(X_tr_s, y_tr_s)
        Xtr_s = X_tr_s[top]; Xval_s = X_val_s[top]; Xte_s = X_te_s[top]

        best_p = optuna_tune(Xtr_s, y_tr_s, Xval_s, y_val_s)

        # XGBoost
        print(f"\n  🌲 XGBoost 회귀")
        xgb_reg = train_xgb_reg(Xtr_s, y_tr_s, Xval_s, y_val_s, best_p)
        xgb_pred = sy.inverse_transform(
            xgb_reg.predict(Xte_s).reshape(-1,1)).flatten()
        m = compute_metrics(y_te.values, xgb_pred)
        print_metrics(m, f"{asset_name} - XGBoost 회귀")
        all_metrics[f"{asset_name}_XGB_Reg"] = m

        print(f"\n  🎯 XGBoost 분류")
        xgb_clf = train_xgb_clf(Xtr_s, y_tr_s, Xval_s, y_val_s)
        cp = xgb_clf.predict_proba(Xte_s)[:,1]
        clf_acc = accuracy_score(
            (y_te.values>0).astype(int),(cp>0.5).astype(int))*100
        print(f"    ✓ 방향성: {clf_acc:.1f}%")
        all_metrics[f"{asset_name}_XGB_Clf"] = {
            "Dir_Acc":clf_acc,"MAE":0,"RMSE":0,"sMAPE":0,"R2":0}

        # LightGBM
        print(f"\n  ⚡ LightGBM 회귀")
        lgb_reg = train_lgb_reg(Xtr_s, y_tr_s, Xval_s, y_val_s)
        lgb_pred = sy.inverse_transform(
            lgb_reg.predict(Xte_s).reshape(-1,1)).flatten()
        m = compute_metrics(y_te.values, lgb_pred)
        print_metrics(m, f"{asset_name} - LightGBM 회귀")
        all_metrics[f"{asset_name}_LGB_Reg"] = m

        print(f"\n  🎯 LightGBM 분류")
        lgb_clf = train_lgb_clf(Xtr_s, y_tr_s, Xval_s, y_val_s)
        lgb_cp  = lgb_clf.predict_proba(Xte_s)[:,1]
        lgb_clf_acc = accuracy_score(
            (y_te.values>0).astype(int),(lgb_cp>0.5).astype(int))*100
        print(f"    ✓ LGB 분류 방향성: {lgb_clf_acc:.1f}%")

        # ★ v6 가중 앙상블
        print(f"\n  🔗 v6 가중 앙상블 (확률 기반)")
        ens, ep = ensemble_predict_v6(
            [xgb_reg, lgb_reg], [xgb_clf, lgb_clf], Xte_s, sy)
        m = compute_metrics(y_te.values, ens)
        print_metrics(m, f"{asset_name} - v6 가중 앙상블")
        all_metrics[f"{asset_name}_Ensemble"] = m

        # SHAP
        run_shap(xgb_reg, Xte_s, asset_name)

        # Walk-forward
        print(f"\n  📅 Walk-forward: {asset_name}")
        bt = walk_forward(Xtr_s, pd.Series(y_tr_s, index=Xtr_s.index))
        plot_backtest(bt, asset_name)
        all_metrics[f"{asset_name}_WF"] = bt["avg_metrics"]

    compare_models(all_metrics)
    print("\n  ✅ 예측 모델 완료 (v6)")
    return all_metrics


if __name__ == "__main__":
    main()
