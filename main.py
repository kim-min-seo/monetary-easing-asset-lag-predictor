# ============================================================
#  main.py — 전체 파이프라인 일괄 실행 (v6)
#  python main.py 하나로 전체 실행
#
#  [실행 전 패키지 설치]
#  pip install python-dotenv fredapi yfinance pandas numpy
#  pip install scikit-learn xgboost lightgbm shap
#  pip install statsmodels scipy matplotlib seaborn plotly optuna
# ============================================================

import sys
import os
import importlib.util
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_module(filename):
    """파일명으로 모듈 로드 후 main() 실행"""
    filepath = os.path.join(os.path.dirname(__file__), f"{filename}.py")
    spec = importlib.util.spec_from_file_location(filename, filepath)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if hasattr(mod, "main"):
        return mod.main()


print("=" * 62)
print("  통화 완화 환경에서의 자산군별 가격 반응 시차 실증 분석")
print("  ★ VERSION 6.0")
print("  A. 시차 존재 여부  B. 반응 순서  C. 예측 가능성")
print(f"  실행 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 62)

# ── STEP 1 ──────────────────────────────────────────────────
print("\n" + "="*62)
print("  STEP 1/6 — 데이터 수집 (FRED + Yahoo Finance)")
print("="*62)
run_module("01_data_collection")

# ── STEP 2 ──────────────────────────────────────────────────
print("\n" + "="*62)
print("  STEP 2/6 — 전처리 (CaseShiller 2차 차분 + TIPS 추가)")
print("="*62)
run_module("02_preprocessing")

# ── STEP 3 ──────────────────────────────────────────────────
print("\n" + "="*62)
print("  STEP 3/6 — 실증 분석 (ADF · 그랜저 · VAR · IRF)")
print("="*62)
run_module("03_analysis")

# ── STEP 4 ──────────────────────────────────────────────────
print("\n" + "="*62)
print("  STEP 4/6 — 시각화")
print("="*62)
run_module("04_visualization")

# ── STEP 5 ──────────────────────────────────────────────────
print("\n" + "="*62)
print("  STEP 5/6 — 예측 모델 (XGBoost · LightGBM · 가중앙상블)")
print("="*62)
run_module("05_modeling")

# ── STEP 6 ──────────────────────────────────────────────────
print("\n" + "="*62)
print("  STEP 6/6 — 예측 모델 (XGBoost · LightGBM · 가중앙상블)")
print("="*62)
run_module("05_modeling")

# ── 완료 ────────────────────────────────────────────────────
print("\n" + "="*62)
print("  ✅ v6 파이프라인 완료!")
print(f"  완료 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("  결과 위치:")
print("    outputs/figures/ → 모든 차트 (PNG, HTML)")
print("    outputs/results/ → 분석 결과 (CSV)")
print("="*62)
