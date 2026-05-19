# ============================================================
#  main.py — 전체 파이프라인 (v7 Regime)
#  python main.py 하나로 전체 실행
# ============================================================

import sys
import os
import importlib.util
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_module(filename):
    filepath = os.path.join(os.path.dirname(__file__), f"{filename}.py")
    spec = importlib.util.spec_from_file_location(filename, filepath)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if hasattr(mod, "main"):
        return mod.main()


print("=" * 62)
print("  통화 완화 환경에서의 자산군별 가격 반응 시차 실증 분석")
print("  ★ VERSION 7.0 (QVAR Regime 국면별 모델)")
print("  A. 시차 존재 여부  B. 반응 순서  C. 예측 가능성")
print(f"  실행 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 62)

print("\n" + "="*62)
print("  STEP 1/6 — 데이터 수집 (FRED + Yahoo Finance)")
print("="*62)
run_module("01_data_collection")

print("\n" + "="*62)
print("  STEP 2/6 — 전처리 (v7: QVAR 국면 피처 추가)")
print("="*62)
run_module("02_preprocessing")

print("\n" + "="*62)
print("  STEP 3/6 — 실증 분석 (ADF · 그랜저 · VAR · IRF)")
print("="*62)
run_module("03_analysis")

print("\n" + "="*62)
print("  STEP 4/6 — 시각화")
print("="*62)
run_module("04_visualization")

print("\n" + "="*62)
print("  STEP 5/6 — QVAR Spillover (경기국면별 전이 구조)")
print("="*62)
run_module("06_qvar_spillover")

print("\n" + "="*62)
print("  STEP 6/6 — 예측 모델 (v7: QVAR 국면별 별도 모델)")
print("="*62)
run_module("05_modeling")

print("\n" + "="*62)
print("  ✅ v7 Regime 파이프라인 완료!")
print(f"  완료 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("  결과 위치:")
print("    outputs/figures/ → 모든 차트 (PNG, HTML)")
print("    outputs/results/ → 분석 결과 (CSV)")
print("="*62)
