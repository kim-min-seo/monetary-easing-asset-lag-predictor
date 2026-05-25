# ============================================================
#  main.py — 파이프라인 실행기 (v8a)
#  · 전체 실행 / 단계별 선택 실행 지원
#  · 인자 없이 실행하면 대화형 메뉴, CLI 플래그로도 지정 가능
#
#  [실행 전 패키지 설치]
#  pip install python-dotenv fredapi yfinance pandas numpy
#  pip install scikit-learn xgboost lightgbm shap
#  pip install statsmodels scipy matplotlib seaborn plotly optuna
#
#  [사용 예시]
#  python main.py                  # 대화형 메뉴
#  python main.py --all            # 전체 실행 (01~07)
#  python main.py --from-saved     # 저장된 raw_data로 학습 (01 건너뜀, 02~07)
#  python main.py --from-processed # 전처리 완료 데이터로 분석 (03~07)
#  python main.py --steps 3 5 7    # 원하는 단계만
#  python main.py --step 5         # 한 단계만
# ============================================================

import sys
import os
import argparse
import importlib.util
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C


# ── 단계 정의 ────────────────────────────────────────────────
STEPS = {
    1: ("01_data_collection",  "데이터 수집 (FRED + Yahoo Finance)"),
    2: ("02_preprocessing",    "전처리 (로그수익률·더미·국면 피처)"),
    3: ("03_analysis",         "실증 분석 (ADF · 그랜저 · VAR · IRF)"),
    4: ("04_visualization",    "시각화"),
    5: ("05_modeling",         "예측 모델 (XGBoost · LightGBM)"),
    6: ("06_qvar_spillover",   "QVAR Spillover (경기국면별 전이 구조)"),
    7: ("07_response_timing",  "반응 타이밍 + QE 규모 분석 (v8a)"),
}
TOTAL = len(STEPS)

# 단계별 선행 산출물 (없으면 경고)
PREREQ = {
    2: (os.path.join(C.DATA_RAW_DIR,  "raw_data.csv"),       "STEP 1(수집) 또는 저장된 raw_data.csv 필요"),
    3: (os.path.join(C.DATA_PROC_DIR, "processed_data.csv"), "STEP 2(전처리) 또는 저장된 processed_data.csv 필요"),
    4: (os.path.join(C.DATA_PROC_DIR, "processed_data.csv"), "processed_data.csv 필요"),
    5: (os.path.join(C.DATA_PROC_DIR, "processed_data.csv"), "processed_data.csv 필요"),
    6: (os.path.join(C.DATA_RAW_DIR,  "raw_data.csv"),       "raw_data.csv 필요"),
    7: (os.path.join(C.DATA_PROC_DIR, "processed_data.csv"), "processed_data.csv 필요"),
}


def run_module(filename):
    """파일명으로 모듈 로드 후 main() 실행"""
    filepath = os.path.join(os.path.dirname(__file__), f"{filename}.py")
    spec = importlib.util.spec_from_file_location(filename, filepath)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if hasattr(mod, "main"):
        return mod.main()


def check_prereq(step):
    """선행 산출물 존재 확인. 없으면 (False, 안내) 반환."""
    if step not in PREREQ:
        return True, ""
    path, msg = PREREQ[step]
    if os.path.exists(path):
        return True, ""
    return False, msg


def run_steps(step_list):
    """주어진 단계 리스트를 순서대로 실행."""
    step_list = sorted(set(step_list))
    print("=" * 62)
    print("  통화 완화 환경에서의 자산군별 가격 반응 시차 실증 분석")
    print("  ★ VERSION 8a")
    print(f"  실행 단계: {', '.join(f'{s:02d}' for s in step_list)}")
    print(f"  실행 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 62)

    # 첫 단계만 선행 산출물 확인 (이전 단계를 같은 실행에서 만들면 통과)
    for i, step in enumerate(step_list):
        prior_made = any(s < step for s in step_list[:i])
        if not prior_made:
            ok, msg = check_prereq(step)
            if not ok:
                print(f"\n  ⚠️  STEP {step:02d} 실행 불가: {msg}")
                print(f"      → 먼저 해당 데이터를 만들거나 단계를 포함해 실행하세요.")
                return

    for step in step_list:
        name, desc = STEPS[step]
        print("\n" + "=" * 62)
        print(f"  STEP {step}/{TOTAL} — {desc}")
        print("=" * 62)
        run_module(name)

    print("\n" + "=" * 62)
    print("  ✅ 파이프라인 완료 (v8a)")
    print(f"  완료 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("  결과 위치:  outputs/figures/ (차트) · outputs/results/ (CSV)")
    print("=" * 62)


def interactive_menu():
    """인자 없이 실행 시 대화형 메뉴."""
    print("=" * 62)
    print("  파이프라인 실행 메뉴 (v8a)")
    print("=" * 62)
    print("  1. 전체 실행 (01~07)")
    print("  2. 저장된 데이터로 학습 (01 건너뜀, 02~07)")
    print("  3. 전처리 완료 데이터로 분석 (03~07)")
    print("  4. 예측 모델만 (05)")
    print("  5. 반응 타이밍 + QE 분석만 (07)")
    print("  6. 단계 직접 선택 (예: 3 5 7)")
    print("  0. 종료")
    print("  ──────────────────────────────────────")
    print("  [참고 단계]")
    for s, (_, desc) in STEPS.items():
        print(f"    {s}. {desc}")
    print("=" * 62)

    choice = input("  선택 > ").strip()

    if choice == "1":
        run_steps(list(STEPS.keys()))
    elif choice == "2":
        run_steps([2, 3, 4, 5, 6, 7])
    elif choice == "3":
        run_steps([3, 4, 5, 6, 7])
    elif choice == "4":
        run_steps([5])
    elif choice == "5":
        run_steps([7])
    elif choice == "6":
        raw = input("  실행할 단계 번호 (공백 구분, 예: 3 5 7) > ").strip()
        try:
            steps = [int(x) for x in raw.split() if int(x) in STEPS]
            if steps:
                run_steps(steps)
            else:
                print("  ⚠️  유효한 단계가 없습니다 (1~7).")
        except ValueError:
            print("  ⚠️  숫자만 입력하세요.")
    elif choice == "0":
        print("  종료합니다.")
    else:
        print("  ⚠️  잘못된 선택입니다.")


def parse_args():
    p = argparse.ArgumentParser(
        description="통화완화 자산 시차 분석 파이프라인 실행기 (v8a)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--all", action="store_true",
                   help="전체 실행 (01~07)")
    g.add_argument("--from-saved", action="store_true",
                   help="저장된 raw_data로 학습 (01 건너뜀, 02~07)")
    g.add_argument("--from-processed", action="store_true",
                   help="전처리 완료 데이터로 분석 (03~07)")
    g.add_argument("--steps", type=int, nargs="+", metavar="N",
                   help="원하는 단계만 (예: --steps 3 5 7)")
    g.add_argument("--step", type=int, metavar="N",
                   help="한 단계만 (예: --step 5)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.all:
        run_steps(list(STEPS.keys()))
    elif args.from_saved:
        run_steps([2, 3, 4, 5, 6, 7])
    elif args.from_processed:
        run_steps([3, 4, 5, 6, 7])
    elif args.steps:
        sel = [s for s in args.steps if s in STEPS]
        if sel:
            run_steps(sel)
        else:
            print("  ⚠️  유효한 단계가 없습니다 (1~7).")
    elif args.step:
        if args.step in STEPS:
            run_steps([args.step])
        else:
            print("  ⚠️  유효한 단계가 아닙니다 (1~7).")
    else:
        interactive_menu()
