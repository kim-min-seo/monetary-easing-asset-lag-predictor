# ============================================================
#  config.py — 공통 설정값 (v7 Hybrid)
#  ★ API 키는 .env 파일에서 로드
#  ★ v7 추가: VIX (SP500 개선), PPI (CPI 개선)
# ============================================================

import os
from datetime import datetime
from pathlib import Path

# ──────────────────────────────────────────────
#  .env 로드 (API 키 보안)
# ──────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv 없으면 환경변수 직접 사용

FRED_API_KEY = os.getenv("FRED_API_KEY", "your_fred_api_key_here")

# ──────────────────────────────────────────────
#  분석 기간
# ──────────────────────────────────────────────
START_DATE = "2000-01-01"
END_DATE   = datetime.today().strftime("%Y-%m-%d")

# ──────────────────────────────────────────────
#  FRED 시리즈 ID
# ──────────────────────────────────────────────
FRED_SERIES = {
    "FedRate":    "FEDFUNDS",
    "Fed_Assets": "WALCL",
    "T10Y":       "DGS10",
    "T2Y":        "DGS2",
    "CPI":        "CPIAUCSL",
    "M2":         "M2SL",
    "GDP":        "GDP",
    "CaseShiller":"CSUSHPISA",
    "TIPS_10Y":   "DFII10",   # 기대 인플레이션
    "PPI":        "PPIACO",   # 생산자물가지수
    "PPI_Core":   "PPIFGS",   # 완제품 PPI
    # ★ v8 추가: WTI 공급측 변수
    "Oil_Stocks": "WTESTUS",  # 미국 원유 재고 (EIA 주간→월간)
    "Oil_Prod":   "MCRFPUS2", # 미국 원유 생산량
}

# ──────────────────────────────────────────────
#  Yahoo Finance 티커
# ──────────────────────────────────────────────
YAHOO_TICKERS = {
    "Gold":  "GC=F",
    "WTI":   "CL=F",
    "DXY":   "DX-Y.NYB",
    "SP500": "^GSPC",
    "VIX":   "^VIX",    # 변동성 지수
    # ★ v8 추가: 자산별 개선 변수
    "OVX":    "^OVX",    # 원유 변동성 (WTI 개선)
    "GLD":    "GLD",     # Gold ETF (금 자금흐름 proxy)
    "GDX":    "GDX",     # 금광업 ETF (Gold 선행)
    "Silver":  "SI=F",   # 은 선물 (금-은 비율용)
}

# ──────────────────────────────────────────────
#  자산별 그랜저 시차 (가설 기반)
# ──────────────────────────────────────────────
ASSET_LAG_MAP = {
    "Gold_LogReturn":         6,
    "WTI_LogReturn":         12,
    "SP500_LogReturn":       12,
    "CaseShiller_LogReturn": 24,
    "CPI_LogReturn":         24,
}

# ──────────────────────────────────────────────
#  통화환경 독립변수
# ──────────────────────────────────────────────
MONETARY_VARS = [
    "FedRate_Change",
    "FedRate_Change3M",
    "Real_Rate",
    "QE_Size",
    "DXY_Change",
    "Yield_Spread",
    "Excess_Liquidity",
    "M2_YoY",
    "M2_MoM",
    "M2_Accel",
    "TIPS_Spread",       # 기대 인플레이션
    "Inflation_Expect",  # 10Y - TIPS (BEI)
    "PPI_YoY",           # ★ v7 추가: 생산자물가 전년비 (CPI 선행)
    "PPI_LogReturn",     # ★ v7 추가: PPI 월간 변화율
    "VIX_Level",         # ★ v7 추가: 변동성 지수 (SP500 역행)
    "VIX_Change",        # VIX 변화율
    # ★ v8 추가
    "OVX_Level",         # 원유 변동성 (WTI 공급 불확실성)
    "Gold_Silver_Ratio", # 금-은 비율 (안전자산 수요 강도)
    "GLD_Flow",          # Gold ETF 자금 흐름 proxy
    "Oil_Stocks_YoY",    # 원유 재고 전년비
    "Oil_Prod_YoY",      # 원유 생산 전년비
]

# ──────────────────────────────────────────────
#  금리인하 사이클
# ──────────────────────────────────────────────
RATE_CUT_CYCLES = [
    ("2001-01-01", "2003-06-30"),
    ("2007-09-01", "2009-12-31"),
    ("2019-08-01", "2020-03-31"),
    ("2020-03-01", "2022-02-28"),
    ("2024-09-01", "2025-12-31"),
]

# ──────────────────────────────────────────────
#  피처 설정
# ──────────────────────────────────────────────
LAG_PERIODS  = [1, 2, 3, 6, 9, 12, 18, 24]
MA_WINDOWS   = [3, 6, 12]
TOP_FEATURES = 50

# ──────────────────────────────────────────────
#  VAR
# ──────────────────────────────────────────────
VAR_MAX_LAG = 12

# ──────────────────────────────────────────────
#  모델 설정
# ──────────────────────────────────────────────
LOOKBACK      = 12
EPOCHS        = 80
BATCH         = 16
GRU_UNITS     = 64
DROPOUT       = 0.3
LR            = 0.001
OPTUNA_TRIALS = 50

# ──────────────────────────────────────────────
#  백테스팅
# ──────────────────────────────────────────────
WF_SPLITS  = 5
MIN_TRAIN  = 60

# ──────────────────────────────────────────────
#  ★ v8: 자산별 예측 시차 (개선)
#  Gold: 6개월 후 방향 (실질금리 → Gold 반응 3~6개월)
#  WTI:  3개월 후 방향 (단기 공급 신호 포착)
#  SP500, CaseShiller, CPI: 1개월 유지
# ──────────────────────────────────────────────
PREDICTION_LAG = {
    "Gold":        6,
    "WTI":         3,
    "SP500":       1,
    "CaseShiller": 1,
    "CPI":         1,   # CPI는 가속도 예측으로 변환
}

# ──────────────────────────────────────────────
#  경로 (자동 생성)
# ──────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
DATA_RAW_DIR  = BASE_DIR / "data" / "raw"
DATA_PROC_DIR = BASE_DIR / "data" / "processed"
FIG_DIR       = BASE_DIR / "outputs" / "figures"
RESULT_DIR    = BASE_DIR / "outputs" / "results"

for d in [DATA_RAW_DIR, DATA_PROC_DIR, FIG_DIR, RESULT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# str 변환 (하위 호환)
DATA_RAW_DIR  = str(DATA_RAW_DIR)
DATA_PROC_DIR = str(DATA_PROC_DIR)
FIG_DIR       = str(FIG_DIR)
RESULT_DIR    = str(RESULT_DIR)
