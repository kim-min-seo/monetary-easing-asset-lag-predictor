# ============================================================
#  config.py — 공통 설정값 (v6)
#  ★ API 키는 .env 파일에서 로드
#  ★ 여기서 분석 파라미터만 수정
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
    "TIPS_10Y":   "DFII10",   # ★ v6 추가: 기대 인플레이션 (Gold 유의성 확보)
}

# ──────────────────────────────────────────────
#  Yahoo Finance 티커
# ──────────────────────────────────────────────
YAHOO_TICKERS = {
    "Gold":  "GC=F",
    "WTI":   "CL=F",
    "DXY":   "DX-Y.NYB",
    "SP500": "^GSPC",
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
    "TIPS_Spread",       # ★ v6 추가: 기대 인플레이션
    "Inflation_Expect",  # ★ v6 추가: 10Y - TIPS (BEI)
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
