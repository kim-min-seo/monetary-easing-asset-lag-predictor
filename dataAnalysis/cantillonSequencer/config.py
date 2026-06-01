"""
config.py — 설정·파라미터 (계획서 §13.1)
================================================================
- .env 로드(FRED_API_KEY, 하드코딩 금지)
- ASSETS(13개; source/id/lag/seasonal/role) + WALCL_SERIES + MONETARY_SERIES
- 윈도우/타이밍/추론/QE탐지/동률/레이트리밋 상수 (계획서 명시 이름 그대로)
================================================================
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ----------------------------------------------------------------------
# 경로
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
RES_DIR = OUT_DIR / "results"
for _d in (DATA_DIR, CACHE_DIR, OUT_DIR, FIG_DIR, RES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# API 키 (.env)
# ----------------------------------------------------------------------
FRED_API_KEY = os.environ.get("FRED_API_KEY", "").strip()

# 버전 (터미널 헤더 배너용)
VERSION = "6.0"

# ----------------------------------------------------------------------
# 분석 기간 (월단위)
# ----------------------------------------------------------------------
START_DATE = "2000-01-01"
END_DATE = "2024-12-31"
FREQ = "MS"

# ----------------------------------------------------------------------
# 13개 자산  (source / id / lag / seasonal / role)
#   source  : 'fred' | 'yahoo'
#   id      : 시리즈 코드 / 티커
#   lag     : 공표 지연(개월) — 실시간 정렬 전방 시프트
#   seasonal: 'SA'(계절조정 완료) | 'NSA'(미조정 → STL 적용)
#   role    : 칸티용 사슬 위치 frontrun < financial < metal < real < tail
# ----------------------------------------------------------------------
ASSETS: dict[str, dict] = {
    "long_bond":   {"label": "장기국채",     "source": "yahoo", "id": "TLT",         "lag": 0, "seasonal": "SA",  "role": "frontrun"},
    "sp500":       {"label": "S&P500",       "source": "yahoo", "id": "^GSPC",       "lag": 0, "seasonal": "SA",  "role": "financial"},
    "gold":        {"label": "금",           "source": "yahoo", "id": "GC=F",        "lag": 0, "seasonal": "SA",  "role": "financial"},
    "copper":      {"label": "구리",         "source": "fred",  "id": "PCOPPUSDM",   "lag": 0, "seasonal": "NSA", "role": "metal"},
    "aluminum":    {"label": "알루미늄",     "source": "fred",  "id": "PALUMUSDM",   "lag": 0, "seasonal": "NSA", "role": "metal"},
    "ind_metals":  {"label": "산업금속종합", "source": "fred",  "id": "PINDUINDEXM", "lag": 0, "seasonal": "NSA", "role": "metal"},
    "steel":       {"label": "철강",         "source": "fred",  "id": "WPU101",      "lag": 1, "seasonal": "NSA", "role": "metal"},
    "wti":         {"label": "WTI원유",      "source": "fred",  "id": "MCOILWTICO",  "lag": 0, "seasonal": "NSA", "role": "metal"},
    "case_shiller":{"label": "부동산",       "source": "fred",  "id": "CSUSHPISA",   "lag": 2, "seasonal": "SA",  "role": "real"},
    "agri":        {"label": "농산물종합",   "source": "fred",  "id": "PFOODINDEXM", "lag": 1, "seasonal": "NSA", "role": "real"},
    "chem":        {"label": "화학",         "source": "fred",  "id": "WPU06",       "lag": 1, "seasonal": "NSA", "role": "real"},
    "wages":       {"label": "임금",         "source": "fred",  "id": "CES0500000003","lag": 1,"seasonal": "SA",  "role": "tail"},
    "cpi":         {"label": "CPI",          "source": "fred",  "id": "CPIAUCSL",    "lag": 1, "seasonal": "SA",  "role": "tail"},
}
ASSET_KEYS = list(ASSETS.keys())
ASSET_LABELS = {k: v["label"] for k, v in ASSETS.items()}
ROLE_ORDER = ["frontrun", "financial", "metal", "real", "tail"]

# ----------------------------------------------------------------------
# 거시/정책 시리즈
# ----------------------------------------------------------------------
WALCL_SERIES = "WALCL"        # 연준 대차대조표($백만 → $조 환산)
MONETARY_SERIES = "FEDFUNDS"  # 실효연방기금금리(정책금리)

# v5: 통합 월간 캐시 (정본) + 무결성 최소 행수
RAW_MONTHLY = CACHE_DIR / "raw_monthly.csv"
CACHE_MIN_ROWS = 200

# v5: 수집 로그용 표시 이름
DISP_NAME = {
    "long_bond": "LongBond", "sp500": "SP500", "gold": "Gold", "copper": "Copper",
    "aluminum": "Aluminum", "ind_metals": "IndMetals", "steel": "Steel", "wti": "WTI",
    "case_shiller": "CaseShiller", "agri": "AgComposite", "chem": "Chemicals",
    "wages": "Wages", "cpi": "CPI",
}

# ----------------------------------------------------------------------
# 통화 이벤트 (이중 채널). QE 는 events.detect_injection_events 가 자동 탐지.
# ----------------------------------------------------------------------
RATE_CUT_EVENTS = ["2001-01-01", "2007-09-01", "2019-08-01"]
QE_ANNOUNCED = ["2008-11-01", "2010-11-01", "2012-09-01", "2020-03-01"]
MIXED_EVENTS = ["2001-01-01", "2007-09-01", "2008-11-01", "2010-11-01",
                "2012-09-01", "2019-08-01", "2020-03-01"]

# ----------------------------------------------------------------------
# 윈도우  (계획서 §13.1)
# ----------------------------------------------------------------------
EVENT_PRE_MONTHS = 60
EVENT_POST_MONTHS = 60
MIN_POST_MONTHS = 36

# ----------------------------------------------------------------------
# 급격한 QE 구간 분석
#   점(point) 이벤트와 별개로, WALCL 이 급격히 증가하는 실제 급격한 QE 구간을
#   start 를 t=0 으로 잡고 [start, end + PROGRAM_POST_YEARS] 까지를 분석창으로 본다.
#   post_m = (end−start 개월) + 12·PROGRAM_POST_YEARS
# ----------------------------------------------------------------------
INJECTION_PROGRAMS = [
    {"label": "QE1",    "start": "2008-11-25", "end": "2010-03-31"},
    {"label": "QE3",    "start": "2012-09-13", "end": "2014-10-29"},
    {"label": "QEinf",  "start": "2020-03-15", "end": "2022-03-10"},
]
PROGRAM_POST_YEARS = 3
PROGRAM_PRE_MONTHS = 12   # 급격한 QE 구간 직전 변동성 밴드 산정용 사전 윈도우

# v6: 이벤트 선택 정교화
QE_EVENT_MIN_DWALCL = 0.10   # QE 증가 이벤트 의미성 하한($조)
QE_EVENT_TARGET_N = 6        # QE 증가 이벤트 최대 개수(약 5~6)
RATE_EVENT_TARGET_N = 6      # 금리 인하 이벤트 최대 개수(약 5~6)
EVENT_MIN_GAP_M = 6          # 이벤트 간 최소 간격(개월, 중복 제거)

# v3: 사후창 2종 병행(+3년 / +1.5년). tag 는 파일명 접미사로 사용.
PROGRAM_POST_YEARS_SET = [3.0, 1.5]
QE_SURGE_HORIZON_YEARS_SET = PROGRAM_POST_YEARS_SET   # v6 정규 명칭 별칭


def post_tag(years: float) -> str:
    """3.0 → '3y', 1.5 → '1p5y'."""
    return f"{years:g}y".replace(".", "p")


# 순서성 강한 자산 선별 — Kendall's W ≥ 임계 를 만족하는 최대 N
STRONG_W_THRESHOLD = 0.80      # 순서성 임계(강한 순서성)
STRONG_N_MIN = 3               # 최소 자산 수
STRONG_N_MAX = len(ASSETS)     # 최대(= 전체)
STRONG_SELECT_METHOD = "max_n_over_thr"   # 'max_n_over_thr' | 'max_w' | 'min_var' | 'role_repr'
# 하위호환 별칭
TOP_W_THRESHOLD = STRONG_W_THRESHOLD
TOP_N_MIN = STRONG_N_MIN
TOP_N_MAX = STRONG_N_MAX
TOP_SELECT_METHOD = STRONG_SELECT_METHOD
TOP_K_ASSETS = 5

# ----------------------------------------------------------------------
# 타이밍  (계획서 §13.1)
#   onset = max(ONSET_VOL_K·pre_vol, ONSET_MIN_PCT/100) 를
#           ONSET_PERSIST_M 개월 지속 돌파하는 최초 시점
# ----------------------------------------------------------------------
ONSET_VOL_K = 1.5
ONSET_MIN_PCT = 0.5     # 최소 반응 floor (%) → 0.5% = 0.005
ONSET_PERSIST_M = 2
HALF_FRAC = 0.5
DRAWDOWN_THETA = 0.30   # peak 종료 드로다운 임계 θ

# ----------------------------------------------------------------------
# 추론  (계획서 §13.1)
# ----------------------------------------------------------------------
SHOCK_AR_LAGS = 3
LP_HORIZON = 36
LP_CONTROL_LAGS = 3
LP_HAC_LAGS = 6
SIG_Z = 1.96
VAR_HORIZON = 36
VAR_MAXLAGS = 6

# ----------------------------------------------------------------------
# QE 주입 국면 탐지  (계획서 §13.1)
# ----------------------------------------------------------------------
INJ_GROWTH_THR = 0.015
INJ_SMOOTH = 3
INJ_MIN_LEN = 3

# ----------------------------------------------------------------------
# 동률  (계획서 §13.1)
# ----------------------------------------------------------------------
TIE_TOL_MONTHS = 1
TIE_TOL_PCT = 0.10

# ----------------------------------------------------------------------
# 레이트리밋  (계획서 §13.2)
# ----------------------------------------------------------------------
FRED_THROTTLE = 0.6
YAHOO_THROTTLE = 1.0
MAX_TRIES = 6
BACKOFF_BASE = 2.0   # 2→4→8→16→32→64s

# ----------------------------------------------------------------------
# 시각화
# ----------------------------------------------------------------------
DPI = 130
FIGSIZE_WIDE = (15, 7)
KOR_FONT_CANDIDATES = ["NanumGothic", "Malgun Gothic", "AppleGothic",
                       "Noto Sans CJK KR", "DejaVu Sans"]

# 재현성
SYNTH_SEED = 20250531


def summary() -> str:
    return (
        f"기간 {START_DATE[:7]}~{END_DATE[:7]} | 자산 {len(ASSETS)} | "
        f"인하 {len(RATE_CUT_EVENTS)} | QE발표 {len(QE_ANNOUNCED)} | "
        f"INJ_THR={INJ_GROWTH_THR} | TIE=({TIE_TOL_MONTHS}M,{TIE_TOL_PCT}) | "
        f"onset(K={ONSET_VOL_K},floor={ONSET_MIN_PCT}%) | LP H={LP_HORIZON} | "
        f"FRED_KEY={'set' if FRED_API_KEY else 'EMPTY→synthetic'}"
    )


if __name__ == "__main__":
    print(summary())


# ======================================================================
# 상세 터미널 출력 (V8 스타일) — 구 logging_util 흡수 (계획서 v4 §13.1)
# ======================================================================
import time as _time
from contextlib import contextmanager as _contextmanager
from datetime import datetime as _datetime

_LOG_W = 62
_VERBOSE = True
_ICON = {"ok": "✓", "data": "📊", "find": "🔎", "calc": "🧮",
         "fig": "🖼", "warn": "⚠️", "info": "·", "time": "⏱", "done": "✅"}


def set_verbose(v: bool) -> None:
    global _VERBOSE
    _VERBOSE = bool(v)


def is_verbose() -> bool:
    return _VERBOSE


def _logline(ch: str = "=") -> str:
    return ch * _LOG_W


def banner(mode: str, synth: bool) -> None:
    now = _datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = "합성(synthetic)" if synth else "실데이터(FRED+Yahoo)"
    print(_logline())
    print("  통화 완화 환경에서의 자산군별 가격 반응 시차 실증 분석")
    print(f"  ★ cantillon-sequencer  VERSION {VERSION}")
    print(f"  실행 모드: {mode}   |   데이터: {data}")
    print(f"  실행 시작: {now}")
    print(_logline())


def step(i: int, n: int, name: str) -> None:
    if not _VERBOSE:
        return
    print(_logline()); print(f"  STEP {i}/{n} — {name}"); print(_logline())


def status(msg: str, level: str = "info") -> None:
    if not _VERBOSE:
        return
    print(f"  {_ICON.get(level, '·')} {msg}")


def warn(msg: str) -> None:
    print(f"  {_ICON['warn']} {msg}")


@_contextmanager
def timed(name: str):
    t0 = _time.time()
    try:
        yield
    finally:
        if _VERBOSE:
            print(f"  {_ICON['time']} {name} 완료 ({_time.time() - t0:.1f}s)")


def footer(n_fig: int, n_csv: int, elapsed: float) -> None:
    now = _datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(_logline())
    print(f"  {_ICON['done']} 완료  |  총 소요 {elapsed:.1f}s  |  완료 시각 {now}")
    print("  결과 위치:")
    print(f"    outputs/figures/ → 차트 PNG (총 {n_fig}개)")
    print(f"    outputs/results/ → 분석 CSV (총 {n_csv}개)")
    print(_logline())


def count_outputs() -> tuple[int, int]:
    return (len(list(FIG_DIR.glob("*.png"))), len(list(RES_DIR.glob("*.csv"))))


# ======================================================================
# v5: 항목 단위 진행 로그 (수집·캐시·그림)
# ======================================================================
def collect_header(n_fred: int, n_yahoo: int) -> None:
    if not _VERBOSE:
        return
    print("  · 실데이터 수집 시작 (FRED + Yahoo)")
    print(f"  · FRED 요청: {n_fred}개  ·  Yahoo 요청: {n_yahoo}개")


def collect_line(i: int, n: int, source: str, name: str, code: str,
                 rows: int | None, resampled: int | None = None,
                 ok: bool = True, retry: int | None = None) -> None:
    # 성공 라인은 verbose 에서만, 실패는 항상 출력
    if ok and not _VERBOSE:
        return
    if ok:
        extra = f"  (월말 리샘플 → {resampled}행)" if (resampled is not None and resampled != rows) else ""
        print(f"  [{i}/{n}] {_ICON['ok']} {source} {name} ({code}): {rows}행{extra}")
    else:
        rtx = f" (재시도 {retry}회 후)" if retry else ""
        print(f"  [{i}/{n}] ✗ {source} {name} ({code}): 수집 실패{rtx} → 스킵")


def collect_summary(ok: int, total: int) -> None:
    print(f"  · 수집 성공: {ok}/{total}  (실패 {total - ok})")


def cache_confirm(path, reused: bool = False, shape=None) -> None:
    kb = (path.stat().st_size / 1024) if path.exists() else 0.0
    if reused:
        sh = f", {shape[0]}×{shape[1]}" if shape else ""
        print(f"  {_ICON['ok']} {path.name} 캐시 사용 ({kb:.0f}KB{sh})")
    else:
        print(f"  {_ICON['ok']} {path.name} 캐시 생성 확인 ({kb:.0f}KB)")


def data_shape(rows: int, cols: int) -> None:
    print(f"  · 데이터 shape: {rows}개월 × {cols}컬럼")


def fig_line(i: int, n: int, name: str) -> None:
    if not _VERBOSE:
        return
    print(f"  [{i}/{n}] {_ICON['fig']} {name}.png")


def fig_summary(ok: int, total: int, fail: int) -> None:
    print(f"  · 그림 저장: {ok}/{total} (실패 {fail})  → outputs/figures/")
