# monetary-easing-asset-lag-predictor

# 통화 완화 환경에서의 자산군별 가격 반응 시차(Time-lag) 실증 분석

미국 거시·자산 데이터로 칸티용 효과(Cantillon Effect)를 실증한다. 통화 완화 이후
자산군이 **순차적으로** 반응하는지, 아니면 **동시에** 반응하는지를 Granger 인과,
VAR/IRF, 이벤트 스터디, 머신러닝 분류로 검증한다.

GitHub: https://github.com/kim-min-seo/monetary-easing-asset-lag-predictor

---

## 연구 질문과 핵심 결과

**원래 가설:** 통화 완화 이후 `금 → WTI → S&P500 → 부동산 → CPI` 순으로 가격이
단계적으로 전이된다.

**실증 결과 (가설 5/5 반박):**

- 데이터 기반 전이 순서(순위 평균): **S&P500 → 금(Gold) → WTI → CPI → 부동산**
- 금융자산(S&P500·금·WTI)은 1~3개월 내 **거의 동시에** 반응하는 한 클러스터를 형성하고,
  실물경제 변수(CPI·부동산)만 6~23개월의 **늦은 반응**을 보인다.
  → 전통적인 단계적 전이가 아니라 **〈금융 vs 실물〉 두 클러스터** 구조.
- 순수 통화변수만으로 월간 자산 방향을 예측하는 것은 사실상 불가능하다
  (전 자산 Hold-out MCC = 0). 200여 개 피처 엔지니어링은 평균적으로 신호가 아니라
  노이즈를 더했다(Raw 모델이 가공 모델보다 우세).

가설을 확인하지 못한 것이 아니라, **엄밀한 검증을 통해 월간 통화변수+분류 접근의
한계와 시장의 높은 정보 효율성을 드러낸 것**이 이 연구의 기여다.

---

## 프로젝트 구조

```
monetary-easing-asset-lag-predictor/
│
├── config.py                  # 공통 설정 (기간·변수·경로, .env에서 API 키 로드)
│
├── 01_data_collection.py      # 데이터 수집 (FRED · Yahoo Finance)
├── 02_preprocessing.py        # 전처리 (로그수익률 · 사이클/국면 더미 · 통화 피처)
├── 03_analysis.py             # 실증 분석 (ADF · Granger · VAR · IRF · 이벤트 스터디)
├── 04_visualization.py        # 시각화 (히트맵 · IRF CI · 전이경로 · 백테스트)
├── 05_modeling.py             # 예측 모델 (XGBoost · LightGBM · SHAP, Walk-forward)
├── 06_qvar_spillover.py       # QVAR Spillover (경기국면별 전이 구조)
├── 07_response_timing.py      # 반응 타이밍 + QE 주입 규모 분석
│
├── main.py                    # 전체 파이프라인 실행기 (단계 선택 지원)
├── make_mock_data.py          # 오프라인 테스트용 모의 데이터 생성
├── requirements.txt
│
├── data/
│   ├── raw/                   # 원본 데이터 (raw_data.csv)
│   └── processed/             # 전처리 완료 데이터 (processed_data.csv)
│
└── outputs/
    ├── figures/               # 차트 (PNG · HTML)
    └── results/               # 분석/모델 결과 (CSV)
```

> `data/`, `outputs/`는 `.gitignore` 처리되어 있어 실행 시 새로 생성된다.

---

## 설치 및 실행

### 1. 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. FRED API 키 설정

프로젝트 루트에 `.env` 파일을 만들고 [FRED](https://fred.stlouisfed.org/docs/api/api_key.html)에서
발급받은 키를 넣는다.

```
FRED_API_KEY=발급받은_키
```

`config.py`는 `.env`에서 키를 읽으므로 코드를 직접 수정할 필요는 없다.

### 3. 실행

```bash
python main.py --all            # 전체 (01~07)
python main.py --from-saved     # 수집 건너뛰고 저장된 raw_data.csv로 (02~07)
python main.py --from-processed # 전처리 완료 데이터로 (03~07)
python main.py --step 5         # 특정 단계만
python main.py --steps 3 5 7    # 여러 단계 선택
python main.py                  # 인자 없이 실행하면 대화형 메뉴
```

각 모듈은 단독 실행도 가능하다 (예: `python 03_analysis.py`).
결과는 `outputs/figures/`(차트)와 `outputs/results/`(CSV)에 저장된다.

---

## 분석 파이프라인

| 단계 | 모듈 | 내용 |
|------|------|------|
| 1 | `01_data_collection` | FRED·Yahoo Finance에서 월간 데이터 수집 (2000-01 ~ 현재) |
| 2 | `02_preprocessing` | 로그수익률, 금리인하 사이클 더미, 경기국면 변수, 통화 피처 생성 |
| 3 | `03_analysis` | ADF 정상성 · Granger 인과(AIC 시차) · VAR · IRF(95% 부트스트랩 CI) · 이벤트 스터디 · 칸티용 전이순서 |
| 4 | `04_visualization` | Granger 히트맵, IRF 신뢰구간, 전이경로 맵, 이벤트 스터디, 백테스트 차트 |
| 5 | `05_modeling` | Walk-forward 5-fold 방향 분류 (XGBoost·LightGBM) + SHAP, Raw vs 가공 비교, 선행연구 벤치마크 |
| 6 | `06_qvar_spillover` | 분위(경기국면)별 QVAR 스필오버 분석 (침체/중립/과열) |
| 7 | `07_response_timing` | 자산별 상승시작/반감/최대 반응 타이밍 + 사이클별 QE 주입 규모 |

---

## 방법론 요약

- **정상성:** ADF 검정. CaseShiller 수익률은 비정상(추세 잔존)으로 한계 명시.
- **인과:** 양변량 VAR을 cause-effect 쌍에 적합 후 **AIC로 단일 시차 선택**, 그 시차에서만
  Granger 검정 (모든 시차 최소 p 선택이 유발하는 다중검정 편향 제거).
- **충격반응:** 1차 차분 로그수익률 기준 VAR + IRF, 95% 부트스트랩 신뢰구간.
  Cholesky 순서 변경에 대한 강건성 확인.
- **이벤트 스터디:** 금리인하 시점을 baseline(=0)으로 정규화, 로그수익률 정확 누적
  (`exp(cumsum)`), **half-peak time**으로 반응 속도 측정. 진행 중 사이클은 표본에서 제외.
- **예측:** Walk-forward 5-fold 방향 분류. 폴드 내부에서만 학습하여 누수 차단.
  지표는 정확도·Base Rate·Gain·MCC·F1·Balanced Accuracy 및 Hold-out 평가.

---

## 주요 산출물

`outputs/results/` (CSV): `granger_results`, `irf_results`, `event_study_results`,
`cantillon_order`, `model_performance`, `qvar_spillover_tau_*`, `response_timing`,
`qe_response_combined` 등.

`outputs/figures/` (PNG·HTML): `granger_heatmap`, `irf_realrate`, `cantillon_path`,
`event_study_avg`/`_cycles`, `shap_*`, `backtest_*`, `model_comparison`,
`timing_profile_all`, `qe_*`, `m2_dashboard.html` 등.

---

## 데이터

- **기간:** 2000년 1월 ~ 현재 (월간)
- **자산:** 금(Gold), WTI 원유, S&P500, Case-Shiller 주택가격지수, CPI
- **통화·거시 변수:** 실질금리(Real_Rate), 기대 인플레이션(TIPS_Spread), QE 규모(Fed 대차대조표),
  M2 증가율, 통화완화지수(Monetary_Ease_Index), 정책금리 변화, 달러인덱스 등
- **출처:** FRED(거시·금리·주택·통화), Yahoo Finance(금·원유·주가)

---

## 기술 스택

| 분류 | 라이브러리 |
|------|-----------|
| 데이터 수집 | `fredapi`, `yfinance` |
| 전처리·통계 | `pandas`, `numpy`, `statsmodels`, `scipy` |
| 머신러닝·해석 | `scikit-learn`, `xgboost`, `lightgbm`, `shap` |
| 시각화 | `matplotlib`, `seaborn`, `plotly` |

> 개발 환경: Python 3.9

---

## 팀 구성

| 역할 | 담당 모듈 |
|------|----------|
| 데이터 수집 | `01_data_collection.py` |
| 전처리 | `02_preprocessing.py` |
| 실증 분석 | `03_analysis.py` |
| 시각화 | `04_visualization.py` |
| 모델링 | `05_modeling.py` |
| 전이구조·타이밍 | `06_qvar_spillover.py`, `07_response_timing.py` |

---

## 한계

- CaseShiller(부동산)는 강한 추세로 수익률 시계열이 비정상이며, 유의한 Granger 원인이
  거의 없어 방향 분류에 부적합하다.
- CPI·CaseShiller는 Base Rate가 높아(상승 편중) 방향 분류의 도전 여지가 작다.
- 월간 빈도·통화변수만으로는 자산 방향에 대한 표본외 예측력이 확인되지 않는다.
  이는 모델 결함이 아니라 시장의 정보 효율성을 시사하는 결과로 해석한다.
