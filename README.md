# monetary-easing-asset-lag-predictor


# 📈 통화 완화 환경에서의 자산군별 가격 반응 시차 실증 분석

> **통화 완화 이후, 금 → WTI → S&P500 → 부동산 → CPI 순서로  
> 가격이 전이되는가?** 칸티용 효과를 머신러닝으로 실증합니다.

---

## 🔬 연구 질문

| # | 질문 | 방법 |
|---|------|------|
| A | 통화 완화 이후 자산군별 가격 반응에 **시차(Time-lag)가 존재하는가?** | Granger 인과검정, VAR |
| B | 존재한다면, **어떤 순서**로 나타나는가? | IRF (충격반응함수) |
| C | 선행 자산으로 후행 자산을 **예측할 수 있는가?** | BiGRU, XGBoost, LightGBM |

---

## 🗂️ 프로젝트 구조

```
proj/
│
├── config.py                  # ⚙️  공통 설정 (API 키, 기간, 변수명, 경로)
│
├── 01_data_collection.py      # 📥  데이터 수집   (FRED / Yahoo Finance)
├── 02_preprocessing.py        # 🧹  전처리        (결측치 · 로그수익률 · 더미변수)
├── 03_analysis.py             # 🔍  실증 분석     (ADF · Granger · VAR · IRF)
├── 04_visualization.py        # 📊  시각화        (히트맵 · 시차 플롯 · IRF 그래프)
├── 05_modeling.py             # 🤖  예측 모델     (BiGRU · XGBoost · LightGBM)
├── 06_modeling2.py            # 🤖  기타 예측 모델......
│
├── data/
│   ├── raw/                   # 원본 데이터 (수집 직후 CSV)
│   └── processed/             # 전처리 완료 데이터 (CSV)
│
├── outputs/
│   ├── figures/               # 시각화 결과 이미지 (PNG)
│   └── results/               # 분석 결과 · 모델 성능 (CSV)
│
└── main.py                    # 🚀  전체 파이프라인 일괄 실행
```

---

## ⚡ 실행 방법

### 환경 설정
```bash
pip install -r requirements.txt
```

### config.py에 API 키 입력
```python
FRED_API_KEY = "your_api_key_here"
```

### 단계별 실행
```bash
python 01_data_collection.py   # 원본 데이터 수집
python 02_preprocessing.py     # 전처리 및 피처 생성
python 03_analysis.py          # Granger / VAR / IRF 분석
python 04_visualization.py     # 차트 및 그래프 출력
python 05_modeling.py          # 예측 모델 학습 및 평가
```

### 전체 파이프라인 한 번에
```bash
python main.py
```

---

## 📦 변수 구성

### 분석 대상 자산군
| 자산 | 데이터 출처 | 역할 |
|------|------------|------|
| 금 (Gold) | Yahoo Finance (GLD) | 선행 지표 후보 |
| WTI 원유 | Yahoo Finance (USO) | 중간 전이 |
| S&P 500 | Yahoo Finance (^GSPC) | 중간 전이 |
| 부동산 (Case-Shiller HPI) | FRED | 후행 지표 |
| CPI | FRED | **종속변수** (최종 인플레이션) |

### 독립변수 (통화 환경 지표)
| 구분 | 변수 | 출처 |
|------|------|------|
| 핵심 | 금리 변화율 | FRED (FEDFUNDS) |
| 핵심 | 실질금리 | FRED (DFII10) |
| 핵심 | QE 규모 (Fed 자산) | FRED (WALCL) |
| 핵심 | 달러인덱스 | Yahoo Finance (DX-Y.NYB) |
| 추가 | 수익률곡선 (10Y-2Y) | FRED |
| 추가 | 초과유동성 | 계산 (M2 - 명목 GDP) |

### 이벤트 더미변수 (금리 인하 사이클)
| 시점 | 이벤트 |
|------|--------|
| 2001 | 닷컴버블 대응 |
| 2007 | 금융위기 QE |
| 2019 | 예방적 인하 |
| 2020 | 팬데믹 긴급 완화 |
| 2024 | 피벗 시작 |

> 각 사이클은 **인하 시작 / 유지 / 인상 전환** 3단계 더미로 처리

---

## 🔄 분석 파이프라인

```
FRED API / Yahoo Finance
        ↓  01_data_collection.py
   원본 데이터 저장 (data/raw/)
        ↓  02_preprocessing.py
   로그수익률 · 더미변수 · 정상성 확보
        ↓  03_analysis.py
   Granger 인과검정 → VAR 모델 → IRF
   ✅ 최적 시차(lag) 도출
        ↓  04_visualization.py
   히트맵 · 시차 플롯 · 전이 순서 플로우
        ↓  05_modeling.py
   BiGRU (PyTorch) + XGBoost + LightGBM
   Optuna 튜닝 → Walk-forward CV → SHAP
        ↓
   outputs/ 에 결과 저장
```

---

## 🌍 검증 계획

```
1차: 미국 데이터  →  모델 구축 및 성능 확인
                        ↓
2차: 한국 데이터  →  데이터만 교체하여 범용성·재현성 검증
```

---

## 🛠️ 기술 스택

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)

| 분류 | 라이브러리 |
|------|-----------|
| 데이터 수집 | `fredapi`, `yfinance`, `pandas-datareader` |
| 전처리 / 분석 | `pandas`, `numpy`, `statsmodels` |
| 시각화 | `matplotlib`, `seaborn` |
| ML 모델 | `xgboost`, `lightgbm`, `torch` |
| 튜닝 / 해석 | `optuna`, `shap` |

---

## 👥 팀 구성

| 역할 | 담당 파일 |
|------|----------|
| 데이터 수집 | `01_data_collection.py` |
| 전처리 | `02_preprocessing.py` |
| 실증 분석 | `03_analysis.py` |
| 시각화 | `04_visualization.py` |
| 모델링 | `05_modeling.py` |

---

> 분석 기간: **2000년 1월 ~ 현재** (월별 데이터)
