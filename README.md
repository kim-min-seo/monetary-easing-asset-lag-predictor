# monetary-easing-asset-lag-predictor



예상구조:
proj/
│
├── config.py                  # 공통 설정 (API 키, 기간, 변수명 등)
│
├── 01_data_collection.py      # 데이터 수집 (FRED, Yahoo Finance)
├── 02_preprocessing.py        # 전처리 (결측치, 로그수익률, 더미변수)
├── 03_analysis.py             # 실증 분석 (ADF, Granger, VAR, IRF)
├── 04_visualization.py        # 시각화 (히트맵, 시차 플롯, IRF 그래프)
├── 05_modeling.py             # 예측 모델 (BiGRU, XGBoost, LightGBM)
│
├── data/
│   ├── raw/                   # 수집한 원본 데이터 (CSV)
│   └── processed/             # 전처리 완료 데이터 (CSV)
│
├── outputs/
│   ├── figures/               # 시각화 결과 이미지
│   └── results/               # 분석 결과 CSV, 모델 성능 표
│
└── main.py                    # 전체 파이프라인 순서대로 실행
