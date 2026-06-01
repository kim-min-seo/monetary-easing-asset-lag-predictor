<div align="center">

# 🌊 cantillon-sequencer

**통화 주입 전이 순서 분석기 · Cantillon Effect Sequencer**

유동성 주입의 *시점·규모*를 기준으로 자산군이 **어떤 순서로 반응하는지** 실증하고,
순서성이 뚜렷한 자산군을 자동 선별한다.

![python](https://img.shields.io/badge/python-3.10%2B-blue)
![status](https://img.shields.io/badge/status-research-orange)
![data](https://img.shields.io/badge/data-FRED%20%2B%20Yahoo-success)
![method](https://img.shields.io/badge/method-Local%20Projection%20%7C%20VAR--IRF-informational)
![tests](https://img.shields.io/badge/tests-pytest-green)

</div>

---

## 📌 한 줄 요약

> 새로 풀린 돈은 **진입점에서 시작해 시차를 두고 순차 전이**된다(칸티용 효과). 이 프로젝트는 자산의 **상승 폭이 아니라 반응 *순서*와 *시점***을 ML/계량으로 검증한다.

- 금리 인하와 양적완화(QE)는 돈이 들어오는 방식이 달라 **채널을 분리**해 분석한다.
- 통화 이벤트는 **급격한 QE 구간**(QE1/QE3/QEinf, 사후 관측 +3년·+1.5년)과 **금리 인하 이벤트**(큰 인하 상위 ~5~6) 2종으로 본다.
- 순서 일치도는 **Kendall's W** 로 계량하고, **W ≥ 0.80 을 만족하는 가장 큰 N개** 자산을 **순서성 강한 자산**으로 자동 선별한다.

---

## ✨ 주요 기능

- **이중 채널 분리** — 금리충격·QE충격을 한 Local Projection 에 함께 넣어 상호 통제(2008·2020 동시 발생도 분리 식별).
- **급격한 QE 구간 자동 정의** — 발표일이 아닌 *실제 확장 구간*을 사용(QE3 → 2013 자동 재지정), 사후 관측 **+3년 / +1.5년 병행**. **+3년 장이 QE 경로의 기본 표현**.
- **금리 인하 이벤트 분석** — 정책금리 하강 사이클에서 **총 인하폭(bp) 상위 ~5~6** 시점을 선택, 순위·일관성·**전이 사슬**·LP 동형 분석.
- **순서성 강한 자산 선별** — 순서성(Kendall's W ≥ **0.80**)이 유지되는 한 **가장 많은 N개** 자산을 선택, 어느 창에서 뽑혔는지 명시.
- **채널 비교** — **급격한 QE 구간 +3년 장 vs 금리 인하 이벤트**의 자산 순서를 슬로프그래프로 비교.
- **추론 교차검증** — Local Projection(주) + LP 순서 수치화(표·일치도), VAR-IRF·Kaplan-Meier 해저드(보조).
- **상세 터미널 출력** — 단계 배너·진행 건수·소요시간·완료 요약. `--quiet` 로 끔.
- **그림 설명서** — `outputs/figures_guide.md` 자동 생성(각 그림의 수치 속성·기법·읽는 법).
- **합성 폴백** — `FRED_API_KEY` 없이도 합성 데이터로 전체 파이프라인 동작(데모·검증).

---

## 🔧 설치 & 실행

```bash
git clone https://github.com/kim-min-seo/cantillon-sequencer.git
cd cantillon-sequencer
pip install -r requirements.txt
```

### 🔑 `.env` 파일 만들기 & 설정

FRED API 키를 `.env` 에 넣으면 실데이터로 동작합니다. (키가 없으면 자동으로 합성 데이터 폴백 — 동작 검증용)

1. **FRED API 키 발급 (무료)**
   - https://fredaccount.stlouisfed.org/apikeys 접속 → 로그인/가입 → **Request API Key** → 32자리 키 복사.
2. **`.env` 파일 생성** — 프로젝트 루트(`cantillon-sequencer/`)에 생성합니다.
   ```bash
   cp .env.example .env          # 예시 파일 복사 (없으면 아래처럼 새로 만들기)
   # 또는 직접 생성:
   #   echo "FRED_API_KEY=여기에_발급받은_키" > .env
   ```
3. **`.env` 내용** — 아래 한 줄을 본인 키로 채웁니다(따옴표·공백 없이).
   ```dotenv
   FRED_API_KEY=abcdef0123456789abcdef0123456789
   ```
   - Yahoo Finance 데이터는 별도 키가 필요 없습니다.
   - `.env` 는 `.gitignore` 에 포함되어 커밋되지 않습니다(키 노출 방지).
4. **확인** — `python main.py --all --fresh` 실행 시 수집 로그에 `합성 데이터` 경고가 **뜨지 않고** `수집 성공: 15/15` 가 보이면 실데이터로 정상 동작하는 것입니다.

### 실행

```bash
python main.py                # 대화형 메뉴
python main.py --all          # 전체(급격한 QE 구간 ±3/1.5년 + 금리 인하 + 채널비교 + 순서성 강한 자산)
python main.py --qe-surge     # 급격한 QE 구간 분석(±3/1.5년)
python main.py --qe-surge --post 3     # +3년 창만
python main.py --rate-cuts    # 금리 인하 이벤트 분석(전이 사슬 포함)
python main.py --strong       # 순서성 강한 자산 선별(W≥0.8 최대 N, 두 창 승자)
python main.py --guide        # outputs/figures_guide.md 만 생성
python main.py --sample       # 합성 데이터(키 불필요)
python main.py --steps 5 6 7  # 특정 단계만
python main.py --clean        # cache·결과 삭제
python main.py --all --fresh  # 초기화 후 전체 새로 생성
```

> ⚠️ 저장소에 동봉된 그림·CSV 는 **합성 데이터** 기반 동작 검증용입니다. 실제 실증 결과는 `.env` 에 본인 `FRED_API_KEY` 를 넣고 `python main.py --all --fresh` 로 재생성해야 합니다.

---

## 🧪 방법론

| 항목 | 결정 |
|------|------|
| 순서 기준 | `onset_month`(반응 시작) 주, `peak`(정점) 보조 |
| onset | 사전 변동성 밴드 `max(1.5·prevol, 0.5%)` 를 2개월 지속 돌파 |
| peak | 부호 인지 + 드로다운 임계(θ=30%) 종료 → 첫 지속 정점 |
| 변환 | 중심화 누적 로그변화 `exp(logLₜ−logL_event)−1`, 정상성 ADF 보조 |
| 발표 시차 | 공표 지연 전방 시프트(CPI/PPI/임금 +1, Case-Shiller +2) |
| 충격 식별 | 정책금리·WALCL 변화의 AR 잔차 (rate/qe/easing) |
| 동률 | onset 차 ≤ 1개월 **AND** peak 상대차 ≤ 10% |
| peak 제약 | peak 는 onset 이후 구간에서만 탐색 → `peak_month ≥ onset_month`(역행 없음) |
| 순서성 강한 자산 | Kendall's W ≥ **0.80** 을 만족하는 **최대 N** 부분집합 |

### 검증 가설

- **H1** — 급격한 QE 구간 / 금리 인하 이벤트 Kendall's W > 혼합 풀링 W (분리·구간화가 순서를 선명하게).
- **H2** — 금융(국채·주식·금) → 금속 → 실물(부동산) → CPI 후행 (role 사슬 단조).
- **H3** — 주입 규모(ΔWALCL)·인하폭(bp) 클수록 빠른·큰 반응.

---

## 🗂 프로젝트 구조


```
cantillon-sequencer/
├── config.py        # 설정·상수·.env + 로깅 헬퍼
├── data.py          # 수집(레이트리밋·캐시·합성) + 변환 + 이벤트(급격한 QE 구간·금리 인하) + 충격
├── analysis.py      # 타이밍(onset/peak) + 순서(Kendall's W) + 순서성 강한 자산 선별 + LP 순서 수치화
├── inference.py     # Local Projection(채널분리) + VAR-IRF + 해저드
├── viz.py           # 시각화 일체(서술·추론·순서·급격한 QE 구간·금리 인하·순서성 강한 자산) + 그림 설명서
├── main.py          # 오케스트레이션 + CLI + 대화형 메뉴
├── tests/test_all.py
├── requirements.txt / .env / .gitignore
└── outputs/{figures,results}/ , data/cache/
```

---

## 📊 산출물

**그림** — WALCL 타임라인(급격한 QE 구간 음영 + 자산 마커 범례), 순위 매트릭스·일관성·lead-lag·전이 사슬(급격한 QE 구간·**금리 인하 이벤트**·순서성 강한 자산), LP 충격반응 패널·**LP 순서 수치 표/일치도**, **채널 비교(QE 3년 vs 금리 인하) 슬로프그래프**, 규모-반응 산점/버블, 타이밍 프로파일, 순서성 강한 자산 동형 시각화. + `outputs/figures_guide.md`.

> ordering_consistency 계열은 평균순위로 정렬하며 **순위 숫자가 클수록 위, 작을수록 아래**로 표시한다.

**결과 CSV** — `qe_surge_events.csv`·`rate_cut_events.csv`·`events.csv`, `timing_*.csv`, `shocks.csv`, `lp_scores_*.csv`·`lp_ordering_scores_*.csv`, `var_irf_scores.csv`, `hazard_onset.csv`, `W_comparison.csv`·`rate_W_comparison.csv`, `channel_comparison.csv`, `strong_assets.csv`(출처 창·N·W 명시) 등.



