# 실데이터 검증 체크리스트 (real-data validation)

> 동봉된 그림·CSV 는 **합성 데이터(`--sample`) 기반 동작 검증용**입니다. 실제 실증 결과는 본인 `FRED_API_KEY` 로 재생성해야 합니다. 아래 순서대로 점검하세요.

---

## 0. 사전 준비
- [ ] `pip install -r requirements.txt` 완료 (pandas/numpy/scipy/statsmodels/matplotlib/fredapi/yfinance).
- [ ] FRED API 키 발급 → `.env` 에 `FRED_API_KEY=발급키` 입력 (`.env.example` 복사).
- [ ] 네트워크에서 `api.stlouisfed.org`(FRED), `query*.finance.yahoo.com`(Yahoo) 접근 가능.
- [ ] 한글 폰트(Noto Sans CJK 등) 설치 — 없으면 그림 한글이 깨질 수 있음(경고만 뜨고 진행).

## 1. 수집 단계 로그 확인 (`python main.py --all --fresh`)
- [ ] `· 수집 성공: N/15` 에서 **15/15** 인지(13 자산 + WALCL + FedRate). 일부 실패 시 해당 시리즈 코드 확인.
- [ ] WALCL 줄에 `(월말 리샘플 → R행)` 표기 — 주간 시리즈가 월말로 정상 변환됐는지.
- [ ] `✓ raw_monthly.csv 캐시 생성 확인 (NN KB)` 출력 — 통합 캐시 기록 성공.
- [ ] `· 데이터 shape: M개월 × 15컬럼` 에서 M 이 2000-01~현재 길이(약 305+)인지.
- [ ] 합성 폴백 경고(`⚠️ ... 합성 데이터`)가 **뜨지 않아야** 함(뜨면 키/네트워크 문제로 합성으로 빠진 것).

## 2. 이벤트 정의 점검
- [ ] **급격한 QE 구간 3개** = QE1/QE3/QEinf, 각 `+$?T` 누적 증가가 양수·합리적 규모(QEinf 가 가장 큼).
- [ ] QE3 시작이 발표(2012-09)가 아닌 **실제 확장(≈2013)** 으로 잡혔는지(자동 재지정).
- [ ] **금리 인하 이벤트 ~5~6건**: 2001·2007·2019·2020(·2024) 사이클이 큰 인하폭(bp) 순으로 선택됐는지(`rate_cut_events.csv`).
- [ ] 이벤트 간 간격이 `EVENT_MIN_GAP_M`(6개월) 이상으로 중복 없이 선택됐는지.

## 3. 타이밍·순서 정합성
- [ ] `timing_*.csv` 에서 **모든 행 `peak_m ≥ onset_m`**(역행 없음). (위반 시 버그 — 보고)
- [ ] `half_m` 이 onset~peak 사이 값인지(이탈 없으면 정상).
- [ ] onset 미발생(=NaN) 비율이 과도하지 않은지(자산 대부분 onset 잡혀야).

## 4. 가설(H1/H2/H3) — 방향성 확인 (확정 인과 아님)
- [ ] **H1**: `W_comparison.csv` 에서 `qe_surge_3y` / `rate_cuts` 의 Kendall's W > `mixed(all)` (구간화·분리가 순서를 선명하게).
- [ ] **H2**: `transmission_chain*` / `ordering_consistency*` 에서 대체로 **금융(국채·주식·금) → 금속 → 실물(부동산) → CPI·임금** 방향인지(role 사슬 단조, Spearman>0).
- [ ] **H3**: `*_magnitude_scatter` 에서 규모(ΔWALCL)·인하폭(bp) 클수록 빠른/큰 반응 경향인지(`*_hypotheses.csv` 의 amp_rho 부호).

## 5. 순서성 강한 자산 (W ≥ 0.80)
- [ ] `strong_assets.csv` 의 `qualified=True` 인지(0.80 미달 시 `False` + N=`STRONG_N_MIN` 경고 — 표본 한계로 해석).
- [ ] `source_window`(3y/1p5y)·`W_selected`·양쪽 `(N,W)` 가 기재됐는지.
- [ ] `ordering_consistency_strong_from*.png` 에서 **순위 큰 자산이 위, 작은 자산이 아래**(필수 규칙) 로 렌더됐는지.

## 6. 채널 비교 & LP 추론
- [ ] `channel_comparison.csv`/`channel_compare.png` 가 **급격한 QE 구간 +3년 vs 금리 인하 이벤트** 순서를 비교하는지(좌=금리, 우=QE 3년).
- [ ] `lp_irf_{qe,rate}.png` 충격반응이 발산하지 않고 신뢰구간이 합리적인지.
- [ ] `lp_concordance_qe`(QE 채널 ↔ QE 3년 장 순서)·`lp_concordance_rate`(금리 채널 ↔ 금리 인하 순서) 의 **Spearman ρ 부호·크기** 확인 — LP와 이벤트-스터디가 대체로 합치하는지.
- [ ] `method_compare.png` 에서 LP·이벤트(3년)·VAR 순서 간 상관이 양(+)인지.

## 7. 산출물 무결성
- [ ] `outputs/figures/` PNG 수와 `· 그림 저장: 성공/예상 (실패 0)` 일치(실패>0 이면 해당 그림 로그 확인).
- [ ] `outputs/figures_guide.md` 가 최신(접미사 `_qe_surge_*`/`_rate_cuts`/`_strong_from*`)으로 생성됐는지(`--guide`).
- [ ] 핵심 CSV 존재: `qe_surge_events.csv`·`rate_cut_events.csv`·`W_comparison.csv`·`rate_W_comparison.csv`·`channel_comparison.csv`·`strong_assets.csv`·`lp_scores_*`·`lp_ordering_scores_*`.

## 8. 민감도(권장)
- [ ] `--qe-surge --post 1.5` 와 `--post 3` 결과 순서가 크게 뒤집히지 않는지(사후 관측구간 민감도).
- [ ] `config.py` 의 `DRAWDOWN_THETA`(0.30)·`ONSET_*`·`STRONG_W_THRESHOLD`(0.80) 를 ±조정해 결론이 견고한지.
- [ ] (가능 시) AR 잔차 충격 대신 고빈도 FOMC 서프라이즈로 교체해 LP 재확인.

## 9. 재현성
- [ ] 두 번째 실행에서 `· raw_monthly.csv 재사용 (R×C)` 로 캐시가 재사용되는지(`--from-cache`).
- [ ] `--fresh` 로 캐시 삭제 후 재수집해도 동일 이벤트·순서가 나오는지.
- [ ] 커밋 전 `python -m pytest -q` 26개 통과 확인.

---

### 빠른 실행 순서
```bash
cp .env.example .env && $EDITOR .env       # FRED_API_KEY 입력
python -m pytest -q                          # 로직 회귀 확인(합성)
python main.py --all --fresh                 # 실데이터 전체 재생성
python main.py --guide                        # figures_guide.md 갱신
```
