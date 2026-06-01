"""tests/test_all.py — 통합 테스트 (합성). 실행: pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as C
import data as D
import analysis as A
import inference as INF
import viz as V
import main as M


@pytest.fixture(scope="module")
def synth():
    a, w, f, s = D.load_data(sample=True)
    prep = D.prepare(a)
    return {"assets": a, "walcl": w, "fedfunds": f, "loglev": prep["loglev"]}


# ---------- data ----------
def test_load_synth(synth):
    assert synth["assets"].shape[1] == 13 and synth["assets"].shape[0] >= 240


def test_events_detect(synth):
    ev = D.build_events(synth["walcl"], synth["fedfunds"])
    assert not ev["qe_surge"].empty
    assert list(ev["program"]["label"]) == ["QE1", "QE3", "QEinf"]


def test_program_events_post_years(synth):
    p3 = D.program_events(synth["walcl"], post_years=3.0)
    p15 = D.program_events(synth["walcl"], post_years=1.5)
    assert (p3["post_m"] == p3["span_m"] + 36).all()
    assert (p15["post_m"] == p15["span_m"] + 18).all()
    assert p3.loc[p3["dWALCL"].idxmax(), "label"] == "QEinf"


# ---------- analysis ----------
def test_timing_and_W(synth):
    ev = D.build_events(synth["walcl"], synth["fedfunds"])
    tt = A.extract_timing(synth["loglev"], ev["qe_surge"], pre=C.PROGRAM_PRE_MONTHS, save=False)
    W = A.kendalls_w(A._rank_matrix(tt))
    assert 0.0 <= W <= 1.0


def test_h1_qe_beats_mixed(synth):
    ev = D.build_events(synth["walcl"], synth["fedfunds"])
    tt_qe = A.extract_timing(synth["loglev"], ev["qe_surge"], pre=C.PROGRAM_PRE_MONTHS, save=False)
    tt_mixed = A.extract_timing(synth["loglev"], ev["mixed"], save=False)
    comp = A.compare_event_sets({"mixed(all)": tt_mixed, "qe_surge": tt_qe})
    W = comp.set_index("set")["kendalls_W"]
    assert W["qe_surge"] > W["mixed(all)"]


def test_post_tag():
    assert C.post_tag(3.0) == "3y" and C.post_tag(1.5) == "1p5y"


def _prog_mat(synth, y):
    tt = A.extract_timing(synth["loglev"], D.program_events(synth["walcl"], post_years=y),
                          pre=C.PROGRAM_PRE_MONTHS, save=False)
    return A._rank_matrix(tt)


def test_select_top_ordered_adaptive(synth):
    res = A.select_top_ordered(_prog_mat(synth, 3.0), w_thr=C.STRONG_W_THRESHOLD, n_min=C.STRONG_N_MIN)
    assert res["N"] >= C.STRONG_N_MIN
    assert 0.0 <= res["W"] <= 1.0
    if res["qualified"]:
        assert res["W"] >= C.STRONG_W_THRESHOLD          # 충족 시 임계 이상
    assert len(res["assets"]) == res["N"]


def test_pick_top_across_windows(synth):
    pick = A.pick_top_across_windows(_prog_mat(synth, 3.0), _prog_mat(synth, 1.5), w_thr=C.STRONG_W_THRESHOLD, n_min=C.STRONG_N_MIN)
    assert pick["source_window"] in ("3y", "1p5y")
    assert 0.0 <= pick["W_selected"] <= 1.0
    assert pick["N"] >= C.STRONG_N_MIN
    assert "qualified" in pick and "threshold" in pick


# ---------- inference ----------
def test_lp_runs(synth):
    ev = D.build_events(synth["walcl"], synth["fedfunds"])
    sh = D.build_shocks(synth["walcl"], synth["fedfunds"])
    lp = INF.run_lp_channels(synth["loglev"], sh, H=12)
    assert set(["curves", "scores_rate", "scores_qe"]).issubset(lp.keys())


# ---------- main orchestration ----------
def test_run_steps_no_figs():
    st = M.run([1, 2, 3, 4, 5], sample=True, make_figs=False)
    W = st["comp"].set_index("set")["kendalls_W"]
    assert "qe_surge_3y" in W.index and "qe_surge_1p5y" in W.index
    assert W["qe_surge_3y"] > W["mixed(all)"]


def test_qe_surge_runs():
    res = M.run_programs(sample=True, make_figs=False)
    assert res["pick_strong"]["source_window"] in ("3y", "1p5y")


def test_run_programs():
    res = M.run_programs(sample=True, make_figs=False)
    W = res["comp"].set_index("set")["kendalls_W"]
    assert W["qe_surge_3y"] > W["mixed(all)"]
    assert W["qe_surge_1p5y"] > W["mixed(all)"]
    assert res["pick_strong"]["source_window"] in ("3y", "1p5y")
    for f in ("qe_surge_events.csv", "ordering_qe_surge_3y.csv", "strong_assets.csv"):
        assert (C.RES_DIR / f).exists(), f"missing {f}"


def test_run_strong():
    pick = M.run_strong(sample=True, make_figs=False)
    assert pick["source_window"] in ("3y", "1p5y")
    assert pick["N"] >= C.STRONG_N_MIN
    # 선택 창의 W·N 이 두 창 중 우선순위(qualified→N→W)에 맞는지 최소 확인
    assert pick["W_selected"] in (pick["W_3y"], pick["W_1p5y"])


def test_full_pipeline_outputs():
    st = M.run([1, 2, 3, 4, 5, 6, 7], sample=True, make_figs=True)
    assert len(st["figs"]) >= 30
    for f in ("W_comparison.csv", "qe_surge_events.csv", "rate_cut_events.csv", "lp_scores_qe.csv",
              "shocks.csv", "var_irf_scores.csv", "hazard_onset.csv", "channel_comparison.csv",
              "rank_matrix_qe_surge_3y.csv", "rank_matrix_qe_surge_1p5y.csv",
              "strong_assets.csv", "lp_ordering_scores_qe.csv"):
        assert (C.RES_DIR / f).exists(), f"missing {f}"


def test_reset_workspace():
    M.reset_workspace(clear_cache=False, clear_outputs=True)
    assert C.RES_DIR.exists() and len(list(C.FIG_DIR.glob("*.png"))) == 0


# ---------- v5: 캐시 신뢰성 ----------
def test_raw_monthly_roundtrip(synth):
    ok = D._save_raw_monthly(synth["assets"], synth["walcl"], synth["fedfunds"])
    assert ok and C.RAW_MONTHLY.exists()
    loaded = D._load_raw_monthly()
    assert loaded is not None
    a, w, f = loaded
    assert set(C.ASSET_KEYS).issubset(a.columns) and len(a) >= C.CACHE_MIN_ROWS
    assert len(w) == len(a) and len(f) == len(a)


def test_raw_monthly_integrity_reject(tmp_path, monkeypatch):
    # 컬럼 부족·행 부족이면 None
    import pandas as pd
    bad = pd.DataFrame({"x": range(10)})
    bad.to_csv(C.RAW_MONTHLY)
    assert D._load_raw_monthly() is None
    C.RAW_MONTHLY.unlink(missing_ok=True)


# ---------- v5: 그림 생성 신뢰성 ----------
def test_render_isolates_failures():
    def boom():
        raise ValueError("forced")
    res = M._render([("ok", lambda: "outputs/x.png"), ("bad", boom), ("none", lambda: None)])
    assert res == ["outputs/x.png"]            # 실패·None 은 제외, 예외로 중단되지 않음


def test_plot_empty_guards():
    import pandas as pd
    assert V.small_multiples(None, pd.DataFrame()) is None
    assert V.event_study_average(None, pd.DataFrame()) is None
    assert V.magnitude_scatter(pd.DataFrame(), pd.DataFrame()) is None


def test_version():
    assert C.STRONG_W_THRESHOLD == 0.80


# ---------- v6: 이벤트 정교화 / 금리 인하 / peak 수정 / LP 수치화 / 그림설명 ----------
def test_event_selection_caps(synth):
    ev = D.build_events(synth["walcl"], synth["fedfunds"])
    assert len(ev["qe_surge"]) == 3                      # QE1/QE3/QEinf
    assert len(ev["rate_cuts"]) <= C.RATE_EVENT_TARGET_N
    assert "qe_events" not in ev                          # QE-only 제거
    assert ev["program"] is ev["qe_surge"]               # 구 키 별칭 유지


def test_rate_cut_detection(synth):
    rc = D.detect_rate_cut_events(synth["fedfunds"])
    assert len(rc) >= 1 and "cut_bp" in rc.columns
    assert (rc["cut_bp"] > 0).all()                      # 인하폭 양수


def test_peak_not_before_onset(synth):
    ev = D.build_events(synth["walcl"], synth["fedfunds"])
    for key in ("qe_surge", "mixed"):
        tt = A.extract_timing(synth["loglev"], ev[key], pre=C.PROGRAM_PRE_MONTHS, save=False)
        m = (~tt["onset_m"].isna()) & (~tt["peak_m"].isna())
        assert (tt.loc[m, "peak_m"] >= tt.loc[m, "onset_m"]).all(), f"{key}: peak<onset"


def test_lp_quantification(synth):
    sh = D.build_shocks(synth["walcl"], synth["fedfunds"])
    lp = INF.run_lp_channels(synth["loglev"], sh, H=12)
    tbl = A.lp_ordering_table(lp["scores_qe"], "qe")
    assert {"onset_h", "peak_h", "peak_beta", "lp_rank"}.issubset(tbl.columns)
    ev = D.build_events(synth["walcl"], synth["fedfunds"])
    order = A.mean_ordering(A._rank_matrix(A.extract_timing(synth["loglev"], ev["qe_surge"], pre=C.PROGRAM_PRE_MONTHS, save=False)))
    conc = A.lp_event_concordance(A.order_from_lp_named(lp["scores_qe"]), order)
    assert "rho" in conc and "table" in conc and "delta" in conc["table"].columns
    assert (C.RES_DIR / "lp_ordering_scores_qe.csv").exists()


def test_run_rate_cuts():
    res = M.run_rate_cuts(sample=True, make_figs=False)
    if res.get("figs") is not None and "W_rate" in res:
        W = res["comp"].set_index("set")["kendalls_W"]
        assert "rate_cuts" in W.index
        assert (C.RES_DIR / "rate_W_comparison.csv").exists()


def test_figures_guide():
    p = V.write_figures_guide()
    assert p and (C.OUT_DIR / "figures_guide.md").exists()
