"""
main.py — 통합 실행기 (v4: 파일 통합 + 적응형 N자산)
================================================================
  python main.py                 대화형 메뉴
  python main.py --all           전체(혼합 + 채널 분리 + 구간 ±3/1.5년 + topN)
  python main.py --qe-only       QE 주입 채널 단독
  python main.py --qe-surge      급격한 QE 구간 분석(±3/1.5년)
  python main.py --qe-surge --post 3      +3년 창만
  python main.py --topN          핵심 N자산(W≥0.7 최대 N, 두 창 승자)
  python main.py --from-cache    캐시로 분석
  python main.py --sample        합성 데이터
  python main.py --steps 5 6 7   특정 단계
  python main.py --clean / --fresh / --quiet
단계: 1 로드·2 변환·3 이벤트·충격·4 타이밍·5 순서·6 추론·7 시각화
================================================================
"""
from __future__ import annotations

import argparse
import shutil
import time

import numpy as np
import pandas as pd

import config as C
import data as D
import analysis as A
import inference as INF
import viz as V

STEP_NAMES = {1: "데이터 로드", 2: "변환", 3: "이벤트·충격", 4: "타이밍",
              5: "순서·Kendall's W", 6: "추론(LP/VAR/해저드)", 7: "시각화"}


def reset_workspace(clear_cache: bool = True, clear_outputs: bool = True) -> None:
    targets = []
    if clear_outputs:
        targets += [C.RES_DIR, C.FIG_DIR]
    if clear_cache:
        targets.append(C.CACHE_DIR)
    for d in targets:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    C.status(f"초기화: {[str(t.relative_to(C.ROOT)) for t in targets]}", "info")


# ----------------------------------------------------------------------
# 급격한 QE 구간(한 사후관측구간) 분석 + 동형 시각화
# ----------------------------------------------------------------------
def _qe_surge_window(loglev, walcl, years) -> dict:
    tag = C.post_tag(years)
    prog = D.program_events(walcl, post_years=years)
    tt = A.extract_timing(loglev, prog, tag=f"qe_surge_{tag}", pre=C.PROGRAM_PRE_MONTHS)
    mat = A._rank_matrix(tt); W = A.kendalls_w(mat)
    order = A.mean_ordering(mat)
    order.to_csv(C.RES_DIR / f"ordering_qe_surge_{tag}.csv", index=False)
    mat.to_csv(C.RES_DIR / f"rank_matrix_qe_surge_{tag}.csv")
    C.status(f"급격한 QE 구간 +{years:g}년 사후관측: W={W} | {len(prog)}구간", "calc")
    return {"tag": tag, "years": years, "prog": prog, "tt": tt, "mat": mat, "W": W, "order": order}


def _qe_surge_specs(loglev, ev, walcl, win) -> list:
    tag, years = win["tag"], win["years"]
    yr = f"+{years:g}년"
    prog, tt, order, mat, W = win["prog"], win["tt"], win["order"], win["mat"], win["W"]
    specs = [
        (f"ordering_rank_matrix_qe_surge_{tag}",
         lambda: V.rank_matrix(mat, W, title_prefix=f"[급격한 QE 구간 {yr}] 자산 반응 순위",
                               name=f"ordering_rank_matrix_qe_surge_{tag}")),
        (f"ordering_consistency_qe_surge_{tag}",
         lambda: V.consistency(order, suffix=f"_qe_surge_{tag}", title=f"자산별 전이 순서 일관성 — 급격한 QE 구간 {yr}")),
        (f"transmission_chain_qe_surge_{tag}",
         lambda: V.chain_diagram(order, name=f"transmission_chain_qe_surge_{tag}", title=f"평균 전이 사슬 — 급격한 QE 구간 {yr}")),
        (f"walcl_timeline_qe_surge_{tag}",
         lambda: V.walcl_timeline(walcl, pd.DataFrame(columns=["event_date", "dWALCL"]),
                                  ev.get("rate_cuts", pd.DataFrame()), tt,
                                  suffix=f"_qe_surge_{tag}", programs=prog)),
        (f"qe_surge_small_multiples_{tag}",
         lambda: V.small_multiples(loglev, prog, name=f"qe_surge_small_multiples_{tag}")),
        (f"qe_surge_study_average_{tag}",
         lambda: V.event_study_average(loglev, prog, name=f"qe_surge_study_average_{tag}")),
        (f"qe_surge_magnitude_scatter_{tag}",
         lambda: V.magnitude_scatter(prog, tt, name=f"qe_surge_magnitude_scatter_{tag}")),
        (f"qe_surge_magnitude_bubble_{tag}",
         lambda: V.magnitude_bubble(prog, tt, name=f"qe_surge_magnitude_bubble_{tag}")),
    ]
    for _, p in prog.iterrows():
        nm = f"timing_profile_{p['label']}_{tag}"

        def _mk(ed=p["event_date"], pm=int(p["post_m"]), lbl=p["label"], nm=nm):
            wr = D.window_response(loglev, ed, pre=C.PROGRAM_PRE_MONTHS, post=pm)
            return V.timing_profile(A.timing_for_window(wr),
                                    f"급격한 QE 구간 {lbl} ({pd.Timestamp(ed):%Y-%m}{yr})", name=nm)
        specs.append((nm, _mk))
    return specs


def _qe_surge_hypotheses(win, tt_mixed) -> pd.DataFrame:
    from scipy.stats import spearmanr
    order, tt, prog, W = win["order"], win["tt"], win["prog"], win["W"]
    role_idx = {r: i for i, r in enumerate(C.ROLE_ORDER)}
    o2 = order.dropna(subset=["mean_rank"]).copy(); o2["ri"] = o2["role"].map(role_idx)
    h2 = spearmanr(o2["ri"], o2["mean_rank"]).statistic if len(o2) >= 3 else np.nan
    mr = tt.merge(prog[["event_date", "magnitude_T"]], on="event_date", how="left")
    msk = (~mr["magnitude_T"].isna()) & (~mr["peak_resp"].isna())
    h3 = spearmanr(mr["magnitude_T"][msk], mr["peak_resp"].abs()[msk]).statistic if msk.sum() >= 3 else np.nan
    W_mix = A.kendalls_w(A._rank_matrix(tt_mixed))
    hyp = pd.DataFrame([
        {"hypothesis": "H1 (W_qe_surge > W_mixed)", "stat": f"{W} vs {W_mix}", "pass": bool(W > W_mix)},
        {"hypothesis": "H2 (role 사슬 단조)", "stat": f"rho={round(float(h2),3)}", "pass": bool(h2 > 0)},
        {"hypothesis": "H3 (규모↑→반응↑)", "stat": f"amp_rho={round(float(h3),3)}", "pass": bool(h3 > 0)},
    ])
    hyp.to_csv(C.RES_DIR / f"qe_surge_hypotheses_{win['tag']}.csv", index=False)
    return hyp


# ----------------------------------------------------------------------
# 순서성 강한 자산 (두 창 승자, W≥0.80) + 동형 시각화
# ----------------------------------------------------------------------
def _strong(windows: dict) -> dict:
    if "3y" not in windows or "1p5y" not in windows:
        return {}
    pick = A.pick_top_across_windows(windows["3y"]["mat"], windows["1p5y"]["mat"],
                                     w_thr=C.STRONG_W_THRESHOLD, n_min=C.STRONG_N_MIN)
    A.save_strong_assets(pick)
    src = pick["source_window"]; yrs = "3년" if src == "3y" else "1.5년"
    flag = f"≥{C.STRONG_W_THRESHOLD:.2f}" if pick["qualified"] else "미충족"
    C.status(f"순서성 강한 자산 출처: {yrs} 창 (N={pick['N']}, W={pick['W_selected']} {flag} | "
             f"3y:N{pick['N_3y']}/W{pick['W_3y']}, 1p5y:N{pick['N_1p5y']}/W{pick['W_1p5y']}) "
             f"→ {[C.ASSET_LABELS[a] for a in pick['assets']]}", "calc")
    return pick


def _strong_specs(pick: dict) -> list:
    if not pick:
        return []
    src = pick["source_window"]; yrs = "3년" if src == "3y" else "1.5년"
    N, W = pick["N"], pick["W_selected"]
    flag = "" if pick["qualified"] else " · 임계 미충족"
    title = f"[급격한 QE 구간·출처 {yrs}창] 순서성 강한 {N}자산 전이 순위{flag}"
    return [
        (f"ordering_rank_matrix_strong_from{src}",
         lambda: V.rank_matrix(pick["sub_mat"], W, title_prefix=title,
                               name=f"ordering_rank_matrix_strong_from{src}")),
        (f"ordering_consistency_strong_from{src}",
         lambda: V.consistency(pick["order"], name="ordering_consistency",
                               suffix=f"_strong_from{src}", title=f"순서성 강한 {N}자산 일관성 — 출처 {yrs}창")),
        (f"transmission_chain_strong_from{src}",
         lambda: V.chain_diagram(pick["order"], name=f"transmission_chain_strong_from{src}",
                                 title=f"순서성 강한 {N}자산 전이 사슬 — 출처 {yrs}창 (W={W})")),
    ]


def _render(specs: list) -> list:
    """그림 spec[(name, thunk)] 를 예외 격리 렌더 + 항목별/집계 로그. 성공 경로 리스트 반환."""
    ok, fail, n = [], 0, len(specs)
    for i, (name, thunk) in enumerate(specs, 1):
        try:
            r = thunk()
        except Exception as e:  # noqa
            r = None; C.warn(f"{name} 생성 실패: {type(e).__name__}: {e}")
        if r:
            ok.append(r); C.fig_line(i, n, name)
        else:
            fail += 1
    C.fig_summary(len(ok), n, fail)
    return ok


# ----------------------------------------------------------------------
# 전체 실행
# ----------------------------------------------------------------------
def run(steps=(1, 2, 3, 4, 5, 6, 7), use_cache=True, sample=False, make_figs=True) -> dict:
    st: dict = {}
    steps = sorted(set(steps))
    need = lambda s: any(x >= s for x in steps)
    t0 = time.time()

    if need(1):
        with C.timed("STEP 1"):
            C.step(1, 7, STEP_NAMES[1])
            a, w, f, synth = D.load_data(use_cache, sample)
            st.update(assets=a, walcl=w, fedfunds=f, synthetic=synth)
            C.status(f"자산 {a.shape[1]}개 · {a.shape[0]}개월 · 합성={synth}", "ok")
    if need(2):
        with C.timed("STEP 2"):
            C.step(2, 7, STEP_NAMES[2])
            st["prep"] = D.prepare(st["assets"]); st["loglev"] = st["prep"]["loglev"]
            C.status("실시간 정렬 → STL(NSA) → 중심화 누적 로그반응", "ok")
    if need(3):
        with C.timed("STEP 3"):
            C.step(3, 7, STEP_NAMES[3])
            st["ev"] = D.build_events(st["walcl"], st["fedfunds"])
            st["shocks"] = D.build_shocks(st["walcl"], st["fedfunds"])
            D.save_events(st["ev"]); D.save_shocks(st["shocks"])
            su = st["ev"]["qe_surge"]
            tags = "  ".join(f"{r['label']} +${r['dWALCL']:.1f}T" for _, r in su.iterrows())
            C.status(f"급격한 QE 구간: {len(su)}개 | {tags}", "find")
            C.status(f"금리 인하 이벤트: {len(st['ev']['rate_cuts'])}건 (큰 인하순)", "find")
    if need(4):
        with C.timed("STEP 4"):
            C.step(4, 7, STEP_NAMES[4])
            st["tt_rate"] = A.extract_timing(st["loglev"], st["ev"]["rate_cuts"], tag="rate_cuts") \
                if len(st["ev"]["rate_cuts"]) else st["ev"]["rate_cuts"]
            st["tt_mixed"] = A.extract_timing(st["loglev"], st["ev"]["mixed"], tag="mixed")
            st["windows"] = {C.post_tag(y): _qe_surge_window(st["loglev"], st["walcl"], y)
                             for y in C.QE_SURGE_HORIZON_YEARS_SET}
            C.status("onset·half·peak 추출 완료 (급격한 QE 구간 2창 + 금리 인하 이벤트)", "ok")
    if need(5):
        with C.timed("STEP 5"):
            C.step(5, 7, STEP_NAMES[5])
            st["mat_mixed"] = A._rank_matrix(st["tt_mixed"]); st["W_mixed"] = A.kendalls_w(st["mat_mixed"])
            st["order_mixed"] = A.mean_ordering(st["mat_mixed"])
            # QE 경로 기본 = 급격한 QE 구간 +3년 장
            st["order_qe3y"] = st["windows"]["3y"]["order"]; st["mat_qe3y"] = st["windows"]["3y"]["mat"]
            st["W_qe3y"] = st["windows"]["3y"]["W"]
            sets = {"mixed(all)": st["tt_mixed"]}
            if isinstance(st["tt_rate"], pd.DataFrame) and len(st["tt_rate"]):
                sets["rate_cuts"] = st["tt_rate"]
                st["mat_rate"] = A._rank_matrix(st["tt_rate"]); st["order_rate"] = A.mean_ordering(st["mat_rate"])
                st["W_rate"] = A.kendalls_w(st["mat_rate"])
            for tag, win in st["windows"].items():
                sets[f"qe_surge_{tag}"] = win["tt"]
            st["comp"] = A.compare_event_sets(sets); st["comp"].to_csv(C.RES_DIR / "W_comparison.csv", index=False)
            st["pick_strong"] = _strong(st["windows"])
            wline = "  ".join(f"{r['set']} {r['kendalls_W']}" for _, r in st["comp"].iterrows())
            C.status(f"Kendall's W | {wline}", "calc")
    if need(6):
        with C.timed("STEP 6"):
            C.step(6, 7, STEP_NAMES[6])
            st["lp"] = INF.run_lp_channels(st["loglev"], st["shocks"]); INF.save_lp(st["lp"])
            st["var_qe"] = INF.run_var_irf(st["loglev"], st["shocks"], "qe_shock"); INF.save_var(st["var_qe"])
            # 해저드는 급격한 QE 구간(3년 장) 기준
            su3 = st["windows"]["3y"]["prog"][["event_date", "magnitude_T"]]
            st["hazard_qe"] = INF.run_hazard(st["windows"]["3y"]["tt"], su3, channel=None)
            INF.save_hazard(st["hazard_qe"])
            st["ord_lp_qe"] = A.order_from_lp_named(st["lp"]["scores_qe"])
            st["ord_lp_rate"] = A.order_from_lp_named(st["lp"]["scores_rate"])
            # LP 순서 수치화 (표 + 일치도): QE 채널↔3년 장, rate 채널↔금리 인하
            st["lp_tbl_qe"] = A.lp_ordering_table(st["lp"]["scores_qe"], "qe")
            st["lp_tbl_rate"] = A.lp_ordering_table(st["lp"]["scores_rate"], "rate")
            st["lp_conc_qe"] = A.lp_event_concordance(st["ord_lp_qe"], st["order_qe3y"])
            st["lp_conc_rate"] = A.lp_event_concordance(
                st["ord_lp_rate"], st.get("order_rate", st["order_mixed"]))
            # 채널 비교 = QE 3년 장 vs 금리 인하 이벤트 (이벤트-스터디 순서)
            st["chan"] = A.channel_compare_orders(st["order_qe3y"], st.get("order_rate", st["order_mixed"]))
            st["chan"].to_csv(C.RES_DIR / "channel_comparison.csv", index=False)
            st["methods"] = A.compare_methods(st["lp"]["scores_qe"], st["order_qe3y"], st["var_qe"])
            A.save_ordering(st["mat_mixed"], st["mat_qe3y"], st["comp"], st["order_qe3y"],
                            st["ord_lp_rate"], st["ord_lp_qe"], st["chan"])
            A.magnitude_response(st["windows"]["3y"]["tt"], st["windows"]["3y"]["prog"])
            C.status("LP(채널분리)·VAR-IRF·해저드 완료", "ok")
    if 7 in steps and make_figs:
        with C.timed("STEP 7"):
            C.step(7, 7, STEP_NAMES[7])
            st["figs"] = _make_all_figs(st)
            C.status(f"그림 {len(st['figs'])}개 저장 → outputs/figures/", "fig")

    n_fig, n_csv = C.count_outputs()
    C.footer(n_fig, n_csv, time.time() - t0)
    return st


def _make_all_figs(st) -> list:
    w, ev = st["walcl"], st["ev"]
    surge = ev["qe_surge"]
    no_qe_lines = pd.DataFrame(columns=["event_date", "dWALCL"])   # QE 점 이벤트 없음(구간 음영으로 표시)
    specs = [
        ("walcl_timeline", lambda: V.walcl_timeline(w, no_qe_lines, ev["rate_cuts"],
                                                    st["windows"]["3y"]["tt"], programs=surge)),
        ("event_small_multiples", lambda: V.small_multiples(st["loglev"], ev["mixed"])),
        ("ordering_rank_matrix", lambda: V.rank_matrix(st["mat_mixed"], st["W_mixed"], name="ordering_rank_matrix")),
        ("ordering_consistency", lambda: V.consistency(st["order_mixed"], name="ordering_consistency")),
        ("ordering_lead_lag", lambda: V.lead_lag(st["mat_qe3y"])),
        ("transmission_chain", lambda: V.chain_diagram(st["order_qe3y"])),
        ("qe_W_comparison", lambda: V.w_comparison(st["comp"])),
    ]
    # 금리 인하 이벤트 분석 그림 (사슬 포함)
    if isinstance(st.get("tt_rate"), pd.DataFrame) and len(st["tt_rate"]):
        specs += [
            ("ordering_rank_matrix_rate_cuts",
             lambda: V.rank_matrix(st["mat_rate"], st["W_rate"],
                                   title_prefix="[금리 인하 이벤트] 자산 반응 순위",
                                   name="ordering_rank_matrix_rate_cuts")),
            ("ordering_consistency_rate_cuts",
             lambda: V.consistency(st["order_rate"], suffix="_rate_cuts",
                                   title="자산별 전이 순서 일관성 — 금리 인하 이벤트")),
            ("transmission_chain_rate_cuts",
             lambda: V.chain_diagram(st["order_rate"], name="transmission_chain_rate_cuts",
                                     title="평균 전이 사슬 — 금리 인하 이벤트")),
            ("walcl_timeline_rate_cuts",
             lambda: V.walcl_timeline(w, no_qe_lines, ev["rate_cuts"], st["tt_rate"], suffix="_rate_cuts")),
            ("rate_magnitude_scatter",
             lambda: V.magnitude_scatter(ev["rate_cuts"], st["tt_rate"], name="rate_magnitude_scatter")),
        ]
    if "lp" in st:
        specs += [
            ("lp_irf_qe", lambda: V.lp_panels(st["lp"]["curves"], "qe", suffix="_qe")),
            ("lp_irf_rate", lambda: V.lp_panels(st["lp"]["curves"], "rate", suffix="_rate")),
            ("lp_ordering_qe", lambda: V.lp_ordering(st["lp"]["scores_qe"], "급격한 QE 구간", suffix="_qe")),
            ("lp_ordering_rate", lambda: V.lp_ordering(st["lp"]["scores_rate"], "금리 인하 이벤트", suffix="_rate")),
            ("lp_ordering_table_qe", lambda: V.lp_ordering_table(st["lp_tbl_qe"], "급격한 QE 구간", suffix="_qe")),
            ("lp_ordering_table_rate", lambda: V.lp_ordering_table(st["lp_tbl_rate"], "금리 인하 이벤트", suffix="_rate")),
            ("lp_concordance_qe", lambda: V.lp_concordance(st["lp_conc_qe"], "급격한 QE 구간 +3년", suffix="_qe")),
            ("lp_concordance_rate", lambda: V.lp_concordance(st["lp_conc_rate"], "금리 인하 이벤트", suffix="_rate")),
            ("channel_compare", lambda: V.channel_compare(st["chan"])),
            ("method_compare", lambda: V.method_compare(st["methods"])),
        ]
    for win in st.get("windows", {}).values():
        specs += _qe_surge_specs(st["loglev"], ev, w, win)
    specs += _strong_specs(st.get("pick_strong", {}))
    figs = _render(specs)
    V.write_figures_guide()
    return figs


# ----------------------------------------------------------------------
# QE 증가 이벤트 분석 (구 qe_pipeline 흡수)
# ----------------------------------------------------------------------
def run_qe_events(*a, **k):
    """[삭제됨] QE 증가 이벤트(점 단위) 분석은 더 이상 제공하지 않는다.
    QE 경로는 '급격한 QE 구간 분석'(run_programs / --qe-surge)으로 다룬다."""
    C.warn("QE 증가 이벤트(점) 분석은 폐지되었습니다. '--qe-surge'(급격한 QE 구간)를 사용하세요.")
    return run_programs(*a, **k)


run_qe_only = run_qe_events   # 하위호환 별칭


# ----------------------------------------------------------------------
# 금리 인하 이벤트 분석
# ----------------------------------------------------------------------
def run_rate_cuts(use_cache=True, sample=False, make_figs=True) -> dict:
    from scipy.stats import spearmanr
    t0 = time.time()
    C.step(1, 3, "데이터·이벤트·충격")
    a, w, f, synth = D.load_data(use_cache, sample)
    loglev = D.prepare(a)["loglev"]
    ev = D.build_events(w, f); D.save_events(ev)
    sh = D.build_shocks(w, f); D.save_shocks(sh)
    rc = ev["rate_cuts"]
    if rc is None or len(rc) == 0:
        C.warn("금리 인하 이벤트가 탐지되지 않았습니다(데이터 확인).")
        C.footer(*C.count_outputs(), time.time() - t0)
        return {"synthetic": synth, "figs": []}

    C.step(2, 3, "타이밍·순서·LP(금리 채널)")
    for _, p in rc.iterrows():
        C.status(f"{p['label']}: -{p['cut_bp']:.0f}bp ({pd.Timestamp(p['event_date']):%Y-%m}→{pd.Timestamp(p['end_date']):%Y-%m})", "info")
    tt_rate = A.extract_timing(loglev, rc, tag="rate_cuts")
    tt_mixed = A.extract_timing(loglev, ev["mixed"], tag="mixed")
    mat_rate = A._rank_matrix(tt_rate); mat_mixed = A._rank_matrix(tt_mixed)
    W_rate = A.kendalls_w(mat_rate); W_mixed = A.kendalls_w(mat_mixed)
    comp = A.compare_event_sets({"mixed(all)": tt_mixed, "rate_cuts": tt_rate})
    comp.to_csv(C.RES_DIR / "rate_W_comparison.csv", index=False)
    order_rate = A.mean_ordering(mat_rate)
    magdf = A.magnitude_response(tt_rate, rc.rename(columns={"cut_bp": "magnitude_T"})
                                 if "magnitude_T" not in rc else rc)
    lp = INF.run_lp_channels(loglev, sh); INF.save_lp(lp)
    ord_lp_rate = A.order_from_lp_named(lp["scores_rate"])
    lp_tbl = A.lp_ordering_table(lp["scores_rate"], "rate")
    lp_conc = A.lp_event_concordance(ord_lp_rate, order_rate)

    role_idx = {r: i for i, r in enumerate(C.ROLE_ORDER)}
    o2 = order_rate.dropna(subset=["mean_rank"]).copy(); o2["ri"] = o2["role"].map(role_idx)
    h2 = spearmanr(o2["ri"], o2["mean_rank"]).statistic if len(o2) >= 3 else np.nan
    amp_rho = float(magdf.iloc[0]["rho"]) if not pd.isna(magdf.iloc[0]["rho"]) else np.nan
    hyp = pd.DataFrame([
        {"hypothesis": "H1 (W_rate > W_mixed)", "stat": f"{W_rate} vs {W_mixed}", "pass": bool(W_rate > W_mixed)},
        {"hypothesis": "H2 (role 사슬 단조)", "stat": f"rho={round(float(h2),3)}", "pass": bool(h2 > 0)},
        {"hypothesis": "H3 (인하폭↑→반응↑)", "stat": f"amp_rho={amp_rho}", "pass": bool(amp_rho > 0)},
    ])
    hyp.to_csv(C.RES_DIR / "rate_hypotheses.csv", index=False)
    C.status(f"Kendall's W | rate_cuts {W_rate} vs mixed {W_mixed} | LP↔이벤트 ρ={lp_conc['rho']}", "calc")

    figs = []
    if make_figs:
        C.step(3, 3, "시각화")
        specs = [
            ("ordering_rank_matrix_rate_cuts",
             lambda: V.rank_matrix(mat_rate, W_rate, title_prefix="[금리 인하 이벤트] 자산 반응 순위",
                                   name="ordering_rank_matrix_rate_cuts")),
            ("ordering_consistency_rate_cuts",
             lambda: V.consistency(order_rate, suffix="_rate_cuts", title="자산별 전이 순서 일관성 — 금리 인하 이벤트")),
            ("transmission_chain_rate_cuts",
             lambda: V.chain_diagram(order_rate, name="transmission_chain_rate_cuts",
                                     title="평균 전이 사슬 — 금리 인하 이벤트")),
            ("rate_W_comparison", lambda: V.w_comparison(comp)),
            ("rate_magnitude_scatter", lambda: V.magnitude_scatter(rc, tt_rate, name="rate_magnitude_scatter")),
            ("lp_irf_rate", lambda: V.lp_panels(lp["curves"], channel="rate", suffix="_rate")),
            ("lp_ordering_rate", lambda: V.lp_ordering(lp["scores_rate"], "금리 인하 이벤트", suffix="_rate")),
            ("lp_ordering_table_rate", lambda: V.lp_ordering_table(lp_tbl, "금리 인하 이벤트", suffix="_rate")),
            ("lp_concordance_rate", lambda: V.lp_concordance(lp_conc, "금리 인하 이벤트", suffix="_rate")),
            ("walcl_timeline_rate_cuts",
             lambda: V.walcl_timeline(w, pd.DataFrame(columns=["event_date", "dWALCL"]), rc, tt_rate, suffix="_rate_cuts")),
        ]
        figs = _render(specs)
        V.write_figures_guide()
    print("\n=== 금리 인하 이벤트 분석 ==="); print(comp.to_string(index=False))
    print("\n가설:"); print(hyp.to_string(index=False))
    n_fig, n_csv = C.count_outputs(); C.footer(n_fig, n_csv, time.time() - t0)
    return {"comp": comp, "hyp": hyp, "order_rate": order_rate, "W_rate": W_rate,
            "W_mixed": W_mixed, "lp": lp, "lp_concordance": lp_conc, "synthetic": synth, "figs": figs}


# ----------------------------------------------------------------------
# 급격한 QE 구간 분석
# ----------------------------------------------------------------------
def run_qe_surge(use_cache=True, sample=False, make_figs=True, post_years_set=None) -> dict:
    post_years_set = post_years_set or C.PROGRAM_POST_YEARS_SET
    t0 = time.time()
    C.step(1, 4, "데이터 로드")
    a, w, f, synth = D.load_data(use_cache, sample)
    loglev = D.prepare(a)["loglev"]
    ev = D.build_events(w, f); D.save_events(ev)
    C.status(f"자산 {a.shape[1]}개 · 합성={synth}", "ok")

    C.step(2, 4, "급격한 QE 구간 정의")
    base = D.program_events(w, post_years=post_years_set[0])
    for _, p in base.iterrows():
        C.status(f"{p['label']}: {pd.Timestamp(p['event_date']):%Y-%m} → {pd.Timestamp(p['end_date']):%Y-%m} "
                 f"(span {p['span_m']}M, +${p['dWALCL']:.1f}T)", "info")

    C.step(3, 4, "창별 분석(타이밍·순서·W·가설)")
    tt_mixed = A.extract_timing(loglev, ev["mixed"], tag="mixed", save=False)
    windows, specs = {}, []
    for yrs in post_years_set:
        with C.timed(f"급격한 QE 구간 +{yrs:g}년"):
            win = _qe_surge_window(loglev, w, yrs)
            win["hyp"] = _qe_surge_hypotheses(win, tt_mixed)
            windows[win["tag"]] = win
            if make_figs:
                specs += _qe_surge_specs(loglev, ev, w, win)
            print(win["hyp"].to_string(index=False))

    sets = {"mixed(all)": tt_mixed}
    for tag, win in windows.items():
        sets[f"qe_surge_{tag}"] = win["tt"]
    comp = A.compare_event_sets(sets); comp.to_csv(C.RES_DIR / "W_comparison.csv", index=False)

    C.step(4, 4, "순서성 강한 자산(두 창 승자)")
    pick = _strong(windows) if len(windows) >= 2 else {}
    if pick and make_figs:
        specs += _strong_specs(pick)
    figs = _render(specs) if make_figs else []
    if make_figs:
        V.write_figures_guide()

    print("\n=== W 비교 ==="); print(comp.to_string(index=False))
    if pick:
        yrs = "3년" if pick["source_window"] == "3y" else "1.5년"
        print(f"\n=== 순서성 강한 {pick['N']}자산 (출처 {yrs}창, W={pick['W_selected']}, 충족={pick['qualified']}) ===")
        print(pick["order"][["label", "role", "mean_rank"]].to_string(index=False))
    n_fig, n_csv = C.count_outputs(); C.footer(n_fig, n_csv, time.time() - t0)
    return {"comp": comp, "windows": windows, "pick_strong": pick, "synthetic": synth, "figs": figs}


run_programs = run_qe_surge   # 하위호환 별칭


def run_strong(use_cache=True, sample=False, make_figs=True) -> dict:
    t0 = time.time()
    C.step(1, 2, "데이터·구간 준비")
    a, w, f, synth = D.load_data(use_cache, sample)
    loglev = D.prepare(a)["loglev"]
    windows = {C.post_tag(y): _qe_surge_window(loglev, w, y) for y in C.QE_SURGE_HORIZON_YEARS_SET}
    C.step(2, 2, "순서성 강한 자산 선별·시각화")
    pick = _strong(windows)
    if make_figs:
        _render(_strong_specs(pick)); V.write_figures_guide()
    yrs = "3년" if pick["source_window"] == "3y" else "1.5년"
    print(f"\n=== 순서성 강한 {pick['N']}자산 (출처 {yrs}창, W={pick['W_selected']}, 충족={pick['qualified']} | "
          f"3y:N{pick['N_3y']}/W{pick['W_3y']}, 1p5y:N{pick['N_1p5y']}/W{pick['W_1p5y']}) ===")
    print(pick["order"][["label", "role", "mean_rank"]].to_string(index=False))
    n_fig, n_csv = C.count_outputs(); C.footer(n_fig, n_csv, time.time() - t0)
    return pick


run_topN = run_strong   # 하위호환 별칭


# ----------------------------------------------------------------------
# 메뉴 / CLI
# ----------------------------------------------------------------------
MENU = """
================ cantillon-sequencer ================
 1) 전체 분석                                 [--all]
 2) 급격한 QE 구간 분석(±3/1.5년)             [--qe-surge]
 3) 금리 인하 이벤트 분석                      [--rate-cuts]
 4) 순서성 강한 자산 선별(W≥0.8 최대 N)       [--strong]
 5) 합성 데이터로 데모                        [--sample]
 6) 그림 설명서 생성                          [--guide]
 7) 초기화만(cache·결과 삭제)               [--clean]
 0) 종료
=====================================================
선택: """


def menu():
    while True:
        try:
            ch = input(MENU).strip()
        except (EOFError, KeyboardInterrupt):
            print("  종료합니다."); return
        if ch == "0":
            print("  종료합니다."); return
        elif ch == "1": C.banner("메뉴 1) 전체", False); run()
        elif ch == "2": C.banner("메뉴 2) 급격한 QE 구간", False); run_programs()
        elif ch == "3": C.banner("메뉴 3) 금리 인하 이벤트", False); run_rate_cuts()
        elif ch == "4": C.banner("메뉴 4) 순서성 강한 자산", False); run_strong()
        elif ch == "5": run(sample=True)
        elif ch == "6": print("  ", V.write_figures_guide())
        elif ch == "7": reset_workspace()
        else: print("  ⚠️  잘못된 선택입니다.")
        print()


def build_parser():
    p = argparse.ArgumentParser(description="cantillon-sequencer")
    p.add_argument("--all", action="store_true")
    p.add_argument("--qe-surge", "--programs", dest="qe_surge", action="store_true",
                   help="급격한 QE 구간 분석(±3/1.5년)")
    p.add_argument("--rate-cuts", dest="rate_cuts", action="store_true",
                   help="금리 인하 이벤트 분석")
    p.add_argument("--strong", "--topN", "--top5", dest="strong", action="store_true",
                   help="순서성 강한 자산(W≥0.8 최대 N)")
    p.add_argument("--guide", action="store_true", help="그림 설명서(figures_guide.md)만 생성")
    p.add_argument("--post", type=float, choices=[1.5, 3.0], help="급격한 QE 구간 사후 관측구간(년) 단일 선택")
    p.add_argument("--from-cache", action="store_true")
    p.add_argument("--sample", action="store_true")
    p.add_argument("--steps", nargs="+", type=int)
    p.add_argument("--clean", action="store_true")
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--quiet", action="store_true", help="상세 로그 끄기")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    C.set_verbose(not args.quiet)
    use_cache = args.from_cache or not args.fresh
    any_mode = (args.all or args.rate_cuts or args.qe_surge or
                args.strong or args.steps or args.guide)
    if args.clean and not any_mode:
        reset_workspace(); return
    if args.fresh:
        reset_workspace(); use_cache = False

    if args.guide:
        print("  ", V.write_figures_guide()); return
    if args.rate_cuts:
        C.banner("--rate-cuts", args.sample); run_rate_cuts(use_cache=use_cache, sample=args.sample)
    elif args.strong:
        C.banner("--strong", args.sample); run_strong(use_cache=use_cache, sample=args.sample)
    elif args.qe_surge:
        pys = [args.post] if args.post else None
        C.banner(f"--qe-surge{(' --post ' + str(args.post)) if args.post else ''}", args.sample)
        run_programs(use_cache=use_cache, sample=args.sample, post_years_set=pys)
    elif args.steps:
        C.banner(f"--steps {args.steps}", args.sample); run(args.steps, use_cache=use_cache, sample=args.sample)
    elif args.all:
        C.banner("--all", args.sample); run(use_cache=use_cache, sample=args.sample)
    else:
        menu()


if __name__ == "__main__":
    main()
