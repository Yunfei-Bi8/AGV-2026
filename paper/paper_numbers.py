"""Every number quoted in Sections IV and V of the paper (arXiv v2 and the CDC version), recomputed from the stored
results; nothing is simulated.

    python3 paper/paper_numbers.py

Each line gives the quantity, the value and the source (result file and key). Percentages are rounded as in the paper.
"""
import glob
import json
import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, 'experiments'), os.path.join(ROOT, 'experiments', 'robust'), os.path.join(ROOT, 'paper')):
    if p not in sys.path:
        sys.path.insert(0, p)
warnings.filterwarnings('ignore')
import make_figures_robust as mfr          # noqa: E402
from make_paper_tables import r5_ratios   # noqa: E402

RES = os.path.join(ROOT, 'results')
RESR = os.path.join(RES, 'robust')
load = lambda *p: json.load(open(os.path.join(*p)))


def out(section, what, value, source):
    print(f'{section:5s} {what:62s} {value:28s} {source}')


def pct(x):
    return f'{100 * x:.0f}%'


# ---------------------------------------------------------------- V-A, kinematic model (E2, E3, E5)
runs = load(RES, 'e3_runs.json')
out('V-A', 'closed-loop runs; all bounds hold', f"{len(runs)}; {all(r['all_ok'] for r in runs)}", 'e3_runs.json: all_ok')
for net in ('nominal', 'slow1', 'slow3'):
    out('V-A', f'largest delay, network {net}', f"{max(r['h'] for r in runs if r['net'] == net):.2f} s", 'e3_runs.json: h')
out('V-A', 'largest rise of W_i relative to max_j W_j(0)', f"{max(r['W_max_rise_rel'] for r in runs):.1e}",
    'e3_runs.json: W_max_rise_rel')
out('V-A', 'largest ratio to the bounds of Theorems 1 and 2',
    f"{max(max(r['eps_ratio'], r['thm2_eps_ratio'], r['thm2_e_ratio'], r['thm2_head_ratio']) for r in runs):.3f}",
    'e3_runs.json: eps_ratio, thm2_*_ratio')
out('V-A', 'largest ratio to the bounds of Theorem 2, t >= 1 s',
    f"{max(max(r['thm2_eps_ratio_t1'], r['thm2_e_ratio_t1'], r['thm2_head_ratio_t1']) for r in runs):.3f}",
    'e3_runs.json: thm2_*_ratio_t1')
out('V-A', 'a priori curvature bound kappa-bar', f"{min(r['curv_apriori'] for r in runs):.0f}-{max(r['curv_apriori'] for r in runs):.0f} 1/m",
    'e3_runs.json: curv_apriori')
out('V-A', 'largest observed curvature of the shifted references', f"{max(r['curv_run'] for r in runs):.1f} 1/m", 'e3_runs.json: curv_run')
out('V-A', 'rate eta_1 of Theorem 2 (ell = 1 m)', f"{min(r['eta1'] for r in runs):.1e}-{max(r['eta1'] for r in runs):.1e} 1/s",
    'e3_runs.json: eta1')
out('V-A', 'measured decay rate of max_i |eps_i|, all runs', f"{min(r['rate_eps'] for r in runs):.2f}-{max(r['rate_eps'] for r in runs):.2f} 1/s",
    'e3_runs.json: rate_eps')
rates = load(RES, 'e2_lemma2.json')['rates']
out('V-A', 'constant delay: largest |measured rate / rate of Lemma 3 - 1|',
    f"{max(abs(r['rate_measured'] / r['rate_lemma2'] - 1) for r in rates):.2%}", 'e2_lemma2.json: rates')
static = [r for r in load(RES, 'e5_formation.json')['static'] if r['slip']]
for net, lab in (('nominal', '60 ms'), ('slow3', '3 s')):
    d = [r['max_dev_mm'] for r in static if r['net'] == net]
    out('V-A', f'speed loss 0.03 m/s: largest deviation from (8), {lab}', f'{max(d):.2g} mm ({len(d)} runs)', 'e5_formation.json: max_dev_mm')
for net, lab in (('nominal', '60 ms'), ('slow3', '3 s')):
    comp = [abs(r['disagreement_ratio'] / r['predicted_factor'] - 1) for r in static
            if r['graph'] == 'complete3' and r['net'] == net and r.get('disagreement_ratio') is not None]
    out('V-A', f'complete graph, {lab}: |measured ratio / factor - 1|', f'<= {max(comp):.1e}', 'e5_formation.json')
path = [(r['disagreement_ratio'], r['predicted_factor']) for r in static if r['graph'] == 'path4' and r.get('disagreement_ratio') is not None]
out('V-A', 'path of four: measured ratio below the factor in all runs', f'{all(m <= f for m, f in path)} ({len(path)} runs)', 'e5_formation.json')

# ---------------------------------------------------------------- V-B, high-fidelity model
S = load(RESR, 'summary.json')
_, _, r2 = mfr.rows('r2_realistic')
for typ in ('uncoupled', 'direct', 'cascade', 'sigleak'):
    v = [r['pre_form'] for r in r2 if r['scen'] == 'warehouse' and r['case'] == 'nominal' and r['type'] == typ]
    out('V-B', f'nominal formation RMS, {typ} (mean, warehouse)', f'{1e3 * np.mean(v):.1f} mm', 'r2_realistic.json')
r4 = S['r4']
f4 = lambda f, t, l, k='pre_form': r4[f'{f}/{t}/{l}'][k]
out('V-B', 'one-factor sweep, no coupling: nominal / no bias / no noise',
    f"{1e3 * f4('loc_noise', 'uncoupled', '0.01'):.1f} / {1e3 * f4('loc_bias', 'uncoupled', '0.0'):.1f} / "
    f"{1e3 * f4('loc_noise', 'uncoupled', '0.0'):.1f} mm", 'summary.json: r4 (medians)')
r6 = S['r6']
for case, lab in (('traction', 'traction loss'), ('switching', 'intermittent links'), ('stop', 'safety stop')):
    u = r6[f'{case}/uncoupled']
    for typ in ('cascade', 'direct'):
        v = r6[f'{case}/{typ}']
        out('V-B', f'{lab}, {typ}: peak, formation IAE, tracking IAE',
            f"{pct(1 - v['ev_peak'] / u['ev_peak'])} lower, {pct(1 - v['ev_iae'] / u['ev_iae'])} lower, "
            f"{pct(v['ev_track_iae'] / u['ev_track_iae'] - 1)}", 'summary.json: r6 (means)')
_, _, r1 = mfr.rows('r1_paper_delays')
by1 = {(r['case'], r['scen'], r['seed'], r['type']): r for r in r1}
for typ in ('sigleak', 'cascade', 'direct'):
    rat = []
    for case in ('init', 'event'):
        for sc in sorted(set(r['scen'] for r in r1)):
            seeds = sorted(set(k[2] for k in by1 if k[0] == case and k[1] == sc))
            rat.append(np.mean([by1[(case, sc, s, typ)]['m_form_iae'] for s in seeds]) /
                       np.mean([by1[(case, sc, s, 'uncoupled')]['m_form_iae'] for s in seeds]))
    out('V-B', f'kinematic model, delays of [1]: whole-run formation IAE, {typ}', f'{min(rat):.2f}-{max(rat):.2f} of no coupling',
        'r1_paper_delays.json (ratio of means per route and case)')
r3 = S['r3']
for typ in ('cascade', 'direct'):
    rr = {l: r3[f'latency/{typ}/{l}']['ev_iae'] / r3['uncoupled']['ev_iae'] for l in (0.01, 0.1, 0.3, 1.0, 2.0, 4.0)}
    out('V-B', f'latency sweep, {typ}: ratio at 0.01-0.3 s | 1 s | 2 s | 4 s',
        f"{min(rr[l] for l in (0.01, 0.1, 0.3)):.2f}-{max(rr[l] for l in (0.01, 0.1, 0.3)):.2f} | {rr[1.0]:.2f} | {rr[2.0]:.2f} | {rr[4.0]:.2f}",
        'summary.json: r3 (ratio of medians)')
_, _, r3r = mfr.rows('r3_network')
base = {r['seed']: r for r in r3r if r['type'] == 'uncoupled'}
for lv in (2.0, 4.0):
    sel = [r for r in r3r if r.get('factor') == 'latency' and r['type'] == 'cascade' and abs(float(r['level']) - lv) < 1e-9]
    out('V-B', f'latency {lv:g} s, cascade worse than no coupling', f"{sum(r['ev_iae'] > base[r['seed']]['ev_iae'] for r in sel)}/{len(sel)} seeds",
        'r3_network.json (paired)')
out('V-B', 'failed runs in the network sweeps (latency, loss, outages, rate)', f"{sum(S['r3_failed'].values())}",
    'summary.json: r3_failed')
T = load(RESR, 'r0_tuning_summary.json')
for typ in ('cascade', 'sigleak', 'direct'):
    rows = T[typ]['rows']
    out('V-B', f'tuning, {typ}: selected {T[typ]["param"]}; failures per gain', f"{T[typ]['best']:g}; " +
        ' '.join(f"{r['value']:g}:{r['failed']}" for r in rows), 'r0_tuning_summary.json')
out('V-B', 'cascade, Lambda = 4 and 8: formation ratio to no coupling',
    ' '.join(f"{r['form_ratio']:.2f}" for r in T['cascade']['rows'] if r['value'] >= 4), 'r0_tuning_summary.json')
b = load(RESR, 'r0b_summary.json')
out('V-B', 'direct coupling without drive limits, c = 1, 2: failed', f"{b['no_limits/c=1.0']}, {b['no_limits/c=2.0']}", 'r0b_summary.json')
r7 = S['r7']
for N in (8, 12):
    out('V-B', f'complete graph, N = {N}, unit weights: cascade ratio', f"{r7[f'{N}/complete/cascade']['ev_iae'] / r7[f'{N}/none']['ev_iae']:.2f}",
        'summary.json: r7 (ratio of medians)')
sparse = [(N, g) for N in (3, 5, 8, 12) for g in ('ring', 'dring', 'chain', 'star') if not (g == 'ring' and N == 3)]
for typ in ('direct', 'cascade'):
    v = [r7[f'{N}/{g}/{typ}']['ev_iae'] / r7[f'{N}/none']['ev_iae'] for N, g in sparse]
    out('V-B', f'sparse graphs, {typ}: ratio of formation IAE (min-max)', f'{min(v):.3f}-{max(v):.3f}', 'summary.json: r7')
v = [r7[f'{N}/chain/cascade']['ev_iae'] / r7[f'{N}/none']['ev_iae'] for N in (3, 5, 8, 12)]
out('V-B', 'directed chains, cascade: ratio (N = 3, 5, 8, 12)', ' '.join(f'{x:.3f}' for x in v), 'summary.json: r7')
_, _, r7r = mfr.rows('r7_scaling')
fails = sorted(set((r['N'], r['graph'], r['type']) for r in r7r if r['m_failed']))
out('V-B', 'R7 failed runs occur only on', str(fails), 'r7_scaling.json')
_, _, r7b = mfr.rows('r7b_normalised')
out('V-B', 'degree-normalized weights: failed runs', f"{sum(bool(r['m_failed']) for r in r7b)}/{len(r7b)}", 'r7b_normalised.json')
r5 = r5_ratios()
for typ in ('direct', 'cascade'):
    out('V-B', f'random draws, {typ}: median form. ratio, p95, median track. ratio',
        f"{r5[typ]['form_med']:.3f}, {r5[typ]['form_p95']:.2f}, {r5[typ]['track_med']:.3f}", 'r5_domain_rand.json')
out('V-B', 'random draws: failed (direct, cascade); exact two-sided test', f"{r5['direct']['failed']}, {r5['cascade']['failed']}; "
    f"p = {min(1.0, 2 * 0.5 ** (r5['direct']['failed'] + r5['cascade']['failed'])):.2f}", 'r5_domain_rand.json')
for l in ('0.25', '1.0', '4.0'):
    v = S['r8'][f'lam/{l}']
    out('V-B', f'lambda = {l}: formation IAE, tracking IAE after the traction loss', f"{v['ev_iae']:.2f}, {v['ev_track_iae']:.2f} m s",
        'summary.json: r8 (medians)')
for f, l, lab in (('loc_noise', '0.02', '2 cm noise'), ('payload', '300.0', '300 kg payload'), ('act_delay', '0.1', '100 ms actuation'),
                  ('ctrl_rate', '10.0', '10 Hz control'), ('loc_noise', '0.04', '4 cm noise')):
    out('V-B', f'{lab}: steady formation error relative to no coupling',
        ', '.join(f"{t} {f4(f, t, l) / f4(f, 'uncoupled', l):.2f}" for t in ('cascade', 'direct')), 'summary.json: r4 (medians)')
casc = fail = 0
for p in glob.glob(os.path.join(RESR, '*.json')):
    d = json.load(open(p))
    if isinstance(d, dict) and 'jobs' in d and 'metrics' in d:
        for j, m in zip(d['jobs'], d['metrics']):
            if j.get('ctrl', {}).get('type') == 'cascade':
                casc += 1
                fail += bool(m.get('failed'))
out('V-B', 'cascade runs in all robustness experiments; failed', f'{casc}; {fail}', 'results/robust/*.json')


# ---------------------------------------------------------------- additions of arXiv v2 / CDC version
from agvsim import theory as th                       # noqa: E402
from agvsim import runner                              # noqa: E402
from agvsim.references import make_reference, TRIANGLE, vehicle_refs   # noqa: E402

rate = []
for name in ('circle', 'figure8', 'warehouse'):
    ref, T = make_reference(name)
    kr, vmin = 0.0, np.inf
    for t in np.arange(0.0, T, 2e-3):
        _, _, _, vr, wr, _ = vehicle_refs(ref.centre(t, np.zeros(2)), TRIANGLE)
        kr, vmin = max(kr, float(np.max(np.abs(wr) / vr))), min(vmin, float(vr.min()))
    rate.append((name, kr, vmin, float(th.asymptotic_rate(0.7, 3.0, kr, vmin))))
out('IV/V-A', 'kappa_r of the planner references (circle, figure-8, warehouse)', ' '.join(f'{r[1]:.2f}' for r in rate) + ' 1/m',
    'references.py (dense evaluation)')
out('IV/V-A', 'asymptotic rate eta_1^inf (Remark 2, ell = 1 m)', ' '.join(f'{r[3]:.4f}' for r in rate) + ' 1/s', 'theory.asymptotic_rate')
for w in (0.175, 0.68):
    out('IV', f'|h_3(j w)| at w = {w} rad/s, tau = 0..3 s', '-'.join(f"{abs(th.filter_response(w, tau, 1, 1, 5, 2, -1)):.3f}" for tau in (0.0, 3.0)),
        'theory.filter_response (Remark 3)')
ws = np.linspace(1e-3, 30, 30001)
sup = {tau: np.abs(th.filter_response(ws, tau, 1, 1, 5, 2, -1)) for tau in (0.0, 0.1, 0.5, 1.0, 2.0, 3.0)}
out('IV', 'sup_w |h_3| for tau = 0 | max over tau = 0.1..3 s', f"{sup[0.0].max():.3f} | {max(v.max() for v in sup.values()):.3f}", 'theory.filter_response')
out('IV', '|h_3| > 1 from (rad/s), tau = 0..3 s', f"{min(ws[v > 1].min() for v in sup.values()):.2f}-{max(ws[v > 1].min() for v in sup.values()):.2f}",
    'theory.filter_response')
e7 = load(RES, 'e7_curved.json')['runs']
for route in ('line', 'circle'):
    dev = max(abs(r['ratio'] / r['H_abs'] - 1) for r in e7 if r['route'] == route)
    out('V-A', f'E7 {route}: largest |measured ratio / predicted - 1|', f'{dev:.1e} ({sum(r["route"] == route for r in e7)} runs)', 'e7_curved.json')
out('V-A', 'E7 circle: measured ratio, Lam = 10, tau = 0.07 / 3.01 s', ' / '.join(f"{r['ratio']:.4f}" for r in e7 if r['route'] == 'circle' and r['Lam'] == 10.0 and r['h'] in (0.06, 3.0)),
    'e7_curved.json')
out('V-A', 'E7: local errors below e-bar, chi = 1 (all runs)', f"{max(r['eps_max_mm'] for r in e7):.1f} mm, chi_min {min(r['chi_min'] for r in e7):.3f}", 'e7_curved.json')
for cd in (4.0, 6.0, 8.0):
    out('IV', f'delay margin of the linearized direct coupling, c d = {cd:g}, v = 0.6 m/s', f"{th.direct_delay_margin(0.7, 3.0, cd, 0.6):.2f} s",
        'theory.direct_delay_margin (Proposition 2)')
out('IV', 'threshold 2 psi (1 + psi), psi = 0.7', f'{2 * 0.7 * 1.7:.2f}', 'Proposition 2')
doc10, ts10 = runner.load('r10_delay_margin')
agree = total = 0
cas_ok, cas_n = True, 0
mism = []
for i, (j, m) in enumerate(zip(doc10['jobs'], doc10['metrics'])):
    t, e = ts10[f'{i}/t'], ts10[f'{i}/track_max']
    late, mid = e[t >= 110.0 - 1e-9].max(), e[(t >= 55.0 - 1e-9) & (t <= 65.0 + 1e-9)].max()
    ok = (not m.get('failed')) and bool(late < 1e-6 or late < mid)
    if j['ctrl']['type'] == 'cascade':
        cas_ok &= ok; cas_n += 1
        continue
    unstable = th.direct_rhp_roots(0.7, 3.0, 2 * j['ctrl']['c'], 0.6, j['latency'] + 0.025) > 0
    total += 1
    agree += (ok != unstable)
    if ok == unstable:
        mism.append((j['ctrl']['c'], j['latency'], j['seed'], 'decayed' if ok else 'did not decay'))
out('V-A', 'R10 direct coupling: runs whose outcome agrees with Proposition 2', f'{agree}/{total}', 'r10_delay_margin.json + theory.direct_rhp_roots')
out('V-A', 'R10 direct coupling: disagreeing runs (c, latency, seed, outcome)', str(sorted(set(mism)))[:200], 'r10_delay_margin.json')
out('V-A', 'R10 cascade (Lam = 1, 4, 8): all runs decayed', f'{cas_ok} ({cas_n} runs)', 'r10_delay_margin.json')
out('V-A', 'R10 failed runs (tracking error above 3 m)', f"{sum(bool(m.get('failed')) for m in doc10['metrics'])}/{len(doc10['metrics'])}", 'r10_delay_margin.json')
_, _, r11 = mfr.rows('r11_highgain')
_, _, r6r = mfr.rows('r6_events')
mean = lambda rs, k: float(np.nanmean([r[k] for r in rs]))
for case in ('traction', 'stop', 'switching'):
    hg = [r for r in r11 if r.get('exp') == 'r6' and r['case'] == case]
    un = [r for r in r6r if r['case'] == case and r['type'] == 'uncoupled']
    out('V-B', f'{case}, high gain vs none: peak, formation IAE, tracking IAE',
        ', '.join(pct(mean(hg, k) / mean(un, k) - 1) for k in ('ev_peak', 'ev_iae', 'ev_track_iae')), 'r11_highgain.json, r6 (means)')
    for typ in ('direct', 'cascade'):
        rr = [r for r in r6r if r['case'] == case and r['type'] == typ]
        hb = {r['seed']: r for r in hg}
        out('V-B', f'{case}, {typ} vs high gain: formation IAE ratio of means; lower on seeds',
            f"{mean(rr, 'ev_iae') / mean(hg, 'ev_iae'):.3f}; {sum(r['ev_iae'] < hb[r['seed']]['ev_iae'] for r in rr)}/{len(rr)}",
            'r11_highgain.json, r6 (paired)')
f11 = {}
for r in r11:
    if r.get('exp') == 'r0':
        f11.setdefault(r['ctrl']['c'], []).append(bool(r['m_failed']))
out('V-B', 'tuning grid, high gain: failures per c', ' '.join(f'{c:g}:{sum(v)}' for c, v in sorted(f11.items())), 'r11_highgain.json')
for e, lab in (('r7', 'unit'), ('r7b', 'normalized')):
    rr = [r for r in r11 if r.get('exp') == e and r['N'] in (8, 12)]
    out('V-B', f'complete graphs N = 8, 12, {lab} weights, high gain: failed', f"{sum(bool(r['m_failed']) for r in rr)}/{len(rr)}", 'r11_highgain.json')
_, _, r12 = mfr.rows('r12_age_gate')
b12 = np.median([r['ev_iae'] for r in r3r if r['type'] == 'uncoupled'])
for typ in ('cascade', 'direct'):
    out('V-B', f'age gate 1.5 s, {typ}: ratio at 0.01 | 0.1 | 0.3 | 1 | 2 | 4 s', ' | '.join(
        f"{np.median([r['ev_iae'] for r in r12 if r['type'] == typ and abs(r['level'] - l) < 1e-9]) / b12:.3f}" for l in (0.01, 0.1, 0.3, 1.0, 2.0, 4.0)),
        'r12_age_gate.json (ratio of medians)')
out('V-B', 'age gate: failed runs', f"{sum(bool(r['m_failed']) for r in r12)}/{len(r12)}", 'r12_age_gate.json')
r2r = mfr.rows('r2_realistic')[2] + [r for r in r11 if r.get('exp') == 'r2']
for typ in ('uncoupled', 'highgain', 'direct', 'cascade'):
    rr = [r for r in r2r if r['case'] == 'nominal' and r['scen'] == 'warehouse' and r['type'] == typ]
    out('V-B', f'nominal, {typ}: RMS of dv/dt, heading RMS', f"{mean(rr, 'm_cmd_acc_rms'):.2f} m/s^2, {np.degrees(mean(rr, 'm_head_rms')):.2f} deg",
        'r2_realistic.json, r11_highgain.json (means)')
casc = fail = 0
for p_ in glob.glob(os.path.join(RESR, '*.json')):
    d_ = json.load(open(p_))
    if isinstance(d_, dict) and 'jobs' in d_ and 'metrics' in d_:
        for j, m in zip(d_['jobs'], d_['metrics']):
            if j.get('ctrl', {}).get('type') == 'cascade':
                casc += 1
                fail += bool(m.get('failed'))
out('V-B', 'cascade runs in all robustness experiments incl. R10, R12; failed', f'{casc}; {fail}', 'results/robust/*.json')
