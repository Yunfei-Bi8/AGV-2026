"""LaTeX tables of the paper (arXiv v2 numbering) from the stored results; nothing is simulated.

    python3 paper/make_paper_tables.py        # -> paper/tables/*.tex

Table I    tab_verif.tex    worst cases of the verification (E3: results/e3_summary.json)
Table II   tab_hifi.tex     main results on the high-fidelity model (R0, R2, R5, R6, R7); tab_hifi_cdc.tex is the
                            compact form of the conference version
Table IV   tab_scale.tex    rate per meter of Lemma 2 (E1: results/e1_prop1.json)
Table V    tab_rates.tex    rate of Lemma 3 for a constant delay (E2: results/e2_lemma2.json)
Table VI   tab_meas.tex     measured decay rates (E3)
Table VII  tab_slip.tex     traction loss on the kinematic model (E5: results/e5_formation.json)
Tables X-XVI tab_r0, tab_r1, tab_r2_nominal, tab_r2_event, tab_r5, tab_r6, tab_r7b (robustness study, through the
           table functions of experiments/robust/make_figures_robust.py)
Tables III, VIII and IX describe the set-up and are written by hand in the paper."""
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, 'experiments', 'robust')):
    if p not in sys.path:
        sys.path.insert(0, p)
OUT = os.path.join(ROOT, 'paper', 'tables')
os.makedirs(OUT, exist_ok=True)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import make_figures_robust as mfr
from agvsim import theory as th

RES = os.path.join(ROOT, 'results')
RESR = os.path.join(RES, 'robust')


def load(*p):
    return json.load(open(os.path.join(RES, *p)))


def sci(x, digits=0):
    """x as a\\cdot10^{b} (LaTeX, math mode)."""
    b = int(math.floor(math.log10(abs(x))))
    a = round(x / 10 ** b, digits)
    if a >= 10:
        a, b = a / 10, b + 1
    return f'{a:.{digits}f}\\cdot10^{{{b}}}'


def order(x):
    return f'10^{{{int(round(math.log10(abs(x))))}}}'


def write(name, lines):
    open(os.path.join(OUT, name + '.tex'), 'w').write('\n'.join(lines) + '\n')
    print('wrote', name)


# ---------------------------------------------------------------- Table I
def tab_verif():
    S = load('e3_summary.json')['summary']
    g = lambda k, f, sel=lambda s: True: f([s[k] for s in S if sel(s)])
    coupled = lambda s: s['graph'] != 'none'
    n = sum(s['runs'] for s in S)
    lem2 = max(1 / g('Vdot_ratio_min_run', min), 1 / g('Vdot_ratio_min_apriori', min))
    fin = [g('e_end_max', max, lambda s, net=net: s['net'] == net) for net in ('nominal', 'slow1', 'slow3')]
    rows = [
        ('Lemma~\\ref{lem:wp}', '$\\min_t\\vh_i-\\hat v_{\\min}$', f"$+{g('vhat_margin_min', min):.3f}$\\,m/s"),
        ('Thm.~\\ref{thm:conv}(i)', 'rise of $W_i$ between samples$/W_i(0)$', f"${sci(g('W_rise_max', max))}$"),
        ('', '$|W_i+Q_i-W_i(0)|$', f"${sci(g('balance_max', max))}$\\,m$^2$"),
        ('Thm.~\\ref{thm:conv}(ii)', 'ratio for $|\\eps_i|$; $|\\zeta_i|$, $|e_i|$ (coupled runs)',
         f"{g('eps_ratio_max', max):.3f};\\ {max(g('zeta_ratio_max', max, coupled), g('e_ratio_max', max, coupled)):.2f}"),
        ('Lemma~\\ref{lem:strict}', 'decrease of $U_1$: guaranteed over actual', f'{lem2:.2f}'),
        ('Lemma~\\ref{lem:halanay}', '$M(t)$ over $\\sup_{s\\le t}P(s)/\\lambda$', f"{g('lemma2_ratio_max', max):.3f}"),
        ('Thm.~\\ref{thm:exp}', 'ratios for $|\\eps_i|$, $|e_i|$, heading, $t\\ge1$\\,s',
         f"{g('thm2_t1_eps', max):.2f}, {g('thm2_t1_e', max):.2f}, {g('thm2_t1_head', max):.2f}"),
        ('Thm.~\\ref{thm:conv}(iii)', 'final $|e_i|$ (m), 60\\,ms/1\\,s/3\\,s network', '$' + '$/$'.join(order(x) for x in fin) + '$'),
    ]
    write('tab_verif', [
        '\\begin{table}[t]',
        f'\\caption{{Worst case over {n} closed-loop runs on the kinematic model (ratio: observed value divided by the '
        'bound, so that every ratio must not exceed 1; $Q_i$: integrated dissipation)}\\label{tab:verif}',
        '\\centering\\footnotesize', '\\setlength{\\tabcolsep}{3pt}', '\\begin{tabular}{@{}llc@{}}', '\\toprule',
        'Result & Quantity & Worst case\\\\', '\\midrule'] + [' & '.join(r) + '\\\\' for r in rows] +
        ['\\bottomrule', '\\end{tabular}', '\\end{table}'])


# ---------------------------------------------------------------- Table II
def r5_ratios():
    doc, ts, rs = mfr.rows('r5_domain_rand')
    seeds = sorted(set(r['seed'] for r in rs))
    by = {(r['seed'], r['type']): r for r in rs}
    out = {}
    for typ in ('cascade', 'direct', 'sigleak'):
        ok = [s for s in seeds if not by[(s, typ)]['m_failed'] and not by[(s, 'uncoupled')]['m_failed']]
        fr = np.array([by[(s, typ)]['m_form_iae'] / by[(s, 'uncoupled')]['m_form_iae'] for s in ok])
        tr = np.array([by[(s, typ)]['m_track_iae'] / by[(s, 'uncoupled')]['m_track_iae'] for s in ok])
        out[typ] = dict(form_med=float(np.median(fr)), form_p95=float(np.quantile(fr, 0.95)), track_med=float(np.median(tr)),
                        failed=int(sum(by[(s, typ)]['m_failed'] for s in seeds)), n=len(seeds))
    out['uncoupled'] = dict(failed=int(sum(by[(s, 'uncoupled')]['m_failed'] for s in seeds)), n=len(seeds))
    return out


def tab_hifi():
    S = json.load(open(os.path.join(RESR, 'summary.json')))
    r0 = json.load(open(os.path.join(RESR, 'r0_tuning_summary.json')))
    d0 = json.load(open(os.path.join(RESR, 'r0_tuning.json')))
    d7 = json.load(open(os.path.join(RESR, 'r7_scaling.json')))
    r5 = r5_ratios()
    C = ('uncoupled', 'direct', 'cascade')
    nom = [1e3 * S['r2'][f'warehouse/nominal/{c}']['pre_form'] for c in C]
    ev = lambda case, k, sc=1.0, nd=2: ' & '.join(f"{sc * S['r6'][f'{case}/{c}'][k]:.{nd}f}" for c in C)
    unc0 = [m for j, m in zip(d0['jobs'], d0['metrics']) if j['ctrl']['type'] == 'uncoupled']
    big = {typ: [r for r in r0[typ]['rows'] if r['value'] == max(x['value'] for x in r0[typ]['rows'])][0] for typ in ('direct', 'cascade')}
    f7 = {}
    for c in C:
        sel = [m for j, m in zip(d7['jobs'], d7['metrics']) if j.get('N') in (8, 12)
               and ((j.get('graph') == 'none') if c == 'uncoupled' else (j.get('graph') == 'complete' and j['ctrl']['type'] == c))]
        f7[c] = (sum(bool(m['failed']) for m in sel), len(sel))
    H = lambda t: '\\multicolumn{4}{@{}l}{' + t + '}\\\\'
    rows = [
        H('Nominal operation (R2)'),
        '\\quad formation RMS (mm) & ' + ' & '.join(f'{x:.1f}' for x in nom) + '\\\\',
        H('Traction loss (R6)'),
        '\\quad peak formation error (mm) & ' + ev('traction', 'ev_peak', 1e3, 0) + '\\\\',
        '\\quad formation IAE (m$\\cdot$s) & ' + ev('traction', 'ev_iae') + '\\\\',
        '\\quad tracking IAE (m$\\cdot$s) & ' + ev('traction', 'ev_track_iae') + '\\\\',
        H('Safety stop (R6)'),
        '\\quad peak formation error (mm) & ' + ev('stop', 'ev_peak', 1e3, 0) + '\\\\',
        '\\quad formation IAE (m$\\cdot$s) & ' + ev('stop', 'ev_iae') + '\\\\',
        '\\quad tracking IAE (m$\\cdot$s) & ' + ev('stop', 'ev_track_iae') + '\\\\',
        H('Random draws, whole run (R5)'),
        f"\\quad formation IAE ratio, median & 1 & {r5['direct']['form_med']:.2f} & {r5['cascade']['form_med']:.2f}\\\\",
        f"\\quad formation IAE ratio, 95th pct. & -- & {r5['direct']['form_p95']:.2f} & {r5['cascade']['form_p95']:.2f}\\\\",
        f"\\quad tracking IAE ratio, median & 1 & {r5['direct']['track_med']:.2f} & {r5['cascade']['track_med']:.2f}\\\\",
        H('Failed runs'),
        f"\\quad largest tuning gains (R0) & {sum(bool(m['failed']) for m in unc0)}/{len(unc0)} & "
        f"{big['direct']['failed']}/20 & {big['cascade']['failed']}/20\\\\",
        '\\quad $N=8$ and 12, complete graph (R7) & ' + ' & '.join(f'{f7[c][0]}/{f7[c][1]}' for c in C) + '\\\\',
        '\\quad random draws (R5) & ' + ' & '.join(f"{r5[c]['failed']}/{r5[c]['n']}" for c in C) + '\\\\']
    write('tab_hifi', [
        '\\begin{table}[t]',
        '\\caption{Main results on the high-fidelity model. R2: warehouse route, mean over 30 seeds; R6: warehouse route, '
        'event of AGV~2 at 30\\,s, mean over 20 seeds, IAE over 15\\,s from the onset; R5: whole-run IAE relative to no '
        'coupling, 150 random draws of all model parameters; failed: a tracking-error coordinate exceeded 3\\,m '
        '(largest tuning gains $\\Lambda=8$\\,s$^{-1}$, $c=4$)}\\label{tab:hifi}',
        '\\centering\\footnotesize', '\\setlength{\\tabcolsep}{3.5pt}', '\\begin{tabular}{@{}lccc@{}}', '\\toprule',
        ' & No coupling & Direct & Cascade\\\\', '\\midrule'] + rows + ['\\bottomrule', '\\end{tabular}', '\\end{table}'])


    num = lambda case, k, sc=1.0, nd=2: [f"{sc * S['r6'][f'{case}/{c}'][k]:.{nd}f}" for c in C]
    cdc = [('Nominal: formation RMS (mm)', [f'{x:.1f}' for x in nom]),
           ('Traction loss: peak (mm)', num('traction', 'ev_peak', 1e3, 0)),
           ('\\quad formation IAE (m$\\cdot$s)', num('traction', 'ev_iae')),
           ('\\quad tracking IAE (m$\\cdot$s)', num('traction', 'ev_track_iae')),
           ('Safety stop: peak (mm)', num('stop', 'ev_peak', 1e3, 0)),
           ('\\quad formation IAE (m$\\cdot$s)', num('stop', 'ev_iae')),
           ('\\quad tracking IAE (m$\\cdot$s)', num('stop', 'ev_track_iae')),
           ('Random draws: formation ratio', ['1', f"{r5['direct']['form_med']:.2f}", f"{r5['cascade']['form_med']:.2f}"]),
           ('\\quad 95th percentile', ['--', f"{r5['direct']['form_p95']:.2f}", f"{r5['cascade']['form_p95']:.2f}"]),
           ('\\quad tracking ratio', ['1', f"{r5['direct']['track_med']:.2f}", f"{r5['cascade']['track_med']:.2f}"]),
           ('Failed: largest tuning gains', [f"{sum(bool(m['failed']) for m in unc0)}/{len(unc0)}",
                                             f"{big['direct']['failed']}/20", f"{big['cascade']['failed']}/20"]),
           ('\\quad $N=8$, 12, complete graph', [f'{f7[c][0]}/{f7[c][1]}' for c in C]),
           ('\\quad random draws', [f"{r5[c]['failed']}/{r5[c]['n']}" for c in C])]
    write('tab_hifi_cdc', [
        '\\begin{table}[t]',
        '\\caption{Main results on the high-fidelity model (random draws: medians and 95th percentile of the whole-run IAE '
        'relative to no coupling)}\\label{tab:hifi}',
        '\\centering\\footnotesize', '\\setlength{\\tabcolsep}{3.5pt}', '\\begin{tabular}{@{}lccc@{}}', '\\toprule',
        ' & No coupling & Direct & Cascade\\\\', '\\midrule'] + [lab + ' & ' + ' & '.join(v) + '\\\\' for lab, v in cdc] +
        ['\\bottomrule', '\\end{tabular}', '\\end{table}'])


# ---------------------------------------------------------------- Tables IV-VII
def tab_scale():
    rows = [r for r in load('e1_prop1.json')['scale_table'] if r['psi'] == 0.7 and r['b'] == 3.0]
    body = [f"{r['kbar']:g} & {r['alpha_1m']:.4f} & {r['rate_1m']:.3f} & {r['ell_opt']:.3f} & {r['rate_opt']:.3f} & "
            f"{r['rate_opt'] / r['rate_1m']:.1f}\\\\" for r in rows]
    write('tab_scale', [
        '\\begin{table}[!t]',
        '\\caption{Guaranteed rate per meter $\\alpha_\\ell b/c_2$ of Lemma~\\ref{lem:strict} for $\\ell=1$\\,m and for the '
        'best $\\ell$ on a logarithmic grid ($\\psi=0.7$, $b=3$\\,m$^{-2}$)}\\label{tab:scale}',
        '\\centering\\footnotesize', '\\setlength{\\tabcolsep}{3.5pt}', '\\begin{tabular}{@{}cccccc@{}}\\toprule',
        '$\\kappa$ & $\\alpha_1$ & rate, $\\ell=1$\\,m & best $\\ell$ & rate, best $\\ell$ & gain\\\\',
        '(m$^{-1}$) & (m) & (m$^{-1}$) & (m) & (m$^{-1}$) & \\\\\\midrule'] + body[:-1] + [body[-1] + '\\bottomrule',
        '\\end{tabular}', '\\end{table}'])


def tab_rates():
    R = load('e2_lemma2.json')['rates']
    write('tab_rates', [
        '\\begin{table}[!t]',
        '\\caption{Decay rate of the filter network for a constant delay on the complete graph of three nodes '
        '($\\lambda=\\Lambda=1$\\,s$^{-1}$, $\\chi_0=1$, $\\beta\\to0$). The simulated delay lies in $[h,h+2$\\,ms$)$, so '
        'Lemma~\\ref{lem:halanay} gives the interval $[\\eta(h+2\\,\\mathrm{ms}),\\eta(h)]$}\\label{tab:rates}',
        '\\centering\\footnotesize', '\\setlength{\\tabcolsep}{3pt}', '\\begin{tabular}{@{}lcccccc@{}}\\toprule',
        '$h$ (s) & ' + ' & '.join(f"{r['h']:.2f}" for r in R) + '\\\\\\midrule',
        '$\\eta(h)$ (s$^{-1}$) & ' + ' & '.join(f"{r['rate_lemma2']:.4f}" for r in R) + '\\\\',
        '$\\eta(h+2$\\,ms$)$ (s$^{-1}$) & ' + ' & '.join(f"{r['rate_lemma2_hTs']:.4f}" for r in R) + '\\\\',
        'measured (s$^{-1}$) & ' + ' & '.join(f"{r['rate_measured']:.4f}" for r in R) + '\\\\\\bottomrule',
        '\\end{tabular}', '\\end{table}'])
    return all(r['rate_lemma2_hTs'] <= r['rate_measured'] <= r['rate_lemma2'] for r in R)


def tab_meas():
    S = load('e3_summary.json')['summary']
    REF = {'circle': 'Circle', 'figure8': 'Figure-8', 'warehouse': 'Warehouse'}
    NET = {'nominal': '60\\,ms', 'slow1': '1\\,s', 'slow3': '3\\,s'}
    body = []
    for ref in REF:
        for net in NET:
            s = [x for x in S if x['ref'] == ref and x['net'] == net and x['graph'] == 'complete' and x['tag'] == 'std'][0]
            body.append(f"{REF[ref] if net == 'nominal' else ''} & {NET[net]} & {s['rate_eps']:.2f} & {s['rate_W']:.2f} & "
                        f"{s['rate_M']:.2f} & {s['rate_e']:.2f} & {s['rate_lemma2']:.2f} & {s['t_form_2cm_med']:.1f}\\\\")
    write('tab_meas', [
        '\\begin{table}[!t]',
        '\\caption{Measured decay rates (median over 20 seeds, complete graph), the rate of Lemma~\\ref{lem:halanay} for the '
        'largest delay of each run, and the time after which the formation error stays below 2\\,cm}\\label{tab:meas}',
        '\\centering\\footnotesize', '\\setlength{\\tabcolsep}{2.2pt}', '\\begin{tabular}{@{}llcccccc@{}}\\toprule',
        'Reference & Network & $|\\eps_i|$ & $W_i$ & $|\\zeta_i|$ & $|e_i|$ & Lemma~\\ref{lem:halanay} & $<2$\\,cm\\\\',
        ' & & (s$^{-1}$) & (s$^{-1}$) & (s$^{-1}$) & (s$^{-1}$) & (s$^{-1}$) & after (s)\\\\\\midrule'] + body[:-1] +
        [body[-1] + '\\bottomrule', '\\end{tabular}', '\\end{table}'])


def tab_slip():
    S = {(r['Lam'], r['net']): r for r in load('e5_formation.json')['slip']}
    cols = [(0.0, 'nominal'), (1.0, 'nominal'), (3.0, 'nominal'), (0.0, 'slow1'), (1.0, 'slow1'), (3.0, 'slow1')]
    line = lambda lab, f: lab + ' & ' + ' & '.join(f(S[c]) for c in cols) + '\\\\'
    t = lambda x: f'{x:.2f}' if x < 10 else f'{x:.1f}'
    write('tab_slip', [
        '\\begin{table*}[!t]',
        '\\caption{Traction loss of AGV~2 (40\\% of the commanded velocities for 2\\,s) on the kinematic model, warehouse '
        'route; recovery times from the end of the traction loss}\\label{tab:slip}',
        '\\centering\\footnotesize', '\\begin{tabular}{@{}lcccccc@{}}\\toprule',
        ' & \\multicolumn{3}{c}{60\\,ms network} & \\multicolumn{3}{c}{1\\,s network}\\\\\\cmidrule(lr){2-4}\\cmidrule(l){5-7}',
        ' & no coupling & $\\Lambda=1$ & $\\Lambda=3$ & no coupling & $\\Lambda=1$ & $\\Lambda=3$\\\\\\midrule',
        line('Peak formation error (mm)', lambda r: f"{r['form_peak_mm']:.0f}"),
        line('Peak tracking error, other AGVs (mm)', lambda r: f"{r['others_peak_mm']:.0f}"),
        line('Formation error below 5\\,cm after (s)', lambda r: t(r['t_form_5cm'])),
        line('All tracking errors below 5\\,cm after (s)', lambda r: t(r['t_track_5cm'])) + '\\bottomrule',
        '\\end{tabular}', '\\end{table*}'])


# ---------------------------------------------------------------- robustness tables (Appendix B-B)
WIDE = {'tab_r1', 'tab_r2_nominal', 'tab_r2_event', 'tab_r6', 'tab_r7b'}
WORDS = [('localisation', 'localization'), ('Localisation', 'Localization'), ('relocalisation', 'relocalization'),
         ('randomised', 'randomized'), ('Domain-randomised', 'Domain-randomized'), ('normalised', 'normalized'),
         ('mm\\,s', 'mm$\\cdot$s'), ('m\\,s', 'm$\\cdot$s'), ('\\,\\%', '\\%'), ('95\\,\\% CI', '95\\% bootstrap confidence interval'),
         ('95\\% CI', '95\\% bootstrap confidence interval'), ('[95\\% confidence interval]', '[95\\% bootstrap confidence interval]'),
         ('95\\% bootstrap CI', '95\\% bootstrap confidence interval'),
         ('the model of the theorems', 'the kinematic model'), ('the uncoupled law', 'no coupling')]
CAPTION = {   # additions and replacements of the study's captions
    'tab_r0': ('Gain tuning on', 'Gain selection (R0) on', ' Ratios of the whole-run IAE; every tuning run contains a traction '
               'loss of a random vehicle at a random time between 25 and 35\\,s.'),
    'tab_r1': ('Holm-corrected', 'Holm-corrected over the three comparisons with the cascade in each block',
               ' IAE over the whole run (60\\,s for the transport maneuver, 75\\,s for the figure-8).'),
    'tab_r2_event': ('Traction loss on the high-fidelity model (15\\,s window from its onset)',
                     'Traction loss of a random vehicle at a random time between 25 and 35\\,s on the high-fidelity model '
                     '(R2), 15\\,s window from its onset', ''),
    'tab_r2_nominal': ('Nominal operation on the high-fidelity model', 'Nominal operation on the high-fidelity model (R2)',
                       " Localization: RMS position error of each vehicle's pose estimate."),
    'tab_r5': ('Domain-randomized Monte Carlo', 'Domain randomization (R5)', ' Ratios of the whole-run IAE.'),
    'tab_r6': ('Disturbance events on the high-fidelity model (warehouse route, 20 seeds',
               'Events of AGV~2 at 30\\,s on the high-fidelity model (R6, warehouse route, 20 seeds',
               ' No coupling uses no network, so that its row for switching links repeats the traction loss.'),
    'tab_r7b': ('Complete graphs with unit weights (R7)', 'Complete graphs with unit weights (R7)', ''),
}


def write_table(name, header, body, caption, label, colspec, note=None):
    def fix(s):
        for a, b in WORDS:
            s = s.replace(a, b)
        return s
    if name == 'tab_r6':                                   # link failures are reported in the text
        out, skip = [], False
        for b in body:
            if b.startswith('link failure'):
                skip = True
            if not skip:
                out.append(b)
            if skip and b == '\\addlinespace':
                skip = False
        body = out[:-1] if out and out[-1] == '\\addlinespace' else out
    old, new, extra = CAPTION.get(name, (None, None, ''))
    caption = fix(caption)
    if old:
        assert old in caption, (name, caption)
        caption = caption.replace(old, new)
    if name in ('tab_r2_event', 'tab_r2_nominal'):
        caption = caption.replace('Holm-corrected', 'Holm-corrected over the three comparisons with the cascade in each scenario')
    caption = caption.rstrip('.') + '.' + extra
    env = 'table*' if name in WIDE else 'table'
    width = '\\textwidth' if name in WIDE else '\\columnwidth'
    lines = ['\\begin{%s}[!t]\\centering\\footnotesize' % env, '\\caption{%s}\\label{%s}' % (caption, label),
             '\\begin{adjustbox}{max width=%s}' % width, '\\begin{tabular}{%s}\\toprule' % colspec, fix(header) + '\\\\\\midrule']
    lines += [fix(b) for b in body]
    lines += ['\\bottomrule\\end{tabular}', '\\end{adjustbox}', '\\end{%s}' % env]
    write(name, lines)


mfr.write_table = write_table
mfr.save = lambda fig, name: plt.close(fig)
_rows = mfr.rows


def rows_paper(name):
    """The localization column of R2 reports the RMS position error of each vehicle's estimate (the stored time series
    only hold the mean over the vehicles); for runs without an event both use the window t >= 15 s."""
    doc, ts, rs = _rows(name)
    if name == 'r2_realistic':
        for r in rs:
            if r.get('case') == 'nominal' and 'm_loc_rms' in r:
                r['pre_loc'] = r['m_loc_rms']
    return doc, ts, rs


mfr.rows = rows_paper

if __name__ == '__main__':
    tab_verif(); tab_hifi(); tab_scale(); tab_meas(); tab_slip()
    assert tab_rates(), 'a measured rate lies outside the interval of Lemma 3'
    for f in ('r0', 'r1', 'r2', 'r5', 'r6', 'r7b'):
        getattr(mfr, f)()
