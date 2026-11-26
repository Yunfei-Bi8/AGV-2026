"""Figures (PDF) and LaTeX tables of the robustness study, generated from results/robust/*.json and *_ts.npz.

Usage: python3 make_figures_robust.py [r1 r2 ...]      (default: everything whose results exist)"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import load, RES, CONTROLLERS, LABEL, event_metrics, window_rms, ts_of
from agvsim.stats import boot_ci, paired_test, holm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG = os.path.join(ROOT, 'figures', 'robust')
TAB = os.path.join(ROOT, 'tables')                    # LaTeX tables of the paper
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)

COL = {'cascade': '#2a78d6', 'direct': '#eb6834', 'sigleak': '#1baf7a', 'uncoupled': '#8b8a83'}
MK = {'cascade': 'o', 'direct': 's', 'sigleak': '^', 'uncoupled': 'D'}
SCN = {'transport': 'transport', 'figure8': 'figure-8', 'warehouse': 'warehouse'}
CASEN = {'init': 'initial errors', 'event': 'traction loss'}
EVN = {'traction': 'traction loss', 'stop': 'safety stop', 'reloc': 'relocalisation', 'switching': 'switching links',
       'linkfail': 'link failure', 'linkfail_wd': 'link failure + watchdog'}
SHORT = {'cascade': 'Cascade', 'direct': 'Direct', 'sigleak': '$\\sigma$-leak', 'uncoupled': 'No coupling'}
INK, INK2, GRID, AXIS = '#0b0b0b', '#52514a', '#e1e0d9', '#c3c2b7'
RAMP = ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#104281']
plt.rcParams.update({'font.size': 8, 'font.family': 'DejaVu Sans', 'axes.edgecolor': AXIS, 'axes.linewidth': 0.8,
                     'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
                     'legend.frameon': False, 'legend.fontsize': 7, 'lines.linewidth': 1.3, 'axes.titlesize': 8.2,
                     'axes.titleweight': 'bold', 'axes.spines.top': False, 'axes.spines.right': False,
                     'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'pdf.fonttype': 42, 'savefig.bbox': 'tight',
                     'savefig.pad_inches': 0.03})
SUMMARY = {}


# ---------------------------------------------------------------- data access
def rows(name):
    """One dict per job: job fields, metrics, and quantities derived from the stored time series."""
    doc, ts = load(name)
    out = []
    for i, (j, m) in enumerate(zip(doc['jobs'], doc['metrics'])):
        r = dict(j); r.update({'m_' + k: v for k, v in m.items()})
        r['type'] = j['ctrl']['type']; r['idx'] = i
        if ts is not None and f'{i}/t' in ts:
            s = ts_of(ts, i)
            t = s['t']
            ev = j.get('events') or []
            t0 = ev[0]['t0'] if ev else None
            t1 = (ev[0].get('t1', ev[0]['t0'] + 1.0)) if ev else None
            hi = (t0 - 0.5) if ev else t[-1]
            r['pre_form'] = window_rms(t, s['form'], 15.0, hi)
            r['pre_form_est'] = window_rms(t, s['form_est'], 15.0, hi)
            r['pre_track'] = window_rms(t, s['track'], 15.0, hi)
            r['run_form'] = window_rms(t, s['form'], 15.0, t[-1])     # largest pair, t >= 15 s, events included
            r['pre_loc'] = window_rms(t, s['loc'], 15.0, hi)
            if ev:
                em = event_metrics(t, s['form'], t0, t1)
                r.update({'ev_' + k: v for k, v in em.items()})
                mt = (t >= t0) & (t <= t0 + 15.0)
                r['ev_track_iae'] = float(np.trapezoid(s['track'][mt], t[mt]))
                r['ev_track_peak'] = float(s['track_max'][mt].max())
        out.append(r)
    return doc, ts, out


def sel(rs, **kw):
    return [r for r in rs if all((r.get(k) == v) for k, v in kw.items())]


def vals(rs, key):
    return np.array([r.get(key, np.nan) if r.get(key) is not None else np.nan for r in rs], float)


def fmt_ci(x, scale=1.0, nd=1):
    m, lo, hi = boot_ci(np.asarray(x) * scale)
    if not np.isfinite(m):
        return '--'
    return f'{m:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]'


def ptxt(p):
    if not np.isfinite(p):
        return '--'
    return '$<$0.001' if p < 1e-3 else f'{p:.3f}'



def save(fig, name):
    fig.savefig(os.path.join(FIG, name + '.pdf'))
    plt.close(fig)


def write_table(name, header, body, caption, label, colspec, note=None):
    lines = ['\\begin{table}[htbp]\\centering\\small', f'\\caption{{{caption}}}\\label{{{label}}}',
             '\\begin{adjustbox}{max width=\\linewidth}',
             f'\\begin{{tabular}}{{{colspec}}}\\toprule', header + '\\\\\\midrule']
    lines += body
    lines += ['\\bottomrule\\end{tabular}', '\\end{adjustbox}']
    if note:
        lines += [f'\\par\\smallskip\\parbox{{0.95\\linewidth}}{{\\footnotesize {note}}}']
    lines += ['\\end{table}']
    open(os.path.join(TAB, name + '.tex'), 'w').write('\n'.join(lines) + '\n')


def band(ax, t, Y, col, label, lw=1.3, alpha=0.18):
    med = np.nanmedian(Y, 0); q1, q3 = np.nanquantile(Y, 0.25, 0), np.nanquantile(Y, 0.75, 0)
    ax.fill_between(t, q1, q3, color=col, alpha=alpha, lw=0)
    ax.plot(t, med, color=col, lw=lw, label=label)


def stack_ts(ts, rs, key):
    L = min(len(ts[f"{r['idx']}/t"]) for r in rs)
    return ts[f"{rs[0]['idx']}/t"][:L], np.stack([ts[f"{r['idx']}/{key}"][:L] for r in rs])


# ---------------------------------------------------------------- R0
def r0():
    S = json.load(open(os.path.join(RES, 'r0_tuning_summary.json')))
    fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.5))
    for typ in ('cascade', 'sigleak', 'direct'):
        rr = S[typ]['rows']
        x = [r['track_ratio'] for r in rr]; y = [r['form_ratio'] for r in rr]
        ax.plot(x, y, color=COL[typ], marker=MK[typ], ms=4, mec='white', mew=0.5, label=SHORT[typ])
        for r in rr:
            if r['value'] == S[typ]['best']:
                ax.plot(r['track_ratio'], r['form_ratio'], marker=MK[typ], ms=8, mfc='none', mec=COL[typ], mew=1.2)
            ax.annotate(f"{r['value']:g}", (r['track_ratio'], r['form_ratio']), textcoords='offset points', xytext=(3, 2), fontsize=5.5, color=INK2)
    ax.axhline(1, color=COL['uncoupled'], lw=0.8, ls=(0, (3, 2))); ax.axvline(1, color=COL['uncoupled'], lw=0.8, ls=(0, (3, 2)))
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('tracking IAE / no coupling'); ax.set_ylabel('formation IAE / no coupling')
    ax.set_title('Tuning seeds: trade-off per coupling gain', loc='left'); ax.legend(loc='lower right')
    save(fig, 'r0_tuning')
    body = []
    for typ in ('cascade', 'sigleak', 'direct'):
        par = S[typ]['param']
        for r in S[typ]['rows']:
            star = '$\\star$' if r['value'] == S[typ]['best'] else ''
            sym = '\\Lambda' if par == 'Lam' else 'c'
            name0 = SHORT[typ] if r is S[typ]['rows'][0] else ''
            body.append(f"{name0} & ${sym}={r['value']:g}${star} & {r['form_ratio']:.3f} & {r['track_ratio']:.3f} & {r['failed']}" + '\\\\')
        body.append('\\addlinespace')
    write_table('tab_r0', 'Controller & gain & formation IAE ratio & tracking IAE ratio & failed', body[:-1],
                'Gain tuning on 10 tuning seeds $\\times$ 2 scenarios with a random traction loss: mean ratios to the uncoupled law over the runs that did not fail, and number of failed runs out of 20 ($\\star$: selected gain).',
                'tab:r0', '@{}llccc@{}')
    SUMMARY['r0'] = {typ: S[typ]['best'] for typ in S}


# ---------------------------------------------------------------- R1
def r1():
    doc, ts, rs = rows('r1_paper_delays')
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
    for ax, case, title in zip(axs, ('init', 'event'), ('(a) Initial errors only', '(b) With a traction loss')):
        for typ in ['uncoupled', 'sigleak', 'direct', 'cascade']:
            rr = sel(rs, scen='transport', case=case, type=typ)
            t, Y = stack_ts(ts, rr, 'form')
            band(ax, t, np.maximum(Y, 1e-6), COL[typ], SHORT[typ])
        ax.set_yscale('log'); ax.set_xlabel('time (s)'); ax.set_title(title, loc='left'); ax.set_xlim(0, 60)
    axs[0].set_ylabel('formation error (m)'); axs[0].legend(loc='lower left', ncol=2)
    save(fig, 'r1_paper_delays')
    body = []
    for sc in ('transport', 'figure8'):
        for case in ('init', 'event'):
            base = {r['seed']: r for r in sel(rs, scen=sc, case=case, type='cascade')}
            ps = {}
            for typ in CONTROLLERS:
                rr = sel(rs, scen=sc, case=case, type=typ)
                if typ != 'cascade':
                    a = [base[r['seed']]['m_form_iae'] for r in rr]; b = [r['m_form_iae'] for r in rr]
                    ps[typ] = paired_test(a, b)
            padj = dict(zip(ps.keys(), holm([v[0] for v in ps.values()])))
            for typ in ['cascade', 'direct', 'sigleak', 'uncoupled']:
                rr = sel(rs, scen=sc, case=case, type=typ)
                p = padj.get(typ, np.nan)
                body.append(f"{(SCN[sc] + ' / ' + CASEN[case]) if typ == 'cascade' else ''} & {SHORT[typ]} & "
                            f"{fmt_ci(vals(rr, 'm_form_iae'), 1e3, 0)} & {fmt_ci(vals(rr, 'm_track_iae'), 1e3, 0)} & "
                            f"{fmt_ci(vals(rr, 'm_form_peak'), 1e3, 0)} & {ptxt(p)}\\\\")
            body.append('\\addlinespace')
    write_table('tab_r1', 'Scenario / case & Controller & formation IAE (mm\\,s) & tracking IAE (mm\\,s) & peak formation (mm) & $p$',
                body[:-1], 'Heterogeneous sinusoidal delays of \\cite{peng2023} on the model of the theorems; mean [95\\,\\% bootstrap CI] over 30 seeds; $p$: Wilcoxon signed-rank test against the cascade (formation IAE), Holm-corrected.',
                'tab:r1', '@{}llcccc@{}')


# ---------------------------------------------------------------- R2
def r2():
    doc, ts, rs = rows('r2_realistic')
    # trajectories figure (seed 7, traction loss)
    tr = np.load(os.path.join(RES, 'r2_trajectories.npz'))
    AGV = ['#2a78d6', '#eb6834', '#1baf7a']
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw=dict(width_ratios=[1.25, 1.0]))
    for ax, sc, title in zip(axs, ('transport', 'warehouse'), ('(a) Transport manoeuvre (L-formation)', '(b) Warehouse route')):
        t, p, P = tr[f'{sc}/cascade/t'], tr[f'{sc}/cascade/p'], tr[f'{sc}/cascade/P']
        for i in range(3):
            ax.plot(P[:, i, 0], P[:, i, 1], color=GRID, lw=2.2, zorder=1)
        for i in range(3):
            ax.plot(p[:, i, 0], p[:, i, 1], color=AGV[i], lw=1.0, zorder=3, label=f'AGV {i + 1}')
        for tm in np.arange(0, t[-1], 6.0 if sc == 'transport' else 7.0):
            k = int(np.searchsorted(t, tm))
            for arr, col, ls in ((P[k], GREY_, (0, (3, 2))), (p[k], INK, '-')):
                tri = np.vstack([arr, arr[:1]])
                ax.plot(tri[:, 0], tri[:, 1], color=col, lw=0.6, ls=ls, zorder=2 if ls != '-' else 4)
        ax.set_aspect('equal'); ax.set_title(title, loc='left'); ax.set_xlabel('x (m)'); ax.grid(False)
    axs[0].set_ylabel('y (m)'); axs[0].legend(loc='upper left')
    save(fig, 'r2_trajectories')
    # relative error AGV1 - AGV2 with and without synchronisation (transport, seed 7, traction loss) + tracking errors
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 4.2), sharex=True)
    for typ in ('uncoupled', 'direct', 'cascade'):
        t, ev = tr[f'transport/{typ}/t'], tr[f'transport/{typ}/evec']
        d = ev[:, 0] - ev[:, 1]
        axs[0, 0].plot(t, 1e3 * d[:, 0], color=COL[typ], lw=1.0, label=SHORT[typ])
        axs[0, 1].plot(t, 1e3 * d[:, 1], color=COL[typ], lw=1.0, label=SHORT[typ])
    for ax, typ, ttl in ((axs[1, 0], 'cascade', '(c) Tracking errors, cascade (mm)'), (axs[1, 1], 'uncoupled', '(d) Tracking errors, no coupling (mm)')):
        t, ev = tr[f'transport/{typ}/t'], tr[f'transport/{typ}/evec']
        for i in range(3):
            ax.plot(t, 1e3 * np.linalg.norm(ev[:, i], axis=1), color=AGV[i], lw=0.9, label=f'AGV {i + 1}')
        ax.set_title(ttl, loc='left'); ax.set_xlabel('time (s)'); ax.set_yscale('log'); ax.set_ylim(1, 2000)
    axs[0, 0].set_title('(a) $x$-component of $e_1-e_2$ (mm)', loc='left'); axs[0, 1].set_title('(b) $y$-component of $e_1-e_2$ (mm)', loc='left')
    axs[0, 1].legend(loc='lower right'); axs[1, 0].legend(loc='upper right', ncol=3, fontsize=6.5); axs[1, 1].legend(loc='upper right', ncol=3, fontsize=6.5)
    ev0 = [e for e in doc['jobs'] if e['scen'] == 'transport' and e['seed'] == 7][0]
    save(fig, 'r2_relative')
    # tables
    for case, cols, name, cap in (
            ('nominal', [('pre_form', 'formation'), ('pre_form_est', 'formation, as estimated'), ('pre_track', 'tracking'), ('pre_loc', 'localisation')],
             'tab_r2_nominal', 'Nominal operation on the high-fidelity model: RMS errors in mm for $t\\ge15$\\,s, mean [95\\,\\% CI] over 30 seeds; $p$ for the estimated formation error against the cascade, Holm-corrected.'),
            ('event', [('ev_peak', 'peak formation'), ('ev_iae', 'formation IAE'), ('ev_track_iae', 'tracking IAE'), ('ev_recovery', 'recovery (s)')],
             'tab_r2_event', 'Traction loss on the high-fidelity model (15\\,s window from its onset): peak in mm, IAE in mm\\,s, recovery time in s from the end of the traction loss until the 1\\,s average formation error stays below 10\\,cm (mean over the runs that recover before the end of the run; one $\\sigma$-leak run on the warehouse route does not); mean [95\\,\\% CI] over 30 seeds; $p$ for the formation IAE against the cascade, Holm-corrected.')):
        body = []
        for sc in ('transport', 'warehouse'):
            base = {r['seed']: r for r in sel(rs, scen=sc, case=case, type='cascade')}
            key = cols[1][0]
            ps = {typ: paired_test([base[r['seed']][key] for r in sel(rs, scen=sc, case=case, type=typ)],
                                   [r[key] for r in sel(rs, scen=sc, case=case, type=typ)]) for typ in CONTROLLERS if typ != 'cascade'}
            padj = dict(zip(ps.keys(), holm([v[0] for v in ps.values()])))
            for typ in ['cascade', 'direct', 'sigleak', 'uncoupled']:
                rr = sel(rs, scen=sc, case=case, type=typ)
                cells = []
                for k, _ in cols:
                    sc_ = 1.0 if k == 'ev_recovery' else 1e3
                    cells.append(fmt_ci(vals(rr, k), sc_, 1 if k == 'ev_recovery' else 1))
                body.append(f"{sc if typ == 'cascade' else ''} & {SHORT[typ]} & " + ' & '.join(cells) + f" & {ptxt(padj.get(typ, np.nan))}\\\\")
            body.append('\\addlinespace')
        write_table(name, 'Scenario & Controller & ' + ' & '.join(c[1] for c in cols) + ' & $p$', body[:-1], cap,
                    'tab:' + name[4:], '@{}ll' + 'c' * (len(cols) + 1) + '@{}')
    SUMMARY['r2'] = {f'{sc}/{case}/{typ}': {k: float(np.nanmean(vals(sel(rs, scen=sc, case=case, type=typ), k)))
                                            for k in ('pre_form', 'pre_track', 'pre_loc', 'ev_peak', 'ev_iae', 'ev_track_iae', 'ev_recovery', 'pre_form_est')}
                     for sc in ('transport', 'warehouse') for case in ('nominal', 'event') for typ in CONTROLLERS}


GREY_ = '#8b8a83'


# ---------------------------------------------------------------- R3
def r3():
    doc, ts, rs = rows('r3_network')
    panels = [('latency', 'mean latency (s)', True), ('loss', 'packet loss probability', False),
              ('outage', 'outage length (s), no watchdog', False), ('outage_wd', 'outage length (s), 0.5-s watchdog', False),
              ('rate', 'broadcast rate (Hz)', True), ('sinus', 'scale of the sinusoidal delays', True)]
    base = sel(rs, type='uncoupled')
    bmed = np.nanmedian(vals(base, 'ev_iae')); bq = np.nanquantile(vals(base, 'ev_iae'), [0.25, 0.75])
    for metric, ylab, fname in (('ev_iae', 'formation IAE (m s)', 'r3_network'),
                                ('ev_track_iae', 'tracking IAE (m s)', 'r3_network_tracking')):
        bmed = np.nanmedian(vals(base, metric)); bq = np.nanquantile(vals(base, metric), [0.25, 0.75])
        fig, axs = plt.subplots(2, 3, figsize=(7.0, 4.3), sharey=True)
        for ax, (fac, xl, logx) in zip(axs.ravel(), panels):
            ax.axhspan(bq[0], bq[1], color=COL['uncoupled'], alpha=0.15, lw=0)
            ax.axhline(bmed, color=COL['uncoupled'], lw=1.0, label=SHORT['uncoupled'])
            for typ in ('sigleak', 'direct', 'cascade'):
                rr = sel(rs, factor=fac, type=typ)
                if not rr:
                    continue
                lv = sorted(set(r['level'] for r in rr))
                med = [np.nanmedian(vals(sel(rr, level=l), metric)) for l in lv]
                q = np.array([np.nanquantile(vals(sel(rr, level=l), metric), [0.25, 0.75]) for l in lv])
                ax.errorbar(lv, med, yerr=[np.array(med) - q[:, 0], q[:, 1] - np.array(med)], color=COL[typ], marker=MK[typ], ms=3.5,
                            mec='white', mew=0.4, capsize=1.5, lw=1.1, label=SHORT[typ])
            if logx:
                ax.set_xscale('log')
                lvs = sorted(set(r['level'] for r in sel(rs, factor=fac)))
                ax.set_xticks(lvs); ax.set_xticklabels([f'{v:g}' for v in lvs]); ax.minorticks_off()
            ax.set_xlabel(xl)
        for ax in axs[:, 0]:
            ax.set_ylabel(ylab)
        axs[0, 0].legend(loc='upper left', ncol=1)
        fig.tight_layout(h_pad=1.0, w_pad=0.6)
        save(fig, fname)
    SUMMARY['r3'] = {f'{f}/{typ}/{l}': {k: float(np.nanmedian(vals(sel(rs, factor=f, type=typ, level=l), k))) for k in ('ev_iae', 'ev_track_iae', 'ev_peak', 'pre_form', 'pre_track', 'm_failed')}
                     for f, _, _ in panels for typ in ('cascade', 'direct') for l in sorted(set(r['level'] for r in sel(rs, factor=f)))}
    SUMMARY['r3_failed'] = {f'{f}/{typ}/{l}': int(np.nansum(vals(sel(rs, factor=f, type=typ, level=l), 'm_failed')))
                            for f, _, _ in panels for typ in ('cascade', 'direct') for l in sorted(set(r['level'] for r in sel(rs, factor=f)))}
    SUMMARY['r3']['uncoupled'] = {k: float(np.nanmedian(vals(base, k))) for k in ('ev_iae', 'ev_track_iae', 'ev_peak', 'pre_form', 'pre_track')}


# ---------------------------------------------------------------- R4
FACT = [('loc_noise', 'localisation noise (m)', 1), ('loc_bias', 'localisation bias (m)', 1),
        ('loc_latency', 'localisation latency (s)', 1), ('loc_rate', 'localisation rate (Hz)', 1),
        ('calibration', 'radius/track error (rel.)', 1), ('slip', 'slip intensity ($\\times$ nominal)', 1),
        ('payload', 'payload (kg)', 1), ('act_delay', 'actuation delay (s)', 1), ('ctrl_rate', 'control rate (Hz)', 1),
        ('clock', 'clock offset bound (s)', 1), ('wheel_limit', 'wheel-speed limit (m/s)', 1)]


NOMINAL = {'loc_noise': 0.01, 'loc_bias': 0.01, 'loc_latency': 0.05, 'loc_rate': 20.0, 'calibration': 0.01, 'slip': 1.0,
           'payload': 50.0, 'act_delay': 0.01, 'ctrl_rate': 50.0, 'clock': 0.005, 'wheel_limit': 1.5}


def r4():
    doc, ts, rs = rows('r4_simreal')
    for metric, ylab, fname, scale in (('pre_form', 'formation RMS (mm)', 'r4_simreal_steady', 1e3),
                                       ('ev_iae', 'formation IAE (m s)', 'r4_simreal_event', 1.0)):
        fig, axs = plt.subplots(3, 4, figsize=(7.0, 5.6))
        for ax, (fac, xl, _) in zip(axs.ravel(), FACT):
            for typ in ('uncoupled', 'sigleak', 'direct', 'cascade'):
                rr = sel(rs, factor=fac, type=typ)
                if not rr:
                    continue
                lv = sorted(set(r['level'] for r in rr))
                med = [scale * np.nanmedian(vals(sel(rr, level=l), metric)) for l in lv]
                ax.plot(lv, med, color=COL[typ], marker=MK[typ], ms=3, mec='white', mew=0.4, lw=1.0, label=SHORT[typ])
            ax.axvline(NOMINAL[fac], color=AXIS, lw=0.8, ls=(0, (2, 2)), zorder=0)
            ax.set_xlabel(xl, fontsize=6.8)
        axs.ravel()[-1].axis('off')
        h, l = axs[0, 0].get_legend_handles_labels()
        axs.ravel()[-1].legend(h, l, loc='center')
        for ax in axs[:, 0]:
            ax.set_ylabel(ylab, fontsize=6.8)
        fig.tight_layout(h_pad=0.8, w_pad=0.5)
        save(fig, fname)
    SUMMARY['r4'] = {f'{f}/{typ}/{l}': {k: float(np.nanmedian(vals(sel(rs, factor=f, type=typ, level=l), k))) for k in ('pre_form', 'pre_track', 'ev_iae', 'pre_loc')}
                     for f, _, _ in FACT for typ in CONTROLLERS for l in sorted(set(r['level'] for r in sel(rs, factor=f)))}


# ---------------------------------------------------------------- R5
def r5():
    doc, ts, rs = rows('r5_domain_rand')
    seeds = sorted(set(r['seed'] for r in rs))
    by = {(r['seed'], r['type']): r for r in rs}
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.4))
    body = []
    for typ in ('cascade', 'direct', 'sigleak'):
        fr = np.array([by[(s, typ)]['m_form_iae'] / by[(s, 'uncoupled')]['m_form_iae'] for s in seeds
                       if not by[(s, typ)]['m_failed'] and not by[(s, 'uncoupled')]['m_failed']])
        tr = np.array([by[(s, typ)]['m_track_iae'] / by[(s, 'uncoupled')]['m_track_iae'] for s in seeds
                       if not by[(s, typ)]['m_failed'] and not by[(s, 'uncoupled')]['m_failed']])
        for ax, x in ((axs[0], fr), (axs[1], tr)):
            xs = np.sort(x); ax.step(xs, np.arange(1, len(xs) + 1) / len(xs), where='post', color=COL[typ], label=SHORT[typ])
        nf = sum(by[(s, typ)]['m_failed'] for s in seeds)
        body.append(f"{SHORT[typ]} & {np.median(fr):.3f} & {100 * np.mean(fr < 1):.0f} & {np.quantile(fr, 0.95):.2f} & {np.median(tr):.3f} & {100 * np.mean(tr < 1):.0f} & {nf}\\\\")
    nf0 = sum(by[(s, 'uncoupled')]['m_failed'] for s in seeds)
    body.append(f"{SHORT['uncoupled']} & 1 & -- & -- & 1 & -- & {nf0}\\\\")
    for typ in ('uncoupled', 'sigleak', 'direct', 'cascade'):
        pk = np.sort([by[(s, typ)]['m_form_peak'] for s in seeds if not by[(s, typ)]['m_failed']])
        axs[2].step(pk, np.arange(1, len(pk) + 1) / len(pk), where='post', color=COL[typ], label=SHORT[typ])
    for ax in axs[:2]:
        ax.axvline(1, color=INK2, lw=0.7, ls=(0, (3, 2))); ax.set_xscale('log')
    axs[0].set_xlabel('formation IAE / no coupling'); axs[1].set_xlabel('tracking IAE / no coupling'); axs[2].set_xlabel('peak formation error (m)')
    axs[2].set_xscale('log')
    axs[0].set_ylabel('fraction of runs'); axs[0].legend(loc='upper left'); axs[2].legend(loc='lower right')
    for ax, ttl in zip(axs, ('(a)', '(b)', '(c)')):
        ax.set_title(ttl, loc='left')
    fig.tight_layout(w_pad=0.8)
    save(fig, 'r5_domain_rand')
    write_table('tab_r5', 'Controller & \\multicolumn{3}{c}{formation IAE ratio} & \\multicolumn{2}{c}{tracking IAE ratio} & failed\\\\'
                '\\cmidrule(lr){2-4}\\cmidrule(lr){5-6} & median & better (\\%) & 95th pct. & median & better (\\%) &',
                body, f'Domain-randomised Monte Carlo, {len(seeds)} parameter draws; ratios to the uncoupled law on the same draw (paired); runs that failed are excluded from the ratios.',
                'tab:r5', '@{}lcccccc@{}')
    def rms_fail_inf(typ):
        return np.array([np.inf if by[(s, typ)]['m_failed'] else by[(s, typ)]['run_form'] for s in seeds])
    SUMMARY['r5'] = {typ: dict(form_rms_median=float(np.median(rms_fail_inf(typ))),
                               form_rms_p90=float(np.quantile(rms_fail_inf(typ), 0.9, method='higher')),
                               form_ratio_max=float(np.max([by[(s, typ)]['m_form_iae'] / by[(s, 'uncoupled')]['m_form_iae'] for s in seeds if not by[(s, typ)]['m_failed']])),
                               failed=int(sum(by[(s, typ)]['m_failed'] for s in seeds))) for typ in CONTROLLERS}


# ---------------------------------------------------------------- R6
def r6():
    doc, ts, rs = rows('r6_events')
    cases = [('traction', '(a) Traction loss of AGV 2 (40 % for 2 s)', 'form'), ('stop', '(b) Safety stop of AGV 2 (3 s)', 'form'),
             ('reloc', '(c) Relocalisation jump of AGV 2 (10 cm)', 'form'), ('switching', '(d) Traction loss with switching links', 'form'),
             ('linkfail', '(e) All links from AGV 1 lost at 20 s', 'track'), ('linkfail_wd', '(f) as (e), with 0.5 s watchdog', 'track')]
    fig, axs = plt.subplots(2, 3, figsize=(7.0, 4.3), sharex=False)
    for ax, (case, ttl, key) in zip(axs.ravel(), cases):
        for typ in ('uncoupled', 'sigleak', 'direct', 'cascade'):
            if case == 'linkfail_wd' and typ == 'uncoupled':
                rr = sel(rs, case='linkfail', type='uncoupled')
            else:
                rr = sel(rs, case=case, type=typ)
            if not rr:
                continue
            t, Y = stack_ts(ts, rr, key)
            band(ax, t, Y, COL[typ], SHORT[typ], lw=1.1, alpha=0.15)
        ax.set_title(ttl, loc='left', fontsize=7.2)
        ax.set_xlim((25, 45) if not case.startswith('linkfail') else (15, 60))
        ax.set_ylabel('formation error (m)' if key == 'form' else 'mean tracking error (m)', fontsize=7)
        ax.set_xlabel('time (s)')
    axs[0, 0].legend(loc='upper right')
    fig.tight_layout(h_pad=1.0, w_pad=0.6)
    save(fig, 'r6_events')
    body = []
    for case, ttl, key in cases[:4]:
        for typ in ['cascade', 'direct', 'sigleak', 'uncoupled']:
            rr = sel(rs, case=case, type=typ)
            body.append(f"{EVN[case] if typ == 'cascade' else ''} & {SHORT[typ]} & {fmt_ci(vals(rr, 'ev_peak'), 1e3, 0)} & {fmt_ci(vals(rr, 'ev_iae'), 1e3, 0)} & "
                        f"{fmt_ci(vals(rr, 'ev_track_iae'), 1e3, 0)} & {fmt_ci(vals(rr, 'ev_recovery'), 1, 2)}\\\\")
        body.append('\\addlinespace')
    for case in ('linkfail', 'linkfail_wd'):
        for typ in ['cascade', 'direct', 'sigleak', 'uncoupled']:
            rr = sel(rs, case=case, type=typ)
            if not rr:
                continue
            trk = [window_rms(ts[str(r['idx']) + '/t'], ts[str(r['idx']) + '/track'], 30.0, 60.0) for r in rr]
            lab = EVN[case] if typ == 'cascade' else ''
            body.append(f"{lab} & {SHORT[typ]} & " + '\\multicolumn{2}{c}{tracking RMS after 30\\,s: ' + fmt_ci(trk, 1e3, 1) + ' mm} & & \\\\')
        body.append('\\addlinespace')
    write_table('tab_r6', 'Event & Controller & peak formation (mm) & formation IAE (mm\\,s) & tracking IAE (mm\\,s) & recovery (s)',
                body[:-1], 'Disturbance events on the high-fidelity model (warehouse route, 20 seeds, mean [95\\,\\% CI]). Recovery: mean over the runs that recover before the end of the run; the $\\sigma$-leak filter does not recover in 5 of 20 runs after the traction loss and in 12 of 20 runs with switching links.', 'tab:r6', '@{}llcccc@{}')
    SUMMARY['r6'] = {f'{c}/{typ}': {k: float(np.nanmean(vals(sel(rs, case=c, type=typ), k))) for k in ('ev_peak', 'ev_iae', 'ev_track_iae', 'ev_recovery')}
                     for c, _, _ in cases for typ in CONTROLLERS}


# ---------------------------------------------------------------- R7
def r7():
    doc, ts, rs = rows('r7_scaling')
    Ns = [3, 5, 8, 12]
    GNAME = {'complete': 'complete', 'ring': 'ring', 'dring': 'directed ring', 'chain': 'directed chain', 'star': 'star'}
    GCOL = {'complete': RAMP[4], 'ring': RAMP[3], 'dring': RAMP[2], 'chain': RAMP[1], 'star': RAMP[0]}
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.5))
    for ax, metric, ylab in ((axs[0], 'ev_iae', 'formation IAE, event (m s)'), (axs[1], 'pre_form', 'formation RMS, steady (m)'),
                             (axs[2], 'ev_track_iae', 'tracking IAE, event (m s)')):
        b = [np.nanmedian(vals(sel(rs, N=N, graph='none'), metric)) for N in Ns]
        ax.plot(Ns, b, color=COL['uncoupled'], marker='D', ms=3, lw=1.0, label='no coupling')
        for g in ('complete', 'ring', 'dring', 'chain', 'star'):
            for typ, ls in (('cascade', '-'), ('direct', (0, (3, 2)))):
                y = [np.nanmedian(vals(sel(rs, N=N, graph=g, type=typ), metric)) for N in Ns]
                ax.plot(Ns, y, color=GCOL[g], ls=ls, marker='o' if typ == 'cascade' else 's', ms=2.8, lw=1.0,
                        label=(GNAME[g] if typ == 'cascade' else None))
        ax.set_xlabel('number of AGVs'); ax.set_ylabel(ylab, fontsize=7); ax.set_xticks(Ns)
    axs[0].set_ylim(0.85, 2.15); axs[1].set_ylim(0.0, 0.11); axs[2].set_ylim(0.2, 1.25)
    # failures of the direct coupling on the complete graph are stated in the caption
    axs[0].legend(loc='upper left', fontsize=6.0, title='Cascade (solid), Direct (dashed)', title_fontsize=6.0)
    fig.tight_layout(w_pad=0.8)
    save(fig, 'r7_scaling')
    SUMMARY['r7'] = {f'{N}/{g}/{typ}': {k: float(np.nanmedian(vals(sel(rs, N=N, graph=g, type=typ), k))) for k in ('ev_iae', 'pre_form', 'ev_track_iae')}
                     for N in Ns for g in ('complete', 'ring', 'dring', 'chain', 'star') for typ in ('cascade', 'direct')}
    SUMMARY['r7'].update({f'{N}/none': {k: float(np.nanmedian(vals(sel(rs, N=N, graph='none'), k))) for k in ('ev_iae', 'pre_form', 'ev_track_iae')} for N in Ns})
    SUMMARY['r7_failed'] = int(sum(r['m_failed'] for r in rs))


# ---------------------------------------------------------------- R7b
def r7b():
    doc, ts, rs = rows('r7b_normalised')
    d7, t7, r7s = rows('r7_scaling')
    body = []
    out = {}
    for N in (3, 5, 8, 12):
        base = sel(r7s, N=N, graph='none')
        bf, bi = np.nanmedian(vals(base, 'run_form')), np.nanmedian(vals(base, 'm_form_iae'))
        cells = []
        for typ in ('cascade', 'direct'):
            un = sel(r7s, N=N, graph='complete', type=typ)
            nm = sel(rs, N=N, type=typ)
            fu, fn = int(np.nansum(vals(un, 'm_failed'))), int(np.nansum(vals(nm, 'm_failed')))
            fr_u = np.nanmedian(vals([r for r in un if not r['m_failed']], 'run_form')) if fu < len(un) else np.nan
            fr_n = np.nanmedian(vals([r for r in nm if not r['m_failed']], 'run_form'))
            cells += [f"{fu}/{len(un)}", ('--' if not np.isfinite(fr_u) else f"{1e3 * fr_u:.0f}"), f"{fn}/{len(nm)}", f"{1e3 * fr_n:.0f}"]
            out[f'{N}/{typ}'] = dict(failed_unit=fu, failed_norm=fn, form_rms_unit=float(fr_u), form_rms_norm=float(fr_n))
        out[f'{N}/none'] = dict(form_rms=float(bf), form_iae=float(bi))
        body.append(f"{N} & {1e3 * bf:.0f} & " + ' & '.join(cells) + '\\\\')
    write_table('tab_r7b', '$N$ & no coupling & \\multicolumn{4}{c}{cascade} & \\multicolumn{4}{c}{direct}\\\\'
                '\\cmidrule(lr){3-6}\\cmidrule(lr){7-10} & RMS & failed, unit & RMS & failed, norm. & RMS & failed, unit & RMS & failed, norm. & RMS',
                body, 'Complete graphs with unit weights (R7) and with degree-normalised weights $a_{ij}=1/(N-1)$ (R7b): failed runs and median RMS of the formation error (largest pair) in mm over $t\\ge15$\\,s, traction loss included, over the runs that did not fail; 6 seeds.',
                'tab:r7b', '@{}r' + 'c' * 9 + '@{}')
    SUMMARY['r7b'] = out


# ---------------------------------------------------------------- R8
def r8():
    doc, ts, rs = rows('r8_sensitivity')
    pars = [('lam', 'leak $\\lambda$ (s$^{-1}$)', True), ('gam', 'filter bandwidth $\\gamma$ (s$^{-1}$)', True), ('rho', 'shift limit $\\rho$', False), ('ebar', 'cap $\\bar e$ (m)', True)]
    fig, axs = plt.subplots(1, 4, figsize=(7.0, 2.0), sharey=True)
    for ax, (par, xl, logx) in zip(axs, pars):
        rr = sel(rs, factor=par)
        lv = sorted(set(r['level'] for r in rr))
        xs = list(range(len(lv))) if par == 'ebar' else lv          # categorical axis for the cap (includes 'no cap')
        for metric, col, lab in (('ev_iae', COL['cascade'], 'formation IAE'), ('ev_track_iae', INK2, 'tracking IAE')):
            med = [np.nanmedian(vals(sel(rr, level=l), metric)) for l in lv]
            q = np.array([np.nanquantile(vals(sel(rr, level=l), metric), [0.25, 0.75]) for l in lv])
            ax.errorbar(xs, med, yerr=[np.array(med) - q[:, 0], q[:, 1] - np.array(med)], color=col, marker='o', ms=3, capsize=1.5, lw=1.0, label=lab)
        if logx and par != 'ebar':
            ax.set_xscale('log')
        ax.set_xticks(xs); ax.set_xticklabels(['no cap' if v >= 100 else f'{v:g}' for v in lv]); ax.minorticks_off()
        ax.set_xlabel(xl)
    axs[0].set_ylabel('IAE after the event (m s)'); axs[0].legend(loc='upper right')
    fig.tight_layout(w_pad=0.5)
    save(fig, 'r8_sensitivity')
    SUMMARY['r8'] = {f'{p}/{l}': {k: float(np.nanmedian(vals(sel(rs, factor=p, level=l), k))) for k in ('ev_iae', 'ev_track_iae', 'pre_form')}
                     for p, _, _ in pars for l in sorted(set(r['level'] for r in sel(rs, factor=p)))}


# ---------------------------------------------------------------- R9
def r9():
    doc, ts, rs = rows('r9_numerics')
    groups = {'run': ['m_form_iae', 'm_track_iae', 'm_form_rms', 'm_track_rms'], 'event': ['ev_iae', 'ev_peak', 'ev_track_iae'],
              'steady': ['pre_form', 'pre_track']}
    out = {}
    for typ in ('cascade', 'direct'):
        ref = {r['seed']: r for r in sel(rs, type=typ, dt=0.0005)}
        for dt in (0.001, 0.002, 0.005, 0.01):
            for g, keys in groups.items():
                d = [max(abs(r[k] - ref[r['seed']][k]) / abs(ref[r['seed']][k]) for k in keys) for r in sel(rs, type=typ, dt=dt)]
                out[f'{typ}/{dt}/{g}'] = float(np.max(d))
    rr = json.load(open(os.path.join(RES, 'r9_rerun.json')))
    out['identical_reruns'] = f"{sum(rr['identical'])}/{len(rr['identical'])}"
    out['code_hash'] = doc['environment']['code_hash']
    SUMMARY['r9'] = out


def walltime():
    tot, runs = 0.0, 0
    for n in ('r0_tuning', 'r0b_direct_divergence', 'r1_paper_delays', 'r2_realistic', 'r3_network', 'r4_simreal', 'r5_domain_rand', 'r6_events', 'r7_scaling', 'r7b_normalised', 'r8_sensitivity', 'r9_numerics'):
        p = os.path.join(RES, n + '.json')
        if os.path.exists(p):
            d = json.load(open(p)); tot += d['wall_s']; runs += d['n_jobs']
            SUMMARY.setdefault('environment', d['environment'])
    SUMMARY['total_runs'] = runs; SUMMARY['total_wall_s'] = tot


# time series each figure function needs besides results/robust/<experiment>.json
NEEDS = {'r0': ['r0_tuning_summary.json'], 'r1': ['r1_paper_delays_ts.npz'],
         'r2': ['r2_realistic_ts.npz', 'r2_trajectories.npz'], 'r3': ['r3_network_ts.npz'], 'r4': ['r4_simreal_ts.npz'],
         'r5': ['r5_domain_rand_ts.npz'], 'r6': ['r6_events_ts.npz'], 'r7': ['r7_scaling_ts.npz'],
         'r7b': ['r7b_normalised_ts.npz', 'r7_scaling_ts.npz'], 'r8': ['r8_sensitivity_ts.npz'], 'r9': ['r9_numerics_ts.npz']}


if __name__ == '__main__':
    which = [w for w in sys.argv[1:] if w in NEEDS] or ['r0', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r7b', 'r8', 'r9']
    names = {'r0': 'r0_tuning', 'r1': 'r1_paper_delays', 'r2': 'r2_realistic', 'r3': 'r3_network', 'r4': 'r4_simreal',
             'r5': 'r5_domain_rand', 'r6': 'r6_events', 'r7': 'r7_scaling', 'r7b': 'r7b_normalised', 'r8': 'r8_sensitivity', 'r9': 'r9_numerics'}
    p = os.path.join(RES, 'summary.json')
    if os.path.exists(p):
        SUMMARY.update(json.load(open(p)))
    for w in which:
        missing = [f for f in [names[w] + '.json'] + NEEDS[w] if not os.path.exists(os.path.join(RES, f))]
        if missing:                                  # never derive numbers from incomplete data: keep the stored summary
            print('skip', w, '(missing ' + ', '.join(missing) + ')')
            continue
        globals()[w]()
        print('done', w)
    walltime()
    json.dump(SUMMARY, open(p, 'w'), indent=1, default=float)
