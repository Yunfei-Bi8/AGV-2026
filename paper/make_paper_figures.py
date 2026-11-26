"""Figures of the paper (conference version and extended version, arXiv v2) from the stored results.

    python3 paper/make_paper_figures.py            # all figures  -> paper/figures/fig_*.pdf
    python3 paper/make_paper_figures.py kin hifi   # selected figures

Needs only the files in results/ and results/robust/ (results and time series are part of the repository); nothing is
simulated. The figure functions of experiments/make_figures.py and experiments/robust/make_figures_robust.py are reused
in the style of the paper; Figs. 2 and 3 and the figures of R0, R4, R6 and R7 are drawn here. Broadcast rates, fix
rates and fix latencies are plotted at the values realized on the control grid (agvsim.hifi.realized_timing)."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, 'experiments'), os.path.join(ROOT, 'experiments', 'robust')):
    if p not in sys.path:
        sys.path.insert(0, p)
OUT = os.path.join(ROOT, 'paper', 'figures')
os.makedirs(OUT, exist_ok=True)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.text as mtext
from matplotlib.figure import Figure

import make_figures as mf                 # verification study (kinematic model)
import make_figures_robust as mfr         # robustness study (high-fidelity model)
from agvsim import hifi as hf

RES = os.path.join(ROOT, 'results') + '/'
RESR = os.path.join(ROOT, 'results', 'robust')

INK, INK2, GRID, AXIS = '#0b0b0b', '#3d3c37', '#e4e3dc', '#9c9b92'
# controller: colour, line style, label, marker (colour and line style both encode the controller)
CTRL = {'cascade': ('#2a78d6', '-', 'Cascade', 'o'), 'direct': ('#eb6834', (0, (5, 1.6)), 'Direct coupling', 's'),
        'sigleak': ('#1baf7a', (0, (4, 1.5, 1, 1.5)), '$\\sigma$-leak', '^'),
        'uncoupled': ('#8b8a83', (0, (1.2, 1.2)), 'No coupling', 'D')}
# networks of the kinematic study: one hue, light to dark with increasing latency
NETS = [('nominal', '60 ms network', '#3f86dc', '-'), ('slow1', '1 s network', '#1c5cab', (0, (5, 1.6))),
        ('slow3', '3 s network', '#0b2f5e', (0, (3, 1.2, 1, 1.2)))]
STYLE = {
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'axes.titleweight': 'normal',
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.8, 'axes.linewidth': 0.6,
    'axes.edgecolor': AXIS, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
    'xtick.major.size': 2.2, 'ytick.major.size': 2.2, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.minor.size': 1.2, 'ytick.minor.size': 1.2, 'lines.linewidth': 1.0,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.4, 'axes.axisbelow': True,
    'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False, 'legend.handlelength': 2.2,
    'legend.borderaxespad': 0.2, 'legend.labelspacing': 0.25, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.015}

# one spelling and one term per concept in all figure text
TEXT = [('relocalisation', 'relocalization'), ('Relocalisation', 'Relocalization'), ('localisation', 'localization'),
        ('Localisation', 'Localization'), ('randomisation', 'randomization'), ('randomised', 'randomized'),
        ('normalised', 'normalized'), ('synchronisation', 'synchronization'), ('manoeuvre', 'maneuver'),
        ('leak $\\lambda$', 'restoring gain $\\lambda$'), ('cap $\\bar e$', 'bound $\\bar e$'), ('no cap', 'no bound'),
        ('cascade, nominal network', 'cascade, 60 ms network'), (' %', '%'), ('(m s)', '(m$\\cdot$s)')]
_set_text = mtext.Text.set_text


def _set_text_paper(self, s):
    if isinstance(s, str):
        for a, b in TEXT:
            s = s.replace(a, b)
    return _set_text(self, s)


mtext.Text.set_text = _set_text_paper


def capfirst(s):
    """Sentence case for axis labels; single-letter variables (x, y) and math stay as they are."""
    w = s.split(' ')[0] if s else ''
    return s[:1].upper() + s[1:] if len(w) > 1 and w[0].isalpha() and w[0].islower() else s


_savefig = Figure.savefig


def _savefig_paper(self, fname, *a, **k):
    for ax in self.axes:
        ax.set_xlabel(capfirst(ax.get_xlabel()))
        ax.set_ylabel(capfirst(ax.get_ylabel()))
    return _savefig(self, fname, *a, **k)


Figure.savefig = _savefig_paper


def save(fig, name):
    if name == 'r5_domain_rand':                       # legend of (a) away from the curves
        h, l = fig.axes[0].get_legend_handles_labels()
        fig.axes[0].legend(h, l, loc='lower right')
    fig.savefig(os.path.join(OUT, 'fig_' + name + '.pdf'))
    plt.close(fig)


def tag(ax, s):
    ax.text(-0.02, 1.03, s, transform=ax.transAxes, ha='right', va='bottom', fontsize=8, color=INK)


# ---------------------------------------------------------------- reuse of the study's figure functions
mf.FIG = OUT + '/'
mfr.save = save
mfr.FIG = OUT
mfr.write_table = lambda *a, **k: None           # tables: paper/make_paper_tables.py
_rows = mfr.rows


def rows_realized(name):
    """Rows of an experiment with timing levels replaced by the values realized on the 50 Hz control grid."""
    doc, ts, rs = _rows(name)
    for r in rs:
        f = r.get('factor')
        if name == 'r3_network' and f == 'rate':
            r['level'] = hf.realized_timing(50.0, r['level'])['fb']
        elif name == 'r4_simreal' and f == 'loc_rate':
            r['level'] = hf.realized_timing(50.0, 20.0, f_fix=r['level'])['f_fix']
        elif name == 'r4_simreal' and f == 'loc_latency':
            r['level'] = hf.realized_timing(50.0, 20.0, d_fix=r['level'])['d_fix']
    return doc, ts, rs


mfr.rows = rows_realized


# ---------------------------------------------------------------- figures drawn here
def load_e3(net, graph='complete'):
    return np.load(RES + f'e3_figure8_{net}_{graph}.npz')


def fig_kin():
    """Fig. 2: (a) vehicle Lyapunov function, (b) tracking error (figure-8, seed 7), (c) steady disagreement ratio
    against the factor of Proposition 1 (persistent-error runs of E5)."""
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 1.45), gridspec_kw=dict(wspace=0.36))
    a1, a2, a3 = axs
    for net, lab, col, ls in NETS:
        d = load_e3(net)
        a1.semilogy(d['t'], d['W'].max(1), color=col, ls=ls, label=lab)
        a2.semilogy(d['t'], np.linalg.norm(d['e'], axis=2).max(1), color=col, ls=ls, label=lab)
    dn = load_e3('nominal', 'none')
    a2.semilogy(dn['t'], np.linalg.norm(dn['e'], axis=2).max(1), color=CTRL['uncoupled'][0], ls=CTRL['uncoupled'][1],
                lw=1.1, label='no coupling')
    a1.set_ylim(1e-26, 10); a1.set_yticks([1e-24, 1e-16, 1e-8, 1]); a1.minorticks_off()   # rounding level below
    a2.set_ylim(1e-15, 5); a2.set_yticks([1e-15, 1e-10, 1e-5, 1]); a2.minorticks_off()
    for a in (a1, a2):
        a.set_xlim(0, 75); a.set_xlabel('Time (s)')
    a1.set_ylabel(r'$\max_i W_i$ (m$^2$)')
    a2.set_ylabel(r'$\max_i |e_i|$ (m)')
    a1.legend(loc='upper right')
    h, l = a2.get_legend_handles_labels()
    a2.legend(h[-1:], l[-1:], loc='upper right')
    d = json.load(open(RES + 'e5_formation.json'))
    st = [r for r in d['static'] if 'disagreement_ratio' in r and r['Lam'] > 0]
    for gname, mk, glab in (('complete3', 'o', 'complete, $N$=3'), ('path4', 's', 'path, $N$=4')):
        for net, col, nlab in (('nominal', NETS[0][2], '60 ms'), ('slow3', NETS[2][2], '3 s')):
            rr = [r for r in st if r['graph'] == gname and r['net'] == net]
            sty = dict(ms=3.6, color=col, mec='white', mew=0.4) if net == 'nominal' else \
                dict(ms=6.4, mfc='none', mec=col, mew=0.9)
            a3.loglog([r['predicted_factor'] for r in rr], [r['disagreement_ratio'] for r in rr], mk, ls='none',
                      label=f'{glab}, {nlab}', zorder=3 if net == 'nominal' else 2, **sty)
    x = np.array([0.02, 1.0])
    a3.loglog(x, x, color=INK2, lw=0.7, ls=(0, (3, 2)), zorder=0)
    a3.text(0.12, 0.05, 'equality', color=INK2, fontsize=6.8)
    a3.set_xlabel(r'Predicted factor $\lambda/(\lambda+\Lambda\lambda_2(L))$')
    a3.set_ylabel('Measured ratio')
    a3.legend(loc='upper left', handlelength=1.0, handletextpad=0.3)
    a3.set_xlim(0.02, 1.2); a3.set_ylim(0.02, 1.2)
    for a, s in zip(axs, ('(a)', '(b)', '(c)')):
        tag(a, s)
    fig.savefig(os.path.join(OUT, 'fig_kin.pdf'))
    plt.close(fig)


def fig_hifi():
    """Fig. 3: (a) traction loss, (b) safety stop of AGV 2 at 30 s (R6, 20 seeds, median and interquartile range);
    (c) formation IAE after the traction loss relative to no coupling against the mean latency (R3, medians)."""
    doc, ts, rs = mfr.rows('r6_events')
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 1.45), gridspec_kw=dict(wspace=0.32))
    for ax, case, span, xl in ((axs[0], 'traction', (30.0, 32.0), (27, 38)), (axs[1], 'stop', (30.0, 33.0), (27, 45))):
        ax.axvspan(*span, color='#efeee8', lw=0, zorder=0)
        for typ in ('uncoupled', 'direct', 'cascade'):
            col, ls, lab, _ = CTRL[typ]
            t, Y = mfr.stack_ts(ts, mfr.sel(rs, case=case, type=typ), 'form')
            m = (t >= xl[0]) & (t <= xl[1])
            q1, med, q3 = np.nanquantile(Y[:, m], [0.25, 0.5, 0.75], axis=0)
            ax.fill_between(t[m], q1, q3, color=col, alpha=0.28, lw=0)
            ax.plot(t[m], med, color=col, ls=ls, label=lab, lw=1.05)
        ax.set_xlim(*xl); ax.set_ylim(0, None)
        ax.set_xlabel('Time (s)'); ax.set_ylabel('Formation error (m)')
    axs[1].legend(loc='upper right', handlelength=2.0)
    S = json.load(open(os.path.join(RESR, 'summary.json')))
    r3 = S['r3']; base = r3['uncoupled']['ev_iae']
    lv = [0.01, 0.1, 0.3, 1.0, 2.0, 4.0]
    a3 = axs[2]
    a3.axhline(1.0, color=CTRL['uncoupled'][0], ls=CTRL['uncoupled'][1], lw=1.1, label='No coupling')
    for typ in ('direct', 'cascade'):
        col, ls, lab, mk = CTRL[typ]
        a3.semilogx(lv, [r3[f'latency/{typ}/{l}']['ev_iae'] / base for l in lv], color=col, ls=ls, marker=mk, ms=3.6,
                    mec='white', mew=0.5, label=lab)
    a3.set_xticks(lv); a3.set_xticklabels(['0.01', '0.1', '0.3', '1', '2', '4']); a3.minorticks_off()
    a3.set_xlabel('Mean latency (s)'); a3.set_ylabel('Formation IAE ratio'); a3.set_ylim(0.65, 1.2)
    a3.legend(loc='upper left', handlelength=2.0)
    for a, s in zip(axs, ('(a)', '(b)', '(c)')):
        tag(a, s)
    fig.savefig(os.path.join(OUT, 'fig_hifi.pdf'))
    plt.close(fig)


def fig_r0():
    """Gain selection (R0): whole-run IAE ratios to no coupling; enlarged open markers: selected gains.
    (a) all gains, (b) the region of small gains enlarged."""
    S = json.load(open(os.path.join(RESR, 'r0_tuning_summary.json')))
    fig, (ax, zx) = plt.subplots(1, 2, figsize=(7.0, 2.3), gridspec_kw=dict(wspace=0.28))
    box = (0.93, 1.36, 0.77, 1.21)                      # x0, x1, y0, y1 of panel (b)
    offs = {'cascade': (3, -8), 'sigleak': (3, 3), 'direct': (-10, 3)}
    offs1 = {('direct', 0.1): (-13, 2), ('direct', 0.25): (-17, -2), ('direct', 0.5): (-17, -3), ('direct', 1.0): (4, -9)}
    for typ in ('cascade', 'sigleak', 'direct'):
        col, ls, lab, mk = CTRL[typ]
        rr = [r for r in S[typ]['rows'] if np.isfinite(r['form_ratio'])]
        for a in (ax, zx):
            a.plot([r['track_ratio'] for r in rr], [r['form_ratio'] for r in rr], color=col, ls=ls, marker=mk,
                   ms=3.8, mec='white', mew=0.5, label=lab)
        for r in rr:
            x, y = r['track_ratio'], r['form_ratio']
            inside = box[0] <= x <= box[1] and box[2] <= y <= box[3]
            for a in ((zx,) if inside else (ax,)):
                if r['value'] == S[typ]['best']:
                    a.plot(x, y, marker=mk, ms=9, mfc='none', mec=col, mew=1.2)
                a.annotate(f"{r['value']:g}", (x, y), textcoords='offset points', fontsize=6.5, color=INK2,
                           xytext=offs1.get((typ, r['value']), offs[typ]) if inside else offs[typ])
    for a in (ax, zx):
        a.axhline(1, color=CTRL['uncoupled'][0], lw=0.8, ls=(0, (3, 2)))
        a.axvline(1, color=CTRL['uncoupled'][0], lw=0.8, ls=(0, (3, 2)))
        a.set_xlabel('Tracking IAE / no coupling'); a.set_ylabel('Formation IAE / no coupling')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.add_patch(plt.Rectangle((box[0], box[2]), box[1] - box[0], box[3] - box[2], fill=False, ec=INK2, lw=0.6,
                               ls=(0, (2, 1.5))))
    zx.set_xlim(box[0], box[1]); zx.set_ylim(box[2], box[3])
    ax.legend(loc='upper left')
    for a, t in zip((ax, zx), ('(a)', '(b)')):
        tag(a, t)
    fig.savefig(os.path.join(OUT, 'fig_r0_tuning.pdf'))
    plt.close(fig)


FACT = [('loc_noise', 'Localization noise (m)'), ('loc_bias', 'Localization bias (m)'),
        ('loc_latency', 'Fix latency, realized (s)'), ('loc_rate', 'Fix rate, realized (Hz)'),
        ('calibration', 'Radius and track error (rel.)'), ('slip', 'Slip intensity ($\\times$ nominal)'),
        ('payload', 'Payload (kg)'), ('act_delay', 'Actuation delay (s)'), ('ctrl_rate', 'Control rate (Hz)'),
        ('clock', 'Clock offset bound (s)'), ('wheel_limit', 'Wheel-speed limit (m/s)')]
NOMINAL = dict(mfr.NOMINAL, loc_rate=hf.realized_timing(50.0, 20.0, f_fix=20.0)['f_fix'],
               loc_latency=hf.realized_timing(50.0, 20.0, d_fix=0.05)['d_fix'])


def fig_r4():
    """Practical factors one at a time (R4): medians over 8 seeds; dotted: nominal (realized) value."""
    doc, ts, rs = mfr.rows('r4_simreal')
    for metric, ylab, name, scale in (('pre_form', 'Formation RMS (mm)', 'r4_simreal_steady', 1e3),
                                      ('ev_iae', 'Formation IAE (m$\\cdot$s)', 'r4_simreal_event', 1.0)):
        fig, axs = plt.subplots(3, 4, figsize=(7.0, 5.9))
        for ax, (fac, xl) in zip(axs.ravel(), FACT):
            for typ in ('uncoupled', 'sigleak', 'direct', 'cascade'):
                rr = mfr.sel(rs, factor=fac, type=typ)
                if not rr:                               # R4 has no sigma-leak runs
                    continue
                lv = sorted(set(r['level'] for r in rr))
                med = [scale * np.nanmedian(mfr.vals(mfr.sel(rr, level=l), metric)) for l in lv]
                col, ls, lab, mk = CTRL[typ]
                ax.plot(lv, med, color=col, ls=ls, marker=mk, ms=3.2, mec='white', mew=0.4, lw=1.0, label=lab)
            ax.axvline(NOMINAL[fac], color=AXIS, lw=0.8, ls=(0, (2, 2)), zorder=0)
            ax.set_xlabel(xl, fontsize=7.6)
            ax.tick_params(labelsize=7)
        axs.ravel()[-1].axis('off')
        h, l = axs[0, 0].get_legend_handles_labels()
        axs.ravel()[-1].legend(h, l, loc='center', fontsize=7.6)
        for ax in axs[:, 0]:
            ax.set_ylabel(ylab, fontsize=7.6)
        fig.tight_layout(h_pad=0.7, w_pad=0.5)
        fig.savefig(os.path.join(OUT, 'fig_' + name + '.pdf'))
        plt.close(fig)


def fig_r6():
    """Events (R6, AGV 2 at 30 s): formation error, median and interquartile range over 20 seeds."""
    doc, ts, rs = mfr.rows('r6_events')
    cases = [('traction', '(a) Traction loss of AGV 2 (40% for 2 s)'), ('stop', '(b) Safety stop of AGV 2 (3 s)'),
             ('reloc', '(c) Relocalization jump of AGV 2 (10 cm)'), ('switching', '(d) Traction loss, switching links')]
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 3.6))
    for ax, (case, ttl) in zip(axs.ravel(), cases):
        for typ in ('uncoupled', 'sigleak', 'direct', 'cascade'):
            col, ls, lab, _ = CTRL[typ]
            t, Y = mfr.stack_ts(ts, mfr.sel(rs, case=case, type=typ), 'form')
            q1, med, q3 = np.nanquantile(Y, [0.25, 0.5, 0.75], axis=0)
            ax.fill_between(t, q1, q3, color=col, alpha=0.2, lw=0)
            ax.plot(t, med, color=col, ls=ls, lw=1.05, label=lab)
        ax.set_title(ttl, loc='left'); ax.set_xlim(25, 45); ax.set_ylim(0, None)
        ax.set_xlabel('Time (s)'); ax.set_ylabel('Formation error (m)')
    axs[0, 1].legend(loc='upper right')
    fig.tight_layout(h_pad=0.8, w_pad=1.0)
    fig.savefig(os.path.join(OUT, 'fig_r6_events.pdf'))
    plt.close(fig)


# graphs of R7: hue and marker per graph (pairwise OKLab distance >= 17 normal, >= 9 under simulated CVD)
GRAPHS = [('complete', 'complete', 'o', '#1f5fb4'), ('ring', 'ring', 's', '#eb6834'),
          ('dring', 'directed ring', '^', '#1baf7a'), ('chain', 'directed chain', 'v', '#b0306a'),
          ('star', 'star', 'D', '#d9b230')]


def fig_r7():
    """Topology and team size (R7, unit weights): medians over the runs that did not fail."""
    doc, ts, rs = mfr.rows('r7_scaling')
    Ns = [3, 5, 8, 12]
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.35), gridspec_kw=dict(wspace=0.4))
    for ax, metric, ylab, scale in ((axs[0], 'ev_iae', 'Formation IAE after the event (m$\\cdot$s)', 1.0),
                                    (axs[1], 'pre_form', 'Steady formation RMS (mm)', 1e3),
                                    (axs[2], 'ev_track_iae', 'Tracking IAE after the event (m$\\cdot$s)', 1.0)):
        b = [scale * np.nanmedian(mfr.vals(mfr.sel(rs, N=N, graph='none'), metric)) for N in Ns]
        ax.plot(Ns, b, color=CTRL['uncoupled'][0], ls=CTRL['uncoupled'][1], lw=1.3, label='no coupling')
        for g, gl, mk, col in GRAPHS:
            for typ, ls, mfc in (('cascade', '-', col), ('direct', (0, (3, 2)), 'white')):
                y = []
                for N in Ns:
                    ok = [r for r in mfr.sel(rs, N=N, graph=g, type=typ) if not r['m_failed']]
                    y.append(scale * np.nanmedian(mfr.vals(ok, metric)) if ok else np.nan)
                ax.plot(Ns, y, color=col, ls=ls, marker=mk, ms=3.4, mfc=mfc, mec=col, mew=0.8, lw=0.9,
                        label=(gl if typ == 'cascade' else None))
        ax.set_xlabel('Number of AGVs $N$'); ax.set_ylabel(ylab, fontsize=7.2); ax.set_xticks(Ns)
    axs[0].set_ylim(0.6, 2.2); axs[1].set_ylim(0, 110); axs[2].set_ylim(0.2, 1.25)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=6, fontsize=6.8, columnspacing=1.2,
               handlelength=2.0, title='Graph (filled markers, solid lines: cascade; open markers, dashed lines: direct)',
               title_fontsize=6.8)
    for a, s in zip(axs, ('(a)', '(b)', '(c)')):
        tag(a, s)
    fig.savefig(os.path.join(OUT, 'fig_r7_scaling.pdf'))
    plt.close(fig)


FIGS = {
    'kin': fig_kin, 'hifi': fig_hifi,
    'traj': lambda: mf.fig_traj(), 'iss': lambda: mf.fig_iss(),
    'r0': fig_r0, 'r1': lambda: mfr.r1(), 'r2': lambda: mfr.r2(), 'r3': lambda: mfr.r3(), 'r4': fig_r4,
    'r5': lambda: mfr.r5(), 'r6': fig_r6, 'r7': fig_r7, 'r8': lambda: mfr.r8(),
}
UNUSED = ('fig_r3_network_tracking.pdf',)            # by-product of mfr.r3, not used in the paper

if __name__ == '__main__':
    for w in (sys.argv[1:] or list(FIGS)):
        plt.rcParams.update(STYLE)
        FIGS[w]()
        print('done', w)
    for f in UNUSED:
        if os.path.exists(os.path.join(OUT, f)):
            os.remove(os.path.join(OUT, f))
