"""Figures of the paper (PDF).  Colours: categorical slots in fixed order for vehicle identity; one-hue ordinal ramp
for ordered quantities (delay level, coupling gain); neutral grey for the uncoupled baseline and for theory lines."""
import sys, json
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from agvsim import theory as th

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, FIG = os.path.join(_ROOT, 'results', ''), os.path.join(_ROOT, 'figures', '')
AGV = ['#2a78d6', '#eb6834', '#1baf7a']                 # slots 1-3: AGV 1, 2, 3
RAMP = ['#86b6ef', '#2a78d6', '#104281']                # ordinal: low -> high
GREY, INK, INK2, GRID, AXIS = '#8b8a83', '#0b0b0b', '#52514a', '#e1e0d9', '#c3c2b7'
plt.rcParams.update({'font.size': 8, 'font.family': 'DejaVu Sans', 'axes.edgecolor': AXIS, 'axes.linewidth': 0.8,
                     'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
                     'legend.frameon': False, 'legend.fontsize': 7.5, 'lines.linewidth': 1.4, 'axes.titlesize': 8.5,
                     'axes.titleweight': 'bold', 'axes.spines.top': False, 'axes.spines.right': False,
                     'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'pdf.fonttype': 42, 'savefig.bbox': 'tight',
                     'savefig.pad_inches': 0.03})
NETS = [('nominal', 'nominal'), ('slow1', '1 s network'), ('slow3', '3 s network')]


def load(ref, net, graph='complete'):
    return np.load(RES + f'e3_{ref}_{net}_{graph}.npz')


def fig_traj():
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.55), gridspec_kw=dict(width_ratios=[1.0, 1.35, 1.6]))
    marks = {'circle': [0, 6, 14, 22, 30, 38], 'figure8': [0, 8, 18, 30, 42, 54, 66], 'warehouse': [0, 7, 14, 22, 30, 37, 44, 52]}
    for ax, (ref, title) in zip(axs, [('circle', 'Circle'), ('figure8', 'Figure-8'), ('warehouse', 'Warehouse route')]):
        d = load(ref, 'nominal')
        t, p, Pr = d['t'], d['p'], d['Pr']
        for i in range(3):
            ax.plot(Pr[:, i, 0], Pr[:, i, 1], color=GRID, lw=2.2, zorder=1)
        for i in range(3):
            ax.plot(p[:, i, 0], p[:, i, 1], color=AGV[i], lw=1.1, zorder=3, label=f'AGV {i + 1}')
        for tm in marks[ref]:
            k = int(np.searchsorted(t, tm))
            tri = np.vstack([Pr[k], Pr[k][:1]])
            ax.plot(tri[:, 0], tri[:, 1], color=GREY, lw=0.7, ls=(0, (3, 2)), zorder=2)
            tri = np.vstack([p[k], p[k][:1]])
            ax.plot(tri[:, 0], tri[:, 1], color=INK, lw=0.7, zorder=4)
            for i in range(3):
                ax.plot(p[k, i, 0], p[k, i, 1], 'o', ms=3.2, color=AGV[i], mec='white', mew=0.5, zorder=5)
        ax.set_aspect('equal')
        ax.set_title(title, loc='left')
        ax.set_xlabel('x (m)')
        ax.grid(False)
    axs[0].set_ylabel('y (m)')
    axs[0].legend(loc='center', ncol=1, handlelength=1.2)
    fig.savefig(FIG + 'fig_traj.pdf')
    plt.close(fig)


def fig_lyap():
    runs = json.load(open(RES + 'e3_runs.json'))
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 4.5))
    psi, b, lam, Lam, d = 0.7, 3.0, 1.0, 1.0, 2.0
    (a1, a2), (a3, a4) = axs
    for (net, lab), col in zip(NETS, RAMP):
        dd = load('figure8', net)
        t, W = dd['t'], dd['W']
        r = [x for x in runs if x['ref'] == 'figure8' and x['net'] == net and x['graph'] == 'complete' and x['seed'] == 7 and x['tag'] == 'std'][0]
        a1.semilogy(t, W.max(1), color=col, label=lab)
        if net == 'nominal':
            al = r['alpha_run']; c1, c2 = th.c12(al, b)
            s = np.concatenate([[0], np.cumsum(0.5 * (dd['vh'][1:] + dd['vh'][:-1]).min(1) * np.diff(t))])
            est = c2 / c1 * np.exp(W[0].max()) * W[0].max() * np.exp(-al * b / c2 * s)
            a1.semilogy(t, est, color=GREY, lw=1.0, ls=(0, (4, 2)))
            a1.text(40, est[np.searchsorted(t, 40)] * 1e-3, 'dashed: guaranteed by Prop. 1(c)', color=INK2, fontsize=7)
        M = np.linalg.norm(dd['zeta'], axis=2).max(1)
        a2.semilogy(t, np.maximum(M, 1e-16), color=col, label=lab)
        eta = th.halanay_rate(lam, Lam, d, r['h'])
        t0 = {'nominal': 12.0, 'slow1': 22.0, 'slow3': 32.0}[net]
        k0 = int(np.searchsorted(t, t0)); tt = t[k0:]
        a2.semilogy(tt, M[k0] * 1.6 * np.exp(-eta * (tt - t0)), color=GREY, lw=1.0, ls=(0, (4, 2)))
        e = np.linalg.norm(dd['e'], axis=2).max(1)
        a3.semilogy(t, e, color=col, label=lab)
        m = (t >= 20) & (t <= 32)
        a4.plot(t[m], dd['tau'][m], color=col, lw=1.0, label=lab)
    dn = load('figure8', 'nominal', 'none')
    a3.semilogy(dn['t'], np.linalg.norm(dn['e'], axis=2).max(1), color=GREY, lw=1.1, label='no coupling')
    a2.text(2, 3e-15, 'dashed: decay rate guaranteed by Lemma 2', color=INK2, fontsize=7)
    a1.set_title('(a) Vehicle Lyapunov function, max$_i$ W$_i$ (m$^2$)', loc='left'); a1.set_ylim(1e-30, 3)
    a2.set_title('(b) Filter state, max$_i$ |$\\zeta_i$| (m)', loc='left'); a2.set_ylim(1e-15, 1)
    a3.set_title('(c) Tracking error, max$_i$ |e$_i$| (m)', loc='left'); a3.set_ylim(1e-15, 2)
    a4.set_title('(d) Largest effective delay over the links (s)', loc='left')
    for a in (a1, a2, a3):
        a.set_xlim(0, 75)
    a3.set_xlabel('time (s)'); a4.set_xlabel('time (s)')
    a1.legend(loc='lower left'); a3.legend(loc='lower left', ncol=2); a4.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.78)); a4.set_ylim(0, 5.2)
    fig.tight_layout(h_pad=1.2)
    fig.savefig(FIG + 'fig_lyap.pdf')
    plt.close(fig)


def fig_iss():
    d = json.load(open(RES + 'e4_iss.json'))['sweep']
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.35), sharex=True)
    cases = [('nominal', 'none', 'no coupling', GREY, 's'), ('nominal', 'complete', 'cascade, nominal network', RAMP[1], 'o'),
             ('slow3', 'complete', 'cascade, 3 s network', RAMP[2], '^')]
    for ax, (key, title) in zip(axs, [('eps', '(a) Local error |$\\varepsilon_i$| (mm)'), ('e', '(b) Tracking error |e$_i$| (mm)'),
                                      ('form', '(c) Formation error (mm)')]):
        for net, g, lab, col, mk in cases:
            rr = sorted([r for r in d if r['net'] == net and r['graph'] == g], key=lambda r: r['Delta'])
            ax.loglog([r['Delta'] for r in rr], [1e3 * r[key] for r in rr], color=col, marker=mk, ms=4, mec='white', mew=0.5, label=lab)
        ax.set_title(title, loc='left')
        ax.set_xlabel('perturbation size $\\Delta$ (m/s)')
    x = np.array([0.02, 0.5]); axs[0].loglog(x, 1e3 * 0.44 * x * 2.2, color=GREY, lw=0.9, ls=(0, (4, 2)))
    axs[0].text(0.03, 60, 'slope 1', color=INK2, fontsize=7)
    axs[0].legend(loc='lower right', fontsize=6.8)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(FIG + 'fig_iss.pdf')
    plt.close(fig)


def fig_form():
    d = json.load(open(RES + 'e5_formation.json'))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw=dict(width_ratios=[1.0, 1.55]))
    st = [r for r in d['static'] if 'disagreement_ratio' in r and r['Lam'] > 0]
    for gname, mk in (('complete3', 'o'), ('path4', 's')):
        for net, col in (('nominal', RAMP[1]), ('slow3', RAMP[2])):
            rr = [r for r in st if r['graph'] == gname and r['net'] == net]
            a1.loglog([r['predicted_factor'] for r in rr], [r['disagreement_ratio'] for r in rr], mk, ms=5, color=col, mec='white', mew=0.6,
                      label=('complete graph, N=3' if gname == 'complete3' else 'path graph, N=4') + (', nominal' if net == 'nominal' else ', 3 s network'))
    x = np.array([0.02, 1.0]); a1.loglog(x, x, color=GREY, lw=0.9, ls=(0, (4, 2)), zorder=0)
    a1.set_xlabel('predicted factor $\\lambda/(\\lambda+\\Lambda\\lambda_2(L))$'); a1.set_ylabel('measured formation-error ratio')
    a1.set_title('(a) Steady formation-error ratio', loc='left'); a1.legend(loc='upper left', fontsize=6.5, handletextpad=0.2)
    a1.text(0.2, 0.028, 'dashed: equality', color=INK2, fontsize=7)
    for (Lam, lab), col in zip(((0.0, 'no coupling'), (1.0, '$\\Lambda$ = 1 s$^{-1}$'), (3.0, '$\\Lambda$ = 3 s$^{-1}$')), (GREY, RAMP[1], RAMP[2])):
        s = np.load(RES + f'e5_slip_Lam{Lam:g}_nominal.npz')
        m = (s['t'] >= 28) & (s['t'] <= 40)
        a2.plot(s['t'][m], 1e3 * s['form'][m], color=col, label=lab)
    a2.axvspan(30, 32, color='#f1f0ea', zorder=0); a2.text(30.08, 455, 'slip', color=INK2, fontsize=7, va='top')
    a2.set_xlabel('time (s)'); a2.set_title('(b) Formation error, 2 s traction loss of AGV 2 (mm)', loc='left'); a2.legend(loc='upper right')
    a2.set_xlim(28, 40); a2.set_ylim(0, None)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(FIG + 'fig_form.pdf')
    plt.close(fig)


if __name__ == '__main__':
    which = sys.argv[1:] or ['traj', 'lyap', 'iss', 'form']
    for w in which:
        globals()['fig_' + w]()
        print('done', w)
