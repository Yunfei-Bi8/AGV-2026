"""Checks of every claim of the theory on a simulated run (kinematic model)."""
import numpy as np
from . import theory as th
from .cascade import formation_error


def decay_rate(t, x, lo, hi):
    """Least-squares slope of -log(x) on the samples with lo >= x >= hi (x decreasing overall)."""
    m = (x <= lo) & (x >= hi)
    if m.sum() < 20:
        return np.nan
    return float(-np.polyfit(t[m], np.log(x[m]), 1)[0])


def verify_run(run, par, bounds, d, A=None):
    """Margins of all inequalities; a claim holds when its 'ok' flag is True."""
    lam, Lam, gam, rho, psi, b = (par[k] for k in ('lam', 'Lam', 'gam', 'rho', 'psi', 'b'))
    t, W = run['t'], run['W']
    h = run['meta']['h']
    E0 = float(np.sqrt(2 * W[0].max()))
    C = th.cascade_constants(par, bounds, d, E0, h)
    r = dict(h=h, E0=E0, T=float(t[-1]))
    tiny = 1e-9
    # Item 1: the shifted reference keeps moving
    r['vhat_min_obs'], r['vhat_min_thm'] = float(run['vh'].min()), float(C['vhat_min'])
    r['ok_item1'] = bool(run['vh'].min() >= C['vhat_min'])
    # Item 2: W non-increasing; energy balance W + D = W(0); identity for dW/dt
    r['W_max_rise_rel'] = float(np.diff(W, axis=0).max() / W[0].max())
    r['balance_resid'] = float(np.abs(W + run['D'] - W[0]).max())
    Wd = run['mu'] * run['mud'] + run['nu'] * run['nud'] + run['vt'] * run['vtd'] / b
    Wd_th = -run['k'] * run['mu'] ** 2 - run['k'] / b * run['vt'] ** 2 - run['pwr']
    scale = run['k'] * (run['mu'] ** 2 + run['nu'] ** 2 + run['vt'] ** 2 / b) + 1e-300
    above = W > 1e-12 * W[0].max()                 # errors above 1e-6 of their initial size: no rounding effects
    r['Wdot_identity_resid'] = float((np.abs(Wd - Wd_th) / scale)[above].max())
    r['ok_item2'] = bool(r['W_max_rise_rel'] <= 1e-9 and r['balance_resid'] <= 1e-8)
    # Item 3: bounds proportional to the initial local errors
    ne = np.linalg.norm(run['eps'], axis=2)
    nz = np.linalg.norm(run['zeta'], axis=2)
    nerr = np.linalg.norm(run['e'], axis=2)
    r['eps_ratio'] = float(ne.max() / E0)
    r['zeta_ratio'] = float(nz.max() / C['zeta_bound']) if C['zeta_bound'] > 0 else 0.0
    r['e_ratio'] = float(nerr.max() / C['e_bound'])
    r['ok_item3'] = bool(max(r['eps_ratio'], r['zeta_ratio'], r['e_ratio']) <= 1 + tiny)
    # Lemma 3 with t0 = 0, beta -> 1:  M(t) <= sup_{s<=t} P(s) / lam
    M, Pm = nz.max(1), np.maximum.accumulate(run['pi'].max(1))
    r['lemma2_ratio'] = float(np.max(M / np.maximum(Pm / lam, 1e-300)))
    r['ok_lemma2'] = bool(r['lemma2_ratio'] <= 1 + tiny)
    # Item 4: convergence
    form = formation_error(run['e'])
    r['e_end'], r['head_end'], r['form_end'] = float(nerr[-1].max()), float(np.abs(run['head'][-1]).max()), float(form[-1])
    r['e_start'], r['form_start'] = float(nerr[0].max()), float(form[0])
    # Lemma 2 (strict Lyapunov function) along the run: with the a-priori curvature bound and with the curvature that occurred
    kap_run = float((np.abs(run['wh']) / run['vh']).max())
    r['curv_run'], r['curv_apriori'] = kap_run, float(C['kbar'])
    s = np.concatenate([np.zeros((1, W.shape[1])),
                        np.cumsum(0.5 * (run['vh'][1:] + run['vh'][:-1]) * np.diff(t)[:, None], axis=0)])
    for tag, kap in (('apriori', C['kbar']), ('run', kap_run)):
        al = float(th.alpha_weight(psi, b, kap))
        c1, c2 = th.c12(al, b)
        F = 1 + al * np.sqrt(b) / 4 - al * run['nu'] * run['vt'] / (1 + W) ** 2
        Vd = F * Wd + al * (run['nud'] * run['vt'] + run['nu'] * run['vtd']) / (1 + W)
        bound = -al * b * run['vh'] * W / (1 + W)
        m = W > 1e-22 * W[0].max()                      # above the rounding floor
        r['alpha_' + tag] = al
        r['Vdot_ratio_min_' + tag] = float((Vd[m] / bound[m]).min())     # >= 1 means the decrease is at least the guaranteed one
        est = c2 / c1 * np.exp(W[0]) * W[0] * np.exp(-al * b / c2 * s)
        r['propc_ratio_max_' + tag] = float((W / est).max())
        Vs = th.V_strict(run['mu'], run['nu'], run['vt'], b, al)
        r['sandwich_ok_' + tag] = bool(np.all(Vs >= c1 * W * (1 - tiny)) and np.all(Vs <= c2 * W * (1 + tiny)))
    r['ok_prop1'] = bool(r['Vdot_ratio_min_apriori'] >= 1 and r['Vdot_ratio_min_run'] >= 1
                         and r['propc_ratio_max_apriori'] <= 1 and r['propc_ratio_max_run'] <= 1
                         and r['sandwich_ok_apriori'] and r['sandwich_ok_run'])
    # Theorem 2: explicit exponential estimates (constants of theory.cascade_constants)
    b1 = C['C1'] * E0 * np.exp(-C['eta1'] * t)
    b2 = C['K'] * E0 * np.exp(-C['eta_star'] * t)
    b3 = C['Kp'] * E0 * np.exp(-C['eta_star'] * t)
    r['thm2_eps_ratio'] = float((ne.max(1) / b1).max())
    r['thm2_e_ratio'] = float((nerr.max(1) / b2).max())
    r['thm2_head_ratio'] = float((np.abs(run['head']).max(1) / b3).max())
    r['ok_thm2'] = bool(max(r['thm2_eps_ratio'], r['thm2_e_ratio'], r['thm2_head_ratio']) <= 1 + tiny)
    # the largest ratio is attained at t = 0 (it equals 1/C1 there); the ratios after 1 s show the conservatism
    m1 = t >= 1.0
    r['thm2_eps_ratio_t1'] = float((ne.max(1)[m1] / b1[m1]).max())
    r['thm2_e_ratio_t1'] = float((nerr.max(1)[m1] / b2[m1]).max())
    r['thm2_head_ratio_t1'] = float((np.abs(run['head']).max(1)[m1] / b3[m1]).max())
    r['eta1'], r['eta_star'] = float(C['eta1']), float(C['eta_star'])
    # measured rates
    Wm = W.max(1)
    r['rate_W'] = decay_rate(t, Wm, 1e-3 * Wm[0], 1e-14 * Wm[0])
    nem = ne.max(1)
    r['rate_eps'] = decay_rate(t, nem, 1e-2 * nem[0], 1e-8 * nem[0])     # comparable with eta1 of Theorem 2
    Mx = M.max() if M.max() > 0 else 1.0
    r['rate_M'] = decay_rate(t, np.maximum(M, 1e-300), 1e-2 * Mx, 1e-8 * Mx)
    em = nerr.max(1)
    r['rate_e'] = decay_rate(t, em, 1e-2 * em[0], 1e-8 * em[0])
    r['rate_lemma2'] = th.halanay_rate(lam, Lam, d, h) if d > 0 else lam
    r['t_form_2cm'] = float(t[np.max(np.nonzero(form > 0.02)[0]) + 1]) if (form > 0.02).any() and form[-1] <= 0.02 else (0.0 if form[-1] <= 0.02 else np.inf)
    r['steps'], r['rejected'], r['err_sum'] = run['meta']['steps'], run['meta']['rejected'], run['meta']['err_sum']
    r['all_ok'] = bool(r['ok_item1'] and r['ok_item2'] and r['ok_item3'] and r['ok_lemma2'] and r['ok_prop1'] and r['ok_thm2'])
    return r
