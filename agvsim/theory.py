"""Closed-form constants of the theory.

Numbering of the paper (arXiv v2): Lemma 2 = strict Lyapunov function, Lemma 3 = Halanay-type estimate of the filter
network, Theorem 2 = exponential estimates. The input-to-state estimate and the delay bound of the direct coupling are
Theorem 3 and Proposition 3 of arXiv v1 (in v1, the first three were Proposition 1, Lemma 2 and Theorem 2; the keys of
the stored results keep the v1 names, e.g. 'lemma2_ratio' and 'ok_prop1')."""
import numpy as np

SQ2 = np.sqrt(2.0)


def sinc(x):
    """sin(x)/x with sinc(0) = 1."""
    return np.sinc(np.asarray(x) / np.pi)


def alpha_weight(psi, b, kbar):
    """Weight alpha of the strict Lyapunov function (Lemma 2), for unit length scale."""
    a1 = 2 * psi / (1 + np.sqrt(b / 2))
    br = (2 * psi * np.sqrt(kbar ** 2 + b) + SQ2 * b
          + (b / psi) * ((1 / 3 + 1 / (2 * SQ2)) * np.sqrt(b) + 0.5 + 1 / np.sqrt(b)))
    return np.minimum(a1, 1 / br)


def alpha_ell(psi, b, kbar, ell=1.0):
    """Dimensional weight alpha_ell = ell * alpha(psi, b ell^2, kbar ell) for the length scale ell."""
    return ell * alpha_weight(psi, b * ell ** 2, kbar * ell)


def c12(al, b):
    """Sandwich constants c1 W <= V <= c2 W."""
    return 1 - 0.75 * al * np.sqrt(b), 1 + 1.25 * al * np.sqrt(b)


def W_fun(mu, nu, vt, b):
    return 0.5 * (mu ** 2 + nu ** 2) + vt ** 2 / (2 * b)


def V_strict(mu, nu, vt, b, al, ell=1.0):
    """Strict Lyapunov function V_ell = (1 + al sqrt(b)/4) W + al nu vt / (1 + W/ell^2)."""
    W = W_fun(mu, nu, vt, b)
    return (1 + al * np.sqrt(b) / 4) * W + al * nu * vt / (1 + W / ell ** 2)


def halanay_rate(lam, Lam, d, h, beta=0.0, chi0=1.0):
    """Positive root of eta = chi0 [(1-beta) lam - Lam d (exp(eta h) - 1)]  (Lemma 3)."""
    f = lambda eta: chi0 * ((1 - beta) * lam - Lam * d * np.expm1(eta * h)) - eta
    lo, hi = 0.0, chi0 * (1 - beta) * lam
    if f(hi) >= 0:
        return hi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def cascade_constants(par, bounds, d, E0, h):
    """A-priori constants of Theorems 1 and 2.  par: dict(lam, Lam, gam, rho, psi, b); bounds: dict(vmin, vmax, amax, wmax)."""
    lam, Lam, gam, rho, psi, b = (par[k] for k in ('lam', 'Lam', 'gam', 'rho', 'psi', 'b'))
    vmin, vmax, amax, wmax = (bounds[k] for k in ('vmin', 'vmax', 'amax', 'wmax'))
    out = {}
    out['gam_min'] = rho * amax / ((1 - rho) * vmin)
    out['vhat_min'] = (1 - rho) * vmin - rho * amax / gam
    out['vhat_max'] = (1 + rho) * vmax + rho * amax / gam
    out['Abar'] = np.sqrt(amax ** 2 + (vmax * wmax) ** 2) + rho * (2 * gam * vmax + amax)
    out['kbar'] = out['Abar'] / out['vhat_min'] ** 2
    al = alpha_weight(psi, b, out['kbar'])
    c1, c2 = c12(al, b)
    out.update(alpha=al, c1=c1, c2=c2)
    out['eta1'] = al * b * out['vhat_min'] / (2 * c2)
    out['C1'] = np.sqrt(c2 / c1) * np.exp(E0 ** 2 / 4)
    out['zeta_bound'] = 2 * Lam * d / lam * E0
    out['gbar'] = 4 * Lam * d * (1 + Lam * d / lam) * E0
    out['e_bound'] = (1 + 2 * Lam * d / lam + 4 * Lam * d / gam * (1 + Lam * d / lam)) * E0
    out['Ccoef'] = 1 + 2 * Lam * d / lam + 4 * Lam * d / gam * (1 + Lam * d / lam)
    out['chi0'] = min(1.0, rho * vmin / out['gbar']) if out['gbar'] > 0 else 1.0
    out['eta2'] = halanay_rate(lam, Lam, d, h, beta=0.5, chi0=out['chi0'])
    out['eta_star'] = 0.5 * min(out['eta1'], out['eta2'], gam)
    # Theorem 2 (arXiv v2): since eta2 > 0 implies exp(eta2 h) < 1 + lam/(2 Lam d) and 2 eta* <= eta2, the constants
    # below do not depend on the delay bound h; only the rate eta* does.
    out['C2'] = out['C1'] * (1 + 4 * Lam * d / lam)
    out['C3'] = (1.25 * lam + 2 * Lam * d) * out['C2'] + (2 * Lam * d + 0.5 * lam) * out['C1']
    out['K'] = out['C1'] + out['C2'] + 2 * out['C3'] / gam
    out['Kp'] = np.sqrt(b) * out['C1'] + np.pi * out['C3'] / vmin
    return out


def iss_constants(al, b, vhat_min, Wmax):
    """Constants of Theorem 3 (local input-to-state estimate), unit length scale."""
    c1, c2 = c12(al, b)
    a = al * b * vhat_min
    return dict(a=a, Delta_max=a / (1 + Wmax) * np.sqrt(c1 * Wmax / (2 * c2)),
                eta3=a / (2 * c2 * (1 + Wmax)), G=2 * (1 + Wmax) / a * np.sqrt(c2 / c1), c1=c1, c2=c2)


def Delta_size(dv, dn, dw, b):
    """Perturbation size Delta of Theorem 3."""
    return 1.25 * np.abs(dv) + 1.75 * np.abs(dn) + (1.75 + 0.18 * np.sqrt(b)) / np.sqrt(b) * np.abs(dw)


def direct_hstar(k1, k2, k3, c, dmax, vmax):
    """Delay bound h* of the direct coupling (Proposition 2)."""
    return 1.0 / (c * dmax * max(k1, k2 * vmax ** 2 / k3))
