"""E1. Proposition 1 as an algebraic statement: evaluate the exact derivative of the strict Lyapunov function at
random points (extended precision) and compare it with the claimed bounds; same for the perturbed inequality of
Theorem 3 and for the length-scaled family V_ell."""
import sys, json, time
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from agvsim import theory as th

LD = np.longdouble


def sincl(x):
    x = np.asarray(x, LD)
    small = np.abs(x) < 1e-4
    xs = np.where(small, 1.0, x)
    return np.where(small, 1 - x * x / 6 + x ** 4 / 120, np.sin(xs) / xs)


def batch(n, rng, focus=False):
    U = lambda lo, hi: (10 ** rng.uniform(lo, hi, n)).astype(LD)
    sgn = lambda: rng.choice([-1.0, 1.0], n)
    psi, b = U(-1, np.log10(5)), U(-1, np.log10(32))
    vh, kbar = U(-3, 1), U(-3, 3)
    fr = rng.uniform(-1, 1, n)
    fr[rng.random(n) < 0.3] = 1.0
    wh = kbar * vh * fr * sgn()
    if focus:        # region where the bound is tightest: moderate errors
        mu, nu, vt = U(-2, 1) * sgn(), U(-2, 1) * sgn(), U(-2, np.log10(8)) * sgn()
    else:
        mu, nu, vt = U(-4, 4) * sgn(), U(-4, 4) * sgn(), U(-4, np.log10(40)) * sgn()
    ell = U(-2, 2)
    dv, dn, dom = U(-4, 1) * sgn(), U(-4, 1) * sgn(), U(-4, 1) * sgn()
    return psi, b, vh, kbar, wh, mu, nu, vt, ell, dv, dn, dom


def evaluate(n, seed, focus=False):
    rng = np.random.default_rng(seed)
    psi, b, vh, kbar, wh, mu, nu, vt, ell, dv, dn, dom = batch(n, rng, focus)
    sq = np.sqrt
    k = 2 * psi * sq(wh ** 2 + b * vh ** 2)
    snc = sincl(vt)
    om = wh + k * vt + b * vh * snc * nu
    W = (mu ** 2 + nu ** 2) / 2 + vt ** 2 / (2 * b)
    out = {}

    def derivs(dv_, dn_, dom_):
        mud = (om + dom_) * nu - k * mu - dv_
        nud = -(om + dom_) * mu + vh * np.sin(vt) - dn_
        vtd = -k * vt - b * vh * snc * nu - dom_
        return mud, nud, vtd

    # --- unperturbed, unit length scale
    al = th.alpha_weight(psi, b, kbar).astype(LD)
    c1, c2 = 1 - 0.75 * al * sq(b), 1 + 1.25 * al * sq(b)
    mud, nud, vtd = derivs(0, 0, 0)
    Wd = mu * mud + nu * nud + vt * vtd / b
    Wd_id = -k * mu ** 2 - k * vt ** 2 / b
    out['Wdot_identity'] = float(np.max(np.abs(Wd - Wd_id) / (np.abs(om * mu * nu) + np.abs(vh * nu * vt) + np.abs(Wd_id) + LD(1e-4000))))
    V = (1 + al * sq(b) / 4) * W + al * nu * vt / (1 + W)
    out['a_lower'] = float(np.min(V / W - c1))
    out['a_upper'] = float(np.min(c2 - V / W))
    Vd = (1 + al * sq(b) / 4) * Wd_id + al * ((nud * vt + nu * vtd) / (1 + W) - nu * vt * Wd_id / (1 + W) ** 2)
    B1 = -(k / 2) * mu ** 2 - k / (2 * b) * vt ** 2 - al * b * vh * nu ** 2 / (2 * (1 + W))
    B2 = -al * b * vh * W / (1 + W)
    out['b_margin1'] = float(np.min((B1 - Vd) / np.abs(B1)))      # >= 0 : first inequality of (b)
    out['b_margin2'] = float(np.min((B2 - B1) / np.abs(B2)))      # >= 0 : second inequality of (b)
    out['b_ratio_min'] = float(np.min(Vd / B2))                   # >= 1 : decrease at least the guaranteed one
    # --- perturbed inequality used in Theorem 3
    mud, nud, vtd = derivs(dv, dn, dom)
    Wdp = Wd_id - mu * dv - nu * dn - vt * dom / b
    Vdp = (1 + al * sq(b) / 4) * Wdp + al * ((nud * vt + nu * vtd) / (1 + W) - nu * vt * Wdp / (1 + W) ** 2)
    Delta = LD(1.25) * np.abs(dv) + LD(1.75) * np.abs(dn) + (LD(1.75) + LD(0.18) * sq(b)) / sq(b) * np.abs(dom)
    rhs = B2 + sq(2 * W) * Delta
    out['pert_margin'] = float(np.min((rhs - Vdp) / (np.abs(B2) + sq(2 * W) * Delta)))
    # --- length-scaled family V_ell
    ale = (ell * th.alpha_weight(psi, b * ell ** 2, kbar * ell)).astype(LD)
    c1e, c2e = 1 - 0.75 * ale * sq(b), 1 + 1.25 * ale * sq(b)
    We = W / ell ** 2
    Ve = (1 + ale * sq(b) / 4) * W + ale * nu * vt / (1 + We)
    mud, nud, vtd = derivs(0, 0, 0)
    Vde = (1 + ale * sq(b) / 4) * Wd_id + ale * ((nud * vt + nu * vtd) / (1 + We) - nu * vt * Wd_id / (ell ** 2 * (1 + We) ** 2))
    B2e = -ale * b * vh * W / (1 + We)
    out['ell_ratio_min'] = float(np.min(Vde / B2e))
    out['ell_sandwich'] = float(min(np.min(Ve / W - c1e), np.min(c2e - Ve / W)))
    out['alpha_sqrt_b_max'] = float(np.max(al * sq(b)))
    return out


if __name__ == '__main__':
    t0 = time.time()
    n, reps = 500_000, 20
    res = [evaluate(n, s) for s in range(reps)] + [evaluate(n, 1000 + s, focus=True) for s in range(reps)]
    agg = {'points': 2 * n * reps}
    for key in res[0]:
        vals = [r[key] for r in res]
        agg[key] = float(max(vals)) if key in ('Wdot_identity', 'alpha_sqrt_b_max') else float(min(vals))
    agg['violations'] = int(sum(1 for r in res if min(r['b_margin1'], r['b_margin2'], r['pert_margin'], r['a_lower'], r['a_upper'], r['ell_sandwich']) < 0 or min(r['b_ratio_min'], r['ell_ratio_min']) < 1))
    # table of alpha and of the scale-optimised rate
    tab = []
    for (psi, b, kb) in [(0.7, 3.0, 1.0), (0.7, 3.0, 5.0), (0.7, 3.0, 15.0), (1.0, 3.0, 1.0), (0.7, 10.0, 1.0)]:
        ells = 10 ** np.linspace(-2, 1, 601)
        al = th.alpha_ell(psi, b, kb, ells)
        rate = al * b / th.c12(al, b)[1]
        i = int(np.argmax(rate))
        a1 = float(th.alpha_weight(psi, b, kb))
        tab.append(dict(psi=psi, b=b, kbar=kb, alpha_1m=a1, rate_1m=float(a1 * b / th.c12(a1, b)[1]),
                        ell_opt=float(ells[i]), rate_opt=float(rate[i])))
    agg['scale_table'] = tab
    agg['wall_s'] = time.time() - t0
    json.dump(agg, open(os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', ''), 'e1_prop1.json'), 'w'), indent=1)
    print(json.dumps(agg, indent=1))
