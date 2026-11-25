"""E4. Robustness (Theorem 3): (a) perturbation-amplitude sweep, (b) literal check of Theorem 3 with constants that are
valid for the run, (c) broadcast-noise gain."""
import json, time
import numpy as np
from multiprocessing import Pool
from common import run_case, RES, GRAPHS
from agvsim import theory as th, cascade as cs

B = 3.0
DCOEF = 1.25 + 1.75 * 0.5 + (1.75 + 0.18 * np.sqrt(B)) / np.sqrt(B) * 2.0     # sup Delta = DCOEF * amplitude


def sin_pert(amp, N=3, seed=0):
    ph = np.random.default_rng(seed).uniform(0, 2 * np.pi, (3, N))
    return lambda t, v, om: (amp * np.sin(0.9 * t + ph[0]), 0.5 * amp * np.sin(0.5 * t + ph[1]), 2 * amp * np.sin(1.3 * t + ph[2]))


def sweep_job(a):
    amp, net, graph = a
    sim, run, bounds = run_case('figure8', net, graph, 7, pert=sin_pert(amp))
    m = run['t'] >= 45.0
    return dict(amp=amp, net=net, graph=graph, Delta=DCOEF * amp, h=run['meta']['h'],
                eps=float(np.linalg.norm(run['eps'][m], axis=2).max()), e=float(np.linalg.norm(run['e'][m], axis=2).max()),
                form=float(cs.formation_error(run['e'][m]).max()), balance=float(np.abs(run['W'] + run['D'] - run['W'][0]).max()))


def thm3_job(net):
    """Literal check.  Constants (alpha from the curvature bound, vhat_min) are valid for the run itself."""
    Wmax = 0.02
    out = {}
    # pass 1 with a guess of the constants, pass 2 with constants valid for the perturbed run
    kap, vhm = 20.0, 0.12
    for it in range(3):
        al = float(th.alpha_weight(0.7, B, kap))
        K = th.iss_constants(al, B, vhm, Wmax)
        amp = 0.1 * K['Delta_max'] / DCOEF                         # sup Delta = Delta_max / 10
        dp0 = 0.12 * np.array([[1.0, 0.0], [-0.5, 0.8], [0.3, -0.9]]) / np.array([[1.0], [0.943], [0.949]])
        vt0 = np.deg2rad([8.0, -6.0, 5.0])
        nbar = 0.002
        rng = np.random.default_rng(5)
        noise = {}
        def bnoise(j, k):
            if (j, k) not in noise:
                r, a = nbar * np.sqrt(rng.random()), rng.uniform(0, 2 * np.pi)
                noise[(j, k)] = r * np.array([np.cos(a), np.sin(a)])
            return noise[(j, k)]
        sim, run, bounds = run_case('figure8', net, 'complete', 7, pert=sin_pert(amp), bnoise=bnoise, dp0=dp0, vt0=vt0)
        kap_run, vh_run = float((np.abs(run['wh']) / run['vh']).max()), float(run['vh'].min())
        if kap_run <= kap and vh_run >= vhm:
            break
        kap, vhm = 1.05 * max(kap, kap_run), 0.95 * min(vhm, vh_run)
    t, W = run['t'], run['W']
    V0 = th.V_strict(run['mu'][0], run['nu'][0], run['vt'][0], B, al)
    supD = DCOEF * amp
    ne = np.linalg.norm(run['eps'], axis=2)
    bound_eps = np.sqrt(2 * K['c2'] / K['c1'] * W[0])[None, :] * np.exp(-K['eta3'] * t / 2)[:, None] + K['G'] * supD
    par = sim.par
    d = 2.0
    C = 1 + 2 * par['Lam'] * d / par['lam'] + 4 * par['Lam'] * d / par['gam'] * (1 + par['Lam'] * d / par['lam'])
    bound_e = C * (np.sqrt(2 * K['c2'] * W[0].max() / K['c1']) + K['G'] * supD) + (C - 1) / 2 * nbar
    out.update(net=net, alpha=al, kap_used=kap, kap_run=kap_run, vhat_min_used=vhm, vhat_min_run=vh_run, Wmax=Wmax,
               Delta_max=K['Delta_max'], G=K['G'], eta3=K['eta3'], supDelta=supD, nbar=nbar, C=C,
               hyp_V0_ok=bool(np.all(V0 <= K['c1'] * Wmax)), W_sup=float(W.max()), ok_invariance=bool(W.max() <= Wmax),
               eps_ratio=float((ne / bound_eps).max()), e_ratio=float(np.linalg.norm(run['e'], axis=2).max() / bound_e),
               eps_end=float(ne[t >= 60].max()), G_measured=float(ne[t >= 60].max() / supD), h=run['meta']['h'])
    out['ok'] = bool(out['hyp_V0_ok'] and out['ok_invariance'] and out['eps_ratio'] <= 1 and out['e_ratio'] <= 1)
    return out


def noise_job(a):
    net, Lam, nbar = a
    rng = np.random.default_rng(11)
    noise = {}
    def bnoise(j, k):
        if (j, k) not in noise:
            ang = rng.uniform(0, 2 * np.pi)
            noise[(j, k)] = nbar * np.array([np.cos(ang), np.sin(ang)])        # worst case: always on the bound
        return noise[(j, k)]
    sim, run, bounds = run_case('figure8', net, 'complete', 7, par=dict(Lam=Lam), bnoise=bnoise,
                                dp0=np.zeros((3, 2)), vt0=np.zeros(3))
    par, d = sim.par, 2.0
    C = 1 + 2 * par['Lam'] * d / par['lam'] + 4 * par['Lam'] * d / par['gam'] * (1 + par['Lam'] * d / par['lam'])
    e = float(np.linalg.norm(run['e'], axis=2).max())
    return dict(net=net, Lam=Lam, nbar=nbar, e_sup=e, bound=(C - 1) / 2 * nbar, ratio=e / ((C - 1) / 2 * nbar),
                eps_sup=float(np.linalg.norm(run['eps'], axis=2).max()), zeta_sup=float(np.linalg.norm(run['zeta'], axis=2).max()),
                zeta_bound=par['Lam'] * d / par['lam'] * nbar)


if __name__ == '__main__':
    t0 = time.time()
    amps = [0.005, 0.01, 0.02, 0.05, 0.10]
    jobs = [(a, net, g) for a in amps for (net, g) in (('nominal', 'complete'), ('slow3', 'complete'), ('nominal', 'none'))]
    with Pool(18) as pool:
        r1 = pool.map_async(sweep_job, jobs, chunksize=1)
        r2 = pool.map_async(thm3_job, ['nominal', 'slow3'], chunksize=1)
        r3 = pool.map_async(noise_job, [(n, L, 0.01) for n in ('nominal', 'slow3') for L in (1.0, 3.0)], chunksize=1)
        out = dict(sweep=r1.get(), thm3=r2.get(), noise=r3.get())
    out['wall_s'] = time.time() - t0
    json.dump(out, open(RES + 'e4_iss.json', 'w'), indent=1, default=float)
    for k in ('sweep', 'thm3', 'noise'):
        for r in out[k]:
            print(k, {q: (round(v, 6) if isinstance(v, float) else v) for q, v in r.items()})
    print('wall', out['wall_s'])
