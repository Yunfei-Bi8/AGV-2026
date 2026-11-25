"""E5. What the coupling does.  (a) Proposition 2: for constant local errors eps^inf the tracking errors converge to
(lam I + Lam L)^(-1) (lam eps^inf + Lam D (eps^inf - phi(eps^inf))), i.e. lam (lam I + Lam L)^(-1) eps^inf below the cap,
for any delays.  (b) A 2 s traction loss of one vehicle: formation error with and without coupling."""
import json, time
import numpy as np
from multiprocessing import Pool
from common import run_case, RES
from agvsim import cascade as cs

LINE4 = np.array([[0.9, 0.3], [0.3, -0.3], [-0.3, 0.3], [-0.9, -0.3]])
GR = {
    'complete3': (np.ones((3, 3)) - np.eye(3), None),
    'chain3': (np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float), None),
    'path4': (np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]], float), LINE4),
    'weighted3': (np.array([[0, 2.0, 0.5], [0.3, 0, 1.0], [1.5, 0, 0]]), None),
}


def static_job(a):
    gname, Lam, lam, net, slip = a
    A, offs = GR[gname]
    N = A.shape[0]
    def pert(t, v, om):
        dv = np.zeros(N); dv[0] = slip
        return dv, np.zeros(N), np.zeros(N)
    sim, run, _ = run_case('line', net, None, 3, par=dict(Lam=Lam, lam=lam), pert=pert, A=A, offs=offs, T=150.0,
                           dp0=np.zeros((N, 2)), vt0=np.zeros(N), rec_dt=0.05)
    eps, e = run['eps'][-1], run['e'][-1]
    ebar = sim.par['ebar']
    L = np.diag(A.sum(1)) - A
    phi = eps * np.minimum(1.0, ebar / np.maximum(np.linalg.norm(eps, axis=1), 1e-300))[:, None]
    pred = np.linalg.solve(lam * np.eye(N) + Lam * L, lam * eps + Lam * A.sum(1)[:, None] * (eps - phi))
    out = dict(graph=gname, Lam=Lam, lam=lam, net=net, slip=slip, h=run['meta']['h'], eps1_mm=float(1e3 * np.linalg.norm(eps[0])),
               capped=bool(np.linalg.norm(eps[0]) > ebar), e_sim_mm=(1e3 * e[:, 0]).tolist(), e_pred_mm=(1e3 * pred[:, 0]).tolist(),
               max_dev_mm=float(1e3 * np.abs(e - pred).max()), form_mm=float(1e3 * cs.formation_error(e)),
               form_uncoupled_mm=float(1e3 * cs.formation_error(eps)), settled=float(np.abs(run['e'][-1] - run['e'][-200]).max()))
    if np.allclose(A, A.T) and not out['capped']:
        l2 = float(np.sort(np.linalg.eigvalsh(L))[1])
        dis = lambda x: np.linalg.norm(x - x.mean(0))
        out.update(l2=l2, disagreement_ratio=float(dis(e) / dis(eps)), predicted_factor=lam / (lam + Lam * l2),
                   mean_shift_mm=float(1e3 * np.abs(e.mean(0) - eps.mean(0)).max()))
    return out


def slip_job(a):
    Lam, net = a
    A = np.ones((3, 3)) - np.eye(3) if Lam > 0 else np.zeros((3, 3))
    f = 0.4
    def pert(t, v, om):
        dv, dom = np.zeros(3), np.zeros(3)
        if 30.0 <= t < 32.0:
            dv[1], dom[1] = -(1 - f) * v[1], -(1 - f) * om[1]
        return dv, np.zeros(3), dom
    sim, run, _ = run_case('warehouse', net, None, 7, par=dict(Lam=max(Lam, 1e-9)), pert=pert, pert_breaks=(30.0, 32.0), A=A,
                           dp0=np.zeros((3, 2)), vt0=np.zeros(3), rec_dt=0.01)
    t = run['t']
    form = cs.formation_error(run['e'])
    ne = np.linalg.norm(run['e'], axis=2)
    after = t >= 32.0
    def settle(x, thr):
        idx = np.nonzero((x > thr) & after)[0]
        return float(t[idx[-1]] - 32.0) if len(idx) else 0.0
    np.savez_compressed(RES + f'e5_slip_Lam{Lam:g}_{net}.npz', t=t, form=form, ne=ne, zeta=run['zeta'], p=run['p'], Pr=run['Pr'])
    return dict(Lam=Lam, net=net, h=run['meta']['h'], form_peak_mm=float(1e3 * form.max()), slip_err_peak_mm=float(1e3 * ne[:, 1].max()),
                others_peak_mm=float(1e3 * ne[:, [0, 2]].max()), t_form_5cm=settle(form, 0.05), t_track_5cm=settle(ne.max(1), 0.05),
                form_int=float(np.trapz(form[t >= 30], t[t >= 30])))


if __name__ == '__main__':
    t0 = time.time()
    sj = [(g, L, 1.0, net, -0.03) for g in ('complete3', 'chain3', 'path4', 'weighted3') for L in (0.0, 1.0, 3.0, 10.0) for net in ('nominal', 'slow3')]
    sj += [('complete3', 3.0, 0.3, 'slow3', -0.03), ('complete3', 1.0, 1.0, 'nominal', -0.12), ('path4', 3.0, 1.0, 'slow3', -0.12)]
    dj = [(L, net) for L in (0.0, 1.0, 3.0) for net in ('nominal', 'slow1')]
    with Pool(18) as pool:
        r1 = pool.map_async(static_job, sj, chunksize=1)
        r2 = pool.map_async(slip_job, dj, chunksize=1)
        out = dict(static=r1.get(), slip=r2.get())
    out['wall_s'] = time.time() - t0
    json.dump(out, open(RES + 'e5_formation.json', 'w'), indent=1, default=float)
    for r in out['static']:
        print({q: (round(v, 4) if isinstance(v, float) else ([round(x, 3) for x in v] if isinstance(v, list) else v)) for q, v in r.items()})
    for r in out['slip']:
        print({q: (round(v, 3) if isinstance(v, float) else v) for q, v in r.items()})
    print('wall', out['wall_s'])
