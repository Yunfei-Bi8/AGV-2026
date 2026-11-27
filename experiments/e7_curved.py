"""E7. Proposition 1 on a curved reference (Remark 3 of the paper): AGV 1 permanently loses 0.03 m/s of speed, complete
graph of three, constant delays h (broadcasts every 20 ms, so that the held sample is h + 10 ms old on average),
Lam/lam = 1, 3, 10. On the straight line the local errors are constant, and the disagreement falls by the factor
lam/(lam + Lam lam_2) of Proposition 1. On the circle they rotate with the reference at w_r = 0.175 rad/s, and the
disagreement falls by |H(j w_r)| (agvsim.theory.filter_response), which depends on the delay."""
import json, time
import numpy as np
from multiprocessing import Pool
from common import run_case, RES
from agvsim import network as nw, theory as th, references as rf

TS = 0.02
DELAYS = [0.06, 1.0, 3.0]


def job(a):
    route, Lam, h = a
    lam, gam = 1.0, 5.0
    A = np.ones((3, 3)) - np.eye(3)

    def pert(t, v, om):
        dv = np.zeros(3)
        dv[0] = -0.03
        return dv, np.zeros(3), np.zeros(3)
    sim, run, _ = run_case(route, nw.const_delay_net(h, TS), None, 3, par=dict(Lam=Lam, lam=lam), pert=pert, A=A, T=150.0,
                           dp0=np.zeros((3, 2)), vt0=np.zeros(3), rec_dt=0.05)
    t = run['t']
    m = t >= 110.0
    dis = lambda x: float(np.sqrt(np.mean(np.sum((x[m] - x[m].mean(1, keepdims=True)) ** 2, axis=(1, 2)))))
    wr = float(rf.make_reference(route)[0].centre(120.0, np.zeros(2))[4])
    H = th.filter_response(wr, h + TS / 2, lam, Lam, gam, 2, -1)
    return dict(route=route, Lam=Lam, lam=lam, h=h, tau=h + TS / 2, w_r=wr, ratio=dis(run['e']) / dis(run['eps']),
                factor=lam / (lam + 3 * Lam), H_abs=float(abs(H)), eps_max_mm=float(1e3 * np.linalg.norm(run['eps'][m], axis=2).max()),
                chi_min=float(run['chi'][m].min()), h_run=run['meta']['h'])


if __name__ == '__main__':
    t0 = time.time()
    jobs = [(r, L, h) for r in ('line', 'circle') for L in (1.0, 3.0, 10.0) for h in DELAYS]
    with Pool(18) as pool:
        out = pool.map(job, jobs, chunksize=1)
    json.dump(dict(runs=out, wall_s=time.time() - t0), open(RES + 'e7_curved.json', 'w'), indent=1)
    for r in out:
        print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})
