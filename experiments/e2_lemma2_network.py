"""E2. Lemma 2 (Halanay-type estimate) on the delayed filter network, simulated EXACTLY.

With newest-sample hold the delayed states are piecewise constant, chi_i and pi_i are chosen piecewise constant,
so between events  zeta_i' = chi_i [ -(lam + Lam d_i) zeta_i + c_i ]  has the closed-form solution used below.
|zeta_i| is convex along each segment, so the maxima over intervals are attained at event times: M and M_h are exact.
"""
import sys, json, time
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from multiprocessing import Pool
from agvsim import theory as th, network as nw


def simulate(A, lam, Lam, cfg, T, seed, chi0=1.0, inp='zero', Pbar=1.0, sw_rate=5.0, rec_dt=None):
    rng = np.random.default_rng(seed)
    A = np.asarray(A, float)
    N = A.shape[0]
    d = A.sum(1)
    sch = nw.Schedule(A, T, cfg, seed + 12345)
    zeta = rng.standard_normal((N, 2))
    zeta /= np.linalg.norm(zeta, axis=1).max()                     # M(0) = 1
    held = np.repeat(zeta[None, :, :], N, axis=0).copy()           # constant initial history
    ht = np.zeros((N, N))
    nsw = rng.poisson(sw_rate * T)
    ev = [(float(t), 2, 0, 0, 0) for t in np.sort(rng.uniform(0, T, nsw))]
    ev += [(float(sch.send_t[j][k]), 0, 0, j, k) for (j, k) in sch.needed_sends()]
    ev += [(float(ta), 1, i, j, k) for (ta, i, j, k) in sch.updates]
    if rec_dt:
        ev += [(float(t), 3, 0, 0, 0) for t in np.arange(0, T + rec_dt / 2, rec_dt)]
    ev.append((float(T), 3, 0, 0, 0))
    ev.sort()

    def draw(t):
        chi = rng.uniform(chi0, 1.0, N)
        chi[rng.random(N) < 0.3] = chi0                            # sit on the lower bound often
        if inp == 'zero':
            pi = np.zeros((N, 2))
        else:
            ang = rng.uniform(0, 2 * np.pi, N)
            amp = Pbar * rng.uniform(0, 1, N)
            amp[rng.integers(N)] = Pbar                            # sup P = Pbar is attained
            if inp == 'decay':
                amp = amp * np.exp(-0.3 * t)
            pi = amp[:, None] * np.stack([np.cos(ang), np.sin(ang)], 1)
        return chi, pi

    chi, pi = draw(0.0)
    samples = {}
    t = 0.0
    ts, Ms, Ps, hrun = [0.0], [np.linalg.norm(zeta, axis=1).max()], [np.linalg.norm(pi, axis=1).max()], 0.0
    rate_a = lam + Lam * d
    for (tb, kind, i, j, k) in ev:
        if tb > t:
            cbar = (Lam * np.einsum('ij,ijk->ik', A, held) + pi) / rate_a[:, None]
            zeta = cbar + (zeta - cbar) * np.exp(-chi * rate_a * (tb - t))[:, None]
            t = tb
            ts.append(t); Ms.append(np.linalg.norm(zeta, axis=1).max()); Ps.append(np.linalg.norm(pi, axis=1).max())
        if kind == 0:
            samples[(j, k)] = zeta[j].copy()
        elif kind == 1:
            hrun = max(hrun, t - ht[i, j])
            held[i, j] = samples[(j, k)]
            ht[i, j] = sch.send_t[j][k]
        elif kind == 2:
            chi, pi = draw(t)
            Ps[-1] = max(Ps[-1], np.linalg.norm(pi, axis=1).max())
    hrun = max(hrun, float((t - ht)[A > 0].max()))
    return np.array(ts), np.array(Ms), np.array(Ps), hrun, float(d.max())


def check(ts, Ms, Ps, h, d, lam, Lam, chi0):
    """Largest ratio M(t) / bound(t) over t0 in a set of start times and beta in a set of weights."""
    worst = 0.0
    for t0 in (0.0, 0.2 * ts[-1], 0.5 * ts[-1]):
        i0 = np.searchsorted(ts, t0)
        lo = np.searchsorted(ts, ts[i0] - h)
        Mh = Ms[lo:i0 + 1].max() if ts[i0] - h >= 0 else max(Ms[:i0 + 1].max(), Ms[0])
        Pc = np.maximum.accumulate(Ps[i0:])
        for beta in (0.1, 0.5, 0.9):
            eta = th.halanay_rate(lam, Lam, d, h, beta, chi0)
            bound = np.maximum(Mh * np.exp(-eta * (ts[i0:] - ts[i0])), Pc / (beta * lam))
            worst = max(worst, float((Ms[i0:] / np.maximum(bound, 1e-300)).max()))
    return worst


def one(args):
    kind, seed = args
    rng = np.random.default_rng(seed)
    N = int(rng.integers(3, 9))
    A = rng.uniform(0.2, 1.5, (N, N)) * (rng.random((N, N)) < rng.uniform(0.3, 0.9))
    np.fill_diagonal(A, 0.0)
    if A.sum() == 0:
        A[0, 1] = 1.0
    lam, Lam = 10 ** rng.uniform(-0.7, 0.7), 10 ** rng.uniform(-0.7, 0.7)
    chi0 = float(rng.choice([1.0, 0.5, 0.1]))
    scale = 10 ** rng.uniform(-1, 0.5)                           # latency scale: 0.1 s ... 3 s
    cfg = dict(Ts=0.05, lat=(scale, 0.2 * scale, 0.3 * scale, 1.6 * scale), spike=(0.05, 0.1 * scale, 0.5 * scale),
               ge=(0.02, 0.85, 0.05, 0.25))
    if kind == 'outage':                                          # delay jumps between ~0 and the blackout length
        cfg = dict(Ts=0.02, lat=(0.005, 0.0, 0.005, 0.005), spike=(0, 0, 0), ge=(0, 0, 0, 1), outage=(2 * scale + 0.5, 2 * scale))
        kind_inp = 'zero'
    else:
        kind_inp = kind
    T = 60.0
    ts, Ms, Ps, h, d = simulate(A, lam, Lam, cfg, T, seed, chi0=chi0, inp=kind_inp)
    return dict(kind=kind, seed=seed, N=N, lam=lam, Lam=Lam, chi0=chi0, h=h, d=d,
                ratio=check(ts, Ms, Ps, h, d, lam, Lam, chi0), M_end=float(Ms[-1]))


def rate_test(h, N=3, lam=1.0, Lam=1.0, T=None):
    """Constant delay on the complete graph: measured decay rate of M versus the rate of Lemma 2 (beta -> 0)."""
    A = np.ones((N, N)) - np.eye(N)
    eta = th.halanay_rate(lam, Lam, N - 1, h)
    T = T or min(max(25.0 / eta, 30.0), 250.0)
    rng = np.random.default_rng(0)
    cfg = nw.const_delay_net(h, Ts=0.002)
    sch = nw.Schedule(A, T, cfg, 1)
    zeta = np.ones((N, 2)) / np.sqrt(2)                            # consensus direction: slowest mode
    held = np.repeat(zeta[None], N, axis=0).copy()
    ev = [(float(sch.send_t[j][k]), 0, 0, j, k) for (j, k) in sch.needed_sends()] + [(float(ta), 1, i, j, k) for (ta, i, j, k) in sch.updates]
    ev.sort()
    t, ts, Ms, samples = 0.0, [], [], {}
    a = lam + Lam * (N - 1)
    for (tb, kind, i, j, k) in ev:
        if tb > t:
            cbar = Lam * np.einsum('ij,ijk->ik', A, held) / a
            zeta = cbar + (zeta - cbar) * np.exp(-a * (tb - t))
            t = tb
            ts.append(t); Ms.append(np.linalg.norm(zeta, axis=1).max())
        if kind == 0:
            samples[(j, k)] = zeta[j].copy()
        else:
            held[i, j] = samples.pop((j, k)) if False else samples[(j, k)]
    ts, Ms = np.array(ts), np.array(Ms)
    m = (ts > 0.4 * T) & (Ms > 1e-290)
    rate = float(-np.polyfit(ts[m], np.log(Ms[m]), 1)[0])
    return dict(h=h, rate_lemma2=eta, rate_measured=rate, rate_lemma2_hTs=th.halanay_rate(lam, Lam, N - 1, h + 0.002))


if __name__ == '__main__':
    t0 = time.time()
    jobs = [(kind, s) for kind in ('zero', 'bounded', 'decay', 'outage') for s in range(250)]
    with Pool(16) as pool:
        res = pool.map(one, jobs, chunksize=4)
        rates = pool.map(rate_test, [0.1, 0.3, 0.58, 1.0, 1.7, 3.44])
    summ = {}
    for kind in ('zero', 'bounded', 'decay', 'outage'):
        rr = [r for r in res if r['kind'] == kind]
        summ[kind] = dict(runs=len(rr), max_ratio=max(r['ratio'] for r in rr), violations=sum(r['ratio'] > 1 + 1e-9 for r in rr),
                          h_min=min(r['h'] for r in rr), h_max=max(r['h'] for r in rr))
    out = dict(summary=summ, rates=rates, wall_s=time.time() - t0)
    json.dump(out, open(os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', ''), 'e2_lemma2.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1))
