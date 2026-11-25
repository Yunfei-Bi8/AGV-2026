"""E6. Direct-coupling baseline (delayed neighbour errors inside the tracking law) and its Lyapunov-Krasovskii functional.
Closed loop: the direct coupling of Proposition 3 of the paper. Delays: sawtooth (newest-sample hold) per directed link.
Checks:  (i) V non-increasing,  (ii) V(t)+eta*int sum D <= V(0),  (iii) convergence.
"""
import os
import numpy as np, sys

def sinc(x):
    return np.sinc(x/np.pi)

def run(seed=0, N=3, T=40.0, dt=1e-3, k1=1.0, k2=3.0, k3=3.0, c=0.5, A=None,
        ref='circle', maxdelay_target=0.9, loss=0.3, verbose=True):
    rng = np.random.default_rng(seed)
    if A is None:
        A = np.ones((N, N)) - np.eye(N)
    d = A.sum(1); dmax = d.max()
    # references: circles about origin, common angular rate (virtual structure)
    wc = 0.7/4.0
    offs = np.array([[0.6, 0.0], [-0.4, 0.5], [-0.4, -0.5], [-1.2, 0.0], [0.2, 1.0]])[:N]
    # structure centre at radius 4, heading tangent; offsets in structure frame
    def ref_state(t):
        th_c = wc*t + np.pi/2
        pc = 4.0*np.array([np.cos(wc*t), np.sin(wc*t)])
        R = np.array([[np.cos(th_c), -np.sin(th_c)], [np.sin(th_c), np.cos(th_c)]])
        P = pc[None, :] + offs @ R.T
        # each point moves on a circle about the origin with rate wc
        r = np.linalg.norm(P, axis=1)
        vr = wc*r
        thr = np.arctan2(P[:, 1], P[:, 0]) + np.pi/2
        wr = wc*np.ones(N)
        return P, thr, vr, wr
    P0, th0, vr0, _ = ref_state(0.0)
    vmax = (wc*np.linalg.norm(P0, axis=1)).max()
    kappa = max(k1, k2*vmax**2/k3)
    hstar = 1.0/(c*dmax*kappa)
    # initial vehicle states
    p = P0 + rng.uniform(-0.3, 0.3, (N, 2))
    th = th0 + rng.uniform(-0.3, 0.3, N)
    # unwrap reference heading continuously: use analytic form (no wrap needed, thr = atan2+pi/2 may wrap)
    # we keep heading error as real number: track reference heading by integration
    thr_c = th0.copy()
    n = int(round(T/dt))
    # network: each directed link i<-j: packets sent every Ts, latency random, loss bursts
    Ts = 0.05
    held = np.zeros((N, N, 2))          # held[i,j] = newest e_j received by i
    held_time = np.zeros((N, N))        # send time of the newest packet held
    e0 = P0 - p
    for i in range(N):
        for j in range(N):
            held[i, j] = e0[j]          # constant initial history
    pending = []                        # (arrival_step, i, j, send_time, value)
    maxdelay = 0.0
    hist_edot2 = np.zeros((n+1, N))
    Vs = np.zeros(n+1); Ds = np.zeros(n+1); taus = np.zeros(n+1)
    e_hist = []
    def f(p, th, t, held):
        P, thr, vr, wr = ref_state(t)
        e = P - p
        return e, vr, wr
    def deriv(state, t, thr_c, held):
        p = state[:2*N].reshape(N, 2); th = state[2*N:3*N]; thr = state[3*N:]
        P, _, vr, wr = ref_state(t)
        e = P - p
        the = thr - th
        s = e + c*(d[:, None]*e - np.einsum('ij,ijk->ik', A, held))
        xi = np.stack([np.cos(th), np.sin(th)], 1); xip = np.stack([-np.sin(th), np.cos(th)], 1)
        mu = (xi*s).sum(1); nu = (xip*s).sum(1)
        v = vr*np.cos(the) + k1*mu
        w = wr + k3*the + k2*vr*sinc(the)*nu
        dp = v[:, None]*xi
        edot = -k1*mu[:, None]*xi + (vr*np.sin(the))[:, None]*xip
        D = k1*mu**2 + (k3/k2)*the**2
        return np.concatenate([dp.ravel(), w, wr]), e, the, edot, D
    state = np.concatenate([p.ravel(), th, thr_c])
    burst = np.zeros((N, N), dtype=bool)
    for k in range(n+1):
        t = k*dt
        # deliver packets arriving now
        newpend = []
        for (ks, i, j, ts, val) in pending:
            if ks <= k:
                if ts > held_time[i, j]:
                    held[i, j] = val; held_time[i, j] = ts
            else:
                newpend.append((ks, i, j, ts, val))
        pending = newpend
        _, e, the, edot, D = deriv(state, t, None, held)
        # send packets
        if k % int(round(Ts/dt)) == 0:
            for i in range(N):
                for j in range(N):
                    if A[i, j] > 0:
                        # Gilbert-Elliott style bursts
                        if burst[i, j]:
                            if rng.random() < 0.35: burst[i, j] = False
                        else:
                            if rng.random() < loss*0.2: burst[i, j] = True
                        lost = rng.random() < (0.8 if burst[i, j] else 0.02)
                        # cap effective delay: force delivery if age would exceed target
                        age_if_lost = t + Ts + 0.2 - held_time[i, j]
                        if lost and age_if_lost < maxdelay_target:
                            continue
                        lat = float(np.clip(0.06 + 0.02*rng.standard_normal(), 0.005, 0.2))
                        if rng.random() < 0.03: lat += rng.uniform(0.05, 0.15)
                        lat = min(lat, 0.2)
                        pending.append((k + max(1, int(round(lat/dt))), i, j, t, e[j].copy()))
        tau_now = (t - held_time)[A > 0].max()
        maxdelay = max(maxdelay, tau_now); taus[k] = tau_now
        hist_edot2[k] = (edot**2).sum(1)
        e_hist.append((e.copy(), the.copy()))
        Ds[k] = D.sum()
        if k == n: break
        # RK4 step with held values constant
        def F(st, tt):
            return deriv(st, tt, None, held)[0]
        k1_ = F(state, t); k2_ = F(state + 0.5*dt*k1_, t + 0.5*dt)
        k3_ = F(state + 0.5*dt*k2_, t + 0.5*dt); k4_ = F(state + dt*k3_, t + dt)
        state = state + dt/6*(k1_ + 2*k2_ + 2*k3_ + k4_)
    h = np.ceil(maxdelay*100)/100
    nh = int(round(h/dt))
    eta = 1 - h/hstar
    # evaluate V
    wts = np.arange(nh+1)*dt   # weight (sigma - t + h) for sigma in [t-h,t]
    for k in range(n+1):
        e, the = e_hist[k]
        V1 = 0.5*(e**2).sum()
        V2 = 0.25*c*sum(A[i, j]*((e[i]-e[j])**2).sum() for i in range(N) for j in range(N))
        V3 = (the**2).sum()/(2*k2)
        lo = k - nh
        if lo >= 0:
            seg = hist_edot2[lo:k+1]
            w_ = wts
        else:
            seg = hist_edot2[0:k+1]          # history before t=0 is constant => edot=0
            w_ = wts[-(k+1):]
        V4 = 0.5*c*(d[None, :]*np.trapezoid(seg*w_[:, None], dx=dt, axis=0)[None, :]).sum()
        Vs[k] = V1 + V2 + V3 + V4
    cumD = np.concatenate([[0], np.cumsum(0.5*(Ds[1:]+Ds[:-1])*dt)])
    stride = 10
    Vq = Vs[::stride]
    rise = np.diff(Vq).max()/Vs[0]
    Q = Vs + eta*cumD
    riseQ = (Q - Q[0]).max()/Vs[0]
    eN = np.array([np.linalg.norm(eh[0], axis=1).max() for eh in e_hist])
    out = dict(hstar=hstar, h=h, eta=eta, maxdelay=maxdelay, V0=Vs[0], Vend=Vs[-1], max_rel_rise=rise,
               max_rel_excess_Q=riseQ, e_end=eN[-1], e0=eN[0])
    if verbose: print(out)
    return out


def crit_delay(k2v2, k3, c, mu_adj, dself):
    """Smallest delay at which s^2 + k3 s + k2v2 [(1 + c dself) - c mu_adj exp(-s tau)] = 0 has a root on the imaginary axis."""
    from scipy.optimize import brentq
    a, bq = k2v2 * (1 + c * dself), -k2v2 * c * mu_adj
    # s = j w:  -w^2 + a + bq cos(w tau) = 0,  k3 w - bq sin(w tau) = 0  =>  (w^2 - a)^2 + (k3 w)^2 = bq^2
    f = lambda w: (w * w - a) ** 2 + (k3 * w) ** 2 - bq * bq
    ws = np.linspace(1e-6, 4 * np.sqrt(a + abs(bq)), 20001)
    best = np.inf
    for w0, w1 in zip(ws[:-1], ws[1:]):
        if f(w0) * f(w1) < 0:
            w = brentq(f, w0, w1)
            th = np.arctan2(k3 * w / bq, (w * w - a) / bq) % (2 * np.pi)
            best = min(best, th / w)
    return best


def job(a):
    kind, seed = a
    if kind == 'complete3':
        return dict(kind=kind, seed=seed, **{k: float(v) for k, v in run(seed=seed, maxdelay_target=0.9, verbose=False).items()})
    A = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]], float)
    return dict(kind=kind, seed=seed, **{k: float(v) for k, v in run(seed=seed, N=4, A=A, c=0.4, maxdelay_target=1.1, verbose=False).items()})


if __name__ == '__main__':
    import json, time
    from multiprocessing import Pool
    t0 = time.time()
    with Pool(18) as pool:
        res = pool.map(job, [(k, s) for k in ('complete3', 'ring4') for s in range(20)], chunksize=1)
    summ = {}
    for kind in ('complete3', 'ring4'):
        rr = [r for r in res if r['kind'] == kind]
        summ[kind] = dict(runs=len(rr), hstar=rr[0]['hstar'], h_min=min(r['h'] for r in rr), h_max=max(r['h'] for r in rr),
                          max_rel_rise=max(r['max_rel_rise'] for r in rr), max_rel_excess_Q=max(r['max_rel_excess_Q'] for r in rr),
                          e_end_max=max(r['e_end'] for r in rr))
    # tightness: ring of four, gains (k1,k2,k3,c) = (1,4,0.3,3), straight line at 0.5 m/s: alternating mode (adjacency eigenvalue -2)
    k2v2, k3, c = 4 * 0.25, 0.3, 3.0
    tight = dict(hstar=1.0 / (c * 2 * max(1.0, k2v2 / k3)), tau_crit=crit_delay(k2v2, k3, c, -2.0, 2.0))
    tight['ratio'] = tight['tau_crit'] / tight['hstar']
    tight['ratios_k3'] = [dict(k3=x, ratio=crit_delay(k2v2, x, c, -2.0, 2.0) * (c * 2 * max(1.0, k2v2 / x))) for x in (1.0, 0.3, 0.1, 0.03, 0.01)]
    out = dict(summary=summ, tightness=tight, wall_s=time.time() - t0)
    json.dump(out, open(os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', ''), 'e6_direct.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1))
