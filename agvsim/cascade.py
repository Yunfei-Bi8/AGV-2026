"""Closed loop of the cascade (reference-shift filter) design on the kinematic model of the theorems.

The control law runs in continuous time.  Received broadcasts are piecewise constant (newest-sample hold), so the
closed loop is an ordinary differential equation between network events; it is integrated with error control from
event to event.  The heading error vt_i = thetahat_i - theta_i is a state, so no angle is ever wrapped.

State vector: p (2N) | vt (N) | z (2N) | w (2N) | D (N) | pc (2)
  D_i integrates the dissipation (and the perturbation power), so that W_i(t) + D_i(t) = W_i(0) exactly.
"""
import numpy as np
from .theory import sinc
from .references import vehicle_refs
from .integrator import integrate, Stats

DEFAULT = dict(lam=1.0, Lam=1.0, gam=5.0, rho=0.6, ebar=0.05, psi=0.7, b=3.0)


class CascadeSim:
    def __init__(self, ref, offs, A, par, sched, pert=None, pert_breaks=(), bnoise=None, init_hold='inactive'):
        self.ref, self.offs = ref, np.asarray(offs, float)
        self.A = np.asarray(A, float)
        self.N = len(self.offs)
        self.par = dict(DEFAULT, **par)
        self.sched, self.pert, self.pert_breaks, self.bnoise = sched, pert, tuple(pert_breaks), bnoise
        self.init_hold = init_hold
        N = self.N
        self.hz = np.zeros((N, N, 2))      # held zeta_j
        self.he = np.zeros((N, N, 2))      # held eps_j (plus broadcast noise)
        self.hk = -np.ones((N, N), int)    # index of the held packet
        self.ht = np.zeros((N, N))         # its send time
        self.active = np.zeros((N, N), bool)
        self._refresh()

    def _refresh(self):
        Aa = self.A * self.active
        self.dact = Aa.sum(1)
        self.hsz = np.einsum('ij,ijk->ik', Aa, self.hz)
        self.hse = np.einsum('ij,ijk->ik', Aa, self.he)
        self.hsum = self.hsz + self.hse

    # ------------------------------------------------------------------ dynamics
    def core(self, t, y, full=False):
        N, P_ = self.N, self.par
        lam, Lam, gam, rho, ebar, psi, b = (P_[k] for k in ('lam', 'Lam', 'gam', 'rho', 'ebar', 'psi', 'b'))
        p = y[0:2 * N].reshape(N, 2)
        vt = y[2 * N:3 * N]
        z = y[3 * N:5 * N].reshape(N, 2)
        w = y[5 * N:7 * N].reshape(N, 2)
        cen = self.ref.centre(t, y[8 * N:8 * N + 2])
        Pr, cr, sr, vr, wr, vrd = vehicle_refs(cen, self.offs)
        eps = Pr - z - p
        zeta = z + w / gam
        ne = np.hypot(eps[:, 0], eps[:, 1])
        phi = eps * np.minimum(1.0, ebar / np.maximum(ne, 1e-300))[:, None]
        g = -lam * zeta - Lam * (self.dact[:, None] * (zeta + phi) - self.hsum)
        ng = np.hypot(g[:, 0], g[:, 1])
        chi = np.minimum(1.0, rho * vr / np.maximum(ng, 1e-300))
        u = g * chi[:, None]
        Vx, Vy = vr * cr - w[:, 0], vr * sr - w[:, 1]            # velocity of the shifted reference
        vh = np.hypot(Vx, Vy)
        ch, sh = Vx / vh, Vy / vh
        dwx, dwy = gam * (u[:, 0] - w[:, 0]), gam * (u[:, 1] - w[:, 1])
        Ax = vrd * cr - vr * wr * sr - dwx                        # its acceleration
        Ay = vrd * sr + vr * wr * cr - dwy
        wh = (Vx * Ay - Vy * Ax) / vh ** 2
        cv, sv = np.cos(vt), np.sin(vt)
        xx, xy = ch * cv + sh * sv, sh * cv - ch * sv             # xi(theta), theta = thetahat - vt
        mu = xx * eps[:, 0] + xy * eps[:, 1]
        nu = -xy * eps[:, 0] + xx * eps[:, 1]
        k = 2 * psi * np.sqrt(wh ** 2 + b * vh ** 2)
        v = vh * cv + k * mu                                      # the validated single-vehicle law
        om = wh + k * vt + b * vh * sinc(vt) * nu
        if self.pert is not None:
            dv, dn, dom = self.pert(t, v, om)
        else:
            dv = dn = dom = np.zeros(N)
        dy = np.empty_like(y)
        dp = dy[0:2 * N].reshape(N, 2)
        dp[:, 0] = (v + dv) * xx - dn * xy
        dp[:, 1] = (v + dv) * xy + dn * xx
        dvt = wh - om - dom
        dy[2 * N:3 * N] = dvt
        dy[3 * N:5 * N] = w.ravel()
        dw = dy[5 * N:7 * N].reshape(N, 2)
        dw[:, 0], dw[:, 1] = dwx, dwy
        dy[7 * N:8 * N] = k * mu ** 2 + (k / b) * vt ** 2 + mu * dv + nu * dn + vt * dom / b
        dy[8 * N] = cen[3] * cen[1]
        dy[8 * N + 1] = cen[3] * cen[2]
        if not full:
            return dy
        # derivatives of the body-frame errors from the simulated state derivatives (not from the theory)
        ex, ey = Vx - dp[:, 0], Vy - dp[:, 1]                     # d eps / dt
        thd = wh - dvt                                            # d theta / dt
        mud = thd * nu + xx * ex + xy * ey
        nud = -thd * mu - xy * ex + xx * ey
        pi = -Lam * self.dact[:, None] * phi + Lam * self.hse
        dlt = np.arctan2(cr * sh - sr * ch, cr * ch + sr * sh)    # thetahat - theta_r, in (-pi/2, pi/2)
        return dy, dict(W=0.5 * (mu ** 2 + nu ** 2) + vt ** 2 / (2 * b), mu=mu, nu=nu, vt=vt.copy(), mud=mud, nud=nud,
                        vtd=dvt, vh=vh, wh=wh, k=k, eps=eps, zeta=zeta, e=Pr - p, head=vt - dlt, g=ng, chi=chi,
                        pi=np.hypot(pi[:, 0], pi[:, 1]), D=y[7 * N:8 * N].copy(), w=w.copy(), z=z.copy(),
                        p=p.copy(), Pr=Pr, vr=vr, v=v, om=om, dv=dv, dn=dn, dom=dom,
                        pwr=mu * dv + nu * dn + vt * dom / b)

    def rhs(self, t, y):
        return self.core(t, y)

    # ------------------------------------------------------------------ simulation
    def initial_state(self, dp0, vt0):
        N = self.N
        y = np.zeros(8 * N + 2)
        pc0 = self.ref.pc0()
        y[8 * N:] = pc0
        Pr = vehicle_refs(self.ref.centre(0.0, pc0), self.offs)[0]
        y[0:2 * N] = (Pr - np.asarray(dp0, float)).ravel()         # e_i(0) = dp0_i
        y[2 * N:3 * N] = vt0
        return y

    def run(self, y0, T, rec_dt=0.01, rtol=1e-10, atol=1e-12):
        N, sch = self.N, self.sched
        ev = [(float(tr), 2, 0, 0, 0) for tr in np.arange(0.0, T + 0.5 * rec_dt, rec_dt)]
        if sch is not None:
            ev += [(float(sch.send_t[j][k]), 0, 0, j, k) for (j, k) in sch.needed_sends()]
            ev += [(float(ta), 1, i, j, k) for (ta, i, j, k) in sch.updates]
        ev += [(float(tb), 3, 0, 0, 0) for tb in self.pert_breaks if 0 < tb < T]
        ev.sort()
        stats = Stats()
        samples, rec, keys = {}, {}, None
        t, y = 0.0, y0.copy()
        if self.init_hold == 'sigma0':                               # convention of assumption A3 in the doc
            _, o = self.core(0.0, y, full=True)
            for i in range(N):
                for j in range(N):
                    if self.A[i, j] > 0:
                        self.hz[i, j], self.he[i, j], self.active[i, j] = o['zeta'][j], o['eps'][j], True
            self._refresh()
        hrun, tau_rec = 0.0, []
        idx, n = 0, len(ev)
        while idx < n:
            tb = ev[idx][0]
            if tb > t:
                y = integrate(self.rhs, t, y, tb, rtol, atol, stats)
                t = tb
            obs = None
            while idx < n and ev[idx][0] == tb:
                _, kind, i, j, k = ev[idx]
                idx += 1
                if kind == 0:
                    if obs is None:
                        _, obs = self.core(t, y, full=True)
                    e_s = obs['eps'][j].copy()
                    if self.bnoise is not None:
                        e_s += self.bnoise(j, k)
                    samples[(j, k)] = (obs['zeta'][j].copy(), e_s)
                elif kind == 1:
                    if self.active[i, j]:
                        hrun = max(hrun, t - self.ht[i, j])          # peak of the sawtooth, just before the update
                    self.hz[i, j], self.he[i, j] = samples[(j, k)]
                    self.hk[i, j], self.ht[i, j], self.active[i, j] = k, sch.send_t[j][k], True
                    self._refresh()
                    obs = None
                elif kind == 2:
                    _, o = self.core(t, y, full=True)
                    if keys is None:
                        keys = list(o.keys())
                        rec = {q: [] for q in keys}
                        rec['t'] = []
                    for q in keys:
                        rec[q].append(o[q])
                    rec['t'].append(t)
                    tau_rec.append(float((t - self.ht)[self.active].max()) if self.active.any() else 0.0)
        if self.active.any():
            hrun = max(hrun, float((t - self.ht)[self.active].max()))
        out = {q: np.array(v) for q, v in rec.items()}
        out['tau'] = np.array(tau_rec)
        out['meta'] = dict(h=hrun, steps=stats.steps, rejected=stats.rejected, fevals=stats.fevals,
                           err_sum=stats.err_sum, T=T, N=N, net=(sch.stats if sch is not None else None))
        out['y_end'] = y
        return out


def formation_error(e):
    """Largest formation error max_{i<j} |e_i - e_j| for an array e of shape (..., N, 2)."""
    N = e.shape[-2]
    f = [np.linalg.norm(e[..., i, :] - e[..., j, :], axis=-1) for i in range(N) for j in range(i)]
    return np.max(f, axis=0)
