"""High-fidelity sampled-data simulation of N differential-drive AGVs for the robustness / sim-to-real study.

Layers (each one can be switched to its ideal version):

* Controller  : runs at f_c on the common time grid and evaluates its reference at the vehicle's local time (clock
                offset + drift); uses its own pose ESTIMATE and the newest received neighbour sample; commands reach
                the drives after an actuation delay.
* Drives      : (v, omega) -> wheel set-points with the NOMINAL geometry, curvature-preserving speed limit, set-point
                ramp limit, PI wheel-speed loops with torque saturation and anti-windup.
* Vehicle     : rigid-body dynamics of a differential-drive AGV in wheel coordinates (mass, yaw inertia, wheel
                inertia, payload), true wheel radii and track width (calibration errors), rolling / viscous / turning
                resistance; wheel slip (longitudinal slip ratio and lateral slip velocity, Ornstein-Uhlenbeck processes)
                acts on the pose kinematics and is invisible to odometry.
* Localisation: LiDAR-type fixes at f_fix with latency, white noise, slowly varying bias and outliers, fused with
                wheel odometry (nominal geometry) at the controller rate, as an AMCL+odometry stack does.
* Network     : periodic broadcasts, per-packet latency (Gaussian with spikes, log-normal, or the sinusoidal delays of
                Peng-Ngo-Liu 2023), Gilbert-Elliott burst loss, outages, random link switching, newest-sample hold.
                The schedule does not depend on the transmitted values, so all controllers see the same realisation.

Random numbers: one SeedSequence per seed, spawned into independent streams (initial errors, network, sensors,
disturbances, clocks, vehicle parameters). The number of draws does not depend on the controller, so all controllers
see the same realization (paired comparisons with common random numbers). The same holds across sweeps of latency,
loss and noise magnitudes; changing a rate (control, broadcast or fix rate) or the graph changes the number of draws
and hence the realization.

Timing: broadcasts and localization fixes are scheduled on the control grid. A broadcast rate fb is realized as one
broadcast every round(fc/fb) control periods, a fix rate f_fix as one fix every round(fc/f_fix) periods, and a fix
latency d_fix as round(d_fix*fc) periods (Python rounds halves to even). With the default fc = 50 Hz, fb = 20 Hz,
f_fix = 20 Hz and d_fix = 0.05 s, the realized values are 25 Hz, 25 Hz and 40 ms. `realized_timing` returns them, and
every run stores them in meta['realized'].
"""
import copy
import warnings
import numpy as np
from scipy.interpolate import CubicSpline
from .theory import sinc


def realized_timing(fc, fb, f_fix=None, d_fix=None):
    """Broadcast rate, fix rate and fix latency that the control grid at fc actually realizes (see module docstring)."""
    nb = max(1, int(round(fc / fb)))
    out = dict(fb=fc / nb)
    if f_fix is not None:
        out['f_fix'] = fc / max(1, int(round(fc / f_fix)))
    if d_fix is not None:
        out['d_fix'] = int(round(d_fix * fc)) / fc
    return out
from .references import vehicle_refs, vehicle_refs_rot, make_scenario

G = 9.81


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------- default configuration (one industrial AGV class)
CTRL = dict(type='cascade', psi=0.7, b=3.0, lam=1.0, Lam=1.0, gam=5.0, rho=0.6, ebar=0.05, c=0.5, fc=50.0,
            watchdog=None)
PLANT = dict(model='dyn', dt=0.005, m=150.0, payload=50.0, Lb=1.0, Wb=0.6, r=0.10, B=0.50, Jw=0.05, vw_max=1.5,
             a_set=1.0, a_stop=1.5, tau_max=20.0, Kp=15.0, Ki=60.0, c_rr=0.015, c_v=5.0, c_w=5.0, c_turn=10.0,
             act_delay=0.01, payload_spread=50.0, cal_r=0.01, cal_B=0.01, slip_mean=0.005, slip_sd=0.005, slip_tau=2.0,
             lat_sd=0.005, lat_tau=1.0)
SENSOR = dict(mode='fusion', f_fix=20.0, d_fix=0.05, sd_p=0.01, sd_th=np.deg2rad(0.3), bias_sd=0.01, bias_tau=20.0,
              bias_th_sd=np.deg2rad(0.2), p_out=0.0, out_mag=0.10)
NET = dict(fb=20.0, lat='gauss', lat_mean=0.03, lat_sd=0.015, lat_min=0.005, lat_max=0.2, spike_p=0.03,
           spike_lo=0.05, spike_hi=0.30, ge=(0.01, 0.6, 0.02, 0.4), scale=1.0, outages=(), switching=None,
           sinus=((0.5, 0.4), (0.3, 0.2), (0.4, 0.3)))
CLOCK = dict(offset=0.005, drift=20e-6)
SIM = dict(t_ss=15.0, init=(0.1, 0.3, np.deg2rad(15.0)), fail_err=3.0, rec_every=2, keep=False, form_thr=0.02)

PROFILES = {
    # the model of the theorems: kinematics, exact pose, common clock, fast control
    'ideal': dict(plant=dict(model='kin', act_delay=0.0, cal_r=0.0, cal_B=0.0, slip_mean=0.0, slip_sd=0.0,
                             lat_sd=0.0),
                  sensor=dict(mode='ideal'), clock=dict(offset=0.0, drift=0.0), ctrl=dict(fc=200.0)),
    # nominal realistic industrial AGV (default values above)
    'realistic': dict(),
}

NETS = {
    'ideal': dict(lat='const', lat_mean=0.0, ge=(0.0, 0.0, 0.0, 1.0), spike_p=0.0),
    'wifi': dict(),
    'paper': dict(lat='sinus', ge=(0.0, 0.0, 0.0, 1.0), spike_p=0.0, fb=50.0),
    'slow1': dict(lat='gauss', lat_mean=1.0, lat_sd=0.1, lat_min=0.7, lat_max=1.4, ge=(0.02, 0.85, 0.04, 0.25)),
    'slow3': dict(lat='gauss', lat_mean=3.0, lat_sd=0.15, lat_min=2.6, lat_max=3.3, ge=(0.02, 0.85, 0.04, 0.25)),
}

GRAPHS = {}


def graph(name, N):
    """Weight matrices; row i lists the vehicles that i listens to."""
    A = np.zeros((N, N))
    if name == 'complete':
        A = np.ones((N, N)) - np.eye(N)
    elif name == 'none':
        pass
    elif name == 'ring':                      # undirected ring
        for i in range(N):
            A[i, (i + 1) % N] = A[i, (i - 1) % N] = 1.0
    elif name == 'dring':                     # directed ring
        for i in range(N):
            A[i, (i - 1) % N] = 1.0
    elif name == 'chain':                     # directed chain 0 -> 1 -> ... (spanning tree rooted at 0)
        for i in range(1, N):
            A[i, i - 1] = 1.0
    elif name == 'star':                      # everybody listens to vehicle 0 only
        A[1:, 0] = 1.0
    elif name == 'complete_norm':             # complete graph with degree-normalised weights (row sums 1)
        A = (np.ones((N, N)) - np.eye(N)) / max(N - 1, 1)
    else:
        raise ValueError(name)
    return A


def merged(base, *over):
    out = copy.deepcopy(base)
    for o in over:
        if o:
            out.update(o)
    return out


# ---------------------------------------------------------------- references with arbitrary-time evaluation
_PC_CACHE = {}


class RefEval:
    def __init__(self, scen, T):
        self.scen, self.ref, self.offs, self.phi = scen, scen['ref'], np.asarray(scen['offs'], float), scen['phi']
        self.N = len(self.offs)
        self.needs_pc = getattr(self.ref, 'needs_pc_state', False)
        if self.needs_pc:
            key = (scen['name'], round(T + 2.0, 3))
            if key not in _PC_CACHE:
                dt = 1e-3
                ts = np.arange(0.0, T + 2.0 + dt / 2, dt)
                pc = np.zeros((len(ts), 2))
                pc[0] = self.ref.pc0()
                f = lambda t, p: (lambda c: np.array([c[3] * c[1], c[3] * c[2]]))(self.ref.centre(t, p))
                for k in range(len(ts) - 1):
                    t, p = ts[k], pc[k]
                    k1 = f(t, p); k2 = f(t + dt / 2, p + dt / 2 * k1); k3 = f(t + dt / 2, p + dt / 2 * k2)
                    k4 = f(t + dt, p + dt * k3)
                    pc[k + 1] = p + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                _PC_CACHE[key] = (CubicSpline(ts, pc[:, 0]), CubicSpline(ts, pc[:, 1]))
            self.spx, self.spy = _PC_CACHE[key]

    def state(self, t):
        t = max(t, 0.0)
        cen = self.ref.centre(t, np.array([float(self.spx(t)), float(self.spy(t))])) if self.needs_pc else self.ref.centre(t)
        if self.phi is None:
            P, cr, sr, vr, wr, vrd = vehicle_refs(cen, self.offs)
        else:
            P, cr, sr, vr, wr, vrd = vehicle_refs_rot(cen, *self.phi(t), self.offs)
        return P, np.arctan2(sr, cr), vr, wr, vrd

    def state_local(self, tl):
        """Per-vehicle evaluation at local times tl (N,): P, theta_r, v_r, omega_r, vdot_r."""
        if np.all(tl == tl[0]):
            return self.state(tl[0])
        out = [self.state(t) for t in tl]
        idx = np.arange(self.N)
        P = np.stack([out[i][0][i] for i in idx])
        return (P,) + tuple(np.array([out[i][q][i] for i in idx]) for q in range(1, 5))


# ---------------------------------------------------------------- network schedule
class Network:
    """Pre-computed packet schedule.  Broadcasts happen at controller steps k = phase_j + m * nb."""

    def __init__(self, A, K, fc, cfg, rng):
        N = A.shape[0]
        self.N, self.K, self.Tc = N, K, 1.0 / fc
        nb = max(1, int(round(fc / cfg['fb'])))
        self.nb = nb
        self.phase = rng.integers(0, nb, N)
        self.links = []
        sw = cfg.get('switching')
        for i in range(N):
            for j in range(N):
                if A[i, j] <= 0:
                    continue
                ks = np.arange(self.phase[j], K, nb)
                n = len(ks)
                ts = ks * self.Tc
                # fixed number of draws per packet (common random numbers across parameter values)
                z = rng.standard_normal(n); u_sp = rng.random(n); u_sp2 = rng.random(n); u_ge = rng.random(n)
                u_loss = rng.random(n); u_sw = rng.random(n)
                lat = self._latency(cfg, ts, j, z, u_sp, u_sp2) * cfg.get('scale', 1.0)
                pg, pb, gb, bg = cfg['ge']
                lost = np.zeros(n, bool)
                bad = False
                for m in range(n):
                    bad = (u_ge[m] >= bg) if bad else (u_ge[m] < gb)
                    lost[m] = u_loss[m] < (pb if bad else pg)
                if sw is not None:                       # random link switching: Markov on/off per packet
                    on_mean, off_mean = sw
                    p_off = self.Tc * nb / on_mean; p_on = self.Tc * nb / off_mean
                    on = True
                    for m in range(n):
                        on = (u_sw[m] >= p_off) if on else (u_sw[m] < p_on)
                        lost[m] |= not on
                for (t0, dur, src, dst) in cfg.get('outages', ()):
                    if (src in ('all', j)) and (dst in ('all', i)):
                        lost |= (ts >= t0) & (ts < t0 + dur)
                arr = ts + lat
                keep = ~lost
                order = np.argsort(arr[keep], kind='stable')
                a_t, a_k = arr[keep][order], ks[keep][order]
                acc_t, acc_k, newest = [], [], -1
                for t_, k_ in zip(a_t, a_k):           # newest-sample hold: discard late / reordered packets
                    if k_ > newest:
                        acc_t.append(t_); acc_k.append(k_); newest = k_
                self.links.append(dict(i=i, j=j, w=A[i, j], arr=np.array(acc_t), ks=np.array(acc_k, int), ptr=0,
                                       lat=lat[keep], sent=n, lost=int(lost.sum())))
        lats = np.concatenate([l['lat'] for l in self.links]) if self.links else np.zeros(1)
        sent = sum(l['sent'] for l in self.links); lost = sum(l['lost'] for l in self.links)
        self.stats = dict(sent=sent, loss=lost / max(sent, 1), lat_mean=float(lats.mean()) if len(lats) else 0.0,
                          lat_max=float(lats.max()) if len(lats) else 0.0)

    @staticmethod
    def _latency(cfg, ts, j, z, u_sp, u_sp2):
        kind = cfg['lat']
        if kind == 'const':
            return np.full(len(ts), cfg['lat_mean'])
        if kind == 'gauss':
            lat = np.clip(cfg['lat_mean'] + cfg['lat_sd'] * z, cfg['lat_min'], cfg['lat_max'])
            lat += np.where(u_sp < cfg['spike_p'], cfg['spike_lo'] + (cfg['spike_hi'] - cfg['spike_lo']) * u_sp2, 0.0)
            return lat
        if kind == 'lognorm':                     # lat_min + median * exp(sd * z)
            return cfg['lat_min'] + cfg['lat_mean'] * np.exp(cfg['lat_sd'] * z)
        if kind == 'sinus':                       # Peng-Ngo-Liu 2023: delay depends on the sender
            a, b = cfg['sinus'][j % len(cfg['sinus'])]
            return a + b * np.sin(ts / 2.0)
        raise ValueError(kind)


# ---------------------------------------------------------------- the simulation
class HifiSim:
    def __init__(self, scen_name, A, ctrl=None, plant=None, sensor=None, net=None, clock=None, sim=None,
                 profile='realistic', netname='wifi', seed=0, events=(), T=None, init_override=None):
        prof = PROFILES[profile]
        self.ctrl = merged(CTRL, prof.get('ctrl'), ctrl)
        self.plant = merged(PLANT, prof.get('plant'), plant)
        self.sensor = merged(SENSOR, prof.get('sensor'), sensor)
        self.clock = merged(CLOCK, prof.get('clock'), clock)
        self.net = merged(NET, NETS[netname], net)
        self.sim = merged(SIM, sim)
        self.scen = make_scenario(scen_name)
        self.T = T or self.scen['T']
        self.ref = RefEval(self.scen, self.T)
        self.N = self.ref.N
        self.A = np.asarray(A, float)
        self.events = events
        self.seed = seed
        self.init_override = init_override

    # ------------------------------------------------------------ helpers
    def _controller(self, P, thr, vr, wr, vrd, pe, the, hold_sum, dact, z, w):
        """Returns v, om, broadcast value, filter input u, aux."""
        C = self.ctrl
        typ, psi, b = C['type'], C['psi'], C['b']
        N = self.N
        c_r, s_r = np.cos(thr), np.sin(thr)
        if typ in ('cascade', 'sigleak'):
            lam, Lam, gam, rho, ebar = C['lam'], C['Lam'], C['gam'], C['rho'], C['ebar']
            eps = P - z - pe
            zeta = z + w / gam
            if typ == 'cascade':
                ne = np.hypot(eps[:, 0], eps[:, 1])
                phi = eps * np.minimum(1.0, ebar / np.maximum(ne, 1e-300))[:, None]
                g = -lam * zeta - Lam * (dact[:, None] * (zeta + phi) - hold_sum)
            else:                                   # literal transfer of the CEP filter: leak on sigma, no cap
                sig = zeta + eps
                g = -lam * sig - Lam * (dact[:, None] * sig - hold_sum)
            ng = np.hypot(g[:, 0], g[:, 1])
            chi = np.minimum(1.0, rho * vr / np.maximum(ng, 1e-300))
            u = g * chi[:, None]
            Vx, Vy = vr * c_r - w[:, 0], vr * s_r - w[:, 1]
            Ax = vrd * c_r - vr * wr * s_r - gam * (u[:, 0] - w[:, 0])
            Ay = vrd * s_r + vr * wr * c_r - gam * (u[:, 1] - w[:, 1])
            vref = np.hypot(Vx, Vy)
            thref = np.arctan2(Vy, Vx)
            wref = (Vx * Ay - Vy * Ax) / vref ** 2
            err = eps
            bcast = zeta + eps
        else:                                       # 'uncoupled' or 'direct'
            u = np.zeros((N, 2))
            e = P - pe
            if typ == 'direct':
                err = e + C['c'] * (dact[:, None] * e - hold_sum)
            else:
                err = e
            vref, thref, wref = vr, thr, wr
            bcast = e
        cx, sx = np.cos(the), np.sin(the)
        mu = cx * err[:, 0] + sx * err[:, 1]
        nu = -sx * err[:, 0] + cx * err[:, 1]
        vt = wrap(thref - the)
        k = 2 * psi * np.sqrt(wref ** 2 + b * vref ** 2)
        v = vref * np.cos(vt) + k * mu
        om = wref + k * vt + b * vref * sinc(vt) * nu
        return v, om, bcast, u

    # ------------------------------------------------------------ run
    def run(self):
        C, Pl, S, Nc, Ck, Sm = self.ctrl, self.plant, self.sensor, self.net, self.clock, self.sim
        N, T = self.N, self.T
        fc = C['fc']; Tc = 1.0 / fc
        K = int(round(T * fc))
        kin = Pl['model'] == 'kin'
        dt = Tc if kin else Pl['dt']
        n_sub = max(1, int(round(Tc / dt))); dt = Tc / n_sub
        ss = np.random.SeedSequence(int(self.seed))
        r_init, r_net, r_sens, r_dist, r_clk, r_cal = [np.random.default_rng(s) for s in ss.spawn(6)]

        # ---------------- initial state
        P0, thr0, vr0, wr0, vrd0 = self.ref.state(0.0)
        rmin, rmax, hmax = Sm['init']
        ang = r_init.uniform(0, 2 * np.pi, N); rad = r_init.uniform(rmin, rmax, N); dth = r_init.uniform(-hmax, hmax, N)
        if self.init_override is not None:
            dp, dth = self.init_override
        else:
            dp = np.stack([rad * np.cos(ang), rad * np.sin(ang)], 1)
        p = P0 - dp
        th = thr0 - dth
        # ---------------- vehicle parameters (true values)
        M = Pl['m'] + np.maximum(Pl['payload'] + r_cal.uniform(-1, 1, N) * Pl.get('payload_spread', 0.0), 0.0)
        I = M * (Pl['Lb'] ** 2 + Pl['Wb'] ** 2) / 12.0
        epsR = r_cal.uniform(-1, 1, N) * Pl['cal_r']; epsL = r_cal.uniform(-1, 1, N) * Pl['cal_r']
        epsB = r_cal.uniform(-1, 1, N) * Pl['cal_B']
        rR, rL, Bt = Pl['r'] * (1 + epsR), Pl['r'] * (1 + epsL), Pl['B'] * (1 + epsB)
        aR, aL, bR, bL = rR / 2, rL / 2, rR / Bt, rL / Bt
        Mw = np.empty((N, 2, 2))
        Mw[:, 0, 0] = M * aR ** 2 + I * bR ** 2 + Pl['Jw']
        Mw[:, 1, 1] = M * aL ** 2 + I * bL ** 2 + Pl['Jw']
        Mw[:, 0, 1] = Mw[:, 1, 0] = M * aR * aL - I * bR * bL
        Minv = np.linalg.inv(Mw)
        r_n, B_n = Pl['r'], Pl['B']
        # clocks
        off = r_clk.uniform(-1, 1, N) * Ck['offset']; drift = r_clk.uniform(-1, 1, N) * Ck['drift']
        # ---------------- network
        net = Network(self.A, K, fc, Nc, r_net)
        # ---------------- states
        wR = np.zeros(N); wL = np.zeros(N); IR = np.zeros(N); IL = np.zeros(N)
        spR = np.zeros(N); spL = np.zeros(N)                     # active (ramped) wheel set-points
        tgtR = np.zeros(N); tgtL = np.zeros(N)                   # target set-points from the controller
        v_kin = np.zeros(N); w_kin = np.zeros(N)
        kap = np.full(N, Pl['slip_mean']); dlat = np.zeros(N)
        odo = np.stack([p[:, 0], p[:, 1], th], 1).copy()        # odometry starts at the true initial pose
        z = np.zeros((N, 2)); w = np.zeros((N, 2))
        hold_z = np.zeros((N, N, 2)); hold_k = -np.ones((N, N), int); active = np.zeros((N, N), bool)
        bc_hist = np.zeros((K + 1, N, 2)); e_hist = np.zeros((K + 1, N, 2))
        odo_hist = np.zeros((K + 1, N, 3))
        # localisation
        n_fix = max(1, int(round(fc / S['f_fix']))) if S['mode'] != 'ideal' else 1
        n_dfix = int(round(S['d_fix'] * fc)) if S['mode'] != 'ideal' else 0
        realized = dict(fb=fc / net.nb)
        if S['mode'] != 'ideal':
            realized.update(f_fix=fc / n_fix, d_fix=n_dfix / fc)
        req = dict(fb=Nc['fb'], f_fix=S['f_fix'], d_fix=S['d_fix'])
        off_grid = [k for k in realized if abs(realized[k] - req[k]) > 1e-9 * max(1.0, abs(req[k]))]
        if off_grid:
            warnings.warn('timing quantized to the control grid (fc = %g Hz): ' % fc
                          + ', '.join('%s %g -> %g' % (k, req[k], realized[k]) for k in off_grid), stacklevel=2)
        fixes = {}                                               # k_fix -> (N,3) measured pose
        bias = np.zeros((N, 3))
        last_fix_k = None
        # command queue (actuation delay)
        n_act = int(round(Pl['act_delay'] / dt))
        queue = []                                               # (plant step index, v, om)
        # events
        trac = [(e['t0'], e['t1'], e['veh'], e['factor']) for e in self.events if e['kind'] == 'traction']
        stops = [(e['t0'], e['t1'], e['veh']) for e in self.events if e['kind'] == 'stop']
        jumps = [(e['t0'], e['veh'], np.asarray(e['delta'], float), e.get('tau', np.inf)) for e in self.events if e['kind'] == 'reloc']
        wd = C.get('watchdog')
        # recording
        R = dict(t=[], e=[], head=[], loc=[], sync=[], form=[], shape=[], v=[], om=[], sat=[], u=[], zeta=[], eps=[],
                 tau=[], p=[], P=[], ee=[], form_est=[], evec=[], eevec=[], th=[], thr=[])
        tsat = 0; nplant = 0; wsat = 0
        failed = False
        plant_step = 0
        a_set_w = Pl['a_set'] / r_n; a_stop_w = Pl['a_stop'] / r_n

        Kp, Ki, tmax = Pl['Kp'], Pl['Ki'], Pl['tau_max']
        crrMg = Pl['c_rr'] * M * G; cv, cturn, cw = Pl['c_v'], Pl['c_turn'], Pl['c_w']
        Mi00, Mi01, Mi10, Mi11 = Minv[:, 0, 0], Minv[:, 0, 1], Minv[:, 1, 0], Minv[:, 1, 1]

        def dyn(Wh, Ii, th_, SP, kap_, dlat_):
            """Wh, Ii, SP: (2, N) wheel speeds, PI integrators, set-points (rows: right, left)."""
            e = SP - Wh
            traw = Kp * e + Ki * Ii
            tq = np.minimum(np.maximum(traw, -tmax), tmax)
            dI = np.where((np.abs(traw) < tmax) | (e * traw < 0.0), e, 0.0)
            v_ = aR * Wh[0] + aL * Wh[1]; om_ = bR * Wh[0] - bL * Wh[1]
            F = crrMg * np.tanh(v_ * 50.0) + cv * v_
            Mr = cturn * np.tanh(om_ * 20.0) + cw * om_
            Q0 = tq[0] - (aR * F + bR * Mr); Q1 = tq[1] - (aL * F - bL * Mr)
            dW = np.array([Mi00 * Q0 + Mi01 * Q1, Mi10 * Q0 + Mi11 * Q1])
            vs = v_ * (1.0 - kap_)
            c_, s_ = np.cos(th_), np.sin(th_)
            return dW, dI, vs * c_ - dlat_ * s_, vs * s_ + dlat_ * c_, om_ * (1.0 - kap_), bool((np.abs(traw) >= tmax).any())

        for k in range(K + 1):
            t = k * Tc
            # ---- network arrivals
            changed = False
            for L in net.links:
                arr, ptr = L['arr'], L['ptr']
                while ptr < len(arr) and arr[ptr] <= t + 1e-12:
                    ptr += 1
                if ptr != L['ptr']:
                    L['ptr'] = ptr
                    kk = L['ks'][ptr - 1]
                    hold_z[L['i'], L['j']] = bc_hist[kk, L['j']]
                    hold_k[L['i'], L['j']] = kk
                    active[L['i'], L['j']] = True
            act = active.copy()
            if wd is not None:
                act &= (t - hold_k * Tc) <= wd
            Wact = self.A * act
            dact = Wact.sum(1)
            hold_sum = np.einsum('ij,ijk->ik', Wact, hold_z)
            # ---- localisation
            odo_hist[k] = odo
            if S['mode'] == 'ideal':
                pe, the = p.copy(), th.copy()
            else:
                if k % n_fix == 0:
                    zb = r_sens.standard_normal((N, 3)); zn = r_sens.standard_normal((N, 3))
                    uo = r_sens.random(N); ao = r_sens.uniform(0, 2 * np.pi, N)
                    a_b = np.exp(-n_fix * Tc / S['bias_tau'])
                    sd_b = np.array([S['bias_sd'], S['bias_sd'], S['bias_th_sd']]) * np.sqrt(1 - a_b ** 2)
                    bias = bias * a_b + sd_b * zb
                    meas = np.stack([p[:, 0], p[:, 1], th], 1) + bias + np.array([S['sd_p'], S['sd_p'], S['sd_th']]) * zn
                    out = uo < S['p_out']
                    meas[out, 0] += S['out_mag'] * np.cos(ao[out]); meas[out, 1] += S['out_mag'] * np.sin(ao[out])
                    for (tj, vj, dj, tauj) in jumps:     # relocalisation error: step that decays with time constant tau
                        if t >= tj:
                            meas[vj, :2] += dj * (np.exp(-(t - tj) / tauj) if np.isfinite(tauj) else 1.0)
                    fixes[k] = meas
                kf = k - n_dfix
                avail = [kk for kk in fixes if kk <= kf]
                if avail:
                    last_fix_k = max(avail)
                    for kk in [kk for kk in fixes if kk < last_fix_k]:
                        del fixes[kk]
                if S['mode'] == 'direct' or last_fix_k is None:
                    base = fixes[last_fix_k] if last_fix_k is not None else np.stack([p[:, 0], p[:, 1], th], 1)
                    pe, the = base[:, :2].copy(), base[:, 2].copy()
                else:                                    # fix (+) odometry increment since the fix time
                    fx = fixes[last_fix_k]; o0 = odo_hist[last_fix_k]; o1 = odo
                    d = o1[:, :2] - o0[:, :2]
                    c0, s0 = np.cos(o0[:, 2]), np.sin(o0[:, 2])
                    dl = np.stack([c0 * d[:, 0] + s0 * d[:, 1], -s0 * d[:, 0] + c0 * d[:, 1]], 1)
                    cf, sf = np.cos(fx[:, 2]), np.sin(fx[:, 2])
                    pe = fx[:, :2] + np.stack([cf * dl[:, 0] - sf * dl[:, 1], sf * dl[:, 0] + cf * dl[:, 1]], 1)
                    the = fx[:, 2] + (o1[:, 2] - o0[:, 2])
            # ---- controller
            tl = t * (1 + drift) + off
            Pr, thr, vr, wr, vrd = self.ref.state_local(tl)
            v_c, om_c, bc, u = self._controller(Pr, thr, vr, wr, vrd, pe, the, hold_sum, dact, z, w)
            bc_hist[k] = bc
            # filter update (exact for a zero-order-hold input)
            if C['type'] in ('cascade', 'sigleak'):
                gam = C['gam']; ex = np.exp(-gam * Tc)
                z = z + u * Tc + (w - u) * (1 - ex) / gam
                w = u + (w - u) * ex
            # ---- true errors and recording (global time)
            Pt, thrt = self.ref.state(t)[:2]
            e_true = Pt - p
            e_hist[k] = e_true
            if not np.all(np.isfinite(e_true)) or np.abs(e_true).max() > Sm['fail_err']:
                failed = True
                break
            if k % Sm['rec_every'] == 0:
                sy = []
                for L in net.links:
                    if act[L['i'], L['j']]:
                        sy.append(np.sum((e_true[L['i']] - e_hist[hold_k[L['i'], L['j']], L['j']]) ** 2))
                dif = e_true[:, None, :] - e_true[None, :, :]
                iu = np.triu_indices(N, 1)
                fd = np.hypot(dif[..., 0], dif[..., 1])[iu]
                dd = np.hypot(*(p[:, None, :] - p[None, :, :]).transpose(2, 0, 1))[iu] - np.hypot(*(Pt[:, None, :] - Pt[None, :, :]).transpose(2, 0, 1))[iu]
                e_est = Pr - pe                              # error as seen by the vehicle (estimate, local clock)
                dife = e_est[:, None, :] - e_est[None, :, :]
                R['ee'].append(np.hypot(e_est[:, 0], e_est[:, 1])); R['form_est'].append(np.hypot(dife[..., 0], dife[..., 1])[iu])
                R['t'].append(t); R['e'].append(np.hypot(e_true[:, 0], e_true[:, 1]))
                R['head'].append(np.abs(wrap(thrt - th)))
                R['loc'].append(np.hypot(*(pe - p).T))
                R['sync'].append(np.sqrt(np.mean(sy)) if sy else np.nan)
                R['form'].append(fd); R['shape'].append(np.abs(dd))
                R['v'].append(v_c); R['om'].append(om_c)
                R['u'].append(np.hypot(u[:, 0], u[:, 1])); R['zeta'].append(np.hypot(*(z + w / C['gam']).T))
                R['eps'].append(np.hypot(*(Pr - z - pe).T)) if C['type'] in ('cascade', 'sigleak') else R['eps'].append(np.hypot(*(Pr - pe).T))
                R['tau'].append(float(np.max(np.where(act, t - hold_k * Tc, 0.0))) if act.any() else 0.0)
                if Sm['keep']:
                    R['p'].append(p.copy()); R['P'].append(Pt.copy()); R['evec'].append(e_true.copy()); R['eevec'].append(e_est.copy())
                    R['th'].append(th.copy()); R['thr'].append(np.asarray(thrt, float).copy())
            if k == K:
                break
            # ---- commands to the drives (after the actuation delay)
            queue.append((plant_step + n_act, v_c.copy(), om_c.copy()))
            # ---- plant
            for sub in range(n_sub):
                tp_ = t + sub * dt
                while queue and queue[0][0] <= plant_step:
                    _, vq, oq = queue.pop(0)
                    if kin:
                        v_kin, w_kin = vq, oq
                    else:
                        tR_ = (vq + oq * B_n / 2) / r_n; tL_ = (vq - oq * B_n / 2) / r_n
                        mx = np.maximum(np.abs(tR_), np.abs(tL_)) * r_n
                        sc = np.minimum(1.0, Pl['vw_max'] / np.maximum(mx, 1e-12))
                        wsat += int((sc < 1).any())
                        tgtR, tgtL = tR_ * sc, tL_ * sc
                stopped = np.zeros(N, bool)
                for (t0, t1, vj) in stops:
                    if t0 <= tp_ < t1:
                        stopped[vj] = True
                kap_eff = kap.copy()
                for (t0, t1, vj, fac) in trac:
                    if t0 <= tp_ < t1:
                        kap_eff[vj] = 1 - fac
                if kin:
                    vv = np.where(stopped, 0.0, v_kin); ww = np.where(stopped, 0.0, w_kin)
                    vs = vv * (1 - kap_eff); ws = ww * (1 - kap_eff)
                    th_mid = th + ws * dt / 2
                    p = p + dt * np.stack([vs * np.cos(th_mid) - dlat * np.sin(th_mid), vs * np.sin(th_mid) + dlat * np.cos(th_mid)], 1)
                    th = th + ws * dt
                    vo, wo = vv, ww
                    odo[:, 0] += dt * vo * np.cos(odo[:, 2] + wo * dt / 2); odo[:, 1] += dt * vo * np.sin(odo[:, 2] + wo * dt / 2)
                    odo[:, 2] += wo * dt
                else:
                    gR = np.where(stopped, 0.0, tgtR); gL = np.where(stopped, 0.0, tgtL)
                    rate = np.where(stopped, a_stop_w, a_set_w) * dt
                    spR = spR + np.minimum(np.maximum(gR - spR, -rate), rate); spL = spL + np.minimum(np.maximum(gL - spL, -rate), rate)
                    Wh = np.array([wR, wL]); Ii = np.array([IR, IL]); SP = np.array([spR, spL])
                    px, py = p[:, 0], p[:, 1]
                    d1 = dyn(Wh, Ii, th, SP, kap_eff, dlat)
                    d2 = dyn(Wh + 0.5 * dt * d1[0], Ii + 0.5 * dt * d1[1], th + 0.5 * dt * d1[4], SP, kap_eff, dlat)
                    d3 = dyn(Wh + 0.5 * dt * d2[0], Ii + 0.5 * dt * d2[1], th + 0.5 * dt * d2[4], SP, kap_eff, dlat)
                    d4 = dyn(Wh + dt * d3[0], Ii + dt * d3[1], th + dt * d3[4], SP, kap_eff, dlat)
                    h6 = dt / 6.0
                    Wn = Wh + h6 * (d1[0] + 2 * d2[0] + 2 * d3[0] + d4[0])
                    In = Ii + h6 * (d1[1] + 2 * d2[1] + 2 * d3[1] + d4[1])
                    pxn = px + h6 * (d1[2] + 2 * d2[2] + 2 * d3[2] + d4[2])
                    pyn = py + h6 * (d1[3] + 2 * d2[3] + 2 * d3[3] + d4[3])
                    th = th + h6 * (d1[4] + 2 * d2[4] + 2 * d3[4] + d4[4])
                    wR_old, wL_old = wR, wL
                    wR, wL, IR, IL = Wn[0], Wn[1], In[0], In[1]
                    p = np.stack([pxn, pyn], 1)
                    tsat += int(d1[5]); nplant += 1
                    wm_R, wm_L = 0.5 * (wR + wR_old), 0.5 * (wL + wL_old)   # encoders, nominal geometry
                    vo, wo = r_n * (wm_R + wm_L) / 2, r_n * (wm_R - wm_L) / B_n
                    odo[:, 0] += dt * vo * np.cos(odo[:, 2] + wo * dt / 2); odo[:, 1] += dt * vo * np.sin(odo[:, 2] + wo * dt / 2)
                    odo[:, 2] += wo * dt
                plant_step += 1
            # slip processes: exact OU update once per control period (independent of the plant step size)
            zs = r_dist.standard_normal((2, N))
            a1 = np.exp(-Tc / Pl['slip_tau']); a2 = np.exp(-Tc / Pl['lat_tau'])
            kap = Pl['slip_mean'] + (kap - Pl['slip_mean']) * a1 + Pl['slip_sd'] * np.sqrt(1 - a1 ** 2) * zs[0]
            dlat = dlat * a2 + Pl['lat_sd'] * np.sqrt(1 - a2 ** 2) * zs[1]

        out = {q: np.array(v) for q, v in R.items() if len(v)}
        out['failed'] = failed
        out['meta'] = dict(net=net.stats, torque_sat_frac=tsat / max(nplant, 1), wheel_sat_frac=wsat / max(K, 1),
                           N=N, T=T, fc=fc, dt=dt, seed=self.seed, realized=realized)
        if Sm.get('record_rng'):                       # final states of all random streams (unit test of common random numbers)
            out['rng_state'] = [g.bit_generator.state for g in (r_init, r_net, r_sens, r_dist, r_clk, r_cal)]
        out['metrics'] = metrics(out, self.sim, self.T)
        return out


def metrics(R, sim, T):
    """Scalar performance indicators over t >= t_ss (steady state) and over the whole run."""
    if R['failed'] or 't' not in R:
        return dict(failed=True)
    t = R['t']
    m = t >= min(sim['t_ss'], 0.5 * T)
    e, form, shape, head, loc = R['e'], R['form'], R['shape'], R['head'], R['loc']
    fmax_t = form.max(1) if form.shape[1] else np.zeros(len(t))
    out = dict(failed=False,
               track_rms=float(np.sqrt(np.mean(e[m] ** 2))), track_max=float(e[m].max()),
               form_rms=float(np.sqrt(np.mean(form[m] ** 2))) if form.shape[1] else 0.0,
               form_max=float(fmax_t[m].max()), shape_rms=float(np.sqrt(np.mean(shape[m] ** 2))) if shape.shape[1] else 0.0,
               shape_max=float(shape[m].max()) if shape.shape[1] else 0.0,
               head_rms=float(np.sqrt(np.mean(head[m] ** 2))),
               sync_rms=float(np.sqrt(np.nanmean(R['sync'][m] ** 2))) if np.isfinite(R['sync'][m]).any() else float('nan'),
               loc_rms=float(np.sqrt(np.mean(loc[m] ** 2))),
               track_iae=float(np.trapezoid(e.mean(1), t)), form_iae=float(np.trapezoid(fmax_t, t)),
               form_peak=float(fmax_t.max()), track_peak=float(e.max()),
               tau_max=float(np.max(R['tau'])) if len(R['tau']) else 0.0)
    fe = R['form_est']
    fe_t = fe.max(1) if fe.shape[1] else np.zeros(len(t))
    out.update(track_est_rms=float(np.sqrt(np.mean(R['ee'][m] ** 2))), form_est_rms=float(np.sqrt(np.mean(fe[m] ** 2))) if fe.shape[1] else 0.0,
               form_est_max=float(fe_t[m].max()), form_est_iae=float(np.trapezoid(fe_t, t)))
    def settle(x, thr):
        above = np.nonzero(x > thr)[0]
        if not len(above):
            return 0.0
        return float(t[above[-1] + 1]) if above[-1] + 1 < len(t) else float('inf')
    out['settle_form'] = settle(fmax_t, 0.05)            # true formation error below 5 cm for good
    out['settle_form_est'] = settle(fe_t, sim['form_thr'])  # formation error seen by the vehicles below 2 cm
    vdot = np.diff(R['v'], axis=0) / np.diff(t)[:, None]
    out['cmd_acc_rms'] = float(np.sqrt(np.mean(vdot[m[1:]] ** 2)))
    return out
