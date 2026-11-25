"""Exact reference trajectories for a rigid virtual structure.

Every vehicle reference is a unicycle trajectory obtained analytically from the motion of the structure centre
(position p_c, heading theta_c, speed v_c, turn rate w_c and their derivatives).  No finite differences are used.
"""
import numpy as np


def smoothstep(x):
    """Quintic smoothstep S(x) and its first two derivatives; S(0)=0, S(1)=1."""
    x = np.clip(x, 0.0, 1.0)
    return 6 * x ** 5 - 15 * x ** 4 + 10 * x ** 3, 30 * x ** 4 - 60 * x ** 3 + 30 * x ** 2, 120 * x ** 3 - 180 * x ** 2 + 60 * x


def smoothstep_int(x):
    """Integral of S from 0 to x (x >= 0)."""
    xc = min(max(x, 0.0), 1.0)
    return xc ** 6 - 3 * xc ** 5 + 2.5 * xc ** 4 + max(x - 1.0, 0.0)


class SpeedProfile:
    """rate(t) = top * (f0 + (1 - f0) S(t / Tr)): starts at the fraction f0 and ramps up smoothly."""

    def __init__(self, top, f0=0.25, Tr=4.0):
        self.top, self.f0, self.Tr = top, f0, Tr

    def __call__(self, t):
        S, S1, S2 = smoothstep(t / self.Tr)
        val = self.top * (self.f0 + (1 - self.f0) * S)
        d1 = self.top * (1 - self.f0) * S1 / self.Tr
        d2 = self.top * (1 - self.f0) * S2 / self.Tr ** 2
        integ = self.top * (self.f0 * t + (1 - self.f0) * self.Tr * smoothstep_int(t / self.Tr))
        return integ, val, d1, d2


class Lissajous:
    """Figure-8  p_c = (a sin u, b sin 2u), u = u(t) from a SpeedProfile (closed form, no integration)."""
    needs_pc_state = False

    def __init__(self, a=5.0, b=2.5, lap=70.0, f0=0.25, Tr=4.0):
        self.a, self.b = a, b
        self.prof = SpeedProfile(2 * np.pi / lap, f0, Tr)

    def centre(self, t, pc_state=None):
        u, ud, udd, uddd = self.prof(t)
        a, b = self.a, self.b
        su, cu, s2, c2 = np.sin(u), np.cos(u), np.sin(2 * u), np.cos(2 * u)
        p = np.array([a * su, b * s2])
        p1 = np.array([a * cu, 2 * b * c2])
        p2 = np.array([-a * su, -4 * b * s2])
        p3 = np.array([-a * cu, -8 * b * c2])
        V = p1 * ud
        Ac = p2 * ud ** 2 + p1 * udd
        J = p3 * ud ** 3 + 3 * p2 * ud * udd + p1 * uddd
        v2 = V @ V
        vc = np.sqrt(v2)
        cr = V[0] * Ac[1] - V[1] * Ac[0]
        dt = V @ Ac
        wc = cr / v2
        vcd = dt / vc
        wcd = (V[0] * J[1] - V[1] * J[0]) / v2 - 2 * cr * dt / v2 ** 2
        return p, V[0] / vc, V[1] / vc, vc, wc, vcd, wcd

    def pc0(self):
        return self.centre(0.0)[0]


class PathRef:
    """Path given by a curvature profile kappa(s) (constant plus raised-cosine pulses) and a SpeedProfile.

    theta_c(s) is analytic; the centre position is integrated together with the vehicles (state pc)."""
    needs_pc_state = True

    def __init__(self, speed, kappa0=0.0, pulses=(), f0=0.25, Tr=4.0, theta0=0.0, p0=(0.0, 0.0)):
        self.prof = SpeedProfile(speed, f0, Tr)
        self.kappa0 = kappa0
        self.pulses = [(float(s0), float(S), 2 * float(dth) / float(S)) for (s0, S, dth) in pulses]  # (start, length, peak)
        self.theta0, self.p0 = theta0, np.asarray(p0, float)

    def _curv(self, s):
        kap, dkap, th = self.kappa0, 0.0, self.theta0 + self.kappa0 * s
        for s0, S, K in self.pulses:
            x = (s - s0) / S
            if x <= 0:
                continue
            if x >= 1:
                th += K * S / 2
                continue
            kap += K * (1 - np.cos(2 * np.pi * x)) / 2
            dkap += K * np.pi * np.sin(2 * np.pi * x) / S
            th += K * S * (x / 2 - np.sin(2 * np.pi * x) / (4 * np.pi))
        return kap, dkap, th

    def centre(self, t, pc_state):
        s, v, vd, _ = self.prof(t)
        kap, dkap, th = self._curv(s)
        return pc_state, np.cos(th), np.sin(th), v, kap * v, vd, dkap * v ** 2 + kap * vd

    def pc0(self):
        return self.p0.copy()


def vehicle_refs(cen, offs):
    """References of all vehicles from the centre kinematics.  Returns P (N,2), cr, sr, vr, wr, vrd (N,)."""
    pc, c, s, vc, wc, vcd, wcd = cen
    ox, oy = offs[:, 0], offs[:, 1]
    Rx, Ry = c * ox - s * oy, s * ox + c * oy
    Vx, Vy = vc * c - wc * Ry, vc * s + wc * Rx
    Ax = vcd * c - vc * wc * s - wcd * Ry - wc ** 2 * Rx
    Ay = vcd * s + vc * wc * c + wcd * Rx - wc ** 2 * Ry
    vr = np.hypot(Vx, Vy)
    P = np.empty((len(ox), 2))
    P[:, 0] = pc[0] + Rx
    P[:, 1] = pc[1] + Ry
    return P, Vx / vr, Vy / vr, vr, (Vx * Ay - Vy * Ax) / vr ** 2, (Vx * Ax + Vy * Ay) / vr


def make_reference(name):
    """The three references of the study.  Returns (reference object, duration in s)."""
    if name == 'circle':      # radius 4 m, 0.70 m/s
        return PathRef(0.70, kappa0=1 / 4.0), 45.0
    if name == 'figure8':     # x = 5 sin u, y = 2.5 sin 2u, one lap in 70 s
        return Lissajous(5.0, 2.5, 70.0), 75.0
    if name == 'warehouse':   # aisles joined by 90-degree turns and a lane change, 0.80 m/s
        q, lc = np.pi / 2, np.deg2rad(25.0)
        pulses = [(5.0, 4.5, q), (14.5, 4.5, -q), (23.0, 3.0, lc), (26.0, 3.0, -lc), (33.0, 4.5, q)]
        return PathRef(0.80, pulses=pulses), 60.0
    if name == 'line':
        return PathRef(0.60, f0=1.0), 60.0
    raise ValueError(name)


TRIANGLE = np.array([[0.6, 0.0], [-0.4, 0.5], [-0.4, -0.5]])


def reference_bounds(ref, offs, T, dt=2e-3):
    """Bounds needed by assumption A1, by dense evaluation of the analytic formulas (pc is irrelevant here)."""
    vmin, vmax, amax, wmax = np.inf, 0.0, 0.0, 0.0
    for t in np.arange(0.0, T + dt, dt):
        cen = ref.centre(t, np.zeros(2))
        _, _, _, vr, wr, vrd = vehicle_refs(cen, offs)
        vmin, vmax = min(vmin, vr.min()), max(vmax, vr.max())
        amax, wmax = max(amax, np.abs(vrd).max()), max(wmax, np.abs(wr).max())
    return dict(vmin=float(vmin), vmax=float(vmax), amax=float(amax), wmax=float(wmax))


# ---------------------------------------------------------------- formations that rotate relative to their path
def vehicle_refs_rot(cen, ph, ph1, ph2, offs):
    """Like vehicle_refs, but the formation is rotated by phi(t) relative to the path heading (object orientation
    theta_o = theta_c + phi).  ph, ph1, ph2: phi and its first two time derivatives."""
    pc, c, s, vc, wc, vcd, wcd = cen
    cp, sp = np.cos(ph), np.sin(ph)
    co, so = c * cp - s * sp, s * cp + c * sp                      # cos, sin of theta_o
    ox, oy = offs[:, 0], offs[:, 1]
    Rx, Ry = co * ox - so * oy, so * ox + co * oy
    wo, ao = wc + ph1, wcd + ph2
    Vx, Vy = vc * c - wo * Ry, vc * s + wo * Rx
    Ax = vcd * c - vc * wc * s - ao * Ry - wo ** 2 * Rx
    Ay = vcd * s + vc * wc * c + ao * Rx - wo ** 2 * Ry
    vr = np.hypot(Vx, Vy)
    P = np.empty((len(ox), 2))
    P[:, 0], P[:, 1] = pc[0] + Rx, pc[1] + Ry
    return P, Vx / vr, Vy / vr, vr, (Vx * Ay - Vy * Ax) / vr ** 2, (Vx * Ax + Vy * Ay) / vr


class PhiProfile:
    """phi(t) = A [S((t - t1)/D) - S((t - t2)/D)]: rotate by A, hold, rotate back (quintic smoothstep S)."""

    def __init__(self, A, t1, t2, D):
        self.A, self.t1, self.t2, self.D = A, t1, t2, D

    def __call__(self, t):
        S1, d1, dd1 = smoothstep((t - self.t1) / self.D)
        S2, d2, dd2 = smoothstep((t - self.t2) / self.D)
        A, D = self.A, self.D
        return A * (S1 - S2), A * (d1 - d2) / D, A * (dd1 - dd2) / D ** 2


# paper-inspired L-shaped transport formation (Peng, Ngo & Liu 2023, Sect. 5.1): l = [0.7, 0.4, 0.5] m,
# alpha = [0.6, 1.3, 1.8] pi, support points relative to the object reference point
L_ARM = np.array([0.7, 0.4, 0.5])
ALPHA = np.array([0.6, 1.3, 1.8]) * np.pi
L_FORMATION = np.stack([L_ARM * np.cos(ALPHA), L_ARM * np.sin(ALPHA)], 1)


def lattice_formation(N, dx=1.2, dy=1.0, per_row=4):
    """Rows of up to per_row vehicles, centred at the origin; N = 3 gives the triangle of the earlier study."""
    if N == 3:
        return TRIANGLE.copy()
    pts = []
    rows = int(np.ceil(N / per_row))
    k = 0
    for r in range(rows):
        n_r = min(per_row, N - k)
        ys = (np.arange(n_r) - (n_r - 1) / 2) * dy
        for y in ys:
            pts.append([-r * dx, y])
        k += n_r
    P = np.array(pts)
    return P - P.mean(0)


def make_scenario(name):
    """Scenarios of the robustness study: dict(ref, offs, phi, T)."""
    if name in ('circle', 'figure8', 'warehouse', 'line'):
        ref, T = make_reference(name)
        return dict(ref=ref, offs=TRIANGLE.copy(), phi=None, T=T, name=name)
    if name == 'transport':          # S-shaped aisle at 0.4 m/s; the L-formation turns by -0.6 rad and back
        pulses = [(3.0, 5.0, np.pi / 3), (10.5, 6.0, -2 * np.pi / 3), (19.0, 5.0, np.pi / 3)]
        ref = PathRef(0.40, pulses=pulses)
        return dict(ref=ref, offs=L_FORMATION.copy(), phi=PhiProfile(-0.6, 12.0, 38.0, 8.0), T=60.0, name=name)
    if name.startswith('loop'):      # wide loop for large formations, 0.8 m/s, curvature radius >= 3.8 m
        N = int(name[4:]) if len(name) > 4 else 3
        pulses = [(6.0, 12.0, np.pi / 2), (26.0, 12.0, np.pi / 2)]
        ref = PathRef(0.80, pulses=pulses)
        return dict(ref=ref, offs=lattice_formation(N), phi=None, T=60.0, name=name)
    raise ValueError(name)
