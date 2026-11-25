"""Error-controlled Dormand-Prince 5(4) integration between break points (no discontinuity inside a segment
except isolated kinks of the saturation functions, which the step-size control resolves)."""
import numpy as np

_A = [[],
      [1 / 5],
      [3 / 40, 9 / 40],
      [44 / 45, -56 / 15, 32 / 9],
      [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
      [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
      [35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84]]
_C = [0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0]
_B5 = np.array([35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0])
_B4 = np.array([5179 / 57600, 0.0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40])
_E = _B5 - _B4


class Stats:
    def __init__(self):
        self.steps = self.rejected = self.fevals = 0
        self.err_sum = 0.0          # accumulated local error estimates (upper estimate of the global error)


def integrate(f, t0, y0, t1, rtol=1e-10, atol=1e-12, stats=None):
    """Integrate y' = f(t, y) from t0 to t1; returns y(t1)."""
    t, y = t0, y0
    h = t1 - t0
    if h <= 0:
        return y0
    k = [None] * 7
    k[0] = f(t, y)
    nf = 1
    while True:
        last = t + h >= t1 - 1e-15 * max(1.0, abs(t1))
        if last:
            h = t1 - t
        for s in range(1, 7):
            ys = y + h * sum(a * k[m] for m, a in enumerate(_A[s]) if a != 0.0)
            k[s] = f(t + _C[s] * h, ys)
        nf += 6
        ynew = ys                                    # stage 7 point is the 5th-order solution (FSAL)
        err = h * sum(e * k[m] for m, e in enumerate(_E) if e != 0.0)
        sc = atol + rtol * np.maximum(np.abs(y), np.abs(ynew))
        en = float(np.sqrt(np.mean((err / sc) ** 2)))
        if en <= 1.0 or h < 1e-12:
            if stats is not None:
                stats.steps += 1
                stats.err_sum += float(np.max(np.abs(err)))
            t, y = t + h, ynew
            if last:
                break
            k[0] = k[6]
            h = min(h * min(5.0, max(0.2, 0.9 * (en + 1e-30) ** -0.2)), t1 - t)
        else:
            if stats is not None:
                stats.rejected += 1
            h *= max(0.1, 0.9 * en ** -0.2)
    if stats is not None:
        stats.fevals += nf
    return y
