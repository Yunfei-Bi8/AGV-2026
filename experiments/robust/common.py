"""Shared settings of the robustness study."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from agvsim import runner, hifi as hf
from agvsim.runner import run_jobs, load, tuned_ctrl, RES

CONTROLLERS = ['uncoupled', 'cascade', 'sigleak', 'direct']
LABEL = {'uncoupled': 'No coupling', 'cascade': 'Cascade (proposed)', 'sigleak': 'sigma-leak filter',
         'direct': 'Direct coupling'}
TEST_SEEDS = list(range(30))
TUNE_SEEDS = list(range(1000, 1010))


def traction_event(seed, t_lo=25.0, t_hi=35.0, N=3, factor=0.4, dur=2.0):
    """Random traction loss (seed-determined): one vehicle delivers `factor` of its motion for `dur` seconds."""
    rng = np.random.default_rng([int(seed), 777])
    t0 = float(np.round(rng.uniform(t_lo, t_hi), 2))
    return dict(kind='traction', t0=t0, t1=t0 + dur, veh=int(rng.integers(N)), factor=factor)


def job(scen, ctrl, seed, **kw):
    d = dict(scen=scen, ctrl=dict(ctrl), seed=int(seed))
    d.update(kw)
    return d


# ---------------------------------------------------------------- post-processing helpers
def window_rms(t, x, lo, hi):
    m = (t >= lo) & (t <= hi)
    return float(np.sqrt(np.mean(x[m] ** 2))) if m.any() else np.nan


def event_metrics(t, form, t0, t1, thr=0.10, horizon=15.0):
    """Peak and IAE of the formation error after an event, and the recovery time after its end (1-s moving average
    below thr for the rest of the window)."""
    m = (t >= t0) & (t <= t0 + horizon)
    peak = float(form[m].max()) if m.any() else np.nan
    iae = float(np.trapezoid(form[m], t[m])) if m.any() else np.nan
    dt = t[1] - t[0]
    n = max(1, int(round(1.0 / dt)))
    ma = np.convolve(form, np.ones(n) / n, mode='same')
    after = np.nonzero(t >= t1)[0]
    rec = np.nan
    if len(after):
        above = np.nonzero(ma[after] > thr)[0]
        rec = 0.0 if not len(above) else (float(t[after[above[-1]]] - t1 + dt) if after[above[-1]] + 1 < len(t) else np.inf)
    return dict(peak=peak, iae=iae, recovery=rec)


def ts_of(ts, i):
    return {k: ts[f'{i}/{k}'] for k in ('t', 'form', 'form_est', 'track', 'track_max', 'sync', 'tau', 'loc')
            if f'{i}/{k}' in ts}
