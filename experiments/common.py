"""Shared helpers for the closed-loop experiments."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from agvsim import references as rf, network as nw, cascade as cs, verify as vf

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', '')
GRAPHS = {
    'complete': np.ones((3, 3)) - np.eye(3),
    'chain': np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float),      # 1 -> 2 -> 3 (row i receives from column j)
    'none': np.zeros((3, 3)),
}


def initial_errors(seed, N, rmin, rmax, hmax):
    rng = np.random.default_rng(1000 + seed)
    ang = rng.uniform(0, 2 * np.pi, N)
    rad = rng.uniform(rmin, rmax, N)
    return np.stack([rad * np.cos(ang), rad * np.sin(ang)], 1), rng.uniform(-hmax, hmax, N)


def run_case(refname, netname, graph, seed, par=None, init=(0.1, 0.5, np.deg2rad(40)), keep=False, T=None,
             pert=None, pert_breaks=(), bnoise=None, rtol=1e-10, atol=1e-12, offs=None, A=None, rec_dt=0.01,
             init_hold='inactive', dp0=None, vt0=None):
    ref, Tref = rf.make_reference(refname)
    T = T or Tref
    offs = rf.TRIANGLE if offs is None else offs
    A = GRAPHS[graph] if A is None else A
    N = len(offs)
    cfg = nw.NETWORKS[netname] if isinstance(netname, str) else netname
    sched = nw.Schedule(A, T, cfg, seed) if A.sum() > 0 else None
    sim = cs.CascadeSim(ref, offs, A, par or {}, sched, pert=pert, pert_breaks=pert_breaks, bnoise=bnoise, init_hold=init_hold)
    if dp0 is None:
        dp0, vt0 = initial_errors(seed, N, *init)
    run = sim.run(sim.initial_state(dp0, vt0), T, rec_dt=rec_dt, rtol=rtol, atol=atol)
    bounds = rf.reference_bounds(ref, offs, T)
    return sim, run, bounds
