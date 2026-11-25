import sys, time, json
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from agvsim import references as rf, network as nw, cascade as cs, verify as vf

name = sys.argv[1] if len(sys.argv) > 1 else 'figure8'
netname = sys.argv[2] if len(sys.argv) > 2 else 'nominal'
ref, T = rf.make_reference(name)
offs = rf.TRIANGLE
A = np.ones((3, 3)) - np.eye(3)
bounds = rf.reference_bounds(ref, offs, T)
sched = nw.Schedule(A, T, nw.NETWORKS[netname], 7)
sim = cs.CascadeSim(ref, offs, A, {}, sched)
rng = np.random.default_rng(7)
ang = rng.uniform(0, 2*np.pi, 3); rad = rng.uniform(0.28, 0.32, 3)
dp0 = np.stack([rad*np.cos(ang), rad*np.sin(ang)], 1)
vt0 = np.deg2rad(rng.uniform(10, 18, 3))*rng.choice([-1, 1], 3)
y0 = sim.initial_state(dp0, vt0)
t0 = time.time()
run = sim.run(y0, T)
print('wall %.1fs' % (time.time()-t0), run['meta'])
r = vf.verify_run(run, sim.par, bounds, A.sum(1).max())
print(json.dumps(r, indent=1, default=float))
