"""R9. Numerical convergence and reproducibility: plant step size, and bitwise re-runs of stored jobs."""
import json, os
import numpy as np
from common import job, run_jobs, tuned_ctrl, load, RES
from agvsim import runner

EV = [dict(kind='traction', t0=30.0, t1=32.0, veh=1, factor=0.4)]

if __name__ == '__main__':
    jobs = []
    for dt in (0.0005, 0.001, 0.002, 0.005, 0.01):
        for s in range(5):
            for typ in ('cascade', 'direct'):
                jobs.append(job('warehouse', tuned_ctrl(typ), s, events=EV, plant=dict(dt=dt), dt=dt))
    doc = run_jobs('r9_numerics', jobs)
    # bitwise reproducibility: re-run 12 stored jobs of R2 and compare all metrics
    d2, _ = load('r2_realistic')
    rng = np.random.default_rng(0)
    idx = rng.choice(len(d2['jobs']), 12, replace=False)
    same = []
    for i in idx:
        m = runner.run_job(d2['jobs'][int(i)])['metrics']
        same.append(all((m[k] == d2['metrics'][int(i)][k]) or (np.isnan(m[k]) and np.isnan(d2['metrics'][int(i)][k]))
                        for k in m if isinstance(m[k], float)))
    json.dump(dict(rerun_indices=[int(i) for i in idx], identical=same), open(os.path.join(RES, 'r9_rerun.json'), 'w'), indent=1)
    print('identical re-runs:', sum(same), 'of', len(same), 'wall', doc['wall_s'])
