"""R8. Sensitivity of the cascade to its own parameters (others at their defaults; warehouse, traction loss)."""
from common import job, run_jobs, tuned_ctrl

SEEDS = range(10)
EV = [dict(kind='traction', t0=30.0, t1=32.0, veh=1, factor=0.4)]
GRID = dict(lam=[0.25, 0.5, 1.0, 2.0, 4.0], gam=[2.5, 5.0, 10.0, 20.0], rho=[0.3, 0.6, 0.9], ebar=[0.02, 0.05, 0.1, 0.2, 1e3])

if __name__ == '__main__':
    jobs = []
    for s in SEEDS:
        for par, vals in GRID.items():
            for v in vals:
                c = tuned_ctrl('cascade'); c[par] = v
                jobs.append(job('warehouse', c, s, events=EV, factor=par, level=v))
    doc = run_jobs('r8_sensitivity', jobs)
    print('wall', doc['wall_s'])
