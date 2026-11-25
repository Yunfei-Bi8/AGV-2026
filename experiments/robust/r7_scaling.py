"""R7. Topology and scalability: N = 3, 5, 8, 12 vehicles on a wide loop, five graphs, traction loss of vehicle 2."""
from common import job, run_jobs, tuned_ctrl

SEEDS = range(6)
EV = [dict(kind='traction', t0=30.0, t1=32.0, veh=1, factor=0.4)]

if __name__ == '__main__':
    jobs = []
    for N in (3, 5, 8, 12):
        for s in SEEDS:
            jobs.append(job(f'loop{N}', tuned_ctrl('uncoupled'), s, events=EV, N=N, graph='none'))
            for g in ('complete', 'ring', 'dring', 'chain', 'star'):
                for typ in ('cascade', 'direct'):
                    jobs.append(job(f'loop{N}', tuned_ctrl(typ), s, events=EV, N=N, graph=g))
    doc = run_jobs('r7_scaling', jobs)
    print('wall', doc['wall_s'])
