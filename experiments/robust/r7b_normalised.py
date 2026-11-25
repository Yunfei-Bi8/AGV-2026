"""R7b. Complete graphs with degree-normalised weights a_ij = 1/(N-1), the remedy for the growth of the total coupling
gain with the team size observed in R7 (same seeds, scenario and event as R7)."""
from common import job, run_jobs, tuned_ctrl

SEEDS = range(6)
EV = [dict(kind='traction', t0=30.0, t1=32.0, veh=1, factor=0.4)]

if __name__ == '__main__':
    jobs = []
    for N in (3, 5, 8, 12):
        for s in SEEDS:
            for typ in ('cascade', 'direct'):
                jobs.append(job(f'loop{N}', tuned_ctrl(typ), s, events=EV, N=N, graph='complete_norm'))
    doc = run_jobs('r7b_normalised', jobs)
    print('wall', doc['wall_s'])
