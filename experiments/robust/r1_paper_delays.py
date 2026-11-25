"""R1. Controller comparison under the delays of Peng-Ngo-Liu (2023, Sect. 5.2): heterogeneous sinusoidal delays
0.5+0.4 sin(t/2), 0.3+0.2 sin(t/2), 0.4+0.3 sin(t/2) s by sender, complete graph, model of the theorems
(kinematics, exact pose, 200 Hz).  (a) initial errors only, (b) plus a traction loss."""
from common import job, run_jobs, tuned_ctrl, CONTROLLERS, TEST_SEEDS, traction_event

if __name__ == '__main__':
    jobs = []
    for case in ('init', 'event'):
        for sc in ('transport', 'figure8'):
            for s in TEST_SEEDS:
                ev = [traction_event(s)] if case == 'event' else []
                for typ in CONTROLLERS:
                    jobs.append(job(sc, tuned_ctrl(typ), s, profile='ideal', netname='paper', events=ev, case=case))
    doc = run_jobs('r1_paper_delays', jobs)
    print('wall', doc['wall_s'])
