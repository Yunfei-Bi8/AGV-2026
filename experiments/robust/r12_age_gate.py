"""R12. Latency sweep of R3 with an age gate: every receiver discards samples older than 1.5 s (controller key
'watchdog'). Assumption 2 admits the gate, and it bounds the delay of every active link by 1.5 s, so that Assumption
4(b) holds with h = 1.5 s whatever the network. Same seeds, event and networks as the latency sweep of R3, so that
the runs are paired with its runs without coupling."""
from common import job, run_jobs, tuned_ctrl
from r3_network import SEEDS, EV

GATE = 1.5
LATENCY = [0.01, 0.1, 0.3, 1.0, 2.0, 4.0]

if __name__ == '__main__':
    jobs = []
    for s in SEEDS:
        for L in LATENCY:
            net = dict(lat='gauss', lat_mean=L, lat_sd=0.1 * L, lat_min=0.5 * L, lat_max=2.0 * L)
            for typ in ('cascade', 'direct'):
                c = tuned_ctrl(typ)
                c['watchdog'] = GATE
                jobs.append(job('warehouse', c, s, events=EV, net=net, factor='latency_gate', level=L))
    doc = run_jobs('r12_age_gate', jobs)
    print('wall', doc['wall_s'])
