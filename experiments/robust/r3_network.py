"""R3. Network robustness on the high-fidelity model (warehouse route, traction loss of AGV 2 at t = 30 s)."""
from common import job, run_jobs, tuned_ctrl, CONTROLLERS

SEEDS = range(10)
EV = [dict(kind='traction', t0=30.0, t1=32.0, veh=1, factor=0.4)]


def sweeps():
    S = []
    for L in [0.01, 0.1, 0.3, 1.0, 2.0, 4.0]:
        S.append(('latency', L, dict(lat='gauss', lat_mean=L, lat_sd=0.1 * L, lat_min=0.5 * L, lat_max=2.0 * L), {}))
    for p in [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]:
        S.append(('loss', p, dict(ge=(p, p, 0.0, 1.0)), {}))
    for D in [0.0, 2.0, 5.0, 10.0, 20.0]:
        S.append(('outage', D, dict(outages=[(28.0, D, 'all', 'all')] if D > 0 else []), {}))
        S.append(('outage_wd', D, dict(outages=[(28.0, D, 'all', 'all')] if D > 0 else []), dict(watchdog=0.5)))
    for fb in [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
        S.append(('rate', fb, dict(fb=fb), {}))
    for sc in [1.0, 2.0, 4.0, 8.0]:
        S.append(('sinus', sc, dict(lat='sinus', ge=(0.0, 0.0, 0.0, 1.0), spike_p=0.0, scale=sc), {}))
    return S


if __name__ == '__main__':
    jobs = []
    for s in SEEDS:
        jobs.append(job('warehouse', tuned_ctrl('uncoupled'), s, events=EV, factor='none', level=0.0))
        for (name, level, net, cextra) in sweeps():
            for typ in ('cascade', 'direct'):
                c = tuned_ctrl(typ); c.update(cextra)
                jobs.append(job('warehouse', c, s, events=EV, net=net, factor=name, level=level))
    doc = run_jobs('r3_network', jobs)
    print('wall', doc['wall_s'])
