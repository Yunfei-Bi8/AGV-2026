"""R6. Disturbance events on the high-fidelity model (warehouse route, 20 seeds)."""
from common import job, run_jobs, tuned_ctrl, CONTROLLERS

SEEDS = range(20)
EVENTS = {
    'traction': [dict(kind='traction', t0=30.0, t1=32.0, veh=1, factor=0.4)],
    'stop': [dict(kind='stop', t0=30.0, t1=33.0, veh=1)],
    'reloc': [dict(kind='reloc', t0=30.0, veh=1, delta=[0.0, 0.10], tau=5.0)],
    'switching': [dict(kind='traction', t0=30.0, t1=32.0, veh=1, factor=0.4)],
}

if __name__ == '__main__':
    jobs = []
    for s in SEEDS:
        for name, ev in EVENTS.items():
            net = dict(switching=(5.0, 2.0)) if name == 'switching' else None
            for typ in CONTROLLERS:
                jobs.append(job('warehouse', tuned_ctrl(typ), s, events=ev, net=net, case=name))
        # permanent failure of all links from AGV 1 at t = 20 s, without and with a 0.5 s watchdog
        for wd in (None, 0.5):
            for typ in CONTROLLERS:
                if typ == 'uncoupled' and wd is not None:
                    continue
                c = tuned_ctrl(typ); c['watchdog'] = wd
                jobs.append(job('warehouse', c, s, net=dict(outages=[(20.0, 1e9, 0, 'all')]), case='linkfail' + ('_wd' if wd else '')))
    doc = run_jobs('r6_events', jobs)
    print('wall', doc['wall_s'])
