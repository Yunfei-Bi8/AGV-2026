"""R0b. Why does the direct coupling diverge at large gains?  Same tuning runs (c = 1, 2) with one effect removed at
a time: ideal kinematic plant; 200-Hz control without actuation delay; no wheel-speed, torque and ramp limits."""
import json, os
from common import job, run_jobs, TUNE_SEEDS, traction_event, RES

COND = {'realistic': {}, 'ideal_plant': dict(profile='ideal'),
        'fast_no_delay': dict(plant=dict(act_delay=0.0)), 'no_limits': dict(plant=dict(vw_max=10.0, tau_max=1e3, a_set=100.0))}

if __name__ == '__main__':
    jobs = []
    for cond, kw in COND.items():
        for c in (1.0, 2.0):
            for sc in ('warehouse', 'transport'):
                for s in TUNE_SEEDS:
                    ctrl = dict(type='direct', c=c, fc=200.0 if cond == 'fast_no_delay' else 50.0)
                    jobs.append(job(sc, ctrl, s, events=[traction_event(s)], cond=cond, **kw))
    doc = run_jobs('r0b_direct_divergence', jobs, keep_ts=False)
    out = {}
    for j, m in zip(doc['jobs'], doc['metrics']):
        k = f"{j['cond']}/c={j['ctrl']['c']}"
        out.setdefault(k, [0, 0]); out[k][0] += int(bool(m.get('failed'))); out[k][1] += 1
    json.dump(out, open(os.path.join(RES, 'r0b_summary.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1))
