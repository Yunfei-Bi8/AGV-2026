"""R4. Sim-to-real factors, one at a time around the nominal realistic configuration (warehouse route, traction loss
of AGV 2 at t = 30 s; steady-state errors are evaluated before the event)."""
import numpy as np
from common import job, run_jobs, tuned_ctrl, CONTROLLERS

SEEDS = range(8)
EV = [dict(kind='traction', t0=30.0, t1=32.0, veh=1, factor=0.4)]
TH = np.deg2rad(0.3)


def factors():
    F = []
    for v in [0.0, 0.005, 0.01, 0.02, 0.04]:
        F.append(('loc_noise', v, dict(sensor=dict(sd_p=v, sd_th=TH * v / 0.01))))
    for v in [0.0, 0.005, 0.01, 0.02, 0.04]:
        F.append(('loc_bias', v, dict(sensor=dict(bias_sd=v, bias_th_sd=np.deg2rad(0.2) * v / 0.01))))
    for v in [0.0, 0.05, 0.1, 0.2, 0.4]:
        F.append(('loc_latency', v, dict(sensor=dict(d_fix=v))))
    for v in [5.0, 10.0, 20.0, 50.0]:
        F.append(('loc_rate', v, dict(sensor=dict(f_fix=v))))
    for v in [0.0, 0.01, 0.02, 0.05]:
        F.append(('calibration', v, dict(plant=dict(cal_r=v, cal_B=v))))
    for v in [0.0, 1.0, 2.0, 5.0, 10.0]:
        F.append(('slip', v, dict(plant=dict(slip_mean=0.005 * v, slip_sd=0.005 * v, lat_sd=0.005 * v))))
    for v in [0.0, 50.0, 150.0, 300.0]:
        F.append(('payload', v, dict(plant=dict(payload=v, payload_spread=min(50.0, v)))))
    for v in [0.0, 0.01, 0.03, 0.1]:
        F.append(('act_delay', v, dict(plant=dict(act_delay=v))))
    for v in [10.0, 20.0, 50.0, 100.0]:
        F.append(('ctrl_rate', v, dict(ctrlx=dict(fc=v))))
    for v in [0.0, 0.005, 0.02, 0.05, 0.1]:
        F.append(('clock', v, dict(clock=dict(offset=v))))
    for v in [1.0, 1.2, 1.5]:
        F.append(('wheel_limit', v, dict(plant=dict(vw_max=v))))
    return F


if __name__ == '__main__':
    jobs = []
    for s in SEEDS:
        for (name, level, kw) in factors():
            for typ in ('uncoupled', 'cascade', 'direct'):
                c = tuned_ctrl(typ)
                c.update(kw.get('ctrlx', {}))
                extra = {k: v for k, v in kw.items() if k != 'ctrlx'}
                jobs.append(job('warehouse', c, s, events=EV, factor=name, level=level, **extra))
    doc = run_jobs('r4_simreal', jobs)
    print('wall', doc['wall_s'])
