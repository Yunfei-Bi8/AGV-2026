"""R5. Domain-randomised Monte Carlo: every run draws scenario, vehicle, sensor, clock and network parameters and
disturbance events from wide ranges (seed-determined); all four controllers see the same draw."""
import numpy as np
from common import job, run_jobs, tuned_ctrl, CONTROLLERS

SEEDS = range(150)


def draw(seed):
    r = np.random.default_rng([int(seed), 555])
    lu = lambda a, b: float(np.exp(r.uniform(np.log(a), np.log(b))))
    scen = str(r.choice(['circle', 'figure8', 'warehouse', 'transport']))
    slip = r.uniform(0, 4)
    fc = float(r.choice([25.0, 50.0, 100.0]))
    sdp = r.uniform(0.002, 0.03)
    lat = lu(0.01, 1.0); p = r.uniform(0, 0.3)
    plant = dict(payload=r.uniform(0, 200), payload_spread=r.uniform(0, 100), cal_r=r.uniform(0, 0.02), cal_B=r.uniform(0, 0.02),
                 slip_mean=0.005 * slip, slip_sd=0.005 * slip, lat_sd=0.005 * slip, act_delay=r.uniform(0, 0.04),
                 vw_max=r.uniform(1.2, 1.8), tau_max=r.uniform(12, 30))
    sensor = dict(sd_p=sdp, sd_th=np.deg2rad(0.3) * sdp / 0.01, bias_sd=r.uniform(0, 0.03), d_fix=r.uniform(0, 0.15),
                  f_fix=float(r.choice([10.0, 20.0, 40.0])))
    net = dict(lat='gauss', lat_mean=lat, lat_sd=0.3 * lat, lat_min=0.2 * lat, lat_max=3 * lat,
               ge=(p, min(0.95, p + 0.5), 0.03, 0.3), fb=float(min(fc, r.choice([5.0, 10.0, 20.0, 50.0]))))
    clock = dict(offset=r.uniform(0, 0.02))
    ev = []
    if r.random() < 0.5:
        t0 = float(r.uniform(20, 40)); ev.append(dict(kind='traction', t0=t0, t1=t0 + r.uniform(1, 3), veh=int(r.integers(3)), factor=float(r.uniform(0.3, 0.7))))
    if r.random() < 0.25:
        t0 = float(r.uniform(20, 40)); ev.append(dict(kind='stop', t0=t0, t1=t0 + r.uniform(1, 4), veh=int(r.integers(3))))
    return scen, dict(fc=fc), plant, sensor, net, clock, ev


if __name__ == '__main__':
    jobs = []
    for s in SEEDS:
        scen, cx, plant, sensor, net, clock, ev = draw(s)
        for typ in CONTROLLERS:
            c = tuned_ctrl(typ); c.update(cx)
            jobs.append(job(scen, c, s, plant=plant, sensor=sensor, net=net, clock=clock, events=ev))
    doc = run_jobs('r5_domain_rand', jobs)
    print('wall', doc['wall_s'])
