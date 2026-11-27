"""R10. Delay margin of the direct coupling on the kinematic model (profile 'ideal'): straight reference at 0.6 m/s,
complete graph of three (d = 2), the same constant latency on every link, 120 s, no event. Direct coupling with
c d = 1-8 and the cascade with Lam = 1, 4, 8 1/s. The linearization of the direct coupling is stable for every delay
iff c d < 2 psi (1 + psi) (agvsim.theory.direct_delay_margin and direct_rhp_roots); the cascade is covered by
Theorem 1 for every delay and gain."""
from common import job, run_jobs

SEEDS = range(4)
LATENCY = [0.03, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.5, 3.0, 3.5, 4.0]
GAINS = {'direct': ('c', [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]), 'cascade': ('Lam', [1.0, 4.0, 8.0])}

if __name__ == '__main__':
    jobs = []
    for h in LATENCY:
        net = dict(lat='const', lat_mean=h, ge=(0.0, 0.0, 0.0, 1.0), spike_p=0.0)
        for typ, (par, vals) in GAINS.items():
            for v in vals:
                for s in SEEDS:
                    jobs.append(job('line', {'type': typ, par: v}, s, profile='ideal', net=net, T=120.0, latency=h))
    doc = run_jobs('r10_delay_margin', jobs)
    print('wall', doc['wall_s'])
