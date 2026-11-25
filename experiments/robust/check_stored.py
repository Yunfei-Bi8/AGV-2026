"""Quick reproducibility check: re-run a few stored jobs of every experiment and compare their metrics with the
stored ones, bit for bit.

    python3 check_stored.py            # 2 random jobs per experiment (about 2 minutes with 12+ cores)
    python3 check_stored.py 5 r4 r6    # 5 jobs each of R4 and R6

Every job in results/robust/<experiment>.json is a complete description of one run, so this needs neither the time
series nor a full re-run of the study.  Exit status 1 if any metric differs."""
import json, os, sys
from multiprocessing import Pool
import numpy as np
from common import RES
from agvsim import runner

EXP = ['r0_tuning', 'r0b_direct_divergence', 'r1_paper_delays', 'r2_realistic', 'r3_network', 'r4_simreal',
       'r5_domain_rand', 'r6_events', 'r7_scaling', 'r7b_normalised', 'r8_sensitivity', 'r9_numerics']


def same(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (np.isnan(a) and np.isnan(b))
    return True                                       # only floating-point metrics are compared


def _rerun(arg):
    name, i, job, stored = arg
    m = runner.run_job(job)['metrics']
    bad = [k for k in m if k in stored and not same(m[k], stored[k])]
    return name, i, bad


def main():
    args = sys.argv[1:]
    k = int(args.pop(0)) if args and args[0].isdigit() else 2
    sel = [e for e in EXP if not args or e.split('_')[0] in args or e in args]
    tasks = []
    for name in sel:
        p = os.path.join(RES, name + '.json')
        if not os.path.exists(p):
            print('missing', p)
            continue
        with open(p) as f:
            doc = json.load(f)
        rng = np.random.default_rng(12345)
        for i in rng.choice(doc['n_jobs'], min(k, doc['n_jobs']), replace=False):
            tasks.append((name, int(i), doc['jobs'][int(i)], doc['metrics'][int(i)]))
    fails = 0
    with Pool(min(len(tasks), os.cpu_count() or 1)) as pool:
        for name, i, bad in pool.imap_unordered(_rerun, tasks):
            fails += bool(bad)
            print(f'{name:24s} job {i:5d}: ' + ('identical' if not bad else 'DIFFERS in ' + ', '.join(bad)), flush=True)
    print(f'{len(tasks) - fails} of {len(tasks)} re-runs identical to the stored results')
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
